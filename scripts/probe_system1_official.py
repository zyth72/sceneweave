#!/usr/bin/env python3
"""通过 DeepSeek 官方 API 并发运行同输入 System1 对照；复用配对、解析与报告。"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_system1_comparison as comp  # noqa: E402
import probe_system1_routing as routing  # noqa: E402

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
SYSTEM_PROMPT = str(
    cast(dict[str, Any], routing.AGENT_CONFIG)["agents"][routing.DEFAULT_AGENT]["system"]
)


def request_body(call: comp.PlannedCall, model: str, thinking: bool) -> dict[str, Any]:
    """保存完整请求体，密钥只由 SDK 放进请求头。"""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": call.prompt},
        ],
        "stream": False,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    if thinking:
        body["reasoning_effort"] = "high"
    return body


def usage_record(raw: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    return {
        "reason": finish_reason,
        "cost": None,  # 官方响应没有账单金额；不按旧通道价格推算。
        "tokens": {
            "input": raw.get("prompt_tokens"),
            "output": raw.get("completion_tokens"),
            "reasoning": details.get("reasoning_tokens"),
            "cache": {
                "read": raw.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens")),
                "write": None,
            },
        },
    }


async def run_calls(
    *,
    client: AsyncOpenAI,
    calls: list[comp.PlannedCall],
    outdir: Path,
    model: str,
    concurrency: int,
    thinking: bool,
) -> list[comp.ComparisonSample]:
    """所有调用并发入队；记录计划序号、实际开始时间和完成行序。"""
    semaphore = asyncio.Semaphore(concurrency)
    fatal = asyncio.Event()
    started_batch = time.monotonic()
    samples: list[comp.ComparisonSample] = []

    async def run(index: int, call: comp.PlannedCall) -> None:
        body = request_body(call, model, thinking)
        sample = comp.ComparisonSample(
            call_index=index,
            pair_id=call.pair_id,
            case_id=call.case.id,
            case_suite=call.case.suite,
            profile_label=call.persona.label,
            profile_real_name=call.persona.real_name,
            repeat=call.repeat,
            mode=call.mode,
            model=model,
            prompt=call.prompt,
            prefix_sha256=call.prefix_sha256,
            expected_routes=call.case.expected_routes,
        )
        sample.raw_events.append({"type": "official_request", "body": body})
        queued = time.monotonic()
        async with semaphore:
            started = time.monotonic()
            sample.raw_events.append(
                {
                    "type": "request_start",
                    "utc": dt.datetime.now(dt.UTC).isoformat(),
                    "queue_seconds": started - queued,
                }
            )
            if fatal.is_set():
                sample.exit_code = 1
                sample.call_error = "skipped_after_auth_or_balance_failure"
            else:
                for attempt in range(3):
                    try:
                        sdk_args: dict[str, Any] = {
                            name: value for name, value in body.items() if name != "thinking"
                        }
                        sdk_args["extra_body"] = {"thinking": body["thinking"]}
                        response = await client.chat.completions.create(**sdk_args)
                        raw = response.model_dump(mode="json")
                        sample.raw_events.append(
                            {"type": "official_response", "attempt": attempt + 1, "body": raw}
                        )
                        choice = response.choices[0]
                        sample.text = choice.message.content or ""
                        sample.reasoning = str(
                            getattr(choice.message, "reasoning_content", "") or ""
                        )
                        sample.message_ids = [response.id]
                        sample.tool_polluted = bool(choice.message.tool_calls)
                        if sample.tool_polluted:
                            sample.tool_events = ["unexpected_api_tool_calls"]
                        if response.usage is not None:
                            sample.usage = usage_record(
                                response.usage.model_dump(mode="json"), choice.finish_reason
                            )
                            sample.usage_source = "deepseek-api"
                        sample.exit_code = 0
                        try:
                            if choice.finish_reason != "stop":
                                raise ValueError(f"finish_reason={choice.finish_reason}")
                            result = comp._parse(sample.text, call.mode)
                            for name in (
                                "route",
                                "argmax",
                                "tie",
                                "margin",
                                "entropy",
                                "probabilities",
                                "note",
                            ):
                                setattr(sample, name, result[name])
                            sample.parse_ok = True
                            value = sample.route if call.mode == "field" else sample.argmax
                            sample.in_expected = (
                                value in call.case.expected_routes
                                if call.case.expected_routes
                                else None
                            )
                        except ValueError as error:
                            sample.parse_error = str(error)
                        break
                    except (APIStatusError, APIConnectionError) as error:
                        status = error.status_code if isinstance(error, APIStatusError) else None
                        sample.raw_events.append(
                            {
                                "type": "api_error",
                                "attempt": attempt + 1,
                                "status": status,
                                "error_type": type(error).__name__,
                            }
                        )
                        if status in (401, 402):
                            fatal.set()
                        retryable = status in (429, 500, 502, 503, 504) or status is None
                        if retryable and attempt < 2:
                            await asyncio.sleep(0.5 * 2**attempt)
                            continue
                        sample.exit_code = 1
                        sample.timed_out = isinstance(error, APITimeoutError)
                        sample.call_error = (
                            f"{type(error).__name__}: HTTP {status}"
                            if status
                            else type(error).__name__
                        )
                        break
                    except Exception as error:
                        # 不保存异常原文（可能带请求头、全局配置或敏感值）。
                        sample.exit_code = 1
                        sample.call_error = type(error).__name__
                        break
            sample.elapsed_seconds = time.monotonic() - started
        sample.raw_events.append(
            {"type": "request_end", "utc": dt.datetime.now(dt.UTC).isoformat()}
        )
        samples.append(sample)
        with (outdir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
        status_text = "ok" if comp._valid(sample) else "invalid"
        print(f"[{len(samples)}/{len(calls)}] {call.pair_id}/{call.mode} {status_text}", flush=True)

    await asyncio.gather(*(run(index, call) for index, call in enumerate(calls, 1)))
    (outdir / "timing.json").write_text(
        json.dumps(
            {
                "batch_elapsed_seconds": time.monotonic() - started_batch,
                "concurrency_limit": concurrency,
                "calls": len(samples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return samples


def unsuccessful(summary: dict[str, Any]) -> bool:
    return bool(
        summary["call_failures"]
        or summary["schema_failures"]
        or summary["tool_polluted"]
        or summary["paired"]["missing_pairs"]
        or summary["paired"]["failed_pairs"]
        or summary["actual_calls"] != summary["expected_calls"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=comp.DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-from", type=Path)
    args = parser.parse_args(argv)
    if args.report_from is not None:
        summary = comp.regenerate_report(args.report_from)
        return int(unsuccessful(summary))
    if args.repeats < 1 or not 1 <= args.concurrency <= 2500:
        raise SystemExit("repeats 必须为正数；Flash concurrency 必须在 1..2500")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("timeout 必须是有限正数")
    bundle = comp.load_source(args.source_dir)
    cases, personas = list(bundle.cases), list(bundle.personas)
    if args.smoke:
        cases, personas = comp._select_smoke(cases, personas)
    repeats = 1 if args.smoke else args.repeats
    pairs = comp.build_pairs(cases, personas, bundle.prompts, repeats, args.seed)
    calls = comp.build_calls(pairs)
    if args.dry_run:
        print(f"官方 {MODEL}：{len(calls)} 次，程序并发上限 {args.concurrency}；仅输出模式不同")
        return 0
    outdir = args.output_dir or Path("sceneweave-runs/system1-official") / dt.datetime.now(
        dt.UTC
    ).strftime("%Y%m%d-%H%M%S")
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆写：{outdir}")
    load_dotenv(Path.cwd() / ".env", override=False, interpolate=False)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY（从环境或当前目录 .env 读取）")
    if os.environ.get("OPENAI_CUSTOM_HEADERS"):
        raise SystemExit("检测到 OPENAI_CUSTOM_HEADERS，拒绝引入未声明的全局请求头")
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = comp.build_manifest(
        bundle=bundle,
        cases=cases,
        personas=personas,
        pairs=pairs,
        prompts=bundle.prompts,
        model=MODEL,
        agent="official-sdk",
        repeats=repeats,
        seed=args.seed,
        timeout=args.timeout,
        smoke=args.smoke,
        no_thinking=args.no_thinking,
        no_db_usage=True,
        usage_db="not-used",
    )
    execution = {
        "transport": "deepseek-official",
        "base_url": BASE_URL,
        "concurrency": args.concurrency,
        "http_max_connections": args.concurrency,
        "system_prompt": SYSTEM_PROMPT,
        "thinking": not args.no_thinking,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "reasoning_effort": None if args.no_thinking else "high",
        "ordering": "计划顺序随机打乱，所有请求并发入队；call_index为计划编号，JSONL行序为完成顺序",
        "timing": "逐请求耗时不含排队，包含显式重试等待；全批耗时见timing.json",
        "retry": "SDK自动重试关闭；429/5xx/连接错误最多额外重试2次，401/402停止后续排队调用",
        "usage": "input=prompt_tokens，output=completion_tokens（包含reasoning，不能再叠加）；"
        "reasoning和cache_hit为子集；cache_write及cost未返回则记n/a",
    }
    manifest["execution"] = execution
    manifest["source"]["transport"] = "deepseek-official"
    manifest["source_hashes"][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    async def execute() -> list[comp.ComparisonSample]:
        async with AsyncOpenAI(
            api_key=key,
            base_url=BASE_URL,
            admin_api_key="",
            organization="",
            project="",
            webhook_secret="",
            timeout=args.timeout,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                follow_redirects=False,
                # SDK 默认池只有 1000；连接数跟随并发参数，避免 2500 被暗中压低。
                limits=httpx.Limits(
                    max_connections=args.concurrency,
                    max_keepalive_connections=args.concurrency,
                ),
            ),
        ) as client:
            return await run_calls(
                client=client,
                calls=calls,
                outdir=outdir,
                model=MODEL,
                concurrency=args.concurrency,
                thinking=not args.no_thinking,
            )

    samples = asyncio.run(execute())
    summary = comp.write_report(
        outdir=outdir,
        samples=samples,
        cases=cases,
        personas=personas,
        prompts=bundle.prompts,
        source_meta=manifest["source"],
        model=MODEL,
        repeats=repeats,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(f"报告：{outdir / 'report.md'}；完整可比较 {summary['paired']['comparable_pairs']} 对")
    return int(unsuccessful(summary))


if __name__ == "__main__":
    raise SystemExit(main())
