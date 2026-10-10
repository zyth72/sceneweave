"""官方 System1 探针：用真实 SDK 与假 HTTP 验证并发、请求证据和失败口径。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_system1_official.py"
_SPEC = importlib.util.spec_from_file_location("probe_system1_official", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
official: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = official
_SPEC.loader.exec_module(official)
comp: Any = official.comp


def _fixture(tmp_path: Path) -> tuple[Path, Any, list[Any]]:
    source = tmp_path / "source"
    source.mkdir()
    persona = comp.SourcePersona("baseline", None, None, None, "无额外人格", ())
    case = comp.SourceCase(
        "noise",
        "influence",
        "无关噪声",
        "安全的巨响",
        "交付信封",
        ("走到收件人处",),
        ("行动待开始",),
        ("例行任务",),
        ("continue",),
        "噪声无关",
    )
    prompt = '事件：安全巨响\n意向：交付信封\n【输出】\n{"probabilities":{"continue":0.9,"replan":0.1,"rethink":0.0},"note":"..."}'
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "probe": "independent-system1-sensitivity",
                "model": "fake-source-model",
                "repeats": 1,
                "personas": [persona.to_manifest()],
                "cases": [case.to_manifest()],
            },
            ensure_ascii=False,
        )
    )
    (source / "samples.jsonl").write_text(
        json.dumps(
            {
                "case_id": "noise",
                "profile_label": "baseline",
                "repeat": 0,
                "prompt": prompt,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    bundle = comp.load_source(source)
    calls = comp.build_calls(comp.build_pairs([case], [persona], bundle.prompts, 2, 7))
    return source, bundle, calls


def _response(
    body: dict[str, Any], *, finish: str = "stop", content: str | None = None
) -> dict[str, Any]:
    is_field = '"route"' in body["messages"][1]["content"]
    text = (
        content
        if content is not None
        else json.dumps(
            {"route": "continue", "note": "无关"}
            if is_field
            else {
                "probabilities": {"continue": 0.9, "replan": 0.1, "rethink": 0.0},
                "note": "无关",
            }
        )
    )
    return {
        "id": "completion-test",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": "已知噪声安全",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_cache_hit_tokens": 60,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


def _run(calls: list[Any], outdir: Path, handler: Any, concurrency: int = 2) -> list[Any]:
    outdir.mkdir(exist_ok=True)

    async def execute() -> list[Any]:
        async with AsyncOpenAI(
            api_key="fake-not-a-secret",
            base_url=official.BASE_URL,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            return list(
                await official.run_calls(
                    client=client,
                    calls=calls,
                    outdir=outdir,
                    model=official.MODEL,
                    concurrency=concurrency,
                    thinking=True,
                )
            )

    return asyncio.run(execute())


def test_concurrency_full_request_evidence_and_offline_report(tmp_path: Path) -> None:
    _, bundle, calls = _fixture(tmp_path)
    active = peak = 0
    bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        body = json.loads(request.content)
        bodies.append(body)
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0.004 if len(bodies) == 1 else 0.001)
        active -= 1
        return httpx.Response(200, json=_response(body))

    out = tmp_path / "out"
    samples = _run(calls, out, handler)
    assert peak == 2
    assert len(samples) == len(bodies) == 4
    assert all(comp._valid(sample) for sample in samples)
    assert {sample.call_index for sample in samples} == {1, 2, 3, 4}
    assert samples[0].call_index == 2
    for sample in samples:
        request_body = sample.raw_events[0]["body"]
        assert request_body in bodies  # 完整保存体与 SDK 实际 HTTP body 一致。
        assert request_body["messages"][0]["content"] == official.SYSTEM_PROMPT
        assert request_body["messages"][1]["content"] == sample.prompt
        assert request_body["thinking"] == {"type": "enabled"}
        assert request_body["reasoning_effort"] == "high"
        assert sample.reasoning == "已知噪声安全"
        assert sample.usage["tokens"]["output"] == 30
        assert sample.usage["tokens"]["reasoning"] == 20
        assert sample.usage["cost"] is None
    assert len((out / "samples.jsonl").read_text().splitlines()) == 4
    assert json.loads((out / "timing.json").read_text())["calls"] == 4
    manifest = comp.build_manifest(
        bundle=bundle,
        cases=list(bundle.cases),
        personas=list(bundle.personas),
        pairs=comp.build_pairs(list(bundle.cases), list(bundle.personas), bundle.prompts, 2, 7),
        prompts=bundle.prompts,
        model=official.MODEL,
        agent="official-sdk",
        repeats=2,
        seed=7,
        timeout=10,
        smoke=False,
        no_thinking=False,
        no_db_usage=True,
        usage_db="unused",
    )
    manifest["source"]["transport"] = "deepseek-official"
    manifest["execution"] = {
        "transport": "deepseek-official",
        "system_prompt": official.SYSTEM_PROMPT,
        "concurrency": 2,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    before = hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest()
    summary = comp.regenerate_report(out)
    assert not official.unsuccessful(summary)
    report = (out / "report.md").read_text()
    assert official.SYSTEM_PROMPT in report
    assert "官方 API 未返回 cost" in report
    assert "不能把 CLI 输入文本等同" not in report
    assert "安全巨响" in report
    assert "官方 API usage" in summary["cost_tokens"]["note"]
    assert "不代表实际执行" in summary["paired"]["order_definition"]
    assert before == hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest()


def test_transient_error_retries_without_leaking_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, calls = _fixture(tmp_path)
    attempts = 0

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(official.asyncio, "sleep", no_wait)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "SECRET_ERROR_BODY"}})
        return httpx.Response(200, json=_response(json.loads(request.content)))

    samples = _run(calls[:1], tmp_path / "out", handler, concurrency=1)
    assert attempts == 2 and comp._valid(samples[0])
    record = samples[0].to_record()
    assert "SECRET_ERROR_BODY" not in json.dumps(record)
    assert any(event.get("status") == 429 for event in record["raw_events"])


def test_auth_failure_stops_queued_calls(tmp_path: Path) -> None:
    _, _, calls = _fixture(tmp_path)
    requests = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, json={"error": {"message": "SECRET_ERROR_BODY"}})

    samples = _run(calls, tmp_path / "out", handler, concurrency=1)
    assert requests == 1 and len(samples) == 4
    assert sum(s.call_error == "skipped_after_auth_or_balance_failure" for s in samples) == 3
    assert all(not comp._valid(s) for s in samples)
    assert "SECRET_ERROR_BODY" not in (tmp_path / "out" / "samples.jsonl").read_text()


@pytest.mark.parametrize("finish,content", [("length", None), ("stop", "{bad-json")])
def test_invalid_output_preserves_original_response(
    tmp_path: Path, finish: str, content: str | None
) -> None:
    _, _, calls = _fixture(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_response(json.loads(request.content), finish=finish, content=content)
        )

    samples = _run(calls[:1], tmp_path / "out", handler)
    assert samples[0].exit_code == 0 and not samples[0].parse_ok
    assert samples[0].parse_error
    assert any(e["type"] == "official_response" for e in samples[0].raw_events)


def test_dry_run_does_not_read_credentials_or_call_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, calls = _fixture(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("离线入口不应读密钥或访问 API")

    monkeypatch.setattr(official, "load_dotenv", forbidden)
    monkeypatch.setattr(official, "AsyncOpenAI", forbidden)
    assert official.main(["--source-dir", str(source), "--dry-run"]) == 0
    body = official.request_body(calls[0], official.MODEL, False)
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    usage = official.usage_record({}, None)
    assert usage["tokens"]["input"] is None and usage["cost"] is None
