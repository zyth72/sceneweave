#!/usr/bin/env python3
"""独立 System1 路由探针：比较「字段选择」与「三路倾向分布」两种表达。

本脚本是独立实验，绝不启动 System2/System3/翻译器/World/engine 完整循环，
不生成动作，不影响正式模拟路由。它只把「该次事件 + 本角色当前私有状态」
交给 System1，让模型二选一地表达：

- ``field`` 模式：直接输出 ``{"route": "continue|replan|rethink", "note": ...}``。
- ``distribution`` 模式：输出 ``{"probabilities": {"continue": .., "replan": ..,
  "rethink": ..}, "note": ...}`` —— 这是「该事件对当前状态的影响所驱动的路径
  倾向」，不是世界事件发生概率，也不是随机抽签；程序再据此离线取 argmax。

每次调用都通过已配置的 ``/usr/sbin/opencode -m opencode-go/deepseek-v4.1-flash``
在临时独立目录、无工具 primary agent、独立新会话中进行；不读取 .env/密钥，
不连接常驻服务（``--standalone`` 私有服务器）。默认输出忽略目录
``sceneweave-runs/system1-probe/<timestamp>/``。

``--dry-run`` 只生成待发 prompt 与任务数，不做任何外部调用。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROUTES = ("continue", "replan", "rethink")
DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash"
DEFAULT_AGENT = "system1-probe"
DEFAULT_OPENCODE = "/usr/sbin/opencode"
DEFAULT_USAGE_DB = "~/.local/share/opencode/opencode.db"
PROBABILITY_SUM_TOLERANCE = 0.02  # 允许有说明的小数舍入容差；不偷偷重归一化
CASES_PATH = Path(__file__).with_name("probe_system1_cases.json")

AGENT_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "default_agent": DEFAULT_AGENT,
    "agents": {
        DEFAULT_AGENT: {
            "mode": "primary",
            "description": "Isolated no-tool strict JSON classifier for the System1 routing probe",
            "system": (
                "You are a strict JSON classifier. Output only the single requested JSON "
                "object and nothing else. Never call tools, never read files, never run "
                "commands."
            ),
            "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
        }
    },
}


@dataclass(frozen=True, slots=True)
class ProbeCase:
    id: str
    kind: str
    title: str
    event: str
    intention: str | None
    plan: tuple[str, ...]
    known_actions: tuple[str, ...]
    memory: tuple[str, ...]
    expected_routes: tuple[str, ...]
    label_rationale: str


def load_cases(path: Path) -> list[ProbeCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise SystemExit(f"用例文件格式错误：{path}")
    cases: list[ProbeCase] = []
    for raw in data["cases"]:
        if not isinstance(raw, dict):
            raise SystemExit(f"用例项必须是对象：{raw!r}")
        expected = tuple(raw["expected_routes"])
        for route in expected:
            if route not in ROUTES:
                raise SystemExit(f"用例 {raw.get('id')} 的 expected_routes 含未知路由：{route}")
        if raw["kind"] not in ("clear", "boundary"):
            raise SystemExit(f"用例 {raw.get('id')} 的 kind 必须是 clear/boundary")
        cases.append(
            ProbeCase(
                id=str(raw["id"]),
                kind=str(raw["kind"]),
                title=str(raw["title"]),
                event=str(raw["event"]),
                intention=None if raw.get("intention") is None else str(raw["intention"]),
                plan=tuple(str(item) for item in raw.get("plan", ())),
                known_actions=tuple(str(item) for item in raw.get("known_actions", ())),
                memory=tuple(str(item) for item in raw.get("memory", ())),
                expected_routes=expected,
                label_rationale=str(raw["label_rationale"]),
            )
        )
    if len(cases) < 12:
        raise SystemExit(f"至少需要 12 个用例，当前 {len(cases)}")
    return cases


def build_prompt(case: ProbeCase, mode: str) -> str:
    """构造 System1 输入：只有该次事件与本角色私有状态，不含参考答案。"""
    intention = case.intention if case.intention is not None else "（无，尚未决定过意向）"
    plan_lines = [f"  - {item}" for item in case.plan] or ["  - 无（尚无已保存计划）"]
    known_lines = [f"  - {item}" for item in case.known_actions] or ["  - 无"]
    memory_lines = [f"  - {item}" for item in case.memory] or ["  - 无"]
    lines = [
        "你是角色内部 System1 的分流判断器。只依据下面这次事件和本角色的私有状态，"
        "判断角色在内部规划上应当走哪条路。不要生成意向文本，不要生成任何动作。",
        "",
        "【本次事件】",
        f"  {case.event}",
        "",
        "【本角色当前私有状态】",
        f"  当前意向：{intention}",
        "  已保存的完整动作链与时间安排：",
        *plan_lines,
        "  已知自身行动状态：",
        *known_lines,
        "  近期私人认知：",
        *memory_lines,
        "",
        "【三路含义】",
        "  - continue：沿用已保存的完整链与排程，只推进最近步骤；不重选意向、不重新落实。"
        "没有已保存计划时不要选。",
        "  - replan：保留当前意向，只按新的已知条件重新落实完整动作链。没有当前意向时不要选。",
        "  - rethink：重新选择意向后再落实。第一次决定意向、或原目标已不再合适时必须选。",
        "",
    ]
    if mode == "field":
        lines += [
            "【输出】只输出一个 JSON 对象，直接给出路由字段，格式：",
            '{"route": "continue|replan|rethink", "note": "简短说明"}',
        ]
    elif mode == "distribution":
        lines += [
            "【输出】根据这次事件对当前状态的影响，给出 continue/replan/rethink 三条路径的"
            "倾向分布。这是模型自报的路径倾向，不是世界事件发生概率，也不是随机抽签；"
            "不要直接给出 route。只输出一个 JSON 对象，格式：",
            '{"probabilities": {"continue": 0.0, "replan": 0.0, "rethink": 0.0}, '
            '"note": "简短说明"}',
            "三个键都要给出，数值在 0 到 1 之间，三者之和为 1。",
        ]
    else:
        raise ValueError(f"未知模式：{mode}")
    return "\n".join(lines)


def parse_field(text: str) -> dict[str, Any]:
    """严格解析字段模式；未知字段/未知路由/空说明都报错。"""
    data = _load_object(text)
    if set(data) != {"route", "note"}:
        raise ValueError(f"字段模式必须恰含 route 与 note，得到：{sorted(data)}")
    route = data["route"]
    if route not in ROUTES:
        raise ValueError(f"未知 route：{route!r}")
    note = data["note"]
    if not isinstance(note, str) or not note:
        raise ValueError("note 必须是非空字符串")
    return {"route": route, "note": note, "tie": False, "margin": None, "entropy": None}


def parse_distribution(text: str) -> dict[str, Any]:
    """严格解析分布模式：三项、有限数值（拒 bool）、0..1、和≈1、取唯一最大值。

    不偷偷重归一化；并列最大单独记 tie，不用字典顺序裁决。
    """
    data = _load_object(text)
    if set(data) != {"probabilities", "note"}:
        raise ValueError(f"分布模式必须恰含 probabilities 与 note，得到：{sorted(data)}")
    probs = data["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(ROUTES):
        raise ValueError(f"probabilities 必须恰含三条路径：{ROUTES}")
    values: dict[str, float] = {}
    for route in ROUTES:
        raw = probs[route]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"probabilities.{route} 必须是数值，不能是 bool 或字符串")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"probabilities.{route} 必须是有限数值")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"probabilities.{route} 必须在 0..1：{value}")
        values[route] = value
    total = math.fsum(values.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"probabilities 之和必须约为 1（容差 {PROBABILITY_SUM_TOLERANCE}），得到 {total}"
        )
    note = data["note"]
    if not isinstance(note, str) or not note:
        raise ValueError("note 必须是非空字符串")
    maximum = max(values.values())
    winners = [route for route in ROUTES if values[route] == maximum]
    tie = len(winners) != 1
    ordered = sorted(values.values(), reverse=True)
    margin = None if tie else ordered[0] - ordered[1]
    entropy = -math.fsum(p * math.log2(p) for p in values.values() if p > 0.0)
    return {
        "route": None if tie else winners[0],
        "tie": tie,
        "margin": margin,
        "entropy": entropy,
        "probabilities": values,
        "note": note,
    }


def _load_object(text: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 字段：{key}")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"输出不是合法 JSON：{error}") from error
    if not isinstance(data, dict):
        raise ValueError("输出必须是 JSON 对象")
    return data


def parse_events(stdout: str) -> dict[str, Any]:
    """解析 opencode NDJSON：去重、抽取文本/推理/usage/工具污染/身份。"""
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_events: list[str] = []
    step_finish: dict[str, Any] | None = None
    session_ids: set[str] = set()
    step_ids: set[str] = set()
    message_ids: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part")
        part_id = part.get("id") if isinstance(part, dict) else None
        key = str(part_id) if part_id else f"{event.get('type')}:{event.get('timestamp')}"
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
        if isinstance(part, dict):
            if isinstance(part.get("sessionID"), str):
                session_ids.add(part["sessionID"])
            if isinstance(part.get("messageID"), str):
                message_ids.add(part["messageID"])
            if isinstance(part.get("id"), str):
                step_ids.add(part["id"])
        etype = event.get("type")
        if etype == "text" and isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif etype == "reasoning" and isinstance(part, dict) and isinstance(part.get("text"), str):
            reasoning_parts.append(part["text"])
        elif etype == "step_finish" and isinstance(part, dict):
            step_finish = part
        if etype in ("tool", "tool_use", "tool_call") or (
            isinstance(part, dict)
            and isinstance(part.get("type"), str)
            and part["type"].startswith("tool")
        ):
            tool_events.append(json.dumps(event, ensure_ascii=False)[:500])
    return {
        "raw_events": events,
        "text": "".join(text_parts).strip(),
        "reasoning": "".join(reasoning_parts).strip(),
        "step_finish": step_finish,
        "tool_polluted": bool(tool_events),
        "tool_events": tool_events,
        "session_ids": sorted(session_ids),
        "step_ids": sorted(step_ids),
        "message_ids": sorted(message_ids),
    }


def fetch_db_usage(db_path: str, session_id: str) -> dict[str, Any] | None:
    """从 OpenCode 会话库读取本次会话的 usage（只取 tokens/cost/finish/model）。

    仅在原始事件流缺少 step_finish 时作为补充来源；只按本次生成的 sessionID
    精确查询，不读取或转述任何对话内容。
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        cursor = connection.cursor()
        rows = cursor.execute(
            "select data from session_message where session_id=? and type='assistant' "
            "order by time_created desc limit 1",
            (session_id,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if not rows:
        return None
    try:
        data = json.loads(rows[0][0])
    except TypeError, ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
        return None
    return {
        "reason": data.get("finish"),
        "cost": data.get("cost"),
        "tokens": data["tokens"],
        "model": data.get("model"),
    }


@dataclass
class Sample:
    task_id: str
    case_id: str
    case_kind: str
    mode: str
    repeat: int
    model: str
    prompt: str
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    call_error: str | None = None
    parse_ok: bool = False
    parse_error: str | None = None
    route: str | None = None
    tie: bool = False
    margin: float | None = None
    entropy: float | None = None
    probabilities: dict[str, float] | None = None
    note: str | None = None
    text: str = ""
    reasoning: str = ""
    session_ids: list[str] = field(default_factory=list)
    step_ids: list[str] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    usage_source: str | None = None
    tool_polluted: bool = False
    tool_events: list[str] = field(default_factory=list)
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    expected_routes: tuple[str, ...] = ()
    correct: bool | None = None

    def to_record(self) -> dict[str, Any]:
        usage = self.usage or {}
        return {
            "task_id": self.task_id,
            "case_id": self.case_id,
            "case_kind": self.case_kind,
            "mode": self.mode,
            "repeat": self.repeat,
            "model": self.model,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "timed_out": self.timed_out,
            "call_error": self.call_error,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "route": self.route,
            "tie": self.tie,
            "margin": self.margin,
            "entropy": self.entropy,
            "probabilities": self.probabilities,
            "note": self.note,
            "text": self.text,
            "reasoning": self.reasoning,
            "session_ids": self.session_ids,
            "step_ids": self.step_ids,
            "message_ids": self.message_ids,
            "tool_polluted": self.tool_polluted,
            "tool_events": self.tool_events,
            "usage_available": bool(usage),
            "usage_source": self.usage_source,
            "step_finish_reason": usage.get("reason"),
            "cost": usage.get("cost"),
            "tokens": usage.get("tokens"),
            "expected_routes": list(self.expected_routes),
            "correct": self.correct,
            "raw_events": self.raw_events,
        }


def run_one(
    *,
    opencode_bin: str,
    workdir: Path,
    agent: str,
    model: str,
    prompt: str,
    timeout: float,
    thinking: bool,
) -> tuple[int | None, float, bool, str, str, str | None]:
    cmd = [
        opencode_bin,
        "run",
        "--standalone",
        "--agent",
        agent,
        "--model",
        model,
        "--format",
        "json",
    ]
    if thinking:
        cmd.append("--thinking")
    cmd.append(prompt)
    # OpenCode 按 PWD 发现项目配置；子进程必须显式对齐，否则找不到临时 agent。
    env = dict(os.environ)
    env["PWD"] = str(workdir)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.monotonic() - started
        return completed.returncode, elapsed, False, completed.stdout, completed.stderr, None
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - started
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return None, elapsed, True, stdout, stderr, f"timeout after {timeout}s"
    except OSError as error:
        elapsed = time.monotonic() - started
        return None, elapsed, False, "", "", f"{type(error).__name__}: {error}"


def ensure_agent_dir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "opencode.jsonc"
    desired = json.dumps(AGENT_CONFIG, ensure_ascii=False, indent=2)
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != desired:
            raise SystemExit(f"探针目录已有不同配置，拒绝覆盖：{config_path}")
        return
    config_path.write_text(desired, encoding="utf-8")


def build_tasks(
    cases: list[ProbeCase], modes: list[str], repeats: int, seed: int
) -> list[tuple[ProbeCase, str, int]]:
    tasks: list[tuple[ProbeCase, str, int]] = []
    for repeat in range(repeats):
        for case in cases:
            for mode in modes:
                tasks.append((case, mode, repeat))
    random.Random(seed).shuffle(tasks)
    return tasks


def _select_cases(
    cases: list[ProbeCase], only: list[str] | None, limit: int | None
) -> list[ProbeCase]:
    selected = cases
    if only:
        wanted = set(only)
        selected = [case for case in selected if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            raise SystemExit(f"未知用例 id：{sorted(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def write_report(
    *,
    outdir: Path,
    samples: list[Sample],
    model: str,
    modes: list[str],
    repeats: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    records = [sample.to_record() for sample in samples]
    usable = [s for s in records if s["parse_ok"] and not s["tool_polluted"]]
    call_failures = [s for s in records if s["call_error"] or s["timed_out"]]
    summary: dict[str, Any] = {
        "probe": "independent-system1-routing",
        "semantics": (
            "分布模式是「该事件对当前状态的影响所驱动的路径倾向」，"
            "不是世界事件发生概率；离线 argmax 评分不是正式路由策略。"
        ),
        "model": model,
        "modes": modes,
        "repeats": repeats,
        "seed": seed,
        "seed_scope": "仅用于实验顺序与两模式交错；无法设置模型真实温度/seed",
        "smoke": smoke,
        "total_samples": len(records),
        "call_failures": len(call_failures),
        "tool_polluted": sum(1 for s in records if s["tool_polluted"]),
        "usage_available": sum(1 for s in records if s["usage_available"]),
        "usage_sources": _usage_sources(records),
        "schema_valid": sum(1 for s in records if s["parse_ok"]),
        "schema_valid_rate": _rate(sum(1 for s in records if s["parse_ok"]), len(records)),
        "usable_samples": len(usable),
        "accuracy_over_all": _rate(sum(1 for s in records if s["correct"] is True), len(records)),
        "accuracy_over_usable": _rate(sum(1 for s in usable if s["correct"] is True), len(usable)),
        "by_mode": {},
        "by_kind": {},
        "confusion": {},
        "repeat_consistency": {},
        "latency_seconds": {
            "median": _median([s["elapsed_seconds"] for s in records]),
            "p95": _p95([s["elapsed_seconds"] for s in records]),
        },
        "cost_tokens": _cost_summary(records),
        "probability_stats": _probability_stats(usable),
        "errors": [
            {
                "task_id": s["task_id"],
                "case_id": s["case_id"],
                "mode": s["mode"],
                "repeat": s["repeat"],
                "call_error": s["call_error"],
                "parse_error": s["parse_error"],
                "timed_out": s["timed_out"],
                "tool_polluted": s["tool_polluted"],
            }
            for s in records
            if (s["call_error"] or s["parse_error"] or s["timed_out"] or s["tool_polluted"])
        ],
    }
    for mode in modes:
        mode_records = [s for s in records if s["mode"] == mode]
        mode_usable = [s for s in mode_records if s["parse_ok"] and not s["tool_polluted"]]
        summary["by_mode"][mode] = {
            "samples": len(mode_records),
            "schema_valid": sum(1 for s in mode_records if s["parse_ok"]),
            "ties": sum(1 for s in mode_records if s["tie"]),
            "accuracy_over_all": _rate(
                sum(1 for s in mode_records if s["correct"] is True), len(mode_records)
            ),
            "accuracy_over_usable": _rate(
                sum(1 for s in mode_usable if s["correct"] is True), len(mode_usable)
            ),
            "confusion": _confusion(mode_records),
            "latency_seconds": {
                "median": _median([s["elapsed_seconds"] for s in mode_records]),
                "p95": _p95([s["elapsed_seconds"] for s in mode_records]),
            },
            "cost_tokens": _cost_summary(mode_records),
            "repeat_consistency": _repeat_consistency(mode_records),
            "by_kind": {
                kind: {
                    "samples": sum(s["case_kind"] == kind for s in mode_records),
                    "accuracy_over_all": _rate(
                        sum(s["correct"] is True and s["case_kind"] == kind for s in mode_records),
                        sum(s["case_kind"] == kind for s in mode_records),
                    ),
                }
                for kind in ("clear", "boundary")
            },
        }
    for kind in ("clear", "boundary"):
        kind_records = [s for s in records if s["case_kind"] == kind]
        kind_usable = [s for s in kind_records if s["parse_ok"] and not s["tool_polluted"]]
        summary["by_kind"][kind] = {
            "samples": len(kind_records),
            "accuracy_over_all": _rate(
                sum(1 for s in kind_records if s["correct"] is True), len(kind_records)
            ),
            "accuracy_over_usable": _rate(
                sum(1 for s in kind_usable if s["correct"] is True), len(kind_usable)
            ),
        }
    summary["confusion"] = _confusion(records)
    summary["repeat_consistency"] = _repeat_consistency(records)
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "report.md").write_text(_render_markdown(summary, records, samples), encoding="utf-8")
    return summary


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _usage_sources(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        source = record.get("usage_source") or "unavailable"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _confusion(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {route: {} for route in ROUTES}
    for record in records:
        predicted = (
            record["route"] if record["parse_ok"] and not record["tool_polluted"] else "invalid"
        )
        if predicted is None:
            predicted = "tie"
        expected = "/".join(record["expected_routes"])
        table.setdefault(expected, {})
        table[expected][predicted] = table[expected].get(predicted, 0) + 1
    return table


def _repeat_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["case_id"], record["mode"]), []).append(record)
    total = 0
    covered = 0
    consistent = 0
    unstable: list[str] = []
    for (case_id, mode), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        total += 1
        valid = [
            item
            for item in items
            if item["parse_ok"] and not item["tool_polluted"] and item["route"] in ROUTES
        ]
        if len(valid) != len(items):
            continue
        covered += 1
        picks = {item["route"] for item in valid}
        if len(picks) == 1:
            consistent += 1
        else:
            unstable.append(f"{case_id}/{mode}: {sorted(picks)}")
    return {
        "groups": total,
        "covered_groups": covered,
        "consistent_groups": consistent,
        "coverage": _rate(covered, total),
        "consistency_rate": _rate(consistent, covered),
        "unstable": unstable,
    }


def _cost_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [r["cost"] for r in records if isinstance(r.get("cost"), (int, float))]
    tokens = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    field_counts = dict.fromkeys(tokens, 0)
    counted = 0
    for record in records:
        usage = record.get("tokens")
        if not isinstance(usage, dict):
            continue
        counted += 1
        for key in ("input", "output", "reasoning"):
            value = usage.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                tokens[key] += int(value)
                field_counts[key] += 1
        cache = usage.get("cache")
        if isinstance(cache, dict):
            for key in ("read", "write"):
                value = cache.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    tokens[f"cache_{key}"] += int(value)
                    field_counts[f"cache_{key}"] += 1
    return {
        "samples_with_usage": counted,
        "total_cost": sum(costs) if costs else None,
        "tokens_sum": {key: value if field_counts[key] else None for key, value in tokens.items()}
        if counted
        else None,
        "tokens_field_coverage": field_counts,
        "note": "仅汇总原始 usage 字段（step_finish 或会话库补取），未把 cache 忽略后当作总 token。",
    }


def _probability_stats(usable: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [r["margin"] for r in usable if isinstance(r.get("margin"), (int, float))]
    entropies = [r["entropy"] for r in usable if isinstance(r.get("entropy"), (int, float))]
    return {
        "samples_with_distribution": len(margins),
        "margin_definition": "argmax 概率减去次大概率；并列最大时为 None。",
        "entropy_definition": "三条路径概率的 Shannon 熵（bits），p=0 项不计。",
        "margin_median": _median(margins),
        "entropy_median": _median(entropies),
    }


def _render_markdown(
    summary: dict[str, Any], records: list[dict[str, Any]], samples: list[Sample]
) -> str:
    lines = [
        "# System1 路由探针报告",
        "",
        f"- 模型：`{summary['model']}`",
        f"- 模式：{', '.join(summary['modes'])}；每例重复 {summary['repeats']} 次",
        f"- 样本总数：{summary['total_samples']}；调用失败：{summary['call_failures']}；"
        f"工具污染：{summary['tool_polluted']}；有 usage：{summary['usage_available']}",
        f"- 合法 schema 率：{_fmt(summary['schema_valid_rate'])}",
        f"- 准确率（失败计入分母）：{_fmt(summary['accuracy_over_all'])}；"
        f"有效样本准确率：{_fmt(summary['accuracy_over_usable'])}",
        f"- 耗时中位数/p95（秒）：{_fmt(summary['latency_seconds']['median'])} / "
        f"{_fmt(summary['latency_seconds']['p95'])}",
        "",
        "> 说明：这是独立的 System1 探针。分布模式是「该事件对当前状态的影响所驱动的路径倾向」，"
        "不是世界事件发生概率；离线 argmax 评分不是正式程序路由策略。小样本不能宣称已证明"
        "校准或统计显著。",
        "",
        "## 分模式",
        "",
    ]
    for mode, data in summary["by_mode"].items():
        lines += [
            f"### {mode}",
            "",
            f"- 样本 {data['samples']}；合法 schema {data['schema_valid']}；并列最大 {data['ties']}",
            f"- 准确率（全部）：{_fmt(data['accuracy_over_all'])}；"
            f"准确率（有效）：{_fmt(data['accuracy_over_usable'])}",
            f"- 耗时中位数/p95（秒）：{_fmt(data['latency_seconds']['median'])} / "
            f"{_fmt(data['latency_seconds']['p95'])}",
            f"- usage样本：{data['cost_tokens']['samples_with_usage']}；"
            f"tokens：{data['cost_tokens']['tokens_sum']}；"
            f"cost：{data['cost_tokens']['total_cost']}",
            "",
        ]
    lines += [
        "## 分案例类型",
        "",
        "| 类型 | 样本 | 准确率(全部) | 准确率(有效) |",
        "| --- | --- | --- | --- |",
    ]
    for kind, data in summary["by_kind"].items():
        lines.append(
            f"| {kind} | {data['samples']} | {_fmt(data['accuracy_over_all'])} | "
            f"{_fmt(data['accuracy_over_usable'])} |"
        )
    lines += [
        "",
        "## 重复一致性",
        "",
        f"- 一致组：{summary['repeat_consistency']['consistent_groups']}/"
        f"{summary['repeat_consistency']['groups']}；覆盖率 "
        f"{_fmt(summary['repeat_consistency']['coverage'])}",
        f"- 完整有效重复组中的一致率："
        f"{_fmt(summary['repeat_consistency']['consistency_rate'])}；"
        "失败、并列、仅一次结果的组不算稳定。",
    ]
    for item in summary["repeat_consistency"]["unstable"]:
        lines.append(f"- 不稳定：{item}")
    lines += [
        "",
        "## 概率 margin / entropy（分布模式）",
        "",
        f"- margin：{summary['probability_stats']['margin_definition']}",
        f"- entropy：{summary['probability_stats']['entropy_definition']}",
        f"- margin 中位数：{_fmt(summary['probability_stats']['margin_median'])}；"
        f"entropy 中位数：{_fmt(summary['probability_stats']['entropy_median'])}",
        "",
        "## tokens / cost",
        "",
        f"- {summary['cost_tokens']['note']}",
        f"- usage 来源：{summary['usage_sources']}",
        f"- 有 usage 的样本：{summary['cost_tokens']['samples_with_usage']}",
        f"- 合计 cost：{summary['cost_tokens']['total_cost']}",
        f"- 合计 tokens：{summary['cost_tokens']['tokens_sum']}",
        "",
        "## 逐条结果",
        "",
        "| case | kind | mode | rep | route | 倾向(C/P/T) | tie | 期望 | 符合标签 | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in sorted(records, key=lambda r: (r["case_id"], r["mode"], r["repeat"])):
        error = (
            record["call_error"]
            or record["parse_error"]
            or ("tool-polluted" if record["tool_polluted"] else "")
        )
        probs = record.get("probabilities")
        distribution = "/".join(f"{probs[route]:.2f}" for route in ROUTES) if probs else "—"
        lines.append(
            f"| {record['case_id']} | {record['case_kind']} | {record['mode']} | "
            f"{record['repeat']} | {record['route']} | {distribution} | {record['tie']} | "
            f"{'/'.join(record['expected_routes'])} | {record['correct']} | {error} |"
        )
    lines += [
        "",
        "## 原始输出示例",
        "",
    ]
    for sample in samples[:6]:
        lines += [
            f"### {sample.task_id}",
            "",
            "```json",
            (sample.text or "（无文本输出）")[:1200],
            "```",
            "",
        ]
    lines += [
        "## 限制",
        "",
        "- 样本量小，不能据此宣称校准或统计显著；准确率只是对人工标签的离线比对。",
        "- seed 只控制实验顺序与两模式交错；无法设置模型真实温度/seed，模型侧仍有随机性。",
        "- 标签由人工/主代理审阅；边界案例允许多个合法路由，不做唯一标准。",
        "- 若某次调用未产生 step_finish，则从本次会话库按 sessionID 精确补取 usage"
        "（只取 tokens/cost/finish/model，不读取对话内容）；仍取不到则记为不可用，不做估算。",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case", action="append", default=None, help="只跑指定用例 id，可重复")
    parser.add_argument("--limit", type=int, default=None, help="最多取前 N 个用例")
    parser.add_argument("--mode", choices=("field", "distribution", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--opencode", default=DEFAULT_OPENCODE)
    parser.add_argument("--workdir", default="/tmp/opencode/system1-probe")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--usage-db", default=DEFAULT_USAGE_DB)
    parser.add_argument(
        "--no-db-usage",
        action="store_true",
        help="不在缺少 step_finish 时从会话库补取 usage",
    )
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--no-thinking", action="store_true", help="不请求 reasoning 事件")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 案例 × 2 模式 × 1 次")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats 必须是正整数")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须是正整数")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout 必须是有限正数")
    cases = load_cases(CASES_PATH)
    selected = _select_cases(cases, args.case, args.limit)
    modes = ["field", "distribution"] if args.mode == "both" else [args.mode]
    repeats = args.repeats
    if args.smoke:
        selected = selected[:2]
        modes = ["field", "distribution"]
        repeats = 1
    tasks = build_tasks(selected, modes, repeats, args.seed)

    if args.dry_run:
        print(f"用例数：{len(selected)}；模式：{modes}；重复：{repeats}；任务数：{len(tasks)}")
        print(f"模型：{args.model}；agent：{args.agent}；opencode：{args.opencode}")
        for case, mode, repeat in tasks[:2]:
            print("=" * 60)
            print(f"task {case.id}/{mode}/repeat{repeat}")
            print(build_prompt(case, mode))
        return 0

    workdir = Path(args.workdir)
    ensure_agent_dir(workdir)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path("sceneweave-runs") / "system1-probe" / timestamp)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    samples_path = outdir / "samples.jsonl"

    samples: list[Sample] = []
    print(f"输出目录：{outdir}")
    print(f"任务数：{len(tasks)}（模型 {args.model}）")
    for index, (case, mode, repeat) in enumerate(tasks, start=1):
        task_id = f"{case.id}/{mode}/r{repeat}"
        prompt = build_prompt(case, mode)
        print(f"[{index}/{len(tasks)}] {task_id} ...", flush=True)
        exit_code, elapsed, timed_out, stdout, stderr, call_error = run_one(
            opencode_bin=args.opencode,
            workdir=workdir,
            agent=args.agent,
            model=args.model,
            prompt=prompt,
            timeout=args.timeout,
            thinking=not args.no_thinking,
        )
        parsed = parse_events(stdout)
        sample = Sample(
            task_id=task_id,
            case_id=case.id,
            case_kind=case.kind,
            mode=mode,
            repeat=repeat,
            model=args.model,
            prompt=prompt,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            call_error=call_error
            if call_error
            else (None if exit_code == 0 else f"exit={exit_code}"),
            text=parsed["text"],
            reasoning=parsed["reasoning"],
            session_ids=parsed["session_ids"],
            step_ids=parsed["step_ids"],
            message_ids=parsed["message_ids"],
            tool_polluted=parsed["tool_polluted"],
            tool_events=parsed["tool_events"],
            raw_events=parsed["raw_events"],
            expected_routes=case.expected_routes,
        )
        step_finish = parsed["step_finish"]
        if step_finish is not None:
            sample.usage = {
                "reason": step_finish.get("reason"),
                "cost": step_finish.get("cost"),
                "tokens": step_finish.get("tokens"),
            }
            sample.usage_source = "step_finish"
        elif parsed["session_ids"] and not args.no_db_usage:
            # 原始事件流可能因极短响应缺少 step_finish；从本次会话库补取 usage。
            db_usage = fetch_db_usage(args.usage_db, parsed["session_ids"][-1])
            if db_usage is not None:
                sample.usage = db_usage
                sample.usage_source = "session-db"
        if sample.call_error is None:
            try:
                result = (
                    parse_field(sample.text) if mode == "field" else parse_distribution(sample.text)
                )
                sample.parse_ok = True
                sample.route = result["route"]
                sample.tie = result["tie"]
                sample.margin = result["margin"]
                sample.entropy = result["entropy"]
                sample.probabilities = result.get("probabilities")
                sample.note = result["note"]
                sample.correct = (
                    sample.route in case.expected_routes and not sample.tool_polluted
                    if sample.route is not None
                    else False
                )
            except ValueError as error:
                sample.parse_error = str(error)
                sample.correct = False
        else:
            sample.correct = False
        samples.append(sample)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
        status = "ok" if sample.parse_ok else ("call-error" if sample.call_error else "parse-error")
        print(f"    {status}: route={sample.route} tie={sample.tie} {sample.elapsed_seconds:.1f}s")

    summary = write_report(
        outdir=outdir,
        samples=samples,
        model=args.model,
        modes=modes,
        repeats=repeats,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(f"报告：{outdir / 'report.md'}")
    print(
        f"完成：{summary['total_samples']} 次调用，调用失败 {summary['call_failures']}，"
        f"合法 schema {summary['schema_valid']}，工具污染 {summary['tool_polluted']}"
    )
    return (
        1
        if (
            summary["call_failures"]
            or summary["tool_polluted"]
            or summary["schema_valid"] != summary["total_samples"]
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
