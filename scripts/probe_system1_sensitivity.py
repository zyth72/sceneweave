#!/usr/bin/env python3
"""System1 敏感度探针（第二轮）：event intensity sweep + role sensitivity。

独立于真实模拟：只调用 System1，绝不启动 System2/System3/翻译器/World/engine，
不生成动作或新意向文本，只对给定状态输出三路倾向分布。正式 System1 仍是严格
枚举路由，本探针不接入正式循环。

真实角色来源（用户已明确授权，只读）：
``/home/zyth/lin/worldbook/persona/{贝尔法斯特,企业,恶毒}.json``。
JSON 会整体读入以校验结构；只把固定白名单字段（personality 的核心特征/优点/
缺点（性格缺陷）/习惯或怪癖、worldview 的道德准则、innerDrive 的核心动机/短期
目标/恐惧与禁忌）抽取并注入 prompt，不抽取或注入任何非白名单字段（外貌/服装/
三围/爱情/亲密台词/关系底账等），不写 ~/lin。抽取文本与来源 sha256 写入输出目录
manifest；私有角色正文不入 Git。

复用 ``scripts/probe_system1_routing.py`` 的严格解析、无工具 OpenCode 调用、usage
补取与统计函数（median/p95、cost/tokens、usage 来源），不复制其整份实现。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_system1_routing as routing  # noqa: E402

# 复用第一轮探针的统计函数，保持两次探针口径一致。
_median = routing._median
_p95 = routing._p95
_rate = routing._rate
_cost_summary = routing._cost_summary
_usage_sources = routing._usage_sources

PERSONA_FIELDS: dict[str, tuple[str, ...]] = {
    "personality": ("核心特征", "优点", "缺点（性格缺陷）", "习惯或怪癖"),
    "worldview": ("道德准则",),
    "innerDrive": ("核心动机", "短期目标", "恐惧与禁忌"),
}
DEFAULT_PERSONA_DIR = Path("/home/zyth/lin/worldbook/persona")
SENS_CASES_PATH = Path(__file__).with_name("probe_system1_sensitivity_cases.json")
DEFAULT_WORKDIR = "/tmp/opencode/system1-sensitivity"
PROFILE_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "real_name": None, "file": None},
    {"id": "P1", "real_name": "贝尔法斯特", "file": "贝尔法斯特.json"},
    {"id": "P2", "real_name": "企业", "file": "企业.json"},
    {"id": "P3", "real_name": "恶毒", "file": "恶毒.json"},
)


class PersonaSourceError(RuntimeError):
    """人格来源缺失或字段不完整；不得用杜撰默认替代。"""


@dataclass(frozen=True, slots=True)
class PersonaExcerpt:
    label: str
    real_name: str | None
    source_path: str | None
    sha256: str | None
    fields: tuple[tuple[str, str], ...]
    text: str

    @property
    def length(self) -> int:
        return len(self.text)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "real_name": self.real_name,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "text_length": self.length,
            "field_paths": [path for path, _ in self.fields],
            "text": self.text,
        }


def load_persona(spec: dict[str, Any], persona_dir: Path = DEFAULT_PERSONA_DIR) -> PersonaExcerpt:
    """按白名单摘取人格字段；缺文件/缺字段明确报错，不造默认。"""
    if spec["file"] is None:
        return PersonaExcerpt(
            label=str(spec["id"]),
            real_name=None,
            source_path=None,
            sha256=None,
            fields=(),
            text="（无额外人格设定）",
        )
    path = persona_dir / str(spec["file"])
    if not path.exists():
        raise PersonaSourceError(f"缺少人格来源文件：{path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as error:
        raise PersonaSourceError(f"人格来源不是合法 JSON：{path}") from error
    if not isinstance(data, dict):
        raise PersonaSourceError(f"人格来源根必须是对象：{path}")
    name = spec.get("real_name")
    fields: list[tuple[str, str]] = []
    for top, keys in PERSONA_FIELDS.items():
        section = data.get(top)
        if not isinstance(section, dict):
            raise PersonaSourceError(f"人格来源缺少 {top}：{path}")
        for key in keys:
            value = section.get(key)
            if not isinstance(value, str) or not value.strip():
                raise PersonaSourceError(f"人格来源缺少字段 {top}.{key}：{path}")
            fields.append((f"{top}.{key}", value.strip()))
    lines = []
    for field_path, value in fields:
        label = field_path.split(".")[-1]
        redacted = value.replace(str(name), "本角色") if name else value
        lines.append(f"- {label}：{redacted}")
    return PersonaExcerpt(
        label=str(spec["id"]),
        real_name=None if name is None else str(name),
        source_path=str(path),
        sha256=sha,
        fields=tuple(fields),
        text="\n".join(lines),
    )


@dataclass(frozen=True, slots=True)
class SensCase:
    id: str
    suite: str
    title: str
    event: str
    intention: str
    plan: tuple[str, ...]
    known_actions: tuple[str, ...]
    memory: tuple[str, ...]
    expected_routes: tuple[str, ...]
    label_rationale: str


def load_cases(path: Path = SENS_CASES_PATH) -> list[SensCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise SystemExit(f"用例文件格式错误：{path}")
    cases: list[SensCase] = []
    for raw in data["cases"]:
        expected = tuple(raw.get("expected_routes", ()))
        for route in expected:
            if route not in routing.ROUTES:
                raise SystemExit(f"用例 {raw.get('id')} 含未知期望路由：{route}")
        if raw["suite"] not in ("intensity", "influence", "unconfirmed"):
            raise SystemExit(f"用例 {raw.get('id')} 的 suite 非法")
        for action in raw.get("known_actions", ()):
            text = str(action)
            if "a2" in text and "registered" in text:
                raise SystemExit(
                    f"用例 {raw.get('id')} 的未来 a2 不得是 registered，应为 saved/private/unsubmitted"
                )
        cases.append(
            SensCase(
                id=str(raw["id"]),
                suite=str(raw["suite"]),
                title=str(raw["title"]),
                event=str(raw["event"]),
                intention=str(raw["intention"]),
                plan=tuple(str(item) for item in raw.get("plan", ())),
                known_actions=tuple(str(item) for item in raw.get("known_actions", ())),
                memory=tuple(str(item) for item in raw.get("memory", ())),
                expected_routes=expected,
                label_rationale=str(raw["label_rationale"]),
            )
        )
    if len(cases) != 12:
        raise SystemExit(f"期望 12 个控制状态，实际 {len(cases)}")
    return cases


def build_prompt(case: SensCase, persona: PersonaExcerpt) -> str:
    """构造 System1 输入：私有设定 + 该次事件 + 本角色状态；不含标签/期望。"""
    header = (
        f"【本角色私有设定（来源编号 {persona.label}）】"
        if persona.real_name
        else "【本角色私有设定】"
    )
    plan_lines = [f"  - {item}" for item in case.plan] or ["  - 无（尚无已保存计划）"]
    known_lines = [f"  - {item}" for item in case.known_actions] or ["  - 无"]
    memory_lines = [f"  - {item}" for item in case.memory] or ["  - 无"]
    lines = [
        "你是角色内部 System1 的分流判断器。只依据下面这次事件和本角色的私有状态，"
        "判断角色在内部规划上应当走哪条路。不要生成意向文本，不要生成任何动作。",
        "",
        header,
        persona.text,
        "",
        "【本次事件】",
        f"  {case.event}",
        "",
        "【本角色当前私有状态】",
        f"  当前意向：{case.intention}",
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
        "【输出】根据这次事件对当前状态的影响，给出 continue/replan/rethink 三条路径的"
        "倾向分布。这是模型自报的路径倾向，不是世界事件发生概率，也不是随机抽签；"
        "不要直接给出 route。只输出一个 JSON 对象，格式：",
        '{"probabilities": {"continue": 0.0, "replan": 0.0, "rethink": 0.0}, "note": "简短说明"}',
        "三个键都要给出，数值在 0 到 1 之间，三者之和为 1。",
    ]
    return "\n".join(lines)


@dataclass
class SensSample:
    task_id: str
    case_id: str
    suite: str
    profile_label: str
    profile_real_name: str | None
    profile_len: int
    profile_sha256: str | None
    repeat: int
    model: str
    prompt: str
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    call_error: str | None = None
    parse_ok: bool = False
    parse_error: str | None = None
    tie: bool = False
    probabilities: dict[str, float] | None = None
    argmax: str | None = None
    note: str | None = None
    text: str = ""
    reasoning: str = ""
    session_ids: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    usage_source: str | None = None
    tool_polluted: bool = False
    tool_events: list[str] = field(default_factory=list)
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    expected_routes: tuple[str, ...] = ()
    in_expected: bool | None = None

    def to_record(self) -> dict[str, Any]:
        usage = self.usage or {}
        return {
            "task_id": self.task_id,
            "case_id": self.case_id,
            "suite": self.suite,
            "profile_label": self.profile_label,
            "profile_real_name": self.profile_real_name,
            "profile_len": self.profile_len,
            "profile_sha256": self.profile_sha256,
            "repeat": self.repeat,
            "model": self.model,
            "prompt": self.prompt,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "timed_out": self.timed_out,
            "call_error": self.call_error,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "tie": self.tie,
            "probabilities": self.probabilities,
            "argmax": self.argmax,
            "note": self.note,
            "text": self.text,
            "reasoning": self.reasoning,
            "session_ids": self.session_ids,
            "tool_polluted": self.tool_polluted,
            "tool_events": self.tool_events,
            "usage_available": bool(usage),
            "usage_source": self.usage_source,
            "cost": usage.get("cost"),
            "tokens": usage.get("tokens"),
            "expected_routes": list(self.expected_routes),
            "in_expected": self.in_expected,
            "raw_events": self.raw_events,
        }


def _select_cases(
    cases: list[SensCase], suite: str, limit: int | None, smoke: bool
) -> list[SensCase]:
    selected = [case for case in cases if suite == "all" or case.suite == suite]
    if smoke:
        first_per_suite: dict[str, SensCase] = {}
        for case in selected:
            first_per_suite.setdefault(case.suite, case)
        selected = list(first_per_suite.values())
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("没有匹配的控制状态")
    return selected


def _select_profiles(only: list[str] | None, smoke: bool) -> list[dict[str, Any]]:
    specs = list(PROFILE_SPECS)
    if only:
        wanted = list(dict.fromkeys(str(item) for item in only))
        known_ids = {str(spec["id"]) for spec in specs}
        known_names = {str(spec["real_name"]) for spec in specs if spec.get("real_name")}
        unknown = [item for item in wanted if item not in known_ids and item not in known_names]
        if unknown:
            raise SystemExit(f"未知 profile：{sorted(unknown)}")
        specs = [
            spec for spec in specs if str(spec["id"]) in wanted or spec.get("real_name") in wanted
        ]
    if smoke:
        specs = [spec for spec in specs if spec["id"] in ("baseline", "P1")]
    if not specs:
        raise SystemExit("没有匹配的 profile")
    return specs


def build_tasks(
    cases: list[SensCase], profiles: list[dict[str, Any]], repeats: int, seed: int
) -> list[tuple[SensCase, dict[str, Any], int]]:
    tasks: list[tuple[SensCase, dict[str, Any], int]] = []
    for repeat in range(repeats):
        for case in cases:
            for profile in profiles:
                tasks.append((case, profile, repeat))
    random.Random(seed).shuffle(tasks)
    return tasks


def _valid(sample: SensSample) -> bool:
    """有效样本：调用成功、退出码 0、无超时、无工具污染且 schema 合法。"""
    return (
        sample.parse_ok
        and sample.probabilities is not None
        and sample.call_error is None
        and not sample.timed_out
        and sample.exit_code == 0
        and not sample.tool_polluted
    )


def _mean_distribution(samples: list[SensSample]) -> dict[str, float] | None:
    usable = [s for s in samples if _valid(s)]
    if not usable:
        return None
    return {
        route: statistics.fmean(s.probabilities[route] for s in usable)  # type: ignore[index]
        for route in routing.ROUTES
    }


def _route_range(samples: list[SensSample]) -> dict[str, float | None]:
    """每路径在全部有效重复上的 max-min（全重复范围），不是相对第一次。"""
    usable = [s for s in samples if _valid(s)]
    if len(usable) < 2:
        return dict.fromkeys(routing.ROUTES)
    return {
        route: max(s.probabilities[route] for s in usable)  # type: ignore[index]
        - min(s.probabilities[route] for s in usable)  # type: ignore[index]
        for route in routing.ROUTES
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _latency_cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [record["elapsed_seconds"] for record in records]
    cost = _cost_summary(records)
    return {
        "latency_seconds": {"median": _median(elapsed), "p95": _p95(elapsed)},
        "cost_tokens": cost,
        "usage_coverage": _rate(cost["samples_with_usage"], len(records)),
    }


def _trend_stats(levels: list[dict[str, Any]]) -> dict[str, Any]:
    means = [level["mean"] for level in levels]
    complete = [mean for mean in means if mean is not None]
    stats: dict[str, Any] = {
        "complete_levels": len(complete),
        "expected_levels": len(means),
    }
    if complete and len(complete) == len(means) and len(complete) >= 2:
        index = [float(i) for i in range(len(complete))]
        for route in routing.ROUTES:
            values = [mean[route] for mean in complete]
            adjacent = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            stats[f"{route}_spearman"] = _spearman(index, values)
            stats[f"{route}_adjacent_delta"] = adjacent
            stats[f"{route}_sign_changes"] = sum(
                1 for i in range(len(adjacent) - 1) if adjacent[i] * adjacent[i + 1] < 0
            )
    return stats


def write_report(
    *,
    outdir: Path,
    samples: list[SensSample],
    personas: list[PersonaExcerpt],
    cases: list[SensCase],
    model: str,
    suite: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    records = [s.to_record() for s in samples]
    valid = [s for s in samples if _valid(s)]
    call_failures = [r for r in records if r["call_error"] or r["timed_out"]]
    schema_failures = [r for r in records if not r["parse_ok"]]
    polluted = [r for r in records if r["tool_polluted"]]
    nonzero_exit = [r for r in records if r["exit_code"] not in (0, None)]
    expected_samples = len(cases) * len(personas) * repeats
    expected_keys = {
        (case.id, persona.label, repeat)
        for case in cases
        for persona in personas
        for repeat in range(repeats)
    }
    actual_keys = {(s.case_id, s.profile_label, s.repeat) for s in samples}
    valid_keys = {(s.case_id, s.profile_label, s.repeat) for s in valid}
    missing_samples = sorted(
        f"{case_id}/{label}/r{repeat}" for case_id, label, repeat in (expected_keys - actual_keys)
    )
    invalid_samples = sorted(
        f"{case_id}/{label}/r{repeat}"
        for case_id, label, repeat in (
            {(s.case_id, s.profile_label, s.repeat) for s in samples} - valid_keys
        )
    )
    expected_groups = len(cases) * len(personas)
    complete_groups = sum(
        1
        for case in cases
        for persona in personas
        if sum(1 for s in valid if s.case_id == case.id and s.profile_label == persona.label)
        == repeats
    )
    overall = _latency_cost(records)
    summary: dict[str, Any] = {
        "probe": "independent-system1-sensitivity",
        "semantics": (
            "只输出三路倾向分布（事件影响驱动的路径倾向），不是世界事件概率；"
            "argmax 仅离线展示/评分，不抽样、不制定正式路由策略。自称概率未校准。"
        ),
        "model": model,
        "suite_filter": suite,
        "repeats": repeats,
        "seed": seed,
        "seed_scope": "仅用于实验顺序；无法设置模型真实温度/seed",
        "expected_samples": expected_samples,
        "actual_samples": len(records),
        "total_samples": len(records),
        "valid_samples": len(valid),
        "coverage": _rate(len(valid), expected_samples),
        "coverage_over_actual": _rate(len(valid), len(records)),
        "complete_groups": complete_groups,
        "expected_groups": expected_groups,
        "call_failures": len(call_failures),
        "timed_out": sum(1 for r in records if r["timed_out"]),
        "schema_valid": sum(1 for r in records if r["parse_ok"]),
        "schema_failures": len(schema_failures),
        "nonzero_exit": len(nonzero_exit),
        "tool_polluted": len(polluted),
        "ties": sum(1 for r in records if r["tie"]),
        "usage_available": sum(1 for r in records if r["usage_available"]),
        "usage_sources": _usage_sources(records),
        "latency_seconds": overall["latency_seconds"],
        "cost_tokens": overall["cost_tokens"],
        "usage_coverage": overall["usage_coverage"],
        "missing_samples": missing_samples,
        "invalid_samples": invalid_samples,
        "profiles": [
            {
                "label": persona.label,
                "real_name": persona.real_name,
                "text_length": persona.length,
                "sha256": persona.sha256,
            }
            for persona in personas
        ],
        "by_suite": {},
        "by_profile": {},
        "by_case": {},
        "intensity_trend": {},
        "influence_layer": {},
        "profile_minus_baseline": {},
        "repeat_fluctuation": {},
        "errors": [
            {
                "task_id": r["task_id"],
                "call_error": r["call_error"],
                "parse_error": r["parse_error"],
                "timed_out": r["timed_out"],
                "tool_polluted": r["tool_polluted"],
            }
            for r in records
            if r["call_error"] or r["parse_error"] or r["timed_out"] or r["tool_polluted"]
        ],
    }
    for suite_name in sorted({s.suite for s in samples}):
        subset = [s for s in samples if s.suite == suite_name]
        suite_records = [s.to_record() for s in subset]
        suite_expected = (
            sum(1 for case in cases if case.suite == suite_name) * len(personas) * repeats
        )
        summary["by_suite"][suite_name] = {
            "samples": len(subset),
            "expected": suite_expected,
            "valid": sum(1 for s in subset if _valid(s)),
            "mean": _mean_distribution(subset),
            **_latency_cost(suite_records),
        }
    for persona in personas:
        subset = [s for s in samples if s.profile_label == persona.label]
        profile_records = [s.to_record() for s in subset]
        profile_expected = len(cases) * repeats
        summary["by_profile"][persona.label] = {
            "real_name": persona.real_name,
            "profile_len": persona.length,
            "samples": len(subset),
            "expected": profile_expected,
            "valid": sum(1 for s in subset if _valid(s)),
            "mean": _mean_distribution(subset),
            **_latency_cost(profile_records),
        }
    for case in cases:
        subset = [s for s in samples if s.case_id == case.id]
        if not subset:
            continue
        by_profile: dict[str, Any] = {}
        for persona in personas:
            persona_subset = [s for s in subset if s.profile_label == persona.label]
            if not persona_subset:
                continue
            by_profile[persona.label] = {
                "mean": _mean_distribution(persona_subset),
                "valid": sum(1 for s in persona_subset if _valid(s)),
                "expected": repeats,
                "raw": [
                    {
                        "repeat": s.repeat,
                        "probabilities": s.probabilities,
                        "argmax": s.argmax,
                        "tie": s.tie,
                        "valid": _valid(s),
                    }
                    for s in sorted(persona_subset, key=lambda item: item.repeat)
                ],
            }
        summary["by_case"][case.id] = {
            "suite": case.suite,
            "expected_routes": list(case.expected_routes),
            "mean": _mean_distribution(subset),
            "by_profile": by_profile,
        }
    summary["intensity_trend"] = _intensity_trend(samples, cases, personas)
    summary["influence_layer"] = _influence_layer(samples, cases)
    summary["profile_minus_baseline"] = _profile_minus_baseline(samples, cases, personas)
    summary["repeat_fluctuation"] = _repeat_fluctuation(samples)
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _render_markdown(summary, samples, cases, personas)
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = {
            "说明": "运行当时保存的参数与源码哈希；不代表本次重导出的生成器版本",
            "实验参数": {
                key: value
                for key, value in manifest.items()
                if key not in ("cases", "personas", "source_hashes")
            },
            "运行时源码": manifest.get("source_hashes"),
            "原始记录文件sha256": {
                name: hashlib.sha256((outdir / name).read_bytes()).hexdigest()
                for name in ("manifest.json", "samples.jsonl")
                if (outdir / name).exists()
            },
        }
        report += (
            "\n## 实验记录来源\n\n"
            + _fenced(json.dumps(provenance, ensure_ascii=False, indent=2), "json")
            + "\n"
        )
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return summary


def regenerate_report(outdir: Path) -> dict[str, Any]:
    """只用保存的实验输入输出重导出，不访问模型、人格仓库或当前用例文件。"""
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["probe"] != "independent-system1-sensitivity":
        raise ValueError("该目录不是 System1 敏感度探针记录")
    cases = [
        SensCase(
            **{
                **item,
                **{
                    key: tuple(item[key])
                    for key in ("plan", "known_actions", "memory", "expected_routes")
                },
            }
        )
        for item in manifest["cases"]
    ]
    personas = [
        PersonaExcerpt(
            label=item["label"],
            real_name=item["real_name"],
            source_path=item["source_path"],
            sha256=item["sha256"],
            fields=(),  # 展示直接用已保存的摘录；不从当前源文件重新抽取。
            text=item["text"],
        )
        for item in manifest["personas"]
    ]
    sample_fields = {item.name for item in fields(SensSample)}
    samples = []
    for line in (outdir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        values = {key: value for key, value in record.items() if key in sample_fields}
        values["expected_routes"] = tuple(record["expected_routes"])
        values["usage"] = (
            {"cost": record["cost"], "tokens": record["tokens"]}
            if record["usage_available"]
            else None
        )
        samples.append(SensSample(**values))
    return write_report(
        outdir=outdir,
        samples=samples,
        personas=personas,
        cases=cases,
        model=manifest["model"],
        suite=manifest["suite_filter"],
        repeats=manifest["repeats"],
        seed=manifest["seed"],
    )


def _intensity_trend(
    samples: list[SensSample], cases: list[SensCase], personas: list[PersonaExcerpt]
) -> dict[str, Any]:
    intensity_cases = [case for case in cases if case.suite == "intensity"]
    if not intensity_cases:
        return {}

    def levels_for(predicate: Any) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case.id,
                "mean": _mean_distribution(
                    [s for s in samples if s.case_id == case.id and predicate(s)]
                ),
            }
            for case in intensity_cases
        ]

    pooled_levels = levels_for(lambda _sample: True)
    by_profile: dict[str, Any] = {}
    for persona in personas:
        label = persona.label
        levels = levels_for(lambda sample, label=label: sample.profile_label == label)
        by_profile[label] = {
            "real_name": persona.real_name,
            "levels": levels,
            "stats": _trend_stats(levels),
        }
    result: dict[str, Any] = {
        "levels": pooled_levels,
        "by_profile": by_profile,
        "note": (
            "逐 profile 各自给出 6 级趋势，不能只看混合全部 profile 的池化结果；"
            "不硬规定单调，只报告趋势与符号变化。"
        ),
    }
    result.update(_trend_stats(pooled_levels))
    return result


def _influence_layer(samples: list[SensSample], cases: list[SensCase]) -> dict[str, Any]:
    influence_cases = [case for case in cases if case.suite == "influence"]
    if not influence_cases:
        return {}
    rows = []
    for case in influence_cases:
        subset = [s for s in samples if s.case_id == case.id]
        rows.append(
            {
                "case_id": case.id,
                "expected_routes": list(case.expected_routes),
                "mean": _mean_distribution(subset),
            }
        )
    return {
        "note": "同一共同任务前提下的三种强度/层次事件，检查有无“越强越 rethink”的错误泛化。",
        "rows": rows,
    }


def _profile_minus_baseline(
    samples: list[SensSample], cases: list[SensCase], personas: list[PersonaExcerpt]
) -> dict[str, Any]:
    baseline = next((p for p in personas if p.label == "baseline"), None)
    if baseline is None:
        return {}
    result: dict[str, Any] = {}
    for case in cases:
        base_subset = [
            s for s in samples if s.case_id == case.id and s.profile_label == baseline.label
        ]
        base_mean = _mean_distribution(base_subset)
        if base_mean is None:
            continue
        base_range = _route_range(base_subset)
        rows: dict[str, Any] = {}
        for persona in personas:
            if persona.label == baseline.label:
                continue
            subset = [
                s for s in samples if s.case_id == case.id and s.profile_label == persona.label
            ]
            mean = _mean_distribution(subset)
            if mean is None:
                continue
            rows[persona.label] = {
                "diff": {route: mean[route] - base_mean[route] for route in routing.ROUTES},
                "baseline_mean": base_mean,
                "profile_mean": mean,
                "baseline_range": base_range,
                "profile_range": _route_range(subset),
                "baseline_repeats": sum(1 for s in base_subset if _valid(s)),
                "profile_repeats": sum(1 for s in subset if _valid(s)),
            }
        if rows:
            result[case.id] = rows
    return result


def _repeat_fluctuation(samples: list[SensSample]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[SensSample]] = {}
    for sample in samples:
        groups.setdefault((sample.case_id, sample.profile_label), []).append(sample)
    rows: dict[str, Any] = {}
    for (case_id, profile), items in sorted(groups.items()):
        usable = [s for s in items if _valid(s)]
        if len(usable) < 2:
            continue
        rows[f"{case_id}/{profile}"] = {
            "repeats": len(usable),
            "argmaxes": [s.argmax for s in usable],
            "route_range": _route_range(usable),
            "range_definition": "每路径 max-min（全重复范围），不是相对第一次。",
        }
    return rows


def _fenced(text: str, language: str = "text") -> str:
    """保留原文；输入中的 Markdown 围栏不能提前结束原文块。"""
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _render_evidence(
    cases: list[SensCase], personas: list[PersonaExcerpt], samples: list[SensSample]
) -> list[str]:
    lines = [
        "## 场景、实际提示词与逐次输出",
        "",
        "这是合成的单次事件与角色状态探针，结合真实角色的人格摘录；"
        "不是从角色故事中截取的场景，也没有执行完整模拟。",
        "当前模拟时刻未单独指定；计划中的起止毫秒是输入原文。未给出的空间、身份或通信条件不补写。",
        "",
        "每个场景依次列出：完整状态 → 各人格两次结果对照 → 实际提示词 → 原始响应与解释。"
        "提示词直接取自每次保存的 `prompt`，不是按当前模板重建；"
        "它是传给 OpenCode CLI 的文本，未记录的 CLI／供应商附加系统提示不能据此还原。",
        "参考路由及理由只用于离线分析，不发给模型。r0/r1 对应第1/2次重复，"
        "调用序号是 samples.jsonl 中的实际记录顺序，重复不保证相邻。",
        "",
        "| 场景 | 分组 | 完整输入与结果 |",
        "| --- | --- | --- |",
    ]
    for index, case in enumerate(cases, 1):
        title = case.title.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {index} | {case.suite} | [{title}](#scene-{index}) |")
    lines += ["", "[跳到汇总统计](#statistics)", "", "### 人格条件及来源", ""]
    for persona in personas:
        lines += [
            f"#### {persona.label}：{persona.real_name or '无额外人格基线'}",
            "",
            _fenced(
                json.dumps(
                    {
                        "source_path": persona.source_path,
                        "source_sha256": persona.sha256,
                        "text_length": persona.length,
                        "实际注入的人格摘录": persona.text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "json",
            ),
            "",
        ]
    for index, case in enumerate(cases, 1):
        lines += [
            f'<a id="scene-{index}"></a>',
            "",
            f"### 场景 {index}：{case.title}",
            "",
            f"编号：`{case.id}`；分组：`{case.suite}`。",
            "",
            "**完整场景状态（实际实验输入）**",
            "",
            _fenced(
                f"本次事件：{case.event}\n\n当前意向：{case.intention}\n\n"
                "已保存计划：\n"
                + ("\n".join(case.plan) or "（空）")
                + "\n\n已知行动状态：\n"
                + ("\n".join(case.known_actions) or "（空）")
                + "\n\n近期记忆：\n"
                + ("\n".join(case.memory) or "（空）")
            ),
            "",
            "**离线参考（未发给模型）**",
            "",
            _fenced(
                f"允许路由：{', '.join(case.expected_routes) or '不设固定标签'}\n"
                f"理由：{case.label_rationale}"
            ),
            "",
            "**各条件逐次对照**（c=continue，r=replan，k=rethink；最大倾向仅离线观察）",
            "",
            "| 人格 | 重复 | 调用序号 | c / r / k | 最大倾向 | 有效 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        matches = [(order, s) for order, s in enumerate(samples, 1) if s.case_id == case.id]
        for order, sample in matches:
            probs = sample.probabilities
            values = " / ".join(str(probs[route]) for route in routing.ROUTES) if probs else "n/a"
            lines.append(
                f"| {sample.profile_label} | r{sample.repeat} | {order} | {values} | "
                f"{sample.argmax or 'n/a'} | {'是' if _valid(sample) else '否'} |"
            )
        lines.append("")
        for persona in personas:
            group = sorted(
                [(order, s) for order, s in matches if s.profile_label == persona.label],
                key=lambda item: (item[1].repeat, item[0]),
            )
            lines += [f"#### {persona.label}：{persona.real_name or '无额外人格基线'}", ""]
            if not group:
                lines += ["缺少调用记录。", ""]
                continue
            shared_prompt = len({sample.prompt for _, sample in group}) == 1
            if shared_prompt:
                lines += [
                    "**实际完整提示词**（以下重复的提示词逐字一致，仅展示一次）",
                    "",
                    _fenced(group[0][1].prompt),
                    "",
                ]
            for order, sample in group:
                lines += [
                    f"**第 {sample.repeat + 1} 次（r{sample.repeat}），调用序号 {order}**",
                    "",
                    _fenced(
                        json.dumps(
                            {
                                "task_id": sample.task_id,
                                "model": sample.model,
                                "session_ids": sample.session_ids,
                                "elapsed_seconds": sample.elapsed_seconds,
                                "exit_code": sample.exit_code,
                                "timed_out": sample.timed_out,
                                "call_error": sample.call_error,
                                "parse_error": sample.parse_error,
                                "tool_polluted": sample.tool_polluted,
                                "有效": _valid(sample),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "json",
                    ),
                    "",
                ]
                if not shared_prompt:
                    lines += ["**本次实际完整提示词**", "", _fenced(sample.prompt), ""]
                lines += [
                    "**原始响应**（不改写、不隐藏无效响应）",
                    "",
                    _fenced(sample.text),
                    "",
                    "**模型解释 note**",
                    "",
                    _fenced(sample.note if sample.note is not None else "（未解析出 note）"),
                    "",
                ]
                if sample.reasoning:
                    lines += [
                        "<details>",
                        "<summary>查看本次保存的 reasoning 原文</summary>",
                        "",
                        _fenced(sample.reasoning),
                        "",
                        "</details>",
                        "",
                    ]
    return lines


def _render_markdown(
    summary: dict[str, Any],
    samples: list[SensSample],
    cases: list[SensCase],
    personas: list[PersonaExcerpt],
) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def dist(mean: dict[str, float] | None) -> str:
        if mean is None:
            return "n/a"
        return " / ".join(f"{route}={mean[route]:.3f}" for route in routing.ROUTES)

    def cost_line(data: dict[str, Any]) -> str:
        tokens = data.get("tokens_sum")
        if tokens is None:
            return "tokens=n/a；cost=n/a"
        rendered = " ".join(
            f"{key}={tokens[key]}"
            for key in ("input", "output", "reasoning", "cache_read", "cache_write")
        )
        return f"tokens({rendered})；cost={fmt(data.get('total_cost'))}"

    def latency(data: dict[str, Any]) -> str:
        return f"{fmt(data['median'])} / {fmt(data['p95'])}"

    lines = [
        "# System1 敏感度探针报告（第二轮）",
        "",
        f"- 模型：`{summary['model']}`；suite：{summary['suite_filter']}；每状态重复 {summary['repeats']} 次",
        f"- 样本：实际 {summary['actual_samples']} / 预期 {summary['expected_samples']}；"
        f"有效 {summary['valid_samples']}；覆盖（有效/预期）：{fmt(summary['coverage'])}；"
        f"完整组 {summary['complete_groups']}/{summary['expected_groups']}",
        f"- 调用失败：{summary['call_failures']}；超时：{summary['timed_out']}；"
        f"schema 失败：{summary['schema_failures']}；非零退出：{summary['nonzero_exit']}；"
        f"工具污染：{summary['tool_polluted']}；并列最大：{summary['ties']}",
        f"- 耗时中位数/p95（秒）：{latency(summary['latency_seconds'])}；"
        f"usage 覆盖：{fmt(summary['usage_coverage'])}；usage 来源：{summary['usage_sources']}",
        f"- 合计 {cost_line(summary['cost_tokens'])}",
        "",
        "> 独立 System1 探针：只输出三路倾向分布（事件影响驱动的路径倾向），不是世界事件概率；"
        "argmax 仅离线展示/评分，不抽样、不制定正式路由策略。自称概率未校准；小样本不宣称统计显著。",
        "",
    ]
    lines += _render_evidence(cases, personas, samples)
    lines += [
        '<a id="statistics"></a>',
        "",
        "## 汇总统计",
        "",
        "### Profile（角色来源）",
        "",
        "| 编号 | 真实角色 | 抽取文本长度 | 来源 sha256 |",
        "| --- | --- | --- | --- |",
    ]
    for profile in summary["profiles"]:
        lines.append(
            f"| {profile['label']} | {profile['real_name'] or '（无人格基线）'} | "
            f"{profile['text_length']} | {(profile['sha256'] or 'n/a')[:16]} |"
        )
    lines += [
        "",
        "## 按 suite（含耗时/tokens/cost/覆盖）",
        "",
        "| suite | 样本/预期 | 有效 | 耗时中位/p95(秒) | usage覆盖 | tokens/cost | 三路均值 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for suite_name, data in summary["by_suite"].items():
        lines.append(
            f"| {suite_name} | {data['samples']}/{data['expected']} | {data['valid']} | "
            f"{latency(data['latency_seconds'])} | {fmt(data['usage_coverage'])} | "
            f"{cost_line(data['cost_tokens'])} | {dist(data['mean'])} |"
        )
    lines += [
        "",
        "## 按 profile（含耗时/tokens/cost/覆盖）",
        "",
        "| profile | 样本/预期 | 有效 | 耗时中位/p95(秒) | usage覆盖 | tokens/cost | 三路均值 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, data in summary["by_profile"].items():
        name = data["real_name"] or "基线"
        lines.append(
            f"| {label}（{name}） | {data['samples']}/{data['expected']} | {data['valid']} | "
            f"{latency(data['latency_seconds'])} | {fmt(data['usage_coverage'])} | "
            f"{cost_line(data['cost_tokens'])} | {dist(data['mean'])} |"
        )
    lines += ["", "## 逐状态（含全部重复原始三元组）", ""]
    for case_id, data in summary["by_case"].items():
        lines.append(
            f"### {case_id}（{data['suite']}；期望 {data['expected_routes'] or '无固定'}）"
        )
        lines.append("")
        lines.append(f"- 全体均值：{dist(data['mean'])}")
        for label, profile_data in data["by_profile"].items():
            raw = "；".join(
                f"r{item['repeat']}="
                + (
                    "n/a"
                    if item["probabilities"] is None
                    else f"(c={item['probabilities']['continue']:.2f},"
                    f"r={item['probabilities']['replan']:.2f},"
                    f"k={item['probabilities']['rethink']:.2f})"
                )
                + ("" if item["valid"] else "[无效]")
                for item in profile_data["raw"]
            )
            lines.append(
                f"  - {label}: 有效 {profile_data['valid']}/{profile_data['expected']}；"
                f"均值 {dist(profile_data['mean'])}；原始 {raw}"
            )
        lines.append("")
    trend = summary["intensity_trend"]
    if trend:
        lines += ["## intensity sweep 趋势（逐 profile，6 级）", ""]
        labels = list(trend["by_profile"].keys())
        header = "| 级别 | " + " | ".join(labels) + " | 池化 |"
        separator = "| --- | " + " | ".join("---" for _ in labels) + " | --- |"
        lines += [header, separator]
        for index, level in enumerate(trend["levels"]):
            cells = [dist(trend["by_profile"][label]["levels"][index]["mean"]) for label in labels]
            lines.append(
                f"| {level['case_id']} | " + " | ".join(cells) + f" | {dist(level['mean'])} |"
            )
        lines.append("")
        lines.append(f"- 池化（混合全部 profile，仅作对照）：{_trend_summary(trend)}")
        for label in labels:
            stats = trend["by_profile"][label]["stats"]
            lines.append(
                f"- {label}（完整级 {stats['complete_levels']}/{stats['expected_levels']}）："
                f"{_trend_summary(stats)}"
            )
        lines += ["", f"> {trend['note']}", ""]
    layer = summary["influence_layer"]
    if layer:
        lines += [
            "## influence layer（同强度不同层）",
            "",
            "| 版本 | 期望 | 三路均值 |",
            "| --- | --- | --- |",
        ]
        for row in layer["rows"]:
            lines.append(f"| {row['case_id']} | {row['expected_routes']} | {dist(row['mean'])} |")
        lines += ["", f"> {layer['note']}", ""]
    if summary["profile_minus_baseline"]:
        lines += [
            "## profile − baseline（差值 + 双方全重复范围同屏）",
            "",
            "| 状态/profile | 差值(c/r/k) | baseline 范围 | profile 范围 | 重复数(b/p) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case_id, rows in summary["profile_minus_baseline"].items():
            for label, data in rows.items():
                diff = " / ".join(f"{route}={data['diff'][route]:+.3f}" for route in routing.ROUTES)
                base_range = " / ".join(
                    f"{base}={fmt(data['baseline_range'][base])}" for base in routing.ROUTES
                )
                prof_range = " / ".join(
                    f"{base}={fmt(data['profile_range'][base])}" for base in routing.ROUTES
                )
                lines.append(
                    f"| {case_id}/{label} | {diff} | {base_range} | {prof_range} | "
                    f"{data['baseline_repeats']}/{data['profile_repeats']} |"
                )
        lines += [
            "",
            "> 差值必须与基线侧、profile 侧各自的全重复 max-min 范围一起看；若差值落在范围量级内，"
            "只能描述现象，不能声称稳定的角色敏感性或统计显著。",
            "",
        ]
    if summary["repeat_fluctuation"]:
        lines += ["## 同状态重复波动（每路径 max-min 全范围）", ""]
        for key, data in summary["repeat_fluctuation"].items():
            route_range = " / ".join(
                f"{route}={fmt(data['route_range'][route])}" for route in routing.ROUTES
            )
            lines.append(
                f"- {key}: 有效重复={data['repeats']}；argmax={data['argmaxes']}；"
                f"每路径范围 {route_range}（{data['range_definition']}）"
            )
        lines.append("")
    lines += ["## 覆盖与缺失（分组记录）", ""]
    lines.append(
        f"- 预期样本 {summary['expected_samples']}；实际 {summary['actual_samples']}；"
        f"有效 {summary['valid_samples']}；完整组 {summary['complete_groups']}/"
        f"{summary['expected_groups']}"
    )
    if summary["missing_samples"]:
        lines.append(f"- 缺失样本：{summary['missing_samples']}")
    else:
        lines.append("- 缺失样本：无")
    if summary["invalid_samples"]:
        lines.append(
            f"- 无效样本（失败/超时/非零退出/工具污染/schema 非法）：{summary['invalid_samples']}"
        )
    else:
        lines.append("- 无效样本：无")
    lines += ["", "## 逐条错误", ""]
    if summary["errors"]:
        for error in summary["errors"]:
            lines.append(f"- {error}")
    else:
        lines.append("- 无")
    lines += [
        "",
        "## 限制",
        "",
        "- 三路数值是模型**自报倾向**，未经校准，不是世界事件概率或真实决策概率。",
        "- 真实姓名替换为中性编号，但角色设定内容可能仍被模型凭已有知识识别，属**非完全盲测**。",
        "- 各 profile 抽取文本长度不同，模型对不同角色的既有知识也不同，二者与人格设定混杂，不能单独归因。",
        "- 小样本不宣称统计显著；profile−baseline 差值须与两侧全重复范围同屏比较，只描述现象。",
        "- cost 是 OpenCode usage 返回的元数据，不是账单。",
        "- JSON 会整体读入校验，但只抽取白名单字段并注入 prompt；不抽取或注入任何非白名单字段"
        "（外貌/服装/三围/爱情/亲密台词/关系底账等）。",
        "- seed 只控制实验顺序；无法设置模型真实温度/seed。",
    ]
    return "\n".join(lines) + "\n"


def _trend_summary(stats: dict[str, Any]) -> str:
    parts = []
    for route in routing.ROUTES:
        spearman = stats.get(f"{route}_spearman")
        if spearman is None:
            continue
        deltas = stats.get(f"{route}_adjacent_delta", [])
        parts.append(
            f"{route}: spearman={spearman:.3f}；"
            f"邻级变化={[round(value, 3) for value in deltas]}；"
            f"符号变化={stats.get(f'{route}_sign_changes')}"
        )
    return "；".join(parts) if parts else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("intensity", "influence", "unconfirmed", "all"), default="all"
    )
    parser.add_argument(
        "--profile", action="append", default=None, help="profile id 或真实角色名，可重复"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=routing.DEFAULT_MODEL)
    parser.add_argument("--agent", default=routing.DEFAULT_AGENT)
    parser.add_argument("--opencode", default=routing.DEFAULT_OPENCODE)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--persona-dir", default=str(DEFAULT_PERSONA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--usage-db", default=routing.DEFAULT_USAGE_DB)
    parser.add_argument("--no-db-usage", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-from", type=Path, help="从已保存目录离线重导出，不调用模型")
    args = parser.parse_args(argv)

    if args.report_from is not None:
        summary = regenerate_report(args.report_from)
        print(f"离线报告：{args.report_from / 'report.md'}；样本 {summary['actual_samples']}")
        return 0

    if args.repeats < 1:
        raise SystemExit("--repeats 必须是正整数")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须是正整数")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout 必须是有限正数")
    cases = load_cases()
    selected_cases = _select_cases(cases, args.suite, args.limit, args.smoke)
    specs = _select_profiles(args.profile, args.smoke)
    try:
        personas = [load_persona(spec, Path(args.persona_dir)) for spec in specs]
    except PersonaSourceError as error:
        print(f"人格来源错误：{error}", file=sys.stderr)
        return 2
    repeats = 1 if args.smoke else args.repeats
    tasks = build_tasks(selected_cases, specs, repeats, args.seed)

    if args.dry_run:
        print(
            f"状态数：{len(selected_cases)}；profile：{[p.label for p in personas]}；"
            f"重复：{repeats}；任务数：{len(tasks)}"
        )
        print(f"模型：{args.model}；agent：{args.agent}")
        for case, spec, repeat in tasks[:2]:
            persona = next(p for p in personas if p.label == spec["id"])
            print("=" * 60)
            print(f"task {case.id}/{persona.label}/r{repeat}")
            print(build_prompt(case, persona))
        return 0

    workdir = Path(args.workdir)
    routing.ensure_agent_dir(workdir)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path("sceneweave-runs") / "system1-sensitivity" / timestamp)
    )
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆写混入旧样本：{outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "probe": "independent-system1-sensitivity",
                "model": args.model,
                "suite_filter": args.suite,
                "repeats": repeats,
                "seed": args.seed,
                "seed_scope": "仅用于实验顺序；无法设置模型真实温度/seed",
                "timeout_seconds": args.timeout,
                "no_thinking": args.no_thinking,
                "no_db_usage": args.no_db_usage,
                "usage_db": args.usage_db,
                "source_hashes": {
                    "cases_file": str(SENS_CASES_PATH),
                    "cases_file_sha256": hashlib.sha256(SENS_CASES_PATH.read_bytes()).hexdigest(),
                    "script": str(Path(__file__).resolve()),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                "personas": [persona.to_manifest() for persona in personas],
                "cases": [
                    {
                        "id": case.id,
                        "suite": case.suite,
                        "title": case.title,
                        "event": case.event,
                        "intention": case.intention,
                        "plan": list(case.plan),
                        "known_actions": list(case.known_actions),
                        "memory": list(case.memory),
                        "expected_routes": list(case.expected_routes),
                        "label_rationale": case.label_rationale,
                    }
                    for case in selected_cases
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    samples_path = outdir / "samples.jsonl"
    print(f"输出目录：{outdir}")
    print(f"任务数：{len(tasks)}（模型 {args.model}）")

    samples: list[SensSample] = []
    for index, (case, spec, repeat) in enumerate(tasks, start=1):
        persona = next(p for p in personas if p.label == spec["id"])
        task_id = f"{case.id}/{persona.label}/r{repeat}"
        prompt = build_prompt(case, persona)
        print(f"[{index}/{len(tasks)}] {task_id} ...", flush=True)
        exit_code, elapsed, timed_out, stdout, stderr, call_error = routing.run_one(
            opencode_bin=args.opencode,
            workdir=workdir,
            agent=args.agent,
            model=args.model,
            prompt=prompt,
            timeout=args.timeout,
            thinking=not args.no_thinking,
        )
        parsed = routing.parse_events(stdout)
        sample = SensSample(
            task_id=task_id,
            case_id=case.id,
            suite=case.suite,
            profile_label=persona.label,
            profile_real_name=persona.real_name,
            profile_len=persona.length,
            profile_sha256=persona.sha256,
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
            db_usage = routing.fetch_db_usage(args.usage_db, parsed["session_ids"][-1])
            if db_usage is not None:
                sample.usage = db_usage
                sample.usage_source = "session-db"
        if sample.call_error is None:
            try:
                result = routing.parse_distribution(sample.text)
                sample.parse_ok = True
                sample.tie = bool(result["tie"])
                sample.probabilities = result.get("probabilities")
                sample.argmax = result["route"]
                sample.note = result["note"]
                sample.in_expected = (
                    sample.argmax in case.expected_routes if case.expected_routes else None
                )
            except ValueError as error:
                sample.parse_error = str(error)
        samples.append(sample)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
        status = "ok" if sample.parse_ok else ("call-error" if sample.call_error else "parse-error")
        print(
            f"    {status}: argmax={sample.argmax} tie={sample.tie} {sample.elapsed_seconds:.1f}s"
        )

    summary = write_report(
        outdir=outdir,
        samples=samples,
        personas=personas,
        cases=selected_cases,
        model=args.model,
        suite=args.suite,
        repeats=repeats,
        seed=args.seed,
    )
    print(f"报告：{outdir / 'report.md'}")
    print(
        f"完成：{summary['total_samples']} 次调用，有效 {summary['valid_samples']}，"
        f"调用失败 {summary['call_failures']}，schema 失败 {summary['schema_failures']}，"
        f"工具污染 {summary['tool_polluted']}"
    )
    if (
        summary["call_failures"]
        or summary["tool_polluted"]
        or summary["nonzero_exit"]
        or summary["schema_valid"] != summary["total_samples"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
