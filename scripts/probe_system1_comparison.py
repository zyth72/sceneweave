#!/usr/bin/env python3
"""独立 System1 对照探针：同一敏感度输入下 route 字段 vs 三路倾向分布。

本脚本是独立 System1 实验，绝不启动 System2/System3/翻译器/World/engine 完整循环，
不生成动作、不影响正式模拟路由。它在旧敏感度批次（默认
``sceneweave-runs/system1-sensitivity/20261010-corrected``）保存的真实人格摘录、
12 个合成状态与原始完整 prompt 之上，构造**成对**的两种输出任务：

- ``field``：在共同输入前缀后输出 ``{"route": "continue|replan|rethink", "note": ...}``。
- ``distribution``：保留旧完整提示词原文，输出三路倾向分布，由程序离线取唯一最大值。

每个 (case, profile, repeat) 是一个 pair：两条调用连续执行、共享逐字一致的输入前缀，
只替换最后的【输出】段。旧 96 次记录只作为输入来源，不混入本轮样本，也不假装是新鲜配对。

来源读取规则：人格摘录、用例、原始 prompt 全部来自旧目录保存的 ``manifest.json`` 与
``samples.jsonl``；不读取当前 ``~/lin``，也不按当前模板重建 prompt。原始记录只读。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_system1_routing as routing  # noqa: E402
import probe_system1_sensitivity as sens  # noqa: E402

ROUTES = routing.ROUTES
MODES = ("field", "distribution")
OUTPUT_MARKER = "【输出】"
FIELD_INSTRUCTION = (
    "【输出】只输出一个 JSON 对象，直接给出路由字段，格式：\n"
    '{"route": "continue|replan|rethink", "note": "简短说明"}'
)
DEFAULT_SOURCE_DIR = Path(
    "/home/zyth/sceneweave/sceneweave-runs/system1-sensitivity/20261010-corrected"
)
DEFAULT_WORKDIR = "/tmp/opencode/system1-comparison"
DEFAULT_MODEL = routing.DEFAULT_MODEL
SOURCE_PROBE = "independent-system1-sensitivity"
COMPARISON_PROBE = "independent-system1-comparison"
SMOKE_PROFILES = ("baseline", "P1")

# 复用第一/二轮探针的统计口径，保持三次实验可比。
_median = routing._median
_p95 = routing._p95
_rate = routing._rate
_cost_summary = routing._cost_summary
_usage_sources = routing._usage_sources
_fenced = sens._fenced


class SourceError(RuntimeError):
    """来源缺失、覆盖不全、身份重复或 prompt 歧义；不得静默取一条替代。"""


@dataclass(frozen=True, slots=True)
class SourcePersona:
    label: str
    real_name: str | None
    source_path: str | None
    sha256: str | None
    text: str
    field_paths: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "real_name": self.real_name,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "text_length": len(self.text),
            "field_paths": list(self.field_paths),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class SourceCase:
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

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "title": self.title,
            "event": self.event,
            "intention": self.intention,
            "plan": list(self.plan),
            "known_actions": list(self.known_actions),
            "memory": list(self.memory),
            "expected_routes": list(self.expected_routes),
            "label_rationale": self.label_rationale,
        }


@dataclass(frozen=True, slots=True)
class PromptPair:
    case_id: str
    profile_label: str
    prefix: str
    distribution_instruction: str
    field_prompt: str
    distribution_prompt: str
    prefix_sha256: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "profile_label": self.profile_label,
            "prefix_sha256": self.prefix_sha256,
            "field_prompt": self.field_prompt,
            "distribution_prompt": self.distribution_prompt,
        }


@dataclass(frozen=True, slots=True)
class SourceBundle:
    source_dir: str
    source_meta: dict[str, Any]
    personas: tuple[SourcePersona, ...]
    cases: tuple[SourceCase, ...]
    prompts: dict[tuple[str, str], PromptPair]


def split_prompt(prompt: str) -> tuple[str, str]:
    """按最后一个独立【输出】段分割为共同前缀与输出指令。"""
    index = prompt.rfind(OUTPUT_MARKER)
    if index < 0:
        raise SourceError("保存的 prompt 缺少【输出】标记，拒绝重建")
    prefix = prompt[:index]
    if prefix and not prefix.endswith("\n"):
        raise SourceError("【输出】标记不在独立行首，拒绝分割")
    return prefix, prompt[index:]


def make_prompt_pair(case_id: str, profile_label: str, distribution_prompt: str) -> PromptPair:
    prefix, instruction = split_prompt(distribution_prompt)
    field_prompt = prefix + FIELD_INSTRUCTION
    return PromptPair(
        case_id=case_id,
        profile_label=profile_label,
        prefix=prefix,
        distribution_instruction=instruction,
        field_prompt=field_prompt,
        distribution_prompt=distribution_prompt,
        prefix_sha256=hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source_personas(manifest: dict[str, Any], path: Path) -> tuple[SourcePersona, ...]:
    raw = manifest.get("personas")
    if not isinstance(raw, list) or not raw:
        raise SourceError(f"manifest 缺少 personas：{path}")
    personas: list[SourcePersona] = []
    labels: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise SourceError(f"manifest personas 项必须是对象：{path}")
        missing = [key for key in ("label", "text") if key not in item]
        if missing:
            raise SourceError(f"manifest personas 缺字段 {missing}：{path}")
        label = str(item["label"])
        if label in labels:
            raise SourceError(f"manifest personas 身份重复：{label}")
        labels.add(label)
        field_paths = item.get("field_paths", [])
        personas.append(
            SourcePersona(
                label=label,
                real_name=None if item.get("real_name") is None else str(item["real_name"]),
                source_path=None if item.get("source_path") is None else str(item["source_path"]),
                sha256=None if item.get("sha256") is None else str(item["sha256"]),
                text=str(item["text"]),
                field_paths=tuple(str(entry) for entry in field_paths),
            )
        )
    return tuple(personas)


def _read_source_cases(manifest: dict[str, Any], path: Path) -> tuple[SourceCase, ...]:
    raw = manifest.get("cases")
    if not isinstance(raw, list) or not raw:
        raise SourceError(f"manifest 缺少 cases：{path}")
    cases: list[SourceCase] = []
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise SourceError(f"manifest cases 项必须是对象：{path}")
        missing = [key for key in ("id", "suite", "event", "intention") if key not in item]
        if missing:
            raise SourceError(f"manifest cases 缺字段 {missing}：{path}")
        case_id = str(item["id"])
        if case_id in ids:
            raise SourceError(f"manifest cases 身份重复：{case_id}")
        ids.add(case_id)
        expected = tuple(str(route) for route in item.get("expected_routes", ()))
        for route in expected:
            if route not in ROUTES:
                raise SourceError(f"用例 {case_id} 含未知期望路由：{route}")
        cases.append(
            SourceCase(
                id=case_id,
                suite=str(item["suite"]),
                title=str(item.get("title", case_id)),
                event=str(item["event"]),
                intention=str(item["intention"]),
                plan=tuple(str(entry) for entry in item.get("plan", ())),
                known_actions=tuple(str(entry) for entry in item.get("known_actions", ())),
                memory=tuple(str(entry) for entry in item.get("memory", ())),
                expected_routes=expected,
                label_rationale=str(item.get("label_rationale", "")),
            )
        )
    return tuple(cases)


def load_source(source_dir: Path) -> SourceBundle:
    """只读旧敏感度批次，校验覆盖、身份与 prompt 一致性后派生两种 prompt。"""
    manifest_path = source_dir / "manifest.json"
    samples_path = source_dir / "samples.jsonl"
    if not manifest_path.exists():
        raise SourceError(f"缺少来源 manifest：{manifest_path}")
    if not samples_path.exists():
        raise SourceError(f"缺少来源 samples：{samples_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SourceError(f"来源 manifest 不是合法 JSON：{manifest_path}") from error
    if not isinstance(manifest, dict):
        raise SourceError(f"来源 manifest 根必须是对象：{manifest_path}")
    if manifest.get("probe") != SOURCE_PROBE:
        raise SourceError(f"来源 probe 不是 {SOURCE_PROBE}：{manifest.get('probe')!r}")
    if "repeats" not in manifest or "model" not in manifest:
        raise SourceError(f"来源 manifest 缺少 repeats/model：{manifest_path}")

    personas = _read_source_personas(manifest, manifest_path)
    cases = _read_source_cases(manifest, manifest_path)
    source_repeats = int(manifest["repeats"])
    if source_repeats < 1:
        raise SourceError(f"来源 repeats 必须为正：{source_repeats}")

    expected_keys = {
        (case.id, persona.label, repeat)
        for case in cases
        for persona in personas
        for repeat in range(source_repeats)
    }
    by_pair: dict[tuple[str, str], dict[int, str]] = {}
    seen: set[tuple[str, str, int]] = set()
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise SourceError(f"样本行必须是对象：{samples_path}")
        missing = [
            key for key in ("case_id", "profile_label", "repeat", "prompt") if key not in record
        ]
        if missing:
            raise SourceError(f"样本缺字段 {missing}：{samples_path}")
        key = (str(record["case_id"]), str(record["profile_label"]), int(record["repeat"]))
        if key in seen:
            raise SourceError(f"来源身份重复（同一 case/profile/repeat 多次出现）：{key}")
        seen.add(key)
        by_pair.setdefault((key[0], key[1]), {})[key[2]] = str(record["prompt"])

    if seen != expected_keys:
        missing_keys = sorted(f"{c}/{p}/r{r}" for c, p, r in expected_keys - seen)
        extra_keys = sorted(f"{c}/{p}/r{r}" for c, p, r in seen - expected_keys)
        raise SourceError(f"来源覆盖不符，缺失={missing_keys}，多余={extra_keys}")

    prompts: dict[tuple[str, str], PromptPair] = {}
    for pair_key, repeats_map in by_pair.items():
        if set(repeats_map) != set(range(source_repeats)):
            raise SourceError(f"来源重复不完整：{pair_key}")
        distinct = set(repeats_map.values())
        if len(distinct) != 1:
            raise SourceError(f"同 case/profile 重复 prompt 不一致，拒绝暗中取第一条：{pair_key}")
        prompts[pair_key] = make_prompt_pair(pair_key[0], pair_key[1], next(iter(distinct)))

    source_meta: dict[str, Any] = {
        "source_dir": str(source_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "samples_path": str(samples_path),
        "samples_sha256": _sha256_file(samples_path),
        "source_probe": manifest.get("probe"),
        "source_model": manifest.get("model"),
        "source_repeats": source_repeats,
        "source_seed": manifest.get("seed"),
        "source_suite_filter": manifest.get("suite_filter"),
        "source_hashes": manifest.get("source_hashes"),
        "case_count": len(cases),
        "persona_labels": [persona.label for persona in personas],
    }
    return SourceBundle(
        source_dir=str(source_dir),
        source_meta=source_meta,
        personas=personas,
        cases=cases,
        prompts=prompts,
    )


@dataclass(frozen=True, slots=True)
class PlannedPair:
    pair_id: str
    case: SourceCase
    persona: SourcePersona
    repeat: int
    mode_order: tuple[str, str]
    prompt: PromptPair


@dataclass(frozen=True, slots=True)
class PlannedCall:
    pair_id: str
    case: SourceCase
    persona: SourcePersona
    repeat: int
    mode: str
    prompt: str
    prefix_sha256: str


def build_pairs(
    cases: list[SourceCase],
    personas: list[SourcePersona],
    prompts: dict[tuple[str, str], PromptPair],
    repeats: int,
    seed: int,
) -> list[PlannedPair]:
    """固定 seed 打乱 pair 顺序，每 pair 随机决定 field/distribution 先后。"""
    combos = [
        (case, persona, repeat)
        for repeat in range(repeats)
        for case in cases
        for persona in personas
    ]
    rng = random.Random(seed)
    rng.shuffle(combos)
    pairs: list[PlannedPair] = []
    for case, persona, repeat in combos:
        key = (case.id, persona.label)
        if key not in prompts:
            raise SourceError(f"缺少 prompt：{key}")
        order: tuple[str, str] = (
            ("field", "distribution") if rng.random() < 0.5 else ("distribution", "field")
        )
        pairs.append(
            PlannedPair(
                pair_id=f"{case.id}/{persona.label}/r{repeat}",
                case=case,
                persona=persona,
                repeat=repeat,
                mode_order=order,
                prompt=prompts[key],
            )
        )
    return pairs


def build_calls(pairs: list[PlannedPair]) -> list[PlannedCall]:
    calls: list[PlannedCall] = []
    for pair in pairs:
        for mode in pair.mode_order:
            prompt = (
                pair.prompt.field_prompt if mode == "field" else pair.prompt.distribution_prompt
            )
            calls.append(
                PlannedCall(
                    pair_id=pair.pair_id,
                    case=pair.case,
                    persona=pair.persona,
                    repeat=pair.repeat,
                    mode=mode,
                    prompt=prompt,
                    prefix_sha256=pair.prompt.prefix_sha256,
                )
            )
    return calls


def _select_smoke(
    cases: list[SourceCase], personas: list[SourcePersona]
) -> tuple[list[SourceCase], list[SourcePersona]]:
    first_per_suite: dict[str, SourceCase] = {}
    for case in cases:
        first_per_suite.setdefault(case.suite, case)
    selected_cases = list(first_per_suite.values())
    selected_personas = [persona for persona in personas if persona.label in SMOKE_PROFILES]
    if not selected_personas:
        raise SourceError(f"smoke 需要 {SMOKE_PROFILES} 中的 profile")
    return selected_cases, selected_personas


@dataclass
class ComparisonSample:
    call_index: int
    pair_id: str
    case_id: str
    case_suite: str
    profile_label: str
    profile_real_name: str | None
    repeat: int
    mode: str
    model: str
    prompt: str
    prefix_sha256: str
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    call_error: str | None = None
    parse_ok: bool = False
    parse_error: str | None = None
    route: str | None = None
    probabilities: dict[str, float] | None = None
    argmax: str | None = None
    tie: bool = False
    margin: float | None = None
    entropy: float | None = None
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
    in_expected: bool | None = None

    def to_record(self) -> dict[str, Any]:
        usage = self.usage or {}
        return {
            "call_index": self.call_index,
            "pair_id": self.pair_id,
            "case_id": self.case_id,
            "case_suite": self.case_suite,
            "profile_label": self.profile_label,
            "profile_real_name": self.profile_real_name,
            "repeat": self.repeat,
            "mode": self.mode,
            "model": self.model,
            "prompt": self.prompt,
            "prefix_sha256": self.prefix_sha256,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "timed_out": self.timed_out,
            "call_error": self.call_error,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "route": self.route,
            "probabilities": self.probabilities,
            "argmax": self.argmax,
            "tie": self.tie,
            "margin": self.margin,
            "entropy": self.entropy,
            "note": self.note,
            "text": self.text,
            "reasoning": self.reasoning,
            "session_ids": self.session_ids,
            "step_ids": self.step_ids,
            "message_ids": self.message_ids,
            "usage_available": bool(usage),
            "usage_source": self.usage_source,
            "cost": usage.get("cost"),
            "tokens": usage.get("tokens"),
            "tool_polluted": self.tool_polluted,
            "tool_events": self.tool_events,
            "expected_routes": list(self.expected_routes),
            "in_expected": self.in_expected,
            "raw_events": self.raw_events,
        }


def _call_ok(sample: ComparisonSample) -> bool:
    return (
        sample.call_error is None
        and not sample.timed_out
        and sample.exit_code == 0
        and not sample.tool_polluted
    )


def _valid(sample: ComparisonSample) -> bool:
    return sample.parse_ok and _call_ok(sample)


def _scorable(sample: ComparisonSample) -> bool:
    """可唯一评分：字段模式有合法 route，分布模式有合法且无并列的 argmax。"""
    if not _valid(sample):
        return False
    if sample.mode == "field":
        return sample.route in ROUTES
    return sample.argmax in ROUTES and not sample.tie


def _parse(text: str, mode: str) -> dict[str, Any]:
    if mode == "field":
        result = routing.parse_field(text)
        return {
            "route": result["route"],
            "tie": False,
            "margin": None,
            "entropy": None,
            "probabilities": None,
            "argmax": None,
            "note": result["note"],
        }
    result = routing.parse_distribution(text)
    return {
        "route": None,
        "tie": bool(result["tie"]),
        "margin": result["margin"],
        "entropy": result["entropy"],
        "probabilities": result["probabilities"],
        "argmax": result["route"],
        "note": result["note"],
    }


def _expected_pair_ids(
    cases: list[SourceCase], personas: list[SourcePersona], repeats: int
) -> list[str]:
    return [
        f"{case.id}/{persona.label}/r{repeat}"
        for repeat in range(repeats)
        for case in cases
        for persona in personas
    ]


def _group_pairs(samples: list[ComparisonSample]) -> dict[str, dict[str, ComparisonSample]]:
    grouped: dict[str, dict[str, ComparisonSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.pair_id, {})[sample.mode] = sample
    return grouped


def _paired_summary(
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
) -> dict[str, Any]:
    grouped = _group_pairs(samples)
    expected_ids = set(_expected_pair_ids(cases, personas, repeats))
    missing = sorted(expected_ids - set(grouped))
    complete = comparable = consistent = inconsistent = 0
    failed: list[str] = []
    field_first = distribution_first = 0
    inconsistent_list: list[dict[str, Any]] = []
    for pair_id in sorted(expected_ids & set(grouped)):
        modes = grouped[pair_id]
        field_sample = modes.get("field")
        dist_sample = modes.get("distribution")
        if field_sample is None or dist_sample is None:
            failed.append(pair_id)
            continue
        if field_sample.call_index < dist_sample.call_index:
            field_first += 1
        else:
            distribution_first += 1
        if not _valid(field_sample) or not _valid(dist_sample):
            failed.append(pair_id)
            continue
        complete += 1
        if not _scorable(field_sample) or not _scorable(dist_sample):
            continue
        comparable += 1
        if field_sample.route == dist_sample.argmax:
            consistent += 1
        else:
            inconsistent += 1
            inconsistent_list.append(
                {
                    "pair_id": pair_id,
                    "field": field_sample.route,
                    "distribution": {
                        "probabilities": dist_sample.probabilities,
                        "argmax": dist_sample.argmax,
                        "tie": dist_sample.tie,
                    },
                }
            )
    return {
        "expected_pairs": len(expected_ids),
        "actual_pairs": len(expected_ids & set(grouped)),
        "complete_pairs": complete,
        "comparable_pairs": comparable,
        "consistent_pairs": consistent,
        "inconsistent_pairs": inconsistent,
        "missing_pairs": len(missing),
        "failed_pairs": len(failed),
        "missing_list": missing,
        "failed_list": failed,
        "coverage_complete_over_expected": _rate(complete, len(expected_ids)),
        "coverage_comparable_over_expected": _rate(comparable, len(expected_ids)),
        "consistent_over_comparable": _rate(consistent, comparable),
        "field_first_pairs": field_first,
        "distribution_first_pairs": distribution_first,
        "order_definition": "按保存的 call_index 判断每 pair 内 field 与 distribution 谁先执行。",
        "inconsistent_list": inconsistent_list,
    }


def _stability(
    samples: list[ComparisonSample],
    mode: str,
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[ComparisonSample]] = {}
    for sample in samples:
        if sample.mode == mode:
            groups.setdefault((sample.case_id, sample.profile_label), []).append(sample)
    compared = stable = 0
    unstable: list[str] = []
    incomplete: list[str] = []
    for case in cases:
        for persona in personas:
            key = f"{case.id}/{persona.label}"
            items = groups.get((case.id, persona.label), [])
            if (
                repeats < 2
                or len(items) != repeats
                or not all(_valid(sample) and _scorable(sample) for sample in items)
            ):
                incomplete.append(key)
                continue
            compared += 1
            picked = {sample.route if mode == "field" else sample.argmax for sample in items}
            if len(picked) == 1:
                stable += 1
            else:
                unstable.append(f"{key}: {sorted(str(value) for value in picked)}")
    groups_total = len(cases) * len(personas)
    return {
        "groups": groups_total,
        "compared_groups": compared,
        "stable_groups": stable,
        "coverage": _rate(compared, groups_total),
        "stability_rate": _rate(stable, compared),
        "unstable": unstable,
        "incomplete": incomplete,
    }


def _mode_latency_cost(samples: list[ComparisonSample], mode: str) -> dict[str, Any]:
    records = [sample.to_record() for sample in samples if sample.mode == mode]
    cost = _cost_summary(records)
    elapsed = [record["elapsed_seconds"] for record in records]
    return {
        "latency_seconds": {"median": _median(elapsed), "p95": _p95(elapsed)},
        "cost_tokens": cost,
        "usage_coverage": _rate(cost["samples_with_usage"], len(records)),
        "usage_available": sum(1 for record in records if record["usage_available"]),
    }


def _mode_summary(
    samples: list[ComparisonSample],
    mode: str,
    expected_total: int,
) -> dict[str, Any]:
    records = [sample.to_record() for sample in samples if sample.mode == mode]
    subset = [sample for sample in samples if sample.mode == mode]
    schema_valid = sum(1 for sample in subset if sample.parse_ok)
    valid = sum(1 for sample in subset if _valid(sample))
    scorable = sum(1 for sample in subset if _scorable(sample))
    return {
        "mode": mode,
        "expected": expected_total,
        "actual": len(records),
        "call_failures": sum(1 for record in records if record["call_error"]),
        "timed_out": sum(1 for record in records if record["timed_out"]),
        "nonzero_exit": sum(1 for record in records if record["exit_code"] not in (0, None)),
        "tool_polluted": sum(1 for record in records if record["tool_polluted"]),
        "schema_valid": schema_valid,
        "schema_valid_rate_over_expected": _rate(schema_valid, expected_total),
        "schema_valid_rate_over_actual": _rate(schema_valid, len(records)),
        "valid": valid,
        "valid_rate_over_expected": _rate(valid, expected_total),
        "ties": sum(1 for sample in subset if sample.tie),
        "scorable": scorable,
        "scorable_rate_over_expected": _rate(scorable, expected_total),
        "scorable_rate_over_valid": _rate(scorable, valid),
        "usage_sources": _usage_sources(records),
        **_mode_latency_cost(samples, mode),
    }


def _latency_diff(
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
) -> dict[str, Any]:
    grouped = _group_pairs(samples)
    expected_ids = set(_expected_pair_ids(cases, personas, repeats))
    diffs: list[float] = []
    for pair_id in sorted(expected_ids & set(grouped)):
        modes = grouped[pair_id]
        field_sample = modes.get("field")
        dist_sample = modes.get("distribution")
        if field_sample is None or dist_sample is None:
            continue
        diffs.append(field_sample.elapsed_seconds - dist_sample.elapsed_seconds)
    return {
        "pairs": len(diffs),
        "median_field_minus_distribution": _median(diffs),
        "definition": "每 pair（field.elapsed − distribution.elapsed）的中位数；不含缺任一模式的 pair。",
    }


def _reference_match(
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
    mode: str,
) -> dict[str, Any]:
    labeled = [case for case in cases if case.expected_routes]
    expected_total = len(labeled) * len(personas) * repeats
    matches = 0
    valid_total = 0
    for sample in samples:
        if sample.mode != mode:
            continue
        case = next((item for item in labeled if item.id == sample.case_id), None)
        if case is None:
            continue
        if _valid(sample) and _scorable(sample):
            valid_total += 1
            value = sample.route if mode == "field" else sample.argmax
            if value in case.expected_routes:
                matches += 1
    return {
        "labeled_cases": [case.id for case in labeled],
        "expected_denominator": expected_total,
        "valid_denominator": valid_total,
        "matches": matches,
        "rate_over_expected": _rate(matches, expected_total),
        "rate_over_valid": _rate(matches, valid_total),
        "note": "失败/未评分样本计入预期分母，不进入有效分母；无固定参考的状态不参与。",
    }


def _by_case(
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
) -> dict[str, Any]:
    grouped = _group_pairs(samples)
    result: dict[str, Any] = {}
    for case in cases:
        by_profile: dict[str, Any] = {}
        for persona in personas:
            rows: list[dict[str, Any]] = []
            for repeat in range(repeats):
                pair_id = f"{case.id}/{persona.label}/r{repeat}"
                modes = grouped.get(pair_id, {})
                field_sample = modes.get("field")
                dist_sample = modes.get("distribution")
                both = (
                    field_sample is not None
                    and dist_sample is not None
                    and _scorable(field_sample)
                    and _scorable(dist_sample)
                )
                consistent: bool | None = None
                if both and field_sample is not None and dist_sample is not None:
                    consistent = field_sample.route == dist_sample.argmax
                rows.append(
                    {
                        "repeat": repeat,
                        "pair_id": pair_id,
                        "field": None
                        if field_sample is None
                        else {
                            "route": field_sample.route,
                            "note": field_sample.note,
                            "valid": _valid(field_sample),
                            "scorable": _scorable(field_sample),
                        },
                        "distribution": None
                        if dist_sample is None
                        else {
                            "probabilities": dist_sample.probabilities,
                            "argmax": dist_sample.argmax,
                            "tie": dist_sample.tie,
                            "note": dist_sample.note,
                            "valid": _valid(dist_sample),
                            "scorable": _scorable(dist_sample),
                        },
                        "consistent": consistent,
                    }
                )
            by_profile[persona.label] = rows
        result[case.id] = {
            "suite": case.suite,
            "title": case.title,
            "expected_routes": list(case.expected_routes),
            "by_profile": by_profile,
        }
    return result


def compute_summary(
    *,
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    model: str,
    repeats: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    records = [sample.to_record() for sample in samples]
    expected_per_mode = len(cases) * len(personas) * repeats
    call_failures = [record for record in records if record["call_error"]]
    timed_out = [record for record in records if record["timed_out"]]
    nonzero_exit = [record for record in records if record["exit_code"] not in (0, None)]
    schema_failures = [record for record in records if not record["parse_ok"]]
    polluted = [record for record in records if record["tool_polluted"]]
    elapsed = [record["elapsed_seconds"] for record in records]
    overall_cost = _cost_summary(records)
    summary: dict[str, Any] = {
        "probe": COMPARISON_PROBE,
        "semantics": (
            "同一输入前缀下 route 字段与三路倾向分布配对；分布是事件影响驱动的路径倾向，"
            "不是世界事件概率；argmax 仅离线展示/评分，不是正式路由策略，自称概率未校准。"
        ),
        "model": model,
        "modes": list(MODES),
        "repeats": repeats,
        "seed": seed,
        "seed_scope": "仅用于实验顺序与两模式先后；无法设置模型真实温度/seed",
        "smoke": smoke,
        "expected_calls": expected_per_mode * len(MODES),
        "actual_calls": len(records),
        "call_failures": len(call_failures),
        "timed_out": len(timed_out),
        "nonzero_exit": len(nonzero_exit),
        "schema_failures": len(schema_failures),
        "schema_valid": sum(1 for record in records if record["parse_ok"]),
        "tool_polluted": len(polluted),
        "usage_available": sum(1 for record in records if record["usage_available"]),
        "usage_sources": _usage_sources(records),
        "latency_seconds": {"median": _median(elapsed), "p95": _p95(elapsed)},
        "cost_tokens": overall_cost,
        "usage_coverage": _rate(overall_cost["samples_with_usage"], len(records)),
        "by_mode": {mode: _mode_summary(samples, mode, expected_per_mode) for mode in MODES},
        "paired": _paired_summary(samples, cases, personas, repeats),
        "stability": {mode: _stability(samples, mode, cases, personas, repeats) for mode in MODES},
        "latency_diff_field_minus_distribution": _latency_diff(samples, cases, personas, repeats),
        "reference_match": {
            mode: _reference_match(samples, cases, personas, repeats, mode) for mode in MODES
        },
        "by_case": _by_case(samples, cases, personas, repeats),
        "errors": [
            {
                "call_index": record["call_index"],
                "pair_id": record["pair_id"],
                "mode": record["mode"],
                "call_error": record["call_error"],
                "parse_error": record["parse_error"],
                "timed_out": record["timed_out"],
                "exit_code": record["exit_code"],
                "tool_polluted": record["tool_polluted"],
            }
            for record in records
            if (
                record["call_error"]
                or record["parse_error"]
                or record["timed_out"]
                or record["exit_code"] not in (0, None)
                or record["tool_polluted"]
            )
        ],
    }
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _metadata_block(sample: ComparisonSample) -> str:
    return _fenced(
        json.dumps(
            {
                "call_index": sample.call_index,
                "pair_id": sample.pair_id,
                "task": f"{sample.case_id}/{sample.profile_label}/r{sample.repeat}/{sample.mode}",
                "mode_order_call": "计划调用序号；实际完成行序见 samples.jsonl",
                "model": sample.model,
                "prefix_sha256": sample.prefix_sha256,
                "session_ids": sample.session_ids,
                "response_ids": sample.message_ids,
                "elapsed_seconds": sample.elapsed_seconds,
                "exit_code": sample.exit_code,
                "timed_out": sample.timed_out,
                "call_error": sample.call_error,
                "parse_ok": sample.parse_ok,
                "parse_error": sample.parse_error,
                "tool_polluted": sample.tool_polluted,
                "usage_source": sample.usage_source,
                "cost": (sample.usage or {}).get("cost"),
                "tokens": (sample.usage or {}).get("tokens"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "json",
    )


def _response_blocks(sample: ComparisonSample) -> list[str]:
    lines = [
        _metadata_block(sample),
        "",
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


def _render_evidence(
    cases: list[SourceCase],
    personas: list[SourcePersona],
    prompts: dict[tuple[str, str], PromptPair],
    samples: list[ComparisonSample],
) -> list[str]:
    lines = [
        "## 每场景：完整输入、两种提示词与逐次响应",
        "",
        "这是合成的单次事件与角色状态探针，结合真实角色的人格摘录；不是从角色故事中截取的场景，"
        "也没有执行完整模拟。当前模拟时刻未单独指定；计划中的起止毫秒是输入原文。未给出的空间、"
        "身份或通信条件不补写。",
        "",
        "每个场景依次列出：完整状态 → 各人格的 field/distribution 两种完整用户提示词 → 各自全部重复的"
        "原始响应与解释。提示词直接取自已保存来源 prompt 派生出的本轮 prompt，不是按当前模板重建；"
        "两种模式共享逐字一致的输入前缀，只在【输出】段不同。"
        "调用通道、额外系统提示与请求参数以本轮记录为准；未记录的供应商附加内容不能据此还原。",
        "",
        "| 场景 | 分组 | 完整输入与结果 |",
        "| --- | --- | --- |",
    ]
    for index, case in enumerate(cases, 1):
        lines.append(f"| {index} | {case.suite} | [{_cell(case.title)}](#scene-{index}) |")
    lines += ["", "### 人格条件及来源", ""]
    for persona in personas:
        lines += [
            f"#### {persona.label}：{persona.real_name or '无额外人格基线'}",
            "",
            _fenced(
                json.dumps(
                    {
                        "source_path": persona.source_path,
                        "source_sha256": persona.sha256,
                        "text_length": len(persona.text),
                        "实际注入来源 prompt 的人格摘录": persona.text,
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
            "**完整场景状态（实际保存输入）**",
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
        ]
        case_samples = [sample for sample in samples if sample.case_id == case.id]
        for persona in personas:
            profile_samples = [
                sample for sample in case_samples if sample.profile_label == persona.label
            ]
            lines += [
                f"#### {persona.label}：{persona.real_name or '无额外人格基线'}",
                "",
            ]
            prompt = prompts.get((case.id, persona.label))
            if prompt is None:
                lines += ["缺少本轮 prompt。", ""]
                continue
            lines += [
                "**field 保存的完整提示词**（实际调用文本若不同，下方逐次另列）",
                "",
                _fenced(prompt.field_prompt),
                "",
                "**distribution 保存的完整提示词**（保留来源完整原文）",
                "",
                _fenced(prompt.distribution_prompt),
                "",
                f"- 共同输入前缀 sha256：`{prompt.prefix_sha256}`",
                "",
            ]
            if not profile_samples:
                lines += ["缺少调用记录。", ""]
                continue
            lines += [
                "| mode | 调用序号 | 原始路由 | 三值(c/r/k) | argmax | tie | 有效 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for sample in sorted(profile_samples, key=lambda item: (item.repeat, item.mode)):
                probs = sample.probabilities
                values = " / ".join(str(probs[route]) for route in ROUTES) if probs else "—"
                lines.append(
                    f"| {sample.mode} | {sample.call_index} | {sample.route or '—'} | "
                    f"{values} | {sample.argmax or '—'} | {sample.tie} | "
                    f"{'是' if _valid(sample) else '否'} |"
                )
            lines.append("")
            for sample in sorted(profile_samples, key=lambda item: (item.repeat, item.mode)):
                lines += [
                    f"**{sample.mode}，第 {sample.repeat + 1} 次（r{sample.repeat}），"
                    f"调用序号 {sample.call_index}**",
                    "",
                ]
                expected_prompt = (
                    prompt.field_prompt if sample.mode == "field" else prompt.distribution_prompt
                )
                if sample.prompt != expected_prompt:
                    lines += [
                        "**本次实际提示词（与 manifest 中保存的文本不同）**",
                        "",
                        _fenced(sample.prompt),
                        "",
                    ]
                lines += _response_blocks(sample)
    return lines


def _render_pair_table(
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    repeats: int,
) -> list[str]:
    grouped = _group_pairs(samples)
    lines = [
        "## 配对结果总表（全部 pair，含所有不一致）",
        "",
        "| pair | field route | field note | distribution c/r/k | argmax | tie | distribution note | 一致 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        for persona in personas:
            for repeat in range(repeats):
                pair_id = f"{case.id}/{persona.label}/r{repeat}"
                modes = grouped.get(pair_id, {})
                field_sample = modes.get("field")
                dist_sample = modes.get("distribution")
                field_route = (
                    field_sample.route if field_sample and _scorable(field_sample) else "—"
                )
                field_note = (
                    _cell(field_sample.note or "")
                    if field_sample and field_sample.note is not None
                    else "—"
                )
                probs = dist_sample.probabilities if dist_sample else None
                dist_values = " / ".join(str(probs[route]) for route in ROUTES) if probs else "—"
                argmax = dist_sample.argmax if dist_sample and _scorable(dist_sample) else "—"
                tie = dist_sample.tie if dist_sample else "—"
                dist_note = (
                    _cell(dist_sample.note or "")
                    if dist_sample and dist_sample.note is not None
                    else "—"
                )
                if field_sample is None or dist_sample is None:
                    consistent = "缺调用"
                elif not _scorable(field_sample) or not _scorable(dist_sample):
                    consistent = "不可评分"
                else:
                    consistent = "是" if field_sample.route == dist_sample.argmax else "否"
                lines.append(
                    f"| {pair_id} | {field_route} | {field_note} | {dist_values} | "
                    f"{argmax} | {tie} | {dist_note} | {consistent} |"
                )
    lines.append("")
    return lines


def _render_markdown(
    summary: dict[str, Any],
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    prompts: dict[tuple[str, str], PromptPair],
    source_meta: dict[str, Any],
    outdir: Path,
) -> str:
    def latency(data: dict[str, Any]) -> str:
        return f"{_fmt(data['median'])} / {_fmt(data['p95'])}"

    def cost_line(data: dict[str, Any]) -> str:
        tokens = data.get("tokens_sum")
        if tokens is None:
            return "tokens=n/a；cost=n/a"
        rendered = " ".join(
            f"{key}={tokens[key]}"
            for key in ("input", "output", "reasoning", "cache_read", "cache_write")
        )
        return f"tokens({rendered})；cost={_fmt(data.get('total_cost'))}"

    paired = summary["paired"]
    official = source_meta.get("transport") == "deepseek-official"
    cost_note = (
        "官方 API 未返回 cost，保持 n/a；output 已包含 reasoning，不把子集再次叠加。"
        if official
        else "cost 是 OpenCode usage 元数据，不是账单；保留 tokens 和 cache 原始口径。"
    )
    lines = [
        "# System1 对照探针报告（相同输入的 route 字段 vs 三路倾向分布）",
        "",
        f"- 模型：`{summary['model']}`；模式：{', '.join(summary['modes'])}；"
        f"每场景/人格重复 {summary['repeats']} 次，每次分别调用两种模式",
        f"- 预期调用：{summary['expected_calls']}；实际：{summary['actual_calls']}；"
        f"调用失败：{summary['call_failures']}；超时：{summary['timed_out']}；"
        f"非零退出：{summary['nonzero_exit']}；schema 失败：{summary['schema_failures']}；"
        f"工具污染：{summary['tool_polluted']}",
        f"- 预期 pair：{paired['expected_pairs']}；实际 pair：{paired['actual_pairs']}；"
        f"完整 pair：{paired['complete_pairs']}；完整可比较 pair：{paired['comparable_pairs']}；"
        f"缺失：{paired['missing_pairs']}；失败：{paired['failed_pairs']}",
        f"- 耗时中位数/p95（秒）：{latency(summary['latency_seconds'])}；"
        f"usage 覆盖：{_fmt(summary['usage_coverage'])}；usage 来源：{summary['usage_sources']}",
        f"- 合计 {cost_line(summary['cost_tokens'])}",
        "",
        "> 独立 System1 对照探针：分布模式是「事件对当前状态影响所驱动的路径倾向」，"
        "不是世界事件发生概率；离线 argmax 评分不是正式路由策略，自称概率未校准。"
        "本轮以旧敏感度批次为**只读输入来源**，旧 96 次响应不与本轮样本混用，也不当作新鲜配对。",
        "",
        "> `field` 与 `distribution` 共享逐字一致的输入前缀，只替换最后的【输出】段；"
        "受控改动只有输出格式；观察到的差异仍可能包含模型随机性和调用时延波动。"
        "前缀 sha256 在每场景每人格下逐条可核对。",
        "",
        "## 来源批次（旧，只读输入，不参与本轮配对）",
        "",
        _fenced(json.dumps(source_meta, ensure_ascii=False, indent=2), "json"),
        "",
        "本轮 prompt 由上述 `manifest.json` 的 personas/cases 与 `samples.jsonl` 的原始 prompt "
        "派生；不读取当前 `~/lin`，也不按当前模板重建。旧批次只提供输入，其响应未进入本轮任何样例。",
        "",
    ]
    if summary.get("execution"):
        lines += [
            "## 官方请求配置（两种模式共用）",
            "",
            _fenced(json.dumps(summary["execution"], ensure_ascii=False, indent=2), "json"),
            "",
            "每次实际请求体及原始 API 响应保存于 samples.jsonl 的 raw_events；不保存密钥或请求头。",
            "",
        ]
    lines += _render_evidence(cases, personas, prompts, samples)
    lines += _render_pair_table(samples, cases, personas, summary["repeats"])
    lines += [
        '<a id="statistics"></a>',
        "",
        "## 汇总统计",
        "",
        "### 每模式覆盖与 schema",
        "",
        "| 模式 | 预期 | 实际 | 调用失败 | 超时 | 非零退出 | 工具污染 | schema合法 | schema率(预期) | "
        "有效 | 有效率(预期) | 并列 | 无并列可评分 | 可评分率(有效) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        data = summary["by_mode"][mode]
        lines.append(
            f"| {mode} | {data['expected']} | {data['actual']} | {data['call_failures']} | "
            f"{data['timed_out']} | {data['nonzero_exit']} | {data['tool_polluted']} | "
            f"{data['schema_valid']} | {_fmt(data['schema_valid_rate_over_expected'])} | "
            f"{data['valid']} | {_fmt(data['valid_rate_over_expected'])} | {data['ties']} | "
            f"{data['scorable']} | {_fmt(data['scorable_rate_over_valid'])} |"
        )
    lines += [
        "",
        "### 配对一致（分母明确）",
        "",
        f"- 预期 pair：{paired['expected_pairs']}；实际 pair（至少一条调用）：{paired['actual_pairs']}；"
        f"缺失 pair：{paired['missing_pairs']}；失败 pair（两条未同时有效）：{paired['failed_pairs']}",
        f"- 完整 pair（两条均有效）：{paired['complete_pairs']}；"
        f"完整可比较 pair（field route + distribution 唯一 argmax）：{paired['comparable_pairs']}",
        f"- 一致：{paired['consistent_pairs']} / 可比较 {paired['comparable_pairs']}"
        f"（{_fmt(paired['consistent_over_comparable'])}）；"
        f"不一致：{paired['inconsistent_pairs']}",
        f"- 可比较率（可比较/预期）：{_fmt(paired['coverage_comparable_over_expected'])}；"
        f"完整率（完整/预期）：{_fmt(paired['coverage_complete_over_expected'])}",
        f"- 顺序均衡：field 先 {paired['field_first_pairs']} 对；distribution 先 "
        f"{paired['distribution_first_pairs']} 对（{paired['order_definition']}）",
    ]
    if paired["missing_list"]:
        lines.append(f"- 缺失 pair：{paired['missing_list']}")
    if paired["failed_list"]:
        lines.append(f"- 失败 pair：{paired['failed_list']}")
    if paired["inconsistent_list"]:
        lines.append("")
        lines.append("全部不一致 pair：")
        for item in paired["inconsistent_list"]:
            lines.append(
                f"- {item['pair_id']}: field={item['field']}；distribution={item['distribution']}"
            )
    lines += [
        "",
        "### 重复稳定性（每模式，同 case/profile 全重复有效才比较）",
        "",
        "| 模式 | 组数 | 可比组 | 稳定组 | 覆盖(可比/组) | 稳定率(稳定/可比) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        data = summary["stability"][mode]
        lines.append(
            f"| {mode} | {data['groups']} | {data['compared_groups']} | {data['stable_groups']} | "
            f"{_fmt(data['coverage'])} | {_fmt(data['stability_rate'])} |"
        )
    for mode in MODES:
        data = summary["stability"][mode]
        if data["unstable"]:
            lines.append(f"- {mode} 不稳定：{data['unstable']}")
        if data["incomplete"]:
            lines.append(f"- {mode} 未满足全重复有效（不计为稳定）：{data['incomplete']}")
    if summary["repeats"] < 2:
        lines.append("- 本批每 pair 重复数 < 2，稳定性对比不具判别力。")
    lines += [
        "",
        "### 耗时 / tokens / cost（每模式）",
        "",
        "| 模式 | 耗时中位/p95(秒) | usage覆盖 | usage来源 | tokens/cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        data = summary["by_mode"][mode]
        lines.append(
            f"| {mode} | {latency(data['latency_seconds'])} | {_fmt(data['usage_coverage'])} | "
            f"{data['usage_sources']} | {cost_line(data['cost_tokens'])} |"
        )
    diff = summary["latency_diff_field_minus_distribution"]
    lines += [
        "",
        f"- 配对耗时差 field − distribution 中位数："
        f"{_fmt(diff['median_field_minus_distribution'])} 秒（{diff['pairs']} 对；{diff['definition']}）",
        f"- {cost_note}",
        "",
        "### 参考标签符合率（离线，不进 prompt）",
        "",
        "| 模式 | 有标签状态数 | 预期分母 | 有效分母 | 符合 | 符合率(预期) | 符合率(有效) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        data = summary["reference_match"][mode]
        lines.append(
            f"| {mode} | {len(data['labeled_cases'])} | {data['expected_denominator']} | "
            f"{data['valid_denominator']} | {data['matches']} | "
            f"{_fmt(data['rate_over_expected'])} | {_fmt(data['rate_over_valid'])} |"
        )
    lines += [
        "",
        "> 失败/未评分样本计入预期分母；无固定参考的 intensity 状态不参与，不虚报正确率。"
        "允许路由集合含多条时不要求唯一答案。",
        "",
        "### 逐条错误",
        "",
    ]
    if summary["errors"]:
        for error in summary["errors"]:
            lines.append(f"- {error}")
    else:
        lines.append("- 无")
    lines += [
        "",
        "## 实验记录来源",
        "",
        _fenced(
            json.dumps(
                {
                    "说明": "原始记录文件哈希及本次报告生成器版本；运行时源码哈希另保存在 manifest",
                    "保存的实验参数": {
                        "probe": summary["probe"],
                        "model": summary["model"],
                        "modes": summary["modes"],
                        "repeats": summary["repeats"],
                        "seed": summary["seed"],
                        "seed_scope": summary["seed_scope"],
                        "smoke": summary["smoke"],
                    },
                    "来源批次哈希": {
                        "manifest_sha256": source_meta.get("manifest_sha256"),
                        "samples_sha256": source_meta.get("samples_sha256"),
                    },
                    "源码": {
                        "script": str(Path(__file__).resolve()),
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    },
                    "原始记录文件sha256": {
                        name: _sha256_file(outdir / name)
                        for name in ("manifest.json", "samples.jsonl")
                        if (outdir / name).exists()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            "json",
        ),
        "",
        "## 限制",
        "",
        "- 样本量小（本默认批 12×4×2×2=192 次），不能据此宣称校准或统计显著；"
        "符合率只是对保存的离线参考标签的比对。",
        "- 两种输出任务格式不同（枚举字段 vs 三路分布），差异可能来自任务格式本身，不能单独归因于模型能力。",
        f"- {summary['seed_scope']}，模型侧仍有随机性。",
        "- 分布数值是模型**自报倾向**，未经校准，不是世界事件概率或真实决策概率；argmax 不是正式路由策略。",
        "- 真实姓名替换为中性编号，但角色设定内容可能仍被模型凭已有知识识别，属**非完全盲测**；"
        "各 profile 来源文本长度不同，不能单独归因。",
        f"- {cost_note}",
        "- 当前模拟时刻未在来源中单独指定；场景是合成状态加真实人格，不补写未给出的位置、身份或剧情。",
        (
            "- 官方请求体和原始响应已保存；未把旧 OpenCode 通道的样本、usage 或附加提示词混入本批。"
            if official
            else "- 未保存的 OpenCode／供应商附加系统提示不在记录范围内，不能把 CLI 输入文本等同于完整供应商请求。"
        ),
    ]
    return "\n".join(lines) + "\n"


def write_report(
    *,
    outdir: Path,
    samples: list[ComparisonSample],
    cases: list[SourceCase],
    personas: list[SourcePersona],
    prompts: dict[tuple[str, str], PromptPair],
    source_meta: dict[str, Any],
    model: str,
    repeats: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    summary = compute_summary(
        samples=samples,
        cases=cases,
        personas=personas,
        model=model,
        repeats=repeats,
        seed=seed,
        smoke=smoke,
    )
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        execution = json.loads(manifest_path.read_text(encoding="utf-8")).get("execution")
        if execution:
            summary["execution"] = execution
            summary["seed_scope"] = "仅控制计划入队顺序；不设置模型seed，实际完成顺序取决于并发响应"
            if execution.get("transport") == "deepseek-official":
                summary["paired"]["order_definition"] = (
                    "按 call_index 统计计划入队先后，不代表实际执行或完成先后；"
                    "实际开始时间与完成行序见 raw_events 和 samples.jsonl"
                )
                usage_note = (
                    "直接汇总官方 API usage；output 含 reasoning，reasoning/cache hit 是子集；"
                    "未返回的 cost/cache write 记 n/a"
                )
                summary["cost_tokens"]["note"] = usage_note
                for mode in MODES:
                    summary["by_mode"][mode]["cost_tokens"]["note"] = usage_note
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _render_markdown(summary, samples, cases, personas, prompts, source_meta, outdir)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return summary


def _load_prompts_from_manifest(manifest: dict[str, Any]) -> dict[tuple[str, str], PromptPair]:
    raw = manifest.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest 缺少 prompts")
    prompts: dict[tuple[str, str], PromptPair] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("manifest prompts 项必须是对象")
        for key in ("case_id", "profile_label", "distribution_prompt"):
            if key not in item:
                raise ValueError(f"manifest prompts 缺字段：{key}")
        if not isinstance(item.get("field_prompt"), str):
            raise ValueError("manifest prompts 缺少保存的 field_prompt")
        distribution_prompt = str(item["distribution_prompt"])
        field_prompt = item["field_prompt"]
        prefix, instruction = split_prompt(distribution_prompt)
        field_prefix, _ = split_prompt(field_prompt)
        if prefix != field_prefix:
            raise ValueError("保存的两种提示词的共同输入前缀不一致")
        pair = PromptPair(
            case_id=str(item["case_id"]),
            profile_label=str(item["profile_label"]),
            prefix=prefix,
            distribution_instruction=instruction,
            field_prompt=field_prompt,
            distribution_prompt=distribution_prompt,
            prefix_sha256=hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        )
        if item.get("prefix_sha256") not in (None, pair.prefix_sha256):
            raise ValueError(f"manifest prefix_sha256 与 prompt 不一致：{pair.case_id}")
        prompt_key = (pair.case_id, pair.profile_label)
        if prompt_key in prompts:
            raise ValueError(f"manifest prompts 身份重复：{prompt_key}")
        prompts[prompt_key] = pair
    return prompts


def regenerate_report(outdir: Path) -> dict[str, Any]:
    """只用本轮保存的 manifest/samples 重导出，不读来源、不初始化 agent、不调用模型。"""
    manifest_path = outdir / "manifest.json"
    samples_path = outdir / "samples.jsonl"
    if not manifest_path.exists() or not samples_path.exists():
        raise ValueError(f"目录缺少 manifest.json/samples.jsonl：{outdir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("probe") != COMPARISON_PROBE:
        raise ValueError("该目录不是 System1 对照探针记录")
    if "repeats" not in manifest or "model" not in manifest or "cases" not in manifest:
        raise ValueError("manifest 缺少 repeats/model/cases")
    cases = [
        SourceCase(
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
        SourcePersona(
            label=str(item["label"]),
            real_name=None if item.get("real_name") is None else str(item["real_name"]),
            source_path=None if item.get("source_path") is None else str(item["source_path"]),
            sha256=None if item.get("sha256") is None else str(item["sha256"]),
            text=str(item["text"]),
            field_paths=tuple(str(entry) for entry in item.get("field_paths", ())),
        )
        for item in manifest["personas"]
    ]
    prompts = _load_prompts_from_manifest(manifest)
    sample_fields = {item.name for item in fields(ComparisonSample)}
    samples: list[ComparisonSample] = []
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        values = {key: value for key, value in record.items() if key in sample_fields}
        values["expected_routes"] = tuple(record.get("expected_routes", ()))
        values["usage"] = (
            {"cost": record.get("cost"), "tokens": record.get("tokens")}
            if record.get("usage_available")
            else None
        )
        samples.append(ComparisonSample(**values))
    return write_report(
        outdir=outdir,
        samples=samples,
        cases=cases,
        personas=personas,
        prompts=prompts,
        source_meta=manifest.get("source", {}),
        model=str(manifest["model"]),
        repeats=int(manifest["repeats"]),
        seed=int(manifest.get("seed", 0)),
        smoke=bool(manifest.get("smoke", False)),
    )


def build_manifest(
    *,
    bundle: SourceBundle,
    cases: list[SourceCase],
    personas: list[SourcePersona],
    pairs: list[PlannedPair],
    prompts: dict[tuple[str, str], PromptPair],
    model: str,
    agent: str,
    repeats: int,
    seed: int,
    timeout: float,
    smoke: bool,
    no_thinking: bool,
    no_db_usage: bool,
    usage_db: str,
) -> dict[str, Any]:
    case_ids = {case.id for case in cases}
    persona_labels = {persona.label for persona in personas}
    return {
        "probe": COMPARISON_PROBE,
        "model": model,
        "agent": agent,
        "modes": list(MODES),
        "repeats": repeats,
        "seed": seed,
        "seed_scope": "仅用于实验顺序与两模式先后；无法设置模型真实温度/seed",
        "timeout_seconds": timeout,
        "smoke": smoke,
        "no_thinking": no_thinking,
        "no_db_usage": no_db_usage,
        "usage_db": usage_db,
        "source": bundle.source_meta,
        "source_hashes": {
            str(path.resolve()): _sha256_file(path)
            for path in (Path(__file__), Path(routing.__file__), Path(sens.__file__))
        },
        "personas": [persona.to_manifest() for persona in personas],
        "cases": [case.to_manifest() for case in cases],
        "prompts": [
            prompts[(case_id, label)].to_manifest()
            for (case_id, label) in sorted(prompts)
            if case_id in case_ids and label in persona_labels
        ],
        "task_order": [
            {
                "pair_id": pair.pair_id,
                "case_id": pair.case.id,
                "profile_label": pair.persona.label,
                "repeat": pair.repeat,
                "mode_order": list(pair.mode_order),
            }
            for pair in pairs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agent", default=routing.DEFAULT_AGENT)
    parser.add_argument("--opencode", default=routing.DEFAULT_OPENCODE)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--usage-db", default=routing.DEFAULT_USAGE_DB)
    parser.add_argument("--no-db-usage", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="每 suite 首例 × baseline/P1 × 每模式 × 1 次"
    )
    parser.add_argument("--report-from", type=Path, help="从本轮保存目录离线重导出，不调用模型")
    args = parser.parse_args(argv)

    if args.report_from is not None:
        try:
            summary = regenerate_report(args.report_from)
        except (OSError, ValueError, KeyError, SourceError) as error:
            print(f"离线重导出失败：{error}", file=sys.stderr)
            return 2
        print(
            f"离线报告：{args.report_from / 'report.md'}；"
            f"实际调用 {summary['actual_calls']}，完整可比较对 {summary['paired']['comparable_pairs']}"
        )
        return 0

    if args.repeats < 1:
        raise SystemExit("--repeats 必须是正整数")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout 必须是有限正数")

    try:
        bundle = load_source(Path(args.source_dir))
    except SourceError as error:
        print(f"来源错误：{error}", file=sys.stderr)
        return 2

    cases = list(bundle.cases)
    personas = list(bundle.personas)
    if args.smoke:
        cases, personas = _select_smoke(cases, personas)
    repeats = 1 if args.smoke else args.repeats
    try:
        pairs = build_pairs(cases, personas, bundle.prompts, repeats, args.seed)
    except SourceError as error:
        print(f"来源错误：{error}", file=sys.stderr)
        return 2
    calls = build_calls(pairs)

    if args.dry_run:
        print(
            f"用例数：{len(cases)}；profile：{[p.label for p in personas]}；"
            f"重复：{repeats}；pair 数：{len(pairs)}；调用数：{len(calls)}"
        )
        print(f"模型：{args.model}；agent：{args.agent}；opencode：{args.opencode}")
        print(f"seed：{args.seed}（仅顺序与两模式先后）")
        for pair in pairs[:2]:
            print("=" * 60)
            print(f"pair {pair.pair_id}：mode_order={pair.mode_order}")
            print("--- field prompt ---")
            print(pair.prompt.field_prompt)
            print("--- distribution prompt ---")
            print(pair.prompt.distribution_prompt)
        return 0

    workdir = Path(args.workdir)
    routing.ensure_agent_dir(workdir)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path("sceneweave-runs") / "system1-comparison" / timestamp)
    )
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆写混入旧样本：{outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        bundle=bundle,
        cases=cases,
        personas=personas,
        pairs=pairs,
        prompts=bundle.prompts,
        model=args.model,
        agent=args.agent,
        repeats=repeats,
        seed=args.seed,
        timeout=args.timeout,
        smoke=args.smoke,
        no_thinking=args.no_thinking,
        no_db_usage=args.no_db_usage,
        usage_db=args.usage_db,
    )
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    samples_path = outdir / "samples.jsonl"
    print(f"输出目录：{outdir}")
    print(f"pair 数：{len(pairs)}；调用数：{len(calls)}（模型 {args.model}）")

    samples: list[ComparisonSample] = []
    for index, call in enumerate(calls, start=1):
        print(f"[{index}/{len(calls)}] {call.pair_id}/{call.mode} ...", flush=True)
        exit_code, elapsed, timed_out, stdout, stderr, call_error = routing.run_one(
            opencode_bin=args.opencode,
            workdir=workdir,
            agent=args.agent,
            model=args.model,
            prompt=call.prompt,
            timeout=args.timeout,
            thinking=not args.no_thinking,
        )
        parsed = routing.parse_events(stdout)
        sample = ComparisonSample(
            call_index=index,
            pair_id=call.pair_id,
            case_id=call.case.id,
            case_suite=call.case.suite,
            profile_label=call.persona.label,
            profile_real_name=call.persona.real_name,
            repeat=call.repeat,
            mode=call.mode,
            model=args.model,
            prompt=call.prompt,
            prefix_sha256=call.prefix_sha256,
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
            expected_routes=call.case.expected_routes,
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
                result = _parse(sample.text, call.mode)
                sample.parse_ok = True
                sample.route = result["route"]
                sample.argmax = result["argmax"]
                sample.tie = bool(result["tie"])
                sample.margin = result["margin"]
                sample.entropy = result["entropy"]
                sample.probabilities = result["probabilities"]
                sample.note = result["note"]
                value = sample.route if call.mode == "field" else sample.argmax
                sample.in_expected = (
                    value in call.case.expected_routes if call.case.expected_routes else None
                )
            except ValueError as error:
                sample.parse_error = str(error)
        samples.append(sample)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
        status = "ok" if sample.parse_ok else ("call-error" if sample.call_error else "parse-error")
        print(
            f"    {status}: route={sample.route} argmax={sample.argmax} tie={sample.tie} "
            f"{sample.elapsed_seconds:.1f}s"
        )

    summary = write_report(
        outdir=outdir,
        samples=samples,
        cases=cases,
        personas=personas,
        prompts=bundle.prompts,
        source_meta=bundle.source_meta,
        model=args.model,
        repeats=repeats,
        seed=args.seed,
        smoke=args.smoke,
    )
    paired = summary["paired"]
    print(f"报告：{outdir / 'report.md'}")
    print(
        f"完成：{summary['actual_calls']} 次调用，完整可比较 pair {paired['comparable_pairs']}，"
        f"一致 {paired['consistent_pairs']}，不一致 {paired['inconsistent_pairs']}，"
        f"缺失 pair {paired['missing_pairs']}，失败 pair {paired['failed_pairs']}"
    )
    if (
        summary["call_failures"]
        or summary["tool_polluted"]
        or summary["nonzero_exit"]
        or summary["schema_failures"]
        or paired["missing_pairs"]
        or paired["failed_pairs"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
