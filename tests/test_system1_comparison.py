"""System1 对照探针的离线测试：前缀分割、来源校验、任务编排、统计分母与离线重导出。

默认测试绝不进行真实模型调用，也不读取真实人格仓库或旧批次目录；来源用合成目录，
涉及 ``main`` 的测试都 monkeypatch 掉 ``routing.run_one``，只喂假输出。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe_system1_comparison.py"

GOOD_FIELD = '{"route": "continue", "note": "ok"}'
GOOD_DIST = '{"probabilities": {"continue": 0.2, "replan": 0.3, "rethink": 0.5}, "note": "ok"}'


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_system1_comparison", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comp: Any = _load_module()


def _events_stdout(text: str, tool: bool = False) -> str:
    records = [
        {
            "type": "text",
            "timestamp": 1,
            "part": {"id": "p1", "sessionID": "ses_x", "messageID": "m1", "text": text},
        }
    ]
    if tool:
        records.append({"type": "tool", "timestamp": 2, "part": {"id": "p2", "type": "tool-use"}})
    return "\n".join(json.dumps(record) for record in records)


def _fake_run_one(
    field_text: str = GOOD_FIELD, dist_text: str = GOOD_DIST, elapsed: float = 0.01
) -> Any:
    def fake(**_kwargs: Any) -> tuple[int, float, bool, str, str, None]:
        prompt = _kwargs["prompt"]
        text = field_text if '"route"' in prompt else dist_text
        return (0, elapsed, False, _events_stdout(text), "", None)

    return fake


def _distribution_prompt(case_id: str, profile_label: str, persona_text: str, event: str) -> str:
    return (
        "你是角色内部 System1 的分流判断器。\n\n"
        f"【本角色私有设定】\n{persona_text}\n\n"
        f"【本次事件】\n  {event}\n\n"
        "【本角色当前私有状态】\n  当前意向：交付信封\n\n"
        "【三路含义】\n  - continue\n  - replan\n  - rethink\n\n"
        "【输出】给出 "
        '{"probabilities": {"continue": 0.0, "replan": 0.0, "rethink": 0.0}, "note": "..."} 分布。'
    )


def _default_personas() -> list[dict[str, Any]]:
    return [
        {
            "label": "baseline",
            "real_name": None,
            "source_path": None,
            "sha256": None,
            "field_paths": [],
            "text": "（无额外人格设定）",
        },
        {
            "label": "P1",
            "real_name": "测试角色",
            "source_path": "/tmp/persona.json",
            "sha256": "deadbeef",
            "field_paths": ["personality.核心特征"],
            "text": "- 核心特征：冷静",
        },
    ]


def _default_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "influence-noise",
            "suite": "influence",
            "title": "无关巨响",
            "event": "巨响但已确认无关且安全",
            "intention": "把例行信封交给指定收件人",
            "plan": ["a1 走到原交付位置 60000→90000ms"],
            "known_actions": ["a1（registered）：走到原交付位置，等待开始"],
            "memory": ["这是今天例行的信件交付"],
            "expected_routes": ["continue"],
            "label_rationale": "已确认无关",
        },
        {
            "id": "intensity-move",
            "suite": "intensity",
            "title": "收件人移动",
            "event": "收件人从原位置移动了约 0.5 米",
            "intention": "把例行信封交给指定收件人",
            "plan": ["a1 走到原交付位置 60000→90000ms"],
            "known_actions": ["a1（registered）：走到原交付位置，等待开始"],
            "memory": ["这是今天例行的信件交付"],
            "expected_routes": [],
            "label_rationale": "无固定答案",
        },
        {
            "id": "unconfirmed-pending",
            "suite": "unconfirmed",
            "title": "抓取未确认",
            "event": "抓取结果暂未确认，确认反馈待来",
            "intention": "把例行信封交给指定收件人",
            "plan": ["a2 交付 95000→120000ms（依赖 a1）"],
            "known_actions": ["a1（unconfirmed）：结果未确认"],
            "memory": ["这是今天例行的信件交付"],
            "expected_routes": ["continue", "replan"],
            "label_rationale": "等待确认",
        },
    ]


def _manifest(
    personas: list[dict[str, Any]] | None = None,
    cases: list[dict[str, Any]] | None = None,
    repeats: int = 2,
    probe: str = "independent-system1-sensitivity",
) -> dict[str, Any]:
    return {
        "probe": probe,
        "model": "fake-model",
        "suite_filter": "all",
        "repeats": repeats,
        "seed": 7,
        "source_hashes": {"cases_file_sha256": "abc"},
        "personas": personas if personas is not None else _default_personas(),
        "cases": cases if cases is not None else _default_cases(),
    }


def _records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        for persona in manifest["personas"]:
            for repeat in range(manifest["repeats"]):
                records.append(
                    {
                        "case_id": case["id"],
                        "profile_label": persona["label"],
                        "repeat": repeat,
                        "prompt": _distribution_prompt(
                            case["id"], persona["label"], persona["text"], case["event"]
                        ),
                        "text": "旧响应",
                        "note": "旧解释",
                    }
                )
    return records


def _write_source(tmp_path: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (source / "samples.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return source


def _synthetic_source(tmp_path: Path) -> Path:
    manifest = _manifest()
    return _write_source(tmp_path, manifest, _records(manifest))


def _run_main(
    monkeypatch: pytest.MonkeyPatch, source: Path, outdir: Path, extra: list[str] | None = None
) -> int:
    monkeypatch.setattr(comp.routing, "run_one", _fake_run_one())
    args = [
        "--source-dir",
        str(source),
        "--output-dir",
        str(outdir),
        "--workdir",
        str(outdir.parent / (outdir.name + "-wd")),
        "--no-db-usage",
    ]
    if extra:
        args += extra
    return int(comp.main(args))


def _sample(
    case_id: str,
    profile: str,
    repeat: int,
    mode: str,
    *,
    route: str | None = None,
    probabilities: dict[str, float] | None = None,
    argmax: str | None = None,
    tie: bool = False,
    parse_ok: bool = True,
    exit_code: int | None = 0,
    call_error: str | None = None,
    timed_out: bool = False,
    tool_polluted: bool = False,
    expected: tuple[str, ...] = (),
    elapsed: float = 1.0,
    note: str = "n",
    text: str = "t",
) -> Any:
    return comp.ComparisonSample(
        call_index=0,
        pair_id=f"{case_id}/{profile}/r{repeat}",
        case_id=case_id,
        case_suite="influence",
        profile_label=profile,
        profile_real_name=None,
        repeat=repeat,
        mode=mode,
        model="m",
        prompt="prompt",
        prefix_sha256="hash",
        exit_code=exit_code,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        call_error=call_error,
        parse_ok=parse_ok,
        route=route,
        probabilities=probabilities,
        argmax=argmax,
        tie=tie,
        note=note,
        text=text,
        tool_polluted=tool_polluted,
        expected_routes=expected,
    )


def _bundle_parts() -> tuple[list[Any], list[Any]]:
    personas = [
        comp.SourcePersona("baseline", None, None, None, "（无额外人格设定）", ()),
        comp.SourcePersona("P1", "X", "s", "h", "- 核心特征：冷静", ()),
    ]
    cases = [
        comp.SourceCase(
            "c1",
            "influence",
            "场景一",
            "事件一",
            "意向一",
            ("a1 走",),
            ("a1（registered）",),
            ("记忆一",),
            ("continue",),
            "理由一",
        ),
        comp.SourceCase(
            "c2",
            "intensity",
            "场景二",
            "事件二",
            "意向二",
            ("a2 交付",),
            ("a2（saved）",),
            ("记忆二",),
            (),
            "理由二",
        ),
    ]
    return cases, personas


# --- prompt splitting / output task difference -------------------------------------------


def test_prompt_split_shares_prefix_and_only_output_differs() -> None:
    dist = _distribution_prompt("c", "P1", "- 核心特征：冷静", "事件")
    pair = comp.make_prompt_pair("c", "P1", dist)
    assert pair.distribution_prompt == dist
    prefix = pair.prefix
    assert pair.field_prompt.startswith(prefix)
    assert pair.distribution_prompt.startswith(prefix)
    assert pair.field_prompt[len(prefix) :].startswith("【输出】")
    assert pair.distribution_prompt[len(prefix) :].startswith("【输出】")
    assert "route" in pair.field_prompt
    assert "route" not in prefix
    assert "probabilities" in pair.distribution_prompt
    assert "probabilities" not in pair.field_prompt
    assert pair.prefix_sha256 == hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def test_saved_field_prompt_survives_instruction_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    dist = _distribution_prompt("c", "P1", "人格", "事件")
    pair = comp.make_prompt_pair("c", "P1", dist)
    saved = pair.to_manifest()
    monkeypatch.setattr(comp, "FIELD_INSTRUCTION", "【输出】后来的不同指令")
    restored = comp._load_prompts_from_manifest({"prompts": [saved]})[("c", "P1")]
    assert restored.field_prompt == saved["field_prompt"]
    assert restored.distribution_prompt == dist


def test_single_repeat_does_not_establish_stability() -> None:
    cases, personas = _bundle_parts()
    sample = _sample("c1", "baseline", 0, "field", route="continue")
    stability = comp._stability([sample], "field", cases[:1], personas[:1], repeats=1)
    assert stability["compared_groups"] == 0
    assert stability["stable_groups"] == 0
    assert stability["stability_rate"] is None


def test_split_prompt_rejects_missing_marker() -> None:
    with pytest.raises(comp.SourceError, match="缺少【输出】"):
        comp.split_prompt("没有输出段的提示词")


def test_split_prompt_rejects_marker_not_on_own_line() -> None:
    with pytest.raises(comp.SourceError, match="独立行首"):
        comp.split_prompt("前缀直接跟着【输出】指令")


# --- source validation -------------------------------------------------------------------


def test_load_source_accepts_well_formed_batch(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    bundle = comp.load_source(source)
    assert len(bundle.cases) == 3
    assert len(bundle.personas) == 2
    assert len(bundle.prompts) == 6
    assert bundle.source_meta["manifest_sha256"]
    assert bundle.source_meta["samples_sha256"]
    for case in bundle.cases:
        for persona in bundle.personas:
            pair = bundle.prompts[(case.id, persona.label)]
            assert pair.field_prompt != pair.distribution_prompt


def test_load_source_rejects_inconsistent_repeat_prompts(tmp_path: Path) -> None:
    manifest = _manifest()
    records = _records(manifest)
    for record in records:
        if (
            record["case_id"] == "influence-noise"
            and record["profile_label"] == "baseline"
            and record["repeat"] == 1
        ):
            record["prompt"] = record["prompt"] + "\n额外差异"
    source = _write_source(tmp_path, manifest, records)
    with pytest.raises(comp.SourceError, match="不一致"):
        comp.load_source(source)


def test_load_source_rejects_duplicate_identity(tmp_path: Path) -> None:
    manifest = _manifest()
    records = _records(manifest)
    records.append(dict(records[0]))
    source = _write_source(tmp_path, manifest, records)
    with pytest.raises(comp.SourceError, match="身份重复"):
        comp.load_source(source)


def test_load_source_rejects_coverage_mismatch(tmp_path: Path) -> None:
    manifest = _manifest()
    records = _records(manifest)[:-1]
    source = _write_source(tmp_path, manifest, records)
    with pytest.raises(comp.SourceError, match="覆盖不符"):
        comp.load_source(source)


def test_load_source_rejects_missing_output_marker(tmp_path: Path) -> None:
    manifest = _manifest()
    records = _records(manifest)
    for record in records:
        if record["case_id"] == "influence-noise" and record["profile_label"] == "baseline":
            record["prompt"] = "没有标记的提示词"
    source = _write_source(tmp_path, manifest, records)
    with pytest.raises(comp.SourceError, match="缺少【输出】"):
        comp.load_source(source)


def test_load_source_rejects_wrong_probe(tmp_path: Path) -> None:
    manifest = _manifest(probe="something-else")
    source = _write_source(tmp_path, manifest, _records(manifest))
    with pytest.raises(comp.SourceError, match="probe"):
        comp.load_source(source)


def test_load_source_rejects_duplicate_case_identity(tmp_path: Path) -> None:
    cases = _default_cases()
    cases.append(dict(cases[0]))
    manifest = _manifest(cases=cases)
    source = _write_source(tmp_path, manifest, _records(manifest))
    with pytest.raises(comp.SourceError, match="身份重复"):
        comp.load_source(source)


# --- task ordering -----------------------------------------------------------------------


def test_build_pairs_deterministic_with_pair_identity(tmp_path: Path) -> None:
    bundle = comp.load_source(_synthetic_source(tmp_path))
    cases = list(bundle.cases)
    personas = list(bundle.personas)
    first = comp.build_pairs(cases, personas, bundle.prompts, 2, seed=11)
    second = comp.build_pairs(cases, personas, bundle.prompts, 2, seed=11)
    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    assert len(first) == 3 * 2 * 2
    for pair in first:
        assert set(pair.mode_order) == {"field", "distribution"}
    calls = comp.build_calls(first)
    assert len(calls) == 2 * len(first)
    for index in range(0, len(calls), 2):
        assert calls[index].pair_id == calls[index + 1].pair_id
        assert {calls[index].mode, calls[index + 1].mode} == {"field", "distribution"}
    # 同一 pair 的两种 prompt 共享前缀。
    for call in calls:
        assert (
            call.prefix_sha256
            == hashlib.sha256(
                call.prompt[: call.prompt.rfind("【输出】")].encode("utf-8")
            ).hexdigest()
        )


def test_build_pairs_repeats_and_modes_within_pair(tmp_path: Path) -> None:
    bundle = comp.load_source(_synthetic_source(tmp_path))
    pairs = comp.build_pairs(list(bundle.cases), list(bundle.personas), bundle.prompts, 2, seed=3)
    keys = {(pair.case.id, pair.persona.label, pair.repeat) for pair in pairs}
    assert len(keys) == len(pairs) == 12
    assert {repeat for _, _, repeat in keys} == {0, 1}


# --- schema parsing ----------------------------------------------------------------------


def test_strict_field_schema() -> None:
    parsed = comp._parse('{"route": "rethink", "note": "x"}', "field")
    assert parsed["route"] == "rethink"
    assert parsed["argmax"] is None
    with pytest.raises(ValueError, match="未知 route"):
        comp._parse('{"route": "go", "note": "x"}', "field")
    with pytest.raises(ValueError):
        comp._parse('{"route": "continue", "note": "x", "extra": 1}', "field")


def test_strict_distribution_and_tie() -> None:
    unique = comp._parse(
        '{"probabilities": {"continue": 0.1, "replan": 0.2, "rethink": 0.7}, "note": "x"}',
        "distribution",
    )
    assert unique["argmax"] == "rethink"
    assert unique["tie"] is False
    tied = comp._parse(
        '{"probabilities": {"continue": 0.5, "replan": 0.5, "rethink": 0.0}, "note": "x"}',
        "distribution",
    )
    assert tied["tie"] is True
    assert tied["argmax"] is None
    with pytest.raises(ValueError, match="之和"):
        comp._parse(
            '{"probabilities": {"continue": 0.1, "replan": 0.1, "rethink": 0.1}, "note": "x"}',
            "distribution",
        )


# --- summary denominators / stability -----------------------------------------------------


def _denominator_samples() -> tuple[list[Any], list[Any], list[Any]]:
    _, personas = _bundle_parts()
    cases = [
        comp.SourceCase(
            "c1", "influence", "场景", "事件", "意向", (), (), (), ("continue",), "理由"
        )
    ]
    samples = [
        _sample("c1", "baseline", 0, "field", route="continue", expected=("continue",)),
        _sample(
            "c1",
            "baseline",
            0,
            "distribution",
            probabilities={"continue": 0.2, "replan": 0.3, "rethink": 0.5},
            argmax="rethink",
            expected=("continue",),
        ),
        _sample("c1", "baseline", 1, "field", route="continue", expected=("continue",)),
        _sample(
            "c1",
            "baseline",
            1,
            "distribution",
            probabilities={"continue": 0.5, "replan": 0.5, "rethink": 0.0},
            argmax=None,
            tie=True,
            expected=("continue",),
        ),
        _sample("c1", "P1", 0, "field", route="continue", expected=("continue",)),
        _sample(
            "c1",
            "P1",
            0,
            "distribution",
            probabilities={"continue": 0.8, "replan": 0.1, "rethink": 0.1},
            argmax="continue",
            expected=("continue",),
        ),
        _sample("c1", "P1", 1, "field", route=None, parse_ok=False, exit_code=0),
        _sample("c1", "P1", 1, "distribution", probabilities=None, argmax=None, parse_ok=False),
    ]
    return samples, cases, personas


def test_summary_denominators_missing_failed_and_not_stable() -> None:
    samples, cases, personas = _denominator_samples()
    summary = comp.compute_summary(
        samples=samples, cases=cases, personas=personas, model="m", repeats=2, seed=1, smoke=False
    )
    paired = summary["paired"]
    assert paired["expected_pairs"] == 4
    assert paired["actual_pairs"] == 4
    # c1/P1/r1 两条都无效 → 失败 pair；其余 3 对完整。
    assert paired["failed_pairs"] == 1
    # baseline/r1 分布并列 → 不可评分；可比较的是 baseline/r0 与 P1/r0。
    assert paired["comparable_pairs"] == 2
    assert paired["consistent_pairs"] == 1
    assert paired["inconsistent_pairs"] == 1
    field_stability = summary["stability"]["field"]
    assert field_stability["compared_groups"] == 1
    assert field_stability["stable_groups"] == 1
    assert "c1/P1" in field_stability["incomplete"]
    # 失败/并列不得被当作稳定。
    dist_stability = summary["stability"]["distribution"]
    assert dist_stability["compared_groups"] == 0
    reference = summary["reference_match"]["field"]
    assert reference["expected_denominator"] == 4
    assert reference["valid_denominator"] == 3
    assert reference["matches"] == 3
    assert reference["rate_over_expected"] == pytest.approx(3 / 4)


def test_summary_missing_pairs_counted(tmp_path: Path) -> None:
    cases, personas = _bundle_parts()
    cases = [cases[0]]
    samples = [
        _sample("c1", "baseline", 0, "field", route="continue"),
        _sample(
            "c1",
            "baseline",
            0,
            "distribution",
            probabilities={"continue": 0.6, "replan": 0.2, "rethink": 0.2},
            argmax="continue",
        ),
    ]
    summary = comp.compute_summary(
        samples=samples, cases=cases, personas=personas, model="m", repeats=2, seed=1, smoke=False
    )
    paired = summary["paired"]
    assert paired["expected_pairs"] == 4
    assert paired["actual_pairs"] == 1
    assert paired["missing_pairs"] == 3
    assert paired["coverage_comparable_over_expected"] == pytest.approx(1 / 4)


# --- report completeness -----------------------------------------------------------------


def test_report_contains_full_inputs_and_raw_outputs(tmp_path: Path) -> None:
    personas = [comp.SourcePersona("P1", "测试角色", "s", "h", "- 核心特征：冷静", ())]
    case = comp.SourceCase(
        "c1",
        "influence",
        "指定收件人移动",
        "收件人仍在房间",
        "交付信封",
        ("a1 走到位置",),
        ("a1 registered",),
        ("已约定交付",),
        ("continue",),
        "理由文本",
    )
    prompt = _distribution_prompt(case.id, "P1", personas[0].text, case.event)
    prompt = prompt + "\n```原文包含围栏```\n不能重新构造"
    pair = comp.make_prompt_pair(case.id, "P1", prompt)
    prompts = {(case.id, "P1"): pair}
    samples = []
    for repeat in range(2):
        samples.append(
            _sample(
                case.id,
                "P1",
                repeat,
                "field",
                route="continue",
                note=f"field解释{repeat}",
                text=f"field原始{repeat}",
            )
        )
        samples.append(
            _sample(
                case.id,
                "P1",
                repeat,
                "distribution",
                probabilities={"continue": 0.7, "replan": 0.2, "rethink": 0.1},
                argmax="continue",
                note=f"dist解释{repeat}",
                text=f"dist原始{repeat}",
            )
        )
    for index, sample in enumerate(samples, 1):
        sample.call_index = index
        sample.session_ids = [f"session-{index}"]
        sample.reasoning = f"推理{index}"
    summary = comp.write_report(
        outdir=tmp_path,
        samples=samples,
        cases=[case],
        personas=personas,
        prompts=prompts,
        source_meta={"source_dir": "src"},
        model="m",
        repeats=2,
        seed=1,
        smoke=False,
    )
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert summary["paired"]["comparable_pairs"] == 2
    for text in (
        case.title,
        case.event,
        case.intention,
        *case.plan,
        *case.known_actions,
        *case.memory,
        case.label_rationale,
    ):
        assert text in report
    assert report.count(pair.field_prompt) == 1, "逐字一致的提示词只合并展示一次"
    assert report.count(pair.distribution_prompt) == 1
    assert "````text" in report, "原文围栏必须被安全包裹"
    for sample in samples:
        assert sample.text in report
        assert sample.note is not None and sample.note in report
        assert sample.session_ids[0] in report
        assert sample.reasoning in report
    assert pair.prefix_sha256 in report


# --- main flow / offline re-export --------------------------------------------------------


def test_main_success_and_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    outdir = tmp_path / "run"
    code = _run_main(monkeypatch, source, outdir)
    assert code == 0
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe"] == "independent-system1-comparison"
    assert manifest["repeats"] == 2
    assert len(manifest["cases"]) == 3
    assert len(manifest["personas"]) == 2
    assert manifest["prompts"] and all("prefix_sha256" in item for item in manifest["prompts"])
    assert manifest["source"]["manifest_sha256"]
    assert manifest["source"]["samples_sha256"]
    task_order = manifest["task_order"]
    assert len(task_order) == 12
    for item in task_order:
        assert set(item["mode_order"]) == {"field", "distribution"}
    records = [
        json.loads(line)
        for line in (outdir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 24
    for record in records:
        assert record["pair_id"]
        assert record["prefix_sha256"]
        assert record["mode"] in ("field", "distribution")
        assert record["prompt"]


def test_main_smoke_is_first_per_suite_baseline_p1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)
    outdir = tmp_path / "smoke"
    code = _run_main(monkeypatch, source, outdir, extra=["--smoke"])
    assert code == 0
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert {case["suite"] for case in manifest["cases"]} == {
        "influence",
        "intensity",
        "unconfirmed",
    }
    assert {persona["label"] for persona in manifest["personas"]} == {"baseline", "P1"}
    records = [
        json.loads(line)
        for line in (outdir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 12


def test_offline_report_from_uses_saved_inputs_without_source_or_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)
    outdir = tmp_path / "saved"
    assert _run_main(monkeypatch, source, outdir) == 0
    manifest_path = outdir / "manifest.json"
    samples_path = outdir / "samples.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["event"] = "保存的历史事件，不在当前来源文件里"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    before = [manifest_path.read_bytes(), samples_path.read_bytes()]
    old_summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("离线导出不允许读取来源、初始化 agent 或调用模型")

    monkeypatch.setattr(comp, "load_source", forbidden)
    monkeypatch.setattr(comp.routing, "run_one", forbidden)
    monkeypatch.setattr(comp.routing, "ensure_agent_dir", forbidden)
    assert comp.main(["--report-from", str(outdir)]) == 0
    assert [manifest_path.read_bytes(), samples_path.read_bytes()] == before
    assert json.loads((outdir / "summary.json").read_text(encoding="utf-8")) == old_summary
    report = (outdir / "report.md").read_text(encoding="utf-8")
    assert manifest["cases"][0]["event"] in report
    for record in (json.loads(line) for line in before[1].decode().splitlines() if line.strip()):
        assert record["prompt"] in report
        assert record["text"] in report


def test_report_from_invalid_directory_returns_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"probe": "other"}), encoding="utf-8")
    (bad / "samples.jsonl").write_text("", encoding="utf-8")
    assert comp.main(["--report-from", str(bad)]) == 2


def test_nonempty_output_dir_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)
    outdir = tmp_path / "existing"
    outdir.mkdir()
    (outdir / "old.json").write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit, match="拒绝覆写"):
        _run_main(monkeypatch, source, outdir)
    assert (outdir / "old.json").read_text(encoding="utf-8") == "old"


def test_dry_run_makes_no_external_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _synthetic_source(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run 不允许外部调用")

    monkeypatch.setattr(comp.routing, "run_one", forbidden)
    monkeypatch.setattr(comp.routing, "ensure_agent_dir", forbidden)
    code = comp.main(["--dry-run", "--source-dir", str(source)])
    assert code == 0
    out = capsys.readouterr().out
    assert "pair 数" in out
    assert "distribution" in out


def test_invalid_numeric_arguments_are_rejected(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    with pytest.raises(SystemExit):
        comp.main(["--dry-run", "--source-dir", str(source), "--repeats", "0"])
    with pytest.raises(SystemExit):
        comp.main(["--dry-run", "--source-dir", str(source), "--timeout", "0"])
    with pytest.raises(SystemExit):
        comp.main(["--dry-run", "--source-dir", str(source), "--timeout", "inf"])


def test_main_returns_nonzero_on_call_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)

    def fake(**_kwargs: Any) -> tuple[None, float, bool, str, str, str]:
        return (None, 0.01, False, "", "", "OSError: boom")

    monkeypatch.setattr(comp.routing, "run_one", fake)
    code = comp.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "err"),
            "--workdir",
            str(tmp_path / "err-wd"),
            "--no-db-usage",
        ]
    )
    assert code == 1


def test_main_returns_nonzero_on_schema_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)
    monkeypatch.setattr(
        comp.routing, "run_one", _fake_run_one(field_text="{不是 JSON", dist_text="{不是 JSON")
    )
    code = comp.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "schema"),
            "--workdir",
            str(tmp_path / "schema-wd"),
            "--no-db-usage",
        ]
    )
    assert code == 1


def test_main_returns_nonzero_on_tool_pollution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _synthetic_source(tmp_path)

    def fake(**_kwargs: Any) -> tuple[int, float, bool, str, str, None]:
        prompt = _kwargs["prompt"]
        text = GOOD_FIELD if '"route"' in prompt else GOOD_DIST
        return (0, 0.01, False, _events_stdout(text, tool=True), "", None)

    monkeypatch.setattr(comp.routing, "run_one", fake)
    code = comp.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "tool"),
            "--workdir",
            str(tmp_path / "tool-wd"),
            "--no-db-usage",
        ]
    )
    assert code == 1
