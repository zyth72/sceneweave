"""System1 敏感度探针的离线测试：夹具一致性、人格白名单、标签不泄露、报告与失败语义。

默认测试绝不进行真实模型调用，也不读取 ~/lin 数据仓库（用合成人格文件）；涉及
``main`` 的测试都 monkeypatch 掉 ``routing.run_one``，只喂假输出。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe_system1_sensitivity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_system1_sensitivity", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sens = _load_module()

_GOOD_DISTRIBUTION = (
    '{"probabilities": {"continue": 0.2, "replan": 0.3, "rethink": 0.5}, "note": "ok"}'
)


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


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, return_value, name: str) -> int:
    def fake_run_one(**_kwargs):
        return return_value

    monkeypatch.setattr(sens.routing, "run_one", fake_run_one)
    return sens.main(
        [
            "--suite",
            "intensity",
            "--limit",
            "1",
            "--profile",
            "baseline",
            "--output-dir",
            str(tmp_path / name),
            "--workdir",
            str(tmp_path / f"{name}-wd"),
        ]
    )


def _synthetic_persona(tmp_path: Path) -> Path:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "name": "测试角色",
                "appearance": "外貌不该被摘取",
                "valentineLines": "亲密台词不该被摘取",
                "personality": {
                    "核心特征": "核心特征是测试角色本人，冷静",
                    "优点": "优点文本",
                    "缺点（性格缺陷）": "缺点文本",
                    "习惯或怪癖": "习惯文本",
                    "完美主义": "非白名单字段",
                },
                "worldview": {"道德准则": "道德文本", "对爱情的看法": "非白名单"},
                "innerDrive": {
                    "核心动机": "动机文本",
                    "短期目标": "短期目标文本",
                    "恐惧与禁忌": "恐惧文本",
                    "长期目标": "非白名单",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _case(case_id: str, suite: str) -> sens.SensCase:
    return sens.SensCase(
        id=case_id,
        suite=suite,
        title=case_id,
        event=f"事件 {case_id}",
        intention="意向",
        plan=(),
        known_actions=(),
        memory=(),
        expected_routes=(),
        label_rationale="理由",
    )


def _sample(
    case_id: str,
    suite: str,
    profile: str,
    repeat: int,
    probs: dict,
    *,
    parse_ok: bool = True,
    exit_code: int | None = 0,
    call_error: str | None = None,
    timed_out: bool = False,
    tool_polluted: bool = False,
) -> sens.SensSample:
    return sens.SensSample(
        task_id=f"{case_id}/{profile}/r{repeat}",
        case_id=case_id,
        suite=suite,
        profile_label=profile,
        profile_real_name=None if profile == "baseline" else "X",
        profile_len=0 if profile == "baseline" else 10,
        profile_sha256=None if profile == "baseline" else "abc",
        repeat=repeat,
        model="m",
        prompt="p",
        exit_code=exit_code,
        call_error=call_error,
        timed_out=timed_out,
        parse_ok=parse_ok,
        tool_polluted=tool_polluted,
        probabilities=probs,
        argmax=max(probs, key=probs.get) if probs else None,
    )


def _personas() -> list[sens.PersonaExcerpt]:
    return [
        sens.PersonaExcerpt("baseline", None, None, None, (), "（无额外人格设定）"),
        sens.PersonaExcerpt(
            "P1", "X", "s", "abc", (("personality.核心特征", "a"),), "- 核心特征：a"
        ),
    ]


def test_cases_file_has_twelve_states_across_three_suites() -> None:
    cases = sens.load_cases()
    assert len(cases) == 12
    assert {case.suite for case in cases} == {"intensity", "influence", "unconfirmed"}
    assert len([c for c in cases if c.suite == "intensity"]) == 6
    assert len([c for c in cases if c.suite == "influence"]) == 3
    assert len([c for c in cases if c.suite == "unconfirmed"]) == 3


def test_intensity_series_changes_only_the_event() -> None:
    cases = [c for c in sens.load_cases() if c.suite == "intensity"]
    base = cases[0]
    for case in cases[1:]:
        assert case.intention == base.intention
        assert case.plan == base.plan
        assert case.known_actions == base.known_actions
        assert case.memory == base.memory
        assert case.event != base.event, "强度序列必须只变事件"


def test_persona_whitelist_excludes_non_whitelisted_fields(tmp_path: Path) -> None:
    _synthetic_persona(tmp_path)
    spec = {"id": "P9", "real_name": "测试角色", "file": "p.json"}
    persona = sens.load_persona(spec, tmp_path)
    assert persona.sha256 is not None
    assert persona.length == len(persona.text)
    assert "测试角色" not in persona.text, "姓名应被移除"
    for forbidden in ("外貌不该被摘取", "亲密台词不该被摘取", "非白名单"):
        assert forbidden not in persona.text
    paths = {path for path, _ in persona.fields}
    assert paths == {
        "personality.核心特征",
        "personality.优点",
        "personality.缺点（性格缺陷）",
        "personality.习惯或怪癖",
        "worldview.道德准则",
        "innerDrive.核心动机",
        "innerDrive.短期目标",
        "innerDrive.恐惧与禁忌",
    }


def test_missing_persona_source_is_explicit_error(tmp_path: Path) -> None:
    spec = {"id": "P1", "real_name": "缺失", "file": "不存在.json"}
    with pytest.raises(sens.PersonaSourceError, match="缺少人格来源文件"):
        sens.load_persona(spec, tmp_path)


def test_baseline_persona_has_no_source() -> None:
    persona = sens.load_persona({"id": "baseline", "real_name": None, "file": None})
    assert persona.sha256 is None
    assert persona.real_name is None
    assert persona.text == "（无额外人格设定）"


def test_prompt_changes_only_persona_block_across_profiles() -> None:
    cases = sens.load_cases()
    baseline = sens.load_persona({"id": "baseline", "real_name": None, "file": None})
    persona = sens.PersonaExcerpt(
        label="P1",
        real_name="某角色",
        source_path="x",
        sha256="abc",
        fields=(("personality.核心特征", "冷静"),),
        text="- 核心特征：冷静",
    )
    for case in cases:
        base_prompt = sens.build_prompt(case, baseline)
        persona_prompt = sens.build_prompt(case, persona)
        marker = "【本次事件】"
        assert (
            base_prompt[base_prompt.index(marker) :]
            == persona_prompt[persona_prompt.index(marker) :]
        )
        assert (
            base_prompt[: base_prompt.index(marker)]
            != persona_prompt[: persona_prompt.index(marker)]
        )


def test_prompt_does_not_leak_labels_or_expectations() -> None:
    cases = sens.load_cases()
    baseline = sens.load_persona({"id": "baseline", "real_name": None, "file": None})
    for case in cases:
        prompt = sens.build_prompt(case, baseline)
        assert "expected_routes" not in prompt
        assert case.label_rationale not in prompt
        assert "目标不变" not in prompt
        for route in case.expected_routes:
            assert f'"route": "{route}"' not in prompt


def test_future_a2_is_never_registered() -> None:
    cases = sens.load_cases()
    for case in cases:
        for action in case.known_actions:
            if "a2" in action:
                assert "registered" not in action, f"{case.id} 的未来 a2 不得是 registered"
                assert "saved" in action, f"{case.id} 的未来 a2 应标为 saved"
    by_id = {case.id: case for case in cases}
    joined = " ".join(by_id["unconfirmed-pending"].known_actions)
    assert "a1（unconfirmed）" in joined
    assert "不得提前提交" in joined
    assert "a1（completed）" in " ".join(by_id["unconfirmed-completed"].known_actions)
    assert "依赖满足，可提交" in " ".join(by_id["unconfirmed-completed"].known_actions)
    assert "a1（failed）" in " ".join(by_id["unconfirmed-failed"].known_actions)


def test_cases_loader_rejects_registered_future_a2(tmp_path: Path) -> None:
    bad = {
        "cases": [
            {
                "id": "x",
                "suite": "influence",
                "title": "t",
                "event": "e",
                "intention": "i",
                "plan": [],
                "known_actions": ["a2（registered）：依赖 a1，等待依赖满足"],
                "memory": [],
                "expected_routes": [],
                "label_rationale": "r",
            }
        ]
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="registered"):
        sens.load_cases(path)


def test_intensity_lowest_level_is_moved_half_meter_within_reach() -> None:
    cases = [c for c in sens.load_cases() if c.suite == "intensity"]
    lowest = cases[0]
    assert "从原位置移动" in lowest.event
    assert "0.5 米" in lowest.event
    assert "伸手可及" in lowest.event
    assert "未移动" not in lowest.event
    assert "没有移动" not in lowest.event


def test_select_profiles_rejects_unknown_even_when_mixed() -> None:
    with pytest.raises(SystemExit, match="未知 profile"):
        sens._select_profiles(["baseline", "ghost"], False)
    with pytest.raises(SystemExit, match="未知 profile"):
        sens._select_profiles(["ghost"], False)
    assert [spec["id"] for spec in sens._select_profiles(["贝尔法斯特"], False)] == ["P1"]


def test_valid_excludes_call_error_timeout_nonzero_exit_and_tool_pollution() -> None:
    probs = {"continue": 0.2, "replan": 0.3, "rethink": 0.5}
    assert sens._valid(_sample("c", "intensity", "baseline", 0, probs))
    assert not sens._valid(_sample("c", "intensity", "baseline", 0, probs, call_error="boom"))
    assert not sens._valid(_sample("c", "intensity", "baseline", 0, probs, timed_out=True))
    assert not sens._valid(_sample("c", "intensity", "baseline", 0, probs, exit_code=2))
    assert not sens._valid(_sample("c", "intensity", "baseline", 0, probs, parse_ok=False))
    assert not sens._valid(_sample("c", "intensity", "baseline", 0, probs, tool_polluted=True))


def test_report_keeps_raw_distributions_coverage_and_baseline_diff(tmp_path: Path) -> None:
    personas = _personas()
    cases = [_case("intensity-a", "intensity"), _case("intensity-b", "intensity")]
    samples = []
    for case in cases:
        for profile in ("baseline", "P1"):
            for repeat in (0, 1):
                shift = 0.1 if profile == "P1" else 0.0
                samples.append(
                    _sample(
                        case.id,
                        case.suite,
                        profile,
                        repeat,
                        {"continue": 0.2 + shift, "replan": 0.3, "rethink": 0.5 - shift},
                    )
                )
    summary = sens.write_report(
        outdir=tmp_path,
        samples=samples,
        personas=personas,
        cases=cases,
        model="m",
        suite="all",
        repeats=2,
        seed=1,
    )
    assert summary["coverage"] == 1.0
    assert summary["valid_samples"] == len(samples)
    assert summary["complete_groups"] == summary["expected_groups"] == 4
    assert "raw" in summary["by_case"]["intensity-a"]["by_profile"]["baseline"]
    assert len(summary["by_case"]["intensity-a"]["by_profile"]["baseline"]["raw"]) == 2
    assert "intensity-a" in summary["profile_minus_baseline"]
    assert "intensity-a/baseline" in summary["repeat_fluctuation"]
    assert summary["profiles"][0]["label"] == "baseline"
    assert (tmp_path / "report.md").exists()


def test_intensity_trend_reports_nonmonotonic_without_fixing(tmp_path: Path) -> None:
    cases = [_case(f"i{n}", "intensity") for n in range(3)]
    personas = [sens.PersonaExcerpt("baseline", None, None, None, (), "（无额外人格设定）")]
    rethink = [0.2, 0.8, 0.3]
    samples = []
    for case, value in zip(cases, rethink, strict=True):
        samples.append(
            _sample(
                case.id,
                "intensity",
                "baseline",
                0,
                {"continue": (1 - value) / 2, "replan": (1 - value) / 2, "rethink": value},
            )
        )
    summary = sens.write_report(
        outdir=tmp_path,
        samples=samples,
        personas=personas,
        cases=cases,
        model="m",
        suite="intensity",
        repeats=1,
        seed=1,
    )
    trend = summary["intensity_trend"]
    assert trend["rethink_adjacent_delta"] == [pytest.approx(0.6), pytest.approx(-0.5)]
    assert trend["rethink_sign_changes"] == 1
    assert [level["mean"]["rethink"] for level in trend["levels"]] == pytest.approx(rethink)


def test_per_profile_intensity_trend_differs_from_pooled(tmp_path: Path) -> None:
    cases = [_case(f"i{n}", "intensity") for n in range(6)]
    personas = _personas()
    samples = []
    for index, case in enumerate(cases):
        base_rethink = 0.1 + 0.1 * index
        samples.append(
            _sample(
                case.id,
                "intensity",
                "baseline",
                0,
                {"continue": 0.0, "replan": 1 - base_rethink, "rethink": base_rethink},
            )
        )
        samples.append(
            _sample(
                case.id,
                "intensity",
                "P1",
                0,
                {"continue": 0.0, "replan": 0.7, "rethink": 0.3},
            )
        )
    summary = sens.write_report(
        outdir=tmp_path,
        samples=samples,
        personas=personas,
        cases=cases,
        model="m",
        suite="intensity",
        repeats=1,
        seed=1,
    )
    trend = summary["intensity_trend"]
    baseline = trend["by_profile"]["baseline"]
    p1 = trend["by_profile"]["P1"]
    assert baseline["stats"]["complete_levels"] == 6
    assert p1["stats"]["complete_levels"] == 6
    assert baseline["stats"]["rethink_spearman"] == pytest.approx(1.0)
    assert p1["stats"]["rethink_spearman"] is None, "恒定序列没有可定义的相关性"
    assert baseline["stats"]["rethink_adjacent_delta"] != p1["stats"]["rethink_adjacent_delta"]
    # 逐 profile 的趋势不能只靠池化结果代表。
    assert trend["by_profile"]["baseline"]["levels"][0]["mean"]["rethink"] == pytest.approx(0.1)
    assert trend["by_profile"]["P1"]["levels"][0]["mean"]["rethink"] == pytest.approx(0.3)


def test_repeat_range_uses_full_max_min_not_first(tmp_path: Path) -> None:
    cases = [_case("i0", "intensity")]
    personas = [sens.PersonaExcerpt("baseline", None, None, None, (), "（无额外人格设定）")]
    continue_values = [0.5, 0.2, 0.9]
    samples = []
    for repeat, value in enumerate(continue_values):
        samples.append(
            _sample(
                "i0",
                "intensity",
                "baseline",
                repeat,
                {"continue": value, "replan": (1 - value) / 2, "rethink": (1 - value) / 2},
            )
        )
    summary = sens.write_report(
        outdir=tmp_path,
        samples=samples,
        personas=personas,
        cases=cases,
        model="m",
        suite="intensity",
        repeats=3,
        seed=1,
    )
    fluctuation = summary["repeat_fluctuation"]["i0/baseline"]
    assert fluctuation["repeats"] == 3
    # 全重复 max-min = 0.9 - 0.2 = 0.7；相对第一次最大差只有 0.4。
    assert fluctuation["route_range"]["continue"] == pytest.approx(0.7)
    assert fluctuation["route_range"]["continue"] != pytest.approx(0.4)


def test_report_groups_partial_coverage_and_missing(tmp_path: Path) -> None:
    cases = [_case("i0", "intensity"), _case("i1", "intensity")]
    personas = _personas()
    probs = {"continue": 0.2, "replan": 0.3, "rethink": 0.5}
    samples = []
    for case in cases:
        for profile in ("baseline", "P1"):
            for repeat in (0, 1):
                # i0/baseline/r1 缺失；i1/P1/r0 无效。
                if case.id == "i0" and profile == "baseline" and repeat == 1:
                    continue
                invalid = case.id == "i1" and profile == "P1" and repeat == 0
                samples.append(
                    _sample(
                        case.id,
                        case.suite,
                        profile,
                        repeat,
                        probs,
                        call_error="boom" if invalid else None,
                        exit_code=None if invalid else 0,
                    )
                )
    summary = sens.write_report(
        outdir=tmp_path,
        samples=samples,
        personas=personas,
        cases=cases,
        model="m",
        suite="intensity",
        repeats=2,
        seed=1,
    )
    assert summary["expected_samples"] == 8
    assert summary["actual_samples"] == 7
    assert summary["valid_samples"] == 6
    assert summary["coverage"] == pytest.approx(6 / 8)
    assert summary["missing_samples"] == ["i0/baseline/r1"]
    assert summary["invalid_samples"] == ["i1/P1/r0"]
    assert summary["complete_groups"] == 2
    assert summary["expected_groups"] == 4
    partial = summary["by_case"]["i1"]["by_profile"]["P1"]
    assert partial["valid"] == 1 and partial["expected"] == 2


def test_dry_run_makes_no_external_call(capsys) -> None:
    code = sens.main(["--dry-run", "--suite", "influence", "--profile", "baseline"])
    assert code == 0
    out = capsys.readouterr().out
    assert "任务数" in out
    assert "probabilities" in out


def test_invalid_numeric_arguments_are_rejected() -> None:
    with pytest.raises(SystemExit):
        sens.main(["--dry-run", "--repeats", "0"])
    with pytest.raises(SystemExit):
        sens.main(["--dry-run", "--limit", "0"])
    with pytest.raises(SystemExit):
        sens.main(["--dry-run", "--timeout", "0"])
    with pytest.raises(SystemExit):
        sens.main(["--dry-run", "--timeout", "inf"])


def test_main_returns_nonzero_on_call_error(monkeypatch, tmp_path: Path) -> None:
    code = _run_main(
        monkeypatch, tmp_path, (None, 0.01, False, "", "", "OSError: boom"), "call-error"
    )
    assert code == 1


def test_main_returns_nonzero_on_schema_error(monkeypatch, tmp_path: Path) -> None:
    code = _run_main(
        monkeypatch,
        tmp_path,
        (0, 0.01, False, _events_stdout("{不是 JSON"), "", None),
        "schema-error",
    )
    assert code == 1


def test_main_returns_nonzero_on_tool_pollution(monkeypatch, tmp_path: Path) -> None:
    code = _run_main(
        monkeypatch,
        tmp_path,
        (0, 0.01, False, _events_stdout(_GOOD_DISTRIBUTION, tool=True), "", None),
        "tool",
    )
    assert code == 1


def test_main_returns_zero_on_success(monkeypatch, tmp_path: Path) -> None:
    code = _run_main(
        monkeypatch,
        tmp_path,
        (0, 0.01, False, _events_stdout(_GOOD_DISTRIBUTION), "", None),
        "success",
    )
    assert code == 0


def test_manifest_records_full_cases_and_source_hashes(monkeypatch, tmp_path: Path) -> None:
    code = _run_main(
        monkeypatch,
        tmp_path,
        (0, 0.01, False, _events_stdout(_GOOD_DISTRIBUTION), "", None),
        "manifest",
    )
    assert code == 0
    manifest = json.loads((tmp_path / "manifest" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"]
    assert manifest["cases"][0]["plan"] and manifest["cases"][0]["known_actions"]
    assert manifest["source_hashes"]["cases_file_sha256"]
    assert manifest["source_hashes"]["script_sha256"]
    assert manifest["personas"][0]["text"] == "（无额外人格设定）"


def test_nonempty_output_dir_is_not_overwritten(monkeypatch, tmp_path: Path) -> None:
    outdir = tmp_path / "existing"
    outdir.mkdir()
    (outdir / "old.json").write_text("old", encoding="utf-8")

    def fake_run_one(**_kwargs):
        return (0, 0.01, False, _events_stdout(_GOOD_DISTRIBUTION), "", None)

    monkeypatch.setattr(sens.routing, "run_one", fake_run_one)
    with pytest.raises(SystemExit, match="拒绝覆写"):
        sens.main(
            [
                "--suite",
                "intensity",
                "--limit",
                "1",
                "--profile",
                "baseline",
                "--output-dir",
                str(outdir),
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert (outdir / "old.json").read_text(encoding="utf-8") == "old"
