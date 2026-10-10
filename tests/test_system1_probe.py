"""System1 路由探针的离线测试：解析、去重、任务编排与 dry-run。

默认测试绝不进行真实模型调用；只验证脚本自身的纯逻辑。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe_system1_routing.py"


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location("probe_system1_routing", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def test_cases_file_has_twelve_balanced_cases() -> None:
    cases = probe.load_cases(probe.CASES_PATH)
    assert len(cases) >= 12
    clear = [c for c in cases if c.kind == "clear"]
    boundary = [c for c in cases if c.kind == "boundary"]
    assert len(clear) >= 9 and len(boundary) >= 3
    covered = {route for case in clear for route in case.expected_routes}
    assert covered == set(probe.ROUTES), "明确案例应覆盖三条路径"
    for case in boundary:
        assert len(case.expected_routes) >= 2, "边界案例允许多个合法路由"


def test_prompt_does_not_leak_expected_routes() -> None:
    cases = probe.load_cases(probe.CASES_PATH)
    for case in cases:
        for mode in ("field", "distribution"):
            prompt = probe.build_prompt(case, mode)
            for route in case.expected_routes:
                assert f'"route": "{route}"' not in prompt
            assert "expected_routes" not in prompt
            assert case.label_rationale not in prompt


def test_prompt_modes_share_context_but_differ_in_output() -> None:
    cases = probe.load_cases(probe.CASES_PATH)
    case = cases[0]
    field_prompt = probe.build_prompt(case, "field")
    dist_prompt = probe.build_prompt(case, "distribution")
    assert case.event in field_prompt and case.event in dist_prompt
    assert "probabilities" not in field_prompt
    assert "probabilities" in dist_prompt
    assert "不是世界事件发生概率" in dist_prompt


def test_parse_field_strict() -> None:
    good = probe.parse_field('{"route": "rethink", "note": "首次"}')
    assert good["route"] == "rethink" and good["tie"] is False
    with pytest.raises(ValueError, match="恰含"):
        probe.parse_field('{"route": "rethink"}')
    with pytest.raises(ValueError, match="未知 route"):
        probe.parse_field('{"route": "go", "note": "x"}')
    with pytest.raises(ValueError, match="非空字符串"):
        probe.parse_field('{"route": "continue", "note": ""}')
    with pytest.raises(ValueError, match="合法 JSON"):
        probe.parse_field("不是 JSON")


def test_parse_distribution_strict_and_unique_argmax() -> None:
    good = probe.parse_distribution(
        '{"probabilities": {"continue": 0.1, "replan": 0.2, "rethink": 0.7}, "note": "x"}'
    )
    assert good["route"] == "rethink"
    assert good["tie"] is False
    assert good["margin"] == pytest.approx(0.5)
    assert good["entropy"] > 0
    with pytest.raises(ValueError, match="必须是数值"):
        probe.parse_distribution(
            '{"probabilities": {"continue": true, "replan": 0.0, "rethink": 1.0}, "note": "x"}'
        )
    with pytest.raises(ValueError, match="0..1"):
        probe.parse_distribution(
            '{"probabilities": {"continue": 1.2, "replan": 0.0, "rethink": -0.2}, "note": "x"}'
        )
    with pytest.raises(ValueError, match="之和"):
        probe.parse_distribution(
            '{"probabilities": {"continue": 0.1, "replan": 0.1, "rethink": 0.1}, "note": "x"}'
        )
    with pytest.raises(ValueError, match="恰含三条路径"):
        probe.parse_distribution('{"probabilities": {"continue": 0.5, "replan": 0.5}, "note": "x"}')


def test_parse_distribution_marks_tie_without_dict_order() -> None:
    tied = probe.parse_distribution(
        '{"probabilities": {"continue": 0.5, "replan": 0.5, "rethink": 0.0}, "note": "x"}'
    )
    assert tied["tie"] is True
    assert tied["route"] is None
    assert tied["margin"] is None


def test_parse_distribution_allows_documented_rounding_tolerance() -> None:
    rounded = probe.parse_distribution(
        '{"probabilities": {"continue": 0.33, "replan": 0.33, "rethink": 0.33}, "note": "x"}'
    )
    assert rounded["tie"] is True, "舍入后并列应记 tie，而不是偷偷重归一化裁决"


def test_parse_events_dedupes_and_extracts_usage() -> None:
    lines = [
        json.dumps(
            {
                "type": "step_start",
                "timestamp": 1,
                "part": {"id": "p1", "sessionID": "ses_x", "messageID": "m1"},
            }
        ),
        json.dumps(
            {
                "type": "text",
                "timestamp": 2,
                "part": {"id": "p2", "sessionID": "ses_x", "messageID": "m1", "text": '{"a":1}'},
            }
        ),
        json.dumps(
            {
                "type": "text",
                "timestamp": 2,
                "part": {"id": "p2", "sessionID": "ses_x", "messageID": "m1", "text": '{"a":1}'},
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "timestamp": 3,
                "part": {
                    "id": "p3",
                    "sessionID": "ses_x",
                    "messageID": "m1",
                    "type": "step-finish",
                    "reason": "stop",
                    "cost": 0.1,
                    "tokens": {"input": 10, "output": 2, "cache": {"read": 1, "write": 0}},
                },
            }
        ),
    ]
    parsed = probe.parse_events("\n".join(lines))
    assert parsed["text"] == '{"a":1}'
    assert len(parsed["raw_events"]) == 3, "重复 part id 去重"
    assert parsed["step_finish"]["tokens"]["input"] == 10
    assert parsed["tool_polluted"] is False


def test_parse_events_flags_tool_pollution() -> None:
    line = json.dumps({"type": "tool", "timestamp": 1, "part": {"id": "p1", "type": "tool-use"}})
    parsed = probe.parse_events(line)
    assert parsed["tool_polluted"] is True
    assert parsed["tool_events"]


def test_build_tasks_interleaves_modes_deterministically() -> None:
    cases = probe.load_cases(probe.CASES_PATH)[:3]
    first = probe.build_tasks(cases, ["field", "distribution"], 2, seed=7)
    second = probe.build_tasks(cases, ["field", "distribution"], 2, seed=7)
    assert first == second, "固定 seed 只影响顺序"
    assert len(first) == 12
    assert {mode for _, mode, _ in first} == {"field", "distribution"}


def test_dry_run_makes_no_external_call(capsys) -> None:
    code = probe.main(["--dry-run", "--limit", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "任务数" in out
    assert "probabilities" in out  # 分布模式提示在 dry-run 中可见


def test_failed_or_single_samples_are_not_reported_as_stable() -> None:
    good = {
        "case_id": "c",
        "mode": "field",
        "parse_ok": True,
        "tool_polluted": False,
        "route": "continue",
    }
    bad = dict(good, parse_ok=False, route=None)
    incomplete = probe._repeat_consistency([good, bad])
    assert incomplete["consistent_groups"] == 0
    assert incomplete["coverage"] == 0
    assert incomplete["consistency_rate"] is None
    single = probe._repeat_consistency([good])
    assert single["groups"] == 0
    assert single["consistency_rate"] is None
    stable = probe._repeat_consistency([good, dict(good)])
    assert stable["coverage"] == stable["consistency_rate"] == 1


def test_unknown_token_fields_remain_unavailable() -> None:
    result = probe._cost_summary([{"tokens": {"output": 3}, "cost": None}])
    assert result["tokens_sum"]["output"] == 3
    assert result["tokens_sum"]["input"] is None
    assert result["tokens_field_coverage"]["input"] == 0


def test_existing_workdir_configuration_is_preserved(tmp_path) -> None:
    config = tmp_path / "opencode.jsonc"
    config.write_text('{"user": true}')
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        probe.ensure_agent_dir(tmp_path)
    assert config.read_text() == '{"user": true}'


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="重复 JSON 字段"):
        probe.parse_field('{"route":"continue","route":"rethink","note":"x"}')
