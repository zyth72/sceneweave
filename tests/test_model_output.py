"""模型输出严格解析的行为测试：类型/字段/引用/依赖校验，错误不转空。"""

import pytest

from sceneweave.domain.action_plan import (
    ActionKind,
    ActionPlan,
    AtomicAction,
    Effector,
    PlanAction,
)
from sceneweave.domain.model_json import ModelJsonError
from sceneweave.roles.model_output import (
    ModelOutputError,
    System1Route,
    System1RoutePolicy,
    System1Tendency,
    System2Intention,
    System3Unfulfillable,
    parse_system1_tendency,
    parse_system2_intention,
    parse_system3_output,
    parse_system3_plan,
)


def test_system1_valid_tendency_parses() -> None:
    tendency = parse_system1_tendency(
        '{"probabilities": {"continue": 0.82, "replan": 0.16, "rethink": 0.02}, '
        '"note": "事件影响中等"}'
    )
    assert tendency == System1Tendency(
        probabilities={
            System1Route.CONTINUE: 0.82,
            System1Route.REPLAN: 0.16,
            System1Route.RETHINK: 0.02,
        },
        note="事件影响中等",
        total=1.0,
    )
    assert tendency.probabilities[System1Route.CONTINUE] == 0.82
    assert abs(tendency.total - 1.0) < 1e-9
    normalized = tendency.normalized()
    assert abs(normalized[System1Route.CONTINUE] - 0.82) < 1e-9


def test_system1_rounding_tolerance_keeps_raw_values() -> None:
    """允许舍入容差，但保存原始值，归一化有记录（不静默改写原始数）。"""
    tendency = parse_system1_tendency(
        '{"probabilities": {"continue": 0.5, "replan": 0.25, "rethink": 0.24}, "note": "舍入"}'
    )
    assert tendency.total == 0.99
    assert tendency.probabilities[System1Route.RETHINK] == 0.24, "原始值保持"
    assert abs(tendency.normalized()[System1Route.CONTINUE] - 0.5 / 0.99) < 1e-9


def test_system1_old_route_schema_is_explicitly_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system1_tendency('{"route": "continue", "note": "旧 schema"}')


def test_system1_outer_and_inner_field_errors() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system1_tendency('{"note": "缺分布"}')
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 1.0, "replan": 0.0, "rethink": 0.0},'
            ' "note": "x", "extra": 1}'
        )
    with pytest.raises(ModelOutputError, match="probabilities 必须是 JSON 对象"):
        parse_system1_tendency('{"probabilities": 1.0, "note": "x"}')
    with pytest.raises(ModelOutputError, match="未知路径"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 0.5, "replan": 0.3, "rethink": 0.2, "wait": 0.0},'
            ' "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="缺少路径"):
        parse_system1_tendency('{"probabilities": {"continue": 0.5, "replan": 0.5}, "note": "x"}')


def test_system1_bad_probability_values_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="不能是 bool 或字符串"):
        parse_system1_tendency(
            '{"probabilities": {"continue": true, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="不能是 bool 或字符串"):
        parse_system1_tendency(
            '{"probabilities": {"continue": "0.5", "replan": 0.5, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="必须在 0..1"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 1.5, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="必须在 0..1"):
        parse_system1_tendency(
            '{"probabilities": {"continue": -0.1, "replan": 0.6, "rethink": 0.5}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="非有限常量|必须是有限数值"):
        parse_system1_tendency(
            '{"probabilities": {"continue": NaN, "replan": 1.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="非有限常量|必须是有限数值"):
        parse_system1_tendency(
            '{"probabilities": {"continue": Infinity, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )


def test_system1_sum_and_note_errors() -> None:
    with pytest.raises(ModelOutputError, match="之和必须约为 1"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 0.5, "replan": 0.5, "rethink": 0.5}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="之和必须为正"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 0.0, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="note 必须是非空字符串"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 1.0, "replan": 0.0, "rethink": 0.0}, "note": ""}'
        )
    with pytest.raises(ModelOutputError, match="不是合法 JSON"):
        parse_system1_tendency("随便一句话")
    with pytest.raises(ModelOutputError, match="输出为空"):
        parse_system1_tendency("   ")


def test_system1_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="重复键"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 0.5, "continue": 0.5, "replan": 0.5}, "note": "x"}'
        )


def test_system1_route_policy_validates_thresholds() -> None:
    assert System1RoutePolicy().certainty_threshold == 0.75
    assert System1RoutePolicy().margin_threshold == 0.25
    assert System1RoutePolicy(certainty_threshold=0.9, margin_threshold=0.1) == (
        System1RoutePolicy(certainty_threshold=0.9, margin_threshold=0.1)
    )
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            System1RoutePolicy(certainty_threshold=bad)
    with pytest.raises(ValueError):
        System1RoutePolicy(margin_threshold=True)  # type: ignore[arg-type]


def test_system2_valid_intention_parses() -> None:
    intention = parse_system2_intention('{"intention": "拿到抽屉里的钥匙"}')
    assert intention.intention == "拿到抽屉里的钥匙"


def test_system2_abandon_is_no_longer_accepted() -> None:
    """协议纠正：System2 只产意向，abandon 字段被拒绝。"""
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system2_intention('{"intention": "x", "abandon": ["a-lin-1"]}')


def test_system2_missing_or_bad_intention_errors() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system2_intention("{}")
    with pytest.raises(ModelOutputError, match="intention 必须是非空字符串"):
        parse_system2_intention('{"intention": ""}')


def test_system3_one_dimensional_full_list_parses() -> None:
    plan = parse_system3_plan(
        '[{"kind": "reach", "target": "drawer", "effector": "right_hand"},'
        ' {"kind": "grasp", "target": "key", "effector": "right_hand",'
        '  "key": "take", "depends_on": ["open"]},'
        ' {"kind": "move", "destination": {"entity": "door"}}]'
    )
    assert [item.action.kind for item in plan.actions] == [
        ActionKind.REACH,
        ActionKind.GRASP,
        ActionKind.MOVE,
    ]
    assert plan.actions[1].depends_on == ("open",)
    assert plan.actions[0].key is None


def test_system3_explicit_empty_is_valid_not_omitted() -> None:
    plan = parse_system3_plan("[]")
    assert plan == ActionPlan()


def test_system3_two_dimensional_output_is_rejected() -> None:
    """二维数组不再是模型协议，明确拒绝。"""
    with pytest.raises(ModelOutputError, match="必须是 JSON 对象"):
        parse_system3_plan('[[{"kind": "wait"}]]')


def test_system3_free_string_action_is_rejected() -> None:
    with pytest.raises(ModelOutputError, match="必须是 JSON 对象"):
        parse_system3_plan('["走过去并拿走钥匙"]')


def test_system3_id_only_object_is_rejected() -> None:
    with pytest.raises(ModelOutputError, match="缺少 kind"):
        parse_system3_plan('[{"id": "a-lin-1"}]')


def test_system3_unknown_kind_and_field_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知动作种类"):
        parse_system3_plan('[{"kind": "fly", "target": "sky"}]')
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system3_plan(
            '[{"kind": "reach", "target": "x", "effector": "right_hand", "description": "伸手"}]'
        )


def test_system3_missing_required_field_is_model_error_not_keyerror() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system3_plan('[{"kind": "reach", "target": "x"}]')
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system3_plan('[{"kind": "move"}]')


def test_system3_non_finite_and_bool_and_float_time_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="非有限常量"):
        parse_system3_plan('[{"kind": "wait", "extra": NaN}]')
    with pytest.raises(ModelOutputError, match="必须是数值"):
        parse_system3_plan('[{"kind": "move", "destination": {"position": {"x": true, "y": 1}}}]')
    # 坐标必须是有限数值：Infinity 常量在解析前即被拒绝。
    with pytest.raises(ModelOutputError, match="非有限常量"):
        parse_system3_plan(
            '[{"kind": "move", "destination": {"position": {"x": Infinity, "y": 1}}}]'
        )


def test_system3_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="重复键"):
        parse_system3_plan('[{"kind": "wait", "kind": "sit"}]')


def test_system3_self_and_duplicate_dependencies_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="依赖自身"):
        parse_system3_plan('[{"kind": "wait", "key": "a", "depends_on": ["a"]}]')
    with pytest.raises(ModelOutputError, match="不得重复引用"):
        parse_system3_plan(
            '[{"kind": "wait", "key": "a"}, {"kind": "sit", "depends_on": ["a", "a"]}]'
        )


def test_system3_cycle_and_key_identity_conflict_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="循环依赖"):
        parse_system3_plan(
            '[{"kind": "wait", "key": "a", "depends_on": ["b"]},'
            ' {"kind": "wait", "key": "b", "depends_on": ["a"]}]'
        )
    with pytest.raises(ModelOutputError, match="重复的局部 key"):
        parse_system3_plan('[{"kind": "wait", "key": "a"}, {"kind": "sit", "key": "a"}]')


def test_system3_undefined_dependency_is_rejected() -> None:
    # 仅解析层无法知道历史 id，外部引用由角色按已知身份解析；角色层测试覆盖未定义。
    plan = parse_system3_plan('[{"kind": "wait", "depends_on": ["a-history"]}]')
    assert plan.actions[0].depends_on == ("a-history",)


def test_system3_unfulfillable_control_object_is_typed_and_strict() -> None:
    outcome = parse_system3_output(
        '{"error": "intention_unfulfillable", "reason": "当前没有任何可触达的路径"}'
    )
    assert isinstance(outcome, System3Unfulfillable)
    assert outcome.reason == "当前没有任何可触达的路径"
    with pytest.raises(ModelOutputError, match="未知 system3 错误码"):
        parse_system3_output('{"error": "other", "reason": "x"}')
    with pytest.raises(ModelOutputError, match="恰含"):
        parse_system3_output('{"error": "intention_unfulfillable"}')
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system3_output('{"error": "intention_unfulfillable", "reason": "x", "actions": []}')
    with pytest.raises(ModelOutputError, match="reason 必须是非空字符串"):
        parse_system3_output('{"error": "intention_unfulfillable", "reason": ""}')
    with pytest.raises(ModelOutputError, match="不是动作数组"):
        parse_system3_plan('{"error": "intention_unfulfillable", "reason": "x"}')


def test_top_level_must_be_object_for_system2() -> None:
    with pytest.raises(ModelOutputError, match="必须是 JSON 对象"):
        parse_system2_intention('["intention"]')


def test_frozen_results_reject_construction_errors() -> None:
    with pytest.raises(ModelOutputError):
        System1Tendency(probabilities={System1Route.CONTINUE: 1.0}, note="", total=1.0)
    with pytest.raises(ModelOutputError):
        System1Tendency(
            probabilities={
                System1Route.CONTINUE: float("nan"),
                System1Route.REPLAN: 0.0,
                System1Route.RETHINK: 0.0,
            },
            note="x",
            total=0.0,
        )
    with pytest.raises(ModelOutputError):
        System2Intention(intention="")
    with pytest.raises(ModelOutputError):
        System3Unfulfillable(reason="")
    with pytest.raises(ModelJsonError):
        AtomicAction(kind=ActionKind.REACH, effector=Effector.RIGHT_HAND)  # 缺 target
    with pytest.raises(ModelJsonError):
        PlanAction(action=AtomicAction(kind=ActionKind.WAIT), depends_on=("a", "a"))


def test_unknown_fields_in_system3_action_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system3_plan('[{"kind": "wait", "surprise": true}]')
