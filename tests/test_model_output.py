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
    System1Decision,
    System1Route,
    System2Intention,
    System3Unfulfillable,
    parse_system1_decision,
    parse_system2_intention,
    parse_system3_output,
    parse_system3_plan,
)


def test_system1_valid_decision_parses() -> None:
    decision = parse_system1_decision('{"route": "rethink", "note": "门开了，需要警惕"}')
    assert decision == System1Decision(route=System1Route.RETHINK, note="门开了，需要警惕")
    assert parse_system1_decision('{"route": "continue", "note": "照旧"}').route is (
        System1Route.CONTINUE
    )
    assert parse_system1_decision('{"route": "replan", "note": "重做"}').route is (
        System1Route.REPLAN
    )


def test_system1_missing_fields_and_bad_types_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system1_decision('{"note": "缺少判断"}')
    with pytest.raises(ModelOutputError, match="未知 route"):
        parse_system1_decision('{"route": "reconsider", "note": "旧字段"}')
    with pytest.raises(ModelOutputError, match="route 必须是字符串"):
        parse_system1_decision('{"route": 1, "note": "类型错误"}')
    with pytest.raises(ModelOutputError, match="note 必须是非空字符串"):
        parse_system1_decision('{"route": "rethink", "note": ""}')
    with pytest.raises(ModelOutputError, match="不是合法 JSON"):
        parse_system1_decision("随便一句话")
    with pytest.raises(ModelOutputError, match="输出为空"):
        parse_system1_decision("   ")


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
        System1Decision(route=System1Route.RETHINK, note="")
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
