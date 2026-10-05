"""模型输出严格解析的行为测试：类型/字段/引用校验，错误不转空。"""

import pytest

from sceneweave.roles.model_output import (
    ModelOutputError,
    ModificationRef,
    System1Decision,
    System2Intention,
    System3Plan,
    parse_system1_decision,
    parse_system2_intention,
    parse_system3_plan,
)


def test_system1_valid_decision_parses() -> None:
    decision = parse_system1_decision('{"reconsider": true, "note": "门开了，需要警惕"}')
    assert decision == System1Decision(reconsider=True, note="门开了，需要警惕")


def test_system1_missing_fields_and_bad_types_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system1_decision('{"note": "缺少判断"}')
    with pytest.raises(ModelOutputError, match="reconsider 必须是布尔值"):
        parse_system1_decision('{"reconsider": 1, "note": "bool 乱作整数"}')
    with pytest.raises(ModelOutputError, match="note 必须是非空字符串"):
        parse_system1_decision('{"reconsider": true, "note": ""}')
    with pytest.raises(ModelOutputError, match="不是合法 JSON"):
        parse_system1_decision("随便一句话")
    with pytest.raises(ModelOutputError, match="输出为空"):
        parse_system1_decision("   ")


def test_system2_valid_intention_parses() -> None:
    intention = parse_system2_intention('{"intention": "拿到抽屉里的钥匙", "abandon": ["a-lin-1"]}')
    assert intention.abandon == ("a-lin-1",)
    assert intention.intention == "拿到抽屉里的钥匙"


def test_system2_bad_abandon_items_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="每项必须是非空字符串身份"):
        parse_system2_intention('{"intention": "x", "abandon": [17]}')
    with pytest.raises(ModelOutputError, match="abandon 必须是数组"):
        parse_system2_intention('{"intention": "x", "abandon": "a-1"}')


def test_system3_full_declaration_parses() -> None:
    plan = parse_system3_plan(
        '{"feasible": true, "maintain": ["a-1"],'
        ' "modify": [{"id": "a-2", "change": "放慢速度"}],'
        ' "new_actions": ["伸手拉开抽屉", "同时说话"]}'
    )
    assert plan.maintain == ("a-1",)
    assert plan.modify == (ModificationRef("a-2", "放慢速度"),)
    assert plan.new_descriptions == ("伸手拉开抽屉", "同时说话")
    assert plan.feasible is True and plan.retry is False


def test_system3_explicit_empty_is_valid_not_omitted() -> None:
    plan = parse_system3_plan('{"feasible": true, "maintain": [], "modify": [], "new_actions": []}')
    assert plan.maintain == () and plan.modify == () and plan.new_descriptions == ()


def test_system3_missing_collection_keys_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system3_plan('{"feasible": true}')
    with pytest.raises(ModelOutputError, match="缺少必填字段"):
        parse_system3_plan('{"feasible": true, "maintain": [], "modify": []}')


def test_system3_conflicting_references_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="不能同时维持与修改"):
        parse_system3_plan(
            '{"feasible": true, "maintain": ["a-1"],'
            ' "modify": [{"id": "a-1", "change": "x"}], "new_actions": []}'
        )
    with pytest.raises(ModelOutputError, match="maintain 行动身份必须唯一"):
        parse_system3_plan(
            '{"feasible": true, "maintain": ["a-1", "a-1"], "modify": [], "new_actions": []}'
        )


def test_system3_infeasible_requires_reason_and_no_declarations() -> None:
    with pytest.raises(ModelOutputError, match="不可行时必须说明原因"):
        parse_system3_plan('{"feasible": false, "maintain": [], "modify": [], "new_actions": []}')
    with pytest.raises(ModelOutputError, match="不可行时不得携带主动集合"):
        parse_system3_plan(
            '{"feasible": false, "reason": "够不到", "maintain": ["a-1"],'
            ' "modify": [], "new_actions": []}'
        )


def test_system3_retry_must_not_carry_declarations() -> None:
    with pytest.raises(ModelOutputError, match="内部重做时不得携带主动集合"):
        parse_system3_plan(
            '{"feasible": true, "retry": true, "maintain": ["a-1"],'
            ' "modify": [], "new_actions": []}'
        )
    plan = parse_system3_plan(
        '{"feasible": true, "retry": true, "reason": "换个姿势",'
        ' "maintain": [], "modify": [], "new_actions": []}'
    )
    assert plan.retry is True


def test_system3_bad_item_types_are_errors() -> None:
    with pytest.raises(ModelOutputError, match="modify 每项必须是对象"):
        parse_system3_plan(
            '{"feasible": true, "maintain": [], "modify": ["a-1"], "new_actions": []}'
        )
    with pytest.raises(ModelOutputError, match="new_actions 每项必须是非空字符串描述"):
        parse_system3_plan('{"feasible": true, "maintain": [], "modify": [], "new_actions": [3]}')
    with pytest.raises(ModelOutputError, match="feasible 必须是布尔值"):
        parse_system3_plan('{"feasible": "yes", "maintain": [], "modify": [], "new_actions": []}')


def test_top_level_must_be_object() -> None:
    with pytest.raises(ModelOutputError, match="必须是 JSON 对象"):
        parse_system2_intention('["intention"]')


def test_frozen_results_reject_construction_errors() -> None:
    with pytest.raises(ModelOutputError):
        System1Decision(reconsider=True, note="")
    with pytest.raises(ModelOutputError):
        System2Intention(intention="", abandon=())
    with pytest.raises(ModelOutputError):
        System3Plan(
            feasible=True,
            reason="",
            retry=False,
            maintain=("a-1", "a-1"),
            modify=(),
            new_descriptions=(),
        )


def test_duplicate_keys_are_rejected() -> None:
    """重复键悄悄覆盖会改变声明语义，直接拒绝。"""
    with pytest.raises(ModelOutputError, match="重复键"):
        parse_system3_plan(
            '{"feasible": true, "maintain": [], "modify": [],'
            ' "new_actions": ["走路"], "new_actions": []}'
        )
    with pytest.raises(ModelOutputError, match="重复键"):
        parse_system1_decision('{"reconsider": true, "note": "a", "note": "b"}')


def test_non_finite_constants_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="非有限常量"):
        parse_system2_intention('{"intention": "x", "abandon": [], "extra": NaN}')


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system2_intention('{"intention": "x", "abandon": [], "extra": 1}')
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system3_plan(
            '{"feasible": true, "maintain": [], "modify": [], "new_actions": [], "surprise": true}'
        )


def test_retry_infeasible_contradiction_is_rejected() -> None:
    with pytest.raises(ModelOutputError, match="不能同时成立"):
        parse_system3_plan(
            '{"feasible": false, "reason": "矛盾", "retry": true,'
            ' "maintain": [], "modify": [], "new_actions": []}'
        )
    with pytest.raises(ModelOutputError, match="内部重做时必须说明原因"):
        parse_system3_plan(
            '{"feasible": true, "retry": true, "maintain": [], "modify": [], "new_actions": []}'
        )


def test_dataclass_construction_matches_parse_rules() -> None:
    """直构路径与解析路径规则一致：bool 身份、矛盾标记同样拒绝。"""
    with pytest.raises(ModelOutputError, match="不能同时成立"):
        System3Plan(
            feasible=False,
            reason="矛盾",
            retry=True,
            maintain=(),
            modify=(),
            new_descriptions=(),
        )
    with pytest.raises(ModelOutputError, match="内部重做时必须说明原因"):
        System3Plan(
            feasible=True,
            reason="",
            retry=True,
            maintain=(),
            modify=(),
            new_descriptions=(),
        )
    with pytest.raises(ModelOutputError, match="必须是非空字符串"):
        ModificationRef(action_id=True, change="x")  # type: ignore[arg-type]


def test_nested_modification_unknown_fields_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system3_plan(
            '{"feasible":true,"maintain":[],"modify":'
            '[{"id":"a-1","change":"放慢","future_actions":["拿钥匙"]}],"new_actions":[]}'
        )


@pytest.mark.parametrize("field,invalid", [("maintain", (True,)), ("modify", ("a-1",))])
def test_direct_plan_construction_rejects_invalid_references(field, invalid) -> None:
    fields = dict(
        feasible=True, reason="", retry=False, maintain=(), modify=(), new_descriptions=()
    )
    fields[field] = invalid
    with pytest.raises(ModelOutputError):
        System3Plan(**fields)
