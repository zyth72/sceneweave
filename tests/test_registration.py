"""登记前时间安排纯校验：结构、引用、提交组、共同非零交集与修改约束。"""

from typing import Any

import pytest

from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    ResponseId,
    RoleId,
)
from sceneweave.domain.messages import (
    ActionPhase,
    MaintainAction,
    ModifiedAction,
    NewAction,
    RoleResponse,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import (
    ActionArrangement,
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationResult,
)
from sceneweave.engine.registration import (
    RegistrationValidationError,
    registration_problems,
    validate_registration,
)


def response(actions: tuple[Any, ...]) -> RoleResponse:
    return RoleResponse(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), actions)


def schedule(
    action_id: str,
    start: int,
    end: int,
    group: str = "g-1",
    checkpoints: tuple[int, ...] = (),
) -> ActionSchedule:
    return ActionSchedule(
        ActionId(action_id),
        CommitGroupId(group),
        SimulationTime(start),
        SimulationTime(end),
        tuple(SimulationTime(item) for item in checkpoints),
    )


def modified_schedule(
    action_id: str,
    start: int,
    end: int,
    checkpoints: tuple[int, ...] = (),
) -> ModifiedActionSchedule:
    return ModifiedActionSchedule(
        ActionId(action_id),
        SimulationTime(start),
        SimulationTime(end),
        tuple(SimulationTime(item) for item in checkpoints),
    )


def arrangement(
    action_id: str,
    start: int,
    end: int,
    checkpoints: tuple[int, ...] = (),
    *,
    phase: ActionPhase = ActionPhase.STARTED,
) -> ActionArrangement:
    return ActionArrangement(
        action_id=ActionId(action_id),
        phase=phase,
        description="既有行动",
        start=SimulationTime(start),
        expected_end=SimulationTime(end),
        checkpoints=tuple(SimulationTime(item) for item in checkpoints),
    )


def translation(
    schedules: tuple[ActionSchedule, ...] = (),
    modified_schedules: tuple[ModifiedActionSchedule, ...] = (),
) -> TranslationResult:
    return TranslationResult(
        CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), schedules, modified_schedules
    )


def parallel_response() -> RoleResponse:
    return response(
        (
            NewAction(ActionId("a-1"), "伸手拉开抽屉", CommitGroupId("g-1")),
            NewAction(ActionId("a-2"), "同时说话", CommitGroupId("g-1")),
        )
    )


def valid_schedules() -> tuple[ActionSchedule, ...]:
    return (
        schedule("a-1", 2_000, 5_000, checkpoints=(3_000, 4_000)),
        schedule("a-2", 3_000, 6_000),
    )


def test_valid_parallel_group_passes() -> None:
    assert (
        registration_problems(
            parallel_response(), translation(valid_schedules()), current_time=SimulationTime(1_000)
        )
        == ()
    )
    validate_registration(
        parallel_response(),
        translation(valid_schedules()),
        current_time=SimulationTime(1_000),
    )


def test_overlapping_group_passes_at_boundary_overlap() -> None:
    problems = registration_problems(
        parallel_response(),
        translation((schedule("a-1", 2_000, 5_000), schedule("a-2", 4_999, 6_000))),
        current_time=SimulationTime(1_000),
    )
    assert problems == ()


@pytest.mark.parametrize("start", [1_000, 999])
def test_start_must_be_strictly_after_current_time(start: int) -> None:
    single = response((NewAction(ActionId("a-1"), "伸手", CommitGroupId("g-1")),))
    schedules = (schedule("a-1", start, 5_000),)
    problems = registration_problems(
        single, translation(schedules), current_time=SimulationTime(1_000)
    )
    assert any("晚于当前时刻" in problem for problem in problems)


@pytest.mark.parametrize(
    ("first", "second"),
    [((2_000, 3_000), (3_000, 6_000)), ((2_000, 3_000), (4_000, 5_000))],
)
def test_group_without_common_nonzero_overlap_fails(
    first: tuple[int, int], second: tuple[int, int]
) -> None:
    schedules = (
        schedule("a-1", *first),
        schedule("a-2", *second),
    )
    problems = registration_problems(
        parallel_response(),
        translation(schedules),
        current_time=SimulationTime(1_000),
    )
    assert any("共同非零时间交集" in problem for problem in problems)


def test_missing_or_extra_schedules_are_rejected() -> None:
    only_first = (schedule("a-1", 2_000, 5_000),)
    problems = registration_problems(
        parallel_response(),
        translation(only_first),
        current_time=SimulationTime(1_000),
    )
    assert any("缺少时间安排" in problem for problem in problems)

    extra = valid_schedules() + (schedule("a-9", 3_000, 4_000),)
    problems = registration_problems(
        parallel_response(),
        translation(extra),
        current_time=SimulationTime(1_000),
    )
    assert any("非本次新行动" in problem for problem in problems)


def test_commit_group_must_match_the_action_group() -> None:
    schedules = (schedule("a-1", 2_000, 5_000, group="g-2"),)
    problems = registration_problems(
        response((NewAction(ActionId("a-1"), "伸手", CommitGroupId("g-1")),)),
        TranslationResult(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), schedules),
        current_time=SimulationTime(1_000),
    )
    assert any("提交组与行动不一致" in problem for problem in problems)


def test_translation_must_reference_the_same_response() -> None:
    other = TranslationResult(
        CycleId("cycle-1"), RoleId("alice"), ResponseId("r-other"), valid_schedules()
    )
    problems = registration_problems(parallel_response(), other, current_time=SimulationTime(1_000))
    assert any("关联同一响应" in problem for problem in problems)


def test_maintained_actions_do_not_need_schedules() -> None:
    maintained = response((MaintainAction(ActionId("a-0")),))
    assert (
        registration_problems(maintained, translation(()), current_time=SimulationTime(1_000)) == ()
    )


def test_failure_reports_all_problems_without_splitting_the_group() -> None:
    schedules = (schedule("a-1", 2_000, 5_000, group="g-2"),)
    problems = registration_problems(
        parallel_response(),
        translation(schedules),
        current_time=SimulationTime(1_000),
    )
    assert any("缺少时间安排" in problem for problem in problems)
    assert any("提交组与行动不一致" in problem for problem in problems)
    with pytest.raises(RegistrationValidationError) as info:
        validate_registration(
            parallel_response(),
            translation(schedules),
            current_time=SimulationTime(1_000),
        )
    assert "缺少时间安排" in str(info.value)
    assert "提交组与行动不一致" in str(info.value)


def test_bool_current_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="非负整数毫秒"):
        registration_problems(
            parallel_response(),
            translation(valid_schedules()),
            current_time=True,  # type: ignore[arg-type]
        )


def test_valid_modification_passes_and_extends_future_only() -> None:
    """已开始行动的修改：保留历史节点，只调整未来部分。"""
    modified = (ModifiedAction(ActionId("a-0"), "跑得更久"),)
    known = (arrangement("a-0", 1000, 5000, (2000, 3000)),)
    new_arrangement = modified_schedule("a-0", 1000, 9000, (2000, 3000, 6000, 8000))
    problems = registration_problems(
        response(modified),
        translation(modified_schedules=(new_arrangement,)),
        current_time=SimulationTime(4000),
        arrangements=known,
    )
    assert problems == ()


def test_modification_cannot_rewrite_history() -> None:
    modified = (ModifiedAction(ActionId("a-0"), "重写历史"),)
    known = (arrangement("a-0", 1000, 5000, (2000,)),)
    rewritten = modified_schedule("a-0", 1000, 9000, (1500,))
    problems = registration_problems(
        response(modified),
        translation(modified_schedules=(rewritten,)),
        current_time=SimulationTime(4000),
        arrangements=known,
    )
    assert any("历史过程节点不可改写" in problem for problem in problems)


def test_started_action_start_cannot_be_changed() -> None:
    modified = (ModifiedAction(ActionId("a-0"), "挪动开始"),)
    known = (arrangement("a-0", 1000, 5000),)
    moved = modified_schedule("a-0", 1200, 9000)
    problems = registration_problems(
        response(modified),
        translation(modified_schedules=(moved,)),
        current_time=SimulationTime(4000),
        arrangements=known,
    )
    assert any("开始时刻不可改写" in problem for problem in problems)


def test_pending_action_modification_can_move_start_after_now() -> None:
    """未开始行动的开始属于未来节点，可以改到更晚的当前时刻之后。"""
    modified = (ModifiedAction(ActionId("a-0"), "延后开始"),)
    known = (arrangement("a-0", 5000, 9000, phase=ActionPhase.REGISTERED),)
    moved = modified_schedule("a-0", 6000, 10_000)
    assert (
        registration_problems(
            response(modified),
            translation(modified_schedules=(moved,)),
            current_time=SimulationTime(4000),
            arrangements=known,
        )
        == ()
    )
    too_early = modified_schedule("a-0", 4000, 10_000)
    problems = registration_problems(
        response(modified),
        translation(modified_schedules=(too_early,)),
        current_time=SimulationTime(4000),
        arrangements=known,
    )
    assert any("修改后的开始时刻必须晚于当前时刻" in problem for problem in problems)


def test_modification_end_must_stay_in_future() -> None:
    modified = (ModifiedAction(ActionId("a-0"), "立即结束"),)
    known = (arrangement("a-0", 1000, 5000),)
    ending = modified_schedule("a-0", 1000, 4000)
    problems = registration_problems(
        response(modified),
        translation(modified_schedules=(ending,)),
        current_time=SimulationTime(4000),
        arrangements=known,
    )
    assert any("修改后的预计结束必须晚于当前时刻" in problem for problem in problems)


def test_modification_schedule_must_match_declarations() -> None:
    modified = (ModifiedAction(ActionId("a-0"), "修改"),)
    known = (arrangement("a-0", 1000, 5000),)
    missing = registration_problems(
        response(modified),
        translation(),
        current_time=SimulationTime(2000),
        arrangements=known,
    )
    assert any("修改声明缺少未来时间安排" in problem for problem in missing)
    extra = registration_problems(
        response((MaintainAction(ActionId("a-0")),)),
        translation(modified_schedules=(modified_schedule("a-0", 1000, 6000),)),
        current_time=SimulationTime(2000),
        arrangements=known,
    )
    assert any("非本次修改声明" in problem for problem in extra)
    unknown = registration_problems(
        response(modified),
        translation(modified_schedules=(modified_schedule("a-0", 1000, 6000),)),
        current_time=SimulationTime(2000),
    )
    assert any("修改声明缺少既有行动安排" in problem for problem in unknown)


def test_modified_actions_do_not_join_group_overlap_constraint() -> None:
    """修改不参与新行动的并行组交集约束。"""
    actions = (
        ModifiedAction(ActionId("a-0"), "修改"),
        NewAction(ActionId("a-1"), "新行动", CommitGroupId("g-1")),
    )
    known = (arrangement("a-0", 1000, 300_000),)
    schedules = (schedule("a-1", 2000, 3000),)
    modified_schedules = (modified_schedule("a-0", 1000, 200_000),)
    assert (
        registration_problems(
            response(actions),
            translation(schedules=schedules, modified_schedules=modified_schedules),
            current_time=SimulationTime(1500),
            arrangements=known,
        )
        == ()
    )
