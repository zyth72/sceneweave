"""登记前时间安排纯校验：结构、引用、提交组与共同非零时间交集。"""

from typing import Any

import pytest

from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    ResponseId,
    RoleId,
)
from sceneweave.domain.messages import MaintainAction, NewAction, RoleResponse
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionSchedule, TranslationResult
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


def translation(schedules: tuple[ActionSchedule, ...]) -> TranslationResult:
    return TranslationResult(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), schedules)


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
