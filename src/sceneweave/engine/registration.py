"""登记前对行动时间安排的纯校验。

校验只判断结构、引用与时间约束，不实现翻译模型、调度队列或完整
行动状态机；失败逐条报告，不拆组、不部分接受。问题分两类：结构
不一致（引用错误、缺安排、提交组关联错误）属于运行错误，停止推进
并保存现场；时间安排不合业务约束（开始晚于当前时刻、组内共同非零
交集、历史不可改写）走打回接口让角色重新决定。新行动与修改声明的
约束不同：新行动构成并行组并须满足共同非零时间交集；修改只能调整
未来节点，历史开始与已过过程节点不可改写。
"""

from collections.abc import Mapping
from dataclasses import dataclass

from sceneweave.domain.identity import ActionId
from sceneweave.domain.messages import ModifiedAction, NewAction, RoleResponse
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.translation import (
    ActionArrangement,
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationResult,
)


class RegistrationValidationError(ValueError):
    """时间安排未通过登记前校验；不得登记部分行动。"""


@dataclass(frozen=True, slots=True)
class RegistrationIssue:
    """一条登记前问题；structural 为真表示结构不一致（运行错误）。"""

    text: str
    structural: bool

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("问题文本必须是非空字符串")
        if type(self.structural) is not bool:
            raise ValueError("结构标记必须是布尔值")


def validate_registration(
    response: RoleResponse,
    translation: TranslationResult,
    *,
    current_time: SimulationTime,
    arrangements: Mapping[ActionId, ActionArrangement] | tuple[ActionArrangement, ...] = (),
) -> None:
    """校验翻译结果与响应一致、时间安排满足登记约束；失败整体拒绝。"""
    problems = registration_problems(
        response, translation, current_time=current_time, arrangements=arrangements
    )
    if problems:
        raise RegistrationValidationError("；".join(problems))


def registration_issues(
    response: RoleResponse,
    translation: TranslationResult,
    *,
    current_time: SimulationTime,
    arrangements: Mapping[ActionId, ActionArrangement] | tuple[ActionArrangement, ...] = (),
) -> tuple[RegistrationIssue, ...]:
    """逐条返回登记前问题；空元组表示可以通过登记。

    新行动覆盖：引用一致性、提交组一致性、新行动完整性、开始时刻
    晚于当前时刻、结束晚于开始、过程节点合法，以及组内共同非零时间
    交集（max(组内开始) < min(组内结束)，仅端点相接不满足）。修改
    行动覆盖：与既有安排一一对应、历史不可改写、修改后的结束晚于
    当前时刻。修改不参与新组交集约束。
    """
    if not isinstance(response, RoleResponse):
        raise ValueError("登记校验的响应必须是 RoleResponse")
    if not isinstance(translation, TranslationResult):
        raise ValueError("登记校验的翻译结果必须是 TranslationResult")
    validate_simulation_time(current_time, "当前剧情时刻")
    known: dict[ActionId, ActionArrangement] = (
        dict(arrangements)
        if isinstance(arrangements, Mapping)
        else {item.action_id: item for item in arrangements}
    )
    issues: list[RegistrationIssue] = []

    def structural(text: str) -> None:
        issues.append(RegistrationIssue(text=text, structural=True))

    def timing(text: str) -> None:
        issues.append(RegistrationIssue(text=text, structural=False))

    if (
        translation.cycle_id,
        translation.role_id,
        translation.response_id,
    ) != (
        response.cycle_id,
        response.role_id,
        response.response_id,
    ):
        structural("翻译结果必须关联同一响应")
    new_actions: dict[ActionId, NewAction] = {
        item.action_id: item for item in response.actions if isinstance(item, NewAction)
    }
    modified_actions: dict[ActionId, ModifiedAction] = {
        item.action_id: item for item in response.actions if isinstance(item, ModifiedAction)
    }
    schedules: dict[ActionId, ActionSchedule] = {
        schedule.action_id: schedule for schedule in translation.schedules
    }
    modified_schedules: dict[ActionId, ModifiedActionSchedule] = {
        schedule.action_id: schedule for schedule in translation.modified_schedules
    }
    for action_id in sorted(set(new_actions) - set(schedules)):
        structural(f"新行动缺少时间安排：{action_id}")
    for action_id in sorted(set(schedules) - set(new_actions)):
        structural(f"时间安排引用了非本次新行动：{action_id}")
    for action_id in sorted(set(schedules) & set(new_actions)):
        schedule = schedules[action_id]
        if schedule.commit_group_id != new_actions[action_id].commit_group_id:
            structural(f"时间安排的提交组与行动不一致：{action_id}")
            continue
        previous = schedule.start
        ordered = True
        for checkpoint in schedule.checkpoints:
            if checkpoint <= previous or checkpoint >= schedule.expected_end:
                ordered = False
            previous = checkpoint
        if not ordered:
            structural(f"过程节点必须严格递增且在开始与结束之间：{action_id}")
            continue
        if schedule.start <= current_time:
            timing(f"新行动开始时刻必须晚于当前时刻：{action_id}")
    starts = [schedule.start for schedule in schedules.values()]
    ends = [schedule.expected_end for schedule in schedules.values()]
    if starts and max(starts) >= min(ends):
        timing("并行组内行动必须有共同非零时间交集：max(开始) < min(结束)")
    for action_id in sorted(set(modified_actions) - set(modified_schedules)):
        structural(f"修改声明缺少未来时间安排：{action_id}")
    for action_id in sorted(set(modified_schedules) - set(modified_actions)):
        structural(f"修改安排引用了非本次修改声明的行动：{action_id}")
    for action_id in sorted(set(modified_schedules) & set(modified_actions)):
        modified_schedule = modified_schedules[action_id]
        arrangement = known.get(action_id)
        if arrangement is None:
            structural(f"修改声明缺少既有行动安排：{action_id}")
            continue
        if arrangement.start <= current_time:
            if modified_schedule.start != arrangement.start:
                timing(f"已开始行动的开始时刻不可改写：{action_id}")
        elif modified_schedule.start <= current_time:
            timing(f"修改后的开始时刻必须晚于当前时刻：{action_id}")
        if _history_checkpoints(
            modified_schedule.checkpoints, current_time
        ) != _history_checkpoints(arrangement.checkpoints, current_time):
            timing(f"历史过程节点不可改写：{action_id}")
        if modified_schedule.expected_end <= current_time:
            timing(f"修改后的预计结束必须晚于当前时刻：{action_id}")
    return tuple(issues)


def registration_problems(
    response: RoleResponse,
    translation: TranslationResult,
    *,
    current_time: SimulationTime,
    arrangements: Mapping[ActionId, ActionArrangement] | tuple[ActionArrangement, ...] = (),
) -> tuple[str, ...]:
    """全部登记前问题的文本；保持既有调用方可读。"""
    return tuple(
        issue.text
        for issue in registration_issues(
            response, translation, current_time=current_time, arrangements=arrangements
        )
    )


def _history_checkpoints(
    checkpoints: tuple[SimulationTime, ...], current_time: SimulationTime
) -> list[SimulationTime]:
    return [checkpoint for checkpoint in checkpoints if checkpoint <= current_time]
