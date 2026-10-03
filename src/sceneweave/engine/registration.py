"""登记前对行动时间安排的纯校验。

校验只判断结构、引用与时间约束，不实现翻译模型、调度队列或完整
行动状态机；失败抛出异常，不拆组、不部分接受。需要角色改变行为
安排时，通过打回接口让角色重新决定。当前只覆盖本次新行动：维持
声明无需重排，修改声明的未来时间安排待行动生命周期实现时补充。
"""

from sceneweave.domain.identity import ActionId
from sceneweave.domain.messages import NewAction, RoleResponse
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.translation import ActionSchedule, TranslationResult


class RegistrationValidationError(ValueError):
    """时间安排未通过登记前校验；不得登记部分行动。"""


def validate_registration(
    response: RoleResponse,
    translation: TranslationResult,
    *,
    current_time: SimulationTime,
) -> None:
    """校验翻译结果与响应一致、时间安排满足登记约束。

    覆盖引用一致性、提交组一致性、新行动完整性、开始时刻晚于当前
    时刻、结束晚于开始、过程节点合法，以及组内共同非零时间交集
    （max(组内开始) < min(组内结束)，仅端点相接不满足）。任何问题
    都导致整体失败：调用者不得拆组或登记部分行动。
    """
    problems = registration_problems(response, translation, current_time=current_time)
    if problems:
        raise RegistrationValidationError("；".join(problems))


def registration_problems(
    response: RoleResponse,
    translation: TranslationResult,
    *,
    current_time: SimulationTime,
) -> tuple[str, ...]:
    """逐条返回登记前校验问题；空元组表示可以通过登记。"""
    if not isinstance(response, RoleResponse):
        raise ValueError("登记校验的响应必须是 RoleResponse")
    if not isinstance(translation, TranslationResult):
        raise ValueError("登记校验的翻译结果必须是 TranslationResult")
    validate_simulation_time(current_time, "当前剧情时刻")
    problems: list[str] = []
    if (
        translation.cycle_id,
        translation.role_id,
        translation.response_id,
    ) != (
        response.cycle_id,
        response.role_id,
        response.response_id,
    ):
        problems.append("翻译结果必须关联同一响应")
    new_actions: dict[ActionId, NewAction] = {
        item.action_id: item for item in response.actions if isinstance(item, NewAction)
    }
    schedules: dict[ActionId, ActionSchedule] = {
        schedule.action_id: schedule for schedule in translation.schedules
    }
    for action_id in sorted(set(new_actions) - set(schedules)):
        problems.append(f"新行动缺少时间安排：{action_id}")
    for action_id in sorted(set(schedules) - set(new_actions)):
        problems.append(f"时间安排引用了非本次新行动：{action_id}")
    for action_id in sorted(set(schedules) & set(new_actions)):
        schedule = schedules[action_id]
        if schedule.commit_group_id != new_actions[action_id].commit_group_id:
            problems.append(f"时间安排的提交组与行动不一致：{action_id}")
        previous = schedule.start
        ordered = True
        for checkpoint in schedule.checkpoints:
            if checkpoint <= previous or checkpoint >= schedule.expected_end:
                ordered = False
            previous = checkpoint
        if not ordered:
            problems.append(f"过程节点必须严格递增且在开始与结束之间：{action_id}")
        if schedule.start <= current_time:
            problems.append(f"新行动开始时刻必须晚于当前时刻：{action_id}")
    starts = [schedule.start for schedule in schedules.values()]
    ends = [schedule.expected_end for schedule in schedules.values()]
    if starts and max(starts) >= min(ends):
        problems.append("并行组内行动必须有共同非零时间交集：max(开始) < min(结束)")
    return tuple(problems)
