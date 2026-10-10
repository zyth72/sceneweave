"""登记前对行动时间安排的纯校验。

校验只判断结构、引用、依赖与时间约束，不实现翻译模型、调度队列或完整
行动状态机；失败逐条报告，不拆组、不部分接受。问题分两类：结构不一致
（引用错误、缺安排、依赖未定义/循环）属于运行错误，停止推进并保存现场；
时间安排不合业务约束（开始晚于当前时刻、历史不可改写）走打回接口让角色
重新决定。

协议纠正后，结构化 ``StepAction`` 使用平坦的逐动作时间安排：每个动作
只有自己的 start/expected_end/checkpoints 约束，不再要求整组共同非零
时间交集；可选依赖由引擎在运行时按 World 真实完成门控。旧的三类声明
（新增/修改）仍按其原约束校验，仅服务低层历史测试。
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sceneweave.domain.identity import ActionId
from sceneweave.domain.messages import (
    ModifiedAction,
    NewAction,
    RoleResponse,
    StepAction,
)
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
    """逐条返回登记前问题；空元组表示可以通过登记。"""
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
    step_actions: dict[ActionId, StepAction] = {
        item.action_id: item for item in response.actions if isinstance(item, StepAction)
    }
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
    step_new = {action_id for action_id in step_actions if action_id not in known}
    step_existing = set(step_actions) - step_new
    # ---- 结构化 StepAction：平坦、逐动作时间安排与依赖 ----
    for action_id in sorted(step_new - set(schedules)):
        structural(f"新动作缺少时间安排：{action_id}")
    for action_id in sorted(set(schedules) - step_new - set(new_actions)):
        structural(f"时间安排引用了非本次新行动：{action_id}")
    for action_id in sorted(step_new & set(schedules)):
        schedule = schedules[action_id]
        if schedule.start <= current_time:
            timing(f"新动作开始时刻必须晚于当前时刻：{action_id}")
    for action_id in sorted(step_existing & set(schedules)):
        structural(f"既有行动不得借新动作时间安排重排：{action_id}")
    _check_step_dependencies(step_actions, known, structural)
    # ---- 旧三类声明：保留原有约束（仅低层历史测试路径） ----
    for action_id in sorted(set(new_actions) - set(schedules)):
        structural(f"新行动缺少时间安排：{action_id}")
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
    legacy_starts = [
        schedules[action_id].start for action_id in new_actions if action_id in schedules
    ]
    legacy_ends = [
        schedules[action_id].expected_end for action_id in new_actions if action_id in schedules
    ]
    if legacy_starts and max(legacy_starts) >= min(legacy_ends):
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


def _check_step_dependencies(
    step_actions: Mapping[ActionId, StepAction],
    known: Mapping[ActionId, ActionArrangement],
    structural: Callable[[str], None],
) -> None:
    """校验结构化依赖：引用必须已知；本声明内不得成环。"""
    declared = set(step_actions)
    edges: dict[ActionId, list[ActionId]] = {}
    for action_id, step in step_actions.items():
        for dep in step.depends_on:
            if dep not in known and dep not in declared:
                structural(f"依赖引用了未定义的动作：{dep}")
                continue
            if dep == action_id:
                structural(f"动作不能依赖自身：{action_id}")
                continue
            if dep in declared:
                edges.setdefault(action_id, []).append(dep)
    _reject_dependency_cycles(declared, edges, structural)


def _reject_dependency_cycles(
    nodes: set[ActionId],
    edges: Mapping[ActionId, list[ActionId]],
    structural: Callable[[str], None],
) -> None:
    state: dict[ActionId, int] = {node: 0 for node in nodes}

    def visit(node: ActionId, path: list[ActionId]) -> None:
        state[node] = 1
        for nxt in edges.get(node, []):
            if state.get(nxt) == 1:
                structural(f"依赖存在循环：{'→'.join(str(x) for x in path + [nxt])}")
                return
            if state.get(nxt) == 0:
                visit(nxt, path + [nxt])
        state[node] = 2

    for node in sorted(nodes, key=str):
        if state[node] == 0:
            visit(node, [node])


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
