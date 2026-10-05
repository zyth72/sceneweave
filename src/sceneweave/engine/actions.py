"""引擎的行动登记记录与登记计划。

引擎保存行动身份、登记阶段和待处理节点；World 维护行动实际进展与
作用，两者通过结算结果同步。登记计划先完整准备并校验再提交：失败
不得留下部分新行动已登记、旧行动已取消或世界已局部改变的混合状态。
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from sceneweave.domain.adjudication import ActionOutcome
from sceneweave.domain.identity import ActionId, CommitGroupId, CycleId, RoleId
from sceneweave.domain.messages import ActionPhase, ModifiedAction, NewAction, RoleResponse
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.translation import (
    ActionArrangement,
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationResult,
)
from sceneweave.engine.registration import registration_issues

# 实际已结束的终态：不能再被维持、修改或重新登记；新行动不得复用终态身份。
TERMINAL_PHASES = frozenset(
    {
        ActionPhase.COMPLETED,
        ActionPhase.FAILED,
        ActionPhase.WITHDRAWN,
    }
)
# 仍可被主动维持或修改的阶段。UNCONFIRMED 表示结果未知：不是已知客观
# 终止，后续节点与主动维持不得仅凭未知结果删除或放弃。
MAINTAINABLE_PHASES = frozenset(
    {ActionPhase.REGISTERED, ActionPhase.STARTED, ActionPhase.UNCONFIRMED}
)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """一个已登记行动的引擎记录：身份、归属、阶段与时间安排。"""

    action_id: ActionId
    role_id: RoleId
    description: str
    phase: ActionPhase
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...]
    registered_cycle: CycleId
    commit_group_id: CommitGroupId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("行动身份必须是非空字符串")
        if not isinstance(self.role_id, str) or not self.role_id:
            raise ValueError("角色身份必须是非空字符串")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("行动描述必须是非空字符串")
        if not isinstance(self.phase, ActionPhase):
            raise ValueError("行动阶段必须是 ActionPhase")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        if self.expected_end <= self.start:
            raise ValueError("预计结束时刻必须晚于开始时刻")
        if not isinstance(self.checkpoints, tuple):
            raise ValueError("过程节点必须是毫秒时间元组")
        previous = self.start
        for checkpoint in self.checkpoints:
            validate_simulation_time(checkpoint, "过程节点")
            if checkpoint <= previous or checkpoint >= self.expected_end:
                raise ValueError("过程节点必须严格递增且在开始与结束之间")
            previous = checkpoint
        if not isinstance(self.registered_cycle, str) or not self.registered_cycle:
            raise ValueError("登记周期必须是非空字符串")
        if self.commit_group_id is not None and (
            not isinstance(self.commit_group_id, str) or not self.commit_group_id
        ):
            raise ValueError("提交组身份必须是非空字符串或 None")

    def arrangement(self) -> ActionArrangement:
        return ActionArrangement(
            action_id=self.action_id,
            phase=self.phase,
            description=self.description,
            start=self.start,
            expected_end=self.expected_end,
            checkpoints=self.checkpoints,
        )


@dataclass(frozen=True, slots=True)
class FinalizedDeclaration:
    """一个角色通过屏障与翻译阶段后的最终声明及其时间安排。"""

    response: RoleResponse
    translation: TranslationResult


@dataclass(frozen=True, slots=True)
class RegistrationPlan:
    """一次统一登记的完整可提交计划；problems 非空表示不得提交。"""

    cycle_id: CycleId
    simulation_time: SimulationTime
    maintained: tuple[ActionRecord, ...] = ()
    registered: tuple[ActionRecord, ...] = ()
    modified: tuple[ActionRecord, ...] = ()
    withdrawn_pending: tuple[ActionRecord, ...] = ()
    withdrawn_active: tuple[ActionRecord, ...] = ()
    problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("周期身份必须是非空字符串")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        for name in (
            "maintained",
            "registered",
            "modified",
            "withdrawn_pending",
            "withdrawn_active",
        ):
            records = getattr(self, name)
            if not isinstance(records, tuple) or not all(
                isinstance(item, ActionRecord) for item in records
            ):
                raise ValueError(f"{name} 必须是 ActionRecord 元组")
        if not isinstance(self.problems, tuple) or not all(
            isinstance(item, str) and item for item in self.problems
        ):
            raise ValueError("登记问题必须是非空字符串元组")

    @property
    def ok(self) -> bool:
        return not self.problems


def _new_record(
    action: NewAction, schedule: ActionSchedule, cycle_id: CycleId, role_id: RoleId
) -> ActionRecord:
    return ActionRecord(
        action_id=action.action_id,
        role_id=role_id,
        description=action.description,
        phase=ActionPhase.REGISTERED,
        start=schedule.start,
        expected_end=schedule.expected_end,
        checkpoints=schedule.checkpoints,
        registered_cycle=cycle_id,
        commit_group_id=action.commit_group_id,
    )


def _modified_record(record: ActionRecord, schedule: ModifiedActionSchedule) -> ActionRecord:
    return ActionRecord(
        action_id=record.action_id,
        role_id=record.role_id,
        description=record.description,
        phase=record.phase,
        start=schedule.start,
        expected_end=schedule.expected_end,
        checkpoints=schedule.checkpoints,
        registered_cycle=record.registered_cycle,
        commit_group_id=record.commit_group_id,
    )


def _withdrawn_record(record: ActionRecord) -> ActionRecord:
    return ActionRecord(
        action_id=record.action_id,
        role_id=record.role_id,
        description=record.description,
        phase=ActionPhase.WITHDRAWN,
        start=record.start,
        expected_end=record.expected_end,
        checkpoints=record.checkpoints,
        registered_cycle=record.registered_cycle,
        commit_group_id=record.commit_group_id,
    )


def prepare_registration(
    cycle_id: CycleId,
    current_time: SimulationTime,
    records: Mapping[ActionId, ActionRecord],
    declarations: Mapping[RoleId, FinalizedDeclaration],
) -> RegistrationPlan:
    """根据全部角色的最终声明构建统一登记计划；纯函数，不修改状态。

    这是可直接调用的纯校验入口，不依赖此前是否经过响应屏障。除归属、
    复用与跨角色唯一性外，这里复用 registration_issues 对每份最终声明
    完整重跑结构/引用/提交组/时间/历史校验（不复制业务规则）：伪造或
    绕过翻译阶段的输入同样被拒绝。任何问题都进入 problems，由调用者
    整体拒绝。
    """
    validate_simulation_time(current_time, "当前剧情时刻")
    problems: list[str] = []
    maintained: list[ActionRecord] = []
    registered: list[ActionRecord] = []
    modified: list[ActionRecord] = []
    withdrawn_pending: list[ActionRecord] = []
    withdrawn_active: list[ActionRecord] = []
    new_ids: dict[ActionId, RoleId] = {}
    for role_id, declaration in sorted(declarations.items(), key=lambda item: str(item[0])):
        if not isinstance(declaration, FinalizedDeclaration):
            problems.append(f"声明必须是 FinalizedDeclaration：{role_id}")
            continue
        response = declaration.response
        if not isinstance(response, RoleResponse):
            problems.append(f"声明必须包含 RoleResponse：{role_id}")
            continue
        if response.role_id != role_id or response.cycle_id != cycle_id:
            problems.append(f"声明周期或归属与当前登记不一致：{role_id}")
            continue
        translation = declaration.translation
        if not isinstance(translation, TranslationResult):
            problems.append(f"翻译结果必须是 TranslationResult：{role_id}")
            continue
        if (
            translation.cycle_id,
            translation.role_id,
            translation.response_id,
        ) != (
            response.cycle_id,
            response.role_id,
            response.response_id,
        ):
            problems.append(f"翻译结果必须关联同一响应：{role_id}")
            continue
        # 独立重跑登记前校验：结构、引用、提交组、时间与历史约束。
        role_arrangements = {
            record.action_id: record.arrangement()
            for record in records.values()
            if record.role_id == role_id
        }
        for issue in registration_issues(
            response, translation, current_time=current_time, arrangements=role_arrangements
        ):
            problems.append(f"{role_id}：{issue.text}")
        schedules = {schedule.action_id: schedule for schedule in translation.schedules}
        modified_schedules = {
            schedule.action_id: schedule for schedule in translation.modified_schedules
        }
        declared: set[ActionId] = set()
        for action in response.actions:
            action_id = action.action_id
            record = records.get(action_id)
            if isinstance(action, NewAction):
                if record is not None:
                    problems.append(f"新行动不得复用既有行动身份：{action_id}")
                    continue
                other = new_ids.get(action_id)
                if other is not None and other != role_id:
                    problems.append(f"新行动身份跨角色重复：{action_id}")
                    continue
                schedule = schedules.get(action_id)
                if schedule is None:
                    problems.append(f"新行动缺少时间安排：{action_id}")
                    continue
                if schedule.commit_group_id != action.commit_group_id:
                    problems.append(f"时间安排的提交组与行动不一致：{action_id}")
                    continue
                new_ids[action_id] = role_id
                registered.append(_new_record(action, schedule, cycle_id, role_id))
                declared.add(action_id)
            elif isinstance(action, ModifiedAction):
                if record is None:
                    problems.append(f"修改声明指向不存在的行动：{action_id}")
                    continue
                if record.role_id != role_id:
                    problems.append(f"不能修改其他角色的行动：{action_id}")
                    continue
                if record.phase not in MAINTAINABLE_PHASES:
                    problems.append(f"终态行动不能被修改：{action_id}")
                    continue
                modified_schedule = modified_schedules.get(action_id)
                if modified_schedule is None:
                    problems.append(f"修改声明缺少未来时间安排：{action_id}")
                    continue
                modified.append(_modified_record(record, modified_schedule))
                declared.add(action_id)
            else:
                if record is None:
                    problems.append(f"维持声明指向不存在的行动：{action_id}")
                    continue
                if record.role_id != role_id:
                    problems.append(f"不能维持其他角色的行动：{action_id}")
                    continue
                if record.phase not in MAINTAINABLE_PHASES:
                    problems.append(f"终态行动不能被重新维持：{action_id}")
                    continue
                maintained.append(record)
                declared.add(action_id)
        # 全量声明语义：本角色仍可维护但未声明的行动即撤回主动维持。
        for record in sorted(
            (item for item in records.values() if item.role_id == role_id),
            key=lambda item: str(item.action_id),
        ):
            if record.phase in MAINTAINABLE_PHASES and record.action_id not in declared:
                target = (
                    withdrawn_active
                    if record.phase != ActionPhase.REGISTERED
                    else withdrawn_pending
                )
                target.append(_withdrawn_record(record))
    return RegistrationPlan(
        cycle_id=cycle_id,
        simulation_time=current_time,
        maintained=tuple(maintained),
        registered=tuple(registered),
        modified=tuple(modified),
        withdrawn_pending=tuple(withdrawn_pending),
        withdrawn_active=tuple(withdrawn_active),
        problems=tuple(problems),
    )


class ActionRegistry:
    """不可变快照式的行动登记表；每次变更返回新实例。"""

    def __init__(self, records: Mapping[ActionId, ActionRecord] | None = None) -> None:
        self._records: dict[ActionId, ActionRecord] = dict(records or {})

    @property
    def records(self) -> Mapping[ActionId, ActionRecord]:
        return MappingProxyType(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def for_role(self, role_id: RoleId) -> tuple[ActionRecord, ...]:
        return tuple(
            sorted(
                (item for item in self._records.values() if item.role_id == role_id),
                key=lambda item: str(item.action_id),
            )
        )

    def arrangements_for(self, role_id: RoleId) -> tuple[ActionArrangement, ...]:
        return tuple(item.arrangement() for item in self.for_role(role_id))

    def apply(self, plan: RegistrationPlan) -> ActionRegistry:
        """应用已通过校验的登记计划；计划问题未清空时整体拒绝。"""
        if not plan.ok:
            raise ValueError("登记计划未通过校验，不得提交")
        updated = dict(self._records)
        for record in (
            *plan.registered,
            *plan.modified,
            *plan.withdrawn_pending,
            *plan.withdrawn_active,
        ):
            updated[record.action_id] = record
        return ActionRegistry(updated)

    def apply_outcomes(
        self, outcomes: Iterable[ActionOutcome]
    ) -> tuple[ActionRegistry, tuple[ActionRecord, ...]]:
        """按 World 提交的结算结果更新行动阶段；返回新登记表与被更新记录。

        登记阶段只反映 World 的实际结果：预计结束时刻到达本身不改变
        阶段。in_progress 保持原阶段；UNCONFIRMED 表示结果未知，不是
        已知客观终止——阶段标记为未确认但行动仍可维持、修改，后续
        节点不因此删除。实际已结束只有完成、失败与撤回。
        """
        changed: list[ActionRecord] = []
        for outcome in outcomes:
            record = self._records.get(outcome.action_id)
            if record is None:
                raise ValueError(f"结算结果引用未登记的行动：{outcome.action_id}")
            if record.phase in TERMINAL_PHASES:
                raise ValueError(f"已结束行动不能再次结算：{outcome.action_id}")
            phase = record.phase
            if outcome.result.value == "started":
                phase = ActionPhase.STARTED
            elif outcome.result.value in ("completed", "failed", "unconfirmed"):
                phase = ActionPhase(outcome.result.value)
            if phase == record.phase:
                continue
            updated = ActionRecord(
                action_id=record.action_id,
                role_id=record.role_id,
                description=record.description,
                phase=phase,
                start=record.start,
                expected_end=record.expected_end,
                checkpoints=record.checkpoints,
                registered_cycle=record.registered_cycle,
                commit_group_id=record.commit_group_id,
            )
            changed.append(updated)
        if not changed:
            return self, ()
        merged = dict(self._records)
        for record in changed:
            merged[record.action_id] = record
        return ActionRegistry(merged), tuple(changed)


def outcome_transition_problems(
    records: Mapping[ActionId, ActionRecord],
    outcomes: Iterable[ActionOutcome],
    allowed_results: Mapping[ActionId, frozenset[Any]],
) -> tuple[str, ...]:
    """在世界事实提交前预检结算结果与登记阶段、节点类型的兼容性。

    引擎在调用 commit_adjudication 之前执行：未登记、已结束或结果
    类别与节点不符都在提交前发现，不在事实提交后才报错。World 可
    返回进行中或未确认；引擎不自行把未知解释为成功或失败。
    """
    problems: list[str] = []
    for outcome in outcomes:
        record = records.get(outcome.action_id)
        if record is None:
            problems.append(f"结算结果引用未登记的行动：{outcome.action_id}")
            continue
        if record.phase in TERMINAL_PHASES:
            problems.append(f"已结束行动不能再次结算：{outcome.action_id}")
            continue
        allowed = allowed_results.get(outcome.action_id, frozenset())
        if outcome.result not in allowed:
            problems.append(
                f"结算结果与节点类型不符：{outcome.action_id} 得到 {outcome.result.value}"
            )
    return tuple(problems)
