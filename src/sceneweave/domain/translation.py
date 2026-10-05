"""行动翻译的输入输出消息。

翻译器把通过检查的声明转换为时间安排；这里只定义消息与结构校验，
不实现翻译模型本身。时间安排是对尝试的安排，不是已经发生的事实。
新行动与修改声明的安排是不同的约束：新行动构成并行组并须满足共同
非零时间交集，修改只能调整尚未发生的节点、不得改写历史。
"""

from dataclasses import dataclass

from sceneweave.domain.adjudication import ObjectiveContext
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.messages import ActionPhase, RoleResponse
from sceneweave.domain.time import SimulationTime, validate_simulation_time


def _validate_window(
    start: SimulationTime, end: SimulationTime, checkpoints: tuple[SimulationTime, ...]
) -> None:
    """开始、过程节点与结束的通用结构约束；与当前时刻无关。"""
    if end <= start:
        raise ValueError("预计结束时刻必须晚于开始时刻")
    if not isinstance(checkpoints, tuple):
        raise ValueError("过程节点必须是毫秒时间元组")
    previous = start
    for checkpoint in checkpoints:
        validate_simulation_time(checkpoint, "过程节点")
        if checkpoint <= previous:
            raise ValueError("过程节点必须严格递增且晚于开始时刻")
        previous = checkpoint
    if end <= previous:
        raise ValueError("过程节点必须早于预计结束时刻")


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    """单个新行动的执行时间安排。

    开始、过程节点与结束时刻都是本次提交的安排，不代表状态变化
    已经发生；实际结果仍由 World 在对应剧情时刻结算。
    """

    action_id: ActionId
    commit_group_id: CommitGroupId
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        require_identifier(self.commit_group_id, "提交组身份")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class ModifiedActionSchedule:
    """对已有行动的修改后的完整时间安排。

    历史不可改写：已到或已过的开始与过程节点必须原样保留，只有
    未来节点可以调整。该约束与当前时刻相关，在登记校验时核对；
    这里只做结构性校验。修改不参与新行动的并行组交集约束。
    """

    action_id: ActionId
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class ActionArrangement:
    """翻译请求携带的既有行动安排与阶段；表达引擎已知的事实性安排。"""

    action_id: ActionId
    phase: ActionPhase
    description: str
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        if not isinstance(self.phase, ActionPhase):
            raise ValueError("行动阶段必须是 ActionPhase")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("行动描述必须是非空字符串")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """交给行动翻译器的输入：已通过检查的完整响应与客观上下文。

    arrangements 是该角色已登记行动的现行安排与阶段（既有进展的
    引擎侧事实）；objective_context 是 World 提供的只读客观条件
    （位置、姿态、障碍、实际进展，携带共同状态版本），只供翻译器
    使用，不得流入角色感知。
    """

    response: RoleResponse
    current_time: SimulationTime
    world_state_version: WorldStateVersion
    arrangements: tuple[ActionArrangement, ...] = ()
    objective_context: ObjectiveContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, RoleResponse):
            raise ValueError("翻译输入必须包含角色响应")
        validate_simulation_time(self.current_time, "当前剧情时刻")
        require_version(self.world_state_version, "共同世界状态版本")
        if not isinstance(self.arrangements, tuple) or not all(
            isinstance(item, ActionArrangement) for item in self.arrangements
        ):
            raise ValueError("既有安排必须是 ActionArrangement 元组")
        require_unique_identifiers(
            [item.action_id for item in self.arrangements], "既有安排行动身份"
        )
        if self.objective_context is not None and not isinstance(
            self.objective_context, ObjectiveContext
        ):
            raise ValueError("客观上下文必须是 ObjectiveContext")
        if (
            self.objective_context is not None
            and self.objective_context.role_id != self.response.role_id
        ):
            raise ValueError("客观上下文必须属于被翻译的角色")
        if (
            self.objective_context is not None
            and self.objective_context.world_state_version != self.world_state_version
        ):
            raise ValueError("客观上下文必须使用翻译请求的共同世界状态版本")


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """翻译输出：保留原响应及提交组关联，给出各声明的未来时间安排。

    schedules 只覆盖本次新行动；modified_schedules 覆盖明确修改的
    行动。维持声明不需要安排：同 ID 维持不重新翻译或重排。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    schedules: tuple[ActionSchedule, ...] = ()
    modified_schedules: tuple[ModifiedActionSchedule, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        if not isinstance(self.schedules, tuple) or not all(
            isinstance(item, ActionSchedule) for item in self.schedules
        ):
            raise ValueError("时间安排必须是 ActionSchedule 元组")
        require_unique_identifiers([item.action_id for item in self.schedules], "行动身份")
        if not isinstance(self.modified_schedules, tuple) or not all(
            isinstance(item, ModifiedActionSchedule) for item in self.modified_schedules
        ):
            raise ValueError("修改安排必须是 ModifiedActionSchedule 元组")
        require_unique_identifiers(
            [item.action_id for item in self.modified_schedules], "修改行动身份"
        )
        groups = {item.commit_group_id for item in self.schedules}
        if len(groups) > 1:
            raise ValueError("翻译结果只能包含一个提交组")
        overlap = {item.action_id for item in self.schedules} & {
            item.action_id for item in self.modified_schedules
        }
        if overlap:
            raise ValueError(f"同一行动不能同时是新行动安排与修改安排：{sorted(overlap)}")
