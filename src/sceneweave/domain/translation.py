"""行动翻译的输入输出消息。

翻译器把通过检查的新行动转换为时间安排；这里只定义消息与结构
校验，不实现翻译模型本身。时间安排是对尝试的安排，不是已经
发生的事实。
"""

from dataclasses import dataclass

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
from sceneweave.domain.messages import RoleResponse
from sceneweave.domain.time import SimulationTime, validate_simulation_time


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
        if self.expected_end <= self.start:
            raise ValueError("预计结束时刻必须晚于开始时刻")
        if not isinstance(self.checkpoints, tuple):
            raise ValueError("过程节点必须是毫秒时间元组")
        previous = self.start
        for checkpoint in self.checkpoints:
            validate_simulation_time(checkpoint, "过程节点")
            if checkpoint <= previous:
                raise ValueError("过程节点必须严格递增且晚于开始时刻")
            previous = checkpoint
        if self.expected_end <= previous:
            raise ValueError("过程节点必须早于预计结束时刻")


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """交给行动翻译器的输入：已通过检查的完整响应与共同状态引用。"""

    response: RoleResponse
    current_time: SimulationTime
    world_state_version: WorldStateVersion

    def __post_init__(self) -> None:
        if not isinstance(self.response, RoleResponse):
            raise ValueError("翻译输入必须包含角色响应")
        validate_simulation_time(self.current_time, "当前剧情时刻")
        require_version(self.world_state_version, "共同世界状态版本")


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """翻译输出：保留原响应及提交组关联，给出各新行动的时间安排。

    当前登记校验只覆盖本次新行动的安排：维持声明无需重新安排；
    修改声明的未来时间安排尚未实现，待行动生命周期实现时补充，
    「修改不产生时间安排」不是长期业务约束。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    schedules: tuple[ActionSchedule, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        if not isinstance(self.schedules, tuple) or not all(
            isinstance(item, ActionSchedule) for item in self.schedules
        ):
            raise ValueError("时间安排必须是 ActionSchedule 元组")
        require_unique_identifiers([item.action_id for item in self.schedules], "行动身份")
        groups = {item.commit_group_id for item in self.schedules}
        if len(groups) > 1:
            raise ValueError("翻译结果只能包含一个提交组")
