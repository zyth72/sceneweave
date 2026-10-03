"""World 裁决候选与校验提交结果的消息。

裁决候选、运行时校验、实际提交与发布是分开的边界：未经校验提交的
候选不是客观事实，不能进入角色感知；提交错误不得留下半提交事实。
联合裁决基于本时刻事件提交前、World 已更新相关未完成行动进展后的
共同状态；角色声明检查才基于事件落地后的共同状态。具体事实模型与
语义判断由 World 实现注入。
"""

from dataclasses import dataclass

from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    WorldStateVersion,
    require_identifier,
    require_text,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.time import SimulationTime, validate_simulation_time


@dataclass(frozen=True, slots=True)
class FactChangeProposal:
    """一条事实变更候选；具体事实模型由 World 实现定义与解释。"""

    description: str

    def __post_init__(self) -> None:
        require_text(self.description, "事实变更描述")


@dataclass(frozen=True, slots=True)
class AdjudicationCandidate:
    """对同刻事件与行动的联合裁决候选。

    候选基于本时刻事件提交前、World 已更新相关未完成行动进展后的
    共同状态，同时考虑相互作用。候选未经校验提交，不是客观事实；
    其中的后续世界事件必须晚于裁决时刻，尚未发生。
    """

    cycle_id: CycleId
    simulation_time: SimulationTime
    event_ids: tuple[EventId, ...] = ()
    action_ids: tuple[ActionId, ...] = ()
    proposed_changes: tuple[FactChangeProposal, ...] = ()
    follow_up_events: tuple[WorldEvent, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.event_ids, tuple):
            raise ValueError("关联事件身份必须是元组")
        require_unique_identifiers(self.event_ids, "关联事件身份")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("关联行动身份必须是元组")
        require_unique_identifiers(self.action_ids, "关联行动身份")
        if not isinstance(self.proposed_changes, tuple) or not all(
            isinstance(item, FactChangeProposal) for item in self.proposed_changes
        ):
            raise ValueError("事实变更候选必须是 FactChangeProposal 元组")
        if not isinstance(self.follow_up_events, tuple) or not all(
            isinstance(item, WorldEvent) for item in self.follow_up_events
        ):
            raise ValueError("后续世界事件必须是 WorldEvent 元组")
        require_unique_identifiers(
            [item.event_id for item in self.follow_up_events], "后续事件身份"
        )
        for item in self.follow_up_events:
            if item.simulation_time <= self.simulation_time:
                raise ValueError("后续世界事件必须晚于裁决时刻")


@dataclass(frozen=True, slots=True)
class AdjudicationValidation:
    """候选的运行时校验结果，关联候选与其所依据的世界状态版本。

    提交方须核对候选与版本关联，拒绝错误或过期的校验结果；不通过时
    逐条给出问题，不提交事实。
    """

    candidate: AdjudicationCandidate
    world_state_version: WorldStateVersion
    passed: bool
    problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, AdjudicationCandidate):
            raise ValueError("校验结果必须关联裁决候选")
        require_version(self.world_state_version, "共同世界状态版本")
        if type(self.passed) is not bool:
            raise ValueError("校验结论必须是布尔值")
        if not isinstance(self.problems, tuple) or not all(
            isinstance(item, str) and item for item in self.problems
        ):
            raise ValueError("校验问题必须是非空字符串元组")
        if self.passed != (not self.problems):
            raise ValueError("通过状态与问题列表必须一致")


@dataclass(frozen=True, slots=True)
class AdjudicationCommit:
    """经运行时校验后实际提交的裁决结果。

    只有提交后的状态变化才是客观事实；发布据此生成过滤后的感知。
    """

    candidate: AdjudicationCandidate
    committed_changes: tuple[FactChangeProposal, ...]
    resulting_version: WorldStateVersion

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, AdjudicationCandidate):
            raise ValueError("提交结果必须关联裁决候选")
        if not isinstance(self.committed_changes, tuple) or not all(
            isinstance(item, FactChangeProposal) for item in self.committed_changes
        ):
            raise ValueError("实际状态变化必须是 FactChangeProposal 元组")
        require_version(self.resulting_version, "共同世界状态版本")
