"""World 裁决候选与校验提交结果的消息。

裁决候选、运行时校验、实际提交与发布是分开的边界：未经校验提交的
候选不是客观事实，不能进入角色感知；提交错误不得留下半提交事实。
联合裁决基于本时刻事件提交前、World 已更新相关未完成行动进展后的
共同状态；角色声明检查才基于事件落地后的共同状态。具体事实模型与
语义判断由 World 实现注入。

本模块还定义引擎交给 World 的行动节点（保留完整内容与安排）、
行动结算结果与进展更新报告；事件队列本身见 domain/scheduling.py。
"""

import enum
from dataclasses import dataclass

from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_text,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.time import SimulationTime, validate_simulation_time


class ActionNodeKind(enum.StrEnum):
    """交给 World 的行动节点类型；开始、过程与结束是不同的结算边界。"""

    START = "start"
    CHECKPOINT = "checkpoint"
    END = "end"
    WITHDRAWN = "withdrawn"


class ActionOutcomeResult(enum.StrEnum):
    """World 结算给出的行动实际结果；完成、失败与未确认保持区分。"""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True)
class FactChangeProposal:
    """一条事实变更候选；具体事实模型由 World 实现定义与解释。"""

    description: str

    def __post_init__(self) -> None:
        require_text(self.description, "事实变更描述")


@dataclass(frozen=True, slots=True)
class AdjudicationActionNode:
    """随事件批交给 World 的行动节点：事件身份、行动身份、类型与登记安排。

    队列不能只把不可解析的身份交给 World；节点携带触发它的引擎事件
    身份、执行角色的归属、登记时的行动描述与原时间安排，供语义判断
    使用。发布感知时观察可以引用该事件身份。
    """

    event_id: EventId
    action_id: ActionId
    role_id: RoleId
    node: ActionNodeKind
    description: str
    start: SimulationTime
    expected_end: SimulationTime

    def __post_init__(self) -> None:
        require_identifier(self.event_id, "节点事件身份")
        require_identifier(self.action_id, "行动身份")
        require_identifier(self.role_id, "执行角色身份")
        if not isinstance(self.node, ActionNodeKind):
            raise ValueError("行动节点类型必须是 ActionNodeKind")
        require_text(self.description, "行动描述")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        if self.expected_end <= self.start:
            raise ValueError("预计结束时刻必须晚于开始时刻")


def _validate_window(
    start: SimulationTime, end: SimulationTime, checkpoints: tuple[SimulationTime, ...]
) -> None:
    validate_simulation_time(start, "开始时刻")
    validate_simulation_time(end, "预计结束时刻")
    if end <= start:
        raise ValueError("预计结束时刻必须晚于开始时刻")
    previous = start
    for checkpoint in checkpoints:
        validate_simulation_time(checkpoint, "过程节点")
        if checkpoint <= previous or checkpoint >= end:
            raise ValueError("过程节点必须严格递增且在开始与结束之间")
        previous = checkpoint


@dataclass(frozen=True, slots=True)
class ActionModificationNode:
    """统一登记时交给 World 的明确修改：归属、变化与新旧未来安排。

    修改必须在 World 实际生效：World 依据原描述、明确变化与新安排
    裁决实际作用（如速度、方向），经校验提交后与登记统一生效；
    历史开始与已过节点保持不变。
    """

    event_id: EventId
    action_id: ActionId
    role_id: RoleId
    description: str
    change: str
    old_start: SimulationTime
    old_expected_end: SimulationTime
    old_checkpoints: tuple[SimulationTime, ...]
    new_start: SimulationTime
    new_expected_end: SimulationTime
    new_checkpoints: tuple[SimulationTime, ...]

    def __post_init__(self) -> None:
        require_identifier(self.event_id, "修改节点事件身份")
        require_identifier(self.action_id, "行动身份")
        require_identifier(self.role_id, "执行角色身份")
        require_text(self.description, "原行动描述")
        require_text(self.change, "行动修改")
        for name, checkpoints in (("旧", self.old_checkpoints), ("新", self.new_checkpoints)):
            if not isinstance(checkpoints, tuple):
                raise ValueError(f"{name}过程节点必须是毫秒时间元组")
        _validate_window(self.old_start, self.old_expected_end, self.old_checkpoints)
        _validate_window(self.new_start, self.new_expected_end, self.new_checkpoints)


@dataclass(frozen=True, slots=True)
class ObjectiveContext:
    """翻译单个角色请求所需的只读客观上下文。

    由 World 提供并携带共同状态版本：位置、姿态、障碍与实际进展等
    条件以 World 的事实性描述表达。该上下文只交给翻译器，不得流入
    角色感知。
    """

    world_state_version: WorldStateVersion
    role_id: RoleId
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        require_version(self.world_state_version, "共同世界状态版本")
        require_identifier(self.role_id, "角色身份")
        if not isinstance(self.conditions, tuple) or not all(
            isinstance(item, str) and item for item in self.conditions
        ):
            raise ValueError("客观条件必须是非空字符串元组")


@dataclass(frozen=True, slots=True)
class WorldActionState:
    """World 诊断快照中的实际行动状态（客观，非角色认知）。"""

    action_id: ActionId
    role_id: RoleId
    state: str
    detail: str

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        require_identifier(self.role_id, "角色身份")
        require_text(self.state, "实际行动状态")
        require_text(self.detail, "行动状态说明")


@dataclass(frozen=True, slots=True)
class WorldDiagnostic:
    """World 对外的只读诊断快照：实际行动状态与被动过程。

    区别于已提交事实的增量文本；界面用它呈现客观现状，不进入角色
    感知，也不可被当作新的世界事实来源。
    """

    world_state_version: WorldStateVersion
    action_states: tuple[WorldActionState, ...] = ()
    passive_processes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_version(self.world_state_version, "共同世界状态版本")
        if not isinstance(self.action_states, tuple) or not all(
            isinstance(item, WorldActionState) for item in self.action_states
        ):
            raise ValueError("实际行动状态必须是 WorldActionState 元组")
        require_unique_identifiers([item.action_id for item in self.action_states], "实际行动身份")
        if not isinstance(self.passive_processes, tuple) or not all(
            isinstance(item, str) and item for item in self.passive_processes
        ):
            raise ValueError("被动过程必须是非空字符串元组")


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """World 对一个行动的结算结果；未提交前只是候选的一部分。"""

    action_id: ActionId
    result: ActionOutcomeResult
    detail: str

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        if not isinstance(self.result, ActionOutcomeResult):
            raise ValueError("行动结果类别必须是 ActionOutcomeResult")
        require_text(self.detail, "行动结果说明")


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """World 按已流逝剧情时间更新行动进展后的报告；只描述，不裁决。"""

    cycle_id: CycleId
    simulation_time: SimulationTime
    updated_action_ids: tuple[ActionId, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.updated_action_ids, tuple):
            raise ValueError("更新行动身份必须是元组")
        require_unique_identifiers(self.updated_action_ids, "更新行动身份")
        if not isinstance(self.notes, tuple) or not all(
            isinstance(item, str) and item for item in self.notes
        ):
            raise ValueError("进展说明必须是非空字符串元组")


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
    action_outcomes: tuple[ActionOutcome, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.event_ids, tuple):
            raise ValueError("关联事件身份必须是元组")
        require_unique_identifiers(self.event_ids, "关联事件身份")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("关联行动身份必须是元组")
        require_unique_identifiers(self.action_ids, "关联行动身份")
        if not isinstance(self.action_outcomes, tuple) or not all(
            isinstance(item, ActionOutcome) for item in self.action_outcomes
        ):
            raise ValueError("行动结算结果必须是 ActionOutcome 元组")
        require_unique_identifiers(
            [item.action_id for item in self.action_outcomes], "结算结果行动身份"
        )
        for item in self.action_outcomes:
            if item.action_id not in self.action_ids:
                raise ValueError(f"结算结果引用了未参与裁决的行动：{item.action_id}")
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
        for follow_up in self.follow_up_events:
            if follow_up.simulation_time <= self.simulation_time:
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
    action_outcomes: tuple[ActionOutcome, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, AdjudicationCandidate):
            raise ValueError("提交结果必须关联裁决候选")
        if not isinstance(self.committed_changes, tuple) or not all(
            isinstance(item, FactChangeProposal) for item in self.committed_changes
        ):
            raise ValueError("实际状态变化必须是 FactChangeProposal 元组")
        require_version(self.resulting_version, "共同世界状态版本")
        if not isinstance(self.action_outcomes, tuple) or not all(
            isinstance(item, ActionOutcome) for item in self.action_outcomes
        ):
            raise ValueError("提交的行动结算结果必须是 ActionOutcome 元组")
        require_unique_identifiers(
            [item.action_id for item in self.action_outcomes], "结算结果行动身份"
        )
        proposed = {item.action_id: item for item in self.candidate.action_outcomes}
        for item in self.action_outcomes:
            if proposed.get(item.action_id) != item:
                raise ValueError(f"提交的结算结果必须来自候选：{item.action_id}")
