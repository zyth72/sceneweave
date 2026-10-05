"""一个完整事件周期结束后对外呈现的报告。

报告区分角色可见内容（感知、自身行动状态）与客观/诊断内容（同刻
全部事件、提交的世界事实、结算结果、登记后的行动记录、World 诊断
快照）；接入层据此构造界面步数据，不得把诊断内容混入角色感知。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sceneweave.domain.adjudication import ActionOutcome, WorldDiagnostic
from sceneweave.domain.identity import ActionId, CycleId, EventId, RoleId
from sceneweave.domain.messages import KnownActionState
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.engine.actions import ActionRecord


@dataclass(frozen=True, slots=True)
class CycleEvent:
    """本周期处理批中的一个事件：身份、类型与完整描述。

    行动节点事件也携带类型与身份；界面据此显示「行动开始轮」的
    实际触发内容，而不只显示外部世界事件。
    """

    event_id: EventId
    kind: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("事件身份必须是非空字符串")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("事件类型必须是非空字符串")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("事件描述必须是非空字符串")


@dataclass(frozen=True, slots=True)
class RoleCycleView:
    """单个角色在本周期的可见视图：主动声明、感知与自身行动状态。"""

    role_id: RoleId
    declarations: tuple[str, ...]
    perceptions: tuple[str, ...]
    action_states: tuple[KnownActionState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role_id, str) or not self.role_id:
            raise ValueError("角色身份必须是非空字符串")
        for name in ("declarations", "perceptions"):
            texts = getattr(self, name)
            if not isinstance(texts, tuple) or not all(
                isinstance(item, str) and item for item in texts
            ):
                raise ValueError(f"{name}必须是非空字符串元组")
        if not isinstance(self.action_states, tuple) or not all(
            isinstance(item, KnownActionState) for item in self.action_states
        ):
            raise ValueError("行动状态必须是 KnownActionState 元组")
        if len({item.action_id for item in self.action_states}) != len(self.action_states):
            raise ValueError("行动状态身份必须唯一")


@dataclass(frozen=True, slots=True)
class CycleReport:
    """一个完整事件周期的结果报告。

    batch_events 是同刻处理批的全部事件（含行动节点与内思考）；
    registry_actions 是本周期统一登记生效后的行动记录（界面显示
    新登记的 REGISTERED、撤回的 WITHDRAWN 等真实阶段）；world_
    diagnostic 是 World 在周期完成后的只读客观快照（实际行动状态
    与被动过程）。角色可见视图保持输入时的投递内容，不被诊断替换。
    """

    cycle_id: CycleId
    step: int
    simulation_time: SimulationTime
    batch_events: tuple[CycleEvent, ...]
    introspection: bool
    committed_facts: tuple[str, ...]
    action_outcomes: tuple[ActionOutcome, ...]
    roles: tuple[RoleCycleView, ...]
    registry_actions: tuple[ActionRecord, ...] = ()
    world_diagnostic: WorldDiagnostic | None = None
    registered: tuple[ActionRecord, ...] = ()
    modified: tuple[ActionRecord, ...] = ()
    withdrawn: tuple[ActionRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("周期身份必须是非空字符串")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("步序号必须是正整数")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.batch_events, tuple) or not all(
            isinstance(item, CycleEvent) for item in self.batch_events
        ):
            raise ValueError("同刻事件必须是 CycleEvent 元组")
        if len({item.event_id for item in self.batch_events}) != len(self.batch_events):
            raise ValueError("同刻事件身份必须唯一")
        for name in ("committed_facts",):
            texts = getattr(self, name)
            if not isinstance(texts, tuple) or not all(
                isinstance(item, str) and item for item in texts
            ):
                raise ValueError(f"{name}必须是非空字符串元组")
        if type(self.introspection) is not bool:
            raise ValueError("内思考标记必须是布尔值")
        if not isinstance(self.action_outcomes, tuple) or not all(
            isinstance(item, ActionOutcome) for item in self.action_outcomes
        ):
            raise ValueError("结算结果必须是 ActionOutcome 元组")
        # 同一行动可以在一个周期内先后出现多个结算结果（例如先开始、
        # 又在同周期撤回）；按发生顺序保留，不合并。
        if not isinstance(self.roles, tuple) or not all(
            isinstance(item, RoleCycleView) for item in self.roles
        ):
            raise ValueError("角色视图必须是 RoleCycleView 元组")
        if len({item.role_id for item in self.roles}) != len(self.roles):
            raise ValueError("角色视图身份必须唯一")
        for name in ("registry_actions", "registered", "modified", "withdrawn"):
            records = getattr(self, name)
            if not isinstance(records, tuple) or not all(
                isinstance(item, ActionRecord) for item in records
            ):
                raise ValueError(f"{name}必须是 ActionRecord 元组")
        if self.world_diagnostic is not None and not isinstance(
            self.world_diagnostic, WorldDiagnostic
        ):
            raise ValueError("World 诊断必须是 WorldDiagnostic")

    @property
    def world_events(self) -> tuple[str, ...]:
        """外部世界事件的描述；兼容既有呈现，完整事件见 batch_events。"""
        return tuple(event.description for event in self.batch_events if event.kind == "world")

    def view_for(self, role_id: RoleId) -> RoleCycleView | None:
        for view in self.roles:
            if view.role_id == role_id:
                return view
        return None

    def action_changes(self) -> Mapping[ActionId, str]:
        """行动身份到变化原因的只读映射，用于诊断呈现。"""
        changes: dict[ActionId, str] = {}
        for record in self.registered:
            changes[record.action_id] = "已登记等待开始"
        for record in self.modified:
            changes[record.action_id] = "修改未来安排"
        for record in self.withdrawn:
            changes[record.action_id] = "撤回主动维持"
        return MappingProxyType(changes)
