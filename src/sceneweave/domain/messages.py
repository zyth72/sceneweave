"""跨模块传递的不可变消息。

所有消息在构造时校验身份、类型、时间与关联一致性；嵌套集合一律
使用元组（传入 list 等可变集合直接拒绝），不暴露共享可变状态。
自然语言字段表达行动与事实语义，身份、归属、时间和关联关系使用
结构字段，文本字段必须是字符串。
"""

import enum
from dataclasses import dataclass

from sceneweave.domain.action_plan import ActionPlan, PlanAction, StepAction
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_text,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.time import SimulationTime, validate_simulation_time

# 显式再导出身份与时间类型：messages 是跨模块消息的稳定入口，
# 旧调用方（如 model_trace）从本模块导入这些名字。
__all__ = [
    "ActionDeclaration",
    "ActionId",
    "ActionPhase",
    "CommitGroupId",
    "CycleId",
    "EventId",
    "ResponseId",
    "RoleId",
    "SimulationTime",
    "WorldStateVersion",
    "ActionCheckResult",
    "Delivery",
    "KnownActionState",
    "MaintainAction",
    "ModifiedAction",
    "NewAction",
    "Observation",
    "Rejection",
    "RoleResponse",
    "StepAction",
    "TriggerReason",
    "TurnInput",
    "WorldEvent",
]


class ActionPhase(enum.StrEnum):
    """角色可获知的行动阶段；提议、登记、开始、完成、失败、未确认与撤回保持可区分。

    引擎登记记录与角色可见状态共用该词汇：REGISTERED 表示已登记等待
    开始，WITHDRAWN 表示角色撤回主动维持（客观结果另由 World 结算）。
    """

    SUBMITTED = "submitted"
    REGISTERED = "registered"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True, slots=True)
class WorldEvent:
    """交给 World 裁决的事件；描述本身不证明客观事实已经成立。"""

    event_id: EventId
    simulation_time: SimulationTime
    description: str

    def __post_init__(self) -> None:
        require_identifier(self.event_id, "事件身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        require_text(self.description, "事件描述")


@dataclass(frozen=True, slots=True)
class Observation:
    """World 确认并按感知范围生成的单条角色输入。"""

    event_id: EventId
    simulation_time: SimulationTime
    content: str

    def __post_init__(self) -> None:
        require_identifier(self.event_id, "事件身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        require_text(self.content, "感知内容")


@dataclass(frozen=True, slots=True)
class Delivery:
    """向指定角色发布的感知，不包含其他角色的私有信息。"""

    role_id: RoleId
    observation: Observation

    def __post_init__(self) -> None:
        require_identifier(self.role_id, "角色身份")
        if not isinstance(self.observation, Observation):
            raise ValueError("投递内容必须是 Observation")


@dataclass(frozen=True, slots=True)
class TriggerReason:
    """本轮输入的触发原因。

    周期性内思考通过 introspection 标记表达，不伪造外部感知；
    两者都为空表示本轮对该角色没有可感知事件、也非内思考唤醒。
    与外部事件同刻时合并为一次输入。
    """

    event_ids: tuple[EventId, ...] = ()
    introspection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.event_ids, tuple):
            raise ValueError("触发事件身份必须是元组")
        require_unique_identifiers(self.event_ids, "触发事件身份")
        if type(self.introspection) is not bool:
            raise ValueError("内思考标记必须是布尔值")


@dataclass(frozen=True, slots=True)
class KnownActionState:
    """角色可获知的自身行动状态及结果反馈；不包含其他主体的私有信息。"""

    action_id: ActionId
    phase: ActionPhase
    detail: str

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        if not isinstance(self.phase, ActionPhase):
            raise ValueError("行动阶段必须是 ActionPhase")
        require_text(self.detail, "行动状态反馈")


@dataclass(frozen=True, slots=True)
class TurnInput:
    """一个处理周期内交给单个角色的输入。

    输入投递前，本轮事件的客观结果已提交，感知已按该角色的范围过滤；
    同刻多个事件合并为一次输入，同一事件也可以产生多条感知，还可以
    完全没有外部感知。感知只能引用本轮触发的外部事件，触发原因没有
    外部事件时不得凭空携带外部感知。
    """

    cycle_id: CycleId
    role_id: RoleId
    simulation_time: SimulationTime
    trigger: TriggerReason
    observations: tuple[Observation, ...] = ()
    known_action_states: tuple[KnownActionState, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.trigger, TriggerReason):
            raise ValueError("触发原因必须是 TriggerReason")
        if not isinstance(self.observations, tuple) or not all(
            isinstance(item, Observation) for item in self.observations
        ):
            raise ValueError("感知必须是 Observation 元组")
        triggered = set(self.trigger.event_ids)
        for item in self.observations:
            if item.simulation_time != self.simulation_time:
                raise ValueError("感知必须与输入属于同一剧情时刻")
            if item.event_id not in triggered:
                raise ValueError(f"感知引用了本轮未触发的事件：{item.event_id}")
        if not isinstance(self.known_action_states, tuple) or not all(
            isinstance(item, KnownActionState) for item in self.known_action_states
        ):
            raise ValueError("已知行动状态必须是 KnownActionState 元组")
        require_unique_identifiers(
            [item.action_id for item in self.known_action_states], "已知行动身份"
        )


@dataclass(frozen=True, slots=True)
class MaintainAction:
    """维持声明：继续主动维持已登记行动，保留身份、进展与时间安排。"""

    action_id: ActionId

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")


@dataclass(frozen=True, slots=True)
class NewAction:
    """本次新行动尝试；同一响应中的全部新行动属于同一个并行提交组。"""

    action_id: ActionId
    description: str
    commit_group_id: CommitGroupId

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        require_text(self.description, "新行动描述")
        require_identifier(self.commit_group_id, "提交组身份")


@dataclass(frozen=True, slots=True)
class ModifiedAction:
    """对已有行动的明确修改；只调整尚未发生的部分，不改写历史。"""

    action_id: ActionId
    change: str

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        require_text(self.change, "行动修改")


type ActionDeclaration = StepAction | MaintainAction | NewAction | ModifiedAction


@dataclass(frozen=True, slots=True)
class RoleResponse:
    """一次完整的主动行为声明。

    ``actions`` 是从现在起要主动实施或维持的全部行为，必须显式提供
    （省略字段是构造错误，不会被当作空集合）；合法空集合表示撤回
    全部主动维持，不是未执行某个内部阶段。字段缺失、解析失败、
    调用尚未完成或运行错误不得被转换为空集合。

    协议纠正后的活跃路径使用结构化 ``StepAction``（原子动作 + 身份），
    并把角色内部的完整动作链放在 ``plan`` 里供翻译器整链拍平；World/
    引擎的检查与翻译消费完整 ``plan``，执行提交只登记 ``actions`` 中的
    最近步骤。旧的三类声明（维持/修改/新增）仅为低层历史测试保留，
    不进入模型活跃路径。

    ``reused`` 为真表示本声明复用角色内部已保存且已通过整链检查的完整
    链与排程（System1 判断无需重新考虑）：引擎不再翻译、不再整链检查，
    直接按 ``actions`` 做最近步骤执行提交。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    actions: tuple[ActionDeclaration, ...]
    plan: ActionPlan | None = None
    reused: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        if not isinstance(self.actions, tuple) or not all(
            isinstance(item, (StepAction, MaintainAction, NewAction, ModifiedAction))
            for item in self.actions
        ):
            raise ValueError("主动行为集合必须是行动声明元组")
        if type(self.reused) is not bool:
            raise ValueError("复用标记必须是布尔值")
        require_unique_identifiers([item.action_id for item in self.actions], "行动身份")
        groups = {item.commit_group_id for item in self.actions if isinstance(item, NewAction)}
        if len(groups) > 1:
            raise ValueError("一次响应的新行动只能属于一个并行提交组")
        if self.plan is not None:
            if not isinstance(self.plan, ActionPlan):
                raise ValueError("plan 必须是 ActionPlan 或 None")
            for plan_item in self.plan.actions:
                if not isinstance(plan_item, PlanAction) or plan_item.action_id is None:
                    raise ValueError("角色保存的完整动作链必须携带已分配的动作身份")
            by_id = {plan_item.action_id: plan_item for plan_item in self.plan.actions}
            for item in self.actions:
                if not isinstance(item, StepAction):
                    raise ValueError("声明与 plan 同时存在时，actions 只能是结构化当前动作")
                entry = by_id.get(item.action_id)
                if entry is None:
                    raise ValueError(f"当前动作不在保存的完整链中：{item.action_id}")
                if entry.action != item.action:
                    raise ValueError(f"当前动作与保存链的动作不一致：{item.action_id}")
                if tuple(ActionId(ref) for ref in entry.depends_on) != item.depends_on:
                    raise ValueError(f"当前动作与保存链的依赖不一致：{item.action_id}")

    @property
    def step_actions(self) -> tuple[StepAction, ...]:
        """只取结构化当前步骤声明；含旧类型时为内部错误的调用方。"""
        return tuple(item for item in self.actions if isinstance(item, StepAction))


@dataclass(frozen=True, slots=True)
class Rejection:
    """World 检查不通过后打回角色的反馈。

    只针对当前周期、当前角色、当前待检查的响应；没有新事件落地，
    不重复投递原始感知，检查打回也不是剧情中的执行失败。
    """

    cycle_id: CycleId
    role_id: RoleId
    rejected_response_id: ResponseId
    reason: str
    action_ids: tuple[ActionId, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.rejected_response_id, "被打回的响应身份")
        require_text(self.reason, "打回原因")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("有关行动身份必须是元组")
        require_unique_identifiers(self.action_ids, "有关行动身份")


@dataclass(frozen=True, slots=True)
class ActionCheckResult:
    """World 对完整声明的检查结果；检查只产生结果，不改变任何状态。

    action_ids 只能引用被检查声明内的行动；不通过时必须提供原因。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    world_state_version: WorldStateVersion
    passed: bool
    reason: str = ""
    action_ids: tuple[ActionId, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        require_version(self.world_state_version, "共同世界状态版本")
        if type(self.passed) is not bool:
            raise ValueError("检查结论必须是布尔值")
        if not isinstance(self.reason, str):
            raise ValueError("检查原因必须是字符串")
        if not self.passed and not self.reason:
            raise ValueError("检查不通过时必须提供原因")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("有关行动身份必须是元组")
        require_unique_identifiers(self.action_ids, "有关行动身份")
