"""跨职责边界传递的不可变消息。"""

from dataclasses import dataclass
from typing import NewType

RoleId = NewType("RoleId", str)
EventId = NewType("EventId", str)
ActionId = NewType("ActionId", str)
# 剧情时间以非负整数 tick 表示；物理单位由场景配置决定，不是墙钟时间。
SimulationTime = NewType("SimulationTime", int)


@dataclass(frozen=True, slots=True)
class WorldEvent:
    """交给 World 落地的事件；描述本身不证明客观事实已经成立。"""

    event_id: EventId
    simulation_time: SimulationTime
    description: str


@dataclass(frozen=True, slots=True)
class Observation:
    """World 确认并按感知范围生成的单条角色输入。"""

    event_id: EventId
    simulation_time: SimulationTime
    content: str


@dataclass(frozen=True, slots=True)
class Delivery:
    """向指定角色投递感知，不包含其他角色的私有信息。"""

    role_id: RoleId
    observation: Observation


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """角色提交的行为尝试；尚未登记、开始或结算。"""

    action_id: ActionId
    role_id: RoleId
    description: str


@dataclass(frozen=True, slots=True)
class RoleResponse:
    """本次响应的新行动提议；空元组不取消已有行动。"""

    actions: tuple[ActionProposal, ...] = ()
