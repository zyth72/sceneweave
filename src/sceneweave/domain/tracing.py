"""领域追踪接口：让感知、响应、检查、打回与错误可沿因果链查找。

domain/world/engine 只依赖本模块的接口，不依赖具体日志实现；
接入层负责把记录写入 Concordia 的组件与结构化日志机制。记录携带
触发来源、感知输入、行动输出、世界版本与重试响应关系，全部使用
不可变元组，不嵌套可变消息。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.time import SimulationTime, validate_simulation_time

type TraceValue = str | int | bool | None


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """一条可追踪事件；身份与因果字段按适用范围提供。

    stage 是组件/阶段标识；cycle/role/response/action 身份用于沿因果
    链检索；event_ids 是触发来源，inputs/outputs 是本轮感知与行动
    输出的文本快照；world_state_version 记录检查所依据的共同状态；
    related_response_id 表达打回与重试之间的新旧响应关系。details
    只接受可无损记录的标量值并冻结为只读映射。
    """

    stage: str
    cycle_id: CycleId
    simulation_time: SimulationTime
    summary: str
    role_id: RoleId | None = None
    response_id: ResponseId | None = None
    related_response_id: ResponseId | None = None
    action_ids: tuple[ActionId, ...] = ()
    event_ids: tuple[EventId, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    world_state_version: WorldStateVersion | None = None
    details: Mapping[str, TraceValue] = field(default_factory=lambda: MappingProxyType({}))
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("追踪阶段必须是非空字符串")
        require_identifier(self.cycle_id, "周期身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError("追踪摘要必须是非空字符串")
        if self.role_id is not None:
            require_identifier(self.role_id, "角色身份")
        if self.response_id is not None:
            require_identifier(self.response_id, "响应身份")
        if self.related_response_id is not None:
            require_identifier(self.related_response_id, "关联响应身份")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("行动身份必须是元组")
        require_unique_identifiers(self.action_ids, "行动身份")
        if not isinstance(self.event_ids, tuple):
            raise ValueError("触发事件身份必须是元组")
        require_unique_identifiers(self.event_ids, "触发事件身份")
        for texts, name in ((self.inputs, "感知输入"), (self.outputs, "行动输出")):
            if not isinstance(texts, tuple) or not all(
                isinstance(item, str) and item for item in texts
            ):
                raise ValueError(f"{name}必须是非空字符串元组")
        if self.world_state_version is not None:
            require_version(self.world_state_version, "共同世界状态版本")
        if not isinstance(self.details, Mapping):
            raise ValueError("追踪细节必须是映射")
        copied: dict[str, TraceValue] = {}
        for key, value in self.details.items():
            if not isinstance(key, str) or not key:
                raise ValueError("追踪细节键必须是非空字符串")
            if value is not None and not isinstance(value, (str, int)):
                raise ValueError(f"追踪细节值类型不支持：{type(value).__name__}")
            copied[key] = value
        object.__setattr__(self, "details", MappingProxyType(copied))
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("错误信息必须是非空字符串或 None")


class TurnTracer(Protocol):
    """追踪接口；实现负责写入具体日志机制，不得影响运行语义。"""

    def trace(self, record: TraceRecord) -> None: ...
