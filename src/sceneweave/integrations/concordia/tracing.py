"""把领域追踪记录接入 Concordia 的结构化日志与组件日志。

领域模块只依赖 sceneweave.domain.tracing 的接口；本模块负责与锁定
版本的 Concordia 对接。不实现运行控制或界面步数据。
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from concordia.typing.logging import LoggingChannel
from concordia.utils import structured_logging

from sceneweave.domain.identity import CycleId
from sceneweave.domain.tracing import TraceRecord, TurnTracer

_ENGINE_ENTITY = "engine"


class CycleStepIndex:
    """把周期身份映射为结构化日志的 step 序号，按首次出现顺序分配。

    独立于 tracer 实例存在：同一份 SimulationLog 的多个追踪器可以
    共享一个索引，保证周期与 step 一一对应。
    """

    def __init__(self) -> None:
        self._steps: dict[str, int] = {}

    def step_for(self, cycle_id: CycleId) -> int:
        if cycle_id not in self._steps:
            self._steps[cycle_id] = len(self._steps)
        return self._steps[cycle_id]


class SimulationLogTracer(TurnTracer):
    """把 TraceRecord 写入 Concordia SimulationLog 的追踪器。

    周期身份经 CycleStepIndex 映射为结构化日志的 step 序号，角色身份
    映射为 entity_name，追踪阶段映射为 component_name；完整的
    周期/角色/响应/行动身份、剧情时间与细节进入 raw_data，可用
    SimulationLog 的索引与 AIAgentLogInterface 沿因果链检索。
    """

    def __init__(
        self,
        log: structured_logging.SimulationLog,
        *,
        wall_clock: Callable[[], str] | None = None,
        steps: CycleStepIndex | None = None,
    ) -> None:
        self._log = log
        self._steps = steps if steps is not None else CycleStepIndex()
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC).isoformat())

    def trace(self, record: TraceRecord) -> None:
        # 保留字段全部位于顶层，details 一律收进独立命名空间，
        # 任意细节键无法覆盖周期/角色/响应/时间等因果链身份。
        raw: dict[str, Any] = {
            "cycle_id": record.cycle_id,
            "simulation_time_ms": record.simulation_time,
            "role_id": record.role_id,
            "response_id": record.response_id,
            "related_response_id": record.related_response_id,
            "action_ids": list(record.action_ids),
            "event_ids": list(record.event_ids),
            "inputs": list(record.inputs),
            "outputs": list(record.outputs),
            "world_state_version": record.world_state_version,
            "error": record.error,
            "details": dict(record.details),
        }
        self._log.add_entry(
            step=self._steps.step_for(record.cycle_id),
            timestamp=self._wall_clock(),
            entity_name=(record.role_id if record.role_id is not None else _ENGINE_ENTITY),
            component_name=record.stage,
            entry_type="sceneweave",
            summary=record.summary,
            raw_data=raw,
        )

    def step_for(self, cycle_id: CycleId) -> int:
        """返回周期对应的日志 step 序号；未记录过的周期按序分配。"""
        return self._steps.step_for(cycle_id)


def component_logging_channel(
    simulation_log: structured_logging.SimulationLog,
    *,
    entity_name: str,
    component_name: str,
    step: Callable[[], int],
    timestamp: Callable[[], str] | None = None,
) -> LoggingChannel:
    """为 Concordia 组件构造写入同一结构化日志的通道。

    Concordia 组件通过 set_logging_channel 接收该通道（真实接入
    路径见 concordia.typing.entity_component.ComponentWithLogging），
    其输出与引擎、角色的追踪记录进入同一 SimulationLog。step 与
    timestamp 由调用方提供，保持剧情身份与墙钟时间分离。
    """
    clock = timestamp or (lambda: datetime.now(UTC).isoformat())

    def channel(datum: Mapping[str, Any]) -> None:
        simulation_log.add_entry(
            step=step(),
            timestamp=clock(),
            entity_name=entity_name,
            component_name=component_name,
            entry_type="component",
            summary=str(datum.get("Key") or component_name),
            raw_data=dict(datum),
        )

    return channel
