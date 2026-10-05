"""角色组件间共享的轮次事实与阶段留痕辅助。"""

import warnings
from dataclasses import dataclass

from sceneweave.domain.identity import ActionId, CycleId, EventId, ResponseId, RoleId
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.tracing import TraceRecord, TraceValue, TurnTracer

# 自身行动摘 要：(行动身份, 阶段, 描述)。
type OwnActionDigest = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """从本轮输入提取的客观事实快照（不含其他角色私有信息）。"""

    cycle_id: CycleId
    role_id: RoleId
    simulation_time: SimulationTime
    event_ids: tuple[EventId, ...]
    introspection: bool
    observations: tuple[str, ...]
    known_actions: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("周期身份必须是非空字符串")
        if not isinstance(self.role_id, str) or not self.role_id:
            raise ValueError("角色身份必须是非空字符串")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        for name in ("event_ids", "observations"):
            items = getattr(self, name)
            if not isinstance(items, tuple):
                raise ValueError(f"{name}必须是元组")
        if type(self.introspection) is not bool:
            raise ValueError("内思考标记必须是布尔值")
        if not isinstance(self.known_actions, tuple) or not all(
            isinstance(item, tuple) and len(item) == 3 for item in self.known_actions
        ):
            raise ValueError("已知行动摘要必须是 (身份, 阶段, 说明) 元组")


def trace_stage(
    tracer: TurnTracer | None,
    stage: str,
    facts: TurnFacts,
    response_id: ResponseId | None,
    summary: str,
    *,
    prompt: str,
    raw_output: str | None,
    details: dict[str, TraceValue] | None = None,
    action_ids: tuple[ActionId, ...] = (),
    error: str | None = None,
) -> None:
    """把一个组件阶段写入与引擎相同的追踪链。

    追踪器自身失败要暴露；已有模型/解析异常时保留原异常，并报告日志失败。
    """
    if tracer is None:
        return
    merged: dict[str, TraceValue] = {"attempt_stage": stage}
    if details:
        merged.update(details)
    try:
        tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=facts.cycle_id,
                simulation_time=facts.simulation_time,
                summary=summary,
                role_id=facts.role_id,
                response_id=response_id,
                action_ids=action_ids,
                event_ids=facts.event_ids,
                inputs=(prompt,),
                outputs=() if raw_output is None else (raw_output,),
                details=merged,
                error=error,
            )
        )
    except Exception as trace_error:
        if error is None:
            raise
        warnings.warn(
            f"阶段 {stage} 的错误日志写入失败：{trace_error}", RuntimeWarning, stacklevel=2
        )
