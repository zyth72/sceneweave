"""Concordia 运行控制与界面步数据接入。

StepControllerGate 把锁定版本的 StepController 适配为引擎运行门：
同步阻塞的 wait_for_step_permission 放到线程中等待，不占用事件循环；
取消时通过 stop() 释放可能阻塞的等待线程。step_data_from_report 把
引擎周期报告转换为真实 StepData：剧情毫秒、同刻全部事件（含行动
节点类型与身份）、完整主动声明、登记后的行动记录、实际行动状态与
角色感知、World 事实与结算结果、诊断变化原因分别呈现，不把调试
内容混入角色感知。final_step_data 用真实结束状态（最终剧情时刻与
停止原因）构造收尾步数据，不复播旧周期报告冒充当前状态。
"""

import asyncio
from collections.abc import Mapping

from concordia.environment.step_controller import StepController, StepData

from sceneweave.domain.adjudication import WorldDiagnostic
from sceneweave.domain.identity import RoleId
from sceneweave.engine.actions import ActionRecord
from sceneweave.engine.report import CycleReport
from sceneweave.engine.scheduler import RunResult


class StepControllerGate:
    """以真实 StepController 控制运行、暂停、继续、单步与停止。

    许可在每个完整周期前由引擎取得；本门只提供等待与停止状态。
    """

    def __init__(self, controller: StepController) -> None:
        self._controller = controller

    @property
    def controller(self) -> StepController:
        return self._controller

    def should_stop(self) -> bool:
        return self._controller.should_stop()

    async def wait_for_permission(self) -> bool:
        task = asyncio.create_task(asyncio.to_thread(self._controller.wait_for_step_permission))
        try:
            return await task
        except asyncio.CancelledError:
            # 释放仍在条件变量上等待的线程；停止是中止运行的明确语义。
            self._controller.stop()
            raise


def _record_entries(records: tuple[ActionRecord, ...]) -> list[dict[str, object]]:
    return [
        {
            "action_id": record.action_id,
            "role_id": record.role_id,
            "phase": record.phase.value,
            "start_ms": record.start,
            "expected_end_ms": record.expected_end,
            "description": record.description,
        }
        for record in records
    ]


def _diagnostic_entries(diagnostic: WorldDiagnostic) -> dict[str, object]:
    return {
        "world_state_version": diagnostic.world_state_version,
        "action_states": [
            {
                "action_id": item.action_id,
                "role_id": item.role_id,
                "state": item.state,
                "detail": item.detail,
            }
            for item in diagnostic.action_states
        ],
        "passive_processes": list(diagnostic.passive_processes),
    }


def _role_entity_keys(role_ids: tuple[RoleId, ...]) -> dict[RoleId, str]:
    """界面保留键不能被合法角色身份覆盖；普通身份保持既有呈现。"""
    reserved = {"world", "diagnostics"}
    used = reserved | set(role_ids)
    result: dict[RoleId, str] = {}
    for role_id in role_ids:
        key = str(role_id)
        if key in reserved:
            key = f"role:{role_id}"
            while key in used:
                key = f"role:{key}"
            used.add(key)
        result[role_id] = key
    return result


def step_data_from_report(report: CycleReport, *, acting_entity: str = "scheduler") -> StepData:
    """把一个完整事件周期的报告转换为界面步数据。

    entity_actions 是各角色的完整主动声明；entity_logs 区分角色可见
    感知、自身行动状态、登记后的行动记录与 World 客观事实、诊断变化
    原因。这些内容只进入界面，不会进入任何角色的感知。
    """
    entity_actions: dict[str, str] = {}
    role_keys = _role_entity_keys(tuple(view.role_id for view in report.roles))
    entity_logs: dict[str, dict[str, object]] = {
        "world": {
            "simulation_time_ms": report.simulation_time,
            "events": [
                {
                    "event_id": event.event_id,
                    "kind": event.kind,
                    "description": event.description,
                }
                for event in report.batch_events
            ],
            "committed_facts": list(report.committed_facts),
            "action_outcomes": [
                {
                    "action_id": outcome.action_id,
                    "result": outcome.result.value,
                    "detail": outcome.detail,
                }
                for outcome in report.action_outcomes
            ],
            "registry_actions": _record_entries(report.registry_actions),
            "world_diagnostic": (
                _diagnostic_entries(report.world_diagnostic)
                if report.world_diagnostic is not None
                else None
            ),
        },
        "diagnostics": {
            "cycle_id": report.cycle_id,
            "change_reasons": {
                action_id: reason for action_id, reason in report.action_changes().items()
            },
        },
    }
    for view in report.roles:
        key = role_keys[view.role_id]
        entity_actions[key] = (
            "；".join(view.declarations) if view.declarations else "（无主动行为声明）"
        )
        entity_logs[key] = {
            "perceptions": list(view.perceptions),
            "action_states": [
                {
                    "action_id": state.action_id,
                    "phase": state.phase.value,
                    "detail": state.detail,
                }
                for state in view.action_states
            ],
        }
        if key != view.role_id:
            entity_logs[key]["role_id"] = view.role_id
    current = "；".join(f"{event.kind} {event.event_id}" for event in report.batch_events)
    if report.introspection:
        current = f"{current}（含周期性内思考）" if current else "周期性内思考"
    action = f"剧情时刻 {report.simulation_time} 毫秒（周期 {report.cycle_id}）：{current}"
    return StepData(
        step=report.step,
        acting_entity=acting_entity,
        action=action,
        entity_actions=entity_actions,
        entity_logs=entity_logs,
        game_master="world",
    )


def final_step_data(
    result: RunResult,
    last_report: CycleReport | None,
    records: tuple[ActionRecord, ...] = (),
    diagnostic: WorldDiagnostic | None = None,
    *,
    acting_entity: str = "scheduler",
) -> StepData:
    """运行结束时的真实最终状态：实际剧情时刻与停止原因。

    不复播旧周期报告：步序沿用最后完成的周期，动作文本与 world 视
    图换成真实的最终时刻、停止原因（与错误信息）。
    """
    step = last_report.step if last_report is not None else 0
    entity_logs: dict[str, dict[str, object]] = {
        "world": {
            "simulation_time_ms": result.final_time,
            "events": [],
            "committed_facts": [],
            "action_outcomes": [],
            "registry_actions": _record_entries(records),
            "world_diagnostic": _diagnostic_entries(diagnostic) if diagnostic else None,
        },
        "diagnostics": {
            "stop_reason": result.reason.value,
            "completed_cycles": result.completed_cycles,
            "error": result.error,
        },
    }
    entity_actions: dict[str, str] = {}
    if last_report is not None:
        role_keys = _role_entity_keys(tuple(view.role_id for view in last_report.roles))
        for view in last_report.roles:
            key = role_keys[view.role_id]
            entity_actions[key] = (
                "；".join(view.declarations) if view.declarations else "（无主动行为声明）"
            )
            if key != view.role_id:
                entity_logs[key] = {"role_id": view.role_id}
    detail = f"运行结束：{result.reason.value}，最终剧情时刻 {result.final_time} 毫秒"
    if result.error:
        detail = f"{detail}；错误：{result.error}"
    return StepData(
        step=step,
        acting_entity=acting_entity,
        action=detail,
        entity_actions=entity_actions,
        entity_logs=entity_logs,
        game_master="world",
    )


def broadcast_report(
    server: object,
    report: CycleReport,
    *,
    acting_entity: str = "scheduler",
) -> StepData:
    """把报告转换并广播到 SimulationServer 的全部 SSE 客户端。"""
    from concordia.utils.simulation_server import SimulationServer  # 局部避免模块级依赖扩散

    if not isinstance(server, SimulationServer):
        raise TypeError("broadcast_report 需要 Concordia SimulationServer")
    data = step_data_from_report(report, acting_entity=acting_entity)
    server.broadcast_step(data)
    return data


def broadcast_final(
    server: object,
    result: RunResult,
    last_report: CycleReport | None,
    records: tuple[ActionRecord, ...] = (),
    diagnostic: WorldDiagnostic | None = None,
) -> StepData:
    """广播真实最终状态并通知完成；供运行结束时的收尾呈现。"""
    from concordia.utils.simulation_server import SimulationServer

    if not isinstance(server, SimulationServer):
        raise TypeError("broadcast_final 需要 Concordia SimulationServer")
    data = final_step_data(result, last_report, records, diagnostic)
    server.broadcast_step(data)
    server.broadcast_completion()
    return data


def records_tuple(records: Mapping) -> tuple[ActionRecord, ...]:  # type: ignore[type-arg]
    """从登记表只读视图构造有序记录元组；供收尾快照使用。"""
    return tuple(sorted(records.values(), key=lambda item: str(item.action_id)))
