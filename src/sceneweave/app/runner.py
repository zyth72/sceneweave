"""运行编排：headless/UI 两种模式、输出目录管理、录制/重放与错误收尾。

输出目录独立（自动 run-id 或显式指定）；已有目录不覆盖；录制目录与
输出目录不得相同或嵌套。成功运行导出失败以非零退出（RunnerError 链
式保留 IO 原因）；取消/运行异常时导出以 add_note 保留原异常，不掩盖。
取消与异常的概要从引擎公开接口读取真实状态（当前时刻、最后完成周期、
世界版本），不伪造 0 值。组件日志通道在此注入：支持
bind_channel_for 的采样器（真实模型、重放）把 Prompt/Response 连同
作用域/周期/剧情时刻写进同一 SimulationLog。底层同步模型线程不可硬
取消是既存限制（见 integrations/concordia/sampling.py 模块说明）。
"""

import asyncio
import json
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from concordia.typing.logging import LoggingChannel
from concordia.utils import structured_logging

from sceneweave.app.assembly import Assembly, assemble
from sceneweave.app.recording import RecordedRun, write_record_manifest
from sceneweave.config.loader import AssemblyConfig
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.engine.control import RunStatus
from sceneweave.integrations.concordia.sampling import (
    RecordingDirectory,
    SupportsChannelBinding,
)
from sceneweave.integrations.concordia.tracing import (
    SimulationLogTracer,
    component_logging_channel,
)


class RunnerError(RuntimeError):
    """运行编排错误。"""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """运行概要：真实停止原因与最终状态。"""

    stop_reason: str
    final_time_ms: int
    completed_cycles: int
    world_state_version: int
    error: str | None = None


def _alias(a: Path, b: Path) -> bool:
    """两个路径相同或互相嵌套；用于阻止录制/输出互相覆盖。"""
    resolved_a, resolved_b = a.resolve(), b.resolve()
    return (
        resolved_a == resolved_b
        or resolved_a in resolved_b.parents
        or (resolved_b in resolved_a.parents)
    )


def _create_output_directory(explicit: Path | None) -> Path:
    """创建独立输出目录；已有目录不覆盖（mkdir 竞态由 exist_ok=False 防护）。"""
    if explicit is not None:
        if explicit.exists():
            raise RunnerError(f"输出目录已存在，不得覆盖：{explicit}")
        explicit.mkdir(parents=True)
        return explicit
    directory = Path("sceneweave-runs") / f"run-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _prepare_directories(explicit: Path | None, record: Path | None) -> Path:
    """预检并创建输出/录制目录；相同或嵌套路径明确拒绝。"""
    if record is not None and record.exists():
        raise RunnerError(f"录制目录已存在：{record}")
    if explicit is not None and record is not None and _alias(explicit, record):
        raise RunnerError(f"录制目录与输出目录路径相同或嵌套：{record} / {explicit}")
    out_dir = _create_output_directory(explicit)
    if record is not None:
        if _alias(out_dir, record):
            raise RunnerError(f"录制目录与输出目录路径相同或嵌套：{record} / {out_dir}")
        try:
            record.mkdir(parents=True)
        except OSError as error:
            raise RunnerError(f"录制目录创建失败：{error}") from error
    return out_dir


class _RunTracer(TurnTracer):
    """把 TraceRecord 同时写入 SimulationLog 与内存列表。"""

    def __init__(self) -> None:
        self.log: Any = structured_logging.SimulationLog()  # type: ignore[no-untyped-call]
        self._log_tracer = SimulationLogTracer(self.log)
        self.records: list[TraceRecord] = []

    def trace(self, record: TraceRecord) -> None:
        self.records.append(record)
        self._log_tracer.trace(record)

    def component_channel(self, scope: str, component: str) -> LoggingChannel:
        """为采样组件构造写入同一日志的通道；身份取最近一条追踪记录。

        step 用既存 SimulationLogTracer 的周期索引（最近记录的周期），
        不用常量 0；剧情时刻进 raw_data，timestamp 保持真实墙钟，
        不以墙钟冒充剧情时间。
        """

        # 在请求开始时捕获，其他角色并行完成的记录不能替换本次身份。
        last = self.records[-1] if self.records else None
        step_index = self._log_tracer.step_for(last.cycle_id) if last is not None else 0

        def step() -> int:
            return step_index

        def extra() -> dict[str, Any]:
            role_id = None
            for prefix in ("role:", "translator:", "world:check:"):
                if scope.startswith(prefix):
                    role_id = scope.removeprefix(prefix)
                    break
            return {
                "scope": scope,
                "cycle_id": last.cycle_id if last is not None else None,
                "simulation_time_ms": last.simulation_time if last is not None else None,
                "role_id": role_id,
                "response_id": last.response_id
                if last is not None and last.role_id == role_id
                else None,
                "action_ids": list(last.action_ids) if last is not None else [],
                "event_ids": list(last.event_ids) if last is not None else [],
                "world_state_version": last.world_state_version if last is not None else None,
            }

        return component_logging_channel(
            self.log,
            entity_name=scope,
            component_name=component,
            step=step,
            extra=extra,
        )


def bind_component_channels(
    factory: Callable[[str], Any], tracer: _RunTracer
) -> Callable[[str], Any]:
    """运行层包装：为支持通道绑定的采样器注入同一日志的组件通道。

    采样器本身不创建日志；录制/重放包装在本包装之外叠加时，绑定
    仍作用于内层真实采样器或重放器，两种模式组件日志都可查询。
    """

    def bound(scope: str) -> Any:
        sampler = factory(scope)
        if isinstance(sampler, SupportsChannelBinding):

            def channel_for(component: str, bound_scope: str = scope) -> LoggingChannel:
                return tracer.component_channel(bound_scope, component)

            sampler.bind_channel_for(channel_for)
        return sampler

    return bound


def _summary_json(summary: RunSummary) -> dict[str, Any]:
    return {
        "stop_reason": summary.stop_reason,
        "final_time_ms": summary.final_time_ms,
        "completed_cycles": summary.completed_cycles,
        "world_state_version": summary.world_state_version,
        "error": summary.error,
    }


def _export(
    directory: Path,
    tracer: _RunTracer,
    summary: RunSummary,
    snapshot_json: str | None,
) -> None:
    """导出日志/概要/快照；IO 失败向上抛出（不静默吞错）。"""
    failures: list[OSError] = []
    for filename, content in (
        ("simulation_log.json", tracer.log.to_json),
        ("summary.json", lambda: json.dumps(_summary_json(summary), ensure_ascii=False, indent=2)),
        ("error_snapshot.json", lambda: snapshot_json),
    ):
        try:
            text = content()
            if text is not None:
                (directory / filename).write_text(text, encoding="utf-8")
        except OSError as error:
            error.add_note(f"导出文件：{directory / filename}")
            failures.append(error)
    if failures:
        for other in failures[1:]:
            failures[0].add_note(str(other))
        raise failures[0]


def _finish_recording(
    recorder: RecordingDirectory | None,
    record_directory: Path | None,
    config: AssemblyConfig,
    out_dir: Path,
    original: BaseException | None,
) -> None:
    """成功、失败和取消都关闭文件并保留清单；清理错误不替换原异常。"""
    if recorder is None or record_directory is None:
        return
    failures: list[Exception] = []
    try:
        recorder.close()
    except Exception as error:
        failures.append(error)
    try:
        write_record_manifest(record_directory, config, out_dir.name, out_dir)
    except Exception as error:
        failures.append(error)
    if failures:
        if original is not None:
            for failure in failures:
                original.add_note(f"录制收尾失败：{failure}")
        else:
            raise RunnerError(f"录制收尾失败：{failures[0]}") from failures[0]


def _safe_export(
    directory: Path,
    tracer: _RunTracer,
    summary: RunSummary,
    snapshot_json: str | None,
    original: BaseException | None,
) -> None:
    """导出辅助：有原始异常时以 add_note 保留，不掩盖。"""
    try:
        _export(directory, tracer, summary, snapshot_json)
    except OSError as export_error:
        if original is not None:
            original.add_note(f"输出导出失败：{export_error}")
        else:
            raise RunnerError(f"输出导出失败：{export_error}") from export_error


def _interrupt_summary(
    assembly: Assembly | None, stop_reason: str, error: str | None
) -> RunSummary:
    """取消/异常时的真实概要：从公开接口读当前时刻与已完成周期。"""
    if assembly is None:
        return RunSummary(
            stop_reason=stop_reason,
            final_time_ms=0,
            completed_cycles=0,
            world_state_version=0,
            error=error,
        )
    scheduler = assembly.scheduler
    completed = scheduler.last_report.step if scheduler.last_report is not None else 0
    return RunSummary(
        stop_reason=stop_reason,
        final_time_ms=scheduler.current_time,
        completed_cycles=completed,
        world_state_version=assembly.world.state_version,
        error=error,
    )


def _interrupt_snapshot(assembly: Assembly | None, error: str | None) -> str | None:
    """取消/异常时的引擎公开快照；如实状态，未装配时无可导出。"""
    if assembly is None:
        return None
    return assembly.scheduler.snapshot(error=error).to_json()


async def run_headless(
    config: AssemblyConfig,
    sampler_factory: Any,
    *,
    output_directory: Path | None = None,
    record_directory: Path | None = None,
    replay_run: RecordedRun | None = None,
    should_end: Callable[[RunStatus], bool] | None = None,
) -> tuple[RunSummary, Path]:
    """无 UI 运行；返回 (概要, 输出目录)。

    record_directory 录制到独立目录；replay_run 用 RecordedRun 的采样器
    （成功导出前检查全部消费）。二者互斥由调用方（CLI）保证。
    """
    if record_directory is not None and replay_run is not None:
        raise RunnerError("录制与重放互斥")
    tracer = _RunTracer()
    out_dir = _prepare_directories(output_directory, record_directory)
    bound_factory = bind_component_channels(sampler_factory, tracer)
    recorder: RecordingDirectory | None = None
    effective_factory: Any
    if record_directory is not None:
        recorder = RecordingDirectory(bound_factory, record_directory, run_id=out_dir.name)
        effective_factory = recorder.for_scope
    elif replay_run is not None:
        # 重放也绑定组件通道：不建真实模型，但每次回放的服务内容
        # （Prompt/Response 原文）仍写入同一日志，可查询。
        effective_factory = bind_component_channels(replay_run.for_scope, tracer)
    else:
        effective_factory = bound_factory
    assembly: Assembly | None = None
    try:
        assembly = assemble(config, effective_factory, tracer=tracer, should_end=should_end)
        result = await assembly.scheduler.run()
        if replay_run is not None:
            replay_run.assert_fully_consumed()
        summary = RunSummary(
            stop_reason=result.reason.value,
            final_time_ms=result.final_time,
            completed_cycles=result.completed_cycles,
            world_state_version=assembly.world.state_version,
            error=result.error,
        )
        snapshot_json = result.snapshot.to_json() if result.snapshot is not None else None
        try:
            _export(out_dir, tracer, summary, snapshot_json)
        except OSError as export_error:
            detail = f"运行错误：{result.error}；" if result.error else ""
            raise RunnerError(f"{detail}输出导出失败：{export_error}") from export_error
        return summary, out_dir
    except asyncio.CancelledError as error:
        _safe_export(
            out_dir,
            tracer,
            _interrupt_summary(assembly, "cancelled", None),
            _interrupt_snapshot(assembly, None),
            error,
        )
        raise
    except BaseException as error:
        _safe_export(
            out_dir,
            tracer,
            _interrupt_summary(assembly, "error", str(error)),
            _interrupt_snapshot(assembly, str(error)),
            error,
        )
        raise
    finally:
        _finish_recording(recorder, record_directory, config, out_dir, sys.exception())
