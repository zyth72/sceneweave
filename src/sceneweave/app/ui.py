"""UI 运行模式：真实 SimulationServer + StepControllerGate + StepData 广播。

初始暂停在首次事件前（无模型调用）；单步完成一个完整事件周期；
play 连续运行；stop/取消如实中止。每个已完成周期经 broadcast_report
广播一次（下一周期边界等待许可前），运行结束由 broadcast_final 广播
真实最终状态（最终时刻/错误/停止原因/版本），不复播旧周期、不重复
广播完成。广播同时被 SnapshotSimulationServer 捕获为只读历史，页面
经 /run-snapshot 查看完整本次运行与解引用的模型输入输出。运行结束后
页面保留到回车/EOF/Ctrl+C；programmatic 调用用 wait_after_finish=False。
所有资源（服务器/门/装配）都在 try/finally 清理；Concordia 的 stop()
只 shutdown 不 server_close，这里补上 socket 关闭与线程收尾，且未成功
start 的服务器不做 shutdown（会永久阻塞）。
"""

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from concordia.utils.simulation_server import SimulationServer

from sceneweave.app.assembly import Assembly, assemble
from sceneweave.app.page import PAGE
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import (
    RunSummary,
    _export,
    _finish_recording,
    _interrupt_snapshot,
    _interrupt_summary,
    _prepare_directories,
    _run_audit,
    _RunTracer,
    _safe_export,
    bind_component_channels,
)
from sceneweave.app.ui_snapshot import RunSnapshotHistory, SnapshotSimulationServer
from sceneweave.audit.session import AuditSession
from sceneweave.config.loader import AssemblyConfig
from sceneweave.engine.control import RunStatus
from sceneweave.engine.report import CycleReport
from sceneweave.engine.scheduler import RunStopReason
from sceneweave.integrations.concordia.control import (
    StepControllerGate,
    broadcast_final,
    broadcast_report,
    records_tuple,
)
from sceneweave.integrations.concordia.sampling import RecordingDirectory


class UIRunnerError(RuntimeError):
    """UI 运行编排错误。"""


def _validate_port(port: int) -> int:
    """端口类型与范围校验；先于任何文件/模型/服务器副作用。"""
    if isinstance(port, bool) or not isinstance(port, int):
        raise UIRunnerError(f"端口必须是整数，得到：{port!r}")
    if not 0 <= port <= 65535:
        raise UIRunnerError(f"端口超出范围 [0, 65535]：{port}")
    return port


class _ManagedServer:
    """包装 SnapshotSimulationServer：暴露实际端口与可靠的资源关闭。

    端口传 0 时由系统分配，绑定后从 socket 读实际端口。Concordia 的
    stop() 只调 shutdown() 并把内部引用置空；这里先取引用再停止，
    最后 server_close() 关闭 socket。请求线程设为 daemon 且不 join
    （SSE 连接可能长期占用），未成功 start 的实例只释放控制器。
    """

    def __init__(
        self,
        html: str,
        port: int,
        *,
        history: RunSnapshotHistory,
        log_provider: Callable[[], Any],
        audit_state: AuditSession | None = None,
    ) -> None:
        self._server = SnapshotSimulationServer(
            port=port,
            html_content=html,
            history=history,
            log_provider=log_provider,
            audit_state=audit_state.state if audit_state is not None else None,
        )
        self._actual_port: int | None = None
        self._started = False

    @property
    def server(self) -> SimulationServer:
        return self._server

    @property
    def actual_port(self) -> int:
        if self._actual_port is None:
            raise UIRunnerError("服务器尚未启动")
        return self._actual_port

    @property
    def address(self) -> str:
        return f"http://127.0.0.1:{self.actual_port}"

    def start(self) -> None:
        self._server.start()
        self._started = True
        underlying = self._server._server  # noqa: SLF001 框架未暴露实际端口
        if underlying is None:
            raise UIRunnerError("服务器未能启动")
        threading_server = cast("Any", underlying)
        threading_server.daemon_threads = True  # SSE 长连接不阻塞关闭
        threading_server.block_on_close = False
        raw = underlying.server_address
        self._actual_port = raw[1] if isinstance(raw, tuple) and len(raw) >= 2 else raw

    def close(self) -> None:
        # 先释放可能阻塞在控制器条件变量上的许可等待线程。
        self._server.step_controller.stop()
        underlying = self._server._server  # noqa: SLF001
        if not self._started:
            if underlying is not None:
                underlying.server_close()
            return  # 未 serve_forever 的实例 shutdown() 会永久阻塞
        thread = self._server._server_thread  # noqa: SLF001
        try:
            self._server.stop()  # shutdown() 并把内部引用置空
        finally:
            if underlying is not None:
                underlying.server_close()  # Concordia 缺失的 socket 关闭
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)


class _BroadcastingGate:
    """门包装：周期边界等待许可前广播刚完成的周期报告。

    引擎在初始化与每个周期前等待许可；本包装在等待前把上一个已完成
    周期（如有）广播一次——每个周期恰好广播一次，运行结束的最后
    周期由 broadcast_final 以真实最终状态收尾。
    """

    def __init__(
        self,
        inner: StepControllerGate,
        report_of: Callable[[], CycleReport | None],
        server: SimulationServer,
    ) -> None:
        self._inner = inner
        self._report_of = report_of
        self._server = server
        self._last_broadcast_step = 0

    @property
    def controller(self) -> Any:
        return self._inner.controller

    def should_stop(self) -> bool:
        return self._inner.should_stop()

    def broadcast_completed(self) -> None:
        report = self._report_of()
        if report is not None and report.step > self._last_broadcast_step:
            broadcast_report(self._server, report)
            self._last_broadcast_step = report.step

    async def wait_for_permission(self) -> bool:
        self.broadcast_completed()
        return await self._inner.wait_for_permission()


async def run_ui(
    config: AssemblyConfig,
    sampler_factory: Any,
    *,
    port: int = 8080,
    output_directory: Path | None = None,
    record_directory: Path | None = None,
    replay_run: RecordedRun | None = None,
    should_end: Callable[[RunStatus], bool] | None = None,
    wait_after_finish: bool = True,
    on_server_ready: Callable[[str, int], None] | None = None,
    audit: AuditSession | None = None,
) -> tuple[RunSummary, Path]:
    """UI 模式运行；返回 (概要, 输出目录)。

    on_server_ready(address, port) 在服务器启动后回调（端口 0 时获得
    实际端口）。wait_after_finish=False 用于 programmatic 测试，运行
    结束后立即关闭服务器。
    """
    _validate_port(port)
    if record_directory is not None and replay_run is not None:
        raise UIRunnerError("录制与重放互斥")
    tracer = _RunTracer()
    history = RunSnapshotHistory()
    out_dir = _prepare_directories(output_directory, record_directory)
    managed = _ManagedServer(
        PAGE,
        port,
        history=history,
        log_provider=lambda: tracer.log,
        audit_state=audit,
    )
    assembly: Assembly | None = None
    gate: _BroadcastingGate | None = None
    recorder: RecordingDirectory | None = None
    exported = False

    try:
        managed.start()
        if on_server_ready is not None:
            on_server_ready(managed.address, managed.actual_port)

        controller = managed.server.step_controller
        inner_gate = StepControllerGate(controller)

        def report_of() -> CycleReport | None:
            if assembly is None:
                return None
            return assembly.scheduler.last_report

        gate = _BroadcastingGate(inner_gate, report_of, managed.server)
        effective_factory: Any = bind_component_channels(sampler_factory, tracer)
        if record_directory is not None:
            recorder = RecordingDirectory(effective_factory, record_directory, run_id=out_dir.name)
            effective_factory = recorder.for_scope
        elif replay_run is not None:
            effective_factory = bind_component_channels(replay_run.for_scope, tracer)
        assembly = assemble(
            config, effective_factory, tracer=tracer, should_end=should_end, gate=gate
        )
        result = await assembly.scheduler.run()
        if replay_run is not None:
            replay_run.assert_fully_consumed()
        # 时限/自然结束可直接返回，不经过下一许可门；补齐最后的真实周期。
        gate.broadcast_completed()

        # 真实最终状态：最后周期步序、最终时刻、停止原因/错误、行动记录。
        broadcast_final(
            managed.server,
            result,
            assembly.scheduler.last_report,
            records=records_tuple(assembly.scheduler.action_records()),
            diagnostic=assembly.world.diagnostic_snapshot(),
        )

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
            raise UIRunnerError(f"{detail}输出导出失败：{export_error}") from export_error
        exported = True
        incomplete = result.error is not None or result.reason is RunStopReason.ERROR
        await _run_audit(audit, config, tracer, summary, out_dir, incomplete=incomplete)
        if wait_after_finish:
            try:
                input("按回车退出并关闭服务器...")
            except EOFError, KeyboardInterrupt:
                pass  # 运行已结束，回车/EOF/Ctrl+C 都退出查看。
        return summary, out_dir
    except asyncio.CancelledError as error:
        if not exported:
            _safe_export(
                out_dir,
                tracer,
                _interrupt_summary(assembly, "cancelled", None),
                _interrupt_snapshot(assembly, None),
                error,
            )
        if audit is not None:
            audit.state.mark_cancelled()
        raise
    except BaseException as error:
        partial_summary = _interrupt_summary(assembly, "error", str(error))
        _safe_export(
            out_dir,
            tracer,
            partial_summary,
            _interrupt_snapshot(assembly, str(error)),
            error,
        )
        try:
            await _run_audit(audit, config, tracer, partial_summary, out_dir, incomplete=True)
        except Exception:  # noqa: BLE001 审计绝不能替换模拟异常
            pass
        raise
    finally:
        if gate is not None:
            gate.controller.stop()  # 先释放许可等待线程
        original = sys.exception()
        try:
            managed.close()
        except Exception as close_error:
            if original is not None:
                original.add_note(f"UI 资源关闭失败：{close_error}")
            else:
                raise
        finally:
            _finish_recording(
                recorder, record_directory, config, out_dir, sys.exception() or original
            )
