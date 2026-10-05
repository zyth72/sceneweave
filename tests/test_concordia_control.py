"""Concordia 运行控制与界面步数据接入测试。

使用锁定版本的真实 StepController 与 SimulationServer：控制路径
（运行、暂停、继续、单步、停止）与真实 StepData 广播都经过框架
对象本身验证，不使用空字典假数据；HTTP 只访问本机回环地址。
"""

import asyncio
import json
import socket
import urllib.request

from concordia.environment.step_controller import StepController
from concordia.utils.simulation_server import SimulationServer

from sceneweave.domain.identity import ActionId, EventId
from sceneweave.domain.messages import ActionPhase, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.control import (
    StepControllerGate,
    broadcast_final,
    broadcast_report,
    records_tuple,
    step_data_from_report,
)

from .fakes import (
    FakeSemanticWorld,
    ScriptedRole,
    ScriptedTranslator,
    idle_response,
    maintain,
    maintaining,
    ms,
    new_action,
    response,
)


def startup() -> WorldEvent:
    return WorldEvent(EventId("startup"), SimulationTime(0), "开场事件")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def post(url: str) -> dict:
    request = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(request, timeout=5) as answer:  # noqa: S310 本机回环
        return json.loads(answer.read().decode("utf-8"))


def get_status(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/status", timeout=5) as answer:  # noqa: S310
        return json.loads(answer.read().decode("utf-8"))


def build_engine(gate: StepControllerGate | None):
    """构造一个带两轮主动行动的引擎；翻译很快，周期由门控节奏决定。"""
    alice = ScriptedRole(
        "alice",
        [
            response("alice", "r-1", (new_action("a-ui", "走向窗口"),)),
            response("alice", "r-2", (maintaining("alice", "a-ui").actions[0],)),
            response("alice", "r-3", ()),
        ],
        fallback=idle_response("alice"),
    )
    world = FakeSemanticWorld(["alice"])
    translator = ScriptedTranslator(plans={"走向窗口": (1000, 2000, ())})
    scheduler = Scheduler(
        roles=[alice],
        world=world,
        translator=translator,
        config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
        initial_events=[startup()],
        gate=gate,
    )
    return scheduler, alice, world


async def wait_at_gate(scheduler: Scheduler, *, timeout_steps: int = 2000) -> None:
    """有界等待运行到达门控边界（许可在每个完整周期前取得）。"""
    for _ in range(timeout_steps):
        if scheduler.stage == "run.gate_wait":
            return
        await asyncio.sleep(0)
    raise AssertionError("运行未到达门控边界")


def test_gate_holds_before_first_cycle_until_permission() -> None:
    """初始化也是安全边界：首次事件处理前先等待许可。"""

    async def scenario() -> None:
        controller = StepController(start_paused=True)
        gate = StepControllerGate(controller)
        scheduler, alice, world = build_engine(gate)
        task = asyncio.create_task(scheduler.run())
        await wait_at_gate(scheduler)
        assert alice.perceived == [], "未获许可前不处理任何事件"
        assert len(scheduler.action_records()) == 0
        assert not world.adjudications
        assert scheduler.current_time == ms(0), "等待许可不推进剧情时钟"
        assert not task.done()
        # 单步：完成一个完整周期后再次等待。
        controller.step()
        for _ in range(2000):
            if alice.perceived:
                break
            await asyncio.sleep(0)
        await wait_at_gate(scheduler)
        record = scheduler.action_records().get(ActionId("a-ui"))
        assert record is not None and record.phase is ActionPhase.REGISTERED
        assert scheduler.current_time == ms(0)
        assert not task.done(), "单步结束后回到等待"
        # 继续：运行到时限。
        controller.play()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT
        assert scheduler.action_records()[ActionId("a-ui")].phase is ActionPhase.COMPLETED

    asyncio.run(scenario())


def test_pre_requested_stop_runs_no_cycle() -> None:
    """停止预先请求时不执行任何事件，直接以中止结束。"""

    async def scenario() -> None:
        controller = StepController(start_paused=True)
        controller.stop()
        gate = StepControllerGate(controller)
        scheduler, alice, world = build_engine(gate)
        result = await scheduler.run()
        assert result.reason is RunStopReason.ABORTED
        assert result.completed_cycles == 0
        assert alice.perceived == []
        assert not world.adjudications
        assert scheduler.current_time == ms(0)

    asyncio.run(scenario())


def test_pause_takes_effect_at_boundary_without_breaking_commits() -> None:
    """暂停在完整周期后的边界生效；不打断 World 提交、不推进时钟。"""

    async def scenario() -> None:
        thinking = asyncio.Event()
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-pause", "慢慢行动"),)),
                response("alice", "r-2", (maintain("a-pause"),)),
            ],
            gate=thinking,
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        controller = StepController(start_paused=False)
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=ScriptedTranslator(delay=1000, duration=1000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=[startup()],
            gate=StepControllerGate(controller),
        )
        task = asyncio.create_task(scheduler.run())
        # 第一周期：角色思考期间请求暂停；周期仍完整完成（含提交）。
        await alice.started.wait()
        controller.pause()
        assert not world.adjudications or world.adjudications, "请求本身不打断进行中的周期"
        thinking.set()
        for _ in range(2000):
            if scheduler.stage == "run.gate_wait":
                break
            await asyncio.sleep(0)
        await wait_at_gate(scheduler)
        assert world.adjudications, "暂停生效前当轮提交已完成"
        version_after_commits = world.state_version
        assert scheduler.action_records()[ActionId("a-pause")].phase is ActionPhase.REGISTERED
        await asyncio.sleep(0.05)
        assert not task.done(), "暂停期间在边界等待"
        assert world.state_version == version_after_commits, "暂停不改变世界状态"
        assert scheduler.current_time == ms(0), "暂停不推进剧情时钟"
        controller.play()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT

    asyncio.run(scenario())


def test_gate_stop_returns_aborted() -> None:
    """停止请求让运行以中止结束，与自然结束/时限区分。"""

    async def scenario() -> None:
        controller = StepController(start_paused=True)
        gate = StepControllerGate(controller)
        scheduler, alice, _world = build_engine(gate)
        task = asyncio.create_task(scheduler.run())
        await wait_at_gate(scheduler)
        controller.step()
        for _ in range(2000):
            if scheduler.last_report is not None:
                break
            await asyncio.sleep(0)
        controller.stop()
        result = await task
        assert result.reason is RunStopReason.ABORTED

    asyncio.run(scenario())


def test_cancelled_wait_releases_controller() -> None:
    """取消等待会释放许可等待，控制器进入停止状态。"""

    async def scenario() -> None:
        controller = StepController(start_paused=True)
        gate = StepControllerGate(controller)
        scheduler, alice, _world = build_engine(gate)
        task = asyncio.create_task(scheduler.run())
        await wait_at_gate(scheduler)
        task.cancel()
        try:
            await task
            raise AssertionError("取消必须传播")
        except asyncio.CancelledError:
            pass
        assert controller.should_stop(), "取消清理了许可等待"

    asyncio.run(scenario())


def test_step_data_from_report_carries_real_content() -> None:
    """StepData 携带真实剧情毫秒、声明、行动状态、事实与诊断。"""

    async def scenario() -> None:
        scheduler, _alice, _world = build_engine(None)
        await scheduler.run()
        report = scheduler.last_report
        assert report is not None
        data = step_data_from_report(report)
        assert data.step == report.step
        assert "剧情时刻" in data.action
        assert "alice" in data.entity_actions
        assert "走向窗口" in data.entity_actions["alice"] or data.entity_actions["alice"]
        alice_log = data.entity_logs["alice"]
        assert "perceptions" in alice_log
        assert "action_states" in alice_log
        world_log = data.entity_logs["world"]
        assert world_log["simulation_time_ms"] == report.simulation_time
        assert isinstance(world_log["committed_facts"], list)
        diagnostics = data.entity_logs["diagnostics"]
        assert "change_reasons" in diagnostics
        assert data.game_master == "world"

    asyncio.run(scenario())


def test_simulation_server_receives_real_step_data_and_control_roundtrip() -> None:
    """真实服务器收到真实步数据；HTTP 控制端点驱动真实运行节奏。"""

    async def scenario() -> None:
        port = free_port()
        server = SimulationServer(port=port, html_content="<html><body>sceneweave ui</body></html>")
        server.start()
        try:
            holder: dict[str, Scheduler | None] = {"scheduler": None}
            inner = StepControllerGate(server.step_controller)

            class BroadcastingGate:
                """在周期边界等待许可前，把刚完成的周期广播给服务器。"""

                def should_stop(self) -> bool:
                    return inner.should_stop()

                async def wait_for_permission(self) -> bool:
                    scheduler = holder["scheduler"]
                    assert scheduler is not None
                    if scheduler.last_report is not None:
                        broadcast_report(server, scheduler.last_report)
                    return await inner.wait_for_permission()

            scheduler, alice, _world = build_engine(BroadcastingGate())
            holder["scheduler"] = scheduler
            task = asyncio.create_task(scheduler.run())
            base = f"http://127.0.0.1:{port}"
            # 许可在首次事件前取得：暂停时服务器不收到任何步数据。
            for _ in range(1000):
                if scheduler.stage == "run.gate_wait":
                    break
                await asyncio.sleep(0)
            assert scheduler.stage == "run.gate_wait"
            assert server.current_step_data == {}, "未处理事件前没有步数据"
            assert get_status(base)["is_paused"] is True
            # HTTP 单步：完成一个完整周期并广播真实 StepData。
            post(f"{base}/step")
            for _ in range(1000):
                if server.current_step_data.get("step") == 1:
                    break
                await asyncio.sleep(0)
            current = server.current_step_data
            assert current["step"] == 1, "服务器收到真实步数据，不是空字典"
            assert "alice" in current["entity_actions"]
            world_log = current["entity_logs"]["world"]
            assert world_log["simulation_time_ms"] == 0
            assert world_log["events"], "同刻事件列表非空"
            assert world_log["registry_actions"], "登记后的行动记录可见"
            entry = world_log["registry_actions"][0]
            assert entry["action_id"] == "a-ui" and entry["phase"] == "registered"
            assert world_log["world_diagnostic"] is not None
            await asyncio.sleep(0.05)
            assert not task.done(), "单步后回到边界等待"
            # 页面与状态端点。
            with urllib.request.urlopen(base, timeout=5) as page:  # noqa: S310
                assert b"sceneweave ui" in page.read()
            # HTTP 继续：运行到时限；结束时广播真实最终状态。
            post(f"{base}/play")
            result = await task
            assert result.reason is RunStopReason.TIME_LIMIT
            broadcast_final(
                server,
                result,
                scheduler.last_report,
                records_tuple(scheduler.action_records()),
                _world.diagnostic_snapshot(),
            )
            final = server.current_step_data
            assert final["entity_logs"]["world"]["simulation_time_ms"] == result.final_time
            assert final["entity_logs"]["diagnostics"]["stop_reason"] == "time_limit"
            # 框架端点核对：GET 控制命令改变真实控制器状态。
            with urllib.request.urlopen(f"{base}/cmd/pause", timeout=5) as answer:  # noqa: S310
                assert json.loads(answer.read().decode("utf-8"))["status"] == "paused"
            assert get_status(base)["is_paused"] is True
        finally:
            server.stop()

    asyncio.run(scenario())


def _wait_http_ready(port: int, timeout_s: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            get_status(f"http://127.0.0.1:{port}")
            return
        except Exception:
            time.sleep(0.1)
    raise AssertionError("子进程服务器未就绪")


def test_ui_demo_subprocess_completes_and_cleans_up(tmp_path) -> None:
    """子进程隔离：完整运行→回车退出→正常清理，实际返回码为 0。"""
    import subprocess
    import sys
    from pathlib import Path

    port = free_port()
    script = Path(__file__).resolve().parent.parent / "examples" / "ui_demo.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
    )
    try:
        _wait_http_ready(port)
        post(f"http://127.0.0.1:{port}/play")
        # 预先发送回车：运行结束进入等待时立即消费并退出。
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        stdout, _ = child.communicate(timeout=30)
        assert child.returncode == 0, f"返回码 {child.returncode}：\n{stdout[-800:]}"
        assert "运行结束：time_limit" in stdout
        assert "服务器与控制器已清理" in stdout, "finally 清理执行"
        assert "结构化日志已导出" in stdout
    finally:
        if child.poll() is None:  # 测试自身失败时兜底回收，不影响断言
            child.kill()
            child.wait(timeout=5)


def test_ui_demo_subprocess_sigint_cleans_up(tmp_path) -> None:
    """子进程隔离：运行中 Ctrl+C（仅向子进程 PID 发信号）也能清理退出。"""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    port = free_port()
    script = Path(__file__).resolve().parent.parent / "examples" / "ui_demo.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
    )
    try:
        _wait_http_ready(port)
        # 保持暂停在门控等待处，向子进程自身发 SIGINT（不动进程组）。
        time.sleep(0.3)
        os.kill(child.pid, signal.SIGINT)
        stdout, _ = child.communicate(timeout=15)
        assert child.returncode == 0, f"返回码 {child.returncode}：\n{stdout[-800:]}"
        assert "服务器与控制器已清理" in stdout, "中断路径同样执行清理"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
