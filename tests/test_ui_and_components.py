"""UI 真实 HTTP 与采样组件日志的集成测试。

真实 SimulationServer/StepControllerGate/broadcast_report/broadcast_final
对完整 ModelWorld 装配验证；真实（假后端）LanguageModel 经工厂完整
运行并断言组件日志；不调用真实 API、不读取密钥。全部等待有界。
"""

import asyncio
import json
import socket
import urllib.request
from pathlib import Path

import pytest
from concordia.utils import structured_logging

from sceneweave.app.assembly import build_live_factory, build_scripted_factory
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import RunnerError, run_headless
from sceneweave.app.ui import UIRunnerError, run_ui
from sceneweave.config.loader import load_assembly_config

from .test_cli_headless import SCRIPTS, setup_config

_FAKE_MODEL_TEMPLATE = '''"""测试用假 LanguageModel 后端：委托 ScriptedModel，不访问网络。"""

import asyncio
import json

from concordia.language_model import language_model

from sceneweave.app.scripted import ScriptedModel

SCRIPTS = json.loads({scripts_json!r})

_MARKERS = (
    ("world-check", "客观检查者"),
    ("world-adjudicate", "联合裁决者"),
    ("world-progress", "进展裁决者"),
    ("system1", "第一步"),
    ("system2", "第二步"),
    ("system3", "第三步"),
    ("translator", "行动翻译器"),
)


def _detect_component(prompt):
  for component, marker in _MARKERS:
    if marker in prompt:
      return component
  raise ValueError("未知提示词：" + prompt[:60])


class _FakeModel(language_model.LanguageModel):
  """同步假后端：按作用域的脚本队列与内置世界语义回答。"""

  def __init__(self, scope):
    self._scope = scope
    self._scripted = ScriptedModel(dict(SCRIPTS.get(scope, {{}})))

  def sample_text(self, prompt, **kw):
    component = _detect_component(prompt)
    return asyncio.run(self._scripted.sample(prompt, component=component))

  def sample_choice(self, prompt, responses, **kw):
    return (0, responses[0], {{}})


def factory(scope, options):
  return _FakeModel(scope)
'''


def _install_fake_model(tmp_path: Path, name: str = "fake_live_for_tests") -> None:
    import sys

    module = tmp_path / f"{name}.py"
    module.write_text(
        _FAKE_MODEL_TEMPLATE.format(scripts_json=json.dumps(SCRIPTS)),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 本机回环
        return json.loads(response.read().decode("utf-8"))


def _http_post(url: str) -> dict:
    request = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def test_ui_real_http_initial_pause_step_play_stop(tmp_path: Path) -> None:
    """真实 HTTP：初始暂停无模型调用、单步完整周期、播放到时限、停止收尾。"""
    ready = asyncio.Event()
    address_holder: dict[str, str] = {}

    def on_ready(address: str, port: int) -> None:
        address_holder["url"] = address
        address_holder["port"] = str(port)
        ready.set()

    config = load_assembly_config(setup_config(tmp_path))

    async def scenario() -> tuple:
        return await run_ui(
            config,
            build_scripted_factory(SCRIPTS),
            port=0,
            output_directory=tmp_path / "ui-out",
            wait_after_finish=False,
            on_server_ready=on_ready,
        )

    async def drive() -> tuple:
        task = asyncio.create_task(scenario())
        await asyncio.wait_for(ready.wait(), timeout=5)
        url = address_holder["url"]
        assert int(address_holder["port"]) > 0, "端口 0 时回调实际端口"
        # 初始暂停：无模型调用、步数为 0。
        status = await asyncio.to_thread(_http_get, f"{url}/status")
        assert status["is_paused"] is True
        assert status["current_step"] == 0
        # 单步：完成一个完整事件周期并广播真实步数据。
        await asyncio.to_thread(_http_post, f"{url}/step")
        current = {"current_step": 0}
        for _ in range(200):
            await asyncio.sleep(0.01)
            current = await asyncio.to_thread(_http_get, f"{url}/status")
            if current["current_step"] >= 1:
                break
        assert current["current_step"] >= 1, "单步完成至少一个周期"
        # 再单步：仍然回到暂停，周期恰好 +1。
        await asyncio.to_thread(_http_post, f"{url}/step")
        for _ in range(200):
            await asyncio.sleep(0.01)
            stepped = await asyncio.to_thread(_http_get, f"{url}/status")
            if stepped["current_step"] >= 2:
                break
        assert stepped["current_step"] == current["current_step"] + 1
        assert stepped["is_paused"] is True
        # 播放：连续运行到时限。
        await asyncio.to_thread(_http_post, f"{url}/play")
        return await asyncio.wait_for(task, timeout=30)

    summary, out_dir = asyncio.run(drive())
    assert summary.stop_reason == "time_limit"
    assert summary.completed_cycles >= 2
    assert summary.error is None
    data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["stop_reason"] == "time_limit"
    log = structured_logging.SimulationLog.from_json(
        (out_dir / "simulation_log.json").read_text(encoding="utf-8")
    )
    assert log.entries, "运行日志非空"


def test_ui_stop_via_http_reports_aborted(tmp_path: Path) -> None:
    """停止指令后如实报告 aborted；页面数据广播最终真实状态。"""
    ready = asyncio.Event()
    holder: dict[str, str] = {}

    def on_ready(address: str, port: int) -> None:
        holder["url"] = address
        ready.set()

    config = load_assembly_config(setup_config(tmp_path))

    async def scenario() -> tuple:
        return await run_ui(
            config,
            build_scripted_factory(SCRIPTS),
            port=0,
            output_directory=tmp_path / "ui-stop-out",
            wait_after_finish=False,
            on_server_ready=on_ready,
        )

    async def drive() -> tuple:
        task = asyncio.create_task(scenario())
        await asyncio.wait_for(ready.wait(), timeout=5)
        url = holder["url"]
        # 先单步再停止（暂停状态下停止，确定性触发 aborted）。
        await asyncio.to_thread(_http_post, f"{url}/step")
        for _ in range(200):
            await asyncio.sleep(0.01)
            status = await asyncio.to_thread(_http_get, f"{url}/status")
            if status["current_step"] >= 1:
                break
        assert status["current_step"] >= 1
        await asyncio.to_thread(_http_post, f"{url}/stop")
        return await asyncio.wait_for(task, timeout=15)

    summary, out_dir = asyncio.run(drive())
    assert summary.stop_reason == "aborted"
    assert summary.error is None
    assert summary.completed_cycles >= 1
    data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["stop_reason"] == "aborted"
    assert data["completed_cycles"] >= 1


def test_ui_cancel_keeps_truthful_state_and_closes_socket(tmp_path: Path) -> None:
    """任务取消：概要保留真实时刻/周期/版本，服务器 socket 真正关闭。"""
    ready = asyncio.Event()
    holder: dict[str, str] = {}

    def on_ready(address: str, port: int) -> None:
        holder["url"] = address
        holder["port"] = str(port)
        ready.set()

    config = load_assembly_config(setup_config(tmp_path))

    async def scenario() -> tuple:
        return await run_ui(
            config,
            build_scripted_factory(SCRIPTS),
            port=0,
            output_directory=tmp_path / "ui-cancel-out",
            wait_after_finish=False,
            on_server_ready=on_ready,
        )

    async def drive() -> None:
        task = asyncio.create_task(scenario())
        await asyncio.wait_for(ready.wait(), timeout=5)
        url = holder["url"]
        await asyncio.to_thread(_http_post, f"{url}/step")
        for _ in range(200):
            await asyncio.sleep(0.01)
            status = await asyncio.to_thread(_http_get, f"{url}/status")
            if status["current_step"] >= 1:
                break
        assert status["current_step"] >= 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    asyncio.run(drive())
    out_dir = tmp_path / "ui-cancel-out"
    data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["stop_reason"] == "cancelled"
    assert data["completed_cycles"] >= 1, "取消概要保留真实已完成周期"
    assert data["world_state_version"] >= 1
    # 服务器 socket 已关闭：端口不再接受连接。
    probe = socket.socket()
    try:
        with pytest.raises(OSError):
            probe.settimeout(2)
            probe.connect(("127.0.0.1", int(holder["port"])))
    finally:
        probe.close()


def test_ui_port_conflict_fails_without_deadlock(tmp_path: Path) -> None:
    """端口被占用：立即失败、写错误概要、不挂起不 traceback。"""
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        config = load_assembly_config(setup_config(tmp_path))

        async def scenario() -> tuple:
            return await run_ui(
                config,
                build_scripted_factory(SCRIPTS),
                port=port,
                output_directory=tmp_path / "ui-conflict-out",
                wait_after_finish=False,
            )

        async def conflict() -> None:
            with pytest.raises(OSError, match="Address already in use|EADDRINUSE"):
                await asyncio.wait_for(scenario(), timeout=10)

        asyncio.run(conflict())
        data = json.loads(
            (tmp_path / "ui-conflict-out" / "summary.json").read_text(encoding="utf-8")
        )
        assert data["stop_reason"] == "error"
    finally:
        blocker.close()


def test_ui_port_validation() -> None:
    """端口类型/范围校验先于文件与服务器副作用。"""
    from sceneweave.app.ui import _validate_port

    assert _validate_port(0) == 0
    assert _validate_port(8080) == 8080
    with pytest.raises(UIRunnerError, match="范围"):
        _validate_port(65536)
    with pytest.raises(UIRunnerError, match="范围"):
        _validate_port(-1)
    with pytest.raises(UIRunnerError, match="整数"):
        _validate_port(True)
    with pytest.raises(UIRunnerError, match="整数"):
        _validate_port("8080")  # type: ignore[arg-type]


def test_world_scope_mapping_namespaced() -> None:
    """World 内部作用域映射到文档命名空间，不再出现 world:world-*。"""
    from sceneweave.app.assembly import AssemblyError, world_to_namespaced

    assert world_to_namespaced("world-progress") == "world:progress"
    assert world_to_namespaced("world-adjudicate") == "world:adjudicate"
    assert world_to_namespaced("world-check-lin") == "world:check:lin"
    # 角色身份为 world-progress 等也不与 World 作用域碰撞：角色恒为
    # role:{id} 前缀。
    with pytest.raises(AssemblyError, match="未知的 World 采样作用域"):
        world_to_namespaced("world-unknown")


def _component_entries(log: structured_logging.SimulationLog) -> list[dict]:
    out = []
    for entry in log.entries:
        if entry.entry_type == "component":
            raw = log.reconstruct_value(entry.deduplicated_data)
            out.append(
                {
                    "step": entry.step,
                    "entity": entry.entity_name,
                    "component": entry.component_name,
                    "raw": raw,
                }
            )
    return out


def test_real_model_full_factory_component_logs(tmp_path: Path) -> None:
    """真实（假后端）LanguageModel 经工厂完整运行：组件日志全链覆盖。

    断言 System1/2/3、World 三个作用域与 translator 的采样都以
    entry_type=component 写入同一 SimulationLog；条目携带原文
    Prompt/Response、scope/周期/剧情时刻身份；step 来自周期索引不恒
    为 0；各作用域条目互相独立。
    """
    _install_fake_model(tmp_path)
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_live_factory("fake_live_for_tests:factory", {})

    async def scenario() -> tuple:
        return await run_headless(config, factory, output_directory=tmp_path / "live-out")

    summary, out_dir = asyncio.run(scenario())
    assert summary.stop_reason == "time_limit", f"意外停止：{summary.error}"

    log = structured_logging.SimulationLog.from_json(
        (out_dir / "simulation_log.json").read_text(encoding="utf-8")
    )
    entries = _component_entries(log)
    assert entries, "真实采样必须产生组件日志条目"

    entities = {item["entity"] for item in entries}
    components = {item["component"] for item in entries}
    # 覆盖：角色三个系统、World 三类、翻译器（全部为真实采样调用）。
    assert {"system1", "system2", "system3"} <= components, components
    assert {"world-progress", "world-adjudicate"} <= components, components
    assert "world-check" in components, components
    assert "translator" in components, components
    # 作用域隔离：角色与 World/翻译器互不混叠。
    assert "role:lin" in entities and "role:shen" in entities, entities
    assert "world:progress" in entities and "world:adjudicate" in entities, entities
    assert any(e.startswith("world:check:") for e in entities), entities
    assert any(e.startswith("translator:") for e in entities), entities
    # 每条含原文与因果身份；step 不恒为 0。
    for item in entries:
        assert item["raw"]["Prompt"], "组件日志携带 Prompt 原文"
        assert "Response" in item["raw"] or "ErrorMessage" in item["raw"]
        assert item["raw"]["scope"] == item["entity"]
        assert item["raw"]["cycle_id"], "组件日志携带周期身份"
        assert isinstance(item["raw"]["simulation_time_ms"], int)
    assert any(item["step"] > 0 for item in entries), "step 取自周期索引，不为常量 0"
    # 同一作用域内组件条目区分阶段。
    lin_components = {item["component"] for item in entries if item["entity"] == "role:lin"}
    assert {"system1"} <= lin_components, lin_components


def test_record_and_replay_keep_component_logs_queryable(tmp_path: Path) -> None:
    """录制与重放的组件日志都可查询；重放不建真实模型也有条目。"""
    _install_fake_model(tmp_path)
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_live_factory("fake_live_for_tests:factory", {})

    async def record_run() -> tuple:
        return await run_headless(
            config,
            factory,
            output_directory=tmp_path / "rec-out",
            record_directory=tmp_path / "record",
        )

    summary, _ = asyncio.run(record_run())
    assert summary.stop_reason == "time_limit"

    record_log = structured_logging.SimulationLog.from_json(
        (tmp_path / "rec-out" / "simulation_log.json").read_text(encoding="utf-8")
    )
    record_components = _component_entries(record_log)
    assert record_components, "录制运行（真实采样）产生组件日志"

    replay_run = RecordedRun(tmp_path / "record", config)

    async def replay_run_once() -> tuple:
        return await run_headless(
            config,
            replay_run.for_scope,
            replay_run=replay_run,
            output_directory=tmp_path / "rp-out",
        )

    replay_summary, replay_out = asyncio.run(replay_run_once())
    assert replay_summary.stop_reason == "time_limit"

    replay_log = structured_logging.SimulationLog.from_json(
        (replay_out / "simulation_log.json").read_text(encoding="utf-8")
    )
    replay_components = _component_entries(replay_log)
    assert replay_components, "重放（文本包装，无真实模型）仍可查询组件日志"
    replay_pairs = {(item["entity"], item["component"]) for item in replay_components}
    record_pairs = {(item["entity"], item["component"]) for item in record_components}
    assert record_pairs <= replay_pairs, "重放条目覆盖录制的全部作用域/组件"


def test_export_io_failure_raises_chained_runner_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功运行但导出 IO 失败：RunnerError 链式保留原始 IO 错误。"""

    def broken_to_json(self: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(structured_logging.SimulationLog, "to_json", broken_to_json)
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_scripted_factory(SCRIPTS)

    async def scenario() -> tuple:
        return await run_headless(config, factory, output_directory=tmp_path / "io-out")

    with pytest.raises(RunnerError, match="导出失败") as excinfo:
        asyncio.run(scenario())
    assert isinstance(excinfo.value.__cause__, OSError)


def test_record_directory_alias_rejected(tmp_path: Path) -> None:
    """录制目录与输出目录相同/嵌套被明确拒绝，不互相覆盖。"""
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_scripted_factory(SCRIPTS)
    same = tmp_path / "same-dir"

    async def alias() -> tuple:
        return await run_headless(
            config,
            factory,
            output_directory=same,
            record_directory=same,
        )

    with pytest.raises(RunnerError, match="相同或嵌套"):
        asyncio.run(alias())
    assert not same.exists(), "拒绝时不留下半成品目录"

    nested_record = tmp_path / "out" / "record"

    async def nested() -> tuple:
        return await run_headless(
            config,
            factory,
            output_directory=tmp_path / "out",
            record_directory=nested_record,
        )

    with pytest.raises(RunnerError, match="相同或嵌套"):
        asyncio.run(nested())
    assert not (tmp_path / "out" / "record").exists()
