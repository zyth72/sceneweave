"""最终入口验收：UI 录放、最后周期、取消与错误日志的真实边界。"""

import asyncio
import json
import socket
import urllib.request
from pathlib import Path

import pytest
from concordia.language_model.language_model import LanguageModel
from concordia.utils.simulation_server import SimulationServer

from sceneweave.app.assembly import build_scripted_factory
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import run_headless
from sceneweave.app.scripted import ScriptedModel
from sceneweave.app.ui import run_ui
from sceneweave.config.loader import load_assembly_config
from sceneweave.integrations.concordia.sampling import LanguageModelSampler

from .fakes import s1_continue

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/configs/run.json"


def post(url: str, command: str) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url + command, method="POST", data=b""), timeout=5
    ) as response:
        assert response.status == 200


def test_ui_record_replay_delivers_every_cycle_and_final_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = []
    original_broadcast = SimulationServer.broadcast_step

    def broadcast(server: SimulationServer, data: object) -> None:
        frames.append(data)
        original_broadcast(server, data)

    monkeypatch.setattr(SimulationServer, "broadcast_step", broadcast)
    config = load_assembly_config(EXAMPLE)
    scripts = json.loads((EXAMPLE.parent / "scripted.json").read_text())
    factory = build_scripted_factory(scripts)
    calls = []

    def tracked(scope: str) -> object:
        inner = factory(scope)

        class Observed:
            async def sample(self, prompt: str, *, component: str) -> str:
                calls.append((scope, component))
                return await inner.sample(prompt, component=component)

        return Observed()

    async def drive(*, replay: RecordedRun | None = None) -> tuple:
        ready = asyncio.Event()
        addresses = []

        def on_ready(address: str, port: int) -> None:
            addresses.append(address)
            ready.set()

        task = asyncio.create_task(
            run_ui(
                config,
                tracked if replay is None else replay.for_scope,
                port=0,
                output_directory=tmp_path / ("record-out" if replay is None else "replay-out"),
                record_directory=tmp_path / "record" if replay is None else None,
                replay_run=replay,
                wait_after_finish=False,
                on_server_ready=on_ready,
            )
        )
        await asyncio.wait_for(ready.wait(), 5)
        if replay is None:
            assert calls == [], "初始暂停之前没有采样"
        await asyncio.to_thread(post, addresses[0], "/play")
        return await asyncio.wait_for(task, 15)

    recorded_summary, out = asyncio.run(drive())
    assert recorded_summary.completed_cycles == 5
    cycle_frames = [f for f in frames if "stop_reason" not in f.entity_logs["diagnostics"]]
    assert [f.step for f in cycle_frames] == [1, 2, 3, 4, 5]
    assert cycle_frames[-1].entity_logs["world"]["simulation_time_ms"] == 5000
    final = frames[-1]
    assert final.entity_logs["diagnostics"]["stop_reason"] == "time_limit"
    assert final.entity_logs["world"]["simulation_time_ms"] == 8000
    assert (
        final.entity_logs["world"]["world_diagnostic"]["world_state_version"]
        == recorded_summary.world_state_version
    )
    assert (tmp_path / "record/manifest.json").is_file()
    frames.clear()
    replay = RecordedRun(tmp_path / "record", config)
    replayed_summary, replay_out = asyncio.run(drive(replay=replay))
    assert replayed_summary == recorded_summary
    assert json.loads((out / "summary.json").read_text()) == json.loads(
        (replay_out / "summary.json").read_text()
    )
    replay.assert_fully_consumed()


def test_cancel_export_failure_preserves_cancel_and_writes_other_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_assembly_config(EXAMPLE)
    scripted = build_scripted_factory(json.loads((EXAMPLE.parent / "scripted.json").read_text()))
    started = asyncio.Event()

    def factory(scope: str) -> object:
        inner = scripted(scope)

        class Blocked:
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "system1":
                    started.set()
                    await asyncio.Event().wait()
                return await inner.sample(prompt, component=component)

        return Blocked()

    original_write = Path.write_text

    def write(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "simulation_log.json":
            raise OSError("disk full for log")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)

    async def scenario() -> None:
        task = asyncio.create_task(
            run_headless(
                config,
                factory,
                output_directory=tmp_path / "out",
                record_directory=tmp_path / "record",
            )
        )
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 5)
        assert any("导出失败" in note for note in caught.value.__notes__)

    asyncio.run(scenario())
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["stop_reason"] == "cancelled"
    assert summary["world_state_version"] == 1
    assert (tmp_path / "out/error_snapshot.json").is_file()
    assert (tmp_path / "record/manifest.json").is_file()


def test_ui_start_failure_never_waits_for_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_input(*args: object) -> str:
        raise AssertionError("启动失败不得进入查看等待")

    monkeypatch.setattr("builtins.input", forbidden_input)
    config = load_assembly_config(EXAMPLE)
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    try:

        async def scenario() -> None:
            with pytest.raises(OSError):
                await asyncio.wait_for(
                    run_ui(
                        config,
                        lambda scope: None,
                        port=blocker.getsockname()[1],
                        output_directory=tmp_path / "out",
                    ),
                    5,
                )

        asyncio.run(scenario())
    finally:
        blocker.close()


def test_model_component_error_preserves_backend_exception_when_logging_fails() -> None:
    class FailedModel(LanguageModel):
        def sample_text(self, prompt: str, **kwargs: object) -> str:
            raise ValueError("backend failure")

        def sample_choice(self, prompt: str, responses: list[str], **kwargs: object) -> tuple:
            raise NotImplementedError

    def broken_channel(data: object) -> None:
        raise OSError("logging unavailable")

    sampler = LanguageModelSampler(FailedModel(), channel_for=lambda component: broken_channel)
    with pytest.raises(ValueError, match="backend failure") as caught:
        asyncio.run(sampler.sample("original prompt", component="world-check"))
    assert any("组件错误日志" in note for note in caught.value.__notes__)


def test_explicit_world_script_is_honored_and_exhaustion_is_an_error() -> None:
    sampler = ScriptedModel({"world-check": ['{"passed":false,"reason":"拒绝","action_ids":[]}']})
    raw = asyncio.run(sampler.sample("check", component="world-check"))
    assert json.loads(raw)["passed"] is False
    with pytest.raises(RuntimeError, match="耗尽"):
        asyncio.run(sampler.sample("check again", component="world-check"))


def test_valid_role_identity_cannot_overwrite_ui_world_diagnostics() -> None:
    from dataclasses import replace

    from sceneweave.app.assembly import assemble
    from sceneweave.integrations.concordia.control import final_step_data, step_data_from_report

    config = load_assembly_config(EXAMPLE)
    # 角色 world 和 role:world 同时存在时，别名也必须避让真实角色。
    config = replace(
        config,
        scene=replace(
            config.scene,
            children=tuple(
                replace(obj, id="world")
                if obj.id == "lin"
                else replace(obj, id="role:world")
                if obj.id == "shen"
                else obj
                for obj in config.scene.children
            ),
        ),
        roles=(
            replace(config.roles[0], role_id="world"),
            replace(config.roles[1], role_id="role:world"),
        ),
    )
    scripts = {
        f"role:{role.role_id}": {
            "system1": [s1_continue("观察")] * 5,
            "system2": ['{"intention": "先观察，不主动行动"}'],
            "system3": ["[]"],
        }
        for role in config.roles
    }
    assembly = assemble(config, build_scripted_factory(scripts))
    result = asyncio.run(assembly.scheduler.run())
    report = assembly.scheduler.last_report
    assert report is not None
    data = step_data_from_report(report)
    assert data.entity_logs["world"]["simulation_time_ms"] == report.simulation_time
    assert (
        data.entity_logs["world"]["world_diagnostic"]["world_state_version"]
        == assembly.world.state_version
    )
    assert len(data.entity_actions) == 2
    assert "world" not in data.entity_actions
    assert "role:world" in data.entity_actions
    assert data.entity_logs["role:role:world"]["role_id"] == "world"
    final = final_step_data(result, report, diagnostic=assembly.world.diagnostic_snapshot())
    assert final.entity_logs["world"]["simulation_time_ms"] == result.final_time
    assert final.entity_logs["diagnostics"]["stop_reason"] == "time_limit"
    assert final.entity_actions.keys() == data.entity_actions.keys()
