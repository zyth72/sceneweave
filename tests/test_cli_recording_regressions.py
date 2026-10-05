"""录放校验实际输入、完整作用域、原始错误及无剩余调用。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sceneweave.app.recording import RecordedRun, write_record_manifest
from sceneweave.config.loader import load_assembly_config
from sceneweave.integrations.concordia.sampling import (
    RecordedModelError,
    RecordingDirectory,
    ReplayMismatchError,
    scope_file,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/configs/run.json"


class Echo:
    async def sample(self, prompt: str, *, component: str) -> str:
        if prompt == "fail":
            raise ValueError("原始失败")
        return f"{component}:{prompt}"


def recorded(tmp_path: Path, *, failure: bool = False) -> tuple:
    config = load_assembly_config(EXAMPLE)
    directory = tmp_path / "recording"
    with RecordingDirectory(lambda scope: Echo(), directory, run_id="recorded-run") as record:
        sampler = record.for_scope("role:lin")
        if failure:
            with pytest.raises(ValueError, match="原始失败"):
                asyncio.run(sampler.sample("fail", component="system1"))
        else:
            asyncio.run(sampler.sample("A", component="system1"))
        record.for_scope("world:progress")  # 被实例化但无需采样的作用域也属于此次运行。
    write_record_manifest(directory, config, "recorded-run")
    return config, directory


def test_recording_relocated_configuration_matches_by_content(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path)
    replay = RecordedRun(directory, replace(config, scene_path=tmp_path / "moved-scene.json"))
    result = asyncio.run(replay.for_scope("role:lin").sample("A", component="system1"))
    assert result == "system1:A"
    replay.for_scope("world:progress")
    replay.assert_fully_consumed()


@pytest.mark.parametrize("change", ["background", "scene", "event", "interval"])
def test_replay_rejects_changed_actual_input(tmp_path: Path, change: str) -> None:
    config, directory = recorded(tmp_path)
    if change == "background":
        config = replace(
            config, roles=(replace(config.roles[0], background="新的私密背景"), *config.roles[1:])
        )
    elif change == "scene":
        config = replace(config, scene=replace(config.scene, description="新的客观场景"))
    elif change == "event":
        config = replace(
            config, initial_events=(replace(config.initial_events[0], description="另一事件"),)
        )
    else:
        config = replace(config, run=replace(config.run, introspection_interval_ms=2000))
    with pytest.raises(ReplayMismatchError, match="配置或场景"):
        RecordedRun(directory, config)


def test_unrequested_scope_cannot_hide_remaining_calls(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path)
    replay = RecordedRun(directory, config)
    replay.for_scope("world:progress")
    with pytest.raises(ReplayMismatchError, match="未回放"):
        replay.assert_fully_consumed()


def test_unrequested_empty_scope_is_still_detected(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path)
    replay = RecordedRun(directory, config)
    asyncio.run(replay.for_scope("role:lin").sample("A", component="system1"))
    with pytest.raises(ReplayMismatchError, match="未被请求"):
        replay.assert_fully_consumed()


@pytest.mark.parametrize("change", ["extra", "missing", "run_id"])
def test_replay_checks_all_record_files_before_first_call(tmp_path: Path, change: str) -> None:
    config, directory = recorded(tmp_path)
    path = scope_file(directory, "world:progress")
    if change == "extra":
        scope_file(directory, "unknown").write_bytes(path.read_bytes())
    elif change == "missing":
        path.unlink()
    else:
        path.write_text(path.read_text().replace("recorded-run", "other-run"))
    with pytest.raises(ReplayMismatchError):
        RecordedRun(directory, config)


def test_replay_failure_preserves_recorded_error_and_consumption(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path, failure=True)
    replay = RecordedRun(directory, config)
    with pytest.raises(RecordedModelError, match="原始失败") as captured:
        asyncio.run(replay.for_scope("role:lin").sample("fail", component="system1"))
    assert captured.value.error_type == "builtins.ValueError"
    replay.for_scope("world:progress")
    replay.assert_fully_consumed()


def test_mismatched_request_does_not_consume_recording(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path)
    replay = RecordedRun(directory, config)
    sampler = replay.for_scope("role:lin")
    with pytest.raises(ReplayMismatchError, match="不一致"):
        asyncio.run(sampler.sample("wrong", component="system1"))
    assert asyncio.run(sampler.sample("A", component="system1")) == "system1:A"
    replay.for_scope("world:progress")
    replay.assert_fully_consumed()


def test_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    config, directory = recorded(tmp_path)
    original = (directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_record_manifest(directory, config, "second-run")
    assert (directory / "manifest.json").read_bytes() == original
    assert json.loads(original)["run_id"] == "recorded-run"


def test_cli_full_record_replay_private_scopes_and_remaining_calls(tmp_path: Path) -> None:
    from sceneweave.__main__ import main

    recording = tmp_path / "cli-record"
    out = tmp_path / "record-out"
    assert (
        main(
            [
                str(EXAMPLE),
                "--scripted",
                str(EXAMPLE.parent / "scripted.json"),
                "--record",
                str(recording),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    original_summary = json.loads((out / "summary.json").read_text())
    assert (
        original_summary["stop_reason"],
        original_summary["final_time_ms"],
        original_summary["completed_cycles"],
    ) == ("time_limit", 8000, 5)
    config = load_assembly_config(EXAMPLE)
    manifest = json.loads((recording / "manifest.json").read_text())
    assert "world:progress" in manifest["scopes"]
    for scope in manifest["scopes"]:
        body = scope_file(recording, scope).read_text()
        if scope != "role:lin":
            assert config.roles[0].background not in body
        if scope != "role:shen":
            assert config.roles[1].background not in body
    replay_out = tmp_path / "replay-out"
    assert main([str(EXAMPLE), "--replay", str(recording), "--output", str(replay_out)]) == 0
    assert json.loads((replay_out / "summary.json").read_text()) == original_summary
    # 末尾多出合法调用也不能静默忽略；成功摘要不能在核对之前写出。
    path = scope_file(recording, "world:check:lin")
    path.write_text(path.read_text() + path.read_text().splitlines()[-1] + "\n")
    leftover_out = tmp_path / "leftover-out"
    assert main([str(EXAMPLE), "--replay", str(recording), "--output", str(leftover_out)]) != 0
    assert json.loads((leftover_out / "summary.json").read_text())["stop_reason"] == "error"


def test_cli_failure_recording_replays_without_live_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sceneweave import __main__ as cli

    scripts = json.loads((EXAMPLE.parent / "scripted.json").read_text())
    scripts["role:lin"]["system1"] = ["invalid JSON output"]
    script_file = tmp_path / "failure-scripts.json"
    script_file.write_text(json.dumps(scripts))
    recording = tmp_path / "recording"
    failed_out = tmp_path / "failed-out"
    assert (
        cli.main(
            [
                str(EXAMPLE),
                "--scripted",
                str(script_file),
                "--record",
                str(recording),
                "--output",
                str(failed_out),
            ]
        )
        == 1
    )
    original = json.loads((failed_out / "summary.json").read_text())
    assert original["stop_reason"] == "error"
    assert original["error"]
    assert (failed_out / "error_snapshot.json").is_file()
    assert (recording / "manifest.json").is_file()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("replay must never construct a live factory")

    monkeypatch.setattr(cli, "build_live_factory", forbidden)
    replay_out = tmp_path / "replay-out"
    assert cli.main([str(EXAMPLE), "--replay", str(recording), "--output", str(replay_out)]) == 1
    replay_summary = json.loads((replay_out / "summary.json").read_text())
    assert replay_summary["error"] == original["error"]
    assert (replay_out / "error_snapshot.json").is_file()


def test_cli_replay_config_mismatch_is_preflight_error(tmp_path: Path) -> None:
    from sceneweave.__main__ import main

    recording = tmp_path / "recording"
    assert (
        main(
            [
                str(EXAMPLE),
                "--scripted",
                str(EXAMPLE.parent / "scripted.json"),
                "--record",
                str(recording),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    data = json.loads(EXAMPLE.read_text())
    data["scene_path"] = str(EXAMPLE.parent / "scene.json")
    data["roles"][0]["background"] = "改变了角色的私人认知起点"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data))
    output = tmp_path / "changed-out"
    assert main([str(changed), "--replay", str(recording), "--output", str(output)]) != 0
    assert not output.exists()
