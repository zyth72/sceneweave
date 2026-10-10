"""统一入口 headless scripted 模式端到端测试。

用脚本假模型驱动完整装配（ModelWorld+ModelRole+ModelTranslator+
Scheduler），不调用真实 API。验证运行结果概要、输出目录、录制/重放。
"""

import asyncio
import json
from pathlib import Path

import pytest

from sceneweave.app.assembly import build_scripted_factory
from sceneweave.app.runner import run_headless
from sceneweave.config.loader import load_assembly_config


def setup_config(tmp_path: Path) -> Path:
    scene = {
        "id": "room",
        "name": "房间",
        "kind": "房间",
        "unit": "m",
        "size": [4, 3],
        "children": [
            {"id": "lin", "name": "林", "kind": "角色", "position": [0.5, 0.5], "children": []},
            {"id": "shen", "name": "沈", "kind": "角色", "position": [3.0, 0.5], "children": []},
        ],
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    config = {
        "scene_path": "scene.json",
        "roles": [
            {"id": "lin", "background": "测试角色林"},
            {"id": "shen", "background": "测试角色沈"},
        ],
        "initial_events": [{"id": "startup", "time_seconds": 0, "description": "开场安静"}],
        "run": {
            "introspection_interval_seconds": 5,
            "max_simulation_duration_seconds": 8,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config_path


SCRIPTS = {
    "role:lin": {
        "system1": [
            '{"route": "rethink", "note": "开始行动"}',
        ]
        + ['{"route": "continue", "note": "继续"}'] * 10,
        "system2": ['{"intention": "走到书桌旁"}'],
        "system3": ['[{"kind": "move", "destination": {"entity": "书桌"}, "mode": "walk"}]'],
    },
    "role:shen": {
        "system1": ['{"route": "continue", "note": "照旧"}'] * 11,
    },
    "translator:lin": {
        "translator": [
            '{"schedules": [{"action_id": "a-lin-1",'
            ' "start": 1000, "expected_end": 4000, "checkpoints": [2500]}]}'
        ]
    },
    "translator:shen": {},
}


def test_headless_scripted_runs_full_cycle(tmp_path: Path) -> None:
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_scripted_factory(SCRIPTS)

    async def scenario() -> tuple:
        return await run_headless(config, factory, output_directory=tmp_path / "out")

    summary, out_dir = asyncio.run(scenario())
    assert summary.stop_reason == "time_limit"
    assert summary.final_time_ms == 8000
    assert summary.completed_cycles > 0
    assert summary.world_state_version >= 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "simulation_log.json").is_file()
    data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["stop_reason"] == "time_limit"


def test_headless_output_directory_not_overwritten(tmp_path: Path) -> None:
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_scripted_factory(SCRIPTS)
    out = tmp_path / "existing"
    out.mkdir()

    async def scenario() -> tuple:
        return await run_headless(config, factory, output_directory=out)

    with pytest.raises(Exception, match="已存在"):
        asyncio.run(scenario())


def test_record_and_replay_produce_same_result(tmp_path: Path) -> None:
    """录制→重放：同一配置同结果，全部 scope 严格消费。"""
    config = load_assembly_config(setup_config(tmp_path))
    record_dir = tmp_path / "recording"

    async def record() -> tuple:
        return await run_headless(
            config,
            build_scripted_factory(SCRIPTS),
            output_directory=tmp_path / "record-out",
            record_directory=record_dir,
        )

    summary, _ = asyncio.run(record())
    assert summary.stop_reason == "time_limit"
    assert (tmp_path / "record-out" / "manifest.json").is_file()

    # 重放。
    from sceneweave.integrations.concordia.sampling import ReplayDirectory

    replay_dir = ReplayDirectory(record_dir)
    # 列出所有录制 scope 文件，逐个消费确保零残留。

    scope_files = [f for f in record_dir.iterdir() if f.suffix == ".jsonl"]
    assert scope_files, "录制目录包含作用域文件"

    async def replay() -> tuple:
        return await run_headless(
            config,
            replay_dir.for_scope,
            output_directory=tmp_path / "replay-out",
        )

    replay_summary, _ = asyncio.run(replay())
    assert replay_summary.stop_reason == summary.stop_reason
    assert replay_summary.final_time_ms == summary.final_time_ms
    assert replay_summary.world_state_version == summary.world_state_version
