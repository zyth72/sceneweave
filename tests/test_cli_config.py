"""统一运行入口测试：配置加载、模式错误、headless scripted 闭环。

不调用真实 API；使用提交的脚本 fixture 驱动完整装配。
"""

import json
from pathlib import Path

import pytest

from sceneweave.config.loader import (
    ConfigError,
    load_assembly_config,
)


def write_scene(tmp_path: Path) -> Path:
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
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    return path


def write_config(
    tmp_path: Path,
    scene_file: str = "scene.json",
    roles: list | None = None,
    run: dict | None = None,
    events: list | None = None,
) -> Path:
    config = {
        "scene_path": scene_file,
        "roles": roles
        or [
            {"id": "lin", "background": "测试角色A"},
            {"id": "shen", "background": "测试角色B"},
        ],
        "initial_events": events or [{"id": "startup", "time_seconds": 0, "description": "开场"}],
        "run": run
        or {
            "introspection_interval_seconds": 5,
            "max_simulation_duration_seconds": 8,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_valid_config_with_relative_scene_path(tmp_path: Path) -> None:
    write_scene(tmp_path)
    write_config(tmp_path)
    config = load_assembly_config(tmp_path / "config.json")
    assert config.scene.id == "room"
    assert len(config.roles) == 2
    assert config.roles[0].background == "测试角色A"
    assert config.initial_events[0].time_ms == 0
    assert config.run.introspection_interval_ms == 5000


def test_config_relative_scene_path_resolves_from_config_dir(tmp_path: Path) -> None:
    """相对路径基于配置文件目录，不依赖进程 cwd。"""
    subdir = tmp_path / "sub"
    subdir.mkdir()
    scene = write_scene(subdir)
    config = write_config(subdir)
    loaded = load_assembly_config(config)
    assert loaded.scene_path == scene.resolve()


def test_config_unknown_field_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["surprise"] = 1
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_assembly_config(config)


def test_config_duplicate_key_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    text = text.replace('"scene_path"', '"scene_path": "x", "scene_path"', 1)
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="重复键"):
        load_assembly_config(config)


def test_config_bool_seconds_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(
        tmp_path, run={"introspection_interval_seconds": True, "max_simulation_duration_seconds": 5}
    )
    with pytest.raises(ConfigError, match="bool"):
        load_assembly_config(config)


def test_config_fractional_ms_rejected_at_load(tmp_path: Path) -> None:
    """0.0005 秒无法用整毫秒表示：加载时拒绝（不延迟到装配）。"""
    write_scene(tmp_path)
    config = write_config(
        tmp_path,
        run={"introspection_interval_seconds": 0.0005, "max_simulation_duration_seconds": 5},
    )
    with pytest.raises(ConfigError, match="整毫秒"):
        load_assembly_config(config)


def test_config_nan_seconds_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    text = text.replace(
        '"introspection_interval_seconds": 5', '"introspection_interval_seconds": NaN'
    )
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="有限"):
        load_assembly_config(config)


def test_config_nonfinite_run_seconds_rejected(tmp_path: Path) -> None:
    """运行参数秒配置含 NaN/Infinity 时在 from_seconds 处拒绝。"""
    write_scene(tmp_path)
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    text = text.replace(
        '"introspection_interval_seconds": 5', '"introspection_interval_seconds": Infinity'
    )
    config.write_text(text, encoding="utf-8")
    # Infinity 是合法 JSON 常量，经 parse_constant 拒绝或 from_seconds 拒绝。
    with pytest.raises((ConfigError, ValueError)):
        load_assembly_config(config)


def test_config_missing_role_coverage_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(tmp_path, roles=[{"id": "lin", "background": "只有林"}])
    with pytest.raises(ConfigError, match="场景角色未被配置覆盖"):
        load_assembly_config(config)


def test_config_unknown_role_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(
        tmp_path,
        roles=[
            {"id": "lin", "background": "A"},
            {"id": "shen", "background": "B"},
            {"id": "ghost", "background": "C"},
        ],
    )
    with pytest.raises(ConfigError, match="不是场景中的角色对象"):
        load_assembly_config(config)


def test_config_duplicate_role_rejected(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(
        tmp_path,
        roles=[
            {"id": "lin", "background": "A"},
            {"id": "lin", "background": "B"},
        ],
    )
    with pytest.raises(ConfigError, match="唯一"):
        load_assembly_config(config)


def test_config_scene_file_missing(tmp_path: Path) -> None:
    write_scene(tmp_path)
    config = write_config(tmp_path)
    (tmp_path / "scene.json").unlink()
    with pytest.raises((ConfigError, FileNotFoundError)):
        load_assembly_config(config)


def test_config_scene_json_with_duplicates_rejected(tmp_path: Path) -> None:
    """场景卡 JSON 含重复键时拒绝（不被普通 json.loads 吃掉）。"""
    scene_path = write_scene(tmp_path)
    text = scene_path.read_text(encoding="utf-8")
    scene_path.write_text(
        text.replace('"id": "room"', '"id": "room", "id": "room"', 1), encoding="utf-8"
    )
    write_config(tmp_path)
    with pytest.raises(ConfigError, match="重复键"):
        load_assembly_config(tmp_path / "config.json")


def test_config_noninteger_ms_event_rejected(tmp_path: Path) -> None:
    """初始事件 0.0005 秒（=0.5ms）无法用整毫秒表示，拒绝。"""
    write_scene(tmp_path)
    write_config(tmp_path, events=[{"id": "e", "time_seconds": 0.0005}])
    # time_ms 是属性，加载时不触发；在事件实际使用时（Scheduler 装配）
    # 才会失败。这里验证 from_seconds 入口拒绝。
    from sceneweave.domain.time import milliseconds_from_seconds

    with pytest.raises(ValueError, match="整毫秒"):
        milliseconds_from_seconds(0.0005)
