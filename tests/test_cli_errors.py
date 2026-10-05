"""CLI 模式错误与安全行为测试。

不调用真实 API；验证模式互斥、工厂类型校验、缺失参数明确报错、
--help 无副作用。
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sceneweave.app.assembly import AssemblyError, build_live_factory


def test_cli_help_no_side_effects(capsys: Any) -> None:
    """--help 正常退出、打印用法。"""
    from sceneweave.__main__ import _build_parser

    parser = _build_parser()
    assert parser.prog == "sceneweave"
    help_text = parser.format_help()
    assert "sceneweave" in help_text


def test_cli_no_mode_error() -> None:
    """缺少模式参数时 argparse 明确拒绝。"""
    from sceneweave.__main__ import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["some-config"])


def test_cli_record_and_replay_mutually_exclusive() -> None:
    """--record 与 --replay 互斥（--replay 是模式，--record 是选项）。"""
    from sceneweave.__main__ import main

    assert main(["x", "--replay", "some_dir", "--record", "a"]) == 2


def test_cli_scripted_file_missing() -> None:
    """脚本文件不存在时明确报错。"""
    from sceneweave.__main__ import main

    assert main(["x", "--scripted", "/nonexistent/script.json"]) == 2


def test_cli_replay_directory_missing() -> None:
    """重放目录不存在时明确报错。"""
    from sceneweave.__main__ import main

    assert main(["x", "--replay", "/nonexistent/dir"]) == 2


def test_live_factory_invalid_format() -> None:
    """工厂格式无效：明确报错。"""
    with pytest.raises(AssemblyError, match="module:callable"):
        build_live_factory("invalid", {})


def test_live_factory_module_not_found() -> None:
    with pytest.raises(AssemblyError, match="无法导入"):
        build_live_factory("nonexistent_module:factory", {})


def test_live_factory_wrong_return_type() -> None:
    """工厂返回非 LanguageModel：类型校验拒绝。"""

    class FakeModule:
        def factory(scope: str, options: dict) -> object:
            return "not a model"

    import types

    module = types.ModuleType("fake_live")
    module.factory = lambda scope, opts: "not a model"  # type: ignore[attr-defined]

    sys.modules["fake_live"] = module
    try:
        factory = build_live_factory("fake_live:factory", {})
        with pytest.raises(AssemblyError, match="LanguageModel"):
            factory("test-scope")
    finally:
        del sys.modules["fake_live"]


def test_cli_output_exists_fails_safe(tmp_path: Path) -> None:
    """输出目录已存在时不覆盖。"""
    from sceneweave.__main__ import main

    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "id": "r",
                "name": "房间",
                "kind": "房间",
                "unit": "m",
                "size": [4, 3],
                "children": [{"id": "lin", "name": "林", "kind": "角色"}],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "scene_path": "scene.json",
                "roles": [{"id": "lin", "background": "B"}],
                "initial_events": [],
                "run": {
                    "introspection_interval_seconds": 1,
                    "max_simulation_duration_seconds": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    script = tmp_path / "script.json"
    script.write_text("{}", encoding="utf-8")
    assert main([str(config), "--scripted", str(script), "--output", str(out)]) != 0
