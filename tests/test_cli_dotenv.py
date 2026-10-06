"""统一入口 --live 的 cwd .env 加载边界测试。

只使用临时假 .env、假 live 工厂与本地假 HTTP 端点：不读取项目真实
.env 或真实密钥，不调用真实 API。测试实际走 main() 的 argparse →
配置加载 → .env 加载 → live 工厂装配链路（含内置 Chat Completions
工厂），并核定 scripted/replay/help 不读 .env。
"""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from sceneweave.app.assembly import AssemblyError

from .test_chat_completions import (
    _FACTORY,
    _FakeChatServer,
    _FullStackResponder,
    _write_full_config,
)

KEY = "SCENEWEAVE_DOTENV_TEST_KEY"
BASE_KEY = "SCENEWEAVE_DOTENV_TEST_BASE"
DOTENV_KEY_ENV = "SCENEWEAVE_DOTENV_TEST_API_KEY"
FAKE_MODULE = "sceneweave_fake_dotenv_factory"


def _write_config(tmp_path: Path) -> Path:
    """最小单角色配置；场景相对配置文件目录解析。"""
    scene = {
        "id": "room",
        "name": "房间",
        "kind": "房间",
        "unit": "m",
        "size": [4, 3],
        "children": [
            {"id": "lin", "name": "林", "kind": "角色", "position": [0.5, 0.5], "children": []},
        ],
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    config = {
        "scene_path": "scene.json",
        "roles": [{"id": "lin", "background": "测试角色"}],
        "initial_events": [{"id": "startup", "time_seconds": 0, "description": "开场"}],
        "run": {"introspection_interval_seconds": 5, "max_simulation_duration_seconds": 8},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def _install_fake_factory(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    """装入假 live 工厂：记录调用现场环境后主动停止，不调真实模型。"""
    module = types.ModuleType(FAKE_MODULE)

    def factory(scope: str, options: dict[str, Any]) -> object:
        seen["scope"] = scope
        seen["value"] = os.environ.get(KEY)
        seen["base"] = os.environ.get(BASE_KEY)
        raise AssemblyError("假工厂完成环境检查后主动停止")

    module.factory = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FAKE_MODULE, module)


def _run_live(config: Path, output: Path) -> int:
    from sceneweave.__main__ import main

    return main(
        [
            str(config),
            "--live",
            f"{FAKE_MODULE}:factory",
            "--output",
            str(output),
        ]
    )


def test_live_loads_cwd_dotenv_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv(BASE_KEY, raising=False)
    config = _write_config(tmp_path)
    (tmp_path / ".env").write_text(
        f"{KEY}=dotenv-${{{BASE_KEY}}}\n{BASE_KEY}=base-from-dotenv\n",
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    assert _run_live(config, tmp_path / "out") == 1  # 已进入装配，被假工厂主动停止
    assert seen["value"] == f"dotenv-${{{BASE_KEY}}}"  # 不插值：${...} 原样保留
    assert seen["base"] == "base-from-dotenv"


def test_terminal_env_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KEY, "from-terminal")
    config = _write_config(tmp_path)
    (tmp_path / ".env").write_text(f"{KEY}=from-dotenv\n", encoding="utf-8")
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    assert _run_live(config, tmp_path / "out") == 1
    assert seen["value"] == "from-terminal"
    assert os.environ[KEY] == "from-terminal"


def test_dotenv_is_not_searched_upward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY, raising=False)
    config = _write_config(tmp_path)
    (tmp_path / ".env").write_text(f"{KEY}=from-parent\n", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    assert _run_live(config, child / "out") == 1
    assert seen["value"] is None
    assert KEY not in os.environ


def test_missing_dotenv_keeps_existing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KEY, "terminal-only")
    config = _write_config(tmp_path)
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    assert _run_live(config, tmp_path / "out") == 1
    assert seen["value"] == "terminal-only"


def test_unreadable_dotenv_fails_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path)
    (tmp_path / ".env").write_bytes(b"SCENEWEAVE_FAKE=fake-value\n\xff\xfeSECRET-MARKER\n")
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    assert _run_live(config, tmp_path / "out") == 2
    captured = capsys.readouterr()
    assert "UnicodeDecodeError" in captured.err
    assert "SECRET-MARKER" not in captured.err
    assert "fake-value" not in captured.err
    assert seen == {}


def test_invalid_config_does_not_read_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sceneweave.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KEY, raising=False)
    (tmp_path / ".env").write_text(f"{KEY}=must-not-load\n", encoding="utf-8")
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    code = main(
        [
            str(tmp_path / "missing.json"),
            "--live",
            f"{FAKE_MODULE}:factory",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert seen == {}
    assert KEY not in os.environ


def test_invalid_live_options_do_not_build_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sceneweave.__main__ import main

    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path)
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    code = main(
        [
            str(config),
            "--live",
            f"{FAKE_MODULE}:factory",
            "--live-options",
            "not-json",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert seen == {}


def test_programmatic_factory_does_not_load_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sceneweave.app.assembly import build_live_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KEY, raising=False)
    (tmp_path / ".env").write_text(f"{KEY}=must-not-load\n", encoding="utf-8")
    seen: dict[str, Any] = {}
    _install_fake_factory(monkeypatch, seen)

    factory = build_live_factory(f"{FAKE_MODULE}:factory", {})
    with pytest.raises(AssemblyError, match="假工厂"):
        factory("role:lin")
    assert seen["value"] is None
    assert KEY not in os.environ


def test_cli_live_builtin_factory_reads_dotenv_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """内置 live 工厂经 CLI 装配后使用 .env 密钥（本地假端点，不联网）。"""
    from sceneweave.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(DOTENV_KEY_ENV, raising=False)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    config = _write_full_config(tmp_path)
    (tmp_path / ".env").write_text(f"{DOTENV_KEY_ENV}=sk-dotenv-fake-secret\n", encoding="utf-8")

    server = _FakeChatServer(_FullStackResponder())
    with server:
        code = main(
            [
                str(config),
                "--live",
                _FACTORY,
                "--live-options",
                json.dumps(
                    {
                        "base_url": server.base_url,
                        "model": "test-model",
                        "api_key_env": DOTENV_KEY_ENV,
                    }
                ),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert server.requests, "内置工厂应已向本地假端点发起请求"
        assert all(
            request["headers"]["authorization"] == "Bearer sk-dotenv-fake-secret"
            for request in server.requests
        )
    assert code == 0
    assert os.environ[DOTENV_KEY_ENV] == "sk-dotenv-fake-secret"


def test_scripted_replay_help_do_not_read_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sceneweave.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KEY, raising=False)
    (tmp_path / ".env").write_text(f"{KEY}=must-not-load\n", encoding="utf-8")
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "configs"
    config = example_dir / "run.json"
    scripts = example_dir / "scripted.json"

    assert (
        main([str(config), "--scripted", str(scripts), "--output", str(tmp_path / "scripted-out")])
        == 0
    )
    assert KEY not in os.environ

    recording = tmp_path / "recording"
    assert (
        main(
            [
                str(config),
                "--scripted",
                str(scripts),
                "--record",
                str(recording),
                "--output",
                str(tmp_path / "record-out"),
            ]
        )
        == 0
    )
    assert KEY not in os.environ

    assert (
        main([str(config), "--replay", str(recording), "--output", str(tmp_path / "replay-out")])
        == 0
    )
    assert KEY not in os.environ

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert KEY not in os.environ
