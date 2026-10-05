"""配置必须在任何运行副作用前完成校验。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sceneweave.config.loader import (
    ConfigError,
    load_assembly_config,
    load_run_config,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/configs/run.json"


def config_file(tmp_path: Path) -> tuple[Path, dict]:
    data = json.loads(EXAMPLE.read_text())
    data["scene_path"] = str(EXAMPLE.parent / "scene.json")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return path, data


@pytest.mark.parametrize("target", ["run", "event"])
def test_fractional_ms_rejected_by_actual_loader(tmp_path: Path, target: str) -> None:
    path, data = config_file(tmp_path)
    if target == "run":
        data["run"]["introspection_interval_seconds"] = 0.0005
    else:
        data["initial_events"][0]["time_seconds"] = 0.0005
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="整毫秒") as captured:
        load_assembly_config(path)
    assert str(path) in str(captured.value)
    assert target in str(captured.value)


def test_two_field_run_loader_preserves_large_integer_precision(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    seconds = 10**50 + 123
    path.write_text(
        json.dumps(
            {"introspection_interval_seconds": 0.001, "max_simulation_duration_seconds": seconds}
        )
    )
    result = load_run_config(path)
    assert result.introspection_interval == 1
    assert result.max_duration == seconds * 1000


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "1e999", "true"])
def test_numeric_config_rejection_has_file_and_field(tmp_path: Path, raw: str) -> None:
    path, data = config_file(tmp_path)
    path.write_text(
        json.dumps(data).replace(
            '"introspection_interval_seconds": 5', f'"introspection_interval_seconds": {raw}'
        )
    )
    with pytest.raises(ConfigError) as captured:
        load_assembly_config(path)
    assert str(path) in str(captured.value)


def test_nested_unknown_field_rejected_before_assembly(tmp_path: Path) -> None:
    path, data = config_file(tmp_path)
    data["roles"][0]["secret_admin_override"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match=r"roles\[0\].*未知字段"):
        load_assembly_config(path)


def test_relative_path_works_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "scene.json").write_bytes((EXAMPLE.parent / "scene.json").read_bytes())
    (cfgdir / "run.json").write_bytes(EXAMPLE.read_bytes())
    monkeypatch.chdir(tmp_path)
    config = load_assembly_config(cfgdir / "run.json")
    assert config.scene_path == cfgdir / "scene.json"
    assert (
        config.initial_events[0].description
        == json.loads(EXAMPLE.read_text())["initial_events"][0]["description"]
    )


def test_direct_configuration_cannot_hold_mutable_role_list() -> None:
    config = load_assembly_config(EXAMPLE)
    with pytest.raises(ConfigError, match="元组"):
        replace(config, roles=list(config.roles))
