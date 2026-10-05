"""录放运行清单：绑定实际输入，并验证所有作用域完整消费。"""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from sceneweave.config.loader import AssemblyConfig, ConfigError, load_json_object
from sceneweave.integrations.concordia.sampling import (
    ReplayMismatchError,
    ReplaySampler,
    scope_file,
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def configuration_data(config: AssemblyConfig) -> dict[str, Any]:
    """路径可以移动；事实、私人背景、事件与运行参数必须相同。"""
    return {
        "scene": _json_value(config.scene),
        "roles": _json_value(config.roles),
        "initial_events": _json_value(config.initial_events),
        "run": _json_value(config.run),
    }


def write_record_manifest(
    directory: Path, config: AssemblyConfig, run_id: str, output_directory: Path | None = None
) -> None:
    scopes = []
    for path in sorted(directory.glob("*.jsonl")):
        scope = unquote(path.name.removeprefix("model-").removesuffix(".jsonl"))
        if scope_file(directory, scope) != path:
            raise ReplayMismatchError(f"录制文件名不是规范作用域路径：{path.name}")
        scopes.append(scope)
    manifest = {
        "version": 1,
        "run_id": run_id,
        "configuration": configuration_data(config),
        "scopes": scopes,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    for target in (directory,) if output_directory is None else (directory, output_directory):
        # 一个运行只写一次清单；错误时也应保存已录制的失败调用。
        with (target / "manifest.json").open("x", encoding="utf-8") as destination:
            destination.write(text)


class RecordedRun:
    """预先核对清单、全部文件和运行身份；重放不持有真实模型。"""

    def __init__(self, directory: Path, config: AssemblyConfig) -> None:
        try:
            manifest = load_json_object(directory / "manifest.json")
        except (ConfigError, OSError) as error:
            raise ReplayMismatchError(f"无法读取录制清单：{error}") from error
        if set(manifest) != {"version", "run_id", "configuration", "scopes"}:
            raise ReplayMismatchError("录制清单字段缺失或未知")
        if type(manifest["version"]) is not int or manifest["version"] != 1:
            raise ReplayMismatchError("录制清单版本无效")
        run_id = manifest["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ReplayMismatchError("录制清单运行身份无效")

        # JSON 类型也必须一致，避免 True == 1 或 1.0 == 1 掩盖篡改。
        def canonical(data: Any) -> str:
            return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)

        if canonical(manifest["configuration"]) != canonical(configuration_data(config)):
            raise ReplayMismatchError("配置或场景内容与录制不一致")
        scopes = manifest["scopes"]
        if not isinstance(scopes, list) or not all(isinstance(s, str) and s for s in scopes):
            raise ReplayMismatchError("录制清单作用域必须是非空字符串数组")
        if len(scopes) != len(set(scopes)):
            raise ReplayMismatchError("录制清单作用域重复")
        expected = {scope_file(directory, scope) for scope in scopes}
        actual = set(directory.glob("*.jsonl"))
        if expected != actual:
            raise ReplayMismatchError("录制作用域文件缺失或有额外文件")
        self._samplers: dict[str, ReplaySampler] = {}
        self._used: set[str] = set()
        for scope in scopes:
            try:
                sampler = ReplaySampler(scope_file(directory, scope))
            except (OSError, ValueError) as error:
                raise ReplayMismatchError(f"录制作用域 {scope} 无效：{error}") from error
            if sampler.header["run_id"] != run_id:
                raise ReplayMismatchError(f"录制作用域 {scope} 与清单不属于同一运行")
            self._samplers[scope] = sampler

    def for_scope(self, scope: str) -> ReplaySampler:
        if scope not in self._samplers:
            raise ReplayMismatchError(f"录制缺少请求作用域：{scope}")
        self._used.add(scope)
        return self._samplers[scope]

    def assert_fully_consumed(self) -> None:
        for sampler in self._samplers.values():
            sampler.assert_fully_consumed()
        unused = sorted(set(self._samplers) - self._used)
        if unused:
            raise ReplayMismatchError(f"录制有未被请求的作用域：{unused}")
