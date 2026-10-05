"""严格读取配置；加载阶段完成校验，不创建模型、任务或运行数据。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sceneweave.domain.model_json import ModelJsonError, reject_constant, reject_duplicate_keys
from sceneweave.domain.time import milliseconds_from_seconds, validate_simulation_time
from sceneweave.engine.config import RunConfig
from sceneweave.world.scene import (
    SceneObject,
    load_scene_card_json,
    scene_role_ids,
    validate_scene_tree,
)


class ConfigError(ValueError):
    """配置不合法；错误包含来源文件与字段。"""


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ModelJsonError(f"JSON 数值必须有限：{raw}")
    return value


def parse_json_object(text: str, source: str) -> dict[str, Any]:
    """配置、脚本和录制清单共用严格 JSON 入口；保持整数精度。"""
    try:
        data = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, OverflowError) as error:
        raise ConfigError(f"{source}：JSON 无效：{error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{source}：必须是 JSON 对象")
    return data


def load_json_object(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise ConfigError("文件路径必须是 Path")
    return parse_json_object(path.read_text(encoding="utf-8"), str(path))


def _fields(data: dict[str, Any], fields: tuple[str, ...], at: str) -> None:
    missing = sorted(set(fields) - set(data))
    unknown = sorted(set(data) - set(fields))
    if missing:
        raise ConfigError(f"{at}：缺少必填字段 {missing}")
    if unknown:
        raise ConfigError(f"{at}：包含未知字段 {unknown}")


def _text(value: Any, at: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty):
        raise ConfigError(f"{at}：必须是{'非空' if not empty else ''}字符串")
    return value


def _run(data: Any, at: str) -> RunConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"{at}：必须是 JSON 对象")
    _fields(data, ("introspection_interval_seconds", "max_simulation_duration_seconds"), at)
    # 先逐字段检查以保留错误位置；所有转换仍由唯一的秒入口完成。
    for key, value in data.items():
        try:
            converted = milliseconds_from_seconds(value)
            if converted <= 0:
                raise ValueError("必须是正数秒")
        except (ValueError, OverflowError) as error:
            raise ConfigError(f"{at}.{key}：{error}") from error
    return RunConfig.from_seconds(
        data["introspection_interval_seconds"], data["max_simulation_duration_seconds"]
    )


def load_run_config(path: Path) -> RunConfig:
    """读取核心文档的两字段秒配置，经 RunConfig.from_seconds 转换。"""
    return _run(load_json_object(path), str(path))


@dataclass(frozen=True, slots=True)
class RoleSpec:
    role_id: str
    background: str

    def __post_init__(self) -> None:
        _text(self.role_id, "role.id")
        _text(self.background, "role.background", empty=True)


@dataclass(frozen=True, slots=True)
class InitialEventSpec:
    event_id: str
    time_ms: int
    description: str

    def __post_init__(self) -> None:
        _text(self.event_id, "event.id")
        _text(self.description, "event.description")
        validate_simulation_time(self.time_ms, "event.time_ms")


@dataclass(frozen=True, slots=True)
class RunSettings:
    """装配层只保存经秒入口转换后的整数毫秒。"""

    introspection_interval_ms: int
    max_duration_ms: int

    def __post_init__(self) -> None:
        self.to_run_config()

    def to_run_config(self) -> RunConfig:
        from sceneweave.domain.time import SimulationTime

        return RunConfig(
            SimulationTime(self.introspection_interval_ms), SimulationTime(self.max_duration_ms)
        )


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    scene: SceneObject
    scene_path: Path
    roles: tuple[RoleSpec, ...]
    initial_events: tuple[InitialEventSpec, ...]
    run: RunSettings

    def __post_init__(self) -> None:
        if not isinstance(self.scene, SceneObject) or not isinstance(self.scene_path, Path):
            raise ConfigError("scene / scene_path 必须是 SceneObject / Path")
        validate_scene_tree(self.scene)
        if (
            not isinstance(self.roles, tuple)
            or not self.roles
            or not all(isinstance(r, RoleSpec) for r in self.roles)
        ):
            raise ConfigError("roles：必须是非空 RoleSpec 元组")
        if not isinstance(self.initial_events, tuple) or not all(
            isinstance(e, InitialEventSpec) for e in self.initial_events
        ):
            raise ConfigError("initial_events：必须是 InitialEventSpec 元组")
        if not isinstance(self.run, RunSettings):
            raise ConfigError("run：必须是 RunSettings")
        config_roles = [r.role_id for r in self.roles]
        if len(config_roles) != len(set(config_roles)):
            raise ConfigError("roles：参与角色身份必须唯一")
        scene_roles = set(scene_role_ids(self.scene))
        unknown = sorted(set(config_roles) - scene_roles)
        missing = sorted(scene_roles - set(config_roles))
        if unknown:
            raise ConfigError(f"roles：参与角色不是场景中的角色对象：{unknown}")
        if missing:
            raise ConfigError(f"roles：场景角色未被配置覆盖：{missing}")
        event_ids = [e.event_id for e in self.initial_events]
        if len(event_ids) != len(set(event_ids)):
            raise ConfigError("initial_events：初始事件身份重复")


def load_assembly_config(path: Path) -> AssemblyConfig:
    """显式场景路径相对于配置目录；加载失败不产生运行副作用。"""
    root = load_json_object(path)
    try:
        _fields(root, ("scene_path", "roles", "initial_events", "run"), "$")
        scene_path = Path(_text(root["scene_path"], "$.scene_path"))
        if not scene_path.is_absolute():
            scene_path = path.parent / scene_path
        scene_path = scene_path.resolve()
        try:
            scene = load_scene_card_json(scene_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise ConfigError(f"场景文件 {scene_path} 无效：{error}") from error
        if not isinstance(root["roles"], list):
            raise ConfigError("$.roles：必须是数组")
        roles: list[RoleSpec] = []
        for index, item in enumerate(root["roles"]):
            at = f"$.roles[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{at}：必须是 JSON 对象")
            _fields(item, ("id", "background"), at)
            roles.append(
                RoleSpec(
                    _text(item["id"], f"{at}.id"),
                    _text(item["background"], f"{at}.background", empty=True),
                )
            )
        if not isinstance(root["initial_events"], list):
            raise ConfigError("$.initial_events：必须是数组")
        events: list[InitialEventSpec] = []
        for index, item in enumerate(root["initial_events"]):
            at = f"$.initial_events[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{at}：必须是 JSON 对象")
            _fields(item, ("id", "time_seconds", "description"), at)
            try:
                time_ms = milliseconds_from_seconds(item["time_seconds"])
            except (ValueError, OverflowError) as error:
                raise ConfigError(f"{at}.time_seconds：{error}") from error
            events.append(
                InitialEventSpec(
                    _text(item["id"], f"{at}.id"),
                    time_ms,
                    _text(item["description"], f"{at}.description"),
                )
            )
        run = _run(root["run"], "$.run")
        return AssemblyConfig(
            scene,
            scene_path,
            tuple(roles),
            tuple(events),
            RunSettings(run.introspection_interval, run.max_duration),
        )
    except ConfigError as error:
        raise ConfigError(f"{path}：{error}") from error
