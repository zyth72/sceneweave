"""运行后审计的严格配置加载：模型列表独立于模拟装配与录制清单。

配置缺省不联网、不读环境变量；加载只做结构校验。每个模型条目通过
已有 OpenAI 兼容工厂（默认 chat_completions:create_model）调用，
选项只传环境变量名；明文密钥字段直接拒绝，避免凭据进入配置或日志。
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sceneweave.config.loader import ConfigError, load_json_object

DEFAULT_AUDIT_FACTORY = "sceneweave.integrations.concordia.chat_completions:create_model"
_FACTORY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "key", "token", "secret", "password", "authorization", "headers"}
)


@dataclass(frozen=True, slots=True)
class AuditModelSpec:
    """一个审计模型：稳定 id、工厂（module:callable）与显式选项。"""

    model_id: str
    factory: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not _MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ConfigError("audit model id 只能包含字母数字与 . _ -，且非空")
        if not isinstance(self.factory, str) or not _FACTORY_PATTERN.fullmatch(self.factory):
            raise ConfigError(f"audit model factory 必须是 module:callable 格式：{self.factory!r}")
        if not isinstance(self.options, Mapping) or not all(
            isinstance(key, str) and key for key in self.options
        ):
            raise ConfigError("audit model options 必须是字符串键对象")
        object.__setattr__(self, "options", dict(self.options))

    @property
    def model_name(self) -> str:
        value = self.options.get("model")
        return value if isinstance(value, str) and value else self.model_id


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """非空审计模型列表；每个模型独立配置、独立失败。"""

    models: tuple[AuditModelSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.models, tuple) or not self.models:
            raise ConfigError("audit.models 必须是非空列表")
        if not all(isinstance(item, AuditModelSpec) for item in self.models):
            raise ConfigError("audit.models 每项必须是 AuditModelSpec")
        ids = [item.model_id for item in self.models]
        if len(set(ids)) != len(ids):
            raise ConfigError("audit.models 的 id 必须唯一")


def _reject_secret_keys(options: Mapping[str, Any], at: str) -> None:
    for key in options:
        normalized = key.lower().replace("-", "_")
        if (
            normalized in _SECRET_KEYS
            or normalized.endswith("_key")
            or normalized.endswith("_token")
        ):
            raise ConfigError(
                f"{at}：不得出现明文密钥字段 {key!r}；请用 api_key_env 指定环境变量名"
            )


def _model_spec(item: Any, index: int) -> AuditModelSpec:
    at = f"$.models[{index}]"
    if not isinstance(item, dict):
        raise ConfigError(f"{at}：必须是 JSON 对象")
    unknown = sorted(set(item) - {"id", "factory", "options"})
    if unknown:
        raise ConfigError(f"{at}：包含未知字段 {unknown}")
    if "id" not in item:
        raise ConfigError(f"{at}：缺少必填字段 id")
    factory = item.get("factory", DEFAULT_AUDIT_FACTORY)
    options = item.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"{at}.options：必须是 JSON 对象")
    _reject_secret_keys(options, f"{at}.options")
    return AuditModelSpec(model_id=item["id"], factory=factory, options=options)


def load_audit_config(path: Path) -> AuditConfig:
    """读取严格审计配置；根对象只允许 models，错误带路径。"""
    root = load_json_object(path)
    unknown = sorted(set(root) - {"models"})
    if unknown:
        raise ConfigError(f"{path}：包含未知字段 {unknown}")
    if "models" not in root:
        raise ConfigError(f"{path}：缺少必填字段 models")
    raw_models = root["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError(f"{path}：models 必须是非空数组")
    try:
        config = AuditConfig(
            tuple(_model_spec(item, index) for index, item in enumerate(raw_models))
        )
    except ConfigError as error:
        raise ConfigError(f"{path}：{error}") from error
    return config
