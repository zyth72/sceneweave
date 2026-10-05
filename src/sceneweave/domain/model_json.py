"""模型 JSON 输出的共享严格解析。

供角色组件与行动翻译器共用：解析只做结构校验（重复键、非有限常量、
未知字段、类型），不猜语义、不做修复；失败抛 ModelJsonError，由调用
方决定如何归类。保持最小直白，不引入通用校验框架。
"""

import json
from typing import Any


class ModelJsonError(ValueError):
    """模型 JSON 输出未通过结构校验；不得转换为空集合或默认行为。"""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """重复键直接拒绝：后值覆盖前值会悄悄改变声明语义。"""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelJsonError(f"JSON 对象存在重复键：{key}")
        result[key] = value
    return result


def reject_constant(name: str) -> Any:
    raise ModelJsonError(f"JSON 不接受非有限常量：{name}")


def require_json_mapping(value: Any, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelJsonError(f"{stage} 输出必须是 JSON 对象")
    return value


def require_json_keys(data: dict[str, Any], keys: tuple[str, ...], stage: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ModelJsonError(f"{stage} 输出缺少必填字段：{missing}")


def parse_model_json(raw: str, stage: str, *, allowed_keys: tuple[str, ...]) -> dict[str, Any]:
    """解析一段模型文本为 JSON 对象；失败是明确错误，不做修复。

    重复键、NaN/Infinity 与 schema 外未知键都拒绝，避免错误输出被
    解释为合法空集合或维持声明。
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ModelJsonError(f"{stage} 输出为空")
    try:
        data = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ModelJsonError(f"{stage} 输出不是合法 JSON：{error}") from error
    data = require_json_mapping(data, stage)
    unknown = sorted(set(data) - set(allowed_keys))
    if unknown:
        raise ModelJsonError(f"{stage} 输出包含未知字段：{unknown}")
    return data


def require_json_int(value: Any, field: str) -> int:
    """毫秒时间必须是整数；bool 与浮点都不是合法毫秒。"""
    if type(value) is not int:
        raise ModelJsonError(f"{field} 必须是整数毫秒，得到：{value!r}")
    if value < 0:
        raise ModelJsonError(f"{field} 必须是非负整数毫秒，得到：{value!r}")
    return value


def require_json_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ModelJsonError(f"{field} 必须是整数数组")
    return [require_json_int(item, field) for item in value]
