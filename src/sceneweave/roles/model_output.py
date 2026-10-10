"""模型输出的严格解析：把阶段 JSON 转为不可变结果或明确错误。

解析只做类型/字段/引用校验，不猜语义：省略主动集合、无法解析、bool
乱作字段、缺关联或引用冲突都抛 ModelOutputError，不会被转换成空
声明。schema 以现有不可变消息承接，保持最小直白。
"""

import enum
from dataclasses import dataclass
from typing import Any

from sceneweave.domain.action_plan import ActionPlan, parse_action_plan
from sceneweave.domain.identity import ActionId
from sceneweave.domain.model_json import (
    ModelJsonError,
)
from sceneweave.domain.model_json import (
    parse_json_text as _shared_parse_json_text,
)
from sceneweave.domain.model_json import (
    parse_model_json as _shared_parse_model_json,
)


class ModelOutputError(ModelJsonError):
    """模型输出未通过结构校验；不得转换为空声明或默认行为。"""


class System1Route(enum.StrEnum):
    """System1 的三路分流；严格枚举，模型不得自造。"""

    CONTINUE = "continue"
    REPLAN = "replan"
    RETHINK = "rethink"


@dataclass(frozen=True, slots=True)
class System1Decision:
    """System1 的判断结果：三路分流与认知更新说明。

    - ``continue``：沿用已保存的完整链与排程，不调用 System2/3/翻译器。
    - ``replan``：保留当前意向，只让 System3 重新落实。
    - ``rethink``：重新选择意向（System2）后再落实；首次无意向必须 rethink。
    """

    route: System1Route
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.route, System1Route):
            raise ModelOutputError("route 必须是 System1Route")
        if not isinstance(self.note, str) or not self.note:
            raise ModelOutputError("note 必须是非空字符串")


@dataclass(frozen=True, slots=True)
class System3Unfulfillable:
    """System3 的独立内部失败信号：当前已知条件无法落实该意向。

    这不是动作，也不是 World 客观事实；只有 System3 明确判断在保持当前
    意向的前提下找不到方案时才输出。解析为独立类型，不并入动作列表，
    也不使用 feasible/retry wrapper 或 diff 分类。
    """

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ModelOutputError("reason 必须是非空字符串")


_SYSTEM3_ERROR = "intention_unfulfillable"


@dataclass(frozen=True, slots=True)
class System2Intention:
    """System2 的意向选择：当前意向（不含物理可执行性与放弃指令）。"""

    intention: str

    def __post_init__(self) -> None:
        if not isinstance(self.intention, str) or not self.intention:
            raise ModelOutputError("intention 必须是非空字符串")


@dataclass(frozen=True, slots=True)
class _Missing:
    """占位以保持模块导入面稳定；不参与活跃路径。"""


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], stage: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ModelOutputError(f"{stage} 输出缺少必填字段：{missing}")


def _require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ModelOutputError(f"{field} 必须是布尔值（true/false），得到：{value!r}")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelOutputError(f"{field} 必须是非空字符串")
    return value


def _require_id_list(value: Any, field: str) -> tuple[ActionId, ...]:
    if not isinstance(value, list):
        raise ModelOutputError(f"{field} 必须是数组")
    items: list[ActionId] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ModelOutputError(f"{field} 每项必须是非空字符串身份")
        items.append(ActionId(item))
    return tuple(items)


def parse_model_json(raw: str, stage: str, *, allowed_keys: tuple[str, ...]) -> dict[str, Any]:
    """共享严格解析的角色侧包装：错误统一为 ModelOutputError。"""
    try:
        return _shared_parse_model_json(raw, stage, allowed_keys=allowed_keys)
    except ModelJsonError as error:
        raise ModelOutputError(str(error)) from error


def parse_system1_decision(raw: str) -> System1Decision:
    """System1 输出：{"route": "continue|replan|rethink", "note": str}，两字段必填。"""
    data = parse_model_json(raw, "system1", allowed_keys=("route", "note"))
    _require_keys(data, ("route", "note"), "system1")
    raw_route = data["route"]
    if not isinstance(raw_route, str):
        raise ModelOutputError(f"route 必须是字符串，得到：{raw_route!r}")
    try:
        route = System1Route(raw_route)
    except ValueError as error:
        raise ModelOutputError(f"未知 route：{raw_route!r}") from error
    return System1Decision(route=route, note=_require_str(data["note"], "note"))


def parse_system2_intention(raw: str) -> System2Intention:
    """System2 输出：{"intention": str}；只产当前意向，不产 abandon。"""
    data = parse_model_json(raw, "system2", allowed_keys=("intention",))
    _require_keys(data, ("intention",), "system2")
    return System2Intention(intention=_require_str(data["intention"], "intention"))


def parse_system3_output(raw: str) -> ActionPlan | System3Unfulfillable:
    """System3 输出：一维全量动作数组，或独立的意向不可落实控制对象。

    数组按 ``parse_action_plan`` 严格解析（无自由文本、无 ID-only、无未知
    字段）。控制对象必须恰为 ``{"error": "intention_unfulfillable",
    "reason": 非空字符串}``，解析为独立内部结果而非动作。其它形状明确报错。
    """
    try:
        data = _shared_parse_json_text(raw, "system3")
    except ModelJsonError as error:
        raise ModelOutputError(str(error)) from error
    if isinstance(data, list):
        try:
            return parse_action_plan(raw, "system3")
        except ModelJsonError as error:
            raise ModelOutputError(str(error)) from error
    if isinstance(data, dict):
        unknown = sorted(set(data) - {"error", "reason"})
        if unknown:
            raise ModelOutputError(f"system3 控制对象包含未知字段：{unknown}")
        if set(data) != {"error", "reason"}:
            raise ModelOutputError("system3 控制对象必须恰含 error 与 reason")
        if data["error"] != _SYSTEM3_ERROR:
            raise ModelOutputError(f"未知 system3 错误码：{data['error']!r}")
        return System3Unfulfillable(reason=_require_str(data["reason"], "reason"))
    raise ModelOutputError("system3 输出必须是 JSON 一维数组或 intention_unfulfillable 控制对象")


def parse_system3_plan(raw: str) -> ActionPlan:
    """System3 输出必须是动作数组；控制对象按解析错误处理。"""
    outcome = parse_system3_output(raw)
    if not isinstance(outcome, ActionPlan):
        raise ModelOutputError("system3 输出是意向不可落实控制对象，不是动作数组")
    return outcome
