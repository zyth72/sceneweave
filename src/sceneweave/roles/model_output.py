"""模型输出的严格解析：把阶段 JSON 转为不可变结果或明确错误。

解析只做类型/字段/引用校验，不猜语义：省略主动集合、无法解析、bool
乱作字段、缺关联或引用冲突都抛 ModelOutputError，不会被转换成空
声明。schema 以现有不可变消息承接，保持最小直白。
"""

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
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


# 三路倾向之和允许的舍入容差；原始值仍按模型给出的数值保存。
SYSTEM1_PROBABILITY_SUM_TOLERANCE = 0.02

# 模型自报倾向：三个路径的原始值构成的不可变分布。
_SYSTEM1_ROUTE_VALUES = tuple(route.value for route in System1Route)


@dataclass(frozen=True, slots=True)
class System1Tendency:
    """System1 模型自报的三路路径倾向分布（原始值 + 归一化记录）。

    三个数是「本次事件对本角色状态影响所驱动的路径倾向」，不是世界事件
    概率，也未经校准。这里只保存模型给出的原始有限数值与总和；归一化只
    在有记录的情况下进行，不静默改写原始值。
    """

    probabilities: Mapping[System1Route, float]
    note: str
    total: float

    def __post_init__(self) -> None:
        if not isinstance(self.note, str) or not self.note:
            raise ModelOutputError("note 必须是非空字符串")
        if type(self.total) is not float or not math.isfinite(self.total) or self.total <= 0.0:
            raise ModelOutputError("probabilities 总和必须是正有限数值")
        if not isinstance(self.probabilities, Mapping) or set(self.probabilities) != set(
            System1Route
        ):
            raise ModelOutputError("probabilities 必须恰好包含三个路径")
        frozen: dict[System1Route, float] = {}
        for route in System1Route:
            value = self.probabilities.get(route)
            if type(value) is not float or not math.isfinite(value):
                raise ModelOutputError(f"probabilities.{route.value} 必须是有限数值")
            if value < 0.0 or value > 1.0:
                raise ModelOutputError(f"probabilities.{route.value} 必须在 0..1")
            frozen[route] = value
        if self.total != math.fsum(frozen.values()):
            raise ModelOutputError("probabilities 总和与三路数值不一致")
        if (
            not 1.0 - SYSTEM1_PROBABILITY_SUM_TOLERANCE
            <= self.total
            <= 1.0 + SYSTEM1_PROBABILITY_SUM_TOLERANCE
        ):
            raise ModelOutputError(
                f"probabilities 之和必须约为 1（容差 {SYSTEM1_PROBABILITY_SUM_TOLERANCE}），"
                f"得到 {self.total}"
            )
        object.__setattr__(self, "probabilities", MappingProxyType(frozen))

    def normalized(self) -> Mapping[System1Route, float]:
        """按总和归一化后的三路权重（总和>0 在解析时已校验）。"""
        return MappingProxyType(
            {route: self.probabilities[route] / self.total for route in System1Route}
        )


@dataclass(frozen=True, slots=True)
class System1RoutePolicy:
    """可注入的 System1 程序路由策略（确定性阈值）。

    - ``certainty_threshold``：合法归一化分布的最大权重达到该值即确定选择。
    - ``margin_threshold``：最大与次大权重之差达到该值即确定选择。
    任一满足且存在唯一最大项时直接确定；否则按合法归一化分布采样。阈值
    必须是 0..1 的有限数值。
    """

    certainty_threshold: float = 0.75
    margin_threshold: float = 0.25

    def __post_init__(self) -> None:
        for name in ("certainty_threshold", "margin_threshold"):
            value = getattr(self, name)
            if type(value) is bool or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            try:
                number = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} 数值超出可表示范围") from error
            if not math.isfinite(number):
                raise ValueError(f"{name} 必须是有限数值")
            if number < 0.0 or number > 1.0:
                raise ValueError(f"{name} 必须在 0..1")
            object.__setattr__(self, name, number)


# 模型只能输出倾向；最终 route 由程序策略决定，选择方式与原因如实记录。
SYSTEM1_METHOD_FORCED = "forced"
SYSTEM1_METHOD_DETERMINISTIC = "deterministic"
SYSTEM1_METHOD_SAMPLED = "sampled"


@dataclass(frozen=True, slots=True)
class System1Decision:
    """System1 的程序路由结果：由合法分布 + 本角色结构状态生成。

    route 与 note 供角色编排使用；其余字段只用于留痕与审计，不进入其他
    角色感知。``method`` 取 forced/deterministic/sampled；``reason`` 是稳定
    原因标记（no_intention/threshold/tie/below_threshold）。
    """

    route: System1Route
    note: str
    method: str
    reason: str
    allowed_routes: tuple[System1Route, ...]
    raw_probabilities: Mapping[System1Route, float]
    normalized_probabilities: Mapping[System1Route, float]
    legal_weights: Mapping[System1Route, float]
    max_weight: float
    margin: float
    random_value: float | None
    certainty_threshold: float
    margin_threshold: float
    route_seed: int
    seed_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.route, System1Route):
            raise ModelOutputError("route 必须是 System1Route")
        if not isinstance(self.note, str) or not self.note:
            raise ModelOutputError("note 必须是非空字符串")
        if self.method not in (
            SYSTEM1_METHOD_FORCED,
            SYSTEM1_METHOD_DETERMINISTIC,
            SYSTEM1_METHOD_SAMPLED,
        ):
            raise ModelOutputError(f"未知选择方式：{self.method!r}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ModelOutputError("选择原因必须是非空字符串")
        if self.route not in self.allowed_routes:
            raise ModelOutputError("route 必须属于允许路径")
        for name in ("raw_probabilities", "normalized_probabilities", "legal_weights"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise ModelOutputError(f"{name} 必须是映射")
            frozen: dict[System1Route, float] = {}
            for key, value in mapping.items():
                if not isinstance(key, System1Route):
                    raise ModelOutputError(f"{name} 的键必须是 System1Route")
                if type(value) is not float or not math.isfinite(value):
                    raise ModelOutputError(f"{name}.{key.value} 必须是有限数值")
                frozen[key] = value
            object.__setattr__(self, name, MappingProxyType(frozen))


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


def parse_system1_tendency(raw: str) -> System1Tendency:
    """System1 输出：恰好两个外层字段与三个内层路径的倾向分布。

    结构约束：``{"probabilities": {"continue": float, "replan": float,
    "rethink": float}, "note": str}``；三个数必须是有限 0..1 数值（拒绝
    bool/字符串/NaN/Infinity），note 非空，总和为正且距 1 不超过容差。
    旧的 ``{"route": ...}`` schema 不是本解析器的合法形状，明确拒绝，
    不做静默兼容。
    """
    data = parse_model_json(raw, "system1", allowed_keys=("probabilities", "note"))
    _require_keys(data, ("probabilities", "note"), "system1")
    raw_probabilities = data["probabilities"]
    if not isinstance(raw_probabilities, dict):
        raise ModelOutputError("probabilities 必须是 JSON 对象")
    unknown = sorted(set(raw_probabilities) - set(_SYSTEM1_ROUTE_VALUES))
    if unknown:
        raise ModelOutputError(f"probabilities 包含未知路径：{unknown}")
    missing = [name for name in _SYSTEM1_ROUTE_VALUES if name not in raw_probabilities]
    if missing:
        raise ModelOutputError(f"probabilities 缺少路径：{missing}")
    values: dict[System1Route, float] = {}
    for route in System1Route:
        raw_value = raw_probabilities[route.value]
        if type(raw_value) is bool or not isinstance(raw_value, (int, float)):
            raise ModelOutputError(f"probabilities.{route.value} 必须是数值，不能是 bool 或字符串")
        try:
            value = float(raw_value)
        except OverflowError as error:
            raise ModelOutputError(f"probabilities.{route.value} 数值超出可表示范围") from error
        if not math.isfinite(value):
            raise ModelOutputError(f"probabilities.{route.value} 必须是有限数值")
        if value < 0.0 or value > 1.0:
            raise ModelOutputError(f"probabilities.{route.value} 必须在 0..1：{value}")
        values[route] = value
    total = math.fsum(values.values())
    if total <= 0.0:
        raise ModelOutputError("probabilities 之和必须为正")
    if (
        not 1.0 - SYSTEM1_PROBABILITY_SUM_TOLERANCE
        <= total
        <= 1.0 + SYSTEM1_PROBABILITY_SUM_TOLERANCE
    ):
        raise ModelOutputError(
            f"probabilities 之和必须约为 1（容差 {SYSTEM1_PROBABILITY_SUM_TOLERANCE}），"
            f"得到 {total}"
        )
    return System1Tendency(
        probabilities=values, note=_require_str(data["note"], "note"), total=total
    )


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
