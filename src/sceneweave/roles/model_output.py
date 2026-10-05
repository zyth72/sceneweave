"""模型输出的严格解析：把阶段 JSON 转为不可变结果或明确错误。

解析只做类型/字段/引用校验，不猜语义：省略主动集合、无法解析、bool
乱作字段、缺关联或引用冲突都抛 ModelOutputError，不会被转换成空
声明。schema 以现有不可变消息承接，保持最小直白。
"""

from dataclasses import dataclass
from typing import Any

from sceneweave.domain.identity import ActionId
from sceneweave.domain.model_json import (
    ModelJsonError,
)
from sceneweave.domain.model_json import (
    parse_model_json as _shared_parse_model_json,
)


class ModelOutputError(ModelJsonError):
    """模型输出未通过结构校验；不得转换为空声明或默认行为。"""


@dataclass(frozen=True, slots=True)
class System1Decision:
    """System1 的判断结果：是否需要重新考虑，以及认知更新说明。"""

    reconsider: bool
    note: str

    def __post_init__(self) -> None:
        if type(self.reconsider) is not bool:
            raise ModelOutputError("reconsider 必须是布尔值")
        if not isinstance(self.note, str) or not self.note:
            raise ModelOutputError("note 必须是非空字符串")


@dataclass(frozen=True, slots=True)
class System2Intention:
    """System2 的意向选择：保持/改变后的意向与放弃的行动。"""

    intention: str
    abandon: tuple[ActionId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intention, str) or not self.intention:
            raise ModelOutputError("intention 必须是非空字符串")
        if not isinstance(self.abandon, tuple) or not all(
            isinstance(item, str) and item for item in self.abandon
        ):
            raise ModelOutputError("abandon 必须是非空字符串身份元组")
        if len(set(self.abandon)) != len(self.abandon):
            raise ModelOutputError("abandon 行动身份必须唯一")


@dataclass(frozen=True, slots=True)
class ModificationRef:
    """System3 输出中对一个已有行动的明确修改引用。"""

    action_id: ActionId
    change: str

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ModelOutputError("修改行动身份必须是非空字符串")
        if not isinstance(self.change, str) or not self.change:
            raise ModelOutputError("change 必须是非空字符串")


@dataclass(frozen=True, slots=True)
class System3Plan:
    """System3 的落实计划：意向可行性、内部重做与完整主动集合。"""

    feasible: bool
    reason: str
    retry: bool
    maintain: tuple[ActionId, ...]
    modify: tuple[ModificationRef, ...]
    new_descriptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.feasible) is not bool:
            raise ModelOutputError("feasible 必须是布尔值")
        if not isinstance(self.reason, str):
            raise ModelOutputError("reason 必须是字符串")
        if type(self.retry) is not bool:
            raise ModelOutputError("retry 必须是布尔值")
        for name, items in (
            ("maintain", self.maintain),
            ("modify", self.modify),
            ("new_descriptions", self.new_descriptions),
        ):
            if not isinstance(items, tuple):
                raise ModelOutputError(f"{name} 必须是元组")
        if not all(isinstance(item, str) and item for item in self.maintain):
            raise ModelOutputError("maintain 每项必须是非空字符串身份")
        if not all(isinstance(item, ModificationRef) for item in self.modify):
            raise ModelOutputError("modify 每项必须是 ModificationRef")
        if len(set(self.maintain)) != len(self.maintain):
            raise ModelOutputError("maintain 行动身份必须唯一")
        if len({item.action_id for item in self.modify}) != len(self.modify):
            raise ModelOutputError("modify 行动身份必须唯一")
        overlap = set(self.maintain) & {item.action_id for item in self.modify}
        if overlap:
            raise ModelOutputError(f"同一行动不能同时维持与修改：{sorted(overlap)}")
        for description in self.new_descriptions:
            if not isinstance(description, str) or not description:
                raise ModelOutputError("new_descriptions 每项必须是非空字符串")
        if not self.feasible and (self.maintain or self.modify or self.new_descriptions):
            raise ModelOutputError("意向不可行时不得携带主动集合")
        if self.retry and (self.maintain or self.modify or self.new_descriptions):
            raise ModelOutputError("内部重做时不得携带主动集合")
        if not self.feasible and not self.reason:
            raise ModelOutputError("意向不可行时必须说明原因")
        if self.retry and not self.reason:
            raise ModelOutputError("内部重做时必须说明原因")
        if self.retry and not self.feasible:
            raise ModelOutputError("retry 与 feasible=false 不能同时成立")


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
    """System1 输出：{"reconsider": bool, "note": str}，两字段必填。"""
    data = parse_model_json(raw, "system1", allowed_keys=("reconsider", "note"))
    _require_keys(data, ("reconsider", "note"), "system1")
    return System1Decision(
        reconsider=_require_bool(data["reconsider"], "reconsider"),
        note=_require_str(data["note"], "note"),
    )


def parse_system2_intention(raw: str) -> System2Intention:
    """System2 输出：{"intention": str, "abandon": [行动身份...]}。"""
    data = parse_model_json(raw, "system2", allowed_keys=("intention", "abandon"))
    _require_keys(data, ("intention", "abandon"), "system2")
    return System2Intention(
        intention=_require_str(data["intention"], "intention"),
        abandon=_require_id_list(data["abandon"], "abandon"),
    )


def parse_system3_plan(raw: str) -> System3Plan:
    """System3 输出：完整主动集合三键必填（显式空用空数组表达）。

    feasible 为 false 表示意向不可行（须说明原因并反馈 System2）；
    retry 为 true 表示只是做法需要重做；两者都不得携带主动集合。
    """
    data = parse_model_json(
        raw,
        "system3",
        allowed_keys=("feasible", "reason", "retry", "maintain", "modify", "new_actions"),
    )
    _require_keys(data, ("feasible", "maintain", "modify", "new_actions"), "system3")
    retry = _require_bool(data.get("retry", False), "retry")
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ModelOutputError("reason 必须是字符串")
    if not isinstance(data["modify"], list):
        raise ModelOutputError("modify 必须是数组")
    modify: list[ModificationRef] = []
    for item in data["modify"]:
        if not isinstance(item, dict):
            raise ModelOutputError("modify 每项必须是对象")
        _require_keys(item, ("id", "change"), "system3.modify")
        if set(item) - {"id", "change"}:
            raise ModelOutputError("system3.modify 输出包含未知字段")
        modify.append(
            ModificationRef(
                action_id=ActionId(_require_str(item["id"], "modify.id")),
                change=_require_str(item["change"], "modify.change"),
            )
        )
    if not isinstance(data["new_actions"], list):
        raise ModelOutputError("new_actions 必须是数组")
    new_descriptions: list[str] = []
    for item in data["new_actions"]:
        if not isinstance(item, str) or not item:
            raise ModelOutputError("new_actions 每项必须是非空字符串描述")
        new_descriptions.append(item)
    return System3Plan(
        feasible=_require_bool(data["feasible"], "feasible"),
        reason=reason,
        retry=retry,
        maintain=_require_id_list(data["maintain"], "maintain"),
        modify=tuple(modify),
        new_descriptions=tuple(new_descriptions),
    )
