"""跨模块身份类型与通用校验。

身份只用于定位与关联，不携带业务语义。所有跨模块消息在构造时校验
身份为非空字符串，并在需要关联的位置校验唯一性。
"""

from collections.abc import Iterable
from typing import NewType

RoleId = NewType("RoleId", str)
CycleId = NewType("CycleId", str)
EventId = NewType("EventId", str)
ActionId = NewType("ActionId", str)
ResponseId = NewType("ResponseId", str)
CommitGroupId = NewType("CommitGroupId", str)
# 共同世界状态版本：由 World 在每次实际提交后递增，只增不改。
WorldStateVersion = NewType("WorldStateVersion", int)


def require_identifier(value: str, field: str) -> None:
    """校验身份为非空字符串；空串或非字符串直接拒绝。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}必须是非空字符串")


def require_text(value: str, field: str) -> None:
    """校验文本为非空字符串；非字符串（含 list/int）或空串直接拒绝。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}必须是非空字符串")


def require_unique_identifiers(values: Iterable[str], field: str) -> None:
    """校验一组身份各自非空且互不重复。"""
    seen: set[str] = set()
    for value in values:
        require_identifier(value, field)
        if value in seen:
            raise ValueError(f"{field}存在重复身份：{value}")
        seen.add(value)


def require_version(value: int, field: str) -> None:
    """校验世界状态版本为非负整数；bool 不是版本。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}必须是非负整数，得到：{value!r}")
    if value < 0:
        raise ValueError(f"{field}必须是非负整数，得到：{value!r}")
