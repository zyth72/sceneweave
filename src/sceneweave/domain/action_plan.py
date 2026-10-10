"""结构化原子动作与一维全量动作计划。

协议最终语义（一维 + 显式依赖）：

- System3 每次输出一个 JSON **一维数组**，完整声明从现在起要做的结构化
  原子动作；没有主动行动时输出 ``[]``（合法空声明，错误不得当作空）。
- 原子动作种类用 :class:`ActionKind` 枚举，字段固定；禁止自由 ``string``
  描述、``do:[散文]`` 或自定义 operation 逃生口。``say`` 的 ``content``
  允许台词文本；伸手与抓取必须分别表达，不塞宏观串行动作。
- 可选 ``depends_on``：依赖表示「World 已实际把前置动作提交为 completed，
  后置动作才允许开始」，不是预计结束、也不是到时就成功；多个依赖需全部
  完成，失败/未确认不放行。没有依赖的动作独立排时，可部分交叠。
- 模型不得自造真实身份：新动作用计划内局部稳定 ``key`` 供同计划引用，
  程序一次性分配真实 ``action_id`` 后把依赖解析到真实身份；已有动作必须
  携带确认的真实 ``id`` 与完整参数。局部 key 只用于本计划内引用，不是
  World 实体、也不是真实行动身份。

``parse_action_plan`` 严格解析模型输出：重复键、非有限常量、未知类型、
未知字段、空/混合类型元素、ID-only 对象、未定义/歧义/自依赖/重复/循环
依赖都明确报错，不修复、不伪空。display text 由 ``render_atomic_action``
代码渲染，仅用于呈现与日志，不是动作接口。
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sceneweave.domain.identity import ActionId, require_identifier
from sceneweave.domain.model_json import ModelJsonError, parse_json_text


class ActionKind(StrEnum):
    """结构化原子动作的种类；每一种的字段固定。"""

    REACH = "reach"
    GRASP = "grasp"
    RELEASE = "release"
    MOVE = "move"
    SAY = "say"
    LOOK = "look"
    WAIT = "wait"
    STAND = "stand"
    SIT = "sit"
    GIVE = "give"


class Effector(StrEnum):
    """执行动作的肢体；固定枚举，模型不得自造。"""

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class MoveMode(StrEnum):
    """移动方式；固定枚举。"""

    WALK = "walk"
    BRISK = "brisk"
    SLOW = "slow"
    RUN = "run"
    SNEAK = "sneak"


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelJsonError(f"{field} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ModelJsonError(f"{field} 必须是有限数值")
    return number


@dataclass(frozen=True, slots=True)
class Position:
    """明确坐标（米）；非有限值与 bool 明确拒绝。"""

    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _require_finite(self.x, "坐标 x"))
        object.__setattr__(self, "y", _require_finite(self.y, "坐标 y"))
        object.__setattr__(self, "z", _require_finite(self.z, "坐标 z"))


@dataclass(frozen=True, slots=True)
class MoveDestination:
    """move 的目的地：实体标识或明确坐标，二者恰有其一。"""

    entity: str | None = None
    position: Position | None = None

    def __post_init__(self) -> None:
        has_entity = self.entity is not None
        has_position = self.position is not None
        if has_entity == has_position:
            raise ModelJsonError("move 目的地必须恰为实体或坐标之一")
        if has_entity and (not isinstance(self.entity, str) or not self.entity):
            raise ModelJsonError("move 目的地实体必须是非空字符串")
        if has_position and not isinstance(self.position, Position):
            raise ModelJsonError("move 目的地坐标必须是 Position")


@dataclass(frozen=True, slots=True)
class AtomicAction:
    """一个结构化的原子动作；字段固定，按种类校验必填与禁用字段。"""

    kind: ActionKind
    target: str | None = None
    effector: Effector | None = None
    destination: MoveDestination | None = None
    content: str | None = None
    mode: MoveMode | None = None
    item: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise ModelJsonError("动作种类必须是 ActionKind")
        allowed, required = _KIND_FIELDS[self.kind]
        present = {
            "target": self.target,
            "effector": self.effector,
            "destination": self.destination,
            "content": self.content,
            "mode": self.mode,
            "item": self.item,
        }
        for name, value in present.items():
            if name not in allowed and value is not None:
                raise ModelJsonError(f"{self.kind.value} 动作不接受字段：{name}")
        missing = [name for name in required if present[name] is None]
        if missing:
            raise ModelJsonError(f"{self.kind.value} 动作缺少必填字段：{missing}")
        for name in ("target", "item"):
            value = present[name]
            if value is not None and (not isinstance(value, str) or not value):
                raise ModelJsonError(f"{name} 必须是非空实体标识")
        if self.effector is not None and not isinstance(self.effector, Effector):
            raise ModelJsonError("effector 必须是 Effector 枚举")
        if self.mode is not None and not isinstance(self.mode, MoveMode):
            raise ModelJsonError("mode 必须是 MoveMode 枚举")
        if self.content is not None and (not isinstance(self.content, str) or not self.content):
            raise ModelJsonError("say 的 content 必须是非空字符串")
        if self.destination is not None and not isinstance(self.destination, MoveDestination):
            raise ModelJsonError("destination 必须是 MoveDestination")

    @property
    def display(self) -> str:
        return render_atomic_action(self)


# 每种动作允许出现的字段与必须出现的字段（kind 除外）。
_KIND_FIELDS: dict[ActionKind, tuple[frozenset[str], tuple[str, ...]]] = {
    ActionKind.REACH: (frozenset({"target", "effector"}), ("target", "effector")),
    ActionKind.GRASP: (frozenset({"target", "effector"}), ("target", "effector")),
    ActionKind.RELEASE: (frozenset({"target", "effector"}), ("target", "effector")),
    ActionKind.MOVE: (frozenset({"destination", "mode"}), ("destination",)),
    ActionKind.SAY: (frozenset({"content", "target"}), ("content",)),
    ActionKind.LOOK: (frozenset({"target"}), ()),
    ActionKind.WAIT: (frozenset(), ()),
    ActionKind.STAND: (frozenset(), ()),
    ActionKind.SIT: (frozenset(), ()),
    ActionKind.GIVE: (frozenset({"target", "item", "effector"}), ("target", "item", "effector")),
}


def render_atomic_action(action: AtomicAction) -> str:
    """把结构化动作渲染为稳定 display text；仅用于呈现，不是动作接口。"""
    if action.kind is ActionKind.REACH:
        assert action.target is not None and action.effector is not None
        return f"伸手触及 {action.target}（{action.effector.value}）"
    if action.kind is ActionKind.GRASP:
        assert action.target is not None and action.effector is not None
        return f"抓取 {action.target}（{action.effector.value}）"
    if action.kind is ActionKind.RELEASE:
        assert action.target is not None and action.effector is not None
        return f"松开 {action.target}（{action.effector.value}）"
    if action.kind is ActionKind.MOVE:
        destination = action.destination
        assert destination is not None
        if destination.entity is not None:
            where = f"实体 {destination.entity}"
        else:
            position = destination.position
            assert position is not None
            where = f"坐标 ({position.x}, {position.y}, {position.z})"
        mode = action.mode.value if action.mode is not None else MoveMode.WALK.value
        return f"移动到 {where}（{mode}）"
    if action.kind is ActionKind.SAY:
        if action.target is not None:
            return f"对 {action.target} 说「{action.content}」"
        return f"说「{action.content}」"
    if action.kind is ActionKind.LOOK:
        return f"看向 {action.target}" if action.target is not None else "环顾"
    if action.kind is ActionKind.WAIT:
        return "等待"
    if action.kind is ActionKind.STAND:
        return "站起"
    if action.kind is ActionKind.SIT:
        return "坐下"
    if action.kind is ActionKind.GIVE:
        assert action.target is not None and action.item is not None and action.effector is not None
        return f"把 {action.item} 交给 {action.target}（{action.effector.value}）"
    raise ModelJsonError(f"未知动作种类：{action.kind!r}")  # pragma: no cover


def _require_entity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelJsonError(f"{field} 必须是非空实体标识")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelJsonError(f"{field} 必须是非空字符串")
    return value


def _require_field(data: dict[str, Any], name: str, stage: str) -> Any:
    if name not in data:
        raise ModelJsonError(f"{stage} 缺少必填字段：{name}")
    return data[name]


def _parse_effector(value: Any, field: str) -> Effector:
    if not isinstance(value, str):
        raise ModelJsonError(f"{field} 必须是字符串")
    try:
        return Effector(value)
    except ValueError as error:
        raise ModelJsonError(f"{field} 未知 effector：{value!r}") from error


def _parse_mode(value: Any, field: str) -> MoveMode:
    if value is None:
        return MoveMode.WALK
    if not isinstance(value, str):
        raise ModelJsonError(f"{field} 必须是字符串")
    try:
        return MoveMode(value)
    except ValueError as error:
        raise ModelJsonError(f"{field} 未知移动方式：{value!r}") from error


def _parse_destination(value: Any, field: str) -> MoveDestination:
    if not isinstance(value, dict):
        raise ModelJsonError(f"{field} 必须是对象")
    unknown = sorted(set(value) - {"entity", "position"})
    if unknown:
        raise ModelJsonError(f"{field} 包含未知字段：{unknown}")
    if ("entity" in value) == ("position" in value):
        raise ModelJsonError(f"{field} 必须恰为 entity 或 position 之一")
    if "entity" in value:
        return MoveDestination(entity=_require_entity(value["entity"], f"{field}.entity"))
    raw = value["position"]
    if not isinstance(raw, dict):
        raise ModelJsonError(f"{field}.position 必须是对象")
    unknown_pos = sorted(set(raw) - {"x", "y", "z"})
    if unknown_pos:
        raise ModelJsonError(f"{field}.position 包含未知字段：{unknown_pos}")
    for name in ("x", "y"):
        if name not in raw:
            raise ModelJsonError(f"{field}.position 缺少必填字段：{name}")
    return MoveDestination(
        position=Position(
            x=_require_finite(raw["x"], f"{field}.position.x"),
            y=_require_finite(raw["y"], f"{field}.position.y"),
            z=_require_finite(raw["z"], f"{field}.position.z") if "z" in raw else 0.0,
        )
    )


def atomic_action_from_json(data: Any, stage: str) -> AtomicAction:
    """从一个 JSON 对象严格构造原子动作；未知 kind/字段与类型错误都报错。"""
    if not isinstance(data, dict):
        raise ModelJsonError(f"{stage} 每个动作必须是 JSON 对象")
    if "kind" not in data:
        raise ModelJsonError(f"{stage} 动作缺少 kind（ID-only 对象非法）")
    kind_raw = data["kind"]
    if not isinstance(kind_raw, str):
        raise ModelJsonError(f"{stage} 动作 kind 必须是字符串")
    try:
        kind = ActionKind(kind_raw)
    except ValueError as error:
        raise ModelJsonError(f"{stage} 未知动作种类：{kind_raw!r}") from error
    allowed, _required = _KIND_FIELDS[kind]
    unknown = sorted(set(data) - (allowed | {"kind"}))
    if unknown:
        raise ModelJsonError(f"{stage} {kind.value} 动作包含未知字段：{unknown}")
    if kind is ActionKind.MOVE:
        destination = _parse_destination(
            _require_field(data, "destination", stage), f"{stage}.destination"
        )
        mode = _parse_mode(data.get("mode"), f"{stage}.mode")
        return AtomicAction(kind=kind, destination=destination, mode=mode)
    if kind is ActionKind.SAY:
        content = _require_text(_require_field(data, "content", stage), f"{stage}.content")
        target = _require_entity(data["target"], f"{stage}.target") if "target" in data else None
        return AtomicAction(kind=kind, content=content, target=target)
    if kind is ActionKind.LOOK:
        target = _require_entity(data["target"], f"{stage}.target") if "target" in data else None
        return AtomicAction(kind=kind, target=target)
    if kind is ActionKind.GIVE:
        return AtomicAction(
            kind=kind,
            target=_require_entity(_require_field(data, "target", stage), f"{stage}.target"),
            item=_require_entity(_require_field(data, "item", stage), f"{stage}.item"),
            effector=_parse_effector(_require_field(data, "effector", stage), f"{stage}.effector"),
        )
    if kind in (ActionKind.REACH, ActionKind.GRASP, ActionKind.RELEASE):
        return AtomicAction(
            kind=kind,
            target=_require_entity(_require_field(data, "target", stage), f"{stage}.target"),
            effector=_parse_effector(_require_field(data, "effector", stage), f"{stage}.effector"),
        )
    return AtomicAction(kind=kind)


@dataclass(frozen=True, slots=True)
class PlanAction:
    """计划中的一个动作条目。

    ``action_id`` 为 ``None`` 表示新动作（由程序分配真实身份）；已有动作
    必须携带确认的真实 ``action_id``。``key`` 仅在本次计划内供相互引用
    （局部稳定 key），``depends_on`` 是原始引用（局部 key 或真实身份），
    由角色按一次性分配的真实身份解析。
    """

    action: AtomicAction
    action_id: ActionId | None = None
    key: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, AtomicAction):
            raise ModelJsonError("计划动作必须是 AtomicAction")
        if self.action_id is not None:
            require_identifier(self.action_id, "动作身份")
        if self.key is not None:
            if not isinstance(self.key, str) or not self.key:
                raise ModelJsonError("动作局部 key 必须是非空字符串")
        if not isinstance(self.depends_on, tuple) or not all(
            isinstance(item, str) and item for item in self.depends_on
        ):
            raise ModelJsonError("depends_on 必须是非空字符串元组")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ModelJsonError("depends_on 不得重复引用")


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """一维全量结构化动作计划；合法空计划表示没有主动行动。"""

    actions: tuple[PlanAction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            raise ModelJsonError("动作计划必须是 PlanAction 元组")
        if not all(isinstance(item, PlanAction) for item in self.actions):
            raise ModelJsonError("动作计划每项必须是 PlanAction")
        keys = [item.key for item in self.actions if item.key is not None]
        if len(set(keys)) != len(keys):
            raise ModelJsonError("计划中存在重复的局部 key")
        ids = [item.action_id for item in self.actions if item.action_id is not None]
        if len(set(ids)) != len(ids):
            raise ModelJsonError("计划中存在重复的动作身份")
        if set(keys) & set(str(item) for item in ids):
            raise ModelJsonError("局部 key 与既有动作身份冲突（引用歧义）")
        key_index = {item.key: index for index, item in enumerate(self.actions) if item.key}
        id_index = {
            str(item.action_id): index
            for index, item in enumerate(self.actions)
            if item.action_id is not None
        }
        edges: dict[int, list[int]] = {}
        for index, item in enumerate(self.actions):
            targets: list[int] = []
            for ref in item.depends_on:
                if ref in key_index:
                    target = key_index[ref]
                elif ref in id_index:
                    target = id_index[ref]
                else:
                    continue  # 外部（历史）引用由角色按已知身份解析
                if target == index:
                    raise ModelJsonError(f"动作不能依赖自身：{ref}")
                if target in targets:
                    raise ModelJsonError(f"动作依赖重复指向同一目标：{ref}")
                targets.append(target)
            if targets:
                edges[index] = targets
        _reject_cycles(len(self.actions), edges)

    @property
    def is_empty(self) -> bool:
        return not self.actions

    def render(self) -> str:
        """稳定 display text；依赖以原始引用展示。"""
        if not self.actions:
            return "（空计划）"
        parts = []
        for item in self.actions:
            suffix = ""
            if item.action_id is not None:
                suffix += f"#{item.action_id}"
            if item.depends_on:
                suffix += f" 依赖[{', '.join(item.depends_on)}]"
            parts.append(item.action.display + suffix)
        return "；".join(parts)


def _reject_cycles(node_count: int, edges: dict[int, list[int]]) -> None:
    """拒绝局部引用构成的循环依赖。"""
    state = [0] * node_count  # 0=未访问 1=在栈 2=已完成

    def visit(node: int, path: list[int]) -> None:
        state[node] = 1
        for nxt in edges.get(node, []):
            if state[nxt] == 1:
                raise ModelJsonError(f"计划存在循环依赖：{path + [nxt]}")
            if state[nxt] == 0:
                visit(nxt, path + [nxt])
        state[node] = 2

    for node in range(node_count):
        if state[node] == 0:
            visit(node, [node])


@dataclass(frozen=True, slots=True)
class StepAction:
    """对外声明的当前原子动作；真实身份与已解析的真实依赖齐备。

    翻译器与 World 检查接收完整候选链中的原子动作；执行登记只接收最近
    步骤，不整批登记未来动作。``depends_on`` 是已解析到
    真实动作身份的依赖：World 已实际提交这些前置动作 completed 后，
    本动作才允许开始。
    """

    action_id: ActionId
    action: AtomicAction
    depends_on: tuple[ActionId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, AtomicAction):
            raise ValueError("当前步骤动作必须是 AtomicAction")
        require_identifier(self.action_id, "动作身份")
        if not isinstance(self.depends_on, tuple):
            raise ValueError("依赖必须是元组")
        for item in self.depends_on:
            require_identifier(item, "依赖动作身份")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("依赖动作身份必须唯一")
        if self.action_id in self.depends_on:
            raise ValueError("动作不能依赖自身")

    @property
    def description(self) -> str:
        """渲染 display text；供需要文本的既有接口，不是动作接口。"""
        return self.action.display


def parse_action_plan(raw: str, stage: str = "system3") -> ActionPlan:
    """把模型输出的一维 JSON 数组严格解析为 ``ActionPlan``。"""
    data = parse_json_text(raw, stage)
    if not isinstance(data, list):
        raise ModelJsonError(f"{stage} 输出必须是 JSON 一维数组")
    actions: list[PlanAction] = []
    for index, item in enumerate(data):
        where = f"{stage}[{index}]"
        if not isinstance(item, dict):
            raise ModelJsonError(f"{where} 必须是 JSON 对象")
        action_id: ActionId | None = None
        if "id" in item:
            raw_id = item["id"]
            if not isinstance(raw_id, str) or not raw_id:
                raise ModelJsonError(f"{where}.id 必须是非空字符串")
            action_id = ActionId(raw_id)
        key: str | None = None
        if "key" in item:
            key = _require_text(item["key"], f"{where}.key")
        depends_on: tuple[str, ...] = ()
        if "depends_on" in item:
            raw_deps = item["depends_on"]
            if not isinstance(raw_deps, list):
                raise ModelJsonError(f"{where}.depends_on 必须是数组")
            deps: list[str] = []
            for ref in raw_deps:
                deps.append(_require_text(ref, f"{where}.depends_on"))
            depends_on = tuple(deps)
        if "kind" not in item:
            raise ModelJsonError(f"{where} 动作缺少 kind（ID-only 对象非法）")
        payload = {
            name: value for name, value in item.items() if name not in {"id", "key", "depends_on"}
        }
        action = atomic_action_from_json(payload, where)
        actions.append(
            PlanAction(action=action, action_id=action_id, key=key, depends_on=depends_on)
        )
    return ActionPlan(actions=tuple(actions))
