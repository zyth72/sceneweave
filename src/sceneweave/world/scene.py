"""不可变场景事实模型与场景卡加载。

对象树按场景卡嵌套组合：children 表示组成或当前容纳关系，物品移动
改变父对象但不改变身份；position 是所在房间的绝对坐标（米），未给
坐标的子对象不臆造精确位置。加载时严格校验（重复身份、未知字段、
bool 冒充数值、非有限浮点、负尺寸、越界坐标、缺失字段），不把任意
未知 JSON 默默接受为事实。深层数据冻结：state 冻结为只读映射、子树
为不可变元组，快照之间不共享可变 dict/list。

事实变更只能经由结构化补丁（设置状态、改变父对象、设置坐标）在
候选校验提交后应用；补丁应用返回新快照，不修改原快照。角色认知/
意向不写入事实模型。可见性策略是最小语义边界：关闭的容器（state
的 open 为 false）隐藏其全部后代；blocks_sight 等更精细的视线判断
属于模型语义，不在此伪称精确视觉物理。
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sceneweave.domain.identity import require_identifier, require_text

StateValue = str | bool | int | float
_KNOWN_FIELDS = frozenset(
    {
        "id",
        "name",
        "kind",
        "unit",
        "size",
        "position",
        "state",
        "description",
        "blocks_movement",
        "blocks_sight",
        "children",
    }
)
_POSITION_TOLERANCE = 1e-9


class SceneDataError(ValueError):
    """场景卡数据未通过结构校验；不得默默接受为事实。"""


@dataclass(frozen=True, slots=True)
class SceneObject:
    """场景中一个对象的不可变事实。"""

    id: str
    name: str
    kind: str
    description: str = ""
    unit: str = "m"
    size: tuple[float, float] | None = None
    position: tuple[float, float] | None = None
    state: Mapping[str, StateValue] = field(default_factory=lambda: MappingProxyType({}))
    blocks_movement: bool = False
    blocks_sight: bool = False
    children: tuple[SceneObject, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.id, "对象身份")
        require_text(self.name, "对象名称")
        require_text(self.kind, "对象类别")
        if not isinstance(self.description, str):
            raise SceneDataError(f"{self.id} 的 description 必须是字符串")
        if not isinstance(self.unit, str) or not self.unit:
            raise SceneDataError(f"{self.id} 的 unit 必须是非空字符串")
        if self.unit != "m":
            raise SceneDataError(f"{self.id} 的 unit 仅支持米（m），得到：{self.unit}")
        for name, pair in (("size", self.size), ("position", self.position)):
            if pair is None:
                continue
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise SceneDataError(f"{self.id} 的 {name} 必须是 [x, y]")
            for component in pair:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise SceneDataError(f"{self.id} 的 {name} 分量必须是数值，得到：{component!r}")
                if not math.isfinite(component):
                    raise SceneDataError(f"{self.id} 的 {name} 分量必须有限")
            if name == "size" and (pair[0] < 0 or pair[1] < 0):
                raise SceneDataError(f"{self.id} 的 size 分量必须非负")
        if not isinstance(self.state, Mapping):
            raise SceneDataError(f"{self.id} 的 state 必须是对象")
        frozen_state: dict[str, StateValue] = {}
        for key, raw_value in self.state.items():
            if not isinstance(key, str) or not key:
                raise SceneDataError(f"{self.id} 的 state 键必须是非空字符串")
            value: StateValue | None = None
            if isinstance(raw_value, bool) or isinstance(raw_value, (int, float)):
                if isinstance(raw_value, float) and not math.isfinite(raw_value):
                    raise SceneDataError(f"{self.id} 的 state[{key}] 必须有限")
                value = raw_value
            elif isinstance(raw_value, str) and raw_value:
                value = raw_value
            if value is None:
                raise SceneDataError(f"{self.id} 的 state[{key}] 必须是非空字符串/布尔/有限数值")
            frozen_state[key] = value
        object.__setattr__(self, "state", MappingProxyType(frozen_state))
        if type(self.blocks_movement) is not bool or type(self.blocks_sight) is not bool:
            raise SceneDataError(f"{self.id} 的 blocks_* 必须是布尔值")
        if not isinstance(self.children, tuple) or not all(
            isinstance(item, SceneObject) for item in self.children
        ):
            raise SceneDataError(f"{self.id} 的 children 必须是对象元组")

    def is_closed_container(self) -> bool:
        """最小可见性策略：state 的 open 为 false 视为关闭容器。"""
        return self.state.get("open") is False

    def walk(self) -> Sequence[SceneObject]:
        """以自身为根的先序遍历。"""
        result: list[SceneObject] = [self]
        for child in self.children:
            result.extend(child.walk())
        return tuple(result)


def _require_number(value: Any, field_name: str, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneDataError(f"{owner} 的 {field_name} 必须是数值，得到：{value!r}")
    if not math.isfinite(value):
        raise SceneDataError(f"{owner} 的 {field_name} 必须有限")
    return float(value)


def _parse_object(
    data: Any, *, seen_ids: dict[str, None], root_size: tuple[float, float] | None
) -> SceneObject:
    if not isinstance(data, dict):
        raise SceneDataError("场景对象必须是 JSON 对象")
    unknown = sorted(set(data) - _KNOWN_FIELDS)
    if unknown:
        raise SceneDataError(f"场景对象包含未知字段：{unknown}")
    for required in ("id", "name", "kind"):
        if required not in data:
            raise SceneDataError(f"场景对象缺少必填字段：{required}")
    object_id = data["id"]
    require_identifier(object_id if isinstance(object_id, str) else "", "对象身份")
    if not isinstance(object_id, str) or not object_id:
        raise SceneDataError("对象身份必须是非空字符串")
    if object_id in seen_ids:
        raise SceneDataError(f"对象身份重复：{object_id}")
    seen_ids[object_id] = None
    size: tuple[float, float] | None = None
    if data.get("size") is not None:
        raw_size = data["size"]
        if not isinstance(raw_size, list) or len(raw_size) != 2:
            raise SceneDataError(f"{object_id} 的 size 必须是 [宽, 深]")
        size = (
            _require_number(raw_size[0], "size", object_id),
            _require_number(raw_size[1], "size", object_id),
        )
        if size[0] < 0 or size[1] < 0:
            raise SceneDataError(f"{object_id} 的 size 分量必须非负")
    position: tuple[float, float] | None = None
    if data.get("position") is not None:
        raw_position = data["position"]
        if not isinstance(raw_position, list) or len(raw_position) != 2:
            raise SceneDataError(f"{object_id} 的 position 必须是 [x, y]")
        position = (
            _require_number(raw_position[0], "position", object_id),
            _require_number(raw_position[1], "position", object_id),
        )
        if root_size is not None and (
            position[0] < -_POSITION_TOLERANCE
            or position[1] < -_POSITION_TOLERANCE
            or position[0] > root_size[0] + _POSITION_TOLERANCE
            or position[1] > root_size[1] + _POSITION_TOLERANCE
        ):
            raise SceneDataError(f"{object_id} 的 position 超出房间范围：{list(position)}")
    state: dict[str, StateValue] = {}
    raw_state = data.get("state", {})
    if not isinstance(raw_state, dict):
        raise SceneDataError(f"{object_id} 的 state 必须是对象")
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key:
            raise SceneDataError(f"{object_id} 的 state 键必须是非空字符串")
        if isinstance(value, bool) or isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise SceneDataError(f"{object_id} 的 state[{key}] 必须有限")
            state[key] = value
        elif isinstance(value, str) and value:
            state[key] = value
        else:
            raise SceneDataError(f"{object_id} 的 state[{key}] 必须是字符串/布尔/有限数值")
    children_raw = data.get("children", [])
    if not isinstance(children_raw, list):
        raise SceneDataError(f"{object_id} 的 children 必须是数组")
    children = tuple(
        _parse_object(child, seen_ids=seen_ids, root_size=root_size) for child in children_raw
    )
    return SceneObject(
        id=object_id,
        name=data["name"] if isinstance(data["name"], str) else "",
        kind=data["kind"] if isinstance(data["kind"], str) else "",
        description=data.get("description", ""),
        unit=data.get("unit", "m"),
        size=size,
        position=position,
        state=state,
        blocks_movement=data.get("blocks_movement", False),
        blocks_sight=data.get("blocks_sight", False),
        children=children,
    )


def load_scene_card(data: Any) -> SceneObject:
    """从场景卡数据构造不可变根对象；路径由调用方显式传入，导入不加载。"""
    if not isinstance(data, dict):
        raise SceneDataError("场景卡必须是 JSON 对象")
    seen_ids: dict[str, None] = {}
    root = _parse_object(data, seen_ids=seen_ids, root_size=None)
    # 根的尺寸用于校验绝对坐标范围；根自身不参与越界判断。
    if root.size is not None:
        for obj in root.walk():
            if obj is root or obj.position is None:
                continue
            if (
                obj.position[0] < -_POSITION_TOLERANCE
                or obj.position[1] < -_POSITION_TOLERANCE
                or obj.position[0] > root.size[0] + _POSITION_TOLERANCE
                or obj.position[1] > root.size[1] + _POSITION_TOLERANCE
            ):
                raise SceneDataError(f"{obj.id} 的 position 超出房间范围：{list(obj.position)}")
    return root


# ---- 结构化事实补丁 ----


_BOOL_STATE_KEYS = frozenset({"open", "locked"})


@dataclass(frozen=True, slots=True)
class SetStatePatch:
    """设置一个对象的状态键值；description 只是诊断文本。

    布尔语义的关键状态键（open/locked）只接受 bool：字符串或数值
    会使关闭容器的可见性判断失真。
    """

    object_id: str
    key: str
    value: StateValue
    description: str

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "补丁对象身份")
        require_text(self.key, "状态键")
        if self.key in _BOOL_STATE_KEYS and type(self.value) is not bool:
            raise SceneDataError(f"状态键 {self.key} 只接受布尔值，得到：{self.value!r}")
        if isinstance(self.value, bool):
            pass
        elif isinstance(self.value, float):
            if not math.isfinite(self.value):
                raise SceneDataError("状态值必须有限")
        elif not isinstance(self.value, (str, int)):
            raise SceneDataError(f"状态值类型不支持：{type(self.value).__name__}")
        if isinstance(self.value, str) and not self.value:
            raise SceneDataError("字符串状态值必须非空")
        require_text(self.description, "事实变更描述")


@dataclass(frozen=True, slots=True)
class MoveObjectPatch:
    """把一个对象移动到新的父对象下；身份不变，只改变容纳关系。"""

    object_id: str
    new_parent_id: str
    description: str

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "补丁对象身份")
        require_identifier(self.new_parent_id, "目标父对象身份")
        require_text(self.description, "事实变更描述")


@dataclass(frozen=True, slots=True)
class SetPositionPatch:
    """设置一个对象的房间绝对坐标。"""

    object_id: str
    position: tuple[float, float]
    description: str

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "补丁对象身份")
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise SceneDataError("坐标必须是 (x, y)")
        for value in self.position:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SceneDataError("坐标分量必须是数值")
            if not math.isfinite(value):
                raise SceneDataError("坐标分量必须有限")
        require_text(self.description, "事实变更描述")


type ScenePatch = SetStatePatch | MoveObjectPatch | SetPositionPatch


def _replace_child(
    parent: SceneObject, target_id: str, replacer: Callable[[SceneObject], SceneObject]
) -> SceneObject | None:
    """在子树中替换目标对象；返回新父对象，未找到返回 None。"""
    if parent.id == target_id:
        return replacer(parent)
    new_children: list[SceneObject] = []
    changed = False
    for child in parent.children:
        replacement = _replace_child(child, target_id, replacer)
        if replacement is not None:
            new_children.append(replacement)
            changed = True
        else:
            new_children.append(child)
    if not changed:
        return None
    return SceneObject(
        id=parent.id,
        name=parent.name,
        kind=parent.kind,
        description=parent.description,
        unit=parent.unit,
        size=parent.size,
        position=parent.position,
        state=parent.state,
        blocks_movement=parent.blocks_movement,
        blocks_sight=parent.blocks_sight,
        children=tuple(new_children),
    )


def _with_children(node: SceneObject, children: tuple[SceneObject, ...]) -> SceneObject:
    """同事实、新子级的不可变副本。"""
    return SceneObject(
        id=node.id,
        name=node.name,
        kind=node.kind,
        description=node.description,
        unit=node.unit,
        size=node.size,
        position=node.position,
        state=node.state,
        blocks_movement=node.blocks_movement,
        blocks_sight=node.blocks_sight,
        children=children,
    )


def _remove_node(node: SceneObject, target_id: str) -> SceneObject | None:
    """从子树摘除目标对象；返回新子树，目标不在其中时返回原子树。"""
    if node.id == target_id:
        return None  # 由调用方保证目标不是根
    new_children: list[SceneObject] = []
    changed = False
    for child in node.children:
        replacement = _remove_node(child, target_id)
        if replacement is None:
            changed = True  # 该子树中的目标已被摘除
            continue
        if replacement is not child:
            changed = True
        new_children.append(replacement)
    if not changed:
        return node
    return _with_children(node, tuple(new_children))


def _detach(root: SceneObject, target_id: str) -> tuple[SceneObject, SceneObject] | None:
    """从树中摘除目标对象；返回 (新根, 被摘除对象)。"""
    if root.id == target_id:
        raise SceneDataError("不能摘除根对象")
    target = find_object(root, target_id)
    if target is None:
        return None
    new_root = _remove_node(root, target_id)
    if new_root is None:  # pragma: no cover - 根已被排除
        raise SceneDataError("不能摘除根对象")
    return new_root, target


def find_object(root: SceneObject, object_id: str) -> SceneObject | None:
    """按身份查找对象；只读。"""
    if root.id == object_id:
        return root
    for child in root.children:
        found = find_object(child, object_id)
        if found is not None:
            return found
    return None


def _is_within(node: SceneObject, candidate_id: str) -> bool:
    if node.id == candidate_id:
        return True
    return any(_is_within(child, candidate_id) for child in node.children)


def apply_patch(root: SceneObject, patch: ScenePatch) -> SceneObject:
    """把一个已校验补丁应用为新快照；原快照不变。"""
    if isinstance(patch, SetStatePatch):

        def replacer(node: SceneObject) -> SceneObject:
            merged = dict(node.state)
            merged[patch.key] = patch.value
            return SceneObject(
                id=node.id,
                name=node.name,
                kind=node.kind,
                description=node.description,
                unit=node.unit,
                size=node.size,
                position=node.position,
                state=merged,
                blocks_movement=node.blocks_movement,
                blocks_sight=node.blocks_sight,
                children=node.children,
            )

        new_root = _replace_child(root, patch.object_id, replacer)
        if new_root is None:
            raise SceneDataError(f"补丁引用不存在的对象：{patch.object_id}")
        return new_root
    if isinstance(patch, SetPositionPatch):
        return _set_position(root, patch)
    # MoveObjectPatch：先在原树校验引用与祖先关系，再摘除挂接。
    target = find_object(root, patch.object_id)
    if target is None:
        raise SceneDataError(f"补丁引用不存在的对象：{patch.object_id}")
    if find_object(root, patch.new_parent_id) is None:
        raise SceneDataError(f"补丁引用不存在的父对象：{patch.new_parent_id}")
    if patch.new_parent_id == patch.object_id or _is_within(target, patch.new_parent_id):
        raise SceneDataError(
            f"不能把对象移动到自身或其后代下：{patch.object_id} → {patch.new_parent_id}"
        )
    detached = _detach(root, patch.object_id)
    if detached is None:  # pragma: no cover - 上面已校验存在
        raise SceneDataError(f"补丁引用不存在的对象：{patch.object_id}")
    new_root, moved = detached

    def add_child(node: SceneObject) -> SceneObject:
        return _with_children(node, (*node.children, moved))

    attached = _replace_child(new_root, patch.new_parent_id, add_child)
    if attached is None:
        raise SceneDataError(f"补丁引用不存在的父对象：{patch.new_parent_id}")
    return attached


def _set_position(root: SceneObject, patch: SetPositionPatch) -> SceneObject:
    if root.size is not None and (
        patch.position[0] < -_POSITION_TOLERANCE
        or patch.position[1] < -_POSITION_TOLERANCE
        or patch.position[0] > root.size[0] + _POSITION_TOLERANCE
        or patch.position[1] > root.size[1] + _POSITION_TOLERANCE
    ):
        raise SceneDataError(f"补丁把 {patch.object_id} 移出房间范围：{list(patch.position)}")

    def replacer(node: SceneObject) -> SceneObject:
        return SceneObject(
            id=node.id,
            name=node.name,
            kind=node.kind,
            description=node.description,
            unit=node.unit,
            size=node.size,
            position=patch.position,
            state=node.state,
            blocks_movement=node.blocks_movement,
            blocks_sight=node.blocks_sight,
            children=node.children,
        )

    new_root = _replace_child(root, patch.object_id, replacer)
    if new_root is None:
        raise SceneDataError(f"补丁引用不存在的对象：{patch.object_id}")
    return new_root


def is_visible(root: SceneObject, object_id: str) -> bool:
    """最小可见性策略：祖先链上没有关闭容器即可见。

    更精细的视线遮挡（blocks_sight、朝向、距离）属于模型语义判断，
    这里不伪称精确视觉物理。
    """
    path = _path_to(root, object_id)
    if path is None:
        return False
    return not any(node.is_closed_container() for node in path[:-1])


def _path_to(node: SceneObject, object_id: str) -> list[SceneObject] | None:
    if node.id == object_id:
        return [node]
    for child in node.children:
        path = _path_to(child, object_id)
        if path is not None:
            return [node, *path]
    return None


def validate_scene_tree(root: SceneObject) -> None:
    """校验整棵树：身份唯一、结构不可变。供直接构造入口复用。"""
    seen: set[str] = set()
    for obj in root.walk():
        if obj.id in seen:
            raise SceneDataError(f"对象身份重复：{obj.id}")
        seen.add(obj.id)
        if not isinstance(obj.children, tuple):
            raise SceneDataError(f"{obj.id} 的 children 必须是元组")
        if obj is not root and obj.position is not None and root.size is not None:
            if not (0 <= obj.position[0] <= root.size[0] and 0 <= obj.position[1] <= root.size[1]):
                raise SceneDataError(f"{obj.id} 的 position 超出房间范围")


def scene_role_ids(root: SceneObject) -> tuple[str, ...]:
    """场景中的角色对象身份（kind 为角色）。"""
    return tuple(obj.id for obj in root.walk() if obj.kind == "角色")


def load_scene_card_json(text: str) -> SceneObject:
    """从场景卡 JSON 文本加载；重复键直接拒绝（不用默认丢重复的 loads）。"""
    import json

    from sceneweave.domain.model_json import ModelJsonError, reject_duplicate_keys

    try:
        data = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise SceneDataError(f"场景卡不是合法 JSON：{error}") from error
    except ModelJsonError as error:
        raise SceneDataError(f"场景卡 JSON 结构非法：{error}") from error
    return load_scene_card(data)


def describe_scene(root: SceneObject, *, only_ids: Sequence[str] | None = None) -> str:
    """场景的文本摘要（供模型提示）；可按对象集合裁剪。

    包含尺寸、状态、通行/视线阻挡与容纳关系；坐标只作语义辅助。
    """
    lines: list[str] = []
    for obj in root.walk():
        if only_ids is not None and obj.id not in only_ids:
            continue
        parts = [f"{obj.id}（{obj.name}，{obj.kind}）"]
        if obj.position is not None:
            parts.append(f"位置 ({obj.position[0]:.2f}, {obj.position[1]:.2f})")
        if obj.size is not None:
            parts.append(f"尺寸 ({obj.size[0]:.2f}×{obj.size[1]:.2f})")
        if obj.state:
            parts.append("状态 " + ", ".join(f"{k}={v}" for k, v in sorted(obj.state.items())))
        if obj.blocks_movement:
            parts.append("阻挡通行")
        if obj.blocks_sight:
            parts.append("遮挡视线")
        parent = parent_of(root, obj.id)
        if parent is not None and (only_ids is None or parent.id in only_ids):
            parts.append(f"在 {parent.name} 之中")
        if obj.description:
            parts.append(obj.description)
        lines.append("- " + "；".join(parts))
    return "\n".join(lines)


def parent_of(root: SceneObject, object_id: str) -> SceneObject | None:
    """返回对象的直接父对象；根或未知对象返回 None。"""

    def search(node: SceneObject) -> SceneObject | None:
        for child in node.children:
            if child.id == object_id:
                return node
            found = search(child)
            if found is not None:
                return found
        return None

    return search(root)
