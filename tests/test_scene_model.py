"""不可变场景事实模型与场景卡加载的行为测试。"""

import math
from types import MappingProxyType

import pytest

from sceneweave.world.scene import (
    MoveObjectPatch,
    SceneDataError,
    SceneObject,
    SetPositionPatch,
    SetStatePatch,
    apply_patch,
    describe_scene,
    find_object,
    is_visible,
    load_scene_card,
)


def study_card() -> dict:
    """与用户在途场景同型的最小书房（内联数据，不依赖未提交文件）。"""
    return {
        "id": "room",
        "name": "书房",
        "kind": "房间",
        "unit": "m",
        "size": [4, 3],
        "children": [
            {
                "id": "lin",
                "name": "林",
                "kind": "角色",
                "position": [0.6, 0.6],
                "state": {"posture": "站立"},
                "children": [],
            },
            {
                "id": "desk",
                "name": "书桌",
                "kind": "家具",
                "position": [2.8, 2.2],
                "size": [1.2, 0.6],
                "blocks_movement": True,
                "children": [
                    {
                        "id": "drawer",
                        "name": "抽屉",
                        "kind": "容器",
                        "state": {"open": False, "locked": False},
                        "children": [{"id": "key", "name": "铜钥匙", "kind": "物品"}],
                    },
                ],
            },
        ],
    }


def test_load_freezes_deep_structure() -> None:
    root = load_scene_card(study_card())
    drawer = find_object(root, "drawer")
    assert drawer is not None and drawer.is_closed_container()
    key = find_object(root, "key")
    assert key is not None and key.name == "铜钥匙"
    # 深层冻结：state 是只读映射，children 是元组。
    assert isinstance(drawer.state, MappingProxyType)
    assert isinstance(root.children, tuple)
    with pytest.raises(TypeError):
        drawer.state["open"] = True  # type: ignore[index]


def test_duplicate_ids_rejected() -> None:
    card = study_card()
    card["children"][1]["children"][0]["children"][0]["id"] = "lin"
    with pytest.raises(SceneDataError, match="重复"):
        load_scene_card(card)


def test_unknown_fields_rejected() -> None:
    card = study_card()
    card["children"][0]["surprise"] = 1
    with pytest.raises(SceneDataError, match="未知字段"):
        load_scene_card(card)


def test_bool_and_nonfinite_numbers_rejected() -> None:
    card = study_card()
    card["children"][0]["position"] = [True, 0.6]
    with pytest.raises(SceneDataError, match="数值"):
        load_scene_card(card)
    card2 = study_card()
    card2["children"][0]["position"] = [float("nan"), 0.6]
    with pytest.raises(SceneDataError, match="有限"):
        load_scene_card(card2)
    card3 = study_card()
    card3["size"] = [-1, 3]
    with pytest.raises(SceneDataError, match="非负"):
        load_scene_card(card3)


def test_out_of_bounds_position_rejected() -> None:
    card = study_card()
    card["children"][0]["position"] = [5.0, 0.6]
    with pytest.raises(SceneDataError, match="超出房间范围"):
        load_scene_card(card)


def test_missing_required_fields_rejected() -> None:
    card = study_card()
    del card["children"][0]["kind"]
    with pytest.raises(SceneDataError, match="必填字段"):
        load_scene_card(card)
    with pytest.raises(SceneDataError, match="JSON 对象"):
        load_scene_card([1, 2, 3])


def test_set_state_patch_immutable() -> None:
    root = load_scene_card(study_card())
    new_root = apply_patch(root, SetStatePatch("drawer", "open", True, "抽屉被拉开"))
    assert find_object(root, "drawer").state["open"] is False, "原快照不变"
    assert find_object(new_root, "drawer").state["open"] is True
    assert find_object(new_root, "key").name == "铜钥匙", "后代保留"


def test_move_patch_changes_parent_keeps_identity() -> None:
    root = load_scene_card(study_card())
    new_root = apply_patch(root, MoveObjectPatch("key", "lin", "林把钥匙放进口袋"))
    key = find_object(new_root, "key")
    assert key is not None and key.name == "铜钥匙", "身份不变"
    lin = find_object(new_root, "lin")
    assert any(child.id == "key" for child in lin.children)
    drawer = find_object(new_root, "drawer")
    assert all(child.id != "key" for child in drawer.children)


def test_move_to_own_descendant_rejected() -> None:
    root = load_scene_card(study_card())
    with pytest.raises(SceneDataError, match="自身或其后代"):
        apply_patch(root, MoveObjectPatch("desk", "key", "非法移动"))


def test_patch_unknown_object_rejected() -> None:
    root = load_scene_card(study_card())
    with pytest.raises(SceneDataError, match="不存在的对象"):
        apply_patch(root, SetStatePatch("ghost", "open", True, "幽灵"))


def test_position_patch_and_bounds_free() -> None:
    root = load_scene_card(study_card())
    new_root = apply_patch(root, SetPositionPatch("lin", (1.5, 1.0), "林走到房间中央"))
    assert find_object(new_root, "lin").position == (1.5, 1.0)
    assert math.isfinite(find_object(new_root, "lin").position[0])


def test_visibility_closed_container_hides_descendants() -> None:
    root = load_scene_card(study_card())
    assert is_visible(root, "drawer")
    assert not is_visible(root, "key"), "关闭抽屉里的钥匙不可见"
    opened = apply_patch(root, SetStatePatch("drawer", "open", True, "打开"))
    assert is_visible(opened, "key"), "打开后可见"


def test_describe_scene_supports_projection() -> None:
    root = load_scene_card(study_card())
    full = describe_scene(root)
    assert "铜钥匙" in full
    projected = describe_scene(root, only_ids=["room", "lin", "desk", "drawer"])
    assert "铜钥匙" not in projected, "投影裁掉不可见对象"


def test_scene_object_direct_construction_validates() -> None:
    with pytest.raises(ValueError, match="对象身份"):
        SceneObject(id="", name="x", kind="y")
    with pytest.raises(SceneDataError):
        SceneObject(id="a", name="x", kind="y", size=(-1, 2))
    with pytest.raises(SceneDataError):
        SceneObject(id="a", name="x", kind="y", blocks_movement="yes")
