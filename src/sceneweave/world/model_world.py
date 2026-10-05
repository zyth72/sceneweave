"""模型驱动的 World：具体场景事实模型上的检查、进展、裁决与发布。

ModelWorld 完整实现 world.base.World 接口：客观事实是不可变场景对象
树（world/scene.py），一切实际变化都经结构化补丁在候选校验提交后原
子应用；未经提交的候选不改变快照、版本或进展基线。语义判断（可执
行性、联合裁决、进展位置、修改实际作用）经注入采样器调用模型，输
出按严格 JSON schema 解析（未知/重复字段、非有限数值、bool 冒充数值
等都是运行错误，不伪空、不修补、不自动 fallback）。

采样作用域：world-check-{角色}、world-progress、world-adjudicate，
每作用域只构造一次采样器并缓存；不同角色检查可并行，同作用域并发
是明确错误，写入阶段（进展/裁决）另有占用保护，commit 在有模型调用
在途时拒绝执行。感知发布是同步的确定性投影：只发布已提交事实变更，
按提交后快照的可见范围过滤（关闭容器隐藏全部后代），不广播全量场
景，也不把模型的全量裁决总结塞给角色；检查提示按受众裁掉不可见对
象，隐藏内容不进入模型输入。检查打回原因来自基于可见投影的模型判
断，不泄漏不可感知事实；错误现场只入诊断日志。

留痕：每次模型调用的完整提示与原始输出、解析候选、实际事实差异、
发布受众、跳过、取消与错误（含 error_kind 与 raw）写入同一
SimulationLog；成功链追踪失败传播，错误链追踪失败以 add_note 保留
原异常。生产模型参数未在真实 API 上验证（测试与演示使用脚本假模型）。
"""

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sceneweave.domain.adjudication import (
    ActionModificationNode,
    ActionNodeKind,
    ActionOutcome,
    ActionOutcomeResult,
    AdjudicationActionNode,
    AdjudicationCandidate,
    AdjudicationCommit,
    AdjudicationValidation,
    FactChangeProposal,
    ObjectiveContext,
    ProgressUpdate,
    WorldActionState,
    WorldDiagnostic,
)
from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    Delivery,
    ModifiedAction,
    NewAction,
    Observation,
    RoleResponse,
    WorldEvent,
)
from sceneweave.domain.model_json import (
    ModelJsonError,
    parse_model_json,
    require_json_int,
)
from sceneweave.domain.modeling import TextSampler
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TraceValue, TurnTracer
from sceneweave.world.base import World
from sceneweave.world.scene import (
    MoveObjectPatch,
    SceneDataError,
    SceneObject,
    ScenePatch,
    SetPositionPatch,
    SetStatePatch,
    StateValue,
    apply_patch,
    describe_scene,
    find_object,
    is_visible,
    scene_role_ids,
    validate_scene_tree,
)

CHECK_SCHEMA = (
    '{"passed": true/false, "reason": "不通过时的原因", "action_ids": ["有关行动身份", ...]}'
)
PROGRESS_SCHEMA = (
    '{"movements": [{"object_id": "对象", "position": [x, y]}, ...],'
    ' "updated_action_ids": ["有实际进展的行动身份", ...],'
    ' "notes": ["进展说明", ...],'
    ' "passive_updates": [{"id": "被动过程身份", "note": "说明", "done": false}, ...]}'
    "——四个字段全部必填（数组可为空；done 省略为 false）。可选 changes 数组沿用"
    "联合裁决的结构补丁；可选 action_progress 数组只更新实际进展，不改变行动阶段："
    '[{"action_id": "进行中的行动", "detail": "实际进展"}]。'
)
ADJUDICATE_SCHEMA = (
    '{"changes": [{"kind": "set_state", "object_id": "对象", "key": "键",'
    ' "value": 值, "description": "诊断描述"} |'
    ' {"kind": "set_position", "object_id": "对象", "position": [x, y],'
    ' "description": "诊断描述"} |'
    ' {"kind": "move", "object_id": "对象", "new_parent": "目标父对象",'
    ' "description": "诊断描述"}],'
    ' "outcomes": [{"action_id": "参与行动身份",'
    ' "result": "started|in_progress|completed|failed|unconfirmed",'
    ' "detail": "结果说明"}],'
    ' "follow_ups": [{"event_id": "事件身份", "time_ms": 整数毫秒,'
    ' "description": "未来世界事件"}],'
    ' "movements": [{"action_id": "参与行动身份", "role_id": "执行角色",'
    ' "speed": 非负数米每秒, "direction": [dx, dy]（非零，将归一化）}],'
    ' "stop_movements": ["结束运动的行动身份", ...],'
    ' "passives": [{"id": "被动过程身份", "description": "被动影响描述",'
    ' "object_id": "被推动对象（可选）", "speed": 米每秒（可选）,'
    ' "direction": [dx, dy]（可选）}],'
    ' "stop_passives": ["结束的被动过程身份", ...]}——七个字段全部必填'
)


class ModelWorldError(RuntimeError):
    """World 侧结构/关联错误；属于运行错误，不得伪空或假通过。"""


@dataclass(frozen=True, slots=True)
class MovementState:
    """一个行动正在驱动的实际运动（米/秒与单位方向）。"""

    action_id: ActionId
    role_id: RoleId
    speed: float
    direction: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ActualAction:
    """World 对一个行动的实际结算状态（客观，非角色认知）。"""

    role_id: RoleId
    description: str
    result: str
    detail: str


@dataclass(frozen=True, slots=True)
class IssuedCandidate:
    """World 为某个候选保留的生成现场；候选提交前不构成事实。

    candidate 是生成时的候选对象本身：validate/commit 以 is 身份核对，
    值相等的伪造对象（dataclasses.replace）不能冒充。内部集合全部是
    不可变元组/只读映射。
    """

    candidate: AdjudicationCandidate
    base_version: WorldStateVersion
    base_time: SimulationTime
    patches: tuple[ScenePatch, ...]
    movements: tuple[MovementState, ...]
    stop_movements: tuple[ActionId, ...]
    passives: tuple[tuple[str, str], ...]
    passive_movements: tuple[tuple[str, str, float, tuple[float, float]], ...]
    stop_passives: tuple[str, ...]
    reference_event_ids: tuple[EventId, ...]
    node_facts: Mapping[ActionId, tuple[RoleId, str]]
    withdrawn_actions: tuple[ActionId, ...]
    perception_patches: tuple[ScenePatch, ...] = ()


def _require_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelWorldError(f"{field} 必须是数值，得到：{value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ModelWorldError(f"{field} 必须有限")
    return result


def _require_xy(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ModelWorldError(f"{field} 必须是 [x, y]")
    return (_require_float(value[0], f"{field}[0]"), _require_float(value[1], f"{field}[1]"))


def _normalize_direction(direction: tuple[float, float]) -> tuple[float, float]:
    """方向向量归一化为单位长度；零向量拒绝。"""
    length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    if length <= 1e-9:
        raise ModelWorldError("方向向量不能是零向量")
    return (direction[0] / length, direction[1] / length)


def _state_value(value: Any, field: str) -> StateValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ModelWorldError(f"{field} 必须有限")
        return value
    raise ModelWorldError(f"{field} 必须是字符串/布尔/有限数值")


@dataclass(frozen=True, slots=True)
class _ProgressPlan:
    """一次解析得到的类型化进展计划；应用前保持不可变。"""

    movements: tuple[tuple[str, tuple[float, float]], ...]
    updated_action_ids: tuple[ActionId, ...]
    notes: tuple[str, ...]
    passive_notes: tuple[tuple[str, str], ...]
    passives_done: tuple[str, ...]
    patches: tuple[ScenePatch, ...] = ()
    action_progress: tuple[tuple[ActionId, str], ...] = ()


class ModelWorld(World):
    """以注入采样器驱动语义判断的生产 World。"""

    def __init__(
        self,
        *,
        scene: SceneObject,
        sampler_for_scope: Callable[[str], TextSampler],
        roles: Sequence[RoleId] | None = None,
        visibility: Callable[[RoleId, SceneObject, str], bool] | None = None,
        tracer: TurnTracer | None = None,
    ) -> None:
        if not isinstance(scene, SceneObject):
            raise ModelWorldError("scene 必须是 SceneObject")
        validate_scene_tree(scene)
        self._scene = scene
        self._sampler_for_scope = sampler_for_scope
        self._tracer = tracer
        # 受众感知策略：默认为关闭容器隐藏后代（对全部角色一致）；可注入
        # 更精细的语义判断。坐标与 blocks_* 只作语义输入，不做精确视觉
        # 物理或寻路。
        self._visibility = visibility or (
            lambda _role, root, object_id: is_visible(root, object_id)
        )
        # 参与本运行的角色（默认场景内全部角色对象）；发布与上下文按此限定。
        scene_roles = scene_role_ids(scene)
        self._roles: tuple[RoleId, ...] = (
            tuple(roles) if roles is not None else tuple(RoleId(name) for name in scene_roles)
        )
        if not self._roles:
            raise ModelWorldError("场景必须至少包含一个参与角色")
        if len(set(self._roles)) != len(self._roles):
            raise ModelWorldError("参与角色身份必须唯一")
        for role in self._roles:
            if str(role) not in scene_roles:
                raise ModelWorldError(f"参与角色不是场景中的角色对象：{role}")
        self._version = WorldStateVersion(0)
        self._last_time = SimulationTime(0)
        self._movements: dict[ActionId, MovementState] = {}
        self._actual: dict[ActionId, ActualAction] = {}
        self._passives: dict[str, str] = {}
        self._issued: list[IssuedCandidate] = []
        self._committed_candidates: list[AdjudicationCandidate] = []
        self._committed: list[AdjudicationCommit] = []
        self._samplers: dict[str, TextSampler] = {}
        self._current_time = SimulationTime(0)
        self._passive_movers: dict[str, str] = {}
        self._passive_movements: dict[str, tuple[str, float, tuple[float, float]]] = {}
        self._in_flight: set[str] = set()
        self._write_busy = False
        self._progress_for: tuple[CycleId, SimulationTime] | None = None
        self._progress_patches: tuple[ScenePatch, ...] = ()

    # ---- 只读观察 ----

    @property
    def state_version(self) -> WorldStateVersion:
        return self._version

    @property
    def scene(self) -> SceneObject:
        """当前不可变场景根；快照间不共享可变结构。"""
        return self._scene

    def scene_snapshot(self) -> SceneObject:
        """只读诊断快照（测试/界面用）；版本归 World。"""
        return self._scene

    def _role_ids(self) -> tuple[RoleId, ...]:
        return self._roles

    def _sampler_for(self, scope: str) -> TextSampler:
        cached = self._samplers.get(scope)
        if cached is not None:
            return cached
        built: TextSampler = self._sampler_for_scope(scope)
        self._samplers[scope] = built
        return built

    # ---- world.base.World 接口 ----

    async def check_response(self, response: RoleResponse) -> ActionCheckResult:
        """基于共同版本对完整主动集合的无副作用检查。

        提示只携带该角色可见的场景投影与自身行动状态；不同角色并行、
        同角色串行。语义由模型判断，结构错误是运行错误。
        """
        scope = f"world-check-{response.role_id}"
        # 检查依据进入时的共同版本与已知剧情时刻；等待期间事实冻结。
        base_version = self._version
        base_time = self._current_time
        prompt = self._check_prompt(response)
        raw: str | None = None
        self._require_scope_free(scope)
        self._require_no_writer("检查")
        self._in_flight.add(scope)
        error_kind = "追踪"
        try:
            self._trace(
                "world.check.invoke",
                response.cycle_id,
                response.response_id,
                "World 检查模型调用",
                summary_time=base_time,
                role_id=response.role_id,
                action_ids=tuple(i.action_id for i in response.actions),
                inputs=(prompt,),
                outputs=(),
                details={"scope": scope, "world_state_version": self._version},
            )
            error_kind = "采样"
            raw = await self._sampler_for(scope).sample(prompt, component="world-check")
            error_kind = "解析"
            data = parse_model_json(
                raw, "world-check", allowed_keys=("passed", "reason", "action_ids")
            )
            for required in ("passed", "reason", "action_ids"):
                if required not in data:
                    raise ModelWorldError(f"world-check 输出缺少必填字段 {required}")
            if type(data["passed"]) is not bool:
                raise ModelWorldError("passed 必须是布尔值")
            passed = data["passed"]
            reason = data["reason"]
            if not isinstance(reason, str):
                raise ModelWorldError("reason 必须是字符串")
            if not passed and not reason:
                raise ModelWorldError("不通过时必须提供原因")
            raw_ids = data["action_ids"]
            if not isinstance(raw_ids, list):
                raise ModelWorldError("action_ids 必须是数组")
            declared = {item.action_id for item in response.actions}
            related: list[ActionId] = []
            for item in raw_ids:
                if not isinstance(item, str) or not item:
                    raise ModelWorldError("action_ids 每项必须是非空字符串")
                if item not in declared:
                    raise ModelWorldError(f"检查引用了声明外的行动：{item}")
                related.append(ActionId(item))
            self._trace(
                "world.check.parsed",
                response.cycle_id,
                response.response_id,
                f"World 检查{'通过' if passed else '未通过'}",
                summary_time=base_time,
                role_id=response.role_id,
                action_ids=tuple(related),
                inputs=(),
                outputs=(raw,),
                details={"passed": passed, "reason": reason},
            )
            if self._version != base_version:
                raise ModelWorldError("检查期间共同状态被修改，检查前提不再成立")
            return ActionCheckResult(
                cycle_id=response.cycle_id,
                role_id=response.role_id,
                response_id=response.response_id,
                world_state_version=base_version,
                passed=passed,
                reason=reason,
                action_ids=tuple(related),
            )
        except BaseException as error:
            self._trace_failure(
                "world.check.error",
                response.cycle_id,
                summary_time=base_time,
                role_id=response.role_id,
                response_id=response.response_id,
                action_ids=tuple(i.action_id for i in response.actions),
                prompt=prompt,
                raw=raw,
                error_kind=error_kind if not isinstance(error, asyncio.CancelledError) else "取消",
                error=error,
            )
            raise
        finally:
            self._in_flight.discard(scope)

    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        """按已流逝剧情时间更新实际进展；一次解析、完整构造后原子切换。

        覆盖运动、持续中的非移动行动与被动过程（含被动驱动的惯性移动、
        显式结束被动）；省略不等于删除。失败/取消不改基线与任何事实；
        无实际变化不递增版本，且 updated_action_ids 必为空。
        """
        prompt = "（未进入模型调用）"
        raw: str | None = None
        self._require_write_free()
        self._require_no_readers("进展更新")
        self._write_busy = True
        error_kind = "追踪"
        try:
            elapsed = simulation_time - self._last_time
            has_ongoing = bool(
                self._movements
                or self._passives
                or any(
                    actual.result
                    in (
                        ActionOutcomeResult.STARTED.value,
                        ActionOutcomeResult.IN_PROGRESS.value,
                        ActionOutcomeResult.UNCONFIRMED.value,
                    )
                    for actual in self._actual.values()
                )
            )
            if elapsed < 0:
                raise ModelWorldError("剧情时刻不能倒退")
            if elapsed == 0 or not has_ongoing:
                update = ProgressUpdate(cycle_id=cycle_id, simulation_time=simulation_time)
                self._trace(
                    "world.progress.skip",
                    cycle_id,
                    None,
                    "无流逝时间或无进行中的实际过程，跳过模型调用",
                    summary_time=simulation_time,
                    inputs=(),
                    outputs=(),
                    details={"elapsed_ms": elapsed},
                )
                self._last_time = simulation_time
                self._current_time = simulation_time
                self._progress_for = (cycle_id, simulation_time)
                self._progress_patches = ()
                return update
            prompt = self._progress_prompt(elapsed, simulation_time)
            self._trace(
                "world.progress.invoke",
                cycle_id,
                None,
                "World 进展模型调用",
                summary_time=simulation_time,
                inputs=(prompt,),
                outputs=(),
                details={"elapsed_ms": elapsed},
            )
            error_kind = "采样"
            raw = await self._sampler_for("world-progress").sample(
                prompt, component="world-progress"
            )
            error_kind = "解析"
            progress = self._parse_progress(raw)
            error_kind = "应用"
            # 在工作副本上构造全部新状态；任何失败不触碰当前事实。
            new_scene = self._scene
            moved_any = False
            perception_patches: list[ScenePatch] = []
            for object_id, position in progress.movements:
                target = find_object(new_scene, object_id)
                if target is None:
                    raise ModelWorldError(f"进展引用未知对象：{object_id}")
                if target.position is not None and (
                    abs(target.position[0] - position[0]) < 1e-9
                    and abs(target.position[1] - position[1]) < 1e-9
                ):
                    continue  # 无实际变化的移动不产生事实差异
                new_scene = apply_patch(
                    new_scene,
                    SetPositionPatch(
                        object_id=object_id,
                        position=position,
                        description=f"{target.name}移动到 ({position[0]:.2f}, {position[1]:.2f})",
                    ),
                )
                moved_any = True
                perception_patches.append(SetPositionPatch(object_id, position, "实际位置变化"))
            for patch in progress.patches:
                prior_scene = new_scene
                new_scene = apply_patch(new_scene, patch)
                if prior_scene != new_scene:
                    perception_patches.append(patch)
            new_actual = dict(self._actual)
            for action_id, detail in progress.action_progress:
                actual = new_actual[action_id]
                new_actual[action_id] = ActualAction(
                    actual.role_id, actual.description, actual.result, detail
                )
            for passive_id, _note in progress.passive_notes:
                if passive_id not in self._passives:
                    raise ModelWorldError(f"进展引用未知被动过程：{passive_id}")
            new_passives = dict(self._passives)
            for passive_id, passive_note in progress.passive_notes:
                new_passives[passive_id] = passive_note
            for passive_id in progress.passives_done:
                if passive_id not in new_passives:
                    raise ModelWorldError(f"进展引用未知被动过程：{passive_id}")
                del new_passives[passive_id]
            # updated_action_ids 必须与实际变化一致。
            new_passive_movers = {
                key: value
                for key, value in self._passive_movers.items()
                if key not in progress.passives_done
            }
            new_passive_movements = {
                key: value
                for key, value in self._passive_movements.items()
                if key not in progress.passives_done
            }
            changed = (
                new_scene != self._scene
                or new_actual != self._actual
                or new_passives != self._passives
            )
            updated_ids = tuple(progress.updated_action_ids) if changed else ()
            update = ProgressUpdate(
                cycle_id=cycle_id,
                simulation_time=simulation_time,
                updated_action_ids=updated_ids,
                notes=tuple(progress.notes),
            )
            # 成功日志与报告构造先完成；之后只切换已验证的本地状态。
            new_version = WorldStateVersion(self._version + int(changed))
            self._trace(
                "world.progress.applied",
                cycle_id,
                None,
                "实际进展已原子应用" if changed else "模型报告无实际变化",
                summary_time=simulation_time,
                inputs=(),
                outputs=(raw,) + tuple(progress.notes),
                world_state_version=new_version,
                action_ids=updated_ids,
                details={
                    "moved": moved_any,
                    "passive_updates": len(progress.passive_notes),
                    "passives_done": len(progress.passives_done),
                    "version": new_version,
                },
            )
            self._scene = new_scene
            self._actual = new_actual
            self._passives = new_passives
            self._passive_movers = new_passive_movers
            self._passive_movements = new_passive_movements
            self._version = new_version
            self._last_time = simulation_time
            self._current_time = simulation_time
            self._progress_for = (cycle_id, simulation_time)
            self._progress_patches = tuple(perception_patches)
            return update
        except BaseException as error:
            self._trace_failure(
                "world.progress.error",
                cycle_id,
                summary_time=simulation_time,
                prompt=prompt,
                raw=raw,
                error_kind=error_kind if not isinstance(error, asyncio.CancelledError) else "取消",
                error=error,
            )
            raise
        finally:
            self._write_busy = False

    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        """该角色的只读客观条件；不进入角色感知。

        包含房间尺寸、可见对象的空间与阻挡属性、容纳关系（复用
        describe_scene 只读投影，与角色认知隔离）、实际运动与持续
        行动进展；坐标只作语义辅助。
        """
        if role_id not in self._roles:
            raise ModelWorldError(f"未知角色：{role_id}")
        conditions: list[str] = []
        if self._scene.size is not None:
            conditions.append(f"房间尺寸 {self._scene.size[0]:.1f}×{self._scene.size[1]:.1f} 米")
        role_object = find_object(self._scene, role_id)
        if role_object is not None:
            if role_object.position is not None:
                conditions.append(
                    f"{role_object.name}位置 ({role_object.position[0]:.2f},"
                    f" {role_object.position[1]:.2f}) 米"
                )
            posture = role_object.state.get("posture")
            if posture is not None:
                conditions.append(f"姿态：{posture}")
        visible_ids = [
            obj.id
            for obj in self._scene.walk()
            if obj.id != str(role_id) and self._is_visible_to(role_id, obj.id)
        ]
        if visible_ids:
            conditions.append(
                "可见对象（含空间、阻挡与容纳）：\n"
                + describe_scene(self._scene, only_ids=sorted(visible_ids))
            )
        for action_id, movement in sorted(self._movements.items(), key=lambda kv: str(kv[0])):
            if movement.role_id == role_id:
                conditions.append(
                    f"{action_id} 正在以 {movement.speed} 米/秒沿方向"
                    f" ({movement.direction[0]:.2f}, {movement.direction[1]:.2f}) 实际移动"
                )
        for action_id, actual in sorted(self._actual.items(), key=lambda kv: str(kv[0])):
            if actual.role_id == role_id and actual.result in (
                "started",
                "in_progress",
                "unconfirmed",
            ):
                conditions.append(f"{action_id} 实际进行中：{actual.detail}")
        return ObjectiveContext(
            world_state_version=self._version,
            role_id=role_id,
            conditions=tuple(conditions) or ("场景中暂无可述条件",),
        )

    def diagnostic_snapshot(self) -> WorldDiagnostic:
        """客观行动状态与被动过程的只读诊断；不含私有认知。"""
        action_states = tuple(
            WorldActionState(
                action_id=action_id,
                role_id=actual.role_id,
                state=actual.result,
                detail=actual.detail,
            )
            for action_id, actual in sorted(self._actual.items(), key=lambda kv: str(kv[0]))
        )
        passives = tuple(
            f"{pid}：{description}" for pid, description in sorted(self._passives.items())
        )
        return WorldDiagnostic(
            world_state_version=self._version,
            action_states=action_states,
            passive_processes=passives,
        )

    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
        events: Sequence[WorldEvent] = (),
        action_nodes: Sequence[AdjudicationActionNode] = (),
        modifications: Sequence[ActionModificationNode] = (),
    ) -> AdjudicationCandidate:
        """同刻事件与行动的一次联合裁决；只生成候选，不改任何事实。

        所有参与行动（含 start/end/withdrawn 节点）必须有明确结算反馈：
        start/checkpoint/end 需 outcome；withdrawn 可无 outcome（撤回
        主动控制不是伪终态）。运动必须归属参与行动的执行角色且行动
        非终态；终态结果的行动不得再启动运动；UNCONFIRMED 不隐式停止
        已知运动。同对象冲突补丁整批拒绝。
        """
        prompt = "（未进入模型调用）"
        raw: str | None = None
        self._require_write_free()
        self._require_no_readers("联合裁决")
        self._write_busy = True
        error_kind = "追踪"
        try:
            prompt = self._adjudicate_prompt(simulation_time, events, action_nodes, modifications)
            self._trace(
                "world.adjudicate.invoke",
                cycle_id,
                None,
                "World 联合裁决模型调用",
                summary_time=simulation_time,
                event_ids=tuple(event_ids),
                action_ids=tuple(action_ids),
                inputs=(prompt,),
                outputs=(),
                details={
                    "events": len(events),
                    "nodes": len(action_nodes),
                    "modifications": len(modifications),
                },
            )
            error_kind = "采样"
            raw = await self._sampler_for("world-adjudicate").sample(
                prompt, component="world-adjudicate"
            )
            error_kind = "解析"
            parsed = self._parse_adjudicate(raw, action_nodes, modifications)
            error_kind = "候选校验"
            problems = self._candidate_problems(
                parsed, action_nodes, modifications, simulation_time
            )
            if problems:
                raise ModelWorldError("裁决输出未通过候选校验：" + "；".join(problems))
            error_kind = "构造"
            patches = parsed["patches"]
            proposed = tuple(FactChangeProposal(p.description) for p in patches)
            outcome_tuple = tuple(
                ActionOutcome(
                    action_id=ActionId(item["action_id"]),
                    result=item["result"],
                    detail=item["detail"],
                )
                for item in parsed["outcomes"]
            )
            follow_ups = tuple(
                WorldEvent(
                    EventId(item["event_id"]),
                    SimulationTime(item["time_ms"]),
                    item["description"],
                )
                for item in parsed["follow_ups"]
            )
            candidate = AdjudicationCandidate(
                cycle_id=cycle_id,
                simulation_time=simulation_time,
                event_ids=tuple(event_ids),
                action_ids=tuple(action_ids),
                proposed_changes=proposed,
                follow_up_events=follow_ups,
                action_outcomes=outcome_tuple,
            )
            from types import MappingProxyType

            node_facts = {node.action_id: (node.role_id, node.description) for node in action_nodes}
            for modification in modifications:
                node_facts.setdefault(
                    modification.action_id, (modification.role_id, modification.description)
                )
            issued = IssuedCandidate(
                candidate=candidate,
                base_version=self._version,
                base_time=self._last_time,
                perception_patches=(
                    self._progress_patches
                    if self._progress_for == (cycle_id, simulation_time)
                    else ()
                ),
                patches=patches,
                movements=tuple(parsed["movements"]),
                stop_movements=tuple(parsed["stop_movements"]),
                passives=tuple(parsed["passives"]),
                passive_movements=tuple(parsed["passive_movements"]),
                stop_passives=tuple(parsed["stop_passives"]),
                reference_event_ids=tuple(event_ids)
                or tuple(node.event_id for node in action_nodes),
                node_facts=MappingProxyType(node_facts),
                withdrawn_actions=tuple(
                    node.action_id for node in action_nodes if node.node is ActionNodeKind.WITHDRAWN
                ),
            )
            # 先成功留痕，再把候选登记为可提交：留痕失败不留半生成记录。
            self._trace(
                "world.adjudicate.candidate",
                cycle_id,
                None,
                "联合裁决候选已生成（未提交）",
                summary_time=simulation_time,
                event_ids=tuple(event_ids),
                action_ids=tuple(action_ids),
                inputs=(),
                outputs=(raw,) + proposed_texts(parsed),
                details={
                    "changes": len(patches),
                    "outcomes": len(outcome_tuple),
                    "follow_ups": len(follow_ups),
                    "base_version": self._version,
                },
            )
            self._issued.append(issued)
            return candidate
        except BaseException as error:
            self._trace_failure(
                "world.adjudicate.error",
                cycle_id,
                summary_time=simulation_time,
                prompt=prompt,
                raw=raw,
                error_kind=error_kind if not isinstance(error, asyncio.CancelledError) else "取消",
                error=error,
                event_ids=tuple(event_ids),
                action_ids=tuple(action_ids),
            )
            raise
        finally:
            self._write_busy = False

    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        """只读校验：候选必须由本 World 生成（身份核对）且与当前状态一致。

        值相等的伪造对象（如 dataclasses.replace 的副本）不是本 World
        生成的候选，整体拒绝。
        """
        problems: list[str] = []
        issued = self._issued_for(candidate)
        if issued is None:
            return AdjudicationValidation(
                candidate=candidate,
                world_state_version=self._version,
                passed=False,
                problems=("候选不是由本 World 生成的（身份不符或已被提交）",),
            )
        if any(item is candidate for item in self._committed_candidates):
            problems.append("候选已被提交过，不得重复提交")
        if issued.base_version != self._version:
            problems.append("候选依据的共同状态版本已过期")
        if issued.base_time != self._last_time:
            problems.append("候选依据的进展基线已过期")
        if not candidate.proposed_changes and not self._has_commit_effects(issued, candidate):
            # 合法无变化裁决允许提交（版本严格递增记录一次裁决），
            # 但必须存在参与事件才不是空转。
            if candidate.event_ids or candidate.action_ids:
                pass
            else:
                problems.append("候选没有任何可提交的实际变化")
        try:
            working = self._scene
            for patch in issued.patches:
                working = apply_patch(working, patch)
        except (SceneDataError, ValueError) as error:
            problems.append(f"补丁无法应用于当前场景：{error}")
        candidate_action_ids = set(candidate.action_ids)
        for outcome in candidate.action_outcomes:
            if outcome.action_id not in candidate_action_ids:
                problems.append(f"结算结果引用未参与裁决的行动：{outcome.action_id}")
        for movement in issued.movements:
            if find_object(self._scene, movement.role_id) is None:
                problems.append(f"运动引用未知角色对象：{movement.role_id}")
        for event in candidate.follow_up_events:
            if event.simulation_time <= candidate.simulation_time:
                problems.append(f"后续世界事件必须晚于裁决时刻：{event.event_id}")
        for passive_id, _ in issued.passives:
            if not passive_id:
                problems.append("被动过程身份必须是非空字符串")
        return AdjudicationValidation(
            candidate=candidate,
            world_state_version=self._version,
            passed=not problems,
            problems=tuple(problems),
        )

    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        """完整预检后在本地副本构造全部新状态，通过后一次性切换。

        任何失败（含模型调用在途、身份不符、重复提交、过期版本/基线）
        都零局部事实；成功后版本严格递增并允许无事实变化的合法裁决
        （环境事件空转）记录一次提交。
        """
        if self._write_busy or self._in_flight:
            raise ModelWorldError("有模型调用在途，事实提交必须独占写入阶段")
        validation = self.validate_adjudication(candidate)
        if not validation.passed or validation.world_state_version != self._version:
            raise ModelWorldError("候选未通过提交前校验：" + "；".join(validation.problems))
        issued = self._issued_for(candidate)
        assert issued is not None  # 校验通过必有现场
        # 本地副本构造全部新状态。
        working_scene = self._scene
        for patch in issued.patches:
            working_scene = apply_patch(working_scene, patch)
        working_actual = dict(self._actual)
        working_movements = dict(self._movements)
        working_passives = dict(self._passives)
        working_passive_movers = dict(self._passive_movers)
        working_passive_movements = dict(self._passive_movements)
        terminal_results = {
            ActionOutcomeResult.COMPLETED,
            ActionOutcomeResult.FAILED,
        }
        for outcome in candidate.action_outcomes:
            node_role, node_description = issued.node_facts.get(
                outcome.action_id,
                (
                    working_actual[outcome.action_id].role_id
                    if outcome.action_id in working_actual
                    else RoleId("unknown"),
                    "",
                ),
            )
            working_actual[outcome.action_id] = ActualAction(
                role_id=node_role,
                description=node_description
                or (
                    working_actual[outcome.action_id].description
                    if outcome.action_id in working_actual
                    else outcome.detail
                ),
                result=outcome.result.value,
                detail=outcome.detail,
            )
            if outcome.result in terminal_results:
                working_movements.pop(outcome.action_id, None)
        for action_id in issued.stop_movements:
            working_movements.pop(action_id, None)
        for movement in issued.movements:
            working_movements[movement.action_id] = movement
        # 撤回主动控制的节点：记录撤回状态（不是伪终态），停止主动运动。
        for action_id in issued.withdrawn_actions:
            if action_id not in {o.action_id for o in candidate.action_outcomes}:
                node_role, node_description = issued.node_facts.get(
                    action_id, (RoleId("unknown"), "")
                )
                prior = working_actual.get(action_id)
                working_actual[action_id] = ActualAction(
                    role_id=prior.role_id if prior else node_role,
                    description=node_description,
                    result="withdrawn",
                    detail="撤回主动控制；被动过程由 World 维护",
                )
                working_movements.pop(action_id, None)
        for passive_id, description in issued.passives:
            working_passives[passive_id] = description
        for passive_id, mover, speed, direction in issued.passive_movements:
            working_passives.setdefault(passive_id, "被动过程持续")
            working_passive_movers[passive_id] = mover
            working_passive_movements[passive_id] = (mover, speed, direction)
        for passive_id in issued.stop_passives:
            working_passives.pop(passive_id, None)
            working_passive_movers.pop(passive_id, None)
            working_passive_movements.pop(passive_id, None)
        new_version = WorldStateVersion(self._version + 1)
        commit = AdjudicationCommit(
            candidate=candidate,
            committed_changes=candidate.proposed_changes,
            resulting_version=new_version,
            action_outcomes=candidate.action_outcomes,
        )
        self._trace(
            "world.commit",
            candidate.cycle_id,
            None,
            "事实已原子提交"
            if self._has_commit_effects(issued, candidate)
            else "合法无变化裁决已提交",
            summary_time=candidate.simulation_time,
            event_ids=tuple(candidate.event_ids),
            action_ids=tuple(candidate.action_ids),
            inputs=(),
            outputs=tuple(p.description for p in issued.patches),
            world_state_version=new_version,
            details={
                "changes": len(issued.patches),
                "outcomes": len(candidate.action_outcomes),
                "version": new_version,
            },
        )
        # 一次性切换：提交对象与全部新状态同批生效。
        self._scene = working_scene
        self._actual = working_actual
        self._movements = working_movements
        self._passives = working_passives
        self._passive_movers = working_passive_movers
        self._passive_movements = working_passive_movements
        self._version = new_version
        self._current_time = candidate.simulation_time
        self._committed_candidates.append(candidate)
        self._committed.append(commit)
        self._progress_for = None
        return commit

    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        """把已提交的结构事实按受众可见投影生成感知；同步确定性。

        不广播模型的自由描述：内容由已提交结构与提交后快照确定性地
        生成（对象名 + 提交后的状态/容纳/位置），隐藏内容不进入任何
        角色输入；打开容器后新可见内容随本次发布传递，关闭不宣告藏
        品。模型的原始描述只留在诊断日志。
        """
        if not any(item is commit for item in self._committed):
            raise ModelWorldError("只能发布本 World 实际提交的裁决")
        if commit.resulting_version != self._version:
            raise ModelWorldError("不能用较新或过期事实冒充本轮提交的观测")
        issued = None
        for item in self._issued:
            if item.candidate is commit.candidate:
                issued = item
                break
        if issued is None:  # pragma: no cover - 已提交候选必有现场
            raise ModelWorldError("提交候选缺少生成现场")
        if not issued.reference_event_ids:
            return ()  # 没有触发事件不编造事件身份投递
        reference = issued.reference_event_ids[0]
        deliveries: list[Delivery] = []
        for patch in (*issued.perception_patches, *issued.patches):
            for role_id in self._role_ids():
                if not self._audience_can_perceive(role_id, patch):
                    continue
                content = self._audience_content(role_id, patch)
                if content is None:
                    continue
                deliveries.append(
                    Delivery(
                        role_id=role_id,
                        observation=Observation(
                            event_id=reference,
                            simulation_time=commit.candidate.simulation_time,
                            content=content,
                        ),
                    )
                )
        # 可见主体的实际行动变化也可触发感知，不把角色的意图描述当事实。
        phase_text = {
            ActionOutcomeResult.STARTED: "开始行动",
            ActionOutcomeResult.IN_PROGRESS: "继续进行行动",
            ActionOutcomeResult.COMPLETED: "结束行动",
            ActionOutcomeResult.FAILED: "动作未能完成",
        }
        for outcome in commit.action_outcomes:
            actual = self._actual[outcome.action_id]
            actor = find_object(self._scene, actual.role_id)
            suffix = phase_text.get(outcome.result)
            if actor is None or suffix is None:
                continue
            for role_id in self._role_ids():
                if self._is_visible_to(role_id, actor.id):
                    deliveries.append(
                        Delivery(
                            role_id,
                            Observation(
                                reference, commit.candidate.simulation_time, f"{actor.name}{suffix}"
                            ),
                        )
                    )
        return tuple(deliveries)

    def _audience_can_perceive(self, role_id: RoleId, patch: ScenePatch) -> bool:
        """受众感知校验：受影响对象与新容纳位置都经统一 helper 可见。"""
        affected = affected_object_id(patch)
        if affected is None:
            return False
        if not self._is_visible_to(role_id, affected):
            return False
        if isinstance(patch, MoveObjectPatch):
            if not self._is_visible_to(role_id, patch.new_parent_id):
                return False
        return True

    def _audience_content(self, role_id: RoleId, patch: ScenePatch) -> str | None:
        """由已提交结构与提交后快照按该受众生成确定性感知文本。

        打开容器只枚举该角色经统一 helper 确实可见的后代（关闭的子
        容器本身可见但其后代仍不可见）；关闭不枚举藏品。无法生成受众
        安全文本的补丁返回 None（原始描述留诊断日志）。
        """
        obj = find_object(self._scene, affected_object_id(patch) or "")
        if obj is None:
            return None
        if isinstance(patch, SetStatePatch):
            value = obj.state.get(patch.key)
            text = f"{obj.name}的{patch.key}现在是{value}"
            if patch.key == "open" and value is True:
                newly_visible = [
                    child.name for child in obj.children if self._is_visible_to(role_id, child.id)
                ]
                if newly_visible:
                    text += "，可以看到：" + "、".join(newly_visible)
            return text
        if isinstance(patch, MoveObjectPatch):
            parent = find_object(self._scene, patch.new_parent_id)
            if parent is None:
                return None
            return f"{obj.name}现在在{parent.name}之中"
        if isinstance(patch, SetPositionPatch):
            return f"{obj.name}移动到 ({patch.position[0]:.2f}, {patch.position[1]:.2f})"
        return None

    def _issued_for(self, candidate: AdjudicationCandidate) -> IssuedCandidate | None:
        """按对象身份查找生成现场；值相等不算。"""
        for item in self._issued:
            if item.candidate is candidate:
                return item
        return None

    def _has_commit_effects(
        self, issued: IssuedCandidate, candidate: AdjudicationCandidate
    ) -> bool:
        return bool(
            candidate.action_outcomes
            or issued.patches
            or issued.movements
            or issued.stop_movements
            or issued.passives
            or issued.passive_movements
            or issued.stop_passives
        )

    def _candidate_problems(
        self,
        parsed: Mapping[str, Any],
        action_nodes: Sequence[AdjudicationActionNode],
        modifications: Sequence[ActionModificationNode],
        simulation_time: SimulationTime,
    ) -> list[str]:
        """候选生成前的运行时一致性校验（提交前还会整体复核）。"""
        problems: list[str] = []
        participating_roles: dict[ActionId, RoleId] = {
            node.action_id: node.role_id for node in action_nodes
        }
        for modification in modifications:
            participating_roles.setdefault(modification.action_id, modification.role_id)
        outcomes_by_action = {item["action_id"]: item for item in parsed["outcomes"]}
        # 1) start/checkpoint/end 节点必须有明确结算反馈。
        for node in action_nodes:
            if (
                node.node
                in (
                    ActionNodeKind.START,
                    ActionNodeKind.CHECKPOINT,
                    ActionNodeKind.END,
                )
                and node.action_id not in outcomes_by_action
            ):
                problems.append(f"参与节点缺少明确结算反馈：{node.action_id}（{node.node.value}）")
        # 2) 运动归属与终态约束。
        seen_movements: set[ActionId] = set()
        for movement in parsed["movements"]:
            if movement.action_id in seen_movements:
                problems.append(f"运动行动重复：{movement.action_id}")
            seen_movements.add(movement.action_id)
            owner = participating_roles.get(movement.action_id)
            if owner is None:
                problems.append(f"运动引用未参与裁决的行动：{movement.action_id}")
                continue
            if owner != movement.role_id:
                problems.append(
                    f"运动归属与行动执行角色不一致：{movement.action_id}"
                    f" 应属 {owner}，得到 {movement.role_id}"
                )
            outcome = outcomes_by_action.get(movement.action_id)
            if outcome is not None and outcome["result"] in (
                ActionOutcomeResult.COMPLETED,
                ActionOutcomeResult.FAILED,
            ):
                problems.append(f"终态结果后不得再启动运动：{movement.action_id}")
            prior = self._actual.get(movement.action_id)
            if prior is not None and prior.result in ("completed", "failed"):
                problems.append(f"已终态行动不能再启动运动：{movement.action_id}")
        # 3) stop_movements 引用与 UNCONFIRMED 禁止隐式停止。
        seen_stop: set[ActionId] = set()
        for action_id in parsed["stop_movements"]:
            if action_id in seen_stop:
                problems.append(f"stop_movements 重复：{action_id}")
            seen_stop.add(action_id)
            if action_id not in participating_roles and action_id not in self._movements:
                problems.append(f"stop_movements 引用未参与且未在进行的行动：{action_id}")
            outcome = outcomes_by_action.get(action_id)
            if outcome is not None and outcome["result"] is ActionOutcomeResult.UNCONFIRMED:
                problems.append(f"未确认结果不得隐式停止已知运动：{action_id}")
        # 4) 冲突补丁：同一对象的移动/换父冲突整批拒绝。
        move_targets: dict[str, str] = {}
        for patch in parsed["patches"]:
            if isinstance(patch, MoveObjectPatch):
                if patch.object_id in move_targets:
                    problems.append(f"同一对象被多次移动：{patch.object_id}")
                move_targets[patch.object_id] = patch.new_parent_id
        # 5) 被动过程身份引用。
        known_passives = set(self._passives)
        for passive_id, _description in parsed["passives"]:
            if passive_id in known_passives:
                problems.append(f"被动过程身份已存在：{passive_id}")
        for passive_id in parsed["stop_passives"]:
            if passive_id not in known_passives:
                problems.append(f"stop_passives 引用未知被动过程：{passive_id}")
        for passive_id, mover, _speed, _direction in parsed["passive_movements"]:
            if passive_id in known_passives:
                problems.append(f"被动运动身份已存在：{passive_id}")
            if find_object(self._scene, mover) is None:
                problems.append(f"被动运动引用未知对象：{mover}")
        del simulation_time
        return problems

    # ---- 提示构造 ----

    def _is_visible_to(self, role_id: RoleId, object_id: str) -> bool:
        """统一受众可见性：关闭容器硬约束优先，注入策略只能收紧。

        自定义 policy 返回 True 不能让关闭容器的后代可见；未知角色
        明确失败。
        """
        if str(role_id) not in {str(r) for r in self._roles}:
            raise ModelWorldError(f"未知角色请求可见性判断：{role_id}")
        if not is_visible(self._scene, object_id):
            return False
        return bool(self._visibility(role_id, self._scene, object_id))

    def _visible_projection(self, role_id: RoleId) -> str:
        """按受众统一过滤（硬约束+策略）；隐藏内容不进入模型输入。"""
        visible_ids = {obj.id for obj in self._scene.walk() if self._is_visible_to(role_id, obj.id)}
        return describe_scene(self._scene, only_ids=sorted(visible_ids))

    def _check_prompt(self, response: RoleResponse) -> str:
        declaration: list[str] = []
        for item in response.actions:
            if isinstance(item, NewAction):
                declaration.append(
                    f"- 新行动 {item.action_id}（组 {item.commit_group_id}）：{item.description}"
                )
            elif isinstance(item, ModifiedAction):
                declaration.append(f"- 修改 {item.action_id}：{item.change}")
            else:
                declaration.append(f"- 维持 {item.action_id}")
        own_states = [
            f"- {action_id}（{actual.result}）：{actual.detail}"
            for action_id, actual in sorted(self._actual.items(), key=lambda kv: str(kv[0]))
            if actual.role_id == response.role_id
        ]
        lines = [
            f"你是场景 World 的客观检查者。共同世界状态版本 {self._version}。",
            f"角色 {response.role_id} 的可见场景（按可见范围过滤）：",
            self._visible_projection(response.role_id),
            "该角色的实际行动状态：",
            *(own_states or ["- 无"]),
            "角色的完整主动行为声明（本轮待检查）：",
            *(declaration or ["- 空集合（撤回全部主动维持）"]),
            "检查客观可执行性：含义、当前步骤边界、未确认结果依赖、客观条件与既有行动的关系。",
            "不替角色决定意愿；被动影响不要求列为主动行为；空集合是合法声明。",
            "只输出一个 JSON 对象，格式：",
            CHECK_SCHEMA,
        ]
        return "\n".join(lines)

    def _progress_prompt(self, elapsed: int, simulation_time: SimulationTime) -> str:
        movement_lines = [
            f"- {action_id}（{m.role_id}）：速度 {m.speed} 米/秒，方向"
            f" ({m.direction[0]:.2f}, {m.direction[1]:.2f})"
            for action_id, m in sorted(self._movements.items(), key=lambda kv: str(kv[0]))
        ]
        passive_lines = []
        for pid, description in sorted(self._passives.items()):
            movement = self._passive_movements.get(pid)
            if movement is not None:
                mover, speed, direction = movement
                passive_lines.append(
                    f"- {pid}：{description}；推动 {mover} 以 {speed:.2f} 米/秒沿方向"
                    f" ({direction[0]:.2f}, {direction[1]:.2f})"
                )
            else:
                passive_lines.append(f"- {pid}：{description}")
        ongoing_lines = [
            f"- {action_id}（{a.role_id}，{a.result}）：{a.detail}"
            for action_id, a in sorted(self._actual.items(), key=lambda kv: str(kv[0]))
            if a.result in ("started", "in_progress", "unconfirmed")
        ]
        object_lines = [
            f"- {obj.id}：({obj.position[0]:.2f}, {obj.position[1]:.2f})"
            for obj in self._scene.walk()
            if obj.position is not None
        ]
        lines = [
            "你是场景 World 的进展裁决者。按已流逝剧情时间更新实际位置、持续中的行动与被动过程。",
            f"当前剧情时刻 {simulation_time} 毫秒；自上次已提交进展以来流逝 {elapsed} 毫秒。",
            "有坐标对象当前位置（房间绝对坐标，米）：",
            *(object_lines or ["- 无"]),
            "进行中的实际运动：",
            *(movement_lines or ["- 无"]),
            "持续中的行动（含非移动过程）：",
            *(ongoing_lines or ["- 无"]),
            "被动过程（显式 done 才结束，省略不等于删除）：",
            *(passive_lines or ["- 无"]),
            "完整场景事实（World 语义判断可用）：",
            describe_scene(self._scene),
            "要求：位置变化按流逝毫秒与速度方向计算；没有实际变化就不要输出移动；"
            "这是实际进展，不是登记安排或预期结果，不提前宣称完成。",
            "只输出一个 JSON 对象，格式：",
            PROGRESS_SCHEMA,
        ]
        return "\n".join(lines)

    def _adjudicate_prompt(
        self,
        simulation_time: SimulationTime,
        events: Sequence[WorldEvent],
        action_nodes: Sequence[AdjudicationActionNode],
        modifications: Sequence[ActionModificationNode],
    ) -> str:
        event_lines = [f"- {e.event_id} @ {e.simulation_time}ms：{e.description}" for e in events]
        node_lines = [
            f"- {n.event_id}：{n.role_id} 的 {n.action_id} {n.node.value}"
            f"（{n.description}，安排 {n.start}→{n.expected_end}ms）"
            for n in action_nodes
        ]
        modification_lines = [
            f"- {m.role_id} 修改 {m.action_id}：{m.change}"
            f"（原 {m.old_start}→{m.old_expected_end}ms，新 {m.new_start}→{m.new_expected_end}ms）"
            for m in modifications
        ]
        actual_lines = [
            f"- {action_id}（{a.role_id}，{a.result}）：{a.detail}"
            for action_id, a in sorted(self._actual.items(), key=lambda kv: str(kv[0]))
        ]
        movement_lines = [
            f"- {action_id}（{m.role_id}）：速度 {m.speed}，方向 ({m.direction[0]:.2f}, {m.direction[1]:.2f})"
            for action_id, m in sorted(self._movements.items(), key=lambda kv: str(kv[0]))
        ]
        lines = [
            "你是场景 World 的联合裁决者：对同刻全部事件与行动做一次联合判断。",
            f"当前剧情时刻 {simulation_time} 毫秒；共同世界状态版本 {self._version}。",
            "完整场景事实：",
            describe_scene(self._scene),
            "同刻事件：",
            *(event_lines or ["- 无"]),
            "行动节点：",
            *(node_lines or ["- 无"]),
            "明确修改：",
            *(modification_lines or ["- 无"]),
            "持续被动过程：",
            *[f"- {pid}：{description}" for pid, description in sorted(self._passives.items())],
            "现行实际行动状态：",
            *(actual_lines or ["- 无"]),
            "进行中的实际运动：",
            *(movement_lines or ["- 无"]),
            "要求：",
            "- 联合考虑相互作用，不按插入顺序制造优先级；同刻两人争取同一对象只给一份归属；",
            "- start 节点也可能失败或未确认；到 end 不保证成功；完成/失败/未确认保持区分；",
            "- 撤回主动控制只撤主动：历史与被动惯性等仍归 World（用 passives 表达）；",
            "- 开始/修改运动用 movements 建立实际速度方向；结束/撤回用 stop_movements；",
            "- 后续世界事件必须晚于当前剧情时刻（整数毫秒）。",
            "只输出一个 JSON 对象，格式：",
            ADJUDICATE_SCHEMA,
        ]
        return "\n".join(lines)

    # ---- 解析 ----

    def _parse_patches(self, items: Any) -> tuple[ScenePatch, ...]:
        """进展与结算共用结构化补丁校验，不用描述文本替代事实。"""
        if not isinstance(items, list):
            raise ModelWorldError("changes 必须是数组")
        patches: list[ScenePatch] = []
        for item in items:
            if not isinstance(item, dict):
                raise ModelWorldError("changes 每项必须是对象")
            kind = item.get("kind")
            if kind == "set_state":
                unknown = sorted(set(item) - {"kind", "object_id", "key", "value", "description"})
                if unknown:
                    raise ModelWorldError(f"changes 项包含未知字段：{unknown}")
                for field_name in ("object_id", "key", "description"):
                    if not isinstance(item.get(field_name), str) or not item.get(field_name):
                        raise ModelWorldError(f"set_state 缺少非空字段 {field_name}")
                patches.append(
                    SetStatePatch(
                        object_id=item["object_id"],
                        key=item["key"],
                        value=_state_value(item.get("value"), "value"),
                        description=item["description"],
                    )
                )
            elif kind == "set_position":
                unknown = sorted(set(item) - {"kind", "object_id", "position", "description"})
                if unknown:
                    raise ModelWorldError(f"changes 项包含未知字段：{unknown}")
                for field_name in ("object_id", "description"):
                    if not isinstance(item.get(field_name), str) or not item.get(field_name):
                        raise ModelWorldError(f"set_position 缺少非空字段 {field_name}")
                patches.append(
                    SetPositionPatch(
                        object_id=item["object_id"],
                        position=_require_xy(item.get("position"), "position"),
                        description=item["description"],
                    )
                )
            elif kind == "move":
                unknown = sorted(set(item) - {"kind", "object_id", "new_parent", "description"})
                if unknown:
                    raise ModelWorldError(f"changes 项包含未知字段：{unknown}")
                for field_name in ("object_id", "new_parent", "description"):
                    if not isinstance(item.get(field_name), str) or not item.get(field_name):
                        raise ModelWorldError(f"move 缺少非空字段 {field_name}")
                patches.append(
                    MoveObjectPatch(
                        object_id=item["object_id"],
                        new_parent_id=item["new_parent"],
                        description=item["description"],
                    )
                )
            else:
                raise ModelWorldError(f"未知补丁类型：{kind!r}")
        return tuple(patches)

    def _parse_progress(self, raw: str) -> _ProgressPlan:
        """一次解析为 typed 进展计划；数组字段全部必填。"""
        data = parse_model_json(
            raw,
            "world-progress",
            allowed_keys=(
                "movements",
                "updated_action_ids",
                "notes",
                "passive_updates",
                "changes",
                "action_progress",
            ),
        )
        for required in ("movements", "updated_action_ids", "notes", "passive_updates"):
            if required not in data:
                raise ModelWorldError(f"world-progress 输出缺少必填字段 {required}")
        raw_movements = data["movements"]
        if not isinstance(raw_movements, list):
            raise ModelWorldError("movements 必须是数组")
        movements: list[tuple[str, tuple[float, float]]] = []
        seen: set[str] = set()
        for item in raw_movements:
            if not isinstance(item, dict):
                raise ModelWorldError("movements 每项必须是对象")
            unknown = sorted(set(item) - {"object_id", "position"})
            if unknown:
                raise ModelWorldError(f"movements 项包含未知字段：{unknown}")
            for field in ("object_id", "position"):
                if field not in item:
                    raise ModelWorldError(f"movements 项缺少必填字段 {field}")
            object_id = item["object_id"]
            if not isinstance(object_id, str) or not object_id:
                raise ModelWorldError("object_id 必须是非空字符串")
            if object_id in seen:
                raise ModelWorldError(f"movements 对象重复：{object_id}")
            seen.add(object_id)
            movements.append((object_id, _require_xy(item["position"], "position")))
        patches = self._parse_patches(data.get("changes", []))
        positioned = {p.object_id for p in patches if isinstance(p, SetPositionPatch)}
        if positioned & seen:
            raise ModelWorldError("同一对象不得同时在 movements 与 changes 设置坐标")
        action_progress: list[tuple[ActionId, str]] = []
        raw_progress = data.get("action_progress", [])
        if not isinstance(raw_progress, list):
            raise ModelWorldError("action_progress 必须是数组")
        seen_progress: set[str] = set()
        for item in raw_progress:
            if not isinstance(item, dict) or set(item) != {"action_id", "detail"}:
                raise ModelWorldError("action_progress 每项只能且必须包含 action_id/detail")
            action_id, detail = item["action_id"], item["detail"]
            if not isinstance(action_id, str) or action_id in seen_progress:
                raise ModelWorldError("action_progress 身份非法或重复")
            actual = self._actual.get(ActionId(action_id))
            if actual is None or actual.result not in ("started", "in_progress", "unconfirmed"):
                raise ModelWorldError(f"进展必须引用实际进行中的行动：{action_id}")
            if not isinstance(detail, str) or not detail:
                raise ModelWorldError("action_progress.detail 必须是非空字符串")
            seen_progress.add(action_id)
            action_progress.append((ActionId(action_id), detail))
        raw_updated = data["updated_action_ids"]
        if not isinstance(raw_updated, list):
            raise ModelWorldError("updated_action_ids 必须是数组")
        updated: list[ActionId] = []
        seen_actions: set[str] = set()
        for item in raw_updated:
            if not isinstance(item, str) or not item:
                raise ModelWorldError("updated_action_ids 每项必须是非空字符串")
            if item in seen_actions:
                raise ModelWorldError(f"updated_action_ids 重复：{item}")
            seen_actions.add(item)
            if item not in self._movements and item not in self._actual:
                raise ModelWorldError(f"进展引用未知行动：{item}")
            actual = self._actual.get(ActionId(item))
            if actual is not None and actual.result not in (
                "started",
                "in_progress",
                "unconfirmed",
            ):
                raise ModelWorldError(f"不能更新已终止主动行动的进展：{item}")
            updated.append(ActionId(item))
        raw_notes = data["notes"]
        if not isinstance(raw_notes, list):
            raise ModelWorldError("notes 必须是数组")
        notes: list[str] = []
        for item in raw_notes:
            if not isinstance(item, str) or not item:
                raise ModelWorldError("notes 每项必须是非空字符串")
            notes.append(item)
        raw_passive_updates = data["passive_updates"]
        if not isinstance(raw_passive_updates, list):
            raise ModelWorldError("passive_updates 必须是数组")
        passive_notes: list[tuple[str, str]] = []
        passives_done: list[str] = []
        seen_passives: set[str] = set()
        for item in raw_passive_updates:
            if not isinstance(item, dict):
                raise ModelWorldError("passive_updates 每项必须是对象")
            unknown = sorted(set(item) - {"id", "note", "done"})
            if unknown:
                raise ModelWorldError(f"passive_updates 项包含未知字段：{unknown}")
            if "id" not in item:
                raise ModelWorldError("passive_updates 项缺少必填字段 id")
            passive_id = item["id"]
            if not isinstance(passive_id, str) or not passive_id:
                raise ModelWorldError("被动过程身份必须是非空字符串")
            if passive_id in seen_passives:
                raise ModelWorldError(f"被动过程更新重复：{passive_id}")
            seen_passives.add(passive_id)
            done = item.get("done", False)
            if type(done) is not bool:
                raise ModelWorldError("done 必须是布尔值")
            note = item.get("note", "")
            if not isinstance(note, str):
                raise ModelWorldError("note 必须是字符串")
            if done:
                if not note:
                    raise ModelWorldError("结束被动过程必须说明原因")
                passives_done.append(passive_id)
            else:
                passive_notes.append((passive_id, note or self._passives.get(passive_id, "")))
        return _ProgressPlan(
            movements=tuple(movements),
            updated_action_ids=tuple(updated),
            notes=tuple(notes),
            passive_notes=tuple(passive_notes),
            passives_done=tuple(passives_done),
            patches=patches,
            action_progress=tuple(action_progress),
        )

    def _parse_adjudicate(
        self,
        raw: str,
        action_nodes: Sequence[AdjudicationActionNode],
        modifications: Sequence[ActionModificationNode],
    ) -> dict[str, Any]:
        """一次解析联合裁决输出；七个数组字段全部必须显式出现。"""
        try:
            data = parse_model_json(
                raw,
                "world-adjudicate",
                allowed_keys=(
                    "changes",
                    "outcomes",
                    "follow_ups",
                    "movements",
                    "stop_movements",
                    "passives",
                    "stop_passives",
                ),
            )
        except ModelJsonError as error:
            raise ModelWorldError(str(error)) from error
        for required in (
            "changes",
            "outcomes",
            "follow_ups",
            "movements",
            "stop_movements",
            "passives",
            "stop_passives",
        ):
            if required not in data or not isinstance(data[required], list):
                raise ModelWorldError(f"world-adjudicate 输出缺少数组字段 {required}")
        known_actions = {str(n.action_id) for n in action_nodes} | {
            str(m.action_id) for m in modifications
        }
        patches = self._parse_patches(data["changes"])
        outcomes: list[dict[str, Any]] = []
        seen_outcomes: set[str] = set()
        for item in data["outcomes"]:
            if not isinstance(item, dict):
                raise ModelWorldError("outcomes 每项必须是对象")
            unknown = sorted(set(item) - {"action_id", "result", "detail"})
            if unknown:
                raise ModelWorldError(f"outcomes 项包含未知字段：{unknown}")
            action_id = item.get("action_id")
            if not isinstance(action_id, str) or action_id not in known_actions:
                raise ModelWorldError(f"结算结果引用未参与裁决的行动：{action_id!r}")
            if action_id in seen_outcomes:
                raise ModelWorldError(f"结算结果行动重复：{action_id}")
            seen_outcomes.add(action_id)
            result = item.get("result")
            if not isinstance(result, str):
                raise ModelWorldError(f"result 必须是字符串，得到：{result!r}")
            try:
                parsed_result = ActionOutcomeResult(result)
            except ValueError as error:
                raise ModelWorldError(f"未知结算结果类别：{result!r}") from error
            detail = item.get("detail")
            if not isinstance(detail, str) or not detail:
                raise ModelWorldError("outcomes 项 detail 必须是非空字符串")
            outcomes.append({"action_id": action_id, "result": parsed_result, "detail": detail})
        follow_ups: list[dict[str, Any]] = []
        for item in data["follow_ups"]:
            if not isinstance(item, dict):
                raise ModelWorldError("follow_ups 每项必须是对象")
            unknown = sorted(set(item) - {"event_id", "time_ms", "description"})
            if unknown:
                raise ModelWorldError(f"follow_ups 项包含未知字段：{unknown}")
            for field_name in ("event_id", "description"):
                if not isinstance(item.get(field_name), str) or not item.get(field_name):
                    raise ModelWorldError(f"follow_ups 项缺少非空字段 {field_name}")
            follow_ups.append(
                {
                    "event_id": item["event_id"],
                    "time_ms": require_json_int(item.get("time_ms"), "time_ms"),
                    "description": item["description"],
                }
            )
        movements: list[MovementState] = []
        for item in data["movements"]:
            if not isinstance(item, dict):
                raise ModelWorldError("movements 每项必须是对象")
            unknown = sorted(set(item) - {"action_id", "role_id", "speed", "direction"})
            if unknown:
                raise ModelWorldError(f"movements 项包含未知字段：{unknown}")
            action_id = item.get("action_id")
            if not isinstance(action_id, str) or action_id not in known_actions:
                raise ModelWorldError(f"运动引用未参与裁决的行动：{action_id!r}")
            role_id = item.get("role_id")
            if not isinstance(role_id, str) or not role_id:
                raise ModelWorldError("movements 项 role_id 必须是非空字符串")
            speed = _require_float(item.get("speed"), "speed")
            if speed < 0:
                raise ModelWorldError("speed 必须非负")
            direction = _normalize_direction(_require_xy(item.get("direction"), "direction"))
            movements.append(MovementState(ActionId(action_id), RoleId(role_id), speed, direction))
        stop_ids: list[ActionId] = []
        for item in data["stop_movements"]:
            if not isinstance(item, str) or not item:
                raise ModelWorldError("stop_movements 每项必须是非空字符串")
            stop_ids.append(ActionId(item))
        passives: list[tuple[str, str]] = []
        passive_movements: list[tuple[str, str, float, tuple[float, float]]] = []
        seen_passives: set[str] = set()
        for item in data["passives"]:
            if not isinstance(item, dict):
                raise ModelWorldError("passives 每项必须是对象")
            unknown = sorted(set(item) - {"id", "description", "object_id", "speed", "direction"})
            if unknown:
                raise ModelWorldError(f"passives 项包含未知字段：{unknown}")
            passive_id = item.get("id")
            if not isinstance(passive_id, str) or not passive_id:
                raise ModelWorldError("passives 项 id 必须是非空字符串")
            if passive_id in seen_passives:
                raise ModelWorldError(f"被动过程身份重复：{passive_id}")
            seen_passives.add(passive_id)
            description = item.get("description")
            if not isinstance(description, str) or not description:
                raise ModelWorldError("passives 项 description 必须是非空字符串")
            passives.append((passive_id, description))
            if "object_id" in item or "speed" in item:
                mover = item.get("object_id")
                if not isinstance(mover, str) or not mover:
                    raise ModelWorldError("被动运动的 object_id 必须是非空字符串")
                speed = _require_float(item.get("speed"), "passives.speed")
                direction = _normalize_direction(
                    _require_xy(item.get("direction"), "passives.direction")
                )
                passive_movements.append((passive_id, mover, speed, direction))
        stop_passives: list[str] = []
        for item in data["stop_passives"]:
            if not isinstance(item, str) or not item:
                raise ModelWorldError("stop_passives 每项必须是非空字符串")
            stop_passives.append(item)
        return {
            "patches": tuple(patches),
            "outcomes": outcomes,
            "follow_ups": follow_ups,
            "movements": movements,
            "stop_movements": stop_ids,
            "passives": passives,
            "passive_movements": passive_movements,
            "stop_passives": stop_passives,
        }

    # ---- 占用保护与留痕 ----

    def _require_scope_free(self, scope: str) -> None:
        if scope in self._in_flight:
            raise ModelWorldError(f"作用域 {scope} 的请求必须串行，检测到并发调用")

    def _require_write_free(self) -> None:
        if self._write_busy:
            raise ModelWorldError("World 写入阶段已被占用，不得并发进入")

    def _require_no_writer(self, phase: str) -> None:
        """检查（读阶段）不得与写入阶段交叉：事实在检查期间必须冻结。"""
        if self._write_busy:
            raise ModelWorldError(f"{phase}期间 World 正在写入，不得进入")

    def _require_no_readers(self, phase: str) -> None:
        """写入阶段不得与在途检查交叉：检查依据的共同版本不能被改变。"""
        if self._in_flight:
            raise ModelWorldError(
                f"{phase}需要独占写入，但有检查请求在途：{sorted(self._in_flight)}"
            )

    def _trace(
        self,
        stage: str,
        cycle_id: CycleId,
        response_id: ResponseId | None,
        summary: str,
        *,
        summary_time: SimulationTime | None,
        role_id: RoleId | None = None,
        action_ids: tuple[ActionId, ...] = (),
        event_ids: tuple[EventId, ...] = (),
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        details: Mapping[str, TraceValue],
        world_state_version: WorldStateVersion | None = None,
    ) -> None:
        """成功链留痕；追踪器失败传播，不静默假成功。"""
        if self._tracer is None:
            return
        self._tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=cycle_id,
                simulation_time=summary_time if summary_time is not None else SimulationTime(0),
                summary=summary,
                role_id=role_id,
                response_id=response_id,
                action_ids=action_ids,
                event_ids=event_ids,
                inputs=inputs,
                outputs=outputs,
                world_state_version=self._version
                if world_state_version is None
                else world_state_version,
                details=dict(details),
            )
        )

    def _trace_failure(
        self,
        stage: str,
        cycle_id: CycleId,
        *,
        summary_time: SimulationTime | None,
        prompt: str,
        raw: str | None,
        error_kind: str,
        error: BaseException,
        role_id: RoleId | None = None,
        response_id: Any = None,
        action_ids: tuple[ActionId, ...] = (),
        event_ids: tuple[EventId, ...] = (),
    ) -> None:
        """失败链留痕；追踪失败以 add_note 保留原异常，不掩盖。"""
        if self._tracer is None:
            return
        message = f"{type(error).__name__}: {error}"
        try:
            self._tracer.trace(
                TraceRecord(
                    stage=stage,
                    cycle_id=cycle_id,
                    simulation_time=summary_time if summary_time is not None else SimulationTime(0),
                    summary=f"World 阶段失败（{error_kind}）",
                    role_id=role_id,
                    response_id=response_id,
                    action_ids=action_ids,
                    event_ids=event_ids,
                    inputs=(prompt,) if prompt else (),
                    outputs=(raw,) if raw else (),
                    world_state_version=self._version,
                    details={"error_kind": error_kind, "raw_output": raw},
                    error=message,
                )
            )
        except Exception as trace_error:
            error.add_note(f"World 错误日志写入失败：{trace_error}")


def proposed_texts(parsed: Mapping[str, Any]) -> tuple[str, ...]:
    """候选阶段的补丁摘要文本（完整对象 repr，供日志检索）。"""
    return tuple(repr(patch) for patch in parsed["patches"])


def affected_object_id(patch: ScenePatch) -> str | None:
    if isinstance(patch, (SetStatePatch, SetPositionPatch)):
        return patch.object_id
    if isinstance(patch, MoveObjectPatch):
        return patch.object_id
    return None
