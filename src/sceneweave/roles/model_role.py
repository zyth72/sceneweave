"""模型驱动的角色实现：内部维护一维全量结构化动作计划。

角色对外声明完整的一维结构化动作链（含尚未满足依赖的后续动作）；
整链检查通过并保存后，只把**最近步骤**作为执行提交交给引擎登记，未来
步骤留在私有缓存。保存成功与执行提交是两个独立边界，不用一个语义
掩盖。

协议要点：

- System2 只产当前意向，不产放弃指令；System3 每次产一维完整结构化
  动作数组。
- 模型不得自造真实身份：新动作无 id 时由角色（程序）分配；同计划内
  相互依赖用局部 ``key`` 引用，角色一次性分配身份后解析到真实 id；
  已有动作必须携带确认的真实 id 与完整参数。生成候选不修改既有已接受
  动作的描述或参数，身份、实际进展与开始时间不因重声明重启。
- 翻译器给出平坦、独立的逐动作时间安排；无依赖动作可部分交叠，不再
  有全体动作共同时间交集约束。依赖由引擎按 World 真实完成门控。
- 候选只在整链检查通过、引擎调用 ``save_plan`` 后原子保存（完整链 +
  全部排程）；打回、解析错误或取消不保存候选，已保存的链与历史不改写。
- 执行提交只选依赖就绪、尚未提交、时间最早的最近步骤；已提交的仍进行
  动作留在完整声明中维持，其他未来动作不一次性登记。转义为撤回主动
  维持的合法空计划不清空 World 的被动过程。
"""

import asyncio
from dataclasses import dataclass

from sceneweave.domain.action_plan import ActionPlan, AtomicAction, PlanAction, StepAction
from sceneweave.domain.identity import ActionId, CycleId, ResponseId, RoleId
from sceneweave.domain.messages import (
    ActionPhase,
    KnownActionState,
    Rejection,
    RoleResponse,
    TurnInput,
)
from sceneweave.domain.tracing import TraceValue, TurnTracer
from sceneweave.domain.translation import (
    ActionSchedule,
    ExecutionSubmission,
    PlanAcceptance,
)
from sceneweave.roles.base import Role
from sceneweave.roles.components import system1, system2, system3
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.components.system1_policy import (
    DEFAULT_SYSTEM1_ROUTE_SEED,
    derive_route_rng,
)
from sceneweave.roles.model_output import ModelOutputError, System1Route, System1RoutePolicy

# 仍可被主动维持的阶段；未确认表示结果未知，不是已知终止。
_MAINTAINABLE = frozenset({ActionPhase.REGISTERED, ActionPhase.STARTED, ActionPhase.UNCONFIRMED})
# 允许被再次声明的阶段（含尚未确认的候选，允许被新计划替换而非报错）。
_DECLARABLE = frozenset(
    {
        ActionPhase.SUBMITTED,
        ActionPhase.REGISTERED,
        ActionPhase.STARTED,
        ActionPhase.UNCONFIRMED,
    }
)
# 可作为依赖依据的已结束阶段：只有真正完成的前置动作才放行后置。
_COMPLETED = ActionPhase.COMPLETED
# 角色私有计划阶段，不属于 World 的行动生命周期；保存不表示执行提交。
_PLANNED = "planned"
_DECLARABLE_PHASES = frozenset(phase.value for phase in _DECLARABLE) | {_PLANNED}


@dataclass(slots=True)
class _OwnAction:
    """角色对一个行动身份的私人认知。"""

    description: str
    phase: str
    confirmed: bool
    maintainable: bool
    action: AtomicAction | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _SavedPlan:
    """角色内部已保存的完整链与全部排程（整链检查通过后的私有缓存）。"""

    plan: ActionPlan
    schedules: tuple[ActionSchedule, ...]


def _require_positive(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field}必须是正整数，得到：{value!r}")
    return value


def _require_action_id(item: PlanAction) -> ActionId:
    """取内部计划动作已分配的真实身份；未分配是内部错误。"""
    if item.action_id is None:  # pragma: no cover - 分配后一定齐备
        raise ModelOutputError("内部计划动作缺少已分配身份")
    return item.action_id


class ModelRole(Role):
    """以注入采样器驱动 System1/2/3 的角色。"""

    def __init__(
        self,
        role_id: RoleId,
        sampler: TextSampler,
        *,
        tracer: TurnTracer | None = None,
        background: str = "",
        max_memory: int = 32,
        max_system2_calls: int = 3,
        max_system3_calls: int = 4,
        route_seed: int = DEFAULT_SYSTEM1_ROUTE_SEED,
        route_policy: System1RoutePolicy | None = None,
    ) -> None:
        if not isinstance(role_id, str) or not role_id:
            raise ValueError("角色身份必须是非空字符串")
        self._role_id = role_id
        self._sampler = sampler
        self._tracer = tracer
        if not isinstance(background, str):
            raise ValueError("角色背景必须是字符串")
        self._background = background
        self._max_memory = _require_positive(max_memory, "记忆上限")
        # System3 明确判断意向不可落实时升级 System2 重选意向；两个上限约束
        # 单个周期内的落实与升级次数。
        self._max_system2_calls = _require_positive(max_system2_calls, "System2 次数上限")
        self._max_system3_calls = _require_positive(max_system3_calls, "System3 次数上限")
        # System1 程序路由的随机源在角色边界显式派生：每角色独立、可复现，
        # 只由 route_seed 与角色身份决定，不受并发调用次序影响。
        self._route_seed = route_seed
        self._route_rng, self._route_seed_hash = derive_route_rng(route_seed, role_id)
        self._route_policy = route_policy if route_policy is not None else System1RoutePolicy()
        self._memory: list[str] = []
        # 全部历史行动身份都保留：新行动身份永不复用；未确认的候选身份
        # 同样保留但不可维持。
        self._actions: dict[ActionId, _OwnAction] = {}
        self._action_seq = 0
        self._response_seq = 0
        self._intention: str | None = None
        # 已保存（整链检查通过）的完整链与全部排程；候选只在 save_plan 时提升。
        self._saved: _SavedPlan | None = None
        self._candidate: ActionPlan | None = None
        # 已经执行提交（登记）过的行动身份；用于挑选最近步骤。
        self._submitted: set[ActionId] = set()
        self._pending: tuple[CycleId, ResponseId, frozenset[ActionId]] | None = None
        self._last_facts: TurnFacts | None = None
        self._last_response_id: ResponseId | None = None
        # 同角色公共方法串行（跨角色并行）；内部方法不再加锁，避免死锁。
        self._lock = asyncio.Lock()

    @property
    def role_id(self) -> RoleId:
        return self._role_id

    # ---- 只读诊断视图（测试与调试；不暴露可变私有状态） ----

    def memory_notes(self) -> tuple[str, ...]:
        """有界私人认知的只读副本。"""
        return tuple(self._memory)

    def own_action_digest(self) -> tuple[tuple[str, str, str, bool], ...]:
        """自身行动登记的只读摘要：(身份, 阶段, 描述, 可维持)。"""
        return tuple(
            (str(action_id), own.phase, own.description, own.maintainable)
            for action_id, own in sorted(self._actions.items(), key=lambda kv: str(kv[0]))
        )

    def saved_plan_digest(self) -> tuple[str, ...]:
        """已保存完整链的只读摘要（含排程）；只读诊断。"""
        if self._saved is None:
            return ()
        schedules = {item.action_id: item for item in self._saved.schedules}
        rows: list[str] = []
        for item in self._saved.plan.actions:
            action_id = _require_action_id(item)
            schedule = schedules.get(action_id)
            window = (
                f"{schedule.start}->{schedule.expected_end}" if schedule is not None else "无排程"
            )
            deps = f" 依赖[{', '.join(item.depends_on)}]" if item.depends_on else ""
            rows.append(f"{action_id}:{item.action.display} {window}{deps}")
        return tuple(rows)

    # ---- 对引擎的交互契约 ----

    async def perceive(self, turn: TurnInput) -> RoleResponse:
        if turn.role_id != self._role_id:
            raise ValueError(f"输入角色身份不一致：{turn.role_id}")
        async with self._lock:
            facts = self._facts(turn)
            self._last_facts = facts
            self._pending = None
            self._align_with_known(turn.known_action_states)
            response_id = self._next_response_id()
            self._last_response_id = response_id
            decision = await system1.run(
                self._sampler,
                facts,
                tuple(self._memory),
                response_id=response_id,
                intention=self._intention,
                plan_digest=self._plan_digest(),
                has_saved_plan=self._saved is not None,
                background=self._background,
                tracer=self._tracer,
                policy=self._route_policy,
                rng=self._route_rng,
                route_seed=self._route_seed,
                seed_hash=self._route_seed_hash,
            )
            self._remember(decision.note)
            if decision.route is System1Route.CONTINUE:
                # 沿用已保存的完整链与排程：不调用 System2/3，也不重译。
                for stage in ("system2.skipped", "system3.skipped"):
                    trace_stage(
                        self._tracer,
                        stage,
                        facts,
                        response_id,
                        "System1 选择 continue，沿用已保存的完整链与时间安排",
                        prompt="（跳过模型调用）",
                        raw_output=None,
                        action_ids=tuple(self._active_ids()),
                    )
                response = self._compose_reuse(response_id)
                self._set_pending(facts.cycle_id, response)
                return response
            if decision.route is System1Route.REPLAN:
                # 保留当前意向，只重调 System3；不调用 System2。
                if self._intention is None:
                    raise ModelOutputError("System1 选择 replan 但当前没有意向")
                trace_stage(
                    self._tracer,
                    "system2.skipped",
                    facts,
                    response_id,
                    "System1 选择 replan，保留当前意向只重新落实",
                    prompt="（跳过 System2）",
                    raw_output=None,
                    action_ids=tuple(self._active_ids()),
                )
                response = await self._produce(facts, response_id, self._intention)
                self._set_pending(facts.cycle_id, response)
                return response
            # RETHINK：重选意向后再落实。
            response = await self._rethink(
                facts, response_id, feedback_reason=None, rejection_reason=None
            )
            self._set_pending(facts.cycle_id, response)
            return response

    async def rejected(self, feedback: Rejection) -> RoleResponse:
        async with self._lock:
            if feedback.role_id != self._role_id:
                raise ValueError(f"拒绝了他人的反馈：{feedback.role_id}")
            if self._pending is None:
                raise ValueError("没有待检查的响应可被打回")
            pending_cycle, pending_response, pending_actions = self._pending
            if feedback.cycle_id != pending_cycle:
                raise ValueError(f"反馈周期不一致：{feedback.cycle_id} != {pending_cycle}")
            if feedback.rejected_response_id != pending_response:
                raise ValueError(
                    f"反馈指向过期响应：{feedback.rejected_response_id} != {pending_response}"
                )
            for action_id in feedback.action_ids:
                if action_id not in pending_actions:
                    raise ValueError(f"反馈引用了当前待检查声明之外的行动：{action_id}")
            if self._last_facts is None:  # pragma: no cover - pending 存在则必有
                raise ValueError("缺少本轮上下文")
            # 被打回的候选不生效：丢弃候选，已保存计划与已完成历史保持不动。
            # World 整链检查拒绝先反馈 System3（保留意向重落实）；只有 System3
            # 明确判断意向不可落实才升级 System2。API/格式错误不在此列。
            self._pending = None
            self._candidate = None
            response_id = self._next_response_id()
            self._last_response_id = response_id
            if self._intention is None:  # pragma: no cover - 待检查响应必有意向
                response = await self._rethink(
                    self._last_facts,
                    response_id,
                    feedback_reason=None,
                    rejection_reason=feedback.reason,
                )
            else:
                response = await self._produce(
                    self._last_facts, response_id, self._intention, context_note=feedback.reason
                )
            self._set_pending(feedback.cycle_id, response)
            return response

    async def save_plan(self, acceptance: PlanAcceptance) -> None:
        """整链检查通过后原子保存完整链与全部排程；不登记执行事件。

        只有通过检查的候选才能保存；保存与执行提交分开。已完成的历史
        不改写，原先已保存的链被本候选替换。保存前严格核对：保存消息必须
        对应本角色**当前待检查的候选**（周期/响应/完整链一致），过期、不相干
        或未被检查的替换 payload 一律拒绝，且拒绝时不得覆写旧计划。
        """
        async with self._lock:
            if acceptance.declaration.role_id != self._role_id:
                raise ValueError(f"保存消息角色不一致：{acceptance.declaration.role_id}")
            if acceptance.check.role_id != self._role_id:
                raise ValueError(f"保存消息检查角色不一致：{acceptance.check.role_id}")
            if self._pending is None:
                raise ValueError("没有待检查的候选可保存")
            pending_cycle, pending_response, _pending_actions = self._pending
            if (
                acceptance.declaration.cycle_id != pending_cycle
                or acceptance.declaration.response_id != pending_response
            ):
                raise ValueError("保存消息与当前待检查候选不一致（过期或不相干）")
            if self._candidate is None:
                raise ValueError("没有当前候选计划可保存")
            if acceptance.plan != self._candidate:
                raise ValueError("保存消息的完整链不是当前候选计划")
            # 全部校验通过后才原子切换：完整链 + 全部排程一起保存。
            self._saved = _SavedPlan(plan=acceptance.plan, schedules=tuple(acceptance.schedules))
            self._candidate = None
            self._pending = None
            if self._tracer is not None and self._last_facts is not None:
                trace_stage(
                    self._tracer,
                    "role.plan_saved",
                    self._last_facts,
                    acceptance.declaration.response_id,
                    "整链检查通过，角色内部保存完整链与全部排程（尚未执行提交）",
                    prompt="（保存私有缓存，无模型调用）",
                    raw_output=None,
                    details={"actions": len(acceptance.plan.actions)},
                    action_ids=tuple(_require_action_id(item) for item in acceptance.plan.actions),
                )

    def execution_submission(self) -> ExecutionSubmission | None:
        """返回最近步骤执行提交；未来步骤留在私有缓存。"""
        if self._saved is None or self._last_facts is None:
            return None
        saved = self._saved
        schedules = {item.action_id: item for item in saved.schedules}
        maintained: list[PlanAction] = []
        pending: list[PlanAction] = []
        for item in saved.plan.actions:
            action_id = _require_action_id(item)
            own = self._actions.get(action_id)
            if own is None or own.phase not in _DECLARABLE_PHASES:
                continue
            if not self._dependencies_met(item):
                continue
            if action_id in self._submitted:
                maintained.append(item)
            elif action_id in schedules:
                pending.append(item)
        selected: list[PlanAction] = list(maintained)
        maintained_ids = {_require_action_id(item) for item in maintained}
        if pending:
            earliest = min(schedules[_require_action_id(item)].start for item in pending)
            for item in pending:
                if schedules[_require_action_id(item)].start == earliest:
                    selected.append(item)
                    self._submitted.add(_require_action_id(item))
                    self._actions[_require_action_id(item)].phase = ActionPhase.SUBMITTED.value
        actions = tuple(
            StepAction(
                _require_action_id(item),
                item.action,
                tuple(ActionId(ref) for ref in item.depends_on),
            )
            for item in selected
        )
        new_schedules = tuple(
            schedules[action_id]
            for action_id in (_require_action_id(item) for item in selected)
            if action_id not in maintained_ids
        )
        return ExecutionSubmission(
            cycle_id=self._last_facts.cycle_id,
            role_id=self._role_id,
            response_id=self._last_response_id or ResponseId("r-unknown"),
            actions=actions,
            schedules=new_schedules,
        )

    # ---- 内部编排 ----

    async def _rethink(
        self,
        facts: TurnFacts,
        response_id: ResponseId,
        *,
        feedback_reason: str | None,
        rejection_reason: str | None,
    ) -> RoleResponse:
        """重选意向（System2）后再由 System3 落实。"""
        intention = await system2.run(
            self._sampler,
            facts,
            tuple(self._memory),
            self._active_digest(),
            response_id=response_id,
            previous_intention=self._intention,
            feedback_reason=feedback_reason,
            rejection_reason=rejection_reason,
            background=self._background,
            tracer=self._tracer,
        )
        self._intention = intention.intention
        return await self._produce(facts, response_id, self._intention)

    async def _produce(
        self,
        facts: TurnFacts,
        response_id: ResponseId,
        intention: str,
        *,
        context_note: str | None = None,
    ) -> RoleResponse:
        """保持意向调用 System3；只有明确失败信号才升级 System2。

        正常输出即完整动作链候选。System3 输出意向不可落实的独立信号时，
        才升级 System2 重选意向（有限上限）；World 一次拒绝、API/格式错误
        都不是意向不可落实。升级或失败候选不覆盖已保存计划。
        """
        s2_calls = 0
        s3_calls = 0
        current = intention
        note = context_note
        while True:
            s3_calls += 1
            if s3_calls > self._max_system3_calls:
                raise RuntimeError(
                    f"角色 {self._role_id} 的 System3 落实尝试超过上限 "
                    f"{self._max_system3_calls}，停止推进"
                )
            outcome = await system3.run(
                self._sampler,
                facts,
                current,
                self._plan_digest(),
                response_id=response_id,
                context_note=note,
                attempt=s3_calls,
                background=self._background,
                memory=tuple(self._memory),
                tracer=self._tracer,
            )
            if isinstance(outcome, ActionPlan):
                assigned = self._assign_plan(outcome)
                self._candidate = assigned
                return self._compose_candidate(assigned, response_id)
            # 独立的意向不可落实信号：升级 System2 重选意向。
            s2_calls += 1
            if s2_calls > self._max_system2_calls:
                raise RuntimeError(
                    f"角色 {self._role_id} 的意向重选超过上限 {self._max_system2_calls}，停止推进"
                )
            reselected = await system2.run(
                self._sampler,
                facts,
                tuple(self._memory),
                self._active_digest(),
                response_id=response_id,
                previous_intention=current,
                feedback_reason=outcome.reason,
                rejection_reason=None,
                background=self._background,
                tracer=self._tracer,
            )
            current = reselected.intention
            self._intention = current
            note = None

    def _assign_plan(self, plan: ActionPlan) -> ActionPlan:
        """给新动作一次性分配身份、解析依赖；返回身份齐备的计划。

        未知、终态或跨角色身份、未定义/歧义/自依赖/重复/循环依赖都明确
        报错；已完成的历史动作可作为已确认依赖，不重新登记、不改写终态。
        生成候选不修改既有已接受动作的描述或参数。
        """
        assigned_ids: list[ActionId | None] = []
        for entry in plan.actions:
            action_id = entry.action_id
            if action_id is None:
                action_id = self._new_action_id()
                self._actions[action_id] = _OwnAction(
                    description=entry.action.display,
                    phase=_PLANNED,
                    confirmed=False,
                    maintainable=False,
                    action=entry.action,
                )
            else:
                own = self._actions.get(action_id)
                if own is None:
                    raise ModelOutputError(f"计划引用了未知行动身份：{action_id}")
                if own.phase not in _DECLARABLE_PHASES:
                    raise ModelOutputError(f"计划引用了终态行动身份，不能重声明：{action_id}")
                if own.action is not None and own.action != entry.action:
                    raise ModelOutputError(
                        "同 ID 行动不能改变目标或参数；结构化链用撤回旧身份并新增动作表达修改："
                        f"{action_id}"
                    )
            assigned_ids.append(action_id)
        final_ids: list[ActionId] = [item for item in assigned_ids if item is not None]
        if len(final_ids) != len(assigned_ids):  # pragma: no cover - 上面必经分配
            raise ModelOutputError("计划动作身份分配不完整")
        # key → 真实身份（仅本计划内引用）。
        key_to_id: dict[str, ActionId] = {
            entry.key: final_ids[index]
            for index, entry in enumerate(plan.actions)
            if entry.key is not None
        }
        resolved: list[PlanAction] = []
        for index, entry in enumerate(plan.actions):
            own_id = final_ids[index]
            deps: list[ActionId] = []
            for ref in entry.depends_on:
                if ref in key_to_id:
                    target = key_to_id[ref]
                elif ref in self._actions:
                    target = ActionId(ref)
                else:
                    raise ModelOutputError(f"依赖引用了未定义的动作：{ref}")
                if target == own_id:
                    raise ModelOutputError("动作不能依赖自身")
                if self._actions[target].phase not in _DECLARABLE_PHASES and (
                    self._actions[target].phase != _COMPLETED.value
                ):
                    raise ModelOutputError(f"依赖引用了终态但未完成的动作：{target}")
                if target in deps:
                    raise ModelOutputError(f"依赖重复指向同一动作：{target}")
                deps.append(target)
            if entry.action_id is None:
                self._actions[own_id].depends_on = tuple(str(item) for item in deps)
            else:
                own = self._actions[own_id]
                if set(own.depends_on) != {str(item) for item in deps}:
                    raise ModelOutputError(
                        f"同 ID 行动不能改变依赖；结构化链用撤回旧身份并新增动作表达修改：{own_id}"
                    )
            resolved.append(
                PlanAction(
                    action=entry.action,
                    action_id=own_id,
                    key=entry.key,
                    depends_on=tuple(str(item) for item in deps),
                )
            )
        return ActionPlan(actions=tuple(resolved))

    def _compose_reuse(self, response_id: ResponseId) -> RoleResponse:
        """复用已保存的完整链与排程；执行提交由 execution_submission 决定。"""
        plan = self._saved.plan if self._saved is not None else ActionPlan()
        response = RoleResponse(
            cycle_id=self._current_cycle(),
            role_id=self._role_id,
            response_id=response_id,
            actions=(),
            plan=plan,
            reused=True,
        )
        self._trace_compose(response_id, plan.actions)
        if self._tracer is not None and self._last_facts is not None:
            trace_stage(
                self._tracer,
                "role.reuse",
                self._last_facts,
                response_id,
                "复用已保存的完整链与排程，输出最近步骤与进行中声明",
                prompt="（复用私有缓存，无模型调用）",
                raw_output=None,
                action_ids=tuple(_require_action_id(item) for item in plan.actions),
            )
        return response

    def _compose_candidate(self, plan: ActionPlan, response_id: ResponseId) -> RoleResponse:
        """把候选完整链作为对外完整声明（含尚未满足依赖的后续动作）。"""
        actions = tuple(
            StepAction(
                _require_action_id(item),
                item.action,
                tuple(ActionId(ref) for ref in item.depends_on),
            )
            for item in plan.actions
        )
        response = RoleResponse(
            cycle_id=self._current_cycle(),
            role_id=self._role_id,
            response_id=response_id,
            actions=actions,
            plan=plan,
        )
        self._trace_compose(response_id, plan.actions)
        return response

    def _dependencies_met(self, item: PlanAction) -> bool:
        for ref in item.depends_on:
            dep = self._actions.get(ActionId(ref))
            if dep is None or dep.phase != _COMPLETED.value:
                return False
        return True

    def _trace_compose(self, response_id: ResponseId, active: tuple[PlanAction, ...]) -> None:
        """最终组装留痕：声明身份与依赖进入同一因果链。"""
        if self._tracer is None or self._last_facts is None:
            return
        facts = self._last_facts
        details: dict[str, TraceValue] = {
            "declared": len(active),
            "with_dependencies": sum(1 for item in active if item.depends_on),
        }
        trace_stage(
            self._tracer,
            "role.compose",
            facts,
            response_id,
            "组装当前完整声明（模型局部 key 已解析为真实身份）",
            prompt="（组装已决定的当前原子动作，无模型调用）",
            raw_output=None,
            details=details,
            action_ids=tuple(_require_action_id(item) for item in active),
        )

    def _align_with_known(self, known: tuple[KnownActionState, ...]) -> None:
        """用引擎的已知自身状态对齐：淘汰终态与未确认撤回。"""
        seen: set[ActionId] = set()
        for state in known:
            seen.add(state.action_id)
            own = self._actions.get(state.action_id)
            if own is None:
                continue  # 引擎侧存在但角色没有认知的行动（如重建后）。
            own.phase = state.phase.value
            own.confirmed = True
            own.maintainable = state.phase in _MAINTAINABLE
        for action_id, own in self._actions.items():
            if own.confirmed and own.maintainable and action_id not in seen:
                # 本轮已知状态中没有出现：不再可维持（撤回/移除）。
                own.maintainable = False

    def _active_ids(self) -> list[ActionId]:
        return [
            action_id
            for action_id, own in self._actions.items()
            if own.confirmed and own.maintainable
        ]

    def _active_digest(self) -> tuple[OwnActionDigest, ...]:
        return tuple(
            (str(action_id), own.phase, own.description)
            for action_id, own in sorted(self._actions.items(), key=lambda kv: str(kv[0]))
            if own.confirmed and own.maintainable
        )

    def _plan_digest(self) -> tuple[OwnActionDigest, ...]:
        """给 System1/System3 的本角色私有摘要：真实身份、确认阶段、依赖与已保存时序。

        只包含本角色已保存的完整链与排程，不含其他角色私有信息，也不含
        World 未感知的内容。
        """
        plan = self._saved.plan if self._saved is not None else self._candidate
        if plan is None:
            return self._active_digest()
        schedules = {
            item.action_id: item for item in (self._saved.schedules if self._saved else ())
        }
        rows: list[OwnActionDigest] = []
        for item in plan.actions:
            action_id = _require_action_id(item)
            own = self._actions.get(action_id)
            description = item.action.display
            if item.depends_on:
                description += "（依赖 " + "、".join(item.depends_on) + "）"
            schedule = schedules.get(action_id)
            if schedule is not None:
                window = f"{schedule.start}→{schedule.expected_end}ms"
                if schedule.checkpoints:
                    window += f"，节点 {list(schedule.checkpoints)}"
                description += f"（已保存排程 {window}）"
            phase = own.phase if own is not None else "unknown"
            rows.append((str(action_id), phase, description))
        return tuple(sorted(rows))

    def _set_pending(self, cycle_id: CycleId, response: RoleResponse) -> None:
        self._pending = (
            cycle_id,
            response.response_id,
            frozenset(item.action_id for item in response.actions),
        )

    def _next_response_id(self) -> ResponseId:
        self._response_seq += 1
        return ResponseId(f"r-{self._role_id}-{self._response_seq}")

    def _new_action_id(self) -> ActionId:
        self._action_seq += 1
        return ActionId(f"a-{self._role_id}-{self._action_seq}")

    def _remember(self, note: str) -> None:
        self._memory.append(note)
        if len(self._memory) > self._max_memory:
            del self._memory[: len(self._memory) - self._max_memory]

    def _current_cycle(self) -> CycleId:
        if self._last_facts is None:  # pragma: no cover - compose 前必有输入
            raise ValueError("缺少当前周期上下文")
        return self._last_facts.cycle_id

    @staticmethod
    def _facts(turn: TurnInput) -> TurnFacts:
        return TurnFacts(
            cycle_id=turn.cycle_id,
            role_id=turn.role_id,
            simulation_time=turn.simulation_time,
            event_ids=turn.trigger.event_ids,
            introspection=turn.trigger.introspection,
            observations=tuple(item.content for item in turn.observations),
            known_actions=tuple(
                (str(state.action_id), state.phase.value, state.detail)
                for state in turn.known_action_states
            ),
        )
