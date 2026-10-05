"""模型驱动的角色实现：内部串行编排 System1/2/3。

角色维护有界的私人认知与自身行动登记，并严格区分候选与已确认行动：
新行动只是提议，在引擎登记并经下一轮已知状态确认前不可被维持或修
改（身份保留以防复用）；明确修改作为待生效候选记录，只有随被接受
的声明得到确认后才进入私人认知的有效做法。打回候选不提前影响既有
安排——放弃/修改都随最终被接受的完整声明生效。

每份响应的身份在阶段开始前分配并贯穿 System1/2/3、内部重试、跳过、
错误与最终组装的留痕；World 打回原因、System3 重做原因与意向不可行
反馈沿调用链传递给下一次模型调用。perceive 与 rejected 由角色锁保证
同角色串行（跨角色并行）；新 perceive 使旧待检查响应失效；反馈还须
引用当前声明内的行动。模型采样经注入的采样器异步进行，超时/错误/
取消如实传播，解析失败不转换为空声明。
"""

import asyncio
from dataclasses import dataclass

from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    ResponseId,
    RoleId,
)
from sceneweave.domain.messages import (
    ActionPhase,
    KnownActionState,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Rejection,
    RoleResponse,
    TurnInput,
)
from sceneweave.domain.tracing import TraceValue, TurnTracer
from sceneweave.roles.base import Role
from sceneweave.roles.components import system1, system2, system3
from sceneweave.roles.components.base import TextSampler
from sceneweave.roles.components.support import OwnActionDigest, TurnFacts, trace_stage
from sceneweave.roles.model_output import ModelOutputError

# 仍可被主动维持的阶段；未确认表示结果未知，不是已知终止。
_MAINTAINABLE = frozenset({ActionPhase.REGISTERED, ActionPhase.STARTED, ActionPhase.UNCONFIRMED})


@dataclass(slots=True)
class _OwnAction:
    """角色对一个行动身份的私人认知。

    confirmed 为假表示只是候选：尚未经引擎登记确认，不可维持或修改，
    但身份保留以防复用。pending_change 是最近一份候选携带的明确修改，
    只有随被接受的声明确认后才写入 approach。
    """

    description: str
    phase: str
    confirmed: bool
    maintainable: bool
    pending_change: str | None = None
    approach: str | None = None


def _require_positive(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field}必须是正整数，得到：{value!r}")
    return value


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
        self._max_system2_calls = _require_positive(max_system2_calls, "System2 次数上限")
        self._max_system3_calls = _require_positive(max_system3_calls, "System3 次数上限")
        self._memory: list[str] = []
        # 全部历史行动身份都保留：新行动身份永不复用；未确认的候选
        # 身份同样保留但永不可维持。
        self._actions: dict[ActionId, _OwnAction] = {}
        self._action_seq = 0
        self._response_seq = 0
        self._group_seq = 0
        self._intention: str | None = None
        self._pending: tuple[CycleId, ResponseId, frozenset[ActionId]] | None = None
        self._last_facts: TurnFacts | None = None
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
        """自身行动登记的只读摘要：(身份, 阶段, 描述含有效做法, 可维持)。"""
        return tuple(
            (
                str(action_id),
                own.phase,
                own.description + (f"；当前做法：{own.approach}" if own.approach else ""),
                own.maintainable,
            )
            for action_id, own in sorted(self._actions.items(), key=lambda kv: str(kv[0]))
        )

    # ---- 对引擎的交互契约 ----

    async def perceive(self, turn: TurnInput) -> RoleResponse:
        if turn.role_id != self._role_id:
            raise ValueError(f"输入角色身份不一致：{turn.role_id}")
        async with self._lock:
            facts = self._facts(turn)
            self._last_facts = facts
            # 新 perceive 开始：旧的待检查响应即刻失效。
            self._pending = None
            self._align_with_known(turn.known_action_states)
            response_id = self._next_response_id()
            decision = await system1.run(
                self._sampler,
                facts,
                tuple(self._memory),
                response_id=response_id,
                background=self._background,
                tracer=self._tracer,
            )
            self._remember(decision.note)
            if not decision.reconsider:
                for stage in ("system2.skipped", "system3.skipped"):
                    trace_stage(
                        self._tracer,
                        stage,
                        facts,
                        response_id,
                        "System1 判断无需重新考虑，跳过模型调用并维持既有行动",
                        prompt="（跳过模型调用）",
                        raw_output=None,
                        action_ids=tuple(self._active_ids()),
                    )
                # 认识更新不强制行动：返回完整维持集合（可为显式空）。
                response = self._compose(
                    response_id,
                    maintain=tuple(self._active_ids()),
                    modify=(),
                    new_descriptions=(),
                )
                self._set_pending(facts.cycle_id, response)
                return response
            response = await self._decide(facts, response_id, rejection_reason=None)
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
            # 原候选已被明确拒绝，即使重做失败，也不能继续使用其反馈或修改。
            self._pending = None
            for own in self._actions.values():
                own.pending_change = None
            # 不重投原始感知：只用既有事实与打回原因重新决定。
            response_id = self._next_response_id()
            response = await self._decide(
                self._last_facts, response_id, rejection_reason=feedback.reason
            )
            self._set_pending(feedback.cycle_id, response)
            return response

    # ---- 内部编排 ----

    async def _decide(
        self, facts: TurnFacts, response_id: ResponseId, *, rejection_reason: str | None
    ) -> RoleResponse:
        """System2 → System3 的串行编排，含反馈与有界重试。

        World 打回原因、System3 内部重做原因与意向不可行反馈都沿调用
        链传入下一次模型调用，使重做真正针对问题。
        """
        system2_calls = 0
        system3_calls = 0
        infeasible_reason: str | None = None
        retry_reason: str | None = rejection_reason
        while True:
            if system2_calls >= self._max_system2_calls:
                raise RuntimeError(
                    f"角色 {self._role_id} 在周期 {facts.cycle_id} 的意向选择次数"
                    f"超过上限 {self._max_system2_calls}"
                )
            intention = await system2.run(
                self._sampler,
                facts,
                tuple(self._memory),
                self._active_digest(),
                response_id=response_id,
                previous_intention=self._intention,
                feedback_reason=infeasible_reason,
                rejection_reason=rejection_reason if system2_calls == 0 else None,
                background=self._background,
                tracer=self._tracer,
            )
            system2_calls += 1
            self._intention = intention.intention
            self._validate_abandon(intention.abandon)
            while True:
                if system3_calls >= self._max_system3_calls:
                    raise RuntimeError(
                        f"角色 {self._role_id} 在周期 {facts.cycle_id} 的落实尝试次数"
                        f"超过上限 {self._max_system3_calls}"
                    )
                context_note = retry_reason if retry_reason is not None else infeasible_reason
                plan = await system3.run(
                    self._sampler,
                    facts,
                    intention.intention,
                    self._active_digest(),
                    response_id=response_id,
                    context_note=(
                        None if system3_calls == 0 and retry_reason is None else context_note
                    ),
                    abandon=tuple(str(item) for item in intention.abandon),
                    attempt=system3_calls + 1,
                    background=self._background,
                    memory=tuple(self._memory),
                    tracer=self._tracer,
                )
                system3_calls += 1
                if plan.retry:
                    retry_reason = plan.reason  # 做法不可行：带着原因重做。
                    continue
                retry_reason = None
                if not plan.feasible:
                    infeasible_reason = plan.reason  # 意向不可行：反馈 System2。
                    break
                return self._compose(
                    response_id,
                    maintain=plan.maintain,
                    modify=tuple((item.action_id, item.change) for item in plan.modify),
                    new_descriptions=plan.new_descriptions,
                )

    def _compose(
        self,
        response_id: ResponseId,
        *,
        maintain: tuple[ActionId, ...],
        modify: tuple[tuple[ActionId, str], ...],
        new_descriptions: tuple[str, ...],
    ) -> RoleResponse:
        """把计划组装为完整 RoleResponse。

        全量声明语义由响应本身表达（未列出即撤回主动维持）；新行动与
        修改都只是本候选的内容：新身份登记为未确认候选，修改记录为
        待生效变更，均待下一轮引擎已知状态确认。
        """
        actions: list[MaintainAction | ModifiedAction | NewAction] = []
        modify_ids = {action_id for action_id, _ in modify}
        # 最近候选未再修改的已确认行动：清除旧的待生效变更。
        for action_id, own in self._actions.items():
            if own.confirmed and own.pending_change is not None and action_id not in modify_ids:
                own.pending_change = None
        for action_id in maintain:
            self._require_maintainable(action_id, "维持")
            actions.append(MaintainAction(action_id))
        for action_id, change in modify:
            own = self._require_maintainable(action_id, "修改")
            own.pending_change = change
            actions.append(ModifiedAction(action_id, change))
        group: CommitGroupId | None = None
        for description in new_descriptions:
            if group is None:
                self._group_seq += 1
                group = CommitGroupId(f"g-{self._role_id}-{self._group_seq}")
            self._action_seq += 1
            action_id = ActionId(f"a-{self._role_id}-{self._action_seq}")
            self._actions[action_id] = _OwnAction(
                description=description,
                phase=ActionPhase.SUBMITTED.value,
                confirmed=False,
                maintainable=False,
            )
            actions.append(NewAction(action_id, description, group))
        self._trace_compose(response_id, actions)
        return RoleResponse(
            cycle_id=self._current_cycle(),
            role_id=self._role_id,
            response_id=response_id,
            actions=tuple(actions),
        )

    def _trace_compose(
        self,
        response_id: ResponseId,
        actions: list[MaintainAction | ModifiedAction | NewAction],
    ) -> None:
        """最终组装留痕：完整声明身份与行动身份进入同一因果链。"""
        if self._tracer is None or self._last_facts is None:
            return
        facts = self._last_facts
        details: dict[str, TraceValue] = {
            "maintain": sum(isinstance(item, MaintainAction) for item in actions),
            "modify": sum(isinstance(item, ModifiedAction) for item in actions),
            "new_actions": sum(isinstance(item, NewAction) for item in actions),
        }
        trace_stage(
            self._tracer,
            "role.compose",
            facts,
            response_id,
            "组装完整主动行为声明",
            prompt="（组装已决定的完整主动声明，无模型调用）",
            raw_output=None,
            details=details,
            action_ids=tuple(item.action_id for item in actions),
        )

    def _require_maintainable(self, action_id: ActionId, verb: str) -> _OwnAction:
        own = self._actions.get(action_id)
        if own is None:
            raise ModelOutputError(f"{verb}声明引用未知行动：{action_id}")
        if not own.confirmed:
            raise ModelOutputError(f"{verb}声明引用尚未经引擎登记确认的候选行动：{action_id}")
        if not own.maintainable:
            raise ModelOutputError(f"不能继续{verb}已终态或已放弃的行动：{action_id}")
        return own

    def _validate_abandon(self, abandon: tuple[ActionId, ...]) -> None:
        """放弃引用必须是已知身份；放弃本身不改变私人信念。

        弃维持属于候选内容：随被接受的完整声明生效，打回后的重试可以
        重新决定维持、修改或撤回。
        """
        for action_id in abandon:
            if action_id not in self._actions:
                raise ModelOutputError(f"放弃声明引用未知行动：{action_id}")

    def _align_with_known(self, known: tuple[KnownActionState, ...]) -> None:
        """用引擎的已知自身状态对齐：确认候选、淘汰终态与未确认撤回。"""
        seen: set[ActionId] = set()
        for state in known:
            seen.add(state.action_id)
            own = self._actions.get(state.action_id)
            if own is None:
                continue  # 引擎侧存在但角色没有认知的行动（如重建后）。
            own.phase = state.phase.value
            own.confirmed = True
            own.maintainable = state.phase in _MAINTAINABLE
            if own.maintainable and own.pending_change is not None:
                # 随被接受的声明确认的修改：进入有效做法的私人认知。
                own.approach = own.pending_change
                own.pending_change = None
            elif not own.maintainable:
                own.pending_change = None
        for action_id, own in self._actions.items():
            if own.confirmed and own.maintainable and action_id not in seen:
                # 本轮已知状态中没有出现：不再可维持（撤回/移除）。
                own.maintainable = False
            # 未确认的候选身份：保留防复用，但永不可维持。

    def _active_ids(self) -> list[ActionId]:
        return [
            action_id
            for action_id, own in self._actions.items()
            if own.confirmed and own.maintainable
        ]

    def _active_digest(self) -> tuple[OwnActionDigest, ...]:
        return tuple(
            (
                str(action_id),
                own.phase,
                own.description + (f"；当前做法：{own.approach}" if own.approach else ""),
            )
            for action_id, own in sorted(self._actions.items(), key=lambda kv: str(kv[0]))
            if own.confirmed and own.maintainable
        )

    def _set_pending(self, cycle_id: CycleId, response: RoleResponse) -> None:
        self._pending = (
            cycle_id,
            response.response_id,
            frozenset(item.action_id for item in response.actions),
        )

    def _next_response_id(self) -> ResponseId:
        self._response_seq += 1
        return ResponseId(f"r-{self._role_id}-{self._response_seq}")

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
