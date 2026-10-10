"""调度与生命周期测试共用的假角色、假世界与假翻译器。

全部使用脚本与事件门控，不依赖墙钟睡眠，也不调用真实模型。
FakeSemanticWorld 自身履行 World 契约：原子提交（校验通过后一次性
应用并递增版本）、进展更新在位置实际变化时递增版本、只读入口不改
变状态。引擎侧的运行时校验责任（候选与版本关联、投递引用、登记计
划完整性、结算结果与节点兼容）在被测的调度器中，测试分别断言。
"""

import asyncio
from collections.abc import Callable, Sequence
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
    CommitGroupId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    Delivery,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Observation,
    Rejection,
    RoleResponse,
    StepAction,
    TurnInput,
    WorldEvent,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import (
    ActionArrangement,
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationRequest,
    TranslationResult,
)
from sceneweave.roles.base import Role
from sceneweave.world.base import World


def rid(name: str) -> RoleId:
    return RoleId(name)


def cycle(name: str) -> CycleId:
    return CycleId(name)


def ms(value: int) -> SimulationTime:
    return SimulationTime(value)


def maintain(action_id: str) -> MaintainAction:
    return MaintainAction(ActionId(action_id))


def new_action(
    action_id: str,
    description: str = "尝试行动",
    group: str = "group-1",
) -> NewAction:
    return NewAction(ActionId(action_id), description, CommitGroupId(group))


def modified(action_id: str, change: str = "放慢速度") -> ModifiedAction:
    return ModifiedAction(ActionId(action_id), change)


def response(
    role_id: str,
    response_id: str,
    actions: tuple[Any, ...] = (),
    *,
    cycle_id: str = "cycle-1",
) -> RoleResponse:
    return RoleResponse(cycle(cycle_id), rid(role_id), ResponseId(response_id), actions)


def idle_response(role_id: str) -> RoleResponse:
    """空集合响应：撤回全部主动维持；可作为脚本耗尽后的默认轮次。"""
    return response(role_id, f"r-{role_id}-idle", ())


def maintaining(role_id: str, *action_ids: str) -> RoleResponse:
    """维持指定行动的响应；作为长行动期间的默认轮次。"""
    return response(
        role_id,
        f"r-{role_id}-keep",
        tuple(maintain(action_id) for action_id in action_ids),
    )


class ScriptedRole(Role):
    """按脚本逐轮返回完整声明的假角色；脚本项为响应或异常。

    gate 提供 asyncio.Event 时 perceive 阻塞到放行，用于验证冻结与
    屏障；perceive 每周期消耗一项，rejected 每次打回消耗一项。脚本
    响应的周期身份由角色对齐到当前周期。
    """

    def __init__(
        self,
        role_id: str,
        rounds: Sequence[RoleResponse | BaseException] = (),
        *,
        gate: asyncio.Event | None = None,
        fallback: RoleResponse | None = None,
    ) -> None:
        self._role_id = rid(role_id)
        self.rounds = list(rounds)
        self.gate = gate
        self.fallback = fallback
        self.perceived: list[TurnInput] = []
        self.rejections: list[Rejection] = []
        self.started = asyncio.Event()
        self.completed = asyncio.Event()

    @property
    def role_id(self) -> RoleId:
        return self._role_id

    def _adapt(self, item: RoleResponse, cycle_id: CycleId) -> RoleResponse:
        """把脚本响应对齐到当前周期；脚本无需预知周期编号。"""
        if item.cycle_id == cycle_id:
            return item
        import dataclasses

        return dataclasses.replace(item, cycle_id=cycle_id)

    def _next(self, cycle_id: CycleId) -> RoleResponse:
        if not self.rounds:
            if self.fallback is not None:
                return self._adapt(self.fallback, cycle_id)
            raise AssertionError(f"角色 {self._role_id} 的脚本已耗尽")
        item = self.rounds.pop(0)
        if isinstance(item, BaseException):
            raise item
        return self._adapt(item, cycle_id)

    async def perceive(self, turn: TurnInput) -> RoleResponse:
        self.perceived.append(turn)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        self.completed.set()
        return self._next(turn.cycle_id)

    async def rejected(self, feedback: Rejection) -> RoleResponse:
        self.rejections.append(feedback)
        return self._next(feedback.cycle_id)


class FakeSemanticWorld(World):
    """带简单语义的假世界：位置推进、行动结算、修改生效与被动影响。

    - update_progress 按 STARTED 行动速度推进位置（只依赖剧情时间），
      位置实际变化时递增共同状态版本；
    - adjudicate 对世界事件落地、行动开始/过程/结束、撤回与修改给出
      候选；修改按「更快/更慢」变化实际调整速度，位置保持连续；
    - 撤回主动维持产生惯性被动效果与后续事件；被动效果不因角色不再
      声明而清空，只由对应后续事件结算；
    - commit 在校验通过后一次性应用并递增版本（假世界自身原子性）；
      行动归属取自节点携带的 role_id，不靠旁路注入。
    """

    def __init__(self, role_ids: Sequence[str]) -> None:
        self.role_ids = tuple(role_ids)
        self._version = WorldStateVersion(0)
        self.positions: dict[str, float] = {name: 0.0 for name in role_ids}
        self.speeds: dict[ActionId, float] = {}
        self.facts: list[str] = []
        self.passive_effects: list[str] = []
        self.adjudications: list[dict[str, Any]] = []
        self.checks: list[tuple[RoleResponse, tuple[ActionArrangement, ...]]] = []
        self.outcome_overrides: dict[ActionId, ActionOutcomeResult] = {}
        self.node_overrides: dict[tuple[ActionId, ActionNodeKind], ActionOutcomeResult] = {}
        self.passive_follow_up_delay = 500
        self.progress_gate: asyncio.Event | None = None
        self.progress_entered = asyncio.Event()
        self.progress_calls: list[tuple[CycleId, int]] = []
        self.adjudication_gate: asyncio.Event | None = None
        self.adjudication_entered = asyncio.Event()
        self._committed: list[AdjudicationCommit] = []
        self._reference = EventId("engine-observed")
        self._last_time = SimulationTime(0)
        self._started_owners: list[tuple[ActionId, str]] = []
        self._pending_nodes: tuple[AdjudicationActionNode, ...] = ()
        self._pending_modifications: tuple[ActionModificationNode, ...] = ()

    @property
    def state_version(self) -> WorldStateVersion:
        return self._version

    def set_speed(self, action_id: str, speed: float) -> None:
        self.speeds[ActionId(action_id)] = speed

    def bump_for_test(self) -> None:
        """测试专用：直接递增版本，模拟外部副作用。"""
        self._version = WorldStateVersion(self._version + 1)

    def rewind_for_test(self) -> None:
        """测试专用：回退版本，模拟谎报提交但实际未变。"""
        self._version = WorldStateVersion(max(0, self._version - 1))

    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        self.progress_entered.set()
        self.progress_calls.append((cycle_id, simulation_time))
        if self.progress_gate is not None:
            await self.progress_gate.wait()
        elapsed = simulation_time - self._last_time
        self._last_time = simulation_time
        updated: list[ActionId] = []
        notes: list[str] = []
        moved = False
        for action_id, owner in self._started_owners:
            speed = self.speeds.get(action_id, 0.0)
            if speed:
                self.positions[owner] += speed * elapsed
                moved = True
                updated.append(action_id)
                notes.append(
                    f"{action_id} 推进 {elapsed} 毫秒，{owner} 位置 {self.positions[owner]:.2f}"
                )
        if moved:
            # 实际位置变化是原子进展提交：递增共同状态版本。
            self._version = WorldStateVersion(self._version + 1)
        return ProgressUpdate(
            cycle_id=cycle_id,
            simulation_time=simulation_time,
            updated_action_ids=tuple(sorted(updated, key=str)),
            notes=tuple(notes),
        )

    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        """只读客观上下文：位置与进行中行动的实际进展描述。"""
        conditions = [f"{role_id} 位置 {self.positions.get(str(role_id), 0.0):.2f} 米"]
        for action_id, owner in self._started_owners:
            if owner == role_id:
                conditions.append(f"{action_id} 实际进行中，速度 {self.speeds.get(action_id, 0.0)}")
        return ObjectiveContext(
            world_state_version=self._version,
            role_id=role_id,
            conditions=tuple(conditions),
        )

    def diagnostic_snapshot(self) -> WorldDiagnostic:
        """只读诊断：实际行动状态与被动过程（区别于事实增量文本）。"""
        return WorldDiagnostic(
            world_state_version=self._version,
            action_states=tuple(
                sorted(
                    (
                        WorldActionState(
                            action_id=action_id,
                            role_id=rid(owner),
                            state="进行中",
                            detail=f"速度 {self.speeds.get(action_id, 0.0)}，位置 "
                            f"{self.positions.get(owner, 0.0):.2f}",
                        )
                        for action_id, owner in self._started_owners
                    ),
                    key=lambda item: str(item.action_id),
                )
            ),
            passive_processes=tuple(self.passive_effects),
        )

    async def check_response(
        self,
        candidate: RoleResponse,
        *,
        arrangements: Sequence[ActionArrangement] = (),
        schedules: Sequence[ActionSchedule] = (),
    ) -> ActionCheckResult:
        self.checks.append((candidate, tuple(arrangements)))
        blocked = tuple(
            item.action_id
            for item in candidate.actions
            if (isinstance(item, NewAction) and "不可能" in item.description)
            or (isinstance(item, StepAction) and "不可能" in item.action.display)
        )
        return ActionCheckResult(
            cycle_id=candidate.cycle_id,
            role_id=candidate.role_id,
            response_id=candidate.response_id,
            world_state_version=self.state_version,
            passed=not blocked,
            reason="" if not blocked else "客观条件不允许该行动",
            action_ids=blocked,
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
        self.adjudication_entered.set()
        if self.adjudication_gate is not None:
            await self.adjudication_gate.wait()
        self.adjudications.append(
            {
                "cycle_id": cycle_id,
                "simulation_time": simulation_time,
                "event_ids": tuple(event_ids),
                "action_ids": tuple(action_ids),
                "events": tuple(events),
                "action_nodes": tuple(action_nodes),
                "modifications": tuple(modifications),
            }
        )
        if tuple(event_ids):
            self._reference = tuple(event_ids)[0]
        elif action_nodes:
            self._reference = action_nodes[0].event_id
        # 记住本次裁决的节点与修改：提交时据此应用归属与实际作用。
        self._pending_nodes = tuple(action_nodes)
        self._pending_modifications = tuple(modifications)
        changes: list[FactChangeProposal] = []
        outcomes: list[ActionOutcome] = []
        follow_ups: list[WorldEvent] = []
        for event in events:
            changes.append(FactChangeProposal(f"事件 {event.event_id} 发生：{event.description}"))
        for node in action_nodes:
            if node.node is ActionNodeKind.START:
                result = self.node_overrides.get(
                    (node.action_id, ActionNodeKind.START), ActionOutcomeResult.STARTED
                )
                outcomes.append(ActionOutcome(node.action_id, result, "行动实际开始"))
                changes.append(FactChangeProposal(f"{node.action_id} 开始：{node.description}"))
            elif node.node is ActionNodeKind.CHECKPOINT:
                result = self.node_overrides.get(
                    (node.action_id, ActionNodeKind.CHECKPOINT), ActionOutcomeResult.IN_PROGRESS
                )
                outcomes.append(ActionOutcome(node.action_id, result, "过程节点"))
                changes.append(FactChangeProposal(f"{node.action_id} 到达过程节点"))
            elif node.node is ActionNodeKind.END:
                result = self.node_overrides.get(
                    (node.action_id, ActionNodeKind.END),
                    self.outcome_overrides.get(node.action_id, ActionOutcomeResult.COMPLETED),
                )
                outcomes.append(ActionOutcome(node.action_id, result, f"结算结果 {result.value}"))
                changes.append(FactChangeProposal(f"{node.action_id} 结算：{result.value}"))
            else:  # WITHDRAWN：撤回主动维持的客观效果。
                changes.append(
                    FactChangeProposal(f"{node.action_id} 撤回主动维持：身体停止，惯性继续")
                )
                outcomes.append(
                    ActionOutcome(
                        node.action_id,
                        ActionOutcomeResult.IN_PROGRESS,
                        "撤回主动维持：身体停止，惯性继续",
                    )
                )
                follow_ups.append(
                    WorldEvent(
                        EventId(f"{node.action_id}-passive-end"),
                        SimulationTime(simulation_time + self.passive_follow_up_delay),
                        f"{node.action_id} 的惯性滑行结束",
                    )
                )
        for node in modifications:
            changes.append(
                FactChangeProposal(
                    f"{node.action_id} 修改生效（{node.role_id}）：{node.change}，"
                    f"预计结束 {node.old_expected_end}→{node.new_expected_end}"
                )
            )
            outcomes.append(
                ActionOutcome(node.action_id, ActionOutcomeResult.IN_PROGRESS, "修改后的实际进展")
            )
        return AdjudicationCandidate(
            cycle_id=cycle_id,
            simulation_time=simulation_time,
            event_ids=tuple(event_ids),
            action_ids=tuple(action_ids),
            proposed_changes=tuple(changes),
            follow_up_events=tuple(follow_ups),
            action_outcomes=tuple(outcomes),
        )

    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        problems: list[str] = []
        if not candidate.proposed_changes:
            problems.append("候选没有可提交的事实变更")
        return AdjudicationValidation(
            candidate=candidate,
            world_state_version=self._version,
            passed=not problems,
            problems=tuple(problems),
        )

    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        validation = self.validate_adjudication(candidate)
        if validation.candidate != candidate or not validation.passed:
            raise ValueError("假世界拒绝提交未通过校验的候选")
        if validation.world_state_version != self._version:
            raise ValueError("假世界检测到状态版本漂移")
        # 原子应用：全部变更一次落地后递增版本。
        self.facts.extend(item.description for item in candidate.proposed_changes)
        terminal: set[ActionId] = set()
        for outcome in candidate.action_outcomes:
            if outcome.result in (
                ActionOutcomeResult.COMPLETED,
                ActionOutcomeResult.FAILED,
            ):
                terminal.add(outcome.action_id)
        self._started_owners = [item for item in self._started_owners if item[0] not in terminal]
        # 开始结果按节点归属登记推进归属；修改按明确变化实际调整速度。
        for node in self._pending_nodes:
            if node.node is ActionNodeKind.START and not any(
                outcome.action_id == node.action_id
                and outcome.result
                not in (ActionOutcomeResult.STARTED, ActionOutcomeResult.UNCONFIRMED)
                for outcome in candidate.action_outcomes
            ):
                self._started_owners.append((node.action_id, str(node.role_id)))
        for node in self._pending_modifications:
            speed = self.speeds.get(node.action_id, 0.0)
            if "更快" in node.change:
                self.speeds[node.action_id] = speed * 2
            elif "更慢" in node.change:
                self.speeds[node.action_id] = speed / 2
        for node in candidate.follow_up_events:
            self.passive_effects.append(node.description)
        for event_id in candidate.event_ids:
            # 被动过程结束：惯性滑行到后续事件结算为止。
            if str(event_id).endswith("-passive-end"):
                self._started_owners = [
                    item
                    for item in self._started_owners
                    if f"{item[0]}-passive-end" == str(event_id)
                ]
        self._pending_nodes = ()
        self._pending_modifications = ()
        self._version = WorldStateVersion(self._version + 1)
        commit = AdjudicationCommit(
            candidate=candidate,
            committed_changes=candidate.proposed_changes,
            resulting_version=self._version,
            action_outcomes=candidate.action_outcomes,
        )
        self._committed.append(commit)
        return commit

    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        if not any(item is commit for item in self._committed):
            raise ValueError("只能发布已提交的裁决")
        return tuple(
            Delivery(
                rid(name),
                Observation(self._reference, commit.candidate.simulation_time, change.description),
            )
            for change in commit.committed_changes
            for name in self.role_ids
        )


class ScriptedTranslator:
    """脚本假翻译器：新行动按统一延迟/耗时安排，修改按脚本安排。

    plans 按行动描述给出（开始延迟、持续毫秒、过程节点偏移毫秒）；
    modifications 按行动身份给出（新预计结束、追加过程节点偏移）。
    需要完全控制时提供 override 回调。
    """

    def __init__(
        self,
        *,
        delay: int = 100,
        duration: int = 1000,
        checkpoints: tuple[int, ...] = (),
        plans: dict[str, tuple[int, int, tuple[int, ...]]] | None = None,
        modifications: dict[str, tuple[int, tuple[int, ...]]] | None = None,
        override: Callable[[TranslationRequest], TranslationResult] | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.delay = delay
        self.duration = duration
        self.checkpoints = checkpoints
        self.plans = plans or {}
        self.modifications = modifications or {}
        self.override = override
        self.gate = gate
        self.requests: list[TranslationRequest] = []

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.override is not None:
            return self.override(request)
        return self.default_result(request)

    def default_result(self, request: TranslationRequest) -> TranslationResult:
        """不经 override 的标准安排；供 override 分支复用。"""
        arrangement_ids = {item.action_id for item in request.arrangements}
        schedules: list[ActionSchedule] = []
        for item in request.response.actions:
            if isinstance(item, StepAction):
                if item.action_id in arrangement_ids:
                    continue  # 既有动作重声明不重新翻译。
                delay, duration, checkpoints = self.plans.get(
                    item.action.display, (self.delay, self.duration, self.checkpoints)
                )
                start = request.current_time + delay
                end = start + duration
                schedules.append(
                    ActionSchedule(
                        action_id=item.action_id,
                        commit_group_id=CommitGroupId(
                            f"g-{request.response.role_id}-{request.response.response_id}"
                        ),
                        start=SimulationTime(start),
                        expected_end=SimulationTime(end),
                        checkpoints=tuple(SimulationTime(start + c) for c in checkpoints),
                    )
                )
            elif isinstance(item, NewAction):
                delay, duration, checkpoints = self.plans.get(
                    item.description, (self.delay, self.duration, self.checkpoints)
                )
                start = request.current_time + delay
                end = start + duration
                schedules.append(
                    ActionSchedule(
                        action_id=item.action_id,
                        commit_group_id=item.commit_group_id,
                        start=SimulationTime(start),
                        expected_end=SimulationTime(end),
                        checkpoints=tuple(SimulationTime(start + c) for c in checkpoints),
                    )
                )
        modified_schedules: list[ModifiedActionSchedule] = []
        for item in request.response.actions:
            if isinstance(item, ModifiedAction):
                arrangement = next(
                    (a for a in request.arrangements if a.action_id == item.action_id), None
                )
                if arrangement is None:
                    continue
                new_end, extra = self.modifications.get(
                    str(item.action_id), (int(arrangement.expected_end), ())
                )
                kept = [c for c in arrangement.checkpoints if c <= request.current_time]
                future = [int(request.current_time) + c for c in extra]
                modified_schedules.append(
                    ModifiedActionSchedule(
                        action_id=item.action_id,
                        start=arrangement.start,
                        expected_end=SimulationTime(new_end),
                        checkpoints=tuple(SimulationTime(c) for c in sorted(kept + future)),
                    )
                )
        return TranslationResult(
            cycle_id=request.response.cycle_id,
            role_id=request.response.role_id,
            response_id=request.response.response_id,
            schedules=tuple(schedules),
            modified_schedules=tuple(modified_schedules),
        )


class ScriptedSampler:
    """脚本采样器：按组件名顺序返回脚本文本；无脚本即错误。

    gate 提供事件时在记录调用后阻塞，用于验证跨角色并行与剧情冻结。
    严格无默认回答：组件缺少脚本直接报错，不悄悄 fallback。
    """

    def __init__(
        self,
        script: dict[str, list[str]] | None = None,
        *,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.script = {name: list(items) for name, items in (script or {}).items()}
        self.gate = gate
        self.calls: list[tuple[str, str]] = []
        self.in_flight = 0

    def components(self) -> list[str]:
        return [component for component, _ in self.calls]

    async def sample(self, prompt: str, *, component: str) -> str:
        self.calls.append((component, prompt))
        self.in_flight += 1
        try:
            if self.gate is not None:
                await self.gate.wait()
            responses = self.script.get(component)
            if responses is None:
                raise AssertionError(f"组件 {component} 没有脚本回答")
            if not responses:
                raise AssertionError(f"组件 {component} 的脚本已耗尽")
            return responses.pop(0)
        finally:
            self.in_flight -= 1
