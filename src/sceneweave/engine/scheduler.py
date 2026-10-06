"""剧情时钟、事件调度与统一登记的完整闭环。

调度器按最近待处理事件跳转剧情时刻：进入时刻后先让 World 按已流逝
剧情时间更新进展（实际变化原子应用并递增版本），再对同刻整批事件做
联合裁决（基于预裁决版本：候选、校验、原子提交、发布感知），随后
全部角色响应（复用 ResponseRunner 屏障）、翻译与时间校验、统一登记
与新声明生效。整轮完成前剧情时钟保持冻结：翻译结束后复核共同状态
版本未被改动，模型调用与现实等待不推进时间，引擎也不凭角色意向更
新客观状态。

统一登记是原子边界：登记计划与队列变更先在独立候选结构中完整准备
并校验（含事件身份避让与跨角色唯一性），撤回与修改的客观效果经
World 裁决提交后，只执行不再会失败的发布步骤。运行门（暂停/继续/
单步/停止）在每个完整周期前的安全边界等待——初始化同样是安全边界，
首次事件处理也须先取得许可；暂停不打断 World 提交、不推进剧情时钟，
也不计入任何墙钟预算（本项目没有墙钟总预算，仅模型请求自身可有
超时）。
"""

import asyncio
import enum
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sceneweave.domain.adjudication import (
    ActionModificationNode,
    ActionNodeKind,
    AdjudicationActionNode,
    AdjudicationCandidate,
    AdjudicationCommit,
    AdjudicationValidation,
    ProgressUpdate,
    WorldDiagnostic,
    allowed_results_by_action,
)
from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_version,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    ActionPhase,
    Delivery,
    KnownActionState,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Rejection,
    RoleResponse,
    TriggerReason,
    TurnInput,
    WorldEvent,
)
from sceneweave.domain.scheduling import EventQueue, ScheduledEvent, ScheduledEventKind
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.tracing import TraceRecord, TraceValue, TurnTracer
from sceneweave.domain.translation import TranslationRequest, TranslationResult
from sceneweave.engine.actions import (
    MAINTAINABLE_PHASES,
    TERMINAL_PHASES,
    ActionRecord,
    ActionRegistry,
    FinalizedDeclaration,
    RegistrationPlan,
    outcome_transition_problems,
    prepare_registration,
)
from sceneweave.engine.config import RunConfig
from sceneweave.engine.control import RunGate, RunStatus
from sceneweave.engine.registration import registration_issues
from sceneweave.engine.report import CycleEvent, CycleReport, RoleCycleView
from sceneweave.engine.responses import AcceptedResponse, CycleOutcome, ResponseRunner
from sceneweave.roles.base import Role
from sceneweave.translator.base import Translator
from sceneweave.world.base import World

# 运行级留痕使用的周期身份；与真实处理周期区分。
_RUN_CYCLE = CycleId("engine")

_NODE_KINDS: Mapping[ScheduledEventKind, ActionNodeKind] = {
    ScheduledEventKind.ACTION_START: ActionNodeKind.START,
    ScheduledEventKind.ACTION_CHECKPOINT: ActionNodeKind.CHECKPOINT,
    ScheduledEventKind.ACTION_END: ActionNodeKind.END,
}


class RunStopReason(enum.StrEnum):
    """运行停止原因；自然结束、时限、用户中止与运行错误分别明确。"""

    NATURAL_END = "natural_end"
    TIME_LIMIT = "time_limit"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """错误现场中保留的未生效候选摘要；不包含角色私有状态。"""

    role_id: RoleId
    response_id: ResponseId
    actions: tuple[str, ...]
    schedules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """错误现场快照：周期、取出批、队列、登记、阶段、版本、候选与报错。

    只导出引擎可拥有的状态；不宣称已恢复角色私有状态，也不录制模型
    输入输出（模型调用录制见 integrations 的在途模块）。
    """

    stage: str
    cycle_id: CycleId | None
    simulation_time: SimulationTime
    world_state_version: WorldStateVersion
    active_batch: tuple[ScheduledEvent, ...]
    pending_events: tuple[ScheduledEvent, ...]
    action_records: tuple[ActionRecord, ...]
    candidates: tuple[CandidateSummary, ...]
    pending_plan: tuple[str, ...]
    adjudication: str | None
    error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("阶段必须是非空字符串")
        if self.cycle_id is not None and (not isinstance(self.cycle_id, str) or not self.cycle_id):
            raise ValueError("周期身份必须是非空字符串或 None")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        require_version(self.world_state_version, "共同世界状态版本")
        for name in ("active_batch", "pending_events", "action_records"):
            items = getattr(self, name)
            if not isinstance(items, tuple):
                raise ValueError(f"{name}必须是元组")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(item, CandidateSummary) for item in self.candidates
        ):
            raise ValueError("候选摘要必须是 CandidateSummary 元组")
        for name in ("pending_plan",):
            items = getattr(self, name)
            if not isinstance(items, tuple) or not all(
                isinstance(item, str) and item for item in items
            ):
                raise ValueError(f"{name}必须是非空字符串元组")
        if self.adjudication is not None and (
            not isinstance(self.adjudication, str) or not self.adjudication
        ):
            raise ValueError("裁决摘要必须是非空字符串或 None")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("错误信息必须是非空字符串或 None")

    def to_json(self) -> str:
        """可检索的 JSON 快照；字段与引擎状态一一对应。"""

        def event_dict(event: ScheduledEvent) -> dict[str, object]:
            return {
                "event_id": event.event_id,
                "simulation_time_ms": event.simulation_time,
                "kind": event.kind.value,
                "description": event.description,
                "action_id": event.action_id,
            }

        return json.dumps(
            {
                "stage": self.stage,
                "cycle_id": self.cycle_id,
                "simulation_time_ms": self.simulation_time,
                "world_state_version": self.world_state_version,
                "error": self.error,
                "active_batch": [event_dict(event) for event in self.active_batch],
                "pending_events": [event_dict(event) for event in self.pending_events],
                "action_records": [
                    {
                        "action_id": record.action_id,
                        "role_id": record.role_id,
                        "phase": record.phase.value,
                        "description": record.description,
                        "start_ms": record.start,
                        "expected_end_ms": record.expected_end,
                        "checkpoints_ms": list(record.checkpoints),
                        "registered_cycle": record.registered_cycle,
                        "commit_group_id": record.commit_group_id,
                    }
                    for record in self.action_records
                ],
                "pending_candidates": [
                    {
                        "role_id": item.role_id,
                        "response_id": item.response_id,
                        "actions": list(item.actions),
                        "schedules": list(item.schedules),
                    }
                    for item in self.candidates
                ],
                "pending_plan": list(self.pending_plan),
                "adjudication": self.adjudication,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    """一次运行的结束结果与原因。"""

    reason: RunStopReason
    final_time: SimulationTime
    completed_cycles: int
    error: str | None = None
    snapshot: EngineSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, RunStopReason):
            raise ValueError("停止原因必须是 RunStopReason")
        validate_simulation_time(self.final_time, "剧情时刻")
        if isinstance(self.completed_cycles, bool) or not isinstance(self.completed_cycles, int):
            raise ValueError("完成周期数必须是整数")
        if self.completed_cycles < 0:
            raise ValueError("完成周期数必须是非负整数")
        if self.reason is RunStopReason.ERROR and not self.error:
            raise ValueError("运行错误必须携带错误信息")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("错误信息必须是非空字符串或 None")


@dataclass(frozen=True, slots=True)
class _QueuePlan:
    """一次统一登记的队列候选变更；准备并校验后应用，不再会失败。"""

    removals: tuple[ActionId, ...]
    additions: tuple[ScheduledEvent, ...]


_ZERO_TIME = SimulationTime(0)


class Scheduler:
    """事件调度与行动生命周期的运行编排。

    一次 run 从起始时刻推进到自然结束、时限、用户中止或运行错误；
    剧情时钟只在周期开始跳转到事件时刻（以及停止前推进到时限），
    其余时间冻结。单个实例不允许并发运行两次；运行错误的实例禁止
    再次推进——队列中可能仍有已取出的未完成事件，恢复须重建实例并
    外部明确处理现场。错误不盲目重放角色调用，也不宣称回滚角色私有
    状态或已提交的世界事实；未生效候选保留在快照里。
    """

    def __init__(
        self,
        *,
        roles: Iterable[Role],
        world: World,
        translator: Translator,
        config: RunConfig,
        start_time: SimulationTime = _ZERO_TIME,
        initial_events: Iterable[WorldEvent] = (),
        tracer: TurnTracer | None = None,
        gate: RunGate | None = None,
        max_rejections: int = 3,
        max_translation_rejections: int = 3,
        should_end: Callable[[RunStatus], bool] | None = None,
        snapshot_path: Path | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise ValueError("运行配置必须是 RunConfig")
        validate_simulation_time(start_time, "起始剧情时刻")
        self._roles: dict[RoleId, Role] = {}
        for role in roles:
            if role.role_id in self._roles:
                raise ValueError(f"重复角色身份：{role.role_id}")
            self._roles[role.role_id] = role
        if not self._roles:
            raise ValueError("场景必须至少包含一个角色")
        self._world = world
        self._translator = translator
        self._config = config
        self._start = start_time
        self._time = start_time
        self._queue = EventQueue()
        # 外部事件身份（初始事件与 World 后续事件）终生保留，引擎生成
        # 内部身份时避让；外部重复身份明确报错。
        self._reserved_ids: set[EventId] = set()
        for event in initial_events:
            if not isinstance(event, WorldEvent):
                raise ValueError("初始事件必须是 WorldEvent")
            if event.simulation_time < start_time:
                raise ValueError(f"初始事件不得早于起始时刻：{event.event_id}")
            if event.event_id in self._reserved_ids:
                raise ValueError(f"初始事件身份重复：{event.event_id}")
            self._reserved_ids.add(event.event_id)
            self._queue.schedule(
                ScheduledEvent(
                    event_id=event.event_id,
                    simulation_time=event.simulation_time,
                    kind=ScheduledEventKind.WORLD,
                    description=event.description,
                )
            )
        self._tracer = tracer
        self._gate = gate
        self._should_end = should_end
        self._snapshot_path = snapshot_path
        self._max_translation_rejections = _require_positive_int(
            max_translation_rejections, "翻译打回次数上限"
        )
        self._registry = ActionRegistry()
        self._runner = ResponseRunner(
            self._roles.values(),
            world,
            tracer=tracer,
            max_rejections=max_rejections,
            arrangements_for=lambda role_id: self._registry.arrangements_for(role_id),
        )
        self._cycle_seq = 0
        self._completed_cycles = 0
        self._introspection_seq = 0
        self._stage = "initialized"
        self._cycle_id: CycleId | None = None
        self._active_batch: tuple[ScheduledEvent, ...] = ()
        # 周期是否打开（事件批已取出且未完成）：中途取消的实例禁止
        # 盲目恢复；门控等待处的取消不涉及已取批，可再次运行。
        self._cycle_open = False
        self._candidates: dict[RoleId, FinalizedDeclaration] = {}
        self._pending_plan: RegistrationPlan | None = None
        self._current_candidate: AdjudicationCandidate | None = None
        self._feedback_pending: set[ActionId] = set()
        self._outcome_details: dict[ActionId, str] = {}
        self._last_report: CycleReport | None = None
        self._running = False
        self._failed = False
        self._schedule_next_introspection(start_time)

    # ---- 只读观察接口（测试与调试使用） ----

    @property
    def current_time(self) -> SimulationTime:
        return self._time

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def last_report(self) -> CycleReport | None:
        return self._last_report

    def pending_events(self) -> tuple[ScheduledEvent, ...]:
        return self._queue.snapshot()

    def action_records(self) -> Mapping[ActionId, ActionRecord]:
        return self._registry.records

    # ---- 运行入口 ----

    async def run(self) -> RunResult:
        """推进到自然结束、时限、用户中止或运行错误。

        许可在每个完整周期前的边界取得：初始化也是安全边界，首次
        事件处理前先等待许可；暂停/停止在当前周期完成后生效，不打断
        World 提交。运行错误的实例再次调用直接拒绝。
        """
        if self._running:
            raise RuntimeError("上一次运行尚未结束")
        if self._failed:
            raise RuntimeError(
                "运行错误或周期中途取消的实例不可再次推进：可能存在已取出未完成的事件；"
                "恢复须重建调度器并外部明确处理现场"
            )
        self._running = True
        try:
            self._trace(
                "engine.run_start",
                _RUN_CYCLE,
                self._time,
                "运行开始",
                details={"max_duration_ms": self._config.max_duration},
            )
            while True:
                try:
                    result = await self._run_step()
                except asyncio.CancelledError:
                    if self._cycle_open:
                        # 事件批已取出未完成：保留现场并禁止盲目恢复。
                        self._failed = True
                        self._safe_trace(
                            "engine.cancelled",
                            self._cycle_id or _RUN_CYCLE,
                            self._time,
                            "周期中途被取消，现场保留，实例禁止再次推进",
                        )
                    else:
                        self._safe_trace(
                            "engine.cancelled",
                            self._cycle_id or _RUN_CYCLE,
                            self._time,
                            "许可等待处被取消，未取出事件批",
                        )
                    raise
                except Exception as error:
                    return self._error_result(error)
                if result is not None:
                    return result
        finally:
            self._running = False

    async def _run_step(self) -> RunResult | None:
        """执行一次外层推进；返回 None 表示继续下一轮。"""
        limit = self._config.limit_from(self._start)
        # 停止与许可在每个完整周期前取得；初始化也是安全边界。
        if self._gate is not None:
            if self._gate.should_stop():
                return self._stop(RunStopReason.ABORTED)
            self._stage = "run.gate_wait"
            permitted = await self._gate.wait_for_permission()
            if not permitted:
                return self._stop(RunStopReason.ABORTED)
        next_time = self._queue.next_time()
        if next_time is None or next_time > limit:
            # 没有更近的事件：按最后流逝的剧情时间更新实际进展后停在
            # 时限；未发生的事件与行动保留，不伪造结算，也不调用角色。
            await self._advance_to_limit(SimulationTime(limit))
            return self._stop(RunStopReason.TIME_LIMIT)
        report = await self._run_cycle(next_time)
        self._last_report = report
        self._completed_cycles += 1
        # 仅追踪：完整周期边界（翻译与统一登记已生效）。
        # ResponseRunner 的 cycle.complete 只是角色屏障完成，不能替代本边界。
        self._trace(
            "engine.cycle_complete",
            report.cycle_id,
            report.simulation_time,
            "完整周期完成",
            {
                "completed_cycles": self._completed_cycles,
                "outcomes": len(report.action_outcomes),
                "registry_actions": len(report.registry_actions),
            },
            # 同一行动身份可能在同周期多条结算里出现；追踪身份去重，
            # 完整结算明细仍在 outputs。
            action_ids=tuple(
                sorted({outcome.action_id for outcome in report.action_outcomes}, key=str)
            ),
            outputs=tuple(
                f"{outcome.action_id}:{outcome.result.value}:{outcome.detail}"
                for outcome in report.action_outcomes
            ),
            world_state_version=self._world.state_version,
        )
        status = RunStatus(self._time, self._completed_cycles, report)
        if self._should_end is not None and self._should_end(status):
            return self._stop(RunStopReason.NATURAL_END)
        if self._time >= limit:
            # 边界时刻的完整周期已结束，到此停止；同刻新登记的事件保留。
            return self._stop(RunStopReason.TIME_LIMIT)
        return None

    async def _advance_to_limit(self, limit: SimulationTime) -> None:
        """推进到时限边界：只更新实际进展，不处理事件、不调用角色。"""
        if limit <= self._time:
            return
        boundary = CycleId(f"cycle-{self._cycle_seq + 1}-limit")
        self._stage = "run.limit_progress"
        self._trace(
            "engine.limit_boundary",
            boundary,
            limit,
            "下一事件越界，推进到时限并更新实际进展",
            {"elapsed_ms": limit - self._time},
        )
        await self._run_progress(boundary, limit)
        self._time = limit

    # ---- 单个完整事件周期 ----

    async def _run_cycle(self, event_time: SimulationTime) -> CycleReport:
        self._cycle_seq += 1
        cycle_id = CycleId(f"cycle-{self._cycle_seq}")
        self._cycle_id = cycle_id
        self._time = event_time
        batch = self._queue.take_batch(event_time)
        self._active_batch = batch
        self._cycle_open = True
        # 新周期清空上一轮的未生效候选与计划；错误现场只保留本轮内容。
        self._candidates = {}
        self._pending_plan = None
        self._current_candidate = None
        self._stage = "cycle.start"
        introspection = any(e.kind is ScheduledEventKind.INTROSPECTION for e in batch)
        if introspection:
            self._schedule_next_introspection(event_time)
        world_events = tuple(
            WorldEvent(e.event_id, e.simulation_time, e.description)
            for e in batch
            if e.kind is ScheduledEventKind.WORLD
        )
        self._trace(
            "engine.cycle_start",
            cycle_id,
            event_time,
            f"进入事件时刻，处理 {len(batch)} 个事件",
            {"events": len(batch), "introspection": introspection},
            event_ids=tuple(e.event_id for e in batch),
        )
        # 1. 按已流逝剧情时间更新进展；此后直到周期结束不再更新。
        self._stage = "cycle.progress"
        await self._run_progress(cycle_id, event_time)
        # 2. 同刻整批事件的联合裁决：候选、校验、原子提交、发布。
        nodes = self._action_nodes_for(batch)
        commits: list[AdjudicationCommit] = []
        deliveries: tuple[Delivery, ...] = ()
        if world_events or nodes:
            self._stage = "cycle.adjudication"
            commit = await self._settle(
                cycle_id,
                event_time,
                events=world_events,
                action_nodes=nodes,
                purpose="同刻事件联合裁决",
            )
            commits.append(commit)
            deliveries = self._publish(
                cycle_id,
                event_time,
                commit,
                world_events=tuple(event.event_id for event in world_events),
                node_event_ids=tuple(node.event_id for node in nodes),
            )
        base_version = self._world.state_version
        # 3. 全部角色的输入与响应屏障。
        self._stage = "cycle.responses"
        inputs = self._build_turn_inputs(cycle_id, event_time, deliveries, introspection)
        outcome = await self._runner.run_cycle(inputs)
        # 4. 屏障之后的翻译与时间校验；失败经 rejected 打回，同周期重试。
        self._stage = "cycle.translation"
        finalized = await self._translate_all(cycle_id, event_time, outcome, base_version)
        # 翻译期间共同状态必须冻结：任何模块在等待中改动世界都不成立。
        if self._world.state_version != base_version:
            raise RuntimeError("翻译期间共同世界状态被修改，本轮前提不再成立")
        # 5. 统一登记：完整准备并校验登记计划与队列候选，再提交。
        self._stage = "cycle.registration"
        plan = prepare_registration(cycle_id, event_time, self._registry.records, finalized)
        self._pending_plan = plan
        self._trace(
            "registration.plan",
            cycle_id,
            event_time,
            "统一登记计划已准备" if plan.ok else "统一登记计划未通过校验",
            {
                "registered": len(plan.registered),
                "modified": len(plan.modified),
                "maintained": len(plan.maintained),
                "withdrawn_pending": len(plan.withdrawn_pending),
                "withdrawn_active": len(plan.withdrawn_active),
                "problems": len(plan.problems),
            },
            action_ids=tuple(
                sorted(
                    (
                        *(r.action_id for r in plan.registered),
                        *(r.action_id for r in plan.modified),
                        *(r.action_id for r in plan.withdrawn_pending),
                        *(r.action_id for r in plan.withdrawn_active),
                    ),
                    key=str,
                )
            ),
            outputs=tuple(
                f"{record.action_id}: {record.start}→{record.expected_end}ms"
                f"（节点 {list(record.checkpoints)}）"
                for record in (*plan.registered, *plan.modified)
            ),
        )
        if not plan.ok:
            raise RuntimeError("统一登记校验失败：" + "；".join(plan.problems))
        queue_plan = self._prepare_queue_plan(event_time, plan)
        # 6. 修改与撤回的客观效果经 World 裁决提交；此时引擎侧尚未发布
        #    任何登记或队列变更，World 提交失败不留下混合状态。
        if plan.modified or plan.withdrawn_active:
            self._stage = "cycle.registration_effects"
            modification_nodes = tuple(
                self._modification_node(record, finalized[record.role_id], cycle_id)
                for record in plan.modified
            )
            withdrawal_nodes = tuple(
                AdjudicationActionNode(
                    event_id=EventId(f"{record.action_id}-withdrawn"),
                    action_id=record.action_id,
                    role_id=record.role_id,
                    node=ActionNodeKind.WITHDRAWN,
                    description=record.description,
                    start=record.start,
                    expected_end=record.expected_end,
                )
                for record in plan.withdrawn_active
            )
            effects_commit = await self._settle(
                cycle_id,
                event_time,
                events=(),
                action_nodes=withdrawal_nodes,
                modifications=modification_nodes,
                purpose="修改与撤回主动维持的客观效果裁决",
                pending_addition_ids={e.event_id for e in queue_plan.additions},
            )
            commits.append(effects_commit)
            for outcome_item in effects_commit.action_outcomes:
                self._feedback_pending.add(outcome_item.action_id)
                self._outcome_details[outcome_item.action_id] = outcome_item.detail
        # 7. 应用登记与队列：全部内容已预检，此处不再会失败。
        self._registry = self._registry.apply(plan)
        self._apply_queue_plan(queue_plan)
        self._trace(
            "registration.commit",
            cycle_id,
            event_time,
            "统一登记生效，新声明替换此前已接受的声明",
            {
                "registered": len(plan.registered),
                "modified": len(plan.modified),
                "withdrawn": len(plan.withdrawn_pending) + len(plan.withdrawn_active),
            },
            action_ids=tuple(
                sorted(
                    (
                        *(r.action_id for r in plan.registered),
                        *(r.action_id for r in plan.modified),
                        *(r.action_id for r in plan.withdrawn_active),
                        *(r.action_id for r in plan.withdrawn_pending),
                    ),
                    key=str,
                )
            ),
        )
        self._stage = "cycle.complete"
        report = self._build_report(
            cycle_id, event_time, batch, introspection, commits, deliveries, inputs, plan
        )
        self._pending_plan = None
        self._cycle_open = False
        self._active_batch = ()
        return report

    # ---- 周期内部步骤 ----

    async def _run_progress(self, cycle_id: CycleId, simulation_time: SimulationTime) -> None:
        """进展更新入口：实际更新必须原子应用并递增共同状态版本。"""
        before = self._world.state_version
        progress = await self._world.update_progress(cycle_id, simulation_time)
        if not isinstance(progress, ProgressUpdate):
            raise RuntimeError(
                f"World.update_progress 必须返回 ProgressUpdate，得到：{type(progress).__name__}"
            )
        if progress.cycle_id != cycle_id or progress.simulation_time != simulation_time:
            raise RuntimeError("进展更新必须关联当前周期与剧情时刻")
        if self._world.state_version < before:
            raise RuntimeError("进展更新不得使共同世界状态版本回退")
        if progress.updated_action_ids and self._world.state_version == before:
            raise RuntimeError("进展实际更新必须递增共同世界状态版本")
        self._trace(
            "world.progress",
            cycle_id,
            simulation_time,
            "World 按流逝剧情时间更新进展",
            {"updated": len(progress.updated_action_ids)},
            action_ids=progress.updated_action_ids,
            outputs=progress.notes,
        )

    def _action_nodes_for(
        self, batch: tuple[ScheduledEvent, ...]
    ) -> tuple[AdjudicationActionNode, ...]:
        nodes: list[AdjudicationActionNode] = []
        for event in batch:
            if event.kind not in _NODE_KINDS:
                continue
            assert event.action_id is not None  # 结构校验保证行动事件带身份
            record = self._registry.records.get(event.action_id)
            if record is None:
                raise RuntimeError(f"行动节点引用未登记的行动：{event.action_id}")
            if record.phase in TERMINAL_PHASES:
                raise RuntimeError(f"已结束行动仍有待处理节点：{event.action_id}")
            nodes.append(
                AdjudicationActionNode(
                    event_id=event.event_id,
                    action_id=record.action_id,
                    role_id=record.role_id,
                    node=_NODE_KINDS[event.kind],
                    description=record.description,
                    start=record.start,
                    expected_end=record.expected_end,
                )
            )
        return tuple(sorted(nodes, key=lambda item: str(item.action_id)))

    def _modification_node(
        self,
        record: ActionRecord,
        declaration: FinalizedDeclaration,
        cycle_id: CycleId,
    ) -> ActionModificationNode:
        """把一次明确修改转换为交给 World 的候选输入（含新旧安排）。"""
        response = declaration.response
        change = next(
            (
                item.change
                for item in response.actions
                if isinstance(item, ModifiedAction) and item.action_id == record.action_id
            ),
            None,
        )
        if change is None:  # pragma: no cover - prepare_registration 已保证存在
            raise RuntimeError(f"修改声明缺少变化说明：{record.action_id}")
        old = next(
            (
                arrangement
                for arrangement in self._registry.arrangements_for(record.role_id)
                if arrangement.action_id == record.action_id
            ),
            None,
        )
        if old is None:  # pragma: no cover - prepare_registration 已保证存在
            raise RuntimeError(f"修改声明缺少既有安排：{record.action_id}")
        return ActionModificationNode(
            event_id=EventId(f"{record.action_id}-modified-{cycle_id}"),
            action_id=record.action_id,
            role_id=record.role_id,
            description=old.description,
            change=change,
            old_start=old.start,
            old_expected_end=old.expected_end,
            old_checkpoints=old.checkpoints,
            new_start=record.start,
            new_expected_end=record.expected_end,
            new_checkpoints=record.checkpoints,
        )

    async def _settle(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        events: tuple[WorldEvent, ...],
        action_nodes: tuple[AdjudicationActionNode, ...],
        purpose: str,
        modifications: tuple[ActionModificationNode, ...] = (),
        pending_addition_ids: set[EventId] | None = None,
    ) -> AdjudicationCommit:
        """裁决候选 → 运行时校验 → 原子提交的完整边界。

        候选基于预裁决版本：adjudicate 不得改变共同状态；校验关联
        裁决前捕获的版本；提交结果版本必须严格递增且等于 World 当前
        实际版本。后续事件身份与结算结果都在事实提交前结构预检，
        提交之后只剩不会失败的发布步骤。
        """
        pre_version = self._world.state_version
        event_ids = tuple(sorted((event.event_id for event in events), key=str))
        action_ids = tuple(
            sorted(
                {node.action_id for node in action_nodes}
                | {node.action_id for node in modifications},
                key=str,
            )
        )
        candidate = await self._world.adjudicate(
            cycle_id,
            simulation_time,
            event_ids=event_ids,
            action_ids=action_ids,
            events=events,
            action_nodes=action_nodes,
            modifications=modifications,
        )
        if not isinstance(candidate, AdjudicationCandidate):
            raise RuntimeError(
                f"World.adjudicate 必须返回 AdjudicationCandidate，得到：{type(candidate).__name__}"
            )
        if candidate.cycle_id != cycle_id or candidate.simulation_time != simulation_time:
            raise RuntimeError("裁决候选必须关联当前周期与剧情时刻")
        if set(candidate.event_ids) != set(event_ids):
            raise RuntimeError("裁决候选必须只引用本次交给 World 的事件")
        if not set(candidate.action_ids) <= set(action_ids):
            raise RuntimeError("裁决候选引用了未参与本次裁决的行动")
        if self._world.state_version != pre_version:
            raise RuntimeError("World.adjudicate 不得改变共同状态（裁决基于候选生成前的版本）")
        self._current_candidate = candidate
        self._trace(
            "world.adjudicate",
            cycle_id,
            simulation_time,
            purpose,
            {
                "changes": len(candidate.proposed_changes),
                "outcomes": len(candidate.action_outcomes),
                "follow_ups": len(candidate.follow_up_events),
            },
            event_ids=event_ids,
            action_ids=candidate.action_ids,
            inputs=tuple(event.description for event in events)
            + tuple(f"{node.node.value}:{node.description}" for node in action_nodes)
            + tuple(
                f"修改 {node.action_id}（{node.role_id}）：{node.change}" for node in modifications
            ),
            outputs=tuple(item.description for item in candidate.proposed_changes),
        )
        validation = self._world.validate_adjudication(candidate)
        if not isinstance(validation, AdjudicationValidation):
            raise RuntimeError(
                "World.validate_adjudication 必须返回 AdjudicationValidation，得到："
                f"{type(validation).__name__}"
            )
        if validation.candidate != candidate:
            raise RuntimeError("校验结果必须关联被提交的候选")
        if validation.world_state_version != pre_version:
            raise RuntimeError("校验依据的世界状态版本与裁决前捕获的版本不一致")
        if not validation.passed:
            raise RuntimeError("同刻裁决候选未通过运行时校验：" + "；".join(validation.problems))
        if self._world.state_version != pre_version:
            raise RuntimeError("World.validate_adjudication 不得改变共同状态")
        self._trace(
            "world.validate",
            cycle_id,
            simulation_time,
            "裁决候选通过运行时校验",
            {"problems": len(validation.problems)},
            world_state_version=validation.world_state_version,
        )
        # 后续事件身份在提交前预检：不得与现存队列、外部历史身份或
        # 本次待登记的内部事件冲突。
        live_ids = {event.event_id for event in self._queue.snapshot()}
        for follow_up in candidate.follow_up_events:
            if (
                follow_up.event_id in live_ids
                or follow_up.event_id in self._reserved_ids
                or (pending_addition_ids and follow_up.event_id in pending_addition_ids)
            ):
                raise RuntimeError(f"后续世界事件身份冲突：{follow_up.event_id}")
        # 结算结果在提交前预检：未登记、已结束或与节点类型不符都在
        # 事实提交前发现。允许结果归口在 domain/adjudication.py。
        allowed = allowed_results_by_action(action_nodes, modifications)
        outcome_problems = outcome_transition_problems(
            self._registry.records, candidate.action_outcomes, allowed
        )
        if outcome_problems:
            raise RuntimeError("结算结果未通过提交前校验：" + "；".join(outcome_problems))
        commit = self._world.commit_adjudication(candidate)
        if not isinstance(commit, AdjudicationCommit):
            raise RuntimeError(
                f"World.commit_adjudication 必须返回 AdjudicationCommit，得到：{type(commit).__name__}"
            )
        if commit.candidate != candidate:
            raise RuntimeError("提交结果必须关联被校验的候选")
        if commit.resulting_version <= pre_version:
            raise RuntimeError("事实提交必须严格递增共同世界状态版本")
        if self._world.state_version != commit.resulting_version:
            raise RuntimeError("提交报告的版本必须等于 World 当前实际状态版本")
        self._trace(
            "world.commit",
            cycle_id,
            simulation_time,
            "事实已原子提交",
            {"changes": len(commit.committed_changes)},
            outputs=tuple(item.description for item in commit.committed_changes),
            world_state_version=commit.resulting_version,
        )
        for follow_up in candidate.follow_up_events:
            entry = ScheduledEvent(
                event_id=follow_up.event_id,
                simulation_time=follow_up.simulation_time,
                kind=ScheduledEventKind.WORLD,
                description=follow_up.description,
            )
            self._reserved_ids.add(follow_up.event_id)
            self._queue.schedule(entry)
            self._trace(
                "engine.follow_up_scheduled",
                cycle_id,
                simulation_time,
                f"登记后续世界事件 {entry.event_id}",
                {"time": entry.simulation_time},
                event_ids=(entry.event_id,),
            )
        self._current_candidate = None
        return commit

    def _publish(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        commit: AdjudicationCommit,
        *,
        world_events: tuple[EventId, ...],
        node_event_ids: tuple[EventId, ...],
    ) -> tuple[Delivery, ...]:
        """提交后的感知发布；投递必须引用本轮事件与当前时刻。"""
        publish_version = self._world.state_version
        deliveries = self._world.publish(commit)
        if not isinstance(deliveries, tuple) or not all(
            isinstance(item, Delivery) for item in deliveries
        ):
            raise RuntimeError("World.publish 必须返回 Delivery 元组")
        if self._world.state_version != publish_version:
            raise RuntimeError("World.publish 是只读边界，不得改变共同状态")
        known_ids = set(world_events) | set(node_event_ids)
        for item in deliveries:
            if item.role_id not in self._roles:
                raise RuntimeError(f"投递指向未知角色：{item.role_id}")
            if item.observation.simulation_time != simulation_time:
                raise RuntimeError("投递的感知必须属于当前剧情时刻")
            if item.observation.event_id not in known_ids:
                raise RuntimeError(f"投递引用了本轮之外的事件：{item.observation.event_id}")
        # 行动结算结果进入登记阶段与角色反馈；终态行动移除剩余节点。
        self._registry, changed = self._registry.apply_outcomes(commit.action_outcomes)
        for record in changed:
            self._feedback_pending.add(record.action_id)
            detail = next(
                (
                    item.detail
                    for item in commit.action_outcomes
                    if item.action_id == record.action_id
                ),
                "",
            )
            self._outcome_details[record.action_id] = detail
            if record.phase in TERMINAL_PHASES:
                self._queue.remove_for_action(record.action_id)
        self._trace(
            "world.publish",
            cycle_id,
            simulation_time,
            "已按角色范围发布感知",
            {"deliveries": len(deliveries)},
            event_ids=tuple(sorted({item.observation.event_id for item in deliveries}, key=str)),
            outputs=tuple(
                f"{item.role_id} 收到：{item.observation.content}" for item in deliveries
            ),
            world_state_version=commit.resulting_version,
        )
        return deliveries

    def _build_turn_inputs(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        deliveries: tuple[Delivery, ...],
        introspection: bool,
    ) -> dict[RoleId, TurnInput]:
        by_role: dict[RoleId, list[Delivery]] = {role_id: [] for role_id in self._roles}
        for item in deliveries:
            by_role[item.role_id].append(item)
        inputs: dict[RoleId, TurnInput] = {}
        for role_id in sorted(self._roles, key=str):
            own = by_role[role_id]
            observations = tuple(item.observation for item in own)
            trigger_ids = tuple(sorted({item.observation.event_id for item in own}, key=str))
            inputs[role_id] = TurnInput(
                cycle_id=cycle_id,
                role_id=role_id,
                simulation_time=simulation_time,
                trigger=TriggerReason(trigger_ids, introspection),
                observations=observations,
                known_action_states=self._known_states_for(role_id),
            )
        return inputs

    def _known_states_for(self, role_id: RoleId) -> tuple[KnownActionState, ...]:
        """角色可获知的自身行动状态；待反馈项投递一次后消费。"""
        states: list[KnownActionState] = []
        consumed: list[ActionId] = []
        for record in self._registry.for_role(role_id):
            feedback = record.action_id in self._feedback_pending
            if record.phase not in MAINTAINABLE_PHASES and not feedback:
                continue
            default = (
                f"已登记等待开始，预计 {record.start}–{record.expected_end} 毫秒"
                if record.phase is ActionPhase.REGISTERED
                else f"阶段 {record.phase.value}"
            )
            detail = self._outcome_details.get(record.action_id) or default
            states.append(KnownActionState(record.action_id, record.phase, detail))
            if feedback:
                consumed.append(record.action_id)
        for action_id in consumed:
            self._feedback_pending.discard(action_id)
            self._outcome_details.pop(action_id, None)
        return tuple(states)

    async def _translate_all(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        outcome: CycleOutcome,
        base_version: WorldStateVersion,
    ) -> dict[RoleId, FinalizedDeclaration]:
        """全部角色通过检查屏障之后依序翻译；先完成者不得抢先登记。"""
        used_ids = set(outcome.used_response_ids)
        tasks: dict[RoleId, asyncio.Task[FinalizedDeclaration]] = {}
        async with asyncio.TaskGroup() as group:
            for role_id in sorted(self._roles, key=str):
                tasks[role_id] = group.create_task(
                    self._finalize_declaration(
                        self._roles[role_id],
                        outcome.accepted[role_id],
                        cycle_id,
                        simulation_time,
                        used_ids,
                        base_version,
                    )
                )
        finalized = {role_id: tasks[role_id].result() for role_id in sorted(tasks, key=str)}
        self._candidates = finalized
        return finalized

    async def _finalize_declaration(
        self,
        role: Role,
        accepted: AcceptedResponse,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        used_ids: set[ResponseId],
        base_version: WorldStateVersion,
    ) -> FinalizedDeclaration:
        """单个角色的翻译与登记校验循环；打回不重投原感知。

        只有维持或空声明的轮次不调用翻译器（维持不重新翻译），直接
        生成关联同一响应的空时间安排并进入统一登记。结构不一致的
        翻译结果是运行错误；只有合法完整翻译中的时间安排不合业务
        约束才走打回。
        """
        role_id = role.role_id
        response = accepted.response

        def _finalize(
            current: RoleResponse, translation: TranslationResult
        ) -> FinalizedDeclaration:
            finalized = FinalizedDeclaration(response=current, translation=translation)
            self._candidates = {**self._candidates, role_id: finalized}
            return finalized

        def _skip_translation(current: RoleResponse) -> FinalizedDeclaration | None:
            """每轮响应（含打回重试后的完整响应）都判断：仅维持或空声明
            不调用翻译器，直接合成关联同一响应的空时间安排。"""
            if any(isinstance(item, (NewAction, ModifiedAction)) for item in current.actions):
                return None
            self._trace(
                "translation.skipped",
                cycle_id,
                simulation_time,
                "仅维持或空声明，无需重新翻译",
                {"maintained": len(current.actions)},
                role_id=role_id,
                response_id=current.response_id,
                outputs=_action_texts(current),
                world_state_version=base_version,
            )
            return _finalize(
                current,
                TranslationResult(
                    cycle_id=current.cycle_id,
                    role_id=current.role_id,
                    response_id=current.response_id,
                ),
            )

        arrangements = self._registry.arrangements_for(role_id)
        attempts = 0
        while True:
            skipped = _skip_translation(response)
            if skipped is not None:
                return skipped
            request = TranslationRequest(
                response=response,
                current_time=simulation_time,
                world_state_version=base_version,
                arrangements=arrangements,
                objective_context=self._world.translation_context(role_id),
            )
            self._trace(
                "translation.request",
                cycle_id,
                simulation_time,
                "翻译器确定时间安排",
                {
                    "new_actions": sum(isinstance(i, NewAction) for i in response.actions),
                    "modified": sum(isinstance(i, ModifiedAction) for i in response.actions),
                    "maintained": sum(isinstance(i, MaintainAction) for i in response.actions),
                },
                role_id=role_id,
                response_id=response.response_id,
                action_ids=tuple(i.action_id for i in response.actions),
                inputs=(
                    *(request.objective_context.conditions if request.objective_context else ()),
                    *(
                        f"既有 {item.action_id}（{item.phase.value}）"
                        f"{item.start}→{item.expected_end}ms"
                        for item in arrangements
                    ),
                ),
                outputs=_action_texts(response),
                world_state_version=base_version,
            )
            translation = await self._translator.translate(request)
            # 类型与身份错误属于运行错误，不得伪装成语义打回。
            if not isinstance(translation, TranslationResult):
                raise RuntimeError(
                    f"翻译器必须返回 TranslationResult，得到：{type(translation).__name__}"
                )
            if (
                translation.cycle_id,
                translation.role_id,
                translation.response_id,
            ) != (
                response.cycle_id,
                response.role_id,
                response.response_id,
            ):
                raise RuntimeError("翻译结果必须关联同一响应")
            issues = registration_issues(
                response,
                translation,
                current_time=simulation_time,
                arrangements=arrangements,
            )
            structural = tuple(issue.text for issue in issues if issue.structural)
            timing = tuple(issue.text for issue in issues if not issue.structural)
            self._trace(
                "translation.result",
                cycle_id,
                simulation_time,
                "翻译完成"
                if not issues
                else f"时间安排未通过登记校验（结构 {len(structural)} 项，约束 {len(timing)} 项）",
                {"structural": len(structural), "timing": len(timing)},
                role_id=role_id,
                response_id=response.response_id,
                outputs=_schedule_texts(translation),
                world_state_version=base_version,
                error=None if not issues else "；".join(issue.text for issue in issues),
            )
            if structural:
                raise RuntimeError("翻译结构不一致：" + "；".join(structural))
            if not timing:
                return _finalize(response, translation)
            # 时间安排不合业务约束：打回角色重新决定，仍属同一周期。
            problems: tuple[str, ...] = timing
            while problems:
                if attempts >= self._max_translation_rejections:
                    raise RuntimeError(
                        f"角色 {role_id} 在周期 {cycle_id} 的翻译打回次数"
                        f"超过上限 {self._max_translation_rejections}，停止推进"
                    )
                attempts += 1
                rejection = Rejection(
                    cycle_id=cycle_id,
                    role_id=role_id,
                    rejected_response_id=response.response_id,
                    reason="；".join(problems),
                    action_ids=tuple(sorted({i.action_id for i in response.actions}, key=str)),
                )
                self._trace(
                    "registration.rejected",
                    cycle_id,
                    simulation_time,
                    "时间安排问题打回角色重新决定",
                    {"reason": rejection.reason, "attempts": attempts},
                    role_id=role_id,
                    response_id=response.response_id,
                    action_ids=rejection.action_ids,
                )
                previous_id = response.response_id
                response = await role.rejected(rejection)
                self._require_valid_retry(response, role_id, cycle_id, used_ids)
                self._trace(
                    "role.response",
                    cycle_id,
                    simulation_time,
                    "角色返回新的完整声明",
                    _response_details(response),
                    role_id=role_id,
                    response_id=response.response_id,
                    action_ids=tuple(i.action_id for i in response.actions),
                    related_response_id=previous_id,
                    outputs=_action_texts(response),
                )
                check = await self._world.check_response(response, arrangements=arrangements)
                self._require_valid_check(check, response, base_version)
                check_details: dict[str, TraceValue] = {
                    "passed": check.passed,
                    "arrangements": len(arrangements),
                }
                if check.reason:
                    check_details["reason"] = check.reason
                self._trace(
                    "world.check",
                    cycle_id,
                    simulation_time,
                    f"World 检查{'通过' if check.passed else '未通过'}",
                    check_details,
                    role_id=role_id,
                    response_id=response.response_id,
                    action_ids=check.action_ids,
                    outputs=_action_texts(response),
                    world_state_version=check.world_state_version,
                )
                if check.passed:
                    break
                problems = (check.reason or "客观条件不允许",)
            arrangements = self._registry.arrangements_for(role_id)

    def _require_valid_retry(
        self,
        response: object,
        role_id: RoleId,
        cycle_id: CycleId,
        used_ids: set[ResponseId],
    ) -> None:
        if not isinstance(response, RoleResponse):
            raise ValueError(
                f"角色 {role_id} 必须返回 RoleResponse，得到：{type(response).__name__}"
            )
        if response.cycle_id != cycle_id or response.role_id != role_id:
            raise ValueError(f"打回后的响应身份不一致：{response.response_id}")
        if response.response_id in used_ids:
            raise RuntimeError(f"响应身份在当前周期已被使用，不得重用：{response.response_id}")
        used_ids.add(response.response_id)

    def _require_valid_check(
        self, check: object, response: RoleResponse, base_version: WorldStateVersion
    ) -> None:
        if not isinstance(check, ActionCheckResult):
            raise ValueError(f"World 检查必须返回 ActionCheckResult，得到：{type(check).__name__}")
        if (
            check.cycle_id,
            check.role_id,
            check.response_id,
        ) != (
            response.cycle_id,
            response.role_id,
            response.response_id,
        ):
            raise ValueError(f"检查结果与被检查响应不一致：{check.response_id}")
        if check.world_state_version != base_version:
            raise ValueError(f"检查依据的共同世界状态与本周期固定版本不一致：{check.response_id}")
        declared = {item.action_id for item in response.actions}
        for action_id in check.action_ids:
            if action_id not in declared:
                raise ValueError(f"检查结果引用了声明外的行动：{action_id}")

    # ---- 队列候选计划 ----

    def _free_event_id(self, base: str) -> EventId:
        """生成内部事件身份时避让现存队列与外部历史身份。"""
        live = {str(event.event_id) for event in self._queue.snapshot()}
        candidate = base
        suffix = 2
        while candidate in live or candidate in {str(item) for item in self._reserved_ids}:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return EventId(candidate)

    def _action_node_events(
        self, record: ActionRecord, *, only_after: SimulationTime
    ) -> tuple[ScheduledEvent, ...]:
        """为一个行动生成开始/过程/结束节点；只登记晚于给定时刻的节点。"""
        entries: list[tuple[SimulationTime, str, ScheduledEventKind]] = []
        if record.start > only_after:
            entries.append(
                (record.start, f"{record.action_id}-start", ScheduledEventKind.ACTION_START)
            )
        for index, checkpoint in enumerate(record.checkpoints, start=1):
            if checkpoint > only_after:
                entries.append(
                    (
                        checkpoint,
                        f"{record.action_id}-cp-{index}",
                        ScheduledEventKind.ACTION_CHECKPOINT,
                    )
                )
        if record.expected_end > only_after:
            entries.append(
                (record.expected_end, f"{record.action_id}-end", ScheduledEventKind.ACTION_END)
            )
        return tuple(
            ScheduledEvent(
                event_id=self._free_event_id(base),
                simulation_time=time,
                kind=kind,
                description=record.description,
                action_id=record.action_id,
            )
            for time, base, kind in entries
        )

    def _prepare_queue_plan(
        self, current_time: SimulationTime, plan: RegistrationPlan
    ) -> _QueuePlan:
        """准备统一登记的队列候选变更；预检通过后应用不再会失败。"""
        removals = tuple(
            sorted(
                {
                    *(record.action_id for record in plan.withdrawn_pending),
                    *(record.action_id for record in plan.withdrawn_active),
                    *(record.action_id for record in plan.modified),
                },
                key=str,
            )
        )
        additions: list[ScheduledEvent] = []
        for record in plan.registered:
            additions.extend(self._action_node_events(record, only_after=current_time))
        for record in plan.modified:
            additions.extend(self._action_node_events(record, only_after=current_time))
        live_ids = {event.event_id for event in self._queue.snapshot()}
        addition_ids = [event.event_id for event in additions]
        if len(set(addition_ids)) != len(addition_ids):
            raise RuntimeError("内部事件身份在登记计划内重复")
        for event_id in addition_ids:
            if event_id in live_ids:
                raise RuntimeError(f"内部事件身份与现存队列冲突：{event_id}")
        return _QueuePlan(removals=removals, additions=tuple(additions))

    def _apply_queue_plan(self, queue_plan: _QueuePlan) -> None:
        for action_id in queue_plan.removals:
            self._queue.remove_for_action(action_id)
        for event in queue_plan.additions:
            self._queue.schedule(event)

    def _schedule_next_introspection(self, after: SimulationTime) -> None:
        self._introspection_seq += 1
        self._queue.schedule(
            ScheduledEvent(
                event_id=self._free_event_id(f"introspection-{self._introspection_seq}"),
                simulation_time=SimulationTime(after + self._config.introspection_interval),
                kind=ScheduledEventKind.INTROSPECTION,
                description="周期性内思考",
            )
        )

    def _diagnostic_snapshot(self) -> WorldDiagnostic | None:
        """只读诊断快照；调用前后共同状态版本必须冻结。"""
        before = self._world.state_version
        diagnostic = self._world.diagnostic_snapshot()
        if self._world.state_version != before:
            raise RuntimeError("World.diagnostic_snapshot 是只读边界，不得改变共同状态")
        return diagnostic

    def _build_report(
        self,
        cycle_id: CycleId,
        event_time: SimulationTime,
        batch: tuple[ScheduledEvent, ...],
        introspection: bool,
        commits: list[AdjudicationCommit],
        deliveries: tuple[Delivery, ...],
        inputs: dict[RoleId, TurnInput],
        plan: RegistrationPlan,
    ) -> CycleReport:
        by_role: dict[RoleId, list[str]] = {role_id: [] for role_id in self._roles}
        for item in deliveries:
            by_role[item.role_id].append(item.observation.content)
        views: list[RoleCycleView] = []
        for role_id in sorted(self._roles, key=str):
            declaration = self._candidates.get(role_id)
            declarations = _action_texts(declaration.response) if declaration is not None else ()
            views.append(
                RoleCycleView(
                    role_id=role_id,
                    declarations=declarations,
                    perceptions=tuple(by_role[role_id]),
                    action_states=inputs[role_id].known_action_states,
                )
            )
        return CycleReport(
            cycle_id=cycle_id,
            step=self._cycle_seq,
            simulation_time=event_time,
            batch_events=tuple(
                CycleEvent(
                    event_id=event.event_id, kind=event.kind.value, description=event.description
                )
                for event in batch
            ),
            introspection=introspection,
            committed_facts=tuple(
                change.description for commit in commits for change in commit.committed_changes
            ),
            action_outcomes=tuple(
                outcome for commit in commits for outcome in commit.action_outcomes
            ),
            roles=tuple(views),
            registry_actions=tuple(
                sorted(self._registry.records.values(), key=lambda item: str(item.action_id))
            ),
            world_diagnostic=self._diagnostic_snapshot(),
            registered=plan.registered,
            modified=plan.modified,
            withdrawn=plan.withdrawn_pending + plan.withdrawn_active,
        )

    # ---- 停止与错误现场 ----

    def _stop(self, reason: RunStopReason) -> RunResult:
        self._stage = "run.stopped"
        self._trace(
            "engine.stop",
            _RUN_CYCLE,
            self._time,
            f"运行停止：{reason.value}",
            {"cycles": self._completed_cycles, "reason": reason.value},
        )
        return RunResult(
            reason=reason,
            final_time=self._time,
            completed_cycles=self._completed_cycles,
        )

    def _error_result(self, error: BaseException) -> RunResult:
        message = _describe_error(error)
        # 先保留出错阶段的现场（含取出批与未生效计划），再标记失败。
        snapshot = self.snapshot(error=message)
        self._stage = "run.error"
        self._failed = True
        if self._snapshot_path is not None:
            try:
                self._snapshot_path.write_text(snapshot.to_json(), encoding="utf-8")
            except OSError as write_error:
                message = f"{message}；快照写入失败：{write_error}"
        self._safe_trace(
            "engine.error",
            self._cycle_id or _RUN_CYCLE,
            self._time,
            "运行错误，停止推进",
            error=message,
        )
        return RunResult(
            reason=RunStopReason.ERROR,
            final_time=self._time,
            completed_cycles=self._completed_cycles,
            error=message,
            snapshot=snapshot,
        )

    def snapshot(self, *, error: str | None = None) -> EngineSnapshot:
        """导出当前引擎状态的可检索快照；不读取密钥或模型记录。"""
        plan = self._pending_plan
        plan_summary = (
            tuple(
                f"{record.action_id}→{record.phase.value}"
                for record in (
                    *plan.registered,
                    *plan.modified,
                    *plan.withdrawn_pending,
                    *plan.withdrawn_active,
                )
            )
            if plan is not None
            else ()
        )
        candidate = self._current_candidate
        adjudication = (
            "；".join(
                [
                    f"事件 {list(candidate.event_ids)}",
                    f"行动 {list(candidate.action_ids)}",
                    f"变更 {len(candidate.proposed_changes)}",
                    f"结算 {len(candidate.action_outcomes)}",
                ]
            )
            if candidate is not None
            else None
        )
        return EngineSnapshot(
            stage=self._stage,
            cycle_id=self._cycle_id,
            simulation_time=self._time,
            world_state_version=self._world.state_version,
            active_batch=self._active_batch,
            pending_events=self._queue.snapshot(),
            action_records=tuple(
                sorted(self._registry.records.values(), key=lambda item: str(item.action_id))
            ),
            candidates=tuple(
                CandidateSummary(
                    role_id=role_id,
                    response_id=item.response.response_id,
                    actions=_action_texts(item.response),
                    schedules=_schedule_texts(item.translation),
                )
                for role_id, item in sorted(self._candidates.items(), key=lambda kv: str(kv[0]))
            ),
            pending_plan=plan_summary,
            adjudication=adjudication,
            error=error,
        )

    def write_snapshot(self, path: Path) -> None:
        """把当前现场写入文件；用于运行错误后的离线检查。"""
        path.write_text(self.snapshot().to_json(), encoding="utf-8")

    # ---- 留痕 ----

    def _trace(
        self,
        stage: str,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        summary: str,
        details: Mapping[str, TraceValue] | None = None,
        *,
        role_id: RoleId | None = None,
        response_id: ResponseId | None = None,
        action_ids: tuple[ActionId, ...] = (),
        event_ids: tuple[EventId, ...] = (),
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
        related_response_id: ResponseId | None = None,
        world_state_version: WorldStateVersion | None = None,
        error: str | None = None,
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=cycle_id,
                simulation_time=simulation_time,
                summary=summary,
                role_id=role_id,
                response_id=response_id,
                related_response_id=related_response_id,
                action_ids=action_ids,
                event_ids=event_ids,
                inputs=inputs,
                outputs=outputs,
                world_state_version=world_state_version,
                details=details or {},
                error=error,
            )
        )

    def _safe_trace(
        self,
        stage: str,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        summary: str,
        *,
        error: str | None = None,
    ) -> None:
        if self._tracer is None:
            return
        try:
            self._tracer.trace(
                TraceRecord(
                    stage=stage,
                    cycle_id=cycle_id,
                    simulation_time=simulation_time,
                    summary=summary,
                    error=error,
                )
            )
        except Exception:
            pass


def _require_positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}必须是正整数，得到：{value!r}")
    if value < 1:
        raise ValueError(f"{field}必须是正整数，得到：{value!r}")
    return value


def _describe_error(error: BaseException) -> str:
    """展开异常组的实际原因；错误信息不得只剩数量描述。"""
    parts: list[str] = []

    def walk(item: BaseException) -> None:
        if isinstance(item, BaseExceptionGroup):
            for sub in item.exceptions:
                walk(sub)
            return
        parts.append(f"{type(item).__name__}: {item}")

    walk(error)
    if not parts:
        parts.append(f"{type(error).__name__}: {error}")
    return "；".join(parts)


def _action_texts(response: RoleResponse) -> tuple[str, ...]:
    texts: list[str] = []
    for item in response.actions:
        if isinstance(item, NewAction):
            texts.append(f"新增 {item.action_id}：{item.description}")
        elif isinstance(item, ModifiedAction):
            texts.append(f"修改 {item.action_id}：{item.change}")
        else:
            texts.append(f"维持 {item.action_id}")
    return tuple(texts)


def _schedule_texts(translation: TranslationResult) -> tuple[str, ...]:
    """登记安排的稳定文本；日志与快照共用。"""
    texts = [
        f"{item.action_id}: {item.start}→{item.expected_end}ms" for item in translation.schedules
    ]
    texts.extend(
        f"{item.action_id}（修改）: {item.start}→{item.expected_end}ms"
        for item in translation.modified_schedules
    )
    return tuple(texts)


def _response_details(response: RoleResponse) -> dict[str, TraceValue]:
    new_group: str | None = None
    for item in response.actions:
        if isinstance(item, NewAction):
            new_group = item.commit_group_id
            break
    return {
        "maintained": sum(isinstance(item, MaintainAction) for item in response.actions),
        "new_actions": sum(isinstance(item, NewAction) for item in response.actions),
        "modified": sum(isinstance(item, ModifiedAction) for item in response.actions),
        "commit_group": new_group,
    }
