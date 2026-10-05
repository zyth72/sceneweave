"""离线可运行演示：事件调度与行动生命周期的完整闭环。

不调用真实模型、不读取密钥：使用脚本角色、假语义世界与脚本翻译器
演示动作持续（位置按剧情时间推进）、感知投递、检查打回与运行时限
停止。真实模型驱动的 System1/2/3 与翻译器不在本演示内。

运行：

    uv run --locked python examples/offline_demo.py
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from concordia.utils import structured_logging

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
    Observation,
    Rejection,
    RoleResponse,
    TurnInput,
    WorldEvent,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.domain.translation import (
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationRequest,
    TranslationResult,
)
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.base import Role
from sceneweave.world.base import World

LIN = RoleId("lin_zhou")
SHEN = RoleId("shen_yan")


class DemoRole(Role):
    """脚本角色：按每轮脚本返回完整主动行为声明。

    脚本按轮次顺序消耗（含打回后的重新决定）；响应的周期身份由
    角色对齐到当前周期，脚本无需预知周期编号。
    """

    def __init__(self, role_id: RoleId, script: Sequence[RoleResponse]) -> None:
        self._id = role_id
        self._script = list(script)
        self.perceived: list[TurnInput] = []

    @property
    def role_id(self) -> RoleId:
        return self._id

    def _declare(self, cycle_id: CycleId) -> RoleResponse:
        import dataclasses

        if self._script:
            item = self._script.pop(0)
            return dataclasses.replace(item, cycle_id=cycle_id)
        return RoleResponse(cycle_id, self._id, ResponseId(f"{self._id}-idle"), ())

    async def perceive(self, turn: TurnInput) -> RoleResponse:
        self.perceived.append(turn)
        return self._declare(turn.cycle_id)

    async def rejected(self, feedback: Rejection) -> RoleResponse:
        print(
            f"    [打回] {feedback.role_id} 的 {feedback.rejected_response_id}：{feedback.reason}"
        )
        return self._declare(feedback.cycle_id)


class DemoWorld(World):
    """假语义世界：位置随剧情时间推进，行动结算、修改生效与被动影响。

    该演示世界的原子提交：校验通过后一次性应用全部变更并递增版本；
    位置在进展更新中实际变化时也递增版本。
    """

    def __init__(self) -> None:
        self._version = WorldStateVersion(0)
        self.position = 0.0
        self.speed = 0.0
        self.facts: list[str] = []
        self._last_time = SimulationTime(0)
        self._committed: list[AdjudicationCommit] = []
        self._reference = EventId("demo")
        self._active: list[tuple[ActionId, str]] = []
        self._passive: list[str] = []
        self._pending_nodes: tuple = ()
        self._pending_modifications: tuple = ()

    @property
    def state_version(self) -> WorldStateVersion:
        return self._version

    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        elapsed = simulation_time - self._last_time
        self._last_time = simulation_time
        if self.speed:
            self.position += self.speed * elapsed
            self._version = WorldStateVersion(self._version + 1)
            return ProgressUpdate(
                cycle_id,
                simulation_time,
                notes=(f"林舟位置推进到 {self.position:.1f} 米",),
            )
        return ProgressUpdate(cycle_id, simulation_time)

    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        return ObjectiveContext(
            world_state_version=self._version,
            role_id=role_id,
            conditions=(
                f"林舟当前位置 {self.position:.2f} 米，速度 {self.speed * 1000:.2f} 米/秒",
            ),
        )

    def diagnostic_snapshot(self) -> WorldDiagnostic:
        return WorldDiagnostic(
            world_state_version=self._version,
            action_states=tuple(
                WorldActionState(
                    action_id=action_id,
                    role_id=RoleId(owner),
                    state="进行中",
                    detail=f"位置 {self.position:.2f} 米",
                )
                for action_id, owner in self._active
            ),
            passive_processes=tuple(self._passive),
        )

    async def check_response(self, response: RoleResponse) -> ActionCheckResult:
        blocked = tuple(
            item.action_id
            for item in response.actions
            if "隔空" in getattr(item, "description", "")
        )
        return ActionCheckResult(
            response.cycle_id,
            response.role_id,
            response.response_id,
            self._version,
            not blocked,
            "" if not blocked else "距离与遮挡不允许隔空取物",
            blocked,
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
        changes: list[FactChangeProposal] = []
        outcomes: list[ActionOutcome] = []
        follow_ups: list[WorldEvent] = []
        if event_ids:
            self._reference = tuple(event_ids)[0]
        elif action_nodes:
            self._reference = action_nodes[0].event_id
        # 记住节点与修改：提交时按归属应用实际作用。
        self._pending_nodes = tuple(action_nodes)
        self._pending_modifications = tuple(modifications)
        for event in events:
            changes.append(FactChangeProposal(f"事件落地：{event.description}"))
        for node in action_nodes:
            if node.node is ActionNodeKind.START:
                outcomes.append(ActionOutcome(node.action_id, ActionOutcomeResult.STARTED, "开始"))
                changes.append(FactChangeProposal(f"{node.action_id} 实际开始：{node.description}"))
            elif node.node is ActionNodeKind.CHECKPOINT:
                outcomes.append(
                    ActionOutcome(node.action_id, ActionOutcomeResult.IN_PROGRESS, "进行中")
                )
                changes.append(FactChangeProposal(f"{node.action_id} 到达过程节点"))
            elif node.node is ActionNodeKind.END:
                outcomes.append(
                    ActionOutcome(node.action_id, ActionOutcomeResult.COMPLETED, "按事实完成")
                )
                changes.append(FactChangeProposal(f"{node.action_id} 完成：{node.description}"))
            else:  # 撤回主动维持的客观效果
                changes.append(
                    FactChangeProposal(f"{node.action_id} 撤回主动维持：脚步停下，惯性前冲一段")
                )
                follow_ups.append(
                    WorldEvent(
                        EventId(f"{node.action_id}-inertia-end"),
                        SimulationTime(simulation_time + 700),
                        f"{node.action_id} 的惯性前冲结束",
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
            cycle_id,
            simulation_time,
            tuple(event_ids),
            tuple(action_ids),
            tuple(changes),
            tuple(follow_ups),
            tuple(outcomes),
        )

    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        return AdjudicationValidation(candidate, self._version, bool(candidate.proposed_changes))

    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        validation = self.validate_adjudication(candidate)
        if not validation.passed or validation.candidate != candidate:
            raise ValueError("演示世界拒绝提交未通过校验的候选")
        self.facts.extend(item.description for item in candidate.proposed_changes)
        terminal = {
            outcome.action_id
            for outcome in candidate.action_outcomes
            if outcome.result in (ActionOutcomeResult.COMPLETED, ActionOutcomeResult.FAILED)
        }
        self._active = [item for item in self._active if item[0] not in terminal]
        for node in self._pending_nodes:
            if node.node is ActionNodeKind.START:
                self._active.append((node.action_id, str(node.role_id)))
                if node.action_id == ActionId("a-walk"):
                    self.speed = 0.25 / 1000  # 每毫秒 0.25 米
        for node in self._pending_modifications:
            if "更快" in node.change:
                self.speed *= 2
            elif "更慢" in node.change:
                self.speed /= 2
        for outcome in candidate.action_outcomes:
            if outcome.action_id == ActionId("a-walk") and outcome.result in (
                ActionOutcomeResult.COMPLETED,
                ActionOutcomeResult.FAILED,
            ):
                self.speed = 0.0
        for follow in candidate.follow_up_events:
            self._passive.append(follow.description)
        for event_id in candidate.event_ids:
            if str(event_id).endswith("-inertia-end"):
                self.speed = 0.0
        self._pending_nodes = ()
        self._pending_modifications = ()
        self._version = WorldStateVersion(self._version + 1)
        commit = AdjudicationCommit(
            candidate, candidate.proposed_changes, self._version, candidate.action_outcomes
        )
        self._committed.append(commit)
        return commit

    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        if not any(item is commit for item in self._committed):
            raise ValueError("只能发布已提交的裁决")
        deliveries: list[Delivery] = []
        for change in commit.committed_changes:
            for role in (LIN, SHEN):
                prefix = "沈言看到" if role == SHEN else "林舟察觉"
                deliveries.append(
                    Delivery(
                        role,
                        Observation(
                            self._reference,
                            commit.candidate.simulation_time,
                            f"{prefix}：{change.description}",
                        ),
                    )
                )
        return tuple(deliveries)


class DemoTranslator:
    """脚本翻译器：按描述给出时间安排（延迟、持续、过程节点，毫秒）。"""

    PLANS: dict[str, tuple[int, int, tuple[int, ...]]] = {
        "走向书桌": (2_000, 6_000, (2_000,)),
        "拉开抽屉": (500, 3_000, ()),
        "跟随林舟": (1_000, 9_000, (3_000,)),
    }

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        schedules: list[ActionSchedule] = []
        modified: list[ModifiedActionSchedule] = []
        from sceneweave.domain.messages import NewAction

        for item in request.response.actions:
            if isinstance(item, NewAction):
                delay, duration, checkpoints = self.PLANS.get(item.description, (1_000, 4_000, ()))
                start = request.current_time + delay
                schedules.append(
                    ActionSchedule(
                        item.action_id,
                        item.commit_group_id,
                        SimulationTime(start),
                        SimulationTime(start + duration),
                        tuple(SimulationTime(start + c) for c in checkpoints),
                    )
                )
        return TranslationResult(
            request.response.cycle_id,
            request.response.role_id,
            request.response.response_id,
            tuple(schedules),
            tuple(modified),
        )


def declare(
    role: RoleId, response_id: str, actions: tuple[object, ...], cycle: str
) -> RoleResponse:
    return RoleResponse(CycleId(cycle), role, ResponseId(response_id), actions)  # type: ignore[arg-type]


class PrintingTracer(TurnTracer):
    """把周期关键边界打印成可读因果链。"""

    INTERESTING = {
        "engine.cycle_start",
        "world.commit",
        "registration.plan",
        "registration.commit",
        "engine.stop",
        "engine.error",
    }

    def trace(self, record: TraceRecord) -> None:
        if record.stage not in self.INTERESTING:
            return
        head = f"t={record.simulation_time:>6}ms [{record.stage}] {record.summary}"
        if record.outputs:
            head += "：" + "；".join(record.outputs[:4])
        print(head)


class CompositeTracer(TurnTracer):
    """组合追踪器：打印关键边界，同时写入 Concordia 结构化日志。"""

    def __init__(self) -> None:
        self.log = structured_logging.SimulationLog()
        self._log_tracer = SimulationLogTracer(self.log)
        self._printer = PrintingTracer()

    def trace(self, record: TraceRecord) -> None:
        self._log_tracer.trace(record)
        self._printer.trace(record)


async def main() -> None:
    from sceneweave.domain.messages import MaintainAction, NewAction

    # 剧情时间线（内思考间隔 8 秒、时限 30 秒）：
    #   t=0     开场事件；林舟先提出不可能的隔空取物被打回，改走向书桌
    #   t=2000  a-walk 开始（位置开始按剧情时间推进）
    #   t=4000  a-walk 过程节点；沈言决定跟随
    #   t=5000  a-follow 开始
    #   t=8000  a-walk 结算完成（与内思考同刻合并）；沈言撤回跟随，
    #           世界裁决惯性前冲并登记后续事件
    #   t=8700  惯性前冲结束；林舟转向抽屉
    #   t=9200/12200  a-drawer 开始/完成
    #   t=16000/24000 周期内思考；t=30000 时限停止
    lin_script = [
        declare(
            LIN,
            "lin-1",
            (NewAction(ActionId("a-reach"), "隔空拿到桌上钥匙", CommitGroupId("g1")),),
            "cycle-x",
        ),
        declare(
            LIN,
            "lin-2",
            (NewAction(ActionId("a-walk"), "走向书桌", CommitGroupId("g2")),),
            "cycle-x",
        ),
        declare(LIN, "lin-3", (MaintainAction(ActionId("a-walk")),), "cycle-x"),
        declare(LIN, "lin-4", (MaintainAction(ActionId("a-walk")),), "cycle-x"),
        declare(LIN, "lin-5", (MaintainAction(ActionId("a-walk")),), "cycle-x"),
        declare(LIN, "lin-6", (), "cycle-x"),
        declare(
            LIN,
            "lin-7",
            (NewAction(ActionId("a-drawer"), "拉开抽屉", CommitGroupId("g3")),),
            "cycle-x",
        ),
        declare(LIN, "lin-8", (MaintainAction(ActionId("a-drawer")),), "cycle-x"),
        declare(LIN, "lin-9", (), "cycle-x"),
        declare(LIN, "lin-10", (), "cycle-x"),
        declare(LIN, "lin-11", (), "cycle-x"),
    ]
    shen_script = [
        declare(SHEN, "shen-1", (), "cycle-x"),
        declare(SHEN, "shen-2", (), "cycle-x"),
        declare(
            SHEN,
            "shen-3",
            (NewAction(ActionId("a-follow"), "跟随林舟", CommitGroupId("g4")),),
            "cycle-x",
        ),
        declare(SHEN, "shen-4", (MaintainAction(ActionId("a-follow")),), "cycle-x"),
        declare(SHEN, "shen-5", (), "cycle-x"),
        declare(SHEN, "shen-6", (), "cycle-x"),
        declare(SHEN, "shen-7", (), "cycle-x"),
        declare(SHEN, "shen-8", (), "cycle-x"),
        declare(SHEN, "shen-9", (), "cycle-x"),
        declare(SHEN, "shen-10", (), "cycle-x"),
    ]
    world = DemoWorld()
    scheduler_tracer = CompositeTracer()
    scheduler = Scheduler(
        roles=[DemoRole(LIN, lin_script), DemoRole(SHEN, shen_script)],
        world=world,
        translator=DemoTranslator(),
        config=RunConfig.from_seconds(
            introspection_interval_seconds=8,
            max_simulation_duration_seconds=30,
        ),
        initial_events=[
            WorldEvent(EventId("startup"), SimulationTime(0), "书房里很安静，午后光线斜照")
        ],
        tracer=scheduler_tracer,
    )
    result = await scheduler.run()
    log_path = Path("offline_demo_log.json")
    log_path.write_text(scheduler_tracer.log.to_json(), encoding="utf-8")
    print(f"结构化日志已导出：{log_path}（可用任意 JSON 工具检索因果链）")
    print()
    print(
        f"运行结束：{result.reason.value}，完成周期 {result.completed_cycles}，"
        f"最终剧情时刻 {result.final_time}ms"
    )
    print(f"林舟最终位置 {world.position:.2f} 米（只随剧情时间推进）")
    print("已提交的世界事实：")
    for fact in world.facts:
        print(f"  - {fact}")
    if result.reason is RunStopReason.ERROR and result.error:
        print(f"运行错误：{result.error}")


if __name__ == "__main__":
    asyncio.run(main())
