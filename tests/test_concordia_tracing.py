"""以真实 Concordia 结构化日志验证追踪接入路径，不访问模型 API。"""

import ast
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from concordia.typing.entity_component import ComponentWithLogging
from concordia.utils import structured_logging
from concordia.utils.structured_logging import AIAgentLogInterface

from sceneweave.domain.adjudication import (
    ObjectiveContext,
    ProgressUpdate,
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
    NewAction,
    Observation,
    RoleResponse,
    TriggerReason,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.engine.responses import ResponseRunner
from sceneweave.integrations.concordia.tracing import (
    SimulationLogTracer,
    component_logging_channel,
)
from sceneweave.roles.base import Role
from sceneweave.world.base import World

FIXED_CLOCK = "2026-01-01T00:00:00+00:00"


class RetryingRole(Role):
    """先提交被拒绝的行动，打回后改用可通过检查的行动。"""

    def __init__(self, cycle: str = "cycle-1") -> None:
        self.cycle = CycleId(cycle)
        self.rejections = 0

    @property
    def role_id(self) -> RoleId:
        return RoleId("lin_zhou")

    async def perceive(self, turn_input: TurnInput) -> RoleResponse:
        return self._response("r-1", "不可能地拉开抽屉")

    async def rejected(self, feedback: Any) -> RoleResponse:
        self.rejections += 1
        return self._response("r-2", "先走近书桌")

    def _response(self, response_id: str, description: str) -> RoleResponse:
        return RoleResponse(
            self.cycle,
            RoleId("lin_zhou"),
            ResponseId(response_id),
            (NewAction(ActionId("a-1"), description, CommitGroupId("g-1")),),
        )


class CheckingWorld(World):
    """拒绝包含“不可能”描述的响应，其余通过。"""

    @property
    def state_version(self) -> WorldStateVersion:
        return WorldStateVersion(0)

    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        return ProgressUpdate(cycle_id=cycle_id, simulation_time=simulation_time)

    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        return ObjectiveContext(
            world_state_version=self.state_version,
            role_id=role_id,
            conditions=("无特殊客观条件",),
        )

    def diagnostic_snapshot(self) -> WorldDiagnostic:
        return WorldDiagnostic(world_state_version=self.state_version)

    async def check_response(self, candidate: RoleResponse) -> ActionCheckResult:
        blocked = tuple(
            item.action_id
            for item in candidate.actions
            if isinstance(item, NewAction) and "不可能" in item.description
        )
        return ActionCheckResult(
            cycle_id=candidate.cycle_id,
            role_id=candidate.role_id,
            response_id=candidate.response_id,
            world_state_version=self.state_version,
            passed=not blocked,
            reason="" if not blocked else "客观条件不允许",
            action_ids=blocked,
        )

    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
    ) -> Any:
        raise AssertionError("本测试不使用裁决")

    def validate_adjudication(self, candidate: Any) -> Any:
        raise AssertionError("本测试不使用裁决校验")

    def commit_adjudication(self, candidate: Any) -> Any:
        raise AssertionError("本测试不使用提交")

    def publish(self, commit: Any) -> tuple[Any, ...]:
        raise AssertionError("本测试不使用发布")


def obs(event: str, time: int, content: str) -> Observation:
    return Observation(EventId(event), SimulationTime(time), content)


def run_traced_cycle(
    tracer: SimulationLogTracer, *, cycle: str = "cycle-1", time: int = 30_000
) -> None:
    role = RetryingRole(cycle)
    turn_input = TurnInput(
        cycle_id=CycleId(cycle),
        role_id=RoleId("lin_zhou"),
        simulation_time=SimulationTime(time),
        trigger=TriggerReason((EventId("door-opened"),)),
        observations=(obs("door-opened", time, "书房门被推开"),),
    )

    async def scenario() -> None:
        runner = ResponseRunner([role], CheckingWorld(), tracer=tracer)
        await runner.run_cycle({RoleId("lin_zhou"): turn_input})

    asyncio.run(scenario())


def traced_log() -> tuple[structured_logging.SimulationLog, SimulationLogTracer]:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)
    run_traced_cycle(tracer)
    return log, tracer


def reconstructed(log: structured_logging.SimulationLog, entry: Any) -> dict[str, Any]:
    return dict(log.reconstruct_value(entry.deduplicated_data))


def traced_cycle_ids(log: structured_logging.SimulationLog) -> list[str]:
    starts = log.get_entries_by_component("cycle.start")
    return [reconstructed(log, entry)["cycle_id"] for entry in starts]


def test_full_cycle_is_recorded_with_searchable_identity_chain() -> None:
    log, _ = traced_log()
    stages = {entry.component_name for entry in log.entries}
    assert {
        "cycle.start",
        "role.perceive",
        "role.response",
        "world.check",
        "role.rejected",
        "cycle.complete",
    } <= stages
    assert log.get_steps() == [0], "同一周期的记录共享同一 step"
    lin_entries = log.get_entries_by_entity("lin_zhou")
    assert lin_entries, "角色记录可按实体检索"
    engine_entries = log.get_entries_by_entity("engine")
    assert {entry.component_name for entry in engine_entries} >= {"cycle.start"}
    checks = log.get_entries_by_component("world.check")
    assert len(checks) == 2, "打回后再次检查"
    first = reconstructed(log, checks[0])
    assert first["cycle_id"] == "cycle-1"
    assert first["simulation_time_ms"] == 30_000
    assert first["response_id"] == "r-1"
    assert first["world_state_version"] == 0
    assert first["details"]["passed"] is False
    assert first["details"]["reason"] == "客观条件不允许"
    assert first["action_ids"] == ["a-1"]
    assert first["outputs"] == ["不可能地拉开抽屉"], "检查记录携带实际行动描述"


def test_perceive_record_carries_trigger_and_observation_texts() -> None:
    log, _ = traced_log()
    perceive = log.get_entries_by_component("role.perceive")[0]
    data = reconstructed(log, perceive)
    assert data["event_ids"] == ["door-opened"], "触发来源可检索"
    assert data["inputs"] == ["书房门被推开"], "感知文本可检索"
    assert data["details"]["introspection"] is False


def test_retry_relation_links_old_and_new_response_ids() -> None:
    log, _ = traced_log()
    responses = [
        reconstructed(log, entry) for entry in log.get_entries_by_component("role.response")
    ]
    initial = next(item for item in responses if item["response_id"] == "r-1")
    retry = next(item for item in responses if item["response_id"] == "r-2")
    assert initial["related_response_id"] is None
    assert retry["related_response_id"] == "r-1", "重试记录关联被打回的旧响应"
    assert retry["outputs"] == ["先走近书桌"]
    rejected = reconstructed(log, log.get_entries_by_component("role.rejected")[0])
    assert rejected["response_id"] == "r-1"
    assert rejected["details"]["reason"] == "客观条件不允许"


def test_records_are_searchable_along_the_identity_chain() -> None:
    log, _ = traced_log()
    interface = AIAgentLogInterface(log)
    found = interface.filter_entries(component_name="role.rejected", include_content=True)
    assert found
    data = found[0]["data"]
    assert data["cycle_id"] == "cycle-1"
    assert data["role_id"] == "lin_zhou"
    assert data["response_id"] == "r-1"
    assert "客观条件不允许" in data["details"]["reason"]
    summary_hits = interface.search_summaries("打回")
    assert summary_hits


def test_separate_cycles_get_separate_steps() -> None:
    log, tracer = traced_log()
    run_traced_cycle(tracer, cycle="cycle-2", time=60_000)
    assert log.get_steps() == [0, 1]
    late = log.get_entries_by_step(1)
    assert late and late[0].component_name == "cycle.start"
    assert traced_cycle_ids(log) == ["cycle-1", "cycle-2"]


def test_details_cannot_override_reserved_fields() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)
    tracer.trace(
        TraceRecord(
            stage="cycle.start",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(5),
            summary="周期开始",
            details={"cycle_id": "伪造周期", "role_id": "伪造角色", "simulation_time_ms": -1},
        )
    )
    data = reconstructed(log, log.entries[0])
    assert data["cycle_id"] == "cycle-1", "保留字段不可被细节覆盖"
    assert data["simulation_time_ms"] == 5
    assert data["role_id"] is None
    assert data["details"]["cycle_id"] == "伪造周期", "细节数据不丢失"


def test_empty_action_set_is_logged_explicitly() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    class WithdrawingRole(RetryingRole):
        async def perceive(self, turn_input: TurnInput) -> RoleResponse:
            return RoleResponse(self.cycle, RoleId("lin_zhou"), ResponseId("r-empty"), ())

        async def rejected(self, feedback: Any) -> RoleResponse:
            raise AssertionError("空集合合法，不应被打回")

    role = WithdrawingRole()

    async def scenario() -> None:
        await ResponseRunner([role], CheckingWorld(), tracer=tracer).run_cycle(
            {
                RoleId("lin_zhou"): TurnInput(
                    CycleId("cycle-1"),
                    RoleId("lin_zhou"),
                    SimulationTime(1),
                    TriggerReason(introspection=True),
                )
            }
        )

    asyncio.run(scenario())
    responses = [
        reconstructed(log, entry) for entry in log.get_entries_by_component("role.response")
    ]
    empty = next(item for item in responses if item["response_id"] == "r-empty")
    assert empty["outputs"] == []
    assert empty["action_ids"] == []
    assert empty["details"]["new_actions"] == 0
    assert empty["details"]["maintained"] == 0


def test_role_error_keeps_identity_and_original_exception() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    class FailingAfterRejected(RetryingRole):
        async def rejected(self, feedback: Any) -> RoleResponse:
            self.rejections += 1
            raise RuntimeError("内部思考失败")

    role = FailingAfterRejected()

    async def scenario() -> None:
        await ResponseRunner([role], CheckingWorld(), tracer=tracer).run_cycle(
            {
                RoleId("lin_zhou"): TurnInput(
                    CycleId("cycle-1"),
                    RoleId("lin_zhou"),
                    SimulationTime(1),
                    TriggerReason(introspection=True),
                )
            }
        )

    with pytest.raises(BaseExceptionGroup):
        asyncio.run(scenario())
    errors = log.get_entries_by_component("role.error")
    assert errors, "角色错误必须留痕"
    data = reconstructed(log, errors[0])
    assert data["response_id"] == "r-1", "错误保留当前响应身份"
    assert data["action_ids"] == ["a-1"]
    assert "RuntimeError" in data["error"]
    assert "内部思考失败" in data["error"]
    cycle_errors = log.get_entries_by_component("cycle.error")
    assert len(cycle_errors) == 1, "任务组内角色错误也必须记录周期失败"
    assert "内部思考失败" in reconstructed(log, cycle_errors[0])["error"]


def test_cycle_error_and_cancellation_leave_traces() -> None:
    class SilentDriftWorld(CheckingWorld):
        """检查通过后悄悄推进版本，触发周期结束时的漂移检测。"""

        def __init__(self) -> None:
            self._v = 0
            self.checks = 0

        @property
        def state_version(self) -> WorldStateVersion:
            return WorldStateVersion(self._v)

        async def check_response(self, candidate: RoleResponse) -> ActionCheckResult:
            self.checks += 1
            result = await super().check_response(candidate)
            if self.checks > 1:
                self._v = 1
            return result

    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    async def scenario() -> None:
        role = RetryingRole()
        await ResponseRunner([role], SilentDriftWorld(), tracer=tracer).run_cycle(
            {
                RoleId("lin_zhou"): TurnInput(
                    CycleId("cycle-1"),
                    RoleId("lin_zhou"),
                    SimulationTime(1),
                    TriggerReason(introspection=True),
                )
            }
        )

    with pytest.raises(RuntimeError, match="共同世界状态版本发生变化"):
        asyncio.run(scenario())
    cycle_errors = log.get_entries_by_component("cycle.error")
    assert cycle_errors, "周期级错误必须留痕"
    data = reconstructed(log, cycle_errors[0])
    assert "共同世界状态版本发生变化" in data["error"]
    assert data["cycle_id"] == "cycle-1"


def test_cancelled_sibling_role_leaves_trace() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    class BrokenRole(RetryingRole):
        async def perceive(self, turn_input: TurnInput) -> RoleResponse:
            raise RuntimeError("启动失败")

    class GatedRole(Role):
        def __init__(self) -> None:
            self._entered = asyncio.Event()

        @property
        def role_id(self) -> RoleId:
            return RoleId("shen_yan")

        async def perceive(self, turn_input: TurnInput) -> RoleResponse:
            self._entered.set()
            await asyncio.Event().wait()  # 模拟长时间思考，随后被兄弟任务的失败取消
            raise AssertionError("不可达")

        async def rejected(self, feedback: Any) -> RoleResponse:
            raise AssertionError("不可达")

    async def scenario() -> None:
        runner = ResponseRunner([BrokenRole(), GatedRole()], CheckingWorld(), tracer=tracer)
        with pytest.raises(BaseExceptionGroup):
            await runner.run_cycle(
                {
                    RoleId("lin_zhou"): TurnInput(
                        CycleId("cycle-1"),
                        RoleId("lin_zhou"),
                        SimulationTime(1),
                        TriggerReason(introspection=True),
                    ),
                    RoleId("shen_yan"): TurnInput(
                        CycleId("cycle-1"),
                        RoleId("shen_yan"),
                        SimulationTime(1),
                        TriggerReason(introspection=True),
                    ),
                }
            )

    asyncio.run(scenario())
    cancelled = log.get_entries_by_component("role.cancelled")
    assert cancelled, "被取消的角色必须留痕"
    data = reconstructed(log, cancelled[0])
    assert data["role_id"] == "shen_yan"
    assert data["cycle_id"] == "cycle-1"


def test_external_cancellation_records_cycle_and_role_and_preserves_cancelled_error() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    class GatedRole(RetryingRole):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()

        async def perceive(self, turn_input: TurnInput) -> RoleResponse:
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("不可达")

    async def scenario() -> None:
        role = GatedRole()
        runner = ResponseRunner([role], CheckingWorld(), tracer=tracer)
        task = asyncio.create_task(
            runner.run_cycle(
                (
                    TurnInput(
                        CycleId("cycle-1"),
                        role.role_id,
                        SimulationTime(1),
                        TriggerReason(introspection=True),
                    ),
                )
            )
        )
        await role.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(log.get_entries_by_component("role.cancelled")) == 1
    assert len(log.get_entries_by_component("cycle.cancelled")) == 1
    assert log.get_entries_by_component("cycle.complete") == []


class FakeSystemComponent(ComponentWithLogging):
    """以真实 Concordia 组件基类验证组件日志通道接入。"""

    def get_state(self) -> dict[str, str]:
        return {}

    def set_state(self, state: Any) -> None:
        return None

    def judge(self, question: str, conclusion: str) -> None:
        self._logging_channel({"Key": question, "Value": conclusion})


def test_concordia_component_logging_channel_writes_into_simulation_log() -> None:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)
    component = FakeSystemComponent()
    component.set_logging_channel(
        component_logging_channel(
            log,
            entity_name="lin_zhou",
            component_name="system1",
            step=lambda: tracer.step_for(CycleId("cycle-1")),
            timestamp=lambda: FIXED_CLOCK,
        )
    )
    component.judge("这件事是否影响我", "需要重新考虑")
    entries = log.get_entries_by_component("system1")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.entity_name == "lin_zhou"
    assert entry.step == 0
    assert entry.timestamp == FIXED_CLOCK
    assert reconstructed(log, entry)["Value"] == "需要重新考虑"
    assert entry.step in log.get_steps()


def test_domain_world_engine_do_not_import_concordia() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "sceneweave"
    integrations = root / "integrations"
    offenders: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if integrations in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".")[0] == "concordia" for name in names):
                offenders.append(path)
    assert not offenders, f"领域模块不得依赖 Concordia：{offenders}"


def test_tracer_entry_types_are_queryable() -> None:
    log, _ = traced_log()
    types = {entry.entry_type for entry in log.entries}
    assert types == {"sceneweave"}
    summary = log.get_summary()
    assert set(summary["entities"]) == {"engine", "lin_zhou"}
    assert summary["total_steps"] == 1


def test_tracer_rejects_empty_stage() -> None:
    with pytest.raises(ValueError, match="追踪阶段"):
        TraceRecord(
            stage="",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(1),
            summary="摘要",
        )


def test_full_scheduler_run_is_searchable_in_simulation_log() -> None:
    """完整调度闭环接入结构化日志：关键输入/输出可沿因果链检索。"""

    from sceneweave.domain.identity import ActionId
    from sceneweave.domain.messages import MaintainAction, NewAction, WorldEvent
    from sceneweave.engine.config import RunConfig
    from sceneweave.engine.scheduler import RunStopReason, Scheduler
    from tests.fakes import FakeSemanticWorld, ScriptedTranslator

    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: FIXED_CLOCK)

    class RetryingThenRunning(Role):
        def __init__(self) -> None:
            self.rounds = 0

        @property
        def role_id(self) -> RoleId:
            return RoleId("lin_zhou")

        async def perceive(self, turn: TurnInput) -> RoleResponse:
            self.rounds += 1
            if self.rounds == 1:
                return RoleResponse(
                    turn.cycle_id,
                    self.role_id,
                    ResponseId("lin-1"),
                    (NewAction(ActionId("a-late"), "立刻开始", CommitGroupId("g1")),),
                )
            if self.rounds >= 4:
                # t=4000 结束节点先结算完成，此后空声明（终态无需维持）。
                return RoleResponse(turn.cycle_id, self.role_id, ResponseId("lin-idle"), ())
            return RoleResponse(
                turn.cycle_id,
                self.role_id,
                ResponseId(f"lin-keep-{self.rounds}"),
                (MaintainAction(ActionId("a-walk")),),
            )

        async def rejected(self, feedback: Any) -> RoleResponse:
            # 时间安排被打回后返回新的完整声明（同周期、不重投感知）。
            return RoleResponse(
                feedback.cycle_id,
                self.role_id,
                ResponseId("lin-2"),
                (NewAction(ActionId("a-walk"), "走向书桌", CommitGroupId("g2")),),
            )

    class RejectTooEarly(ScriptedTranslator):
        async def translate(self, request):  # type: ignore[no-untyped-def]
            result = await super().translate(request)
            if request.response.response_id == "lin-1":
                # 时间约束不合法：触发引擎侧打回（不是角色侧）。
                import dataclasses

                bad = dataclasses.replace(
                    result.schedules[0],
                    start=request.current_time,
                    expected_end=SimulationTime(request.current_time + 100),
                )
                import sceneweave.domain.translation as translation_mod

                return translation_mod.TranslationResult(
                    result.cycle_id, result.role_id, result.response_id, (bad,)
                )
            return result

    world = FakeSemanticWorld(["lin_zhou"])
    world.set_speed("a-walk", 0.001)
    scheduler = Scheduler(
        roles=[RetryingThenRunning()],
        world=world,
        translator=RejectTooEarly(plans={"走向书桌": (1000, 3000, (1000,))}),
        config=RunConfig(
            introspection_interval=SimulationTime(10_000),
            max_duration=SimulationTime(10_000),
        ),
        initial_events=[WorldEvent(EventId("startup"), SimulationTime(0), "书房安静")],
        tracer=tracer,
    )

    async def scenario() -> None:
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT

    asyncio.run(scenario())

    def raw(entry: Any) -> dict[str, Any]:
        return dict(log.reconstruct_value(entry.deduplicated_data))

    # 引擎/世界/角色/翻译的阶段都可检索。
    stages = {entry.component_name for entry in log.entries}
    assert {
        "engine.cycle_start",
        "world.progress",
        "world.adjudicate",
        "world.validate",
        "world.commit",
        "world.publish",
        "role.perceive",
        "role.response",
        "translation.request",
        "translation.result",
        "registration.rejected",
        "registration.plan",
        "registration.commit",
        "translation.skipped",
    } <= stages
    # 翻译请求携带 World 客观上下文与既有安排（不是只记录成功摘要）。
    requests = [raw(e) for e in log.get_entries_by_component("translation.request")]
    first = next(r for r in requests if r["response_id"] == "lin-2")
    assert any("位置" in item for item in first["inputs"]), "客观上下文进入日志"
    assert any("a-walk" in item or "新增" in item for item in first["outputs"])
    # 世界裁决记录携带节点内容与修改输入。
    adjudications = [raw(e) for e in log.get_entries_by_component("world.adjudicate")]
    start_entry = next(r for r in adjudications if any("start:" in item for item in r["inputs"]))
    assert any("走向书桌" in item for item in start_entry["inputs"]), "节点描述进入日志"
    # 登记计划输出包含完整时间安排。
    plans = [raw(e) for e in log.get_entries_by_component("registration.plan")]
    assert any("a-walk:" in item for p in plans for item in p["outputs"])
    # JSON 导出可离线检索（往返解析）。
    payload = json.loads(log.to_json())
    assert payload["entries"], "结构化日志可导出为 JSON"
