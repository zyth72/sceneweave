"""模型驱动行动翻译器的行为测试（脚本采样器，无真实模型）。

覆盖：合法组合解析与完整关联、维持/空跳过模型、结构错误不伪空、
业务时间约束经引擎打回、冻结、修改携带原安排与客观上下文、并发
串行/并行、真实 Concordia 桥与结构化日志、端到端 StepData、按作用域
录制重放并行运行。
"""

import asyncio

import pytest
from concordia.utils import structured_logging

from sceneweave.domain.adjudication import ObjectiveContext
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
)
from sceneweave.domain.messages import ActionPhase, NewAction, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.domain.translation import ActionArrangement, TranslationRequest
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.control import step_data_from_report
from sceneweave.integrations.concordia.sampling import (
    LanguageModelSampler,
    RecordingDirectory,
    ReplayDirectory,
)
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator, ModelTranslatorError

from .fakes import (
    FakeSemanticWorld,
    ScriptedSampler,
    maintain,
    ms,
    response,
    rid,
)

VALID_NEW = (
    '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
    ' "start": 2000, "expected_end": 6000, "checkpoints": [3000, 5000]}],'
    ' "modified_schedules": []}'
)
VALID_COMBO = (
    '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
    ' "start": 2000, "expected_end": 6000}],'
    ' "modified_schedules": [{"action_id": "a-0", "start": 1000,'
    ' "expected_end": 9000, "checkpoints": [2000, 7000]}]}'
)


def g1_action(action_id: str, description: str = "走向书桌") -> NewAction:
    return NewAction(ActionId(action_id), description, CommitGroupId("g-1"))


def request_for(
    actions: tuple,
    *,
    role: str = "lin",
    time: int = 1000,
    version: int = 3,
    arrangements: tuple = (),
    conditions: tuple[str, ...] | None = ("林舟位置 0.00 米，速度 0.0",),
) -> TranslationRequest:
    from sceneweave.domain.messages import RoleResponse

    role_id = rid(role)
    return TranslationRequest(
        response=RoleResponse(CycleId("cycle-1"), role_id, ResponseId("r-1"), actions),
        current_time=SimulationTime(time),
        world_state_version=WorldStateVersion(version),
        arrangements=arrangements,
        objective_context=(
            ObjectiveContext(
                world_state_version=WorldStateVersion(version),
                role_id=role_id,
                conditions=conditions,
            )
            if conditions is not None
            else None
        ),
    )


def arrangement_for(
    action_id: str, start: int, end: int, checkpoints: tuple = ()
) -> ActionArrangement:
    return ActionArrangement(
        action_id=ActionId(action_id),
        phase=ActionPhase.STARTED,
        description="既有行动",
        start=SimulationTime(start),
        expected_end=SimulationTime(end),
        checkpoints=tuple(SimulationTime(c) for c in checkpoints),
    )


class Collector(TurnTracer):
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def trace(self, record: TraceRecord) -> None:
        self.records.append(record)

    def stages(self) -> set[str]:
        return {r.stage for r in self.records}


def test_valid_new_group_and_combo_parse_with_association() -> None:
    """合法新组与修改组合：结果关联响应、组一致、毫秒为整数。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"translator": [VALID_COMBO]})
        translator = ModelTranslator.shared(sampler)
        from sceneweave.domain.messages import ModifiedAction

        request = request_for(
            (
                maintain("a-keep"),
                g1_action("a-1"),
                ModifiedAction(ActionId("a-0"), "放慢速度"),
            ),
            arrangements=(arrangement_for("a-0", 1000, 5000, (2000,)),),
        )
        result = await translator.translate(request)
        assert result.cycle_id == CycleId("cycle-1")
        assert result.role_id == rid("lin")
        assert result.response_id == ResponseId("r-1")
        [schedule] = result.schedules
        assert schedule.action_id == ActionId("a-1")
        assert schedule.commit_group_id == CommitGroupId("g-1")
        assert schedule.start == SimulationTime(2000)
        assert schedule.checkpoints == ()
        [modified] = result.modified_schedules
        assert modified.action_id == ActionId("a-0")
        assert modified.checkpoints == (SimulationTime(2000), SimulationTime(7000))

    asyncio.run(scenario())


def test_maintain_or_empty_skips_model_with_log() -> None:
    """仅维持/空声明：合成空结果、零模型调用、显式跳过留痕。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"translator": []})
        collector = Collector()
        translator = ModelTranslator.shared(sampler, tracer=collector)
        empty = await translator.translate(request_for((), conditions=("条件",)))
        assert empty.schedules == () and empty.modified_schedules == ()
        kept = await translator.translate(request_for((maintain("a-0"),), conditions=("条件",)))
        assert kept.schedules == () and kept.modified_schedules == ()
        assert sampler.calls == [], "零模型调用"
        assert collector.stages() == {"translator.skip"}
        assert all(r.response_id == ResponseId("r-1") for r in collector.records)

    asyncio.run(scenario())


def test_missing_objective_context_is_explicit_error() -> None:
    """新行动/修改缺客观上下文：明确错误，不猜默认延迟。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"translator": []})
        translator = ModelTranslator.shared(sampler)
        with pytest.raises(ModelTranslatorError, match="缺少只读客观上下文"):
            await translator.translate(request_for((g1_action("a-1", "走"),), conditions=None))
        with pytest.raises(ModelTranslatorError, match="缺少只读客观上下文"):
            await translator.translate(
                request_for(
                    (NewAction(ActionId("a-9"), "x", CommitGroupId("g")),),
                    conditions=None,
                )
            )
        assert sampler.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "raw",
    [
        "不是 JSON",
        '{"schedules": [], "modified_schedules": [], "extra": 1}',
        '{"schedules": [], "schedules": [], "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
        ' "start": true, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
        ' "start": 2000.5, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
        ' "start": 2000, "expected_end": 6000, "checkpoints": [3000, true]}],'
        ' "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
        ' "start": NaN, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": "not-a-list", "modified_schedules": []}',
        '{"modified_schedules": []}',
        '{"schedules": [{"action_id": "a-wrong", "commit_group_id": "g-1",'
        ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-other",'
        ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
        ' "start": 2000, "expected_end": 6000},'
        ' {"action_id": "a-extra", "commit_group_id": "g-1",'
        ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}',
        '{"schedules": [], "modified_schedules": [{"action_id": "a-none",'
        ' "start": 1000, "expected_end": 5000}]}',
    ],
)
def test_structural_errors_are_runtime_errors(raw: str) -> None:
    """结构/关联错误：运行错误，不伪空、不部分接受。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"translator": [raw]})
        translator = ModelTranslator.shared(sampler)
        request = request_for(
            (g1_action("a-1", "走"),),
            arrangements=(arrangement_for("a-0", 1000, 5000),),
        )
        with pytest.raises(ModelTranslatorError):
            await translator.translate(request)

    asyncio.run(scenario())


def test_maintained_action_must_not_be_rescheduled() -> None:
    """维持行动出现在安排中：运行错误（维护不重译）。"""

    async def scenario() -> None:
        raw = (
            '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
            ' "start": 2000, "expected_end": 6000},'
            ' {"action_id": "a-keep", "commit_group_id": "g-1",'
            ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}'
        )
        sampler = ScriptedSampler({"translator": [raw]})
        translator = ModelTranslator.shared(sampler)
        with pytest.raises(ModelTranslatorError, match="维持行动不得重新安排"):
            await translator.translate(request_for((maintain("a-keep"), g1_action("a-1", "x"))))

    asyncio.run(scenario())


def test_modified_requires_arrangement_and_carries_context_in_prompt() -> None:
    """修改缺既有安排是错误；提示携带原安排与客观上下文。"""

    async def scenario() -> None:
        raw = (
            '{"schedules": [], "modified_schedules": [{"action_id": "a-0",'
            ' "start": 1000, "expected_end": 9000}]}'
        )
        sampler = ScriptedSampler({"translator": [raw]})
        translator = ModelTranslator.shared(sampler)
        request = request_for(
            (NewAction(ActionId("a-x"), "新", CommitGroupId("g")),),
            arrangements=(arrangement_for("a-other", 100, 500),),
        )
        # 声明没有修改但输出有 → 关联错误路径。
        with pytest.raises(ModelTranslatorError):
            await translator.translate(request)
        # 正向：修改声明的提示包含历史安排与客观条件。
        sampler2 = ScriptedSampler({"translator": [raw]})
        translator2 = ModelTranslator.shared(sampler2)
        from sceneweave.domain.messages import ModifiedAction

        request3 = TranslationRequest(
            response=response("lin", "r-1", (ModifiedAction(ActionId("a-0"), "放慢"),)),
            current_time=SimulationTime(3000),
            world_state_version=WorldStateVersion(3),
            arrangements=(arrangement_for("a-0", 1000, 5000, (2000,)),),
            objective_context=ObjectiveContext(
                WorldStateVersion(3), rid("lin"), ("林舟位置 2.30 米，速度 0.25",)
            ),
        )
        await translator2.translate(request3)
        [(_, prompt)] = sampler2.calls
        assert "【角色主观声明】" in prompt and "【World 客观上下文】" in prompt
        assert "放慢" in prompt, "主观修改意向进入提示"
        assert "1000→5000ms" in prompt and "2000" in prompt, "历史安排进入提示"
        assert "位置 2.30 米" in prompt, "客观条件进入提示"

    asyncio.run(scenario())


def test_same_role_concurrent_refused_and_roles_parallel() -> None:
    """同角色并发是明确错误；不同角色可并行等待。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        sampler = ScriptedSampler({"translator": [VALID_NEW, VALID_NEW]}, gate=gate)
        translator = ModelTranslator.shared(sampler)
        first = asyncio.create_task(translator.translate(request_for((g1_action("a-1", "x"),))))
        for _ in range(1000):
            if sampler.in_flight:
                break
            await asyncio.sleep(0)
        second = asyncio.create_task(translator.translate(request_for((g1_action("a-1", "x"),))))
        with pytest.raises(ModelTranslatorError, match="串行"):
            await second
        other = asyncio.create_task(
            translator.translate(request_for((g1_action("a-1", "x"),), role="shen"))
        )
        await asyncio.sleep(0)
        assert not other.done(), "不同角色并行等待"
        gate.set()
        await first
        await other

    asyncio.run(scenario())


def test_scheduler_rejects_bad_timing_via_role_single_perceive() -> None:
    """合法结构但时间不合约束：经引擎 rejected 重试，单次 perceive。"""

    async def scenario() -> None:
        bad = (
            '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
            ' "start": 0, "expected_end": 6000}], "modified_schedules": []}'
        )
        good = (
            '{"schedules": [{"action_id": "a-lin-2", "commit_group_id": "g-lin-2",'
            ' "start": 2000, "expected_end": 6000, "checkpoints": [4000]}],'
            ' "modified_schedules": []}'
        )
        role_sampler = ScriptedSampler(
            {
                "system1": ['{"reconsider": true, "note": "行动"}']
                + ['{"reconsider": false, "note": "照旧"}'] * 6,
                "system2": [
                    '{"intention": "去书桌", "abandon": []}',
                    '{"intention": "去书桌", "abandon": []}',
                ],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}',
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}',
                ],
            }
        )
        translator_sampler = ScriptedSampler({"translator": [bad, good]})

        class CountingRole(ModelRole):
            def __init__(self) -> None:
                super().__init__(rid("lin"), role_sampler)
                self.perceive_count = 0
                self.perceive_times: list[SimulationTime] = []
                self.reject_count = 0
                self.reasons: list[str] = []

            async def perceive(self, turn):  # type: ignore[no-untyped-def]
                self.perceive_count += 1
                self.perceive_times.append(turn.simulation_time)
                return await super().perceive(turn)

            async def rejected(self, feedback):  # type: ignore[no-untyped-def]
                self.reject_count += 1
                self.reasons.append(feedback.reason)
                return await super().rejected(feedback)

        role = CountingRole()
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(translator_sampler),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert role.perceive_times.count(SimulationTime(0)) == 1, (
            "打回重试不重投原始感知：同时刻只有一次 perceive"
        )
        assert len(set(role.perceive_times)) == len(role.perceive_times)
        assert role.reject_count == 1, "时间问题一次打回"
        assert any("晚于当前时刻" in reason for reason in role.reasons)
        records = scheduler.action_records()
        assert set(records) == {ActionId("a-lin-2")}, "未通过的候选不登记"
        assert records[ActionId("a-lin-2")].start == SimulationTime(2000)

    asyncio.run(scenario())


def test_translation_wait_freezes_clock_and_queue() -> None:
    """模型等待期间剧情时钟与队列冻结。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        role_sampler = ScriptedSampler(
            {
                "system1": ['{"reconsider": true, "note": "行动"}']
                + ['{"reconsider": false, "note": "照旧"}'] * 6,
                "system2": ['{"intention": "去窗户", "abandon": []}'],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向窗户"]}'
                ],
            }
        )
        translator_sampler = ScriptedSampler(
            {"translator": [VALID_NEW.replace("a-1", "a-lin-1").replace("g-1", "g-lin-1")]},
            gate=gate,
        )
        role = ModelRole(rid("lin"), role_sampler)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(translator_sampler),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        task = asyncio.create_task(scheduler.run())
        for _ in range(2000):
            if translator_sampler.in_flight:
                break
            await asyncio.sleep(0)
        assert translator_sampler.in_flight == 1
        assert scheduler.current_time == SimulationTime(0), "翻译等待时钟冻结"
        assert scheduler.stage == "cycle.translation"
        gate.set()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT
        assert ActionId("a-lin-1") in scheduler.action_records()

    asyncio.run(scenario())


def test_end_to_end_model_role_translator_stepdata_and_log() -> None:
    """ModelRole + ModelTranslator + Scheduler + 假 World + StepData + 日志。"""

    async def scenario() -> None:
        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        walk_plan = '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}'
        role_sampler = ScriptedSampler(
            {
                "system1": [
                    '{"reconsider": true, "note": "门开了"}',
                    '{"reconsider": false, "note": "继续"}',
                    '{"reconsider": false, "note": "继续"}',
                    '{"reconsider": false, "note": "继续"}',
                    '{"reconsider": false, "note": "继续"}',
                ],
                "system2": ['{"intention": "查看书桌", "abandon": []}'],
                "system3": [walk_plan],
            }
        )
        translator_sampler = ScriptedSampler(
            {
                "translator": [
                    '{"schedules": [{"action_id": "a-lin-1",'
                    ' "commit_group_id": "g-lin-1", "start": 1000,'
                    ' "expected_end": 3000, "checkpoints": [2000]}],'
                    ' "modified_schedules": []}'
                ]
            }
        )
        world = FakeSemanticWorld(["lin"])
        world.set_speed("a-lin-1", 0.001)
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_sampler, tracer=tracer)],
            world=world,
            translator=ModelTranslator.shared(translator_sampler, tracer=tracer),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-lin-1")]
        assert record.phase is ActionPhase.COMPLETED
        assert record.start == SimulationTime(1000)
        stages = {entry.component_name for entry in log.entries}
        assert {"translator.invoke", "translator.parsed", "system3.declare"} <= stages
        invoke = [
            dict(log.reconstruct_value(e.deduplicated_data))
            for e in log.get_entries_by_component("translator.invoke")
        ]
        assert any("走向书桌" in i["inputs"][0] for i in invoke), "原始提示全量入日志"
        parsed = [
            dict(log.reconstruct_value(e.deduplicated_data))
            for e in log.get_entries_by_component("translator.parsed")
        ]
        assert any("a-lin-1" in o for p in parsed for o in p["outputs"])
        assert any('"checkpoints": [2000]' in p["outputs"][0] for p in parsed)
        assert any("组 g-lin-1" in o and "节点[2000]" in o for p in parsed for o in p["outputs"])
        report = scheduler.last_report
        assert report is not None
        data = step_data_from_report(report)
        assert "lin" in data.entity_actions

    asyncio.run(scenario())


def test_cancelled_translation_logs_context_and_releases_actor() -> None:
    """取消保留剧情因果现场，并让同角色后续调用继续使用原采样流。"""

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingSampler:
            async def sample(self, prompt: str, *, component: str) -> str:
                entered.set()
                await release.wait()
                return VALID_NEW

        collector = Collector()
        translator = ModelTranslator.shared(WaitingSampler(), tracer=collector)
        request = request_for((g1_action("a-1"),))
        task = asyncio.create_task(translator.translate(request))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        [error] = [r for r in collector.records if r.stage == "translator.error"]
        assert error.details["error_kind"] == "取消"
        assert error.error == "CancelledError: "
        assert error.response_id == ResponseId("r-1")
        assert "走向书桌" in error.inputs[0]
        assert not any(r.stage == "translator.parsed" for r in collector.records)
        release.set()
        result = await translator.translate(request)
        assert result.schedules[0].action_id == ActionId("a-1")

    asyncio.run(scenario())


def test_real_concordia_bridge_logs_translator_raw_io() -> None:
    """LanguageModelSampler + channel_for：translator 组件日志含原始 IO。"""

    async def scenario() -> None:
        from .test_concordia_sampling import StubModel

        walk_plan = '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向窗户"]}'
        schedule_json = (
            '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
            ' "start": 1000, "expected_end": 3000}], "modified_schedules": []}'
        )
        skips = ['{"reconsider": false, "note": "照旧"}'] * 8
        model = StubModel(
            [
                '{"reconsider": true, "note": "行动"}',
                '{"intention": "x", "abandon": []}',
                walk_plan,
                schedule_json,
                *skips,
            ]
        )
        log = structured_logging.SimulationLog()
        steps = iter(range(100))
        sampler = LanguageModelSampler(
            model,
            channel_for=lambda component: __import__(
                "sceneweave.integrations.concordia.tracing", fromlist=["component_logging_channel"]
            ).component_logging_channel(
                log,
                entity_name="lin",
                component_name=component,
                step=lambda: next(steps),
            ),
        )
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), sampler)],
            world=world,
            translator=ModelTranslator.shared(sampler),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        entries = log.get_entries_by_component("translator")
        assert entries, "真实采样组件按 translator 阶段记录"
        data = dict(log.reconstruct_value(entries[0].deduplicated_data))
        assert "Prompt" in data and "Response" in data
        assert "【World 客观上下文】" in data["Prompt"], "原始提示含客观上下文"

    asyncio.run(scenario())


def test_scoped_record_replay_reproduces_parallel_two_role_run(tmp_path) -> None:
    """角色+翻译器按作用域录制，重放零后端调用并复现并行运行。"""

    async def run_once(assembly) -> tuple:
        def role_sampler_factory(name: str) -> ScriptedSampler:
            return ScriptedSampler(
                {
                    "system1": [
                        '{"reconsider": true, "note": "' + name + ' 行动"}',
                        '{"reconsider": false, "note": "照旧"}',
                        '{"reconsider": false, "note": "照旧"}',
                        '{"reconsider": false, "note": "照旧"}',
                    ],
                    "system2": ['{"intention": "看窗户", "abandon": []}'],
                    "system3": [
                        '{"feasible": true, "maintain": [], "modify": [],'
                        ' "new_actions": ["走向窗户"]}'
                    ],
                }
            )

        def translator_sampler_factory(scope: str) -> ScriptedSampler:
            name = scope.split("-", 1)[1]
            return ScriptedSampler(
                {
                    "translator": [
                        '{"schedules": [{"action_id": "a-' + name + '-1",'
                        ' "commit_group_id": "g-' + name + '-1", "start": 1000,'
                        ' "expected_end": 3000}], "modified_schedules": []}'
                    ]
                }
            )

        roles_samplers, translator = assembly(role_sampler_factory, translator_sampler_factory)
        world = FakeSemanticWorld(["alice", "bob"])
        scheduler = Scheduler(
            roles=[ModelRole(rid(name), roles_samplers[name]) for name in ("alice", "bob")],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        return result, world

    async def scenario() -> None:
        directory = tmp_path / "recordings"
        with RecordingDirectory(
            lambda scope: ScriptedSampler(
                {
                    "system1": [
                        '{"reconsider": true, "note": "x"}',
                        '{"reconsider": false, "note": "y"}',
                        '{"reconsider": false, "note": "y"}',
                        '{"reconsider": false, "note": "y"}',
                    ],
                    "system2": ['{"intention": "看窗户", "abandon": []}'],
                    "system3": [
                        '{"feasible": true, "maintain": [], "modify": [],'
                        ' "new_actions": ["走向窗户"]}'
                    ],
                }
            ),
            directory,
            run_id="translator-dual",
        ) as role_dir:
            with RecordingDirectory(
                lambda scope: ScriptedSampler(
                    {
                        "translator": [
                            '{"schedules": [{"action_id": "a-'
                            + scope.split("-", 1)[1]
                            + '-1", "commit_group_id": "g-'
                            + scope.split("-", 1)[1]
                            + '-1", "start": 1000, "expected_end": 3000}],'
                            ' "modified_schedules": []}'
                        ]
                    }
                ),
                directory,
                run_id="translator-dual",
            ) as translator_rec_dir:

                def assemble(role_factory, translator_factory):
                    samplers = {name: role_dir.for_scope(name) for name in ("alice", "bob")}
                    translator = ModelTranslator(
                        lambda role: translator_rec_dir.for_scope(f"translator-{role}")
                    )
                    return samplers, translator

                result, world = await run_once(assemble)
        assert result.reason is RunStopReason.TIME_LIMIT

        role_replay = ReplayDirectory(directory)
        translator_replay = ReplayDirectory(directory)

        def assemble_replay(role_factory, translator_factory):
            samplers = {name: role_replay.for_scope(name) for name in ("alice", "bob")}
            translator = ModelTranslator(
                lambda role: translator_replay.for_scope(f"translator-{role}")
            )
            return samplers, translator

        replay_result, replay_world = await run_once(assemble_replay)
        role_replay.assert_fully_consumed()
        translator_replay.assert_fully_consumed()
        assert replay_result.reason is result.reason
        assert replay_result.completed_cycles == result.completed_cycles
        assert replay_world.facts == world.facts
        assert replay_world.adjudications == world.adjudications

    asyncio.run(scenario())


def test_scoped_factory_constructs_once_per_actor_with_replay(tmp_path) -> None:
    """同角色多次翻译：工厂只构造一次，请求不同也能严格全消费重放。"""

    async def scenario() -> None:
        from sceneweave.domain.messages import ModifiedAction

        first_raw = (
            '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
            ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}'
        )
        second_raw = (
            '{"schedules": [{"action_id": "a-lin-2", "commit_group_id": "g-lin-2",'
            ' "start": 3000, "expected_end": 7000, "checkpoints": [5000]}],'
            ' "modified_schedules": [{"action_id": "a-lin-1", "start": 1000,'
            ' "expected_end": 9000, "checkpoints": [2000, 8000]}]}'
        )
        retry_raw = (
            '{"schedules": [{"action_id": "a-lin-3", "commit_group_id": "g-lin-3",'
            ' "start": 4000, "expected_end": 8000}], "modified_schedules": []}'
        )
        construction_counts: dict[str, int] = {}

        def factory(scope: str) -> ScriptedSampler:
            construction_counts[scope] = construction_counts.get(scope, 0) + 1
            return ScriptedSampler(
                {
                    "translator": {
                        "translator-lin": [first_raw, second_raw, retry_raw],
                    }[scope]
                }
            )

        directory = tmp_path / "recordings"
        with RecordingDirectory(factory, directory, run_id="multi") as recorder:
            translator = ModelTranslator.scoped(recorder)
            # 第一次：新行动。
            first_request = request_for(
                (NewAction(ActionId("a-lin-1"), "首次", CommitGroupId("g-lin-1")),)
            )
            r1 = await translator.translate(first_request)
            # 第二次：不同请求（另一新行动 + 明确修改）。
            from sceneweave.domain.messages import RoleResponse

            second_request = TranslationRequest(
                response=RoleResponse(
                    CycleId("cycle-2"),
                    rid("lin"),
                    ResponseId("r-2"),
                    (
                        NewAction(ActionId("a-lin-2"), "再看", CommitGroupId("g-lin-2")),
                        ModifiedAction(ActionId("a-lin-1"), "延长"),
                    ),
                ),
                current_time=SimulationTime(1500),
                world_state_version=WorldStateVersion(4),
                arrangements=(arrangement_for("a-lin-1", 1000, 6000),),
                objective_context=ObjectiveContext(
                    WorldStateVersion(4), rid("lin"), ("位置 1.0 米",)
                ),
            )
            r2 = await translator.translate(second_request)
            # 第三次：同周期打回后的重试请求。
            retry_request = TranslationRequest(
                response=RoleResponse(
                    CycleId("cycle-2"),
                    rid("lin"),
                    ResponseId("r-3"),
                    (NewAction(ActionId("a-lin-3"), "重试", CommitGroupId("g-lin-3")),),
                ),
                current_time=SimulationTime(1500),
                world_state_version=WorldStateVersion(4),
                objective_context=ObjectiveContext(
                    WorldStateVersion(4), rid("lin"), ("位置 1.0 米",)
                ),
            )
            r3 = await translator.translate(retry_request)
        assert r1.schedules[0].action_id == ActionId("a-lin-1")
        assert r2.modified_schedules[0].action_id == ActionId("a-lin-1")
        assert r3.schedules[0].action_id == ActionId("a-lin-3")
        assert construction_counts == {"translator-lin": 1}, "工厂每角色只构造一次"

        replay_dir = ReplayDirectory(directory)
        replay_counts: dict[str, int] = {}

        def replay_factory(role: RoleId) -> object:
            scope = f"translator-{role}"
            replay_counts[scope] = replay_counts.get(scope, 0) + 1
            return replay_dir.for_scope(scope)

        replayer = ModelTranslator(replay_factory)  # type: ignore[arg-type]
        c1 = await replayer.translate(first_request)
        c2 = await replayer.translate(second_request)
        c3 = await replayer.translate(retry_request)
        assert c1 == r1 and c2 == r2 and c3 == r3, "重放逐字节复现不同请求的结果"
        replay_dir.assert_fully_consumed()
        assert replay_counts == {"translator-lin": 1}, "重放工厂也只构造一次"

    asyncio.run(scenario())


def test_scoped_actors_independent_and_prompt_no_contradiction(tmp_path) -> None:
    """跨角色缓存相互独立；提示区分新行动与修改的时间规则不自相矛盾。"""

    async def scenario() -> None:

        raw_shen = (
            '{"schedules": [{"action_id": "a-shen-1", "commit_group_id": "g-shen-1",'
            ' "start": 2500, "expected_end": 6500}], "modified_schedules": []}'
        )
        prompts: dict[str, list[str]] = {}

        class Recording(ScriptedSampler):
            def __init__(self, script: dict, name: str) -> None:
                super().__init__(script)
                self.name = name

            async def sample(self, prompt: str, *, component: str) -> str:
                prompts.setdefault(self.name, []).append(prompt)
                return await super().sample(prompt, component=component)

        directory = tmp_path / "recordings"
        with RecordingDirectory(
            lambda scope: Recording(
                {
                    "translator": [
                        '{"schedules": [{"action_id": "a-lin-1",'
                        ' "commit_group_id": "g-lin-1", "start": 2000,'
                        ' "expected_end": 6000}], "modified_schedules": []}'
                    ]
                }
                if "lin" in scope
                else {"translator": [raw_shen]},
                scope,
            ),
            directory,
            run_id="independent",
        ) as recorder:
            translator = ModelTranslator.scoped(recorder)
            await asyncio.gather(
                translator.translate(
                    request_for(
                        (NewAction(ActionId("a-lin-1"), "首次", CommitGroupId("g-lin-1")),),
                        role="lin",
                    )
                ),
                translator.translate(
                    request_for(
                        (NewAction(ActionId("a-shen-1"), "跟随", CommitGroupId("g-shen-1")),),
                        role="shen",
                    )
                ),
            )
        lin_prompt = prompts["translator-lin"][0]
        # 提示规则区分新行动与修改，不强迫已开始行动改写历史。
        assert "新行动的 start 晚于当前剧情时刻" in lin_prompt
        assert "已开始的行动保持原 start" in lin_prompt
        assert "expected_end" in lin_prompt and "max(start) < min(expected_end)" in lin_prompt
        assert "整数毫秒" in lin_prompt
        # 角色身份与既有行动的自然语言描述都进入提示。
        assert "角色 lin" in lin_prompt
        shen_prompt = prompts["translator-shen"][0]
        assert "角色 shen" in shen_prompt

    asyncio.run(scenario())


def test_prompt_shows_actor_and_arrangement_descriptions() -> None:
    """提示携带角色身份与既有行动自然语言描述。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"translator": [VALID_NEW.replace('"a-1"', '"a-9"').replace('"g-1"', '"g-9"')]}
        )
        translator = ModelTranslator.shared(sampler)
        request = request_for(
            (NewAction(ActionId("a-9"), "新尝试", CommitGroupId("g-9")),),
            arrangements=(
                ActionArrangement(
                    action_id=ActionId("a-old"),
                    phase=ActionPhase.STARTED,
                    description="原来在整理书桌",
                    start=SimulationTime(1000),
                    expected_end=SimulationTime(5000),
                    checkpoints=(SimulationTime(2000),),
                ),
            ),
        )
        await translator.translate(request)
        [(_, prompt)] = sampler.calls
        assert "角色 lin" in prompt
        assert "原来在整理书桌" in prompt, "既有行动自然语言描述进入提示"
        assert "1000→5000ms" in prompt

    asyncio.run(scenario())


def test_scheduler_started_action_modification_keeps_history() -> None:
    """已开始行动在 now>start 的明确修改：保留历史与进展、未来新节点。"""

    async def scenario() -> None:

        collector = Collector()
        raw_first = (
            '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
            ' "start": 1000, "expected_end": 4000, "checkpoints": [2000]}],'
            ' "modified_schedules": []}'
        )
        raw_modify = (
            '{"schedules": [], "modified_schedules": [{"action_id": "a-lin-1",'
            ' "start": 1000, "expected_end": 7000, "checkpoints": [2000, 5000]}]}'
        )
        role_sampler = ScriptedSampler(
            {
                "system1": [
                    '{"reconsider": true, "note": "行动"}',
                    '{"reconsider": false, "note": "照旧"}',
                    '{"reconsider": true, "note": "想走更久"}',
                ]
                + ['{"reconsider": false, "note": "照旧"}'] * 8,
                "system2": [
                    '{"intention": "慢慢走", "abandon": []}',
                    '{"intention": "走得更久", "abandon": []}',
                ],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}',
                    '{"feasible": true, "maintain": [],'
                    ' "modify": [{"id": "a-lin-1", "change": "走得更久"}],'
                    ' "new_actions": []}',
                ],
            }
        )
        translator_sampler = ScriptedSampler({"translator": [raw_first, raw_modify]})
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_sampler)],
            world=world,
            translator=ModelTranslator.shared(translator_sampler, tracer=collector),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-lin-1")]
        # 历史保留：start 与已过节点不变；未来按新安排。
        assert record.start == SimulationTime(1000)
        assert record.checkpoints == (SimulationTime(2000), SimulationTime(5000))
        assert record.expected_end == SimulationTime(7000)
        from sceneweave.domain.adjudication import ActionNodeKind

        checkpoint_times = [
            entry["simulation_time"]
            for entry in world.adjudications
            for node in entry["action_nodes"]
            if node.node is ActionNodeKind.CHECKPOINT
        ]
        assert checkpoint_times == [SimulationTime(2000), SimulationTime(5000)]
        # World 收到结构化修改（归属、原描述、明确 change、新旧安排）。
        modifications = [node for entry in world.adjudications for node in entry["modifications"]]
        assert modifications, "修改送交 World 裁决"
        node = modifications[0]
        assert node.role_id == "lin"
        assert node.change == "走得更久"
        assert node.old_start == SimulationTime(1000)
        assert node.new_expected_end == SimulationTime(7000)
        # 修改后的提示不自相矛盾：既要求保留历史又给出明确规则。
        prompts = [p for c, p in translator_sampler.calls]
        modify_prompt = prompts[1]
        assert "已开始的行动保持原 start" in modify_prompt
        assert "已到达的过程节点保持不变" in modify_prompt
        assert "走向书桌" in modify_prompt, "原行动描述进入修改提示"

    asyncio.run(scenario())


def test_skip_path_is_guarded_against_same_actor_concurrency() -> None:
    """门控挂起新行动翻译时，同角色的空/维持请求也被并发拒绝。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        sampler = ScriptedSampler({"translator": [VALID_NEW]}, gate=gate)
        translator = ModelTranslator.shared(sampler)
        first = asyncio.create_task(translator.translate(request_for((g1_action("a-1"),))))
        for _ in range(1000):
            if sampler.in_flight:
                break
            await asyncio.sleep(0)
        assert sampler.in_flight == 1
        skip = asyncio.create_task(
            translator.translate(request_for((maintain("a-keep"),), conditions=("条件",)))
        )
        with pytest.raises(ModelTranslatorError, match="串行"):
            await skip
        gate.set()
        await first
        # 占用释放后同角色可再次调用（含跳过路径）。
        empty = await translator.translate(request_for((), conditions=("条件",)))
        assert empty.schedules == ()

    asyncio.run(scenario())


def test_backend_and_factory_errors_propagate_with_trace() -> None:
    """后端异常按原类型传播；工厂异常同样传播且不缓存失败。"""

    async def scenario() -> None:
        collector = Collector()

        class Failing(ScriptedSampler):
            async def sample(self, prompt: str, *, component: str) -> str:
                raise TimeoutError("模型超时")

        translator = ModelTranslator.shared(Failing(), tracer=collector)
        with pytest.raises(TimeoutError):
            await translator.translate(request_for((g1_action("a-1"),)))
        [backend_error] = [r for r in collector.records if r.stage == "translator.error"]
        assert backend_error.error == "TimeoutError: 模型超时"
        assert backend_error.details["error_kind"] == "采样"
        assert "World 客观上下文" in backend_error.inputs[0]
        assert backend_error.role_id == rid("lin")
        assert backend_error.response_id == ResponseId("r-1")
        assert backend_error.simulation_time == SimulationTime(1000)
        assert backend_error.world_state_version == WorldStateVersion(3)
        assert not any(r.stage == "translator.parsed" for r in collector.records)

        factory_calls: list[str] = []

        def factory(role: RoleId) -> object:
            factory_calls.append(str(role))
            if len(factory_calls) == 1:
                raise RuntimeError("工厂失败")
            return ScriptedSampler({"translator": [VALID_NEW]})

        translator2 = ModelTranslator(factory, tracer=collector)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="工厂失败"):
            await translator2.translate(request_for((g1_action("a-1"),)))
        errors = [r for r in collector.records if r.stage == "translator.error"]
        assert len(errors) == 2
        assert errors[-1].error == "RuntimeError: 工厂失败"
        assert errors[-1].details["error_kind"] == "工厂"
        assert "走向书桌" in errors[-1].inputs[0]
        # 工厂失败不缓存：下一次调用重新构造并成功。
        result = await translator2.translate(request_for((g1_action("a-1"),)))
        assert result.schedules[0].action_id == ActionId("a-1")
        assert factory_calls == ["lin", "lin"]

    asyncio.run(scenario())


def test_nested_and_duplicate_errors_carry_trace_events() -> None:
    """嵌套结构错误、重复安排身份：错误事件留痕且为 ModelTranslatorError。"""

    async def scenario() -> None:
        collector = Collector()
        cases = [
            '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
            ' "start": 2000, "expected_end": 6000, "surprise": 1}],'
            ' "modified_schedules": []}',
            '{"schedules": [{"action_id": "a-1", "commit_group_id": "g-1",'
            ' "start": 2000, "expected_end": 6000},'
            ' {"action_id": "a-1", "commit_group_id": "g-1",'
            ' "start": 2000, "expected_end": 6000}], "modified_schedules": []}',
            '{"schedules": [], "modified_schedules": [{"action_id": "a-0",'
            ' "start": 1000, "expected_end": 5000},'
            ' {"action_id": "a-0", "start": 1000, "expected_end": 6000}]}',
        ]
        from sceneweave.domain.messages import ModifiedAction

        cases_with_actions = [
            (cases[0], (g1_action("a-1"),), ()),
            (cases[1], (g1_action("a-1"),), ()),
            (
                cases[2],
                (ModifiedAction(ActionId("a-0"), "改"),),
                (arrangement_for("a-0", 1000, 5000),),
            ),
        ]
        for raw, actions, arrangements in cases_with_actions:
            sampler = ScriptedSampler({"translator": [raw]})
            translator = ModelTranslator.shared(sampler, tracer=collector)
            request = request_for(actions, arrangements=arrangements)
            with pytest.raises(ModelTranslatorError):
                await translator.translate(request)
            last_error = [r for r in collector.records if r.stage == "translator.error"][-1]
            assert last_error.outputs == (raw,)
            assert "World 客观上下文" in last_error.inputs[0]
            assert last_error.response_id == request.response.response_id
        errors = [r for r in collector.records if r.stage == "translator.error"]
        assert len(errors) == 3, "每个错误恰有一个含完整原始输入输出的错误事件"

    asyncio.run(scenario())


def test_tracer_failure_on_success_propagates_and_error_keeps_original() -> None:
    """成功链追踪失败传播；已有原异常时错误留痕失败不掩盖原异常。"""

    async def scenario() -> None:
        from sceneweave.domain.tracing import TurnTracer

        class AlwaysBroken(TurnTracer):
            """完全故障：成功链追踪失败必须传播。"""

            def trace(self, record) -> None:  # type: ignore[no-untyped-def]
                raise RuntimeError("追踪器故障")

        class ErrorBroken(TurnTracer):
            """仅在错误事件上故障：检验原异常优先于追踪失败。"""

            def trace(self, record) -> None:  # type: ignore[no-untyped-def]
                if record.stage == "translator.error":
                    raise RuntimeError("追踪器故障")

        sampler = ScriptedSampler({"translator": [VALID_NEW]})
        translator = ModelTranslator.shared(sampler, tracer=AlwaysBroken())
        with pytest.raises(RuntimeError, match="追踪器故障"):
            await translator.translate(request_for((g1_action("a-1"),)))
        # 同一实例在追踪器恢复后可再次调用（占用已被 finally 释放）。
        recovered = ModelTranslator.shared(sampler)
        ok = await recovered.translate(request_for((g1_action("a-1"),)))
        assert ok.schedules[0].action_id == ActionId("a-1")

        # 原异常路径：解析失败 + 错误事件追踪也坏 → 原解析异常优先。
        class BadOutput(ScriptedSampler):
            async def sample(self, prompt: str, *, component: str) -> str:
                return "不是 JSON"

        translator2 = ModelTranslator.shared(BadOutput(), tracer=ErrorBroken())
        with pytest.raises(ModelTranslatorError, match="不是合法 JSON") as caught:
            await translator2.translate(request_for((g1_action("a-1"),)))
        assert any("错误日志写入失败" in note for note in caught.value.__notes__)

    asyncio.run(scenario())
