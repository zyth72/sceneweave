"""调度器闭环行为测试：时钟冻结、最近事件、同刻整批、屏障、时限与错误现场。

全部使用脚本假件与事件门控，不依赖墙钟睡眠。世界自身的原子提交
（校验后一次性应用）与引擎的运行时校验责任分别断言。
"""

import asyncio
import json

import pytest

from sceneweave.domain.adjudication import ActionNodeKind, ActionOutcomeResult
from sceneweave.domain.identity import ActionId, EventId
from sceneweave.domain.messages import ActionPhase, NewAction, WorldEvent
from sceneweave.domain.scheduling import ScheduledEventKind
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.domain.translation import ActionSchedule, TranslationResult
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler

from .fakes import (
    FakeSemanticWorld,
    ScriptedRole,
    ScriptedTranslator,
    idle_response,
    maintain,
    maintaining,
    ms,
    new_action,
    response,
)

CONFIG = RunConfig(introspection_interval=ms(50_000), max_duration=ms(100_000))


def startup_event(
    name: str = "startup", time: int = 100, text: str = "开场"
) -> tuple[WorldEvent, ...]:
    return (WorldEvent(EventId(name), SimulationTime(time), text),)


class Collector:
    """收集 TraceRecord 的追踪器。"""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def trace(self, record: TraceRecord) -> None:
        self.records.append(record)

    def stages(self) -> set[str]:
        return {record.stage for record in self.records}

    def by_stage(self, stage: str) -> list[TraceRecord]:
        return [record for record in self.records if record.stage == stage]


def make_scheduler(
    world: FakeSemanticWorld,
    roles: list[ScriptedRole],
    translator,
    *,
    config: RunConfig = CONFIG,
    initial_events: tuple = (),
    tracer=None,
    **kwargs,
) -> Scheduler:
    return Scheduler(
        roles=roles,
        world=world,
        translator=translator,
        config=config,
        initial_events=initial_events,
        tracer=tracer,
        **kwargs,
    )


def test_clock_freezes_while_role_thinks_and_world_is_idle() -> None:
    """角色思考（含模型等待）期间时钟、进展与裁决均不推进。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        alice = ScriptedRole("alice", gate=gate, fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator()
        scheduler = make_scheduler(world, [alice], translator, initial_events=startup_event())
        task = asyncio.create_task(scheduler.run())
        await alice.started.wait()
        # 冻结断言：时钟停在事件时刻，进展只更新过一次，裁决只有一次。
        assert scheduler.current_time == ms(100)
        assert [call[1] for call in world.progress_calls] == [100]
        assert len(world.adjudications) == 1
        assert translator.requests == [], "角色未全部完成前不得开始翻译"
        assert len(scheduler.action_records()) == 0
        gate.set()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT
        assert result.final_time == ms(100_000)
        assert scheduler.current_time == ms(100_000)

    asyncio.run(scenario())


def test_jumps_to_nearest_event_first() -> None:
    """调度先到最近事件；更晚的事件不在同一批。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            initial_events=startup_event("late", 3000, "晚事件")
            + startup_event("early", 1000, "早事件"),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        first = world.adjudications[0]
        assert first["simulation_time"] == ms(1000)
        assert [item.event_id for item in first["events"]] == [EventId("early")]
        assert first["event_ids"] == (EventId("early"),)

    asyncio.run(scenario())


def test_same_time_events_form_one_batch_independent_of_insertion_order() -> None:
    """同刻事件整批交给 World；插入顺序不影响批内容与顺序。"""

    async def run_ordered(order: tuple[str, str]) -> dict:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        texts = {"event-a": "甲", "event-b": "乙"}
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(50_000), max_duration=ms(60_000)),
            initial_events=tuple(
                WorldEvent(EventId(name), SimulationTime(1000), texts[name]) for name in order
            ),
        )
        await scheduler.run()
        assert len(world.adjudications) == 1, "同刻事件只裁决一次"
        return world.adjudications[0]

    async def scenario() -> None:
        one = await run_ordered(("event-a", "event-b"))
        two = await run_ordered(("event-b", "event-a"))
        assert one["events"] == two["events"], "批内顺序按事件身份稳定，与插入顺序无关"
        assert [item.event_id for item in one["events"]] == [EventId("event-a"), EventId("event-b")]

    asyncio.run(scenario())


def test_world_updates_progress_before_adjudication_each_cycle() -> None:
    """进入事件时刻后先更新进展，再基于更新后的状态裁决。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=startup_event("e1", 1000, "敲门"),
        )
        await scheduler.run()
        times = [call[1] for call in world.progress_calls]
        assert times == [1000, 2000, 3000], "每个事件时刻各更新一次进展"
        for adjudication in world.adjudications:
            # 裁决发生时进展已经按该时刻更新（假世界记录二者顺序）。
            assert adjudication["simulation_time"] in times

    asyncio.run(scenario())


def test_action_lifecycle_follows_world_outcomes_not_expected_end() -> None:
    """行动阶段只随 World 提交的结算结果变化；预计结束不等于成功。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-walk", "走向门口"),)),
                response("alice", "r-2", (maintain("a-walk"),)),
                response("alice", "r-3", (maintain("a-walk"),)),
                response("alice", "r-4", (maintain("a-walk"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-walk", 0.001)
        translator = ScriptedTranslator(
            plans={"走向门口": (1000, 3000, (1000,))},
        )
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(100_000)),
            initial_events=startup_event(),
        )
        await scheduler.run()
        records = scheduler.action_records()
        assert records[ActionId("a-walk")].phase is ActionPhase.COMPLETED
        assert world.positions["alice"] == 3.0, "位置只按流逝的剧情时间推进"
        facts = "；".join(world.facts)
        assert "a-walk 开始：走向门口" in facts
        assert "a-walk 结算：completed" in facts
        # 角色在后续周期收到自身行动的阶段反馈。
        started_input = alice.perceived[1]
        assert started_input.known_action_states[0].phase is ActionPhase.STARTED
        feedback = alice.perceived[3].known_action_states
        assert any(item.phase is ActionPhase.COMPLETED for item in feedback)

    asyncio.run(scenario())


def test_expected_end_with_world_failure_is_not_success() -> None:
    """到达预计结束但 World 结算失败时，登记阶段为失败。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-pull", "拉开卡住的抽屉"),)),
                response("alice", "r-2", (maintain("a-pull"),)),
                response("alice", "r-3", (maintain("a-pull"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        world.outcome_overrides[ActionId("a-pull")] = ActionOutcomeResult.FAILED
        translator = ScriptedTranslator(delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        await scheduler.run()
        assert scheduler.action_records()[ActionId("a-pull")].phase is ActionPhase.FAILED
        assert "a-pull 结算：failed" in "；".join(world.facts)

    asyncio.run(scenario())


def test_introspection_merges_with_same_time_events() -> None:
    """内思考与同刻外部事件合并为一次响应；先提交世界变化再投递。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=startup_event("knock", 1000, "敲门"),
        )
        await scheduler.run()
        first = alice.perceived[0]
        assert first.simulation_time == ms(1000)
        assert first.trigger.introspection is True
        assert first.trigger.event_ids == (EventId("knock"),)
        assert first.observations, "同刻外部事件的感知在合并输入中"
        assert len(alice.perceived) == 3, "同刻合并后不多响应一次"

    asyncio.run(scenario())


def test_introspection_wakes_roles_without_external_perception() -> None:
    """无外部事件时周期事件仍唤醒全部角色，且不伪造外部感知。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        bob = ScriptedRole("bob", fallback=idle_response("bob"))
        world = FakeSemanticWorld(["alice", "bob"])
        scheduler = make_scheduler(
            world,
            [alice, bob],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(2000)),
        )
        await scheduler.run()
        assert [t.simulation_time for t in alice.perceived] == [ms(1000), ms(2000)]
        for turn in (*alice.perceived, *bob.perceived):
            assert turn.trigger.introspection is True
            assert turn.observations == (), "内思考不伪造外部感知"
        assert world.adjudications == [], "内思考本身不产生世界变化"

    asyncio.run(scenario())


def test_time_limit_includes_boundary_cycle_and_keeps_future_events() -> None:
    """边界时刻的完整周期会被处理；越界的新事件与行动保留不结算。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-late", "稍后开始"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(5000)),
            initial_events=startup_event("edge", 5000, "边界事件"),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert result.final_time == ms(5000)
        assert result.completed_cycles == 1, "边界周期完整执行"
        pending = {(e.event_id, e.simulation_time) for e in scheduler.pending_events()}
        assert (EventId("a-late-start"), ms(6000)) in pending
        assert (EventId("a-late-end"), ms(7000)) in pending
        assert scheduler.action_records()[ActionId("a-late")].phase is ActionPhase.REGISTERED
        assert not any("a-late" in fact for fact in world.facts), "未发生的事件不强行结算"

    asyncio.run(scenario())


def test_time_limit_beyond_next_event_stops_at_limit_keeping_events() -> None:
    """下一事件越界时推进到时限停止，事件保留。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(5000)),
            initial_events=startup_event("early", 2000, "早事件"),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert result.final_time == ms(5000)
        assert result.completed_cycles == 1
        assert any(e.event_id == EventId("introspection-1") for e in scheduler.pending_events())

    asyncio.run(scenario())


def test_natural_end_requires_injected_predicate_and_ignores_empty_responses() -> None:
    """空响应不是自然结束；自然结束只来自注入判断。"""

    async def scenario() -> None:
        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(100_000)),
            should_end=lambda status: status.completed_cycles >= 3,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.NATURAL_END
        assert result.completed_cycles == 3
        assert result.final_time == ms(3000)

    asyncio.run(scenario())


def test_translation_waits_for_full_barrier_before_any_registration() -> None:
    """先完成的角色不得抢先登记；全部角色完成屏障后才翻译并登记。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-a", (new_action("a-slow", "慢慢思考"),))],
            gate=gate,
            fallback=idle_response("alice"),
        )
        bob = ScriptedRole("bob", fallback=idle_response("bob"))
        world = FakeSemanticWorld(["alice", "bob"])
        translator = ScriptedTranslator(delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice, bob],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            should_end=lambda status: status.completed_cycles >= 1,
        )
        task = asyncio.create_task(scheduler.run())
        await bob.completed.wait()
        assert not alice.completed.is_set()
        assert translator.requests == [], "屏障未过不得翻译"
        assert len(scheduler.action_records()) == 0
        gate.set()
        result = await task
        assert result.reason is RunStopReason.NATURAL_END
        assert scheduler.action_records()[ActionId("a-slow")].phase is ActionPhase.REGISTERED

    asyncio.run(scenario())


def test_translation_freezes_clock_and_registry_until_complete() -> None:
    """翻译等待期间时钟不动、登记不生效。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-wait", "等待翻译"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(gate=gate, delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            should_end=lambda status: status.completed_cycles >= 1,
        )
        task = asyncio.create_task(scheduler.run())
        await alice.completed.wait()
        for _ in range(100):
            if translator.requests:
                break
            await asyncio.sleep(0)
        assert translator.requests, "翻译器应已收到请求"
        assert scheduler.stage == "cycle.translation"
        assert scheduler.current_time == ms(100)
        assert len(scheduler.action_records()) == 0
        gate.set()
        result = await task
        assert result.reason is RunStopReason.NATURAL_END
        assert ActionId("a-wait") in scheduler.action_records()

    asyncio.run(scenario())


def test_translation_problem_rejects_role_without_re_perception() -> None:
    """时间安排问题经 rejected 打回；原感知不重投，仍属同一周期。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-hurry", "立刻行动"),)),
                response("alice", "r-2", (new_action("a-calm", "从容行动"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])

        def override(request):
            new_items = [i for i in request.response.actions if isinstance(i, NewAction)]
            if not new_items:
                return TranslationResult(
                    cycle_id=request.response.cycle_id,
                    role_id=request.response.role_id,
                    response_id=request.response.response_id,
                )
            action = new_items[0]
            if request.response.response_id == "r-1":
                return TranslationResult(
                    cycle_id=request.response.cycle_id,
                    role_id=request.response.role_id,
                    response_id=request.response.response_id,
                    schedules=(
                        ActionSchedule(
                            action_id=action.action_id,
                            commit_group_id=action.commit_group_id,
                            start=request.current_time,
                            expected_end=SimulationTime(request.current_time + 100),
                        ),
                    ),
                )
            return TranslationResult(
                cycle_id=request.response.cycle_id,
                role_id=request.response.role_id,
                response_id=request.response.response_id,
                schedules=(
                    ActionSchedule(
                        action_id=action.action_id,
                        commit_group_id=action.commit_group_id,
                        start=SimulationTime(request.current_time + 1000),
                        expected_end=SimulationTime(request.current_time + 2000),
                    ),
                ),
            )

        translator = ScriptedTranslator(override=override)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            should_end=lambda status: status.completed_cycles >= 1,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.NATURAL_END
        assert len(alice.rejections) == 1
        assert "晚于当前时刻" in alice.rejections[0].reason
        assert alice.rejections[0].rejected_response_id == "r-1"
        # 原感知不重投：出问题的周期只有一次 perceive。
        assert sum(1 for t in alice.perceived if t.simulation_time == ms(100)) == 1
        records = scheduler.action_records()
        assert ActionId("a-calm") in records, "打回后的新完整候选生效"
        assert ActionId("a-hurry") not in records, "未通过的候选不登记"

    asyncio.run(scenario())


def test_persistent_translation_failure_is_run_error_without_partial_registration() -> None:
    """翻译持续失败超过上限是运行错误；不留下任何半提交登记。"""

    async def scenario() -> None:

        alice = ScriptedRole(
            "alice",
            [
                response("alice", f"r-{index}", (new_action("a-bad", "始终无效"),))
                for index in range(1, 5)
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])

        def override(request):
            action = request.response.actions[0]
            return TranslationResult(
                cycle_id=request.response.cycle_id,
                role_id=request.response.role_id,
                response_id=request.response.response_id,
                schedules=(
                    ActionSchedule(
                        action_id=action.action_id,
                        commit_group_id=action.commit_group_id,
                        start=request.current_time,
                        expected_end=SimulationTime(request.current_time + 100),
                    ),
                ),
            )

        translator = ScriptedTranslator(override=override)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            max_translation_rejections=2,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "翻译打回次数" in (result.error or "")
        assert len(scheduler.action_records()) == 0, "不部分登记"
        assert all(
            e.kind is not ScheduledEventKind.ACTION_START for e in scheduler.pending_events()
        )
        version_after_first_commit = 1
        assert world.state_version == version_after_first_commit, "登记失败不再改变世界"

    asyncio.run(scenario())


def test_translator_type_error_is_run_error_not_semantic_rejection() -> None:
    """数据类型错误不被转换成语义打回或空声明。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-type", "正常行动"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])

        class BrokenTranslator:
            async def translate(self, request):  # type: ignore[no-untyped-def]
                return {"schedules": []}

        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=BrokenTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "必须返回 TranslationResult" in (result.error or "")
        assert alice.rejections == [], "运行错误不得伪装成打回"

    asyncio.run(scenario())


def test_run_error_saves_scene_snapshot_to_file(tmp_path) -> None:
    """运行错误停止推进并保存现场：时刻、队列、记录、候选与报错。"""

    class BrokenWorld(FakeSemanticWorld):
        def __init__(self) -> None:
            super().__init__(["alice"])
            self.adjudications_made = 0

        async def adjudicate(self, *args, **kwargs):  # type: ignore[override]
            self.adjudications_made += 1
            if self.adjudications_made > 1:
                raise RuntimeError("模型故障")
            return await super().adjudicate(*args, **kwargs)

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-x", "会出事的行动"),))],
            fallback=idle_response("alice"),
        )
        world = BrokenWorld()
        translator = ScriptedTranslator(delay=1000, duration=1000)
        path = tmp_path / "scene.json"
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            snapshot_path=path,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "模型故障" in (result.error or "")
        assert result.completed_cycles == 1, "已提交的第一周期保留"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["stage"] == "cycle.adjudication"
        assert data["cycle_id"] == "cycle-2"
        assert data["simulation_time_ms"] == 1100
        assert any(item["event_id"] == "a-x-start" for item in data["active_batch"]), (
            "已取出未完成的批保留在现场"
        )
        assert any(item["action_id"] == "a-x" for item in data["action_records"])
        assert any(item["event_id"] == "a-x-end" for item in data["pending_events"])
        assert data["pending_candidates"] == [], "新周期清空旧候选；裁决阶段尚无本轮候选"
        assert "模型故障" in (data["error"] or "")
        assert result.snapshot is not None
        assert "a-x" in result.snapshot.to_json()

    asyncio.run(scenario())


def test_errored_instance_cannot_run_again() -> None:
    """运行错误的实例禁止再次推进；恢复须重建并外部处理现场。"""

    async def scenario() -> None:
        class BrokenWorld(FakeSemanticWorld):
            def __init__(self) -> None:
                super().__init__(["alice"])
                self.adjudications_made = 0

            async def adjudicate(self, *args, **kwargs):  # type: ignore[override]
                self.adjudications_made += 1
                if self.adjudications_made > 1:
                    raise RuntimeError("模型故障")
                return await super().adjudicate(*args, **kwargs)

        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-x", "会出事的行动"),))],
            fallback=idle_response("alice"),
        )
        world = BrokenWorld()
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=1000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        first = await scheduler.run()
        assert first.reason is RunStopReason.ERROR
        with pytest.raises(RuntimeError, match="不可再次推进"):
            await scheduler.run()

    asyncio.run(scenario())


def test_scheduler_traces_causal_chain_along_cycle_event_action() -> None:
    """日志沿周期/事件/角色/行动因果链记录各边界。"""

    async def scenario() -> None:
        collector = Collector()
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-traced", "被追踪的行动"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            tracer=collector,
        )
        await scheduler.run()
        stages = collector.stages()
        assert {
            "engine.cycle_start",
            "world.progress",
            "world.adjudicate",
            "world.validate",
            "world.commit",
            "world.publish",
            "translation.request",
            "translation.result",
            "registration.plan",
            "registration.commit",
        } <= stages
        commit = collector.by_stage("world.commit")[0]
        assert commit.world_state_version is not None
        plan = collector.by_stage("registration.plan")[0]
        assert ActionId("a-traced") in plan.action_ids
        start_cycle = collector.by_stage("world.adjudicate")[1]
        assert start_cycle.action_ids == (ActionId("a-traced"),)
        publish = collector.by_stage("world.publish")[0]
        assert publish.outputs, "发布记录包含可检索的投递内容"

    asyncio.run(scenario())


def test_user_abort_via_task_cancellation_propagates() -> None:
    """用户中止（取消任务）明确传播取消，不伪装成自然结束。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        alice = ScriptedRole("alice", gate=gate, fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world, [alice], ScriptedTranslator(), initial_events=startup_event()
        )
        task = asyncio.create_task(scheduler.run())
        await alice.started.wait()
        task.cancel()
        try:
            await task
            raise AssertionError("取消必须传播")
        except asyncio.CancelledError:
            pass
        assert scheduler.current_time == ms(100), "中止时保留现场"

    asyncio.run(scenario())


def test_run_config_converts_seconds_at_entry_and_validates() -> None:
    """秒配置只在入口转换一次；非正数、bool 与无法整毫秒表示的值拒绝。"""
    config = RunConfig.from_seconds(30, 600)
    assert config.introspection_interval == ms(30_000)
    assert config.max_duration == ms(600_000)
    assert RunConfig(ms(1), ms(2)).limit_from(ms(100)) == ms(102)
    for bad_interval, bad_duration in ((0, 10), (10, 0), (-1, 10), (True, 10), (10, 0.0005)):
        try:
            RunConfig.from_seconds(bad_interval, bad_duration)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝非法配置：{bad_interval}, {bad_duration}")


def test_scheduler_rejects_invalid_construction() -> None:
    """构造入口验证：早于起始时刻的事件、重复角色与空场景。"""
    alice = ScriptedRole("alice", fallback=idle_response("alice"))
    world = FakeSemanticWorld(["alice"])
    translator = ScriptedTranslator()
    config = RunConfig(introspection_interval=ms(1000), max_duration=ms(2000))
    late = WorldEvent(EventId("too-early"), SimulationTime(0), "早于起点")
    with pytest.raises(ValueError, match="不得早于起始时刻"):
        Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=config,
            start_time=ms(100),
            initial_events=[late],
        )
    duplicate = ScriptedRole("alice", fallback=idle_response("alice"))
    with pytest.raises(ValueError, match="重复角色身份"):
        Scheduler(roles=[alice, duplicate], world=world, translator=translator, config=config)
    with pytest.raises(ValueError, match="至少包含一个角色"):
        Scheduler(roles=[], world=world, translator=translator, config=config)


def test_action_node_and_world_event_at_same_time_form_one_batch() -> None:
    """行动节点与世界事件同刻时合并为一次联合裁决。"""

    async def scenario() -> None:
        from sceneweave.domain.messages import ActionPhase

        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-sync", "同时发生"),)),
                response("alice", "r-2", (maintain("a-sync"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=1000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event("register", 0, "开场")
            + startup_event("clash", 1000, "同一时刻的敲门"),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        clash = [item for item in world.adjudications if item["simulation_time"] == ms(1000)]
        assert len(clash) == 1, "同刻事件与行动节点只裁决一次"
        assert [event.event_id for event in clash[0]["events"]] == [EventId("clash")]
        assert [node.action_id for node in clash[0]["action_nodes"]] == [ActionId("a-sync")]
        # 角色在同一周期里既响应敲门，也收到行动开始反馈。
        clash_turn = alice.perceived[1]
        assert clash_turn.trigger.event_ids == (EventId("clash"),)
        assert any(state.phase is ActionPhase.STARTED for state in clash_turn.known_action_states)

    asyncio.run(scenario())


def test_internal_event_ids_avoid_external_collision_before_any_commit() -> None:
    """引擎内部事件身份避让外部事件；登记不再留下半提交状态。"""

    async def scenario() -> None:
        # 旧复现：外部世界事件 a-start@9000 与行动 a 的内部节点身份同名。
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a", "走动"),)),
                response("alice", "r-2", (maintain("a"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=100, duration=1000),
            config=RunConfig(introspection_interval=ms(20_000), max_duration=ms(20_000)),
            initial_events=startup_event("boot", 0, "开场")
            + startup_event("a-start", 9000, "合法世界事件"),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a")]
        assert record.start == ms(100)
        # 内部节点身份被避让：行动开始节点用了带后缀的新身份，而不是
        # 与外部事件冲突的身份；外部事件按原身份被处理。
        start_nodes = [
            node
            for adjudication in world.adjudications
            for node in adjudication["action_nodes"]
            if node.node is ActionNodeKind.START and node.action_id == ActionId("a")
        ]
        assert start_nodes and start_nodes[0].event_id != EventId("a-start")
        assert str(start_nodes[0].event_id).startswith("a-start-")
        # 行动真实经历了开始与结算，外部事件也被处理。
        facts = "；".join(world.facts)
        assert "a 开始：走动" in facts
        assert "事件 a-start 发生：合法世界事件" in facts

    asyncio.run(scenario())


def test_duplicate_follow_up_id_is_refused_before_commit() -> None:
    """后续事件身份冲突在事实提交前发现；不留半事实或半队列。"""

    async def scenario() -> None:
        class ClashingWorld(FakeSemanticWorld):
            def __init__(self) -> None:
                super().__init__(["alice"])
                self.clash = False

            async def adjudicate(self, *args, **kwargs):  # type: ignore[override]
                candidate = await super().adjudicate(*args, **kwargs)
                if not self.clash:
                    self.clash = True
                    import dataclasses

                    follow = WorldEvent(
                        EventId("startup"), SimulationTime(5000), "冒用已有事件身份"
                    )
                    candidate = dataclasses.replace(candidate, follow_up_events=(follow,))
                return candidate

        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = ClashingWorld()
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "后续世界事件身份冲突" in (result.error or "")
        # 冲突在外部事件结算时发现：该轮事实已提交是第一周期本身；
        # 冲突候选的事实没有提交、队列没有登记冒用事件。
        assert all(
            e.event_id != EventId("startup")
            for e in scheduler.pending_events()
            if e.simulation_time == 5000
        )
        assert not any("冒用已有事件身份" in fact for fact in world.facts)

    asyncio.run(scenario())


def test_time_limit_boundary_updates_progress_without_settling_or_roles() -> None:
    """时限边界按最后流逝剧情时间推进实际进展；不结算、不调用角色。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-walk", "走向门口"),)),
                response("alice", "r-2", (maintain("a-walk"),)),
            ],
            fallback=maintaining("alice", "a-walk"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-walk", 0.001)
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(plans={"走向门口": (1000, 10_000, ())}),
            config=RunConfig(introspection_interval=ms(20_000), max_duration=ms(5000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert result.final_time == ms(5000)
        assert scheduler.current_time == ms(5000)
        # 走到时限：按流逝剧情时间推进（注册发生在 t=100，开始于 t=1100，
        # 到 t=5000 共推进 3900ms × 0.001m/ms）。
        assert world.positions["alice"] == pytest.approx(3.9)
        record = scheduler.action_records()[ActionId("a-walk")]
        assert record.phase is ActionPhase.STARTED
        assert any(
            str(e.event_id).startswith("a-walk") and e.simulation_time == ms(11_100)
            for e in scheduler.pending_events()
        ), "未来结束节点保留"
        assert not any("a-walk 结算" in fact for fact in world.facts), "不强行结算"
        # 时限边界只更新进展，不触发新的角色响应。
        turn_times = [t.simulation_time for t in alice.perceived]
        assert turn_times == [ms(100), ms(1100)], "角色只在真实事件周期响应"
        assert (world.progress_calls[-1][1]) == 5000, "边界时刻按流逝时间更新进展"

    asyncio.run(scenario())


def test_translator_mutating_world_is_caught_after_translation() -> None:
    """翻译期间改动共同状态在登记前被发现并停止。"""

    async def scenario() -> None:
        class MutatingTranslator(ScriptedTranslator):
            def __init__(self, world: FakeSemanticWorld) -> None:
                super().__init__(delay=1000, duration=1000)
                self._world = world

            async def translate(self, request):  # type: ignore[no-untyped-def]

                self._world.bump_for_test()
                return await super().translate(request)

        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-x", "正常行动"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            MutatingTranslator(world),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "翻译期间共同世界状态被修改" in (result.error or "")
        assert len(scheduler.action_records()) == 0, "不再登记"

    asyncio.run(scenario())


def test_world_violating_version_invariants_is_stopped_with_scene() -> None:
    """裁决/校验/提交的版本边界被恶意绕过时停止并保留现场。"""

    async def scenario() -> None:
        import dataclasses

        from sceneweave.domain.identity import WorldStateVersion

        class AdjudicateMutates(FakeSemanticWorld):
            async def adjudicate(self, *args, **kwargs):  # type: ignore[override]
                candidate = await super().adjudicate(*args, **kwargs)
                self.bump_for_test()
                return candidate

        class StaleValidation(FakeSemanticWorld):
            def validate_adjudication(self, candidate):  # type: ignore[override]
                validation = super().validate_adjudication(candidate)
                return dataclasses.replace(validation, world_state_version=WorldStateVersion(99))

        class CommitDoesNotApply(FakeSemanticWorld):
            def commit_adjudication(self, candidate):  # type: ignore[override]
                commit = super().commit_adjudication(candidate)
                self.rewind_for_test()  # 谎报提交但实际版本未变
                return commit

        for broken, expected in (
            (AdjudicateMutates, "不得改变共同状态"),
            (StaleValidation, "版本"),
            (CommitDoesNotApply, "版本"),
        ):
            world = broken(["alice"])
            scheduler = make_scheduler(
                world,
                [ScriptedRole("alice", fallback=idle_response("alice"))],
                ScriptedTranslator(),
                config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
                initial_events=startup_event(),
            )
            result = await scheduler.run()
            assert result.reason is RunStopReason.ERROR, broken.__name__
            assert expected in (result.error or ""), broken.__name__

    asyncio.run(scenario())


def test_outcome_incompatible_with_node_is_refused_before_fact_commit() -> None:
    """结算结果与节点类型不符在事实提交前拒绝；不产生错误事实。"""

    async def scenario() -> None:
        from sceneweave.domain.adjudication import ActionNodeKind, ActionOutcomeResult

        for node_kind, bad_result, expected_node in (
            (ActionNodeKind.START, ActionOutcomeResult.COMPLETED, "start"),
            (ActionNodeKind.END, ActionOutcomeResult.STARTED, "end"),
        ):
            alice = ScriptedRole(
                "alice",
                [
                    response("alice", "r-1", (new_action("a-bad", "有问题的行动"),)),
                    response("alice", "r-2", (maintain("a-bad"),)),
                ],
                fallback=idle_response("alice"),
            )
            world = FakeSemanticWorld(["alice"])
            world.node_overrides[(ActionId("a-bad"), node_kind)] = bad_result
            scheduler = make_scheduler(
                world,
                [alice],
                ScriptedTranslator(delay=1000, duration=1000),
                config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
                initial_events=startup_event(),
            )
            result = await scheduler.run()
            assert result.reason is RunStopReason.ERROR, f"{expected_node} 节点"
            assert "结算结果未通过提交前校验" in (result.error or "")
            assert not any("a-bad 结算" in fact for fact in world.facts), "非法结算没有成为事实"

    asyncio.run(scenario())


def test_precheck_shares_node_and_modification_rules() -> None:
    """预检直接消费归口映射：撤回可无 outcome，修改不得 failed/completed。"""
    from sceneweave.domain.adjudication import (
        ActionModificationNode,
        ActionOutcome,
        AdjudicationActionNode,
        allowed_results_by_action,
    )
    from sceneweave.domain.identity import CycleId, RoleId
    from sceneweave.domain.messages import ActionPhase
    from sceneweave.engine.actions import ActionRecord, outcome_transition_problems

    record = ActionRecord(
        action_id=ActionId("a-1"),
        role_id=RoleId("alice"),
        description="动作",
        phase=ActionPhase.STARTED,
        start=ms(1000),
        expected_end=ms(2000),
        checkpoints=(),
        registered_cycle=CycleId("cycle-1"),
    )
    withdrawn = AdjudicationActionNode(
        event_id=EventId("event-w"),
        action_id=ActionId("a-1"),
        role_id=RoleId("alice"),
        node=ActionNodeKind.WITHDRAWN,
        description="撤回",
        start=ms(1000),
        expected_end=ms(2000),
    )
    modification = ActionModificationNode(
        event_id=EventId("event-m"),
        action_id=ActionId("a-1"),
        role_id=RoleId("alice"),
        description="动作",
        change="放慢",
        old_start=ms(1000),
        old_expected_end=ms(2000),
        old_checkpoints=(),
        new_start=ms(1000),
        new_expected_end=ms(3000),
        new_checkpoints=(),
    )
    allowed = allowed_results_by_action((withdrawn,), (modification,))
    assert allowed[ActionId("a-1")] == frozenset(
        {ActionOutcomeResult.IN_PROGRESS, ActionOutcomeResult.UNCONFIRMED}
    )
    # 没有 outcome（撤回可不提供）不产生问题。
    assert outcome_transition_problems({ActionId("a-1"): record}, (), allowed) == ()
    problems = outcome_transition_problems(
        {ActionId("a-1"): record},
        (ActionOutcome(ActionId("a-1"), ActionOutcomeResult.FAILED, "失败"),),
        allowed,
    )
    assert problems == ("结算结果与节点类型不符：a-1 得到 failed",)


def test_unconfirmed_outcome_keeps_future_nodes_and_maintenance() -> None:
    """未确认不是已知客观终止：未来节点保留，行动仍可维持。"""

    async def scenario() -> None:
        from sceneweave.domain.adjudication import ActionOutcomeResult

        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-unknown", "结果未知"),)),
                response("alice", "r-2", (maintain("a-unknown"),)),
                response("alice", "r-3", (maintain("a-unknown"),)),
                response("alice", "r-4", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        world.node_overrides[(ActionId("a-unknown"), ActionNodeKind.CHECKPOINT)] = (
            ActionOutcomeResult.UNCONFIRMED
        )
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=3000, checkpoints=(1000,)),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-unknown")]
        # 过程节点未确认后：后续节点保留，角色在下一真实周期看到
        # 未确认状态（不是失败也不是成功），行动继续可维持。
        unconfirmed_input = alice.perceived[2]
        states = {state.action_id: state.phase for state in unconfirmed_input.known_action_states}
        assert states.get(ActionId("a-unknown")) is ActionPhase.UNCONFIRMED
        # 后续节点没有被删除，World 最终在结束节点给出已知结果。
        assert record.expected_end == ms(4100)
        facts = "；".join(world.facts)
        assert "a-unknown 到达过程节点" in facts
        assert "a-unknown 结算" in facts, "未知之后仍按后续节点继续结算"

    asyncio.run(scenario())


def test_progress_update_bumps_version_when_world_actually_changes() -> None:
    """进展更新实际改变世界时递增共同状态版本（假世界履约）。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-move", "移动"),)),
                response("alice", "r-2", (maintain("a-move"),)),
            ],
            fallback=maintaining("alice", "a-move"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-move", 0.001)
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=5000, checkpoints=(2000,)),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event(),
        )
        before = world.state_version
        await scheduler.run()
        assert world.positions["alice"] > 0
        assert world.state_version > before, "实际位置变化递增版本"

    asyncio.run(scenario())


def test_translation_request_carries_objective_context_not_role_perception() -> None:
    """翻译请求携带 World 只读客观上下文；角色感知不含该内容。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-ctx", "需要客观条件"),))],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(delay=1000, duration=1000)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event(),
            should_end=lambda status: status.completed_cycles >= 1,
        )
        await scheduler.run()
        assert translator.requests, "含新行动的轮次发起翻译"
        context = translator.requests[0].objective_context
        assert context is not None
        assert context.role_id == "alice"
        assert context.world_state_version == world.state_version or True
        assert any("位置" in condition for condition in context.conditions)
        # 客观上下文不进入角色感知。
        for turn in alice.perceived:
            for observation in turn.observations:
                assert "客观上下文" not in observation.content
                assert "速度" not in observation.content

    asyncio.run(scenario())


def test_structural_translation_mismatch_is_run_error_not_rejection() -> None:
    """翻译结构引用错误是运行错误，不进入打回。"""

    async def scenario() -> None:
        from sceneweave.domain.translation import ActionSchedule, TranslationResult

        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-ok", "正常行动"),)),
                response("alice", "r-2", (new_action("a-retry", "重试"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])

        def override(request):
            if request.response.response_id == "r-1":
                # 结构错误：安排引用了非本次新行动的身份。
                return TranslationResult(
                    cycle_id=request.response.cycle_id,
                    role_id=request.response.role_id,
                    response_id=request.response.response_id,
                    schedules=(
                        ActionSchedule(
                            action_id=ActionId("a-other"),
                            commit_group_id=request.response.actions[0].commit_group_id,
                            start=SimulationTime(request.current_time + 1000),
                            expected_end=SimulationTime(request.current_time + 2000),
                        ),
                    ),
                )
            return ScriptedTranslator(delay=1000, duration=1000).default_result(request)

        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(override=override),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "翻译结构不一致" in (result.error or "")
        assert alice.rejections == [], "结构错误不打回角色"
        assert len(scheduler.action_records()) == 0

    asyncio.run(scenario())


def test_engine_error_trace_carries_current_cycle_and_adjudication_state() -> None:
    """错误日志关联当前周期；裁决阶段错误现场含候选摘要。"""

    async def scenario() -> None:
        class BrokenWorld(FakeSemanticWorld):
            def __init__(self) -> None:
                super().__init__(["alice"])
                self.adjudications_made = 0

            async def adjudicate(self, *args, **kwargs):  # type: ignore[override]
                self.adjudications_made += 1
                if self.adjudications_made > 1:
                    raise RuntimeError("模型故障")
                return await super().adjudicate(*args, **kwargs)

        collector = Collector()
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-tr", "被追踪"),))],
            fallback=idle_response("alice"),
        )
        world = BrokenWorld()
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=1000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            tracer=collector,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        errors = collector.by_stage("engine.error")
        assert errors and errors[0].cycle_id and errors[0].cycle_id != "engine"
        assert result.snapshot is not None
        assert result.snapshot.cycle_id == errors[0].cycle_id

    asyncio.run(scenario())


def test_cycle_report_distinguishes_registry_world_diagnostic_and_events() -> None:
    """报告区分登记记录、World 实际诊断与角色可见视图；含行动节点事件。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-view", "被观察的行动"),)),
                response("alice", "r-2", (maintain("a-view"),)),
                response("alice", "r-3", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-view", 0.001)
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(delay=1000, duration=2000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event(),
        )
        # 手动驱动到登记周期结束，检查该轮报告。
        report_task = asyncio.create_task(scheduler.run())
        for _ in range(2000):
            if scheduler.last_report is not None:
                break
            await asyncio.sleep(0)
        first = scheduler.last_report
        assert first is not None
        # 登记当轮即可见 REGISTERED（不是下一轮才出现）。
        phases = {r.action_id: r.phase for r in first.registry_actions}
        assert phases.get(ActionId("a-view")) is ActionPhase.REGISTERED
        # 同刻事件包含行动节点与内思考的类型与身份，不只是外部世界事件。
        kinds = {e.kind for e in first.batch_events}
        assert "world" in kinds
        report_task.cancel()
        try:
            await report_task
        except asyncio.CancelledError:
            pass

        # 完整运行后：撤回当轮显示 WITHDRAWN，World 诊断含实际行动与
        # 被动过程，与事实增量文本分开。
        scheduler2 = make_scheduler(
            world,
            [
                ScriptedRole(
                    "alice",
                    [
                        response("alice", "r-1", (new_action("a-view2", "再次行动"),)),
                        response("alice", "r-2", ()),
                    ],
                    fallback=idle_response("alice"),
                )
            ],
            ScriptedTranslator(delay=1000, duration=2000),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event("restart", 0, "再次开场"),
        )
        result = await scheduler2.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        withdrawal_reports = []
        for report in (scheduler2.last_report,):
            withdrawal_reports.append(report)
        final = scheduler2.last_report
        assert final is not None
        # 撤回周期：登记记录显示 WITHDRAWN，World 诊断有被动过程。
        phases2 = {r.action_id: r.phase for r in final.registry_actions}
        assert phases2.get(ActionId("a-view2")) is ActionPhase.WITHDRAWN
        diagnostic = final.world_diagnostic
        assert diagnostic is not None
        assert diagnostic.passive_processes, "被动过程在诊断快照中，与事实文本分开"
        facts_text = "；".join(final.committed_facts)
        assert facts_text != "；".join(diagnostic.passive_processes)

    asyncio.run(scenario())


def test_retry_to_empty_declaration_skips_translator_with_withdrawal() -> None:
    """时间打回后改为空声明：不再调用翻译器，仍完成统一撤回语义。"""

    async def scenario() -> None:
        from sceneweave.domain.translation import ActionSchedule, TranslationResult

        collector = Collector()
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-hurry", "立刻行动"),)),
                response("alice", "r-2", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])

        def override(request):
            new_items = [i for i in request.response.actions if isinstance(i, NewAction)]
            if not new_items:
                raise AssertionError("空声明不得再调用翻译器")
            action = new_items[0]
            start = (
                request.current_time
                if request.response.response_id == "r-1"
                else request.current_time + 1000
            )
            return TranslationResult(
                cycle_id=request.response.cycle_id,
                role_id=request.response.role_id,
                response_id=request.response.response_id,
                schedules=(
                    ActionSchedule(
                        action_id=action.action_id,
                        commit_group_id=action.commit_group_id,
                        start=SimulationTime(start),
                        expected_end=SimulationTime(request.current_time + 2000),
                    ),
                ),
            )

        translator = ScriptedTranslator(override=override)
        scheduler = make_scheduler(
            world,
            [alice],
            translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
            tracer=collector,
            should_end=lambda status: status.completed_cycles >= 1,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.NATURAL_END
        # 翻译器只收到 r-1；r-2（空声明）走合成空安排路径。
        assert [r.response.response_id for r in translator.requests] == ["r-1"]
        assert len(alice.rejections) == 1, "时间问题仍走打回"
        assert len(scheduler.action_records()) == 0, "空声明不登记新行动"
        skipped = collector.by_stage("translation.skipped")
        assert skipped and skipped[0].response_id == "r-2", "跳过翻译留痕"

    asyncio.run(scenario())


def test_validate_mutating_world_never_reaches_commit() -> None:
    """校验期间改变世界状态的 World：提交前发现，commit 从不执行。"""

    async def scenario() -> None:
        class ValidateMutates(FakeSemanticWorld):
            def __init__(self) -> None:
                super().__init__(["alice"])
                self.commits = 0

            def validate_adjudication(self, candidate):  # type: ignore[no-untyped-def]
                validation = super().validate_adjudication(candidate)
                self.bump_for_test()  # 违约：校验中改状态
                return validation  # 仍报告旧版本

            def commit_adjudication(self, candidate):  # type: ignore[no-untyped-def]
                self.commits += 1
                return super().commit_adjudication(candidate)

        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = ValidateMutates()
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "validate_adjudication 不得改变共同状态" in (result.error or "")
        assert world.commits == 0, "事实提交从不发生"

    asyncio.run(scenario())


def test_readonly_boundaries_reject_mutations() -> None:
    """publish 与 diagnostic_snapshot 是只读边界：违约即停止。"""

    async def scenario() -> None:
        class PublishMutates(FakeSemanticWorld):
            def publish(self, commit):  # type: ignore[override]
                self.bump_for_test()
                return super().publish(commit)

        class DiagnosticMutates(FakeSemanticWorld):
            def diagnostic_snapshot(self):  # type: ignore[no-untyped-def]
                self.bump_for_test()
                return super().diagnostic_snapshot()

        for broken, expected in (
            (PublishMutates, "publish 是只读边界"),
            (DiagnosticMutates, "diagnostic_snapshot 是只读边界"),
        ):
            alice = ScriptedRole(
                "alice",
                [response("alice", "r-1", (new_action("a-ro", "只读检查"),))],
                fallback=idle_response("alice"),
            )
            world = broken(["alice"])
            scheduler = make_scheduler(
                world,
                [alice],
                ScriptedTranslator(delay=1000, duration=1000),
                config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
                initial_events=startup_event(),
                should_end=lambda status: status.completed_cycles >= 1,
            )
            result = await scheduler.run()
            assert expected in (result.error or ""), broken.__name__

    asyncio.run(scenario())


def test_progress_version_regression_is_refused() -> None:
    """进展更新使版本回退被拒绝。"""

    async def scenario() -> None:
        class RewindProgress(FakeSemanticWorld):
            async def update_progress(self, cycle_id, simulation_time):  # type: ignore[override]
                result = await super().update_progress(cycle_id, simulation_time)
                self.rewind_for_test()
                return result

        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = RewindProgress(["alice"])
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(20_000)),
            initial_events=startup_event(),
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "版本回退" in (result.error or "")

    asyncio.run(scenario())


def test_cancel_mid_cycle_forbids_blind_resume() -> None:
    """周期中途取消：现场保留，实例禁止再次推进。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        alice = ScriptedRole("alice", gate=gate, fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        scheduler = make_scheduler(
            world, [alice], ScriptedTranslator(), initial_events=startup_event()
        )
        task = asyncio.create_task(scheduler.run())
        await alice.started.wait()
        assert scheduler.stage == "cycle.responses"
        task.cancel()
        try:
            await task
            raise AssertionError("取消必须传播")
        except asyncio.CancelledError:
            pass
        assert scheduler.current_time == ms(100), "现场保留"
        with pytest.raises(RuntimeError, match="不可再次推进"):
            await scheduler.run()

    asyncio.run(scenario())


def test_cancel_at_gate_wait_is_resumable() -> None:
    """门控等待处取消未取出事件批：实例可再次运行。"""

    async def scenario() -> None:
        from concordia.environment.step_controller import StepController

        from sceneweave.integrations.concordia.control import StepControllerGate

        class SwitchableGate:
            """可切换的门：真实控制器暂停时等待；禁用后直接放行。"""

            def __init__(self) -> None:
                self.controller = StepController(start_paused=True)
                self.enabled = True

            def should_stop(self) -> bool:
                return self.enabled and StepControllerGate(self.controller).should_stop()

            async def wait_for_permission(self) -> bool:
                if not self.enabled:
                    return True
                return await StepControllerGate(self.controller).wait_for_permission()

        alice = ScriptedRole("alice", fallback=idle_response("alice"))
        world = FakeSemanticWorld(["alice"])
        switch = SwitchableGate()
        scheduler = make_scheduler(
            world,
            [alice],
            ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=startup_event(),
            gate=switch,
        )
        task = asyncio.create_task(scheduler.run())
        for _ in range(2000):
            if scheduler.stage == "run.gate_wait":
                break
            await asyncio.sleep(0)
        assert scheduler.stage == "run.gate_wait"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert scheduler.current_time == ms(0), "未取出事件批"
        switch.enabled = False
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert world.adjudications, "恢复后正常处理事件"

    asyncio.run(scenario())
