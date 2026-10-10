"""整链检查→保存→最近步骤提交→剧情跳转，以及依赖/平坦排程回归。

覆盖：一维结构化链的严格解析（二维/自由文本/ID-only 拒绝）、依赖引用与
环拒绝、Translator 平坦独立排程（无共同时间交集要求）、整链检查在翻译
之后、检查成功不等于执行提交、最近步骤提交与剧情跳转、依赖真实完成门控。
"""

import asyncio

from concordia.utils import structured_logging

from sceneweave.domain.action_plan import ActionKind, ActionPlan, parse_action_plan
from sceneweave.domain.adjudication import ActionOutcomeResult
from sceneweave.domain.identity import ActionId, EventId
from sceneweave.domain.messages import ActionPhase, WorldEvent
from sceneweave.domain.model_json import ModelJsonError
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import TranslationRequest, TranslationResult
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator

from .fakes import FakeSemanticWorld, ScriptedSampler, ms, rid


def test_action_plan_one_dimensional_and_rejections() -> None:
    plan = parse_action_plan(
        '[{"kind": "reach", "target": "a", "effector": "right_hand"},'
        ' {"kind": "grasp", "target": "a", "effector": "right_hand",'
        ' "depends_on": ["a1"]}]'
    )
    assert isinstance(plan, ActionPlan)
    assert len(plan.actions) == 2
    for bad in ('[[{"kind": "wait"}]]', '["自由文本动作"]', '[{"id": "a1"}]'):
        try:
            parse_action_plan(bad)
        except ModelJsonError:
            continue
        raise AssertionError(f"应拒绝：{bad}")


def _role_script() -> dict[str, list[str]]:
    return {
        "system1": ['{"route": "rethink", "note": "开始"}']
        + ['{"route": "continue", "note": "继续"}'] * 12,
        "system2": ['{"intention": "依次取物"}'],
        "system3": [
            '[{"kind": "move", "destination": {"entity": "a1"}, "key": "k1"},'
            ' {"kind": "look", "target": "a2", "key": "k2"},'
            ' {"kind": "reach", "target": "a3", "effector": "right_hand",'
            ' "depends_on": ["k1"]}]'
        ],
    }


def _translator_script() -> dict[str, list[str]]:
    return {
        "translator": [
            '{"schedules": [{"action_id": "a-lin-1", "start": 60000, "expected_end": 120000},'
            ' {"action_id": "a-lin-2", "start": 90000, "expected_end": 138000},'
            ' {"action_id": "a-lin-3", "start": 126000, "expected_end": 150000}]}',
        ]
    }


def test_dependent_step_submitted_only_after_dependency_completes() -> None:
    """a1 60000~120000、a2 90000~138000、a3 126000~150000；a3 只依赖 a1。"""

    async def scenario() -> None:
        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()), tracer=tracer)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script()), tracer=tracer),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(130000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        records = scheduler.action_records()
        assert records[ActionId("a-lin-1")].phase is ActionPhase.COMPLETED
        assert records[ActionId("a-lin-2")].phase is ActionPhase.STARTED, "a2 仍在执行"
        assert records[ActionId("a-lin-3")].phase is ActionPhase.STARTED, "a3 已开始"
        assert records[ActionId("a-lin-3")].start == SimulationTime(126000), "a3 在 126000 开始"
        assert records[ActionId("a-lin-3")].depends_on == (ActionId("a-lin-1"),)
        # 顺序可见：翻译在整链检查之前；整链检查后有保存与最近步骤提交日志。
        order = [entry.component_name for entry in log.entries]
        assert order.index("translator.invoke") < order.index("engine.chain_checked")
        assert "engine.step_submitted" in order
        assert "engine.chain_checked" in order
        assert "engine.step_submitted" in order

    asyncio.run(scenario())


def test_future_dependent_not_submitted_before_dependency_and_no_started_fact() -> None:
    """依赖未真实完成前不提交后续步骤；检查成功不产生任何 started 事实。"""

    async def scenario() -> None:
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()))
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script())),
            # 在 a1 完成(120000)与 a3 开始(126000)之间停止：a3 已登记但未执行。
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(123000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        records = scheduler.action_records()
        assert set(records) == {
            ActionId("a-lin-1"),
            ActionId("a-lin-2"),
            ActionId("a-lin-3"),
        }
        assert records[ActionId("a-lin-3")].phase is ActionPhase.REGISTERED, "尚未执行"
        assert not any("a-lin-3" in fact for fact in world.facts), "检查成功无 started 事实"

    asyncio.run(scenario())


def test_future_steps_stay_internal_until_ready() -> None:
    """依赖未完成的未来动作参与整链翻译和检查，保留在私有缓存中而不登记。"""

    async def scenario() -> None:
        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()), tracer=tracer)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script())),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(1000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        await scheduler.run()
        records = scheduler.action_records()
        assert ActionId("a-lin-3") not in records, "依赖未完成的后续动作不得登记"
        assert ActionId("a-lin-1") in records, "只登记最早尚未提交的最近步骤"
        assert ActionId("a-lin-2") not in records, "未来动作不一次性登记"

    asyncio.run(scenario())


def test_registration_accepts_nonoverlapping_flat_schedules() -> None:
    """三个独立区间（无共同交集）合法：不再要求全组共同非零交集。"""
    from sceneweave.domain.action_plan import AtomicAction
    from sceneweave.domain.identity import CycleId, ResponseId
    from sceneweave.domain.messages import RoleResponse, StepAction
    from sceneweave.domain.translation import ActionSchedule, TranslationResult
    from sceneweave.engine.registration import registration_issues

    role_id = rid("lin")
    actions = tuple(
        StepAction(ActionId(f"a-lin-{index}"), AtomicAction(kind=ActionKind.WAIT))
        for index in (1, 2, 3)
    )
    response = RoleResponse(CycleId("c1"), role_id, ResponseId("r1"), actions)
    schedules = (
        ActionSchedule(ActionId("a-lin-1"), "g", SimulationTime(60000), SimulationTime(120000)),
        ActionSchedule(ActionId("a-lin-2"), "g", SimulationTime(90000), SimulationTime(138000)),
        ActionSchedule(ActionId("a-lin-3"), "g", SimulationTime(126000), SimulationTime(150000)),
    )
    translation = TranslationResult(CycleId("c1"), role_id, ResponseId("r1"), schedules=schedules)
    issues = registration_issues(
        response, translation, current_time=SimulationTime(0), arrangements=()
    )
    assert issues == (), f"不应有登记问题：{issues}"


class _SpyWorld(FakeSemanticWorld):
    """记录 check_response 的真实调用与传参，并共享调用顺序事件流。"""

    def __init__(self, role_ids, events: list[str]) -> None:
        super().__init__(role_ids)
        self._events = events
        self.check_calls: list[tuple[object, tuple, tuple]] = []

    async def check_response(self, candidate, *, arrangements=(), schedules=()):
        self._events.append("check")
        self.check_calls.append((candidate, tuple(arrangements), tuple(schedules)))
        return await super().check_response(
            candidate, arrangements=arrangements, schedules=schedules
        )


class _CountingTranslator:
    """委托真实翻译器并记录调用次数与顺序。"""

    def __init__(self, inner, events: list[str]) -> None:
        self._inner = inner
        self._events = events
        self.calls = 0

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        self.calls += 1
        self._events.append("translate")
        return await self._inner.translate(request)


class _RecordingRole(ModelRole):
    """在 save_plan 成功的瞬间抓取引擎登记与世界事实，验证保存不登记。"""

    def __init__(self, *args, on_save, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_save = on_save
        self.save_count = 0

    async def save_plan(self, acceptance) -> None:
        await super().save_plan(acceptance)
        self.save_count += 1
        self._on_save()


def test_whole_chain_translate_then_check_then_save_then_nearest_submit() -> None:
    """真实 spy：翻译在检查之前、只调用一次且拿全链；保存不登记；只提交最近一步。"""

    async def scenario() -> None:
        events: list[str] = []
        sampler = ScriptedSampler(_role_script())
        world = _SpyWorld(["lin"], events)
        holder: dict[str, object] = {}

        def on_save() -> None:
            holder["registry_at_save"] = dict(scheduler.action_records())
            holder["facts_at_save"] = tuple(world.facts)

        role = _RecordingRole(rid("lin"), sampler, on_save=on_save)
        inner_sampler = ScriptedSampler(_translator_script())
        translator = _CountingTranslator(ModelTranslator.shared(inner_sampler), events)
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(1000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        await scheduler.run()

        # 翻译一次拿全链；检查在翻译之后且只一次，拿到完整 3 条链 + 3 条排程。
        assert events == ["translate", "check"], f"顺序必须是先翻译后检查：{events}"
        assert translator.calls == 1
        assert len(world.check_calls) == 1
        checked, _arrangements, schedules = world.check_calls[0]
        assert checked.plan is not None
        assert [str(item.action_id) for item in checked.plan.actions] == [
            "a-lin-1",
            "a-lin-2",
            "a-lin-3",
        ]
        assert {str(item.action_id) for item in schedules} == {
            "a-lin-1",
            "a-lin-2",
            "a-lin-3",
        }
        assert {item.start for item in schedules} == {60000, 90000, 126000}
        # 翻译器提示确实包含完整链三个身份。
        translator_prompts = [
            prompt for component, prompt in inner_sampler.calls if component == "translator"
        ]
        assert len(translator_prompts) == 1
        for action_id in ("a-lin-1", "a-lin-2", "a-lin-3"):
            assert action_id in translator_prompts[0], "翻译器一次拿到完整链"

        # 保存那一刻：registry 尚未登记任何行动，也没有该行动的 started 事实。
        assert role.save_count == 1
        assert holder["registry_at_save"] == {}
        assert not any("a-lin-1" in fact for fact in holder["facts_at_save"])
        # 只登记最早尚未提交的最近步骤 a1，而非 a2/a3。
        records = scheduler.action_records()
        assert set(records) == {ActionId("a-lin-1")}
        assert records[ActionId("a-lin-1")].phase is ActionPhase.REGISTERED
        assert not any("a-lin-1" in fact for fact in world.facts), "检查/保存不产生 started 事实"

    asyncio.run(scenario())


def test_late_dependency_completion_retimes_cached_step_and_does_not_start_early() -> None:
    """a1 实际晚于 a3 预计开始才完成：a3 不得提前开始，按真实完成后平移激活。"""

    async def scenario() -> None:
        # a1 预计 60000~140000（晚于 a3 的 126000），a3 只依赖 a1。
        script = {
            "translator": [
                '{"schedules": ['
                '{"action_id": "a-lin-1", "start": 60000, "expected_end": 140000},'
                '{"action_id": "a-lin-2", "start": 90000, "expected_end": 138000},'
                '{"action_id": "a-lin-3", "start": 126000, "expected_end": 150000}]}',
            ]
        }
        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()), tracer=tracer)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(script), tracer=tracer),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(150000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        await scheduler.run()
        records = scheduler.action_records()
        assert records[ActionId("a-lin-1")].phase is ActionPhase.COMPLETED
        a3 = records[ActionId("a-lin-3")]
        assert a3.start > SimulationTime(140000), "不得在 a1 完成前开始"
        assert a3.start == SimulationTime(140001), "在真实完成后的合法时刻激活"
        assert a3.expected_end - a3.start == SimulationTime(24000), "保持原持续时长"
        assert any(entry.component_name == "engine.schedule_retimed" for entry in log.entries), (
            "平移激活必须留痕"
        )

    asyncio.run(scenario())


def test_failed_dependency_never_releases_later_step() -> None:
    """前置确认失败：后置动作不放行、不登记。"""

    async def scenario() -> None:
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()))
        world = FakeSemanticWorld(["lin"])
        world.outcome_overrides[ActionId("a-lin-1")] = ActionOutcomeResult.FAILED
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script())),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(150000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        await scheduler.run()
        records = scheduler.action_records()
        assert records[ActionId("a-lin-1")].phase is ActionPhase.FAILED
        assert ActionId("a-lin-3") not in records, "失败不放行依赖"

    asyncio.run(scenario())


def test_unconfirmed_dependency_never_releases_later_step() -> None:
    """前置未确认（结果未知，不是已知失败）：后置仍不放行。"""

    async def scenario() -> None:
        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()))
        world = FakeSemanticWorld(["lin"])
        world.outcome_overrides[ActionId("a-lin-1")] = ActionOutcomeResult.UNCONFIRMED
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script())),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(150000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        await scheduler.run()
        records = scheduler.action_records()
        assert records[ActionId("a-lin-1")].phase is ActionPhase.UNCONFIRMED
        assert ActionId("a-lin-3") not in records, "未确认不放行依赖"

    asyncio.run(scenario())


def test_world_check_mutating_state_fails_before_save_or_register() -> None:
    """无副作用的整链检查若改动共同状态，必须立即失败，不得保存/登记。"""

    async def scenario() -> None:
        class MutatingCheckWorld(FakeSemanticWorld):
            async def check_response(self, candidate, *, arrangements=(), schedules=()):
                result = await super().check_response(
                    candidate, arrangements=arrangements, schedules=schedules
                )
                self.bump_for_test()
                return result

        role = ModelRole(rid("lin"), ScriptedSampler(_role_script()))
        world = MutatingCheckWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ModelTranslator.shared(ScriptedSampler(_translator_script())),
            config=RunConfig(introspection_interval=ms(1_000_000), max_duration=ms(150000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert result.error is not None and "只读边界" in result.error
        assert scheduler.action_records() == {}, "检查改动状态时不得登记任何行动"

    asyncio.run(scenario())
