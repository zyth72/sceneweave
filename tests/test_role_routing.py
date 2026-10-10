"""System1 三路分流与 System3 独立失败信号的调用链回归。

覆盖：continue 复用已保存计划且不调用 System2/3/翻译器；replan 跳过 System2
只调 System3；rethink 重选意向；System3 独立失败信号才升级 System2；World
一次拒绝只先反馈 System3；API/格式错误不被当作意向不可落实；失败候选不覆盖
已保存计划；重试/升级有限上限。
"""

import asyncio

import pytest

from sceneweave.domain.action_plan import ActionKind
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    EventId,
    WorldStateVersion,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    ActionPhase,
    KnownActionState,
    TriggerReason,
    TurnInput,
    WorldEvent,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionSchedule, PlanAcceptance
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.roles.model_role import ModelRole

from .fakes import (
    FakeSemanticWorld,
    ScriptedSampler,
    ScriptedTranslator,
    ms,
    rid,
    s1_continue,
    s1_replan,
    s1_rethink,
)

S1_CONTINUE = s1_continue("照旧")
S1_REPLAN = s1_replan("换做法")
S1_RETHINK = s1_rethink("重新决定")
S2_A = '{"intention": "拿到钥匙"}'
S2_B = '{"intention": "改找备用钥匙"}'
S3_GRASP = '[{"kind": "grasp", "target": "key", "effector": "right_hand"}]'
S3_WAIT = '[{"kind": "wait"}]'
S3_FAIL = '{"error": "intention_unfulfillable", "reason": "当前已知条件没有可触达路径"}'
WAIT_DISPLAY = "等待"


def turn(*, cycle: str = "cycle-1", time: int = 100, states: tuple = ()) -> TurnInput:
    return TurnInput(
        cycle_id=CycleId(cycle),
        role_id=rid("lin"),
        simulation_time=SimulationTime(time),
        trigger=TriggerReason((), False),
        known_action_states=states,
    )


def known(action_id: str, phase: ActionPhase) -> KnownActionState:
    return KnownActionState(ActionId(action_id), phase, "状态")


async def save_plan(role: ModelRole, response, schedule_map) -> None:
    assert response.plan is not None
    schedules = tuple(
        ActionSchedule(
            ActionId(action_id),
            CommitGroupId("g"),
            SimulationTime(start),
            SimulationTime(end),
        )
        for action_id, start, end in schedule_map
    )
    await role.save_plan(
        PlanAcceptance(
            declaration=response,
            plan=response.plan,
            schedules=schedules,
            check=ActionCheckResult(
                response.cycle_id,
                response.role_id,
                response.response_id,
                WorldStateVersion(0),
                True,
            ),
        )
    )


def test_continue_reuses_saved_plan_without_system2_or_3() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_RETHINK, S1_CONTINUE], "system2": [S2_A], "system3": [S3_GRASP]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        before = list(sampler.components())
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        assert sampler.components() == before + ["system1"], "continue 不调用 System2/3"
        assert second.reused is True
        assert second.plan is not None
        submission = role.execution_submission()
        assert [item.action_id for item in submission.actions] == [action_id], "提交最近步骤"

    asyncio.run(scenario())


def test_replan_skips_system2_and_calls_system3() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK, S1_REPLAN],
                "system2": [S2_A],
                "system3": [S3_GRASP, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        await save_plan(role, first, [(str(first.actions[0].action_id), 100, 200)])
        second = await role.perceive(turn(cycle="cycle-2"))
        assert sampler.components().count("system2") == 1, "replan 跳过 System2"
        assert sampler.components().count("system3") == 2, "replan 只重调 System3"
        assert second.reused is False
        assert [item.action.kind for item in second.actions] == [ActionKind.WAIT]

    asyncio.run(scenario())


def test_rethink_calls_system2_and_system3() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK, S1_RETHINK],
                "system2": [S2_A, S2_B],
                "system3": [S3_GRASP, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        await save_plan(role, first, [(str(first.actions[0].action_id), 100, 200)])
        second = await role.perceive(turn(cycle="cycle-2"))
        assert sampler.components().count("system2") == 2
        assert sampler.components().count("system3") == 2
        assert [item.action.kind for item in second.actions] == [ActionKind.WAIT]

    asyncio.run(scenario())


def test_system3_unfulfillable_escalates_to_system2() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK],
                "system2": [S2_A, S2_B],
                "system3": [S3_FAIL, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        response = await role.perceive(turn())
        assert sampler.components().count("system2") == 2, "失败信号才升级 System2"
        assert sampler.components().count("system3") == 2
        assert [item.action.kind for item in response.actions] == [ActionKind.WAIT]
        # 失败信号是独立内部结果，不进入动作列表，也不改变已保存计划（此时尚无）。
        assert role.saved_plan_digest() == ()

    asyncio.run(scenario())


def test_format_error_is_not_unfulfillable() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_RETHINK], "system2": [S2_A], "system3": ["坏 JSON"]}
        )
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="不是合法 JSON|必须是 JSON"):
            await role.perceive(turn())
        assert sampler.components().count("system2") == 1, "格式错误不升级 System2"

    asyncio.run(scenario())


def test_failed_candidate_does_not_overwrite_saved_plan() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK, S1_RETHINK, S1_CONTINUE],
                "system2": [S2_A, S2_B],
                "system3": [S3_GRASP, "坏 JSON"],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        before = role.saved_plan_digest()
        with pytest.raises(Exception, match="不是合法 JSON|必须是 JSON"):
            await role.perceive(turn(cycle="cycle-2"))
        assert role.saved_plan_digest() == before, "失败候选不覆盖已保存计划"
        third = await role.perceive(
            turn(cycle="cycle-3", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        assert third.reused is True
        assert [item.action_id for item in role.execution_submission().actions] == [action_id]

    asyncio.run(scenario())


def test_unfulfillable_retry_limit_is_reported() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK],
                "system2": [S2_A, S2_B, S2_A],
                "system3": [S3_FAIL, S3_FAIL, S3_FAIL],
            }
        )
        role = ModelRole(rid("lin"), sampler, max_system2_calls=2, max_system3_calls=5)
        with pytest.raises(RuntimeError, match="意向重选超过上限"):
            await role.perceive(turn())

    asyncio.run(scenario())


def test_failure_signal_pipeline_translates_checks_saves_and_submits() -> None:
    """失败信号→System2 重选→System3 新链→完整翻译/World 检查→保存→最近提交。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK] + [S1_CONTINUE] * 6,
                "system2": [S2_A, S2_B],
                "system3": [S3_FAIL, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        world = FakeSemanticWorld(["lin"])
        translator = ScriptedTranslator(plans={WAIT_DISPLAY: (100, 1000, ())})
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(5_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert sampler.components().count("system2") == 2
        assert sampler.components().count("system3") == 2
        records = scheduler.action_records()
        assert len(records) == 1, "只有重选后的新链被登记"
        [only] = records.values()
        assert only.phase is ActionPhase.COMPLETED
        assert len(translator.requests) == 1, "完整链只翻译一次"

    asyncio.run(scenario())


def test_world_rejection_only_recalls_system3_first() -> None:
    """World 整链检查拒绝先反馈 System3；不重选意向，只有失败信号才升级 System2。"""

    async def scenario() -> None:
        impossible = '[{"kind": "reach", "target": "不可能地隔空取物", "effector": "right_hand"}]'
        sampler = ScriptedSampler(
            {
                "system1": [S1_RETHINK] + [S1_CONTINUE] * 8,
                "system2": [S2_A],
                "system3": [impossible, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        world = FakeSemanticWorld(["lin"])
        translator = ScriptedTranslator(plans={WAIT_DISPLAY: (100, 1000, ())})
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(5_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert sampler.components().count("system2") == 1, "普通拒绝不重选意向"
        assert sampler.components().count("system3") == 2, "先重调 System3"
        records = scheduler.action_records()
        assert len(records) == 1
        assert not any("不可能" in fact for fact in world.facts), "被打回候选未生效"

    asyncio.run(scenario())


def test_continue_cycle_does_not_call_translator() -> None:
    """continue 复用已保存排程：整轮只翻译一次，后续周期不重译。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_RETHINK] + [S1_CONTINUE] * 8, "system2": [S2_A], "system3": [S3_WAIT]}
        )
        role = ModelRole(rid("lin"), sampler)
        world = FakeSemanticWorld(["lin"])
        translator = ScriptedTranslator(plans={WAIT_DISPLAY: (100, 3000, (1000,))})
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(500), max_duration=ms(3_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        await scheduler.run()
        assert len(translator.requests) == 1, "continue 不重复调用翻译器"
        assert sampler.components().count("system2") == 1
        assert sampler.components().count("system3") == 1

    asyncio.run(scenario())
