"""端到端集成：模型角色 + 调度器 + 假 World/Translator + 结构化日志。

覆盖：完整闭环（含 World 检查打回后角色重做）、阶段日志沿因果链检索、
真实 StepData 呈现模型角色的声明、跨角色并行采样与剧情时钟冻结。
"""

import asyncio

from concordia.utils import structured_logging

from sceneweave.domain.identity import EventId
from sceneweave.domain.messages import ActionPhase, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.control import step_data_from_report
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole

from .fakes import (
    FakeSemanticWorld,
    ScriptedSampler,
    ScriptedTranslator,
    ms,
    rid,
    s1_continue,
    s1_rethink,
)

S1_GO = s1_rethink("门被推开，需要行动")
S1_SKIP = s1_continue("没有新情况")
S2 = '{"intention": "查看书桌上的东西"}'
S2_AGAIN = '{"intention": "查看书桌上的东西"}'
S3_IMPOSSIBLE = '[{"kind": "grasp", "target": "不可能地隔空取物", "effector": "right_hand"}]'
S3_WALK = '[{"kind": "move", "destination": {"entity": "书桌"}}]'
WALK_DISPLAY = "移动到 实体 书桌（walk）"


def build_log_and_tracer() -> tuple[structured_logging.SimulationLog, SimulationLogTracer]:
    log = structured_logging.SimulationLog()
    tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
    return log, tracer


def test_end_to_end_model_role_with_scheduler_and_log_chain() -> None:
    """完整闭环：模型角色声明→World 打回→角色重做→登记→结算→日志。"""

    async def scenario() -> None:
        log, tracer = build_log_and_tracer()
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_SKIP, S1_SKIP, S1_SKIP],
                "system2": [S2, S2_AGAIN],
                "system3": [S3_IMPOSSIBLE, S3_WALK],
            }
        )
        role = ModelRole(rid("lin"), sampler, tracer=tracer)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ScriptedTranslator(plans={WALK_DISPLAY: (1000, 2000, ())}),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        facts = "；".join(world.facts)
        assert "书桌" in facts, "重做后的新动作真实开始与结算"
        assert "不可能" not in facts, "被打回的候选未生效"
        record = scheduler.action_records()
        assert len(record) == 1
        [only] = record.values()
        assert only.phase is ActionPhase.COMPLETED
        assert only.role_id == "lin"
        stages = {entry.component_name for entry in log.entries}
        assert {
            "system1.decide",
            "system2.intend",
            "system3.declare",
            "role.response",
            "engine.chain_checked",
            "registration.commit",
            "world.commit",
        } <= stages
        s1 = [
            dict(log.reconstruct_value(e.deduplicated_data))
            for e in log.get_entries_by_component("system1.decide")
        ]
        assert any("门被推开" in item["inputs"][0] for item in s1), "提示含真实感知"
        assert any(item["outputs"] for item in s1), "原始模型输出被记录"
        s3 = [
            dict(log.reconstruct_value(e.deduplicated_data))
            for e in log.get_entries_by_component("system3.declare")
        ]
        assert any("不可能地隔空取物" in item["outputs"][0] for item in s3)
        assert any("书桌" in item["outputs"][0] for item in s3)
        report = scheduler.last_report
        assert report is not None
        data = step_data_from_report(report)
        assert "lin" in data.entity_actions

    asyncio.run(scenario())


def test_cross_role_parallel_sampling_with_frozen_clock() -> None:
    """两个模型角色并行采样；等待期间剧情时钟冻结。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        sampler = ScriptedSampler(
            {
                "system1": [S1_SKIP, S1_SKIP, S1_SKIP, S1_SKIP],
                # 首次没有意向/计划：为两名角色各提供一次思考并接受空计划，
                # 之后才走 continue。
                "system2": [
                    '{"intention": "先观察，不主动行动"}',
                    '{"intention": "先观察，不主动行动"}',
                ],
                "system3": ["[]", "[]"],
            },
            gate=gate,
        )
        alice = ModelRole(rid("alice"), sampler)
        bob = ModelRole(rid("bob"), sampler)
        world = FakeSemanticWorld(["alice", "bob"])
        scheduler = Scheduler(
            roles=[alice, bob],
            world=world,
            translator=ScriptedTranslator(),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        task = asyncio.create_task(scheduler.run())
        for _ in range(2000):
            if sampler.in_flight >= 2:
                break
            await asyncio.sleep(0)
        assert sampler.in_flight == 2, "跨角色并行采样"
        components = [component for component, _ in sampler.calls]
        assert components.count("system1") == 2
        assert scheduler.current_time == SimulationTime(0), "采样期间剧情时钟冻结"
        gate.set()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT
        assert alice.memory_notes() and bob.memory_notes()

    asyncio.run(scenario())


def test_model_role_survives_withdrawal_and_reregistration() -> None:
    """模型角色跨周期：新增→重声明→完成后再次新增，身份不复用。"""

    async def scenario() -> None:
        log, tracer = build_log_and_tracer()
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO, S1_GO, S1_SKIP, S1_SKIP, S1_SKIP],
                "system2": [S2, S2, S2],
                "system3": [
                    S3_WALK,
                    '[{"id": "a-lin-1", "kind": "move", "destination": {"entity": "书桌"}}]',
                    S3_WALK,
                ],
            }
        )
        role = ModelRole(rid("lin"), sampler, tracer=tracer)
        world = FakeSemanticWorld(["lin"])
        scheduler = Scheduler(
            roles=[role],
            world=world,
            translator=ScriptedTranslator(plans={WALK_DISPLAY: (1000, 1000, ())}),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        records = scheduler.action_records()
        ids = sorted(str(action_id) for action_id in records)
        assert ids == ["a-lin-1", "a-lin-2"], "新身份不复用"
        assert all(record.phase is ActionPhase.COMPLETED for record in records.values())
        facts = "；".join(world.facts)
        assert facts.count("书桌") >= 2
        assert "需要行动" not in facts, "认知说明未混入客观事实"

    asyncio.run(scenario())
