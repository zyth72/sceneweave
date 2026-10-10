"""模型驱动角色的内部编排行为测试（脚本采样器，无真实模型）。

协议纠正后：System2 只产意向；System3 产一维全量结构化动作计划；
角色对外声明当前可执行动作（含真实身份与已解析依赖）。覆盖：System1
跳过仍复用已接受计划、完成事件走完整周期、真实完成满足依赖、已完成
动作不重新提交、身份不重启、空计划、错误不伪空、候选确认与打回。
"""

import asyncio

import pytest

from sceneweave.domain.action_plan import (
    ActionKind,
    ActionPlan,
    AtomicAction,
    Effector,
    PlanAction,
)
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    EventId,
    ResponseId,
    WorldStateVersion,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    ActionPhase,
    KnownActionState,
    Observation,
    Rejection,
    RoleResponse,
    StepAction,
    TriggerReason,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionSchedule, PlanAcceptance
from sceneweave.roles.model_role import ModelRole

from .fakes import ScriptedSampler, rid

S1_SKIP = '{"route": "continue", "note": "一切照旧"}'
S1_GO = '{"route": "rethink", "note": "情况有变"}'
S2 = '{"intention": "拿到抽屉里的钥匙"}'
S2_OTHER = '{"intention": "改去找备用钥匙"}'
S3_GRASP = '[{"kind": "grasp", "target": "key", "effector": "right_hand"}]'
S3_TWO = (
    '[{"kind": "reach", "target": "key", "effector": "right_hand",'
    ' "key": "r", "depends_on": ["g"]},'
    ' {"kind": "grasp", "target": "key", "effector": "right_hand", "key": "g"}]'
)
S3_EMPTY = "[]"
S3_STRING = '["走过去并拿走钥匙"]'
S3_LEGACY = '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}'


def turn(*, cycle: str = "cycle-1", time: int = 100, states: tuple = ()) -> TurnInput:
    return TurnInput(
        cycle_id=CycleId(cycle),
        role_id=rid("lin"),
        simulation_time=SimulationTime(time),
        trigger=TriggerReason((), False),
        known_action_states=states,
    )


def known(action_id: str, phase: ActionPhase, detail: str = "状态") -> KnownActionState:
    return KnownActionState(ActionId(action_id), phase, detail)


async def save_plan(role: ModelRole, response, schedule_map) -> None:
    """模拟引擎在整链检查通过后调用 save_plan；schedule_map 为 (身份, start, end)。"""
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
    acceptance = PlanAcceptance(
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
    await role.save_plan(acceptance)


def test_system1_skip_reuses_saved_plan_and_advances_dependency() -> None:
    """不重新考虑：复用已保存计划；真实完成满足依赖后最近步骤前移。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_SKIP], "system2": [S2], "system3": [S3_TWO]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        # 完整链声明两个动作（含依赖未满足的伸手），供翻译与整链检查。
        assert [item.action.kind for item in first.actions] == [ActionKind.REACH, ActionKind.GRASP]
        reach = first.actions[0].action_id
        grasp = first.actions[1].action_id
        assert first.actions[0].depends_on == (grasp,), "依赖解析为真实身份"
        assert sampler.components() == ["system1", "system2", "system3"]
        await save_plan(role, first, [(str(grasp), 100, 200), (str(reach), 300, 400)])
        # 最近步骤：伸手依赖未满足，只提交抓取。
        submission = role.execution_submission()
        assert [item.action.kind for item in submission.actions] == [ActionKind.GRASP]

        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(grasp), ActionPhase.COMPLETED),))
        )
        # System1 跳过：不调用 System2/3；抓取已完成，伸手成为最近步骤。
        assert sampler.components() == ["system1", "system2", "system3", "system1"]
        assert f"{reach}（planned）" in sampler.calls[-1][1], (
            "System1 须看见未来动作只是私有计划，不是已提交或已登记"
        )
        assert second.reused is True
        submission2 = role.execution_submission()
        assert [item.action.kind for item in submission2.actions] == [ActionKind.REACH]
        assert submission2.actions[0].depends_on == (grasp,), "依赖解析为真实身份"

    asyncio.run(scenario())


def test_completed_action_is_not_resubmitted_and_identity_not_restarted() -> None:
    """已完成动作不重新提交；持续动作保留身份与阶段。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_SKIP, S1_SKIP], "system2": [S2], "system3": [S3_GRASP]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        first_submission = role.execution_submission()
        assert [item.action_id for item in first_submission.actions] == [action_id]
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        assert second.reused is True
        second_submission = role.execution_submission()
        assert [item.action_id for item in second_submission.actions] == [action_id], (
            "持续动作身份不变"
        )
        third = await role.perceive(
            turn(cycle="cycle-3", states=(known(str(action_id), ActionPhase.COMPLETED),))
        )
        assert third.reused is True
        assert role.execution_submission().actions == (), "已完成动作不再声明"

    asyncio.run(scenario())


def test_failed_or_unconfirmed_does_not_release_dependency() -> None:
    """失败/未确认不是 completed：依赖不放行。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_SKIP, S1_SKIP, S1_SKIP], "system2": [S2], "system3": [S3_TWO]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        reach = first.actions[0].action_id
        grasp = first.actions[1].action_id
        await save_plan(role, first, [(str(grasp), 100, 200), (str(reach), 300, 400)])
        assert [item.action.kind for item in role.execution_submission().actions] == [
            ActionKind.GRASP
        ]
        for phase in (ActionPhase.FAILED, ActionPhase.UNCONFIRMED, ActionPhase.STARTED):
            await role.perceive(turn(cycle="cycle-2", states=(known(str(grasp), phase),)))
            kinds = [item.action.kind for item in role.execution_submission().actions]
            assert ActionKind.REACH not in kinds, f"{phase.value} 不放行依赖"
            # 未终态时抓取自身仍被维持；失败时不重新提交。
            if phase is ActionPhase.FAILED:
                assert kinds == []
            else:
                assert kinds == [ActionKind.GRASP]

    asyncio.run(scenario())


def test_dependency_on_completed_history_action_is_allowed() -> None:
    """已完成的历史动作可作为已确认依赖，不重新登记、不改写终态。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_GO, S1_SKIP], "system2": [S2, S2], "system3": [S3_GRASP]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        history_id = first.actions[0].action_id
        # 第二轮：System1 重想，新计划把已完成动作作为依赖。
        sampler.script["system3"] = [
            '[{"kind": "reach", "target": "key", "effector": "right_hand",'
            ' "depends_on": ["' + str(history_id) + '"]}]'
        ]
        second = await role.perceive(
            turn(
                cycle="cycle-2",
                states=(known(str(history_id), ActionPhase.COMPLETED, "已完成"),),
            )
        )
        assert [item.action.kind for item in second.actions] == [ActionKind.REACH]
        assert second.actions[0].depends_on == (history_id,)

    asyncio.run(scenario())


def test_unknown_terminal_and_self_dependency_are_errors() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP]})
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        finished = first.actions[0].action_id
        # 未知身份。
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2]
        sampler.script["system3"] = ['[{"kind": "wait", "depends_on": ["a-unknown"]}]']
        with pytest.raises(Exception, match="未定义"):
            await role.perceive(turn(cycle="cycle-2"))
        # 终态身份（FAILED）不可作依赖。
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2]
        sampler.script["system3"] = ['[{"kind": "wait", "depends_on": ["' + str(finished) + '"]}]']
        with pytest.raises(Exception, match="终态"):
            await role.perceive(
                turn(
                    cycle="cycle-3",
                    states=(known(str(finished), ActionPhase.FAILED),),
                )
            )

    asyncio.run(scenario())


def test_legacy_three_category_and_free_string_outputs_are_rejected() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_LEGACY]})
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="控制对象包含未知字段"):
            await role.perceive(turn())
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2]
        sampler.script["system3"] = [S3_STRING]
        with pytest.raises(Exception, match="必须是 JSON 对象"):
            await role.perceive(turn(cycle="cycle-2"))

    asyncio.run(scenario())


def test_invalid_json_does_not_revoke_accepted_plan() -> None:
    """模型错误/解析失败不撤销已接受计划，也不改写已完成历史。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO, S1_SKIP],
                "system2": [S2, S2],
                "system3": [S3_GRASP, "坏 JSON"],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        # 第二轮重想时解析失败：错误传播。
        with pytest.raises(Exception, match="不是合法 JSON|必须是 JSON"):
            await role.perceive(
                turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.STARTED),))
            )
        # 已保存计划仍在：跳过路径继续复用该动作。
        sampler.script["system1"] = [S1_SKIP]
        third = await role.perceive(
            turn(cycle="cycle-3", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        assert third.reused is True
        assert [item.action_id for item in role.execution_submission().actions] == [action_id]

    asyncio.run(scenario())


def test_rejected_discards_candidate_and_regenerates() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP, S3_EMPTY]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        second = await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-1"),
                role_id=rid("lin"),
                rejected_response_id=first.response_id,
                reason="客观条件不允许",
            )
        )
        assert second.response_id != first.response_id
        assert second.actions == (), "打回后重做为空计划"
        # 打回只重调 System3（保留意向），System2 调用次数不增加。
        assert sampler.components().count("system2") == 1
        s3_prompts = [prompt for component, prompt in sampler.calls if component == "system3"]
        assert any("客观条件不允许" in prompt for prompt in s3_prompts)

    asyncio.run(scenario())


def test_rejected_validates_stale_foreign_or_wrong_cycle_feedback() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP]})
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        with pytest.raises(ValueError, match="他人"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("shen"),
                    rejected_response_id=first.response_id,
                    reason="他人反馈",
                )
            )
        with pytest.raises(ValueError, match="周期"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-9"),
                    role_id=rid("lin"),
                    rejected_response_id=first.response_id,
                    reason="错误周期",
                )
            )
        with pytest.raises(ValueError, match="过期"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=ResponseId("r-old"),
                    reason="过期响应",
                )
            )
        with pytest.raises(ValueError, match="没有待检查"):
            role2 = ModelRole(rid("lin"), ScriptedSampler())
            await role2.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=ResponseId("r-x"),
                    reason="无待检查响应",
                )
            )

    asyncio.run(scenario())


def test_explicit_empty_plan_is_legal_and_not_leftover() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_SKIP], "system2": [S2], "system3": [S3_EMPTY]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        assert first.actions == ()
        assert first.plan is not None and first.plan.is_empty
        second = await role.perceive(turn(cycle="cycle-2"))
        assert second.actions == ()

    asyncio.run(scenario())


def test_system1_updates_cognition_only() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_SKIP]})
        role = ModelRole(rid("lin"), sampler, max_memory=2)
        await role.perceive(turn())
        assert role.memory_notes() == ("一切照旧",)
        sampler.script["system1"] = ['{"route": "continue", "note": "又一条"}']
        await role.perceive(turn(cycle="cycle-2"))
        assert role.memory_notes() == ("一切照旧", "又一条")
        sampler.script["system1"] = ['{"route": "continue", "note": "第三条"}']
        await role.perceive(turn(cycle="cycle-3"))
        assert role.memory_notes() == ("又一条", "第三条")

    asyncio.run(scenario())


def test_parse_failure_and_sampler_error_propagate() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": ["不是 JSON"]})
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="不是合法 JSON|必须是 JSON"):
            await role.perceive(turn())

        class Broken(ScriptedSampler):
            async def sample(self, prompt: str, *, component: str) -> str:
                raise TimeoutError("模型超时")

        role2 = ModelRole(rid("lin"), Broken())
        with pytest.raises(TimeoutError):
            await role2.perceive(turn())

    asyncio.run(scenario())


def test_hallucinated_id_is_error_not_silent_drop() -> None:
    async def scenario() -> None:
        bad = '[{"id": "a-不存在", "kind": "wait"}]'
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [bad]})
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="未知行动身份"):
            await role.perceive(turn())

    asyncio.run(scenario())


def test_prompts_carry_context_and_plan_identity() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP]})
        role = ModelRole(rid("lin"), sampler, background="书房里的侦探，行动谨慎。")
        rich = TurnInput(
            cycle_id=CycleId("cycle-1"),
            role_id=rid("lin"),
            simulation_time=SimulationTime(500),
            trigger=TriggerReason((EventId("door"),), False),
            observations=(Observation(EventId("door"), SimulationTime(500), "门被推开，有风"),),
            known_action_states=(known("a-old", ActionPhase.STARTED, "正在窗边，位置 2.3 米"),),
        )
        first = await role.perceive(rich)
        prompts = {component: prompt for component, prompt in sampler.calls}
        assert "书房里的侦探" in prompts["system1"]
        assert "门被推开，有风" in prompts["system2"]
        assert "位置 2.3 米" in prompts["system3"]
        assert "拿到抽屉里的钥匙" in prompts["system3"]
        # 第二轮重想时 System3 提示携带当前计划真实身份。
        action_id = str(first.actions[0].action_id)
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2_OTHER]
        sampler.script["system3"] = [S3_GRASP]
        await role.perceive(turn(cycle="cycle-2", states=(known(action_id, ActionPhase.STARTED),)))
        s3 = [prompt for component, prompt in sampler.calls if component == "system3"][-1]
        assert action_id in s3, "System3 看到当前计划的真实身份"

    asyncio.run(scenario())


def test_perceive_rejected_serialized_by_role_lock() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        sampler = ScriptedSampler(
            {"system1": [S1_SKIP, S1_GO], "system2": [S2], "system3": [S3_GRASP]},
            gate=gate,
        )
        role = ModelRole(rid("lin"), sampler)
        perceive_task = asyncio.create_task(role.perceive(turn()))
        for _ in range(1000):
            if sampler.in_flight:
                break
            await asyncio.sleep(0)
        assert sampler.in_flight == 1
        rejected_task = asyncio.create_task(
            role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=ResponseId("r-lin-1"),
                    reason="并发反馈",
                )
            )
        )
        await asyncio.sleep(0)
        assert not rejected_task.done(), "锁保证同角色串行"
        gate.set()
        first = await perceive_task
        assert first.response_id == ResponseId("r-lin-1")
        second = await rejected_task
        assert second.response_id != first.response_id

    asyncio.run(scenario())


def test_feedback_action_ids_must_be_in_pending_declaration() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP]})
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        with pytest.raises(ValueError, match="声明之外的行动"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=first.response_id,
                    reason="引用无关行动",
                    action_ids=(ActionId("a-无关"),),
                )
            )

    asyncio.run(scenario())


def test_response_identity_allocated_before_stages_and_logged() -> None:
    async def scenario() -> None:
        from concordia.utils import structured_logging

        from sceneweave.integrations.concordia.tracing import SimulationLogTracer

        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP, S3_GRASP]}
        )
        role = ModelRole(rid("lin"), sampler, tracer=tracer)

        def records_by_stage() -> dict[str, list[dict]]:
            grouped: dict[str, list[dict]] = {}
            for entry in log.entries:
                data = dict(log.reconstruct_value(entry.deduplicated_data))
                grouped.setdefault(entry.component_name, []).append(data)
            return grouped

        first = await role.perceive(turn())
        grouped = records_by_stage()
        assert grouped["system1.decide"][0]["response_id"] == str(first.response_id)
        assert grouped["role.compose"][0]["response_id"] == str(first.response_id)
        assert grouped["role.compose"][0]["action_ids"]

        second = await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-1"),
                role_id=rid("lin"),
                rejected_response_id=first.response_id,
                reason="客观条件不允许",
            )
        )
        assert str(second.response_id) == "r-lin-2"
        grouped = records_by_stage()
        redo_s3 = [r for r in grouped["system3.declare"] if r["response_id"] == "r-lin-2"]
        assert redo_s3, "打回只重调 System3，使用新响应身份"
        assert all(r["response_id"] in {"r-lin-1", "r-lin-2"} for r in grouped["system1.decide"])
        sampler.script["system1"] = [S1_SKIP]
        skipped = await role.perceive(turn(cycle="cycle-2"))
        grouped = records_by_stage()
        for stage in ("system2.skipped", "system3.skipped"):
            assert grouped[stage][0]["response_id"] == skipped.response_id
        assert sampler.components().count("system2") == 1, "打回不重选意向，跳过也不触发调用"

    asyncio.run(scenario())


def test_failed_rejection_invalidates_consumed_feedback() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2], "system3": [S3_GRASP, "坏 JSON"]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        feedback = Rejection(first.cycle_id, first.role_id, first.response_id, "重新决定")
        with pytest.raises(ValueError, match="不是合法 JSON"):
            await role.rejected(feedback)
        with pytest.raises(ValueError, match="没有待检查"):
            await role.rejected(feedback)

    asyncio.run(scenario())


def test_logging_failure_is_visible_and_preserves_model_error() -> None:
    class BrokenTracer:
        def trace(self, record) -> None:
            raise RuntimeError("日志不可写")

    async def scenario() -> None:
        role = ModelRole(rid("lin"), ScriptedSampler({"system1": [S1_SKIP]}), tracer=BrokenTracer())
        with pytest.raises(RuntimeError, match="日志不可写"):
            await role.perceive(turn())
        role = ModelRole(
            rid("lin"), ScriptedSampler({"system1": ["坏 JSON"]}), tracer=BrokenTracer()
        )
        with pytest.warns(RuntimeWarning, match="错误日志写入失败"):
            with pytest.raises(ValueError, match="不是合法 JSON"):
                await role.perceive(turn())

    asyncio.run(scenario())


def test_perceive_refuses_foreign_input() -> None:
    async def scenario() -> None:
        role = ModelRole(rid("lin"), ScriptedSampler({"system1": [S1_SKIP]}))
        bad_input = TurnInput(
            cycle_id=CycleId("cycle-1"),
            role_id=rid("shen"),
            simulation_time=SimulationTime(0),
            trigger=TriggerReason((EventId("e"),), False),
            observations=(Observation(EventId("e"), SimulationTime(0), "外来输入"),),
        )
        with pytest.raises(ValueError, match="不一致"):
            await role.perceive(bad_input)

    asyncio.run(scenario())


def test_same_id_target_change_is_rejected_at_candidate_boundary() -> None:
    """同 ID 改变目标/参数在候选检查边界明确拒绝，旧保存计划不被覆写。"""

    async def scenario() -> None:
        changed = '[{"id": "a-lin-1", "kind": "grasp", "target": "door", "effector": "right_hand"}]'
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_GO], "system2": [S2, S2], "system3": [S3_GRASP, changed]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        before = role.saved_plan_digest()
        with pytest.raises(Exception, match="不能改变目标或参数"):
            await role.perceive(turn(cycle="cycle-2"))
        assert role.saved_plan_digest() == before, "拒绝后旧保存计划不变"

    asyncio.run(scenario())


def test_same_id_dependency_change_is_rejected_at_candidate_boundary() -> None:
    """同 ID 改变依赖在候选检查边界明确拒绝。"""

    async def scenario() -> None:
        changed = (
            '[{"id": "a-lin-2", "kind": "grasp", "target": "key", "effector": "right_hand",'
            ' "depends_on": ["a-lin-1"]}]'
        )
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_GO], "system2": [S2, S2], "system3": [S3_TWO, changed]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        await save_plan(
            role,
            first,
            [
                (str(first.actions[0].action_id), 100, 200),
                (str(first.actions[1].action_id), 100, 200),
            ],
        )
        before = role.saved_plan_digest()
        with pytest.raises(Exception, match="不能改变依赖"):
            await role.perceive(turn(cycle="cycle-2"))
        assert role.saved_plan_digest() == before

    asyncio.run(scenario())


def test_pure_maintenance_same_payload_does_not_restart() -> None:
    """同 ID 同参数重声明是纯维持：允许且不重新携带排程、不重启。"""

    async def scenario() -> None:
        same = '[{"id": "a-lin-1", "kind": "grasp", "target": "key", "effector": "right_hand"}]'
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_GO], "system2": [S2, S2], "system3": [S3_GRASP, same]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        assert [item.action_id for item in role.execution_submission().actions] == [action_id]
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        assert second.reused is False
        await save_plan(role, second, [(str(action_id), 100, 200)])
        submission = role.execution_submission()
        assert [item.action_id for item in submission.actions] == [action_id]
        assert submission.schedules == (), "已有动作不重新计时，保持原安排"

    asyncio.run(scenario())


def test_save_rejects_stale_or_mismatched_candidate() -> None:
    """保存必须对应本角色最新待检查候选；过期/错配 payload 拒绝且不覆写旧计划。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO, S1_GO], "system2": [S2, S2], "system3": [S3_GRASP, S3_GRASP]}
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        saved = role.saved_plan_digest()

        # 未经过 perceive/rejected 的旧候选：拒绝保存，旧计划不变。
        stale = PlanAcceptance(
            declaration=first,
            plan=first.plan,
            schedules=(
                ActionSchedule(
                    ActionId(str(action_id)),
                    CommitGroupId("g"),
                    SimulationTime(100),
                    SimulationTime(200),
                ),
            ),
            check=ActionCheckResult(
                first.cycle_id,
                first.role_id,
                first.response_id,
                WorldStateVersion(0),
                True,
            ),
        )
        with pytest.raises(ValueError, match="没有待检查的候选"):
            await role.save_plan(stale)
        assert role.saved_plan_digest() == saved

    asyncio.run(scenario())


def test_plan_digest_exposes_saved_schedule_to_system1_and_system3() -> None:
    """System1/System3 的角色私有摘要包含已保存时序，不只身份/阶段。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO],
                "system2": [S2, S2],
                "system3": [S3_GRASP, S3_GRASP],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 60000, 120000)])
        await role.perceive(
            turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.STARTED),))
        )
        system1_prompts = [prompt for component, prompt in sampler.calls if component == "system1"]
        system3_prompts = [prompt for component, prompt in sampler.calls if component == "system3"]
        assert "60000" in system1_prompts[-1] and "120000" in system1_prompts[-1]
        assert "60000" in system3_prompts[-1] and "120000" in system3_prompts[-1]
        assert "已保存排程" in system1_prompts[-1]

    asyncio.run(scenario())


def test_plan_acceptance_requires_plan_to_equal_declaration() -> None:
    """PlanAcceptance 必须保证保存链与候选声明完全一致（同 IDs 不同 payload 也拒绝）。"""

    first = RoleResponse(
        CycleId("cycle-1"),
        rid("lin"),
        ResponseId("r-lin-1"),
        (
            StepAction(
                ActionId("a-lin-1"),
                AtomicAction(kind=ActionKind.GRASP, target="key", effector=Effector.RIGHT_HAND),
            ),
        ),
    )
    different = ActionPlan(
        actions=(
            PlanAction(
                action=AtomicAction(
                    kind=ActionKind.GRASP, target="door", effector=Effector.RIGHT_HAND
                ),
                action_id=ActionId("a-lin-1"),
            ),
        )
    )
    schedules = (
        ActionSchedule(
            ActionId("a-lin-1"),
            CommitGroupId("g"),
            SimulationTime(100),
            SimulationTime(200),
        ),
    )
    with pytest.raises(ValueError, match="必须与候选声明的 plan 完全一致"):
        PlanAcceptance(
            declaration=first,
            plan=different,
            schedules=schedules,
            check=ActionCheckResult(
                CycleId("cycle-1"), rid("lin"), ResponseId("r-lin-1"), WorldStateVersion(0), True
            ),
        )
