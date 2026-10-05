"""模型驱动角色的内部编排行为测试（脚本采样器，无真实模型）。

覆盖：System1 跳过仍完整维持且终态排除、认知单独更新、System2→System3
与两类回退、内部上限、打回校验与重新决定、显式空、错误不伪空、
身份唯一与不复用。
"""

import asyncio

import pytest

from sceneweave.domain.identity import ActionId, CycleId, EventId, ResponseId
from sceneweave.domain.messages import (
    ActionPhase,
    KnownActionState,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Observation,
    Rejection,
    TriggerReason,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.roles.model_role import ModelRole

from .fakes import ScriptedSampler, rid

S1_SKIP = '{"reconsider": false, "note": "一切照旧"}'
S1_GO = '{"reconsider": true, "note": "情况有变"}'
S2 = '{"intention": "拿到抽屉里的钥匙", "abandon": []}'
S2_OTHER = '{"intention": "改去找备用钥匙", "abandon": []}'
S3_NEW = '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}'
S3_NEW_IMPOSSIBLE = (
    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["不可能地隔空取物"]}'
)
S3_EMPTY = '{"feasible": true, "maintain": [], "modify": [], "new_actions": []}'
S3_KEEP = '{"feasible": true, "maintain": ["__KEEP__"], "modify": [], "new_actions": []}'
S3_INFEASIBLE = (
    '{"feasible": false, "reason": "抽屉锁着，意向无法落实",'
    ' "maintain": [], "modify": [], "new_actions": []}'
)
S3_RETRY = (
    '{"feasible": true, "retry": true, "reason": "换个姿势",'
    ' "maintain": [], "modify": [], "new_actions": []}'
)


def turn(*, cycle: str = "cycle-1", time: int = 100, states: tuple = ()) -> TurnInput:
    return TurnInput(
        cycle_id=CycleId(cycle),
        role_id=rid("lin"),
        simulation_time=SimulationTime(time),
        trigger=TriggerReason((), bool(states) is False and False),
        known_action_states=states,
    )


def known(action_id: str, phase: ActionPhase, detail: str = "状态") -> KnownActionState:
    return KnownActionState(ActionId(action_id), phase, detail)


def test_system1_skip_returns_full_maintain_set_without_terminal() -> None:
    """不重新考虑：返回完整维持集合；终态行动被对齐排除。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO, S1_SKIP, S1_SKIP]})
        role = ModelRole(rid("lin"), sampler)
        # 第一轮：注册一个新行动（走 System2/3）。
        sampler.script["system2"] = [S2]
        sampler.script["system3"] = [S3_NEW]
        first = await role.perceive(turn())
        assert isinstance(first.actions[0], NewAction)
        new_id = first.actions[0].action_id
        # 第二轮：行动已开始，System1 跳过 → 完整维持集合。
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(new_id), ActionPhase.STARTED),))
        )
        assert sampler.components() == [
            "system1",
            "system2",
            "system3",
            "system1",
        ], "跳过时不再调用 System2/3"
        assert [type(item) for item in second.actions] == [MaintainAction]
        assert second.actions[0].action_id == new_id, "仍在维持的行动保留"
        # 第三轮：行动已完成（终态）→ 不再出现在维持集合。
        third = await role.perceive(
            turn(cycle="cycle-3", states=(known(str(new_id), ActionPhase.COMPLETED),))
        )
        assert third.actions == (), "终态行动不继续维持"
        digest = dict((row[0], row[3]) for row in role.own_action_digest())
        assert digest[str(new_id)] is False

    asyncio.run(scenario())


def test_system1_updates_cognition_only() -> None:
    """System1 只更新认知：记忆有界保存，不触发后续组件。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_SKIP]})
        role = ModelRole(rid("lin"), sampler, max_memory=2)
        await role.perceive(turn())
        assert role.memory_notes() == ("一切照旧",)
        sampler.script["system1"] = ['{"reconsider": false, "note": "又一条"}']
        await role.perceive(turn(cycle="cycle-2"))
        # 记忆上限 2：最早的记录被淘汰。
        assert role.memory_notes() == ("一切照旧", "又一条")
        sampler.script["system1"] = ['{"reconsider": false, "note": "第三条"}']
        await role.perceive(turn(cycle="cycle-3"))
        assert role.memory_notes() == ("又一条", "第三条")

    asyncio.run(scenario())


def test_system2_to_system3_produces_new_parallel_group() -> None:
    """System2→System3：新行动获得稳定身份与一个并行提交组。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_NEW]})
        role = ModelRole(rid("lin"), sampler)
        response = await role.perceive(turn())
        new_actions = [item for item in response.actions if isinstance(item, NewAction)]
        assert len(new_actions) == 1
        assert new_actions[0].description == "走向书桌"
        assert str(new_actions[0].action_id).startswith("a-lin-")
        assert str(new_actions[0].commit_group_id).startswith("g-lin-")
        assert response.response_id == ResponseId("r-lin-1")
        assert response.cycle_id == CycleId("cycle-1")

    asyncio.run(scenario())


def test_system3_infeasible_feeds_back_to_system2() -> None:
    """意向不可行反馈 System2 重选，再由 System3 落实。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2, S2_OTHER],
                "system3": [S3_INFEASIBLE, S3_NEW],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        response = await role.perceive(turn())
        assert sampler.components() == [
            "system1",
            "system2",
            "system3",
            "system2",
            "system3",
        ]
        new_actions = [item for item in response.actions if isinstance(item, NewAction)]
        assert new_actions and new_actions[0].description == "走向书桌"

    asyncio.run(scenario())


def test_system3_redo_same_intention_with_retry() -> None:
    """做法不可行只重做 System3，不重选意向。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2], "system3": [S3_RETRY, S3_NEW]}
        )
        role = ModelRole(rid("lin"), sampler)
        response = await role.perceive(turn())
        assert sampler.components() == ["system1", "system2", "system3", "system3"]
        assert isinstance(response.actions[0], NewAction)

    asyncio.run(scenario())


def test_system3_retry_cap_is_error_not_empty() -> None:
    """内部重试超过上限：错误传播，不转换为空声明。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2],
                "system3": [S3_RETRY, S3_RETRY, S3_RETRY, S3_RETRY],
            }
        )
        role = ModelRole(rid("lin"), sampler, max_system3_calls=3)
        with pytest.raises(RuntimeError, match="落实尝试次数"):
            await role.perceive(turn())
        assert len(sampler.calls) == 1 + 1 + 3, "上限后不再调用模型"

    asyncio.run(scenario())


def test_system2_reselect_cap_is_error_not_empty() -> None:
    """意向重选超过上限：错误传播。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2, S2_OTHER, S2],
                "system3": [S3_INFEASIBLE, S3_INFEASIBLE, S3_INFEASIBLE],
            }
        )
        role = ModelRole(rid("lin"), sampler, max_system2_calls=2)
        with pytest.raises(RuntimeError, match="意向选择次数"):
            await role.perceive(turn())
        assert len(sampler.calls) == 1 + 2 + 2, "上限后不再调用模型"

    asyncio.run(scenario())


def test_parse_failure_propagates_as_error() -> None:
    """模型输出无法解析：错误传播，不伪空。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": ["不是 JSON"]})
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="不是合法 JSON|必须是 JSON"):
            await role.perceive(turn())

    asyncio.run(scenario())


def test_sampler_error_propagates() -> None:
    """采样器异常如实传播（超时/错误不吞）。"""

    async def scenario() -> None:
        class Broken(ScriptedSampler):
            async def sample(self, prompt: str, *, component: str) -> str:
                raise TimeoutError("模型超时")

        role = ModelRole(rid("lin"), Broken())
        with pytest.raises(TimeoutError):
            await role.perceive(turn())

    asyncio.run(scenario())


def test_rejected_validates_and_returns_new_complete_declaration() -> None:
    """打回只接受当前待检查响应；重新决定返回新的完整声明。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2, S2],
                "system3": [S3_NEW_IMPOSSIBLE, S3_NEW],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        second = await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-1"),
                role_id=rid("lin"),
                rejected_response_id=first.response_id,
                reason="客观条件不允许该行动",
            )
        )
        assert second.response_id != first.response_id, "新响应身份"
        assert isinstance(second.actions[0], NewAction)
        assert second.actions[0].description == "走向书桌", "打回原因驱动重做"
        # 响应身份在感知与打回之间全局唯一。
        assert {first.response_id, second.response_id} == {
            ResponseId("r-lin-1"),
            ResponseId("r-lin-2"),
        }

    asyncio.run(scenario())


def test_rejected_refuses_stale_foreign_or_wrong_cycle_feedback() -> None:
    """过期/他人/错误周期的反馈被拒绝。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_NEW]})
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


def test_explicit_empty_and_maintain_modify_types() -> None:
    """显式空合法；维持与修改在类型上可区分；新身份不复用。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO, S1_SKIP],
                "system2": [S2, S2],
                "system3": [S3_NEW, S3_EMPTY],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        new_id = first.actions[0].action_id
        # 显式空：撤回全部主动维持。
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(new_id), ActionPhase.STARTED),))
        )
        assert second.actions == (), "显式空是合法完整声明"
        # 显式空之后角色信念在下一轮对齐时淘汰该行动；跳过路径也返回空。
        third = await role.perceive(turn(cycle="cycle-3"))
        assert third.actions == (), "无可维持行动时跳过路径为显式空"
        digest = dict((row[0], row[3]) for row in role.own_action_digest())
        assert digest[str(new_id)] is False, "显式空之后不再维持"

    asyncio.run(scenario())


S3_MAINTAIN_MODIFY = (
    '{"feasible": true, "maintain": ["__MAINTAIN__"],'
    ' "modify": [{"id": "__MODIFY__", "change": "改为更慢的速度"}],'
    ' "new_actions": ["再试一次"]}'
)


def test_maintain_modify_and_new_types_distinguished() -> None:
    """维持/修改/新增三类声明类型准确，身份稳定不复用。"""

    async def scenario() -> None:
        # 第一轮注册两个新行动。
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [],'
                    ' "new_actions": ["行动甲", "行动乙"]}'
                ],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        ids = [item.action_id for item in first.actions]
        plan = S3_MAINTAIN_MODIFY.replace("__MAINTAIN__", str(ids[0])).replace(
            "__MODIFY__", str(ids[1])
        )
        sampler.script["system3"] = [plan]
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2]
        second = await role.perceive(
            turn(
                cycle="cycle-2",
                states=(
                    known(str(ids[0]), ActionPhase.STARTED),
                    known(str(ids[1]), ActionPhase.STARTED),
                ),
            )
        )
        types = {type(item) for item in second.actions}
        assert types == {MaintainAction, ModifiedAction, NewAction}
        assert second.actions[0].action_id == ids[0], "维持身份稳定"
        assert isinstance(second.actions[1], ModifiedAction)
        assert second.actions[1].action_id == ids[1]
        assert str(second.actions[2].action_id) not in {str(i) for i in ids}, "新身份不复用"

    asyncio.run(scenario())


def test_hallucinated_reference_is_error_not_silent_drop() -> None:
    """引用未知或已终态行动：明确错误，不悄悄丢弃。"""

    async def scenario() -> None:
        bad = '{"feasible": true, "maintain": ["a-不存在"], "modify": [], "new_actions": []}'
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [bad]})
        role = ModelRole(rid("lin"), sampler)
        with pytest.raises(Exception, match="未知行动"):
            await role.perceive(turn())

    asyncio.run(scenario())


def test_abandon_is_candidate_content_not_irrevocable_prune() -> None:
    """放弃属于候选内容：打回后可重新维持；确认撤回后才退出可维持。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO, S1_SKIP, S1_SKIP],
                "system2": [S2],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["某行动"]}'
                ],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        new_id = first.actions[0].action_id
        started = turn(cycle="cycle-2", states=(known(str(new_id), ActionPhase.STARTED),))
        # 候选一：放弃该行动并提交空集合，随后被 World 打回。
        sampler.script["system2"] = [
            '{"intention": "放弃之前的目标", "abandon": ["' + str(new_id) + '"]}',
            S2,
        ]
        keep_plan = S3_KEEP.replace("__KEEP__", str(new_id))
        sampler.script["system3"] = [S3_EMPTY, keep_plan]
        candidate = await role.perceive(started)
        assert candidate.actions == (), "候选一显式空"
        redo = await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-2"),
                role_id=rid("lin"),
                rejected_response_id=candidate.response_id,
                reason="客观条件不允许",
            )
        )
        # 候选一被打回：重试可以重新决定维持原行动（未被提前剪枝）。
        assert isinstance(redo.actions[0], MaintainAction)
        assert redo.actions[0].action_id == new_id
        digest = dict((row[0], row[3]) for row in role.own_action_digest())
        assert digest[str(new_id)] is True, "打回候选不提前影响原安排"
        # 显式空被接受后（下一轮已知状态不再出现）：才退出可维持。
        sampler.script["system3"] = [S3_EMPTY]
        await role.perceive(
            turn(
                cycle="cycle-3",
                states=(known(str(new_id), ActionPhase.STARTED),),
            )
        )
        await role.perceive(turn(cycle="cycle-4"))
        digest = dict((row[0], row[3]) for row in role.own_action_digest())
        assert digest[str(new_id)] is False, "撤回确认后才不可维持"

    asyncio.run(scenario())


def test_perceive_refuses_foreign_input() -> None:
    """输入角色身份不一致被拒绝。"""

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


def test_candidate_new_action_not_maintainable_in_same_cycle_retry() -> None:
    """候选新行动未经引擎确认：同周期重试里不能当作已登记行动维持。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2, S2],
                "system3": [S3_NEW],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        candidate_id = str(first.actions[0].action_id)
        sampler.script["system3"].append(
            '{"feasible": true, "maintain": ["' + candidate_id + '"],'
            ' "modify": [], "new_actions": []}'
        )
        with pytest.raises(Exception, match="尚未经引擎登记确认"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=first.response_id,
                    reason="客观条件不允许",
                )
            )
        # 候选身份保留（防复用）但不可维持。
        digest = dict((row[0], row[3]) for row in role.own_action_digest())
        assert digest[candidate_id] is False

    asyncio.run(scenario())


def test_pending_modification_committed_only_after_confirmation() -> None:
    """明确修改：被接受的声明在下一轮进入有效做法；被打回的不生效。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["快跑"]}'
                ],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        new_id = str(first.actions[0].action_id)
        slow = (
            '{"feasible": true, "maintain": [],'
            ' "modify": [{"id": "' + new_id + '", "change": "放慢速度"}],'
            ' "new_actions": []}'
        )
        keep = '{"feasible": true, "maintain": ["' + new_id + '"], "modify": [], "new_actions": []}'
        # 候选一（修改）被打回 → 候选二维持原行动：修改未生效。
        sampler.script["system1"] = [S1_GO]
        sampler.script["system2"] = [S2, S2]
        sampler.script["system3"] = [slow, keep]
        started = turn(cycle="cycle-2", states=(known(new_id, ActionPhase.STARTED),))
        candidate = await role.perceive(started)
        assert isinstance(candidate.actions[0], ModifiedAction)
        redo = await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-2"),
                role_id=rid("lin"),
                rejected_response_id=candidate.response_id,
                reason="客观条件不允许",
            )
        )
        assert isinstance(redo.actions[0], MaintainAction), "打回的修改不作为已生效"
        # 被接受的修改：提交的下一轮确认后进入有效做法的私人认知。
        sampler.script["system1"] = [S1_GO, S1_SKIP]
        sampler.script["system2"] = [S2]
        sampler.script["system3"] = [slow]
        await role.perceive(turn(cycle="cycle-3", states=(known(new_id, ActionPhase.STARTED),)))
        digest = dict((row[0], row[2]) for row in role.own_action_digest())
        assert "放慢速度" not in digest[new_id], "提交当轮尚未确认"
        await role.perceive(turn(cycle="cycle-4", states=(known(new_id, ActionPhase.STARTED),)))
        digest = dict((row[0], row[2]) for row in role.own_action_digest())
        assert "放慢速度" in digest[new_id], "确认后保留新做法认知"

    asyncio.run(scenario())


def test_prompts_carry_full_context_and_background() -> None:
    """System2/3 提示携带原始观测、已知状态细节、私人认知与背景。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO],
                "system2": [S2],
                "system3": [S3_NEW],
            }
        )
        role = ModelRole(
            rid("lin"),
            sampler,
            background="书房里的侦探，行动谨慎。",
        )
        rich = TurnInput(
            cycle_id=CycleId("cycle-1"),
            role_id=rid("lin"),
            simulation_time=SimulationTime(500),
            trigger=TriggerReason((EventId("door"),), False),
            observations=(Observation(EventId("door"), SimulationTime(500), "门被推开，有风"),),
            known_action_states=(known("a-old", ActionPhase.STARTED, "正在窗边，位置 2.3 米"),),
        )
        await role.perceive(rich)
        prompts = {component: prompt for component, prompt in sampler.calls}
        assert "书房里的侦探" in prompts["system1"]
        assert "门被推开，有风" in prompts["system2"], "原始观测进入 System2"
        assert "位置 2.3 米" in prompts["system2"], "已知状态细节进入 System2"
        assert "门被推开，有风" in prompts["system3"], "原始观测进入 System3"
        assert "位置 2.3 米" in prompts["system3"]
        assert "拿到抽屉里的钥匙" in prompts["system3"], "意向进入 System3"
        assert "情况有变" in prompts["system3"], "私人认知进入落实阶段"

    asyncio.run(scenario())


def test_reasons_flow_to_next_model_calls() -> None:
    """打回原因到 System2/3；重做原因与不可行反馈沿链传递。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [S1_GO, S1_GO],
                "system2": [S2, S2, S2_OTHER],
                "system3": [S3_NEW_IMPOSSIBLE, S3_RETRY, S3_INFEASIBLE, S3_NEW],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        # 打回后的重做：System2 收到打回原因。
        await role.rejected(
            Rejection(
                cycle_id=CycleId("cycle-1"),
                role_id=rid("lin"),
                rejected_response_id=first.response_id,
                reason="距离太远，无法立即行动",
            )
        )
        prompts = [prompt for component, prompt in sampler.calls if component == "system2"]
        assert any("距离太远" in prompt for prompt in prompts), "打回原因进入 System2"
        s3_prompts = [prompt for component, prompt in sampler.calls if component == "system3"]
        assert any("距离太远" in prompt for prompt in s3_prompts), "打回原因进入 System3"
        assert any("换个姿势" in prompt for prompt in s3_prompts), "重做原因传递"
        assert any("抽屉锁着" in prompt for prompt in s3_prompts), (
            "不可行反馈传递到重选后的 System3"
        )

    asyncio.run(scenario())


def test_perceive_rejected_serialized_by_role_lock() -> None:
    """同角色公共方法串行：rejected 等待 perceive 完成后才执行。"""

    async def scenario() -> None:
        gate = asyncio.Event()
        sampler = ScriptedSampler(
            {"system1": [S1_SKIP, S1_GO], "system2": [S2], "system3": [S3_NEW]},
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
        # perceive 完成后 rejected 才执行：此时反馈有效（不再是过期）。
        second = await rejected_task
        assert second.response_id != first.response_id

    asyncio.run(scenario())


def test_new_perceive_invalidates_old_pending_feedback() -> None:
    """新 perceive 开始后旧反馈失效；采样失败后不接受旧周期反馈。"""

    async def scenario() -> None:
        class FailingThenSkip(ScriptedSampler):
            def __init__(self) -> None:
                super().__init__({"system1": [S1_SKIP]})
                self.failed_once = False

            async def sample(self, prompt: str, *, component: str) -> str:
                if not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("第一次采样失败")
                return await super().sample(prompt, component=component)

        role = ModelRole(rid("lin"), FailingThenSkip())
        with pytest.raises(RuntimeError, match="第一次采样失败"):
            await role.perceive(turn())
        # 失败后没有待检查响应。
        with pytest.raises(ValueError, match="没有待检查"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-1"),
                    role_id=rid("lin"),
                    rejected_response_id=ResponseId("r-any"),
                    reason="过期",
                )
            )
        # 成功 perceive 后，旧响应身份的反馈同样失效（周期一致也拒绝）。
        ok = await role.perceive(turn(cycle="cycle-2"))
        assert ok.response_id == ResponseId("r-lin-2")
        with pytest.raises(ValueError, match="过期响应"):
            await role.rejected(
                Rejection(
                    cycle_id=CycleId("cycle-2"),
                    role_id=rid("lin"),
                    rejected_response_id=ResponseId("r-lin-1"),
                    reason="旧身份",
                )
            )

    asyncio.run(scenario())


def test_feedback_action_ids_must_be_in_pending_declaration() -> None:
    """反馈引用的行动必须属于当前待检查声明。"""

    async def scenario() -> None:
        sampler = ScriptedSampler({"system1": [S1_GO], "system2": [S2], "system3": [S3_NEW]})
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
    """响应身份在阶段开始前分配并贯穿各阶段、重做与最终组装。"""

    async def scenario() -> None:
        from concordia.utils import structured_logging

        from sceneweave.integrations.concordia.tracing import SimulationLogTracer

        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2, S2], "system3": [S3_NEW, S3_NEW]}
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
        s1 = grouped["system1.decide"][0]
        assert s1["response_id"] == str(first.response_id), "身份先于阶段分配"
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
        redo_s2 = [r for r in grouped["system2.intend"] if r["response_id"] == "r-lin-2"]
        assert redo_s2, "重做阶段使用新响应身份"
        assert all(r["response_id"] in {"r-lin-1", "r-lin-2"} for r in grouped["system1.decide"])
        sampler.script["system1"] = [S1_SKIP]
        skipped = await role.perceive(turn(cycle="cycle-2"))
        grouped = records_by_stage()
        for stage in ("system2.skipped", "system3.skipped"):
            assert grouped[stage][0]["response_id"] == skipped.response_id
        assert sampler.components().count("system2") == 2, "跳过记录不触发额外模型调用"

    asyncio.run(scenario())


def test_failed_rejection_invalidates_consumed_feedback() -> None:
    """打回后采样失败：原反馈已消费，不能用旧响应身份再次重做。"""

    async def scenario() -> None:
        sampler = ScriptedSampler(
            {"system1": [S1_GO], "system2": [S2, "坏 JSON"], "system3": [S3_NEW]}
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
