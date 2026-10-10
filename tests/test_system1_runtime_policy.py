"""System1 正式三路倾向分布的运行策略测试（脚本假模型，无真实模型）。

覆盖：严格分布解析、结构 mask 与零剩余错误、已保存空计划可 continue、
确定性/采样两个阈值分支与并列、随机可复现与角色隔离、ModelRole 不绕过
依赖、replan 成功保存后 continue 不重复重排、待确认期间仍可 rethink、
完整与错误策略 trace 的留痕。
"""

import asyncio
import json

import pytest

from sceneweave.domain.action_plan import ActionKind
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    WorldStateVersion,
)
from sceneweave.domain.messages import (
    ActionCheckResult,
    ActionPhase,
    KnownActionState,
    TriggerReason,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionSchedule, PlanAcceptance
from sceneweave.roles.components.system1_policy import (
    allowed_routes,
    decide_route,
    derive_route_rng,
)
from sceneweave.roles.model_output import (
    ModelOutputError,
    System1Route,
    System1RoutePolicy,
    parse_system1_tendency,
)
from sceneweave.roles.model_role import ModelRole

from .fakes import ScriptedSampler, rid, s1_continue, s1_replan, s1_rethink


def tendency(continue_: float, replan: float, rethink: float, note: str = "说明"):
    return parse_system1_tendency(
        json.dumps(
            {
                "probabilities": {
                    "continue": continue_,
                    "replan": replan,
                    "rethink": rethink,
                },
                "note": note,
            }
        )
    )


def test_old_route_schema_and_bad_probabilities_are_rejected() -> None:
    with pytest.raises(ModelOutputError, match="未知字段"):
        parse_system1_tendency('{"route": "continue", "note": "旧"}')
    with pytest.raises(ModelOutputError, match="不能是 bool 或字符串"):
        parse_system1_tendency(
            '{"probabilities": {"continue": true, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="必须在 0..1"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 2.0, "replan": 0.0, "rethink": 0.0}, "note": "x"}'
        )
    with pytest.raises(ModelOutputError, match="之和必须约为 1"):
        parse_system1_tendency(
            '{"probabilities": {"continue": 0.4, "replan": 0.4, "rethink": 0.4}, "note": "x"}'
        )


def test_no_intention_forces_rethink_even_with_zero_rethink_mass() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(1.0, 0.0, 0.0),
        has_intention=False,
        has_saved_plan=False,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.route is System1Route.RETHINK
    assert decision.method == "forced" and decision.reason == "no_intention"
    assert decision.random_value is None


def test_continue_masked_without_saved_plan_and_zero_remaining_errors() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    # 有意向但没有已保存计划：continue 不合法；全部权重在 continue 时零剩余报错。
    with pytest.raises(ModelOutputError, match="权重之和为 0"):
        decide_route(
            tendency(1.0, 0.0, 0.0),
            has_intention=True,
            has_saved_plan=False,
            rng=rng,
            policy=System1RoutePolicy(),
            route_seed=0,
            seed_hash=seed_hash,
        )


def test_mask_renormalizes_over_legal_paths_when_continue_masked() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(0.6, 0.4, 0.0),
        has_intention=True,
        has_saved_plan=False,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.route is System1Route.REPLAN
    assert decision.legal_weights[System1Route.REPLAN] == pytest.approx(1.0)
    assert set(decision.legal_weights) == {System1Route.REPLAN, System1Route.RETHINK}


def test_saved_empty_plan_allows_continue() -> None:
    """已保存的 [] 是有效空计划：has_saved_plan=True 时 continue 合法。"""
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(1.0, 0.0, 0.0),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.route is System1Route.CONTINUE
    assert decision.method == "deterministic"


def test_certainty_and_margin_threshold_branches() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    # 最大权重 >= 0.75：确定选择。
    by_max = decide_route(
        tendency(0.8, 0.15, 0.05),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert by_max.route is System1Route.CONTINUE
    assert by_max.method == "deterministic" and by_max.random_value is None
    # 最大权重 < 0.75 但边际 >= 0.25：确定选择。
    by_margin = decide_route(
        tendency(0.6, 0.3, 0.1),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert by_margin.route is System1Route.CONTINUE
    assert by_margin.method == "deterministic"
    assert by_margin.margin == pytest.approx(0.3)


def test_below_threshold_is_sampled_and_consumes_random() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(0.5, 0.3, 0.2),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.method == "sampled" and decision.reason == "below_threshold"
    assert decision.random_value is not None and 0.0 <= decision.random_value < 1.0


def test_tie_enters_sampling_not_traversal_order() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(0.4, 0.4, 0.2),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.method == "sampled" and decision.reason == "tie"
    assert decision.random_value is not None


def test_custom_thresholds_are_injectable() -> None:
    rng, seed_hash = derive_route_rng(0, "lin")
    decision = decide_route(
        tendency(0.5, 0.3, 0.2),
        has_intention=True,
        has_saved_plan=True,
        rng=rng,
        policy=System1RoutePolicy(certainty_threshold=0.4, margin_threshold=0.9),
        route_seed=0,
        seed_hash=seed_hash,
    )
    assert decision.route is System1Route.CONTINUE
    assert decision.method == "deterministic"
    assert decision.certainty_threshold == 0.4


def test_random_source_is_reproducible_and_role_isolated() -> None:
    body = tendency(0.5, 0.3, 0.2)
    policy = System1RoutePolicy()

    def draw(role: str, seed: int) -> tuple[System1Route, float | None]:
        rng, seed_hash = derive_route_rng(seed, role)
        decision = decide_route(
            body,
            has_intention=True,
            has_saved_plan=True,
            rng=rng,
            policy=policy,
            route_seed=seed,
            seed_hash=seed_hash,
        )
        return decision.route, decision.random_value

    # 相同种子 / 身份 / 策略复现相同抽值。
    assert draw("lin", 7) == draw("lin", 7)
    # 不同角色派生独立序列：抽值不同。
    lin_draw = draw("lin", 7)
    shen_draw = draw("shen", 7)
    assert lin_draw[1] != shen_draw[1]


def test_derive_route_rng_validates_seed_and_role() -> None:
    with pytest.raises(ValueError, match="route_seed"):
        derive_route_rng(True, "lin")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="角色身份"):
        derive_route_rng(0, "")


def test_allowed_routes_reflect_structural_state() -> None:
    assert allowed_routes(has_intention=False, has_saved_plan=False) == (System1Route.RETHINK,)
    assert allowed_routes(has_intention=True, has_saved_plan=False) == (
        System1Route.REPLAN,
        System1Route.RETHINK,
    )
    assert allowed_routes(has_intention=True, has_saved_plan=True) == (
        System1Route.CONTINUE,
        System1Route.REPLAN,
        System1Route.RETHINK,
    )


# ---- ModelRole 编排与真实结构状态 ----


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


S2_A = '{"intention": "拿到钥匙"}'
S3_GRASP = '[{"kind": "grasp", "target": "key", "effector": "right_hand"}]'
S3_WAIT = '[{"kind": "wait"}]'


def test_replan_saves_then_continue_does_not_replan_again() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [s1_rethink("开始"), s1_replan("换做法"), s1_continue("照旧")],
                "system2": [S2_A],
                "system3": [S3_GRASP, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        await save_plan(role, first, [(str(first.actions[0].action_id), 100, 200)])
        second = await role.perceive(turn(cycle="cycle-2"))
        assert second.reused is False
        assert [item.action.kind for item in second.actions] == [ActionKind.WAIT]
        await save_plan(role, second, [(str(second.actions[0].action_id), 100, 200)])
        before = len(sampler.components())
        third = await role.perceive(
            turn(
                cycle="cycle-3",
                states=(known(str(first.actions[0].action_id), ActionPhase.STARTED),),
            )
        )
        assert third.reused is True
        assert sampler.components()[before:] == ["system1"], (
            "成功 replan 保存后下一轮 continue 不重排"
        )
        assert sampler.components().count("system2") == 1, "replan 不重选意向"
        assert sampler.components().count("system3") == 2, "只有 replan 重调一次 System3"

    asyncio.run(scenario())


def test_rethink_allowed_while_result_unconfirmed() -> None:
    async def scenario() -> None:
        sampler = ScriptedSampler(
            {
                "system1": [s1_rethink("开始"), s1_rethink("结果未知，重新考虑")],
                "system2": [S2_A, S2_A],
                "system3": [S3_GRASP, S3_WAIT],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        first = await role.perceive(turn())
        action_id = first.actions[0].action_id
        await save_plan(role, first, [(str(action_id), 100, 200)])
        # 行动结果未确认：不得据此禁止 rethink。
        second = await role.perceive(
            turn(cycle="cycle-2", states=(known(str(action_id), ActionPhase.UNCONFIRMED),))
        )
        assert sampler.components().count("system2") == 2, "待确认期间仍可重选意向"
        assert [item.action.kind for item in second.actions] == [ActionKind.WAIT]

    asyncio.run(scenario())


def test_policy_zero_remaining_error_does_not_overwrite_accepted_plan() -> None:
    async def scenario() -> None:
        # 首轮 rethink 保存计划；次轮有意向但模型给 continue=1 且在屏蔽后零剩余：
        # 构造 has_saved_plan=False 需要先撤销保存——用新角色只设置意向。
        sampler = ScriptedSampler(
            {
                "system1": [s1_rethink("先想"), s1_continue("照旧")],
                "system2": [S2_A],
                "system3": [S3_GRASP],
            }
        )
        role = ModelRole(rid("lin"), sampler)
        await role.perceive(turn())
        assert role.saved_plan_digest() == ()
        with pytest.raises(ModelOutputError, match="权重之和为 0"):
            await role.perceive(turn(cycle="cycle-2"))
        assert role.saved_plan_digest() == (), "策略错误不提前产生或更改已接受计划"

    asyncio.run(scenario())


def test_system1_trace_records_full_policy_and_preserves_floats() -> None:
    async def scenario() -> None:
        from concordia.utils import structured_logging

        from sceneweave.integrations.concordia.tracing import SimulationLogTracer

        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        raw = (
            '{"probabilities": {"continue": 0.5, "replan": 0.3, "rethink": 0.2}, "note": "看情况"}'
        )
        sampler = ScriptedSampler(
            {
                "system1": [s1_rethink("先想"), raw],
                "system2": [S2_A, S2_A, S2_A],
                "system3": [S3_GRASP, S3_GRASP, S3_GRASP],
            }
        )
        role = ModelRole(rid("lin"), sampler, tracer=tracer)
        first = await role.perceive(turn())
        await save_plan(role, first, [(str(first.actions[0].action_id), 100, 200)])
        await role.perceive(turn(cycle="cycle-2"))
        entries = [e for e in log.entries if e.component_name == "system1.decide"]
        data = dict(log.reconstruct_value(entries[-1].deduplicated_data))
        assert data["inputs"][0], "完整输入（含提示）"
        assert data["outputs"][0] == raw, "原始模型 JSON"
        details = data["details"]
        assert details["raw_probabilities"]["continue"] == 0.5, "原始倾向保留 float"
        assert details["route"] in {"continue", "replan", "rethink"}
        assert details["method"] in {"forced", "deterministic", "sampled"}
        assert isinstance(details["legal_weights"], dict)
        assert details["random_value"] is not None, "采样分支记录实际抽值"
        assert "certainty_threshold" in details and "margin_threshold" in details
        assert details["route_seed"] == 0 and details["seed_hash"]
        assert data["cycle_id"] == "cycle-2"
        assert data["role_id"] == "lin"
        assert data["response_id"] == "r-lin-2"

    asyncio.run(scenario())


def test_system1_error_trace_is_recorded_and_not_success() -> None:
    async def scenario() -> None:
        from concordia.utils import structured_logging

        from sceneweave.integrations.concordia.tracing import SimulationLogTracer

        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        sampler = ScriptedSampler({"system1": ['{"route": "continue", "note": "旧"}']})
        role = ModelRole(rid("lin"), sampler, tracer=tracer)
        with pytest.raises(ModelOutputError):
            await role.perceive(turn())
        [entry] = [e for e in log.entries if e.component_name == "system1.decide"]
        data = dict(log.reconstruct_value(entry.deduplicated_data))
        assert data["error"], "解析错误必须留痕，不能报成功"
        assert data["details"]["attempt_stage"] == "system1.decide"
        assert data["outputs"][0] == '{"route": "continue", "note": "旧"}'

    asyncio.run(scenario())
