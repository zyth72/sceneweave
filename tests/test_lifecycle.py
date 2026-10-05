"""行动生命周期测试：维持、撤回、修改、终态与归属的运行时验证。

场景通过完整调度闭环驱动（假世界/假翻译器/脚本角色），断言登记
记录、事件队列与世界客观效果的可观察行为。起始事件在 t=0，便于
按翻译延迟推算各节点时刻。
"""

import asyncio

import pytest

from sceneweave.domain.adjudication import ActionNodeKind
from sceneweave.domain.identity import ActionId, CommitGroupId, CycleId, EventId, ResponseId
from sceneweave.domain.messages import ActionPhase, NewAction, RoleResponse, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import TranslationResult
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler

from .fakes import (
    FakeSemanticWorld,
    ScriptedRole,
    ScriptedTranslator,
    idle_response,
    maintain,
    maintaining,
    modified,
    ms,
    new_action,
    response,
)


def startup(text: str = "开场") -> WorldEvent:
    return WorldEvent(EventId("startup"), SimulationTime(0), text)


def pending_pairs(scheduler: Scheduler) -> set[tuple[str, int, str]]:
    return {
        (str(event.event_id), event.simulation_time, event.kind.value)
        for event in scheduler.pending_events()
    }


def test_maintain_keeps_identity_schedule_and_events() -> None:
    """同 ID 维持保留原安排：不重排、不重复登记、不再次开始。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-walk", "走向门口"),))],
            fallback=maintaining("alice", "a-walk"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(plans={"走向门口": (900, 3000, (1000,))})
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        await scheduler.run()
        record = scheduler.action_records()[ActionId("a-walk")]
        assert record.phase is ActionPhase.COMPLETED
        assert record.start == ms(900), "维持不改变原开始时刻"
        assert record.checkpoints == (ms(1900),)
        assert record.expected_end == ms(3900)
        assert record.registered_cycle == "cycle-1", "维持不是重新登记"
        starts = [
            node
            for adjudication in world.adjudications
            for node in adjudication["action_nodes"]
            if node.node is ActionNodeKind.START
        ]
        assert len(starts) == 1, "维持不产生第二次开始"
        # 维持与空声明轮次不调用翻译器：同 ID 维持不重新翻译或重排。
        assert len(translator.requests) == 1, "只有含新行动的轮次才翻译"
        assert translator.requests[0].response.actions[0].action_id == ActionId("a-walk")

    asyncio.run(scenario())


def test_maintain_and_empty_rounds_skip_translator_and_keep_nodes() -> None:
    """维持/空声明轮次不调用翻译器；原节点与预计结束保持不变。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-hold", "持续推动"),))],
            fallback=maintaining("alice", "a-hold"),
        )
        bob = ScriptedRole("bob", fallback=idle_response("bob"))
        world = FakeSemanticWorld(["alice", "bob"])
        translator = ScriptedTranslator(plans={"持续推动": (900, 4000, (1000, 2000))})
        scheduler = Scheduler(
            roles=[alice, bob],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert len(translator.requests) == 1, "维持与空声明轮次不增加翻译请求"
        record = scheduler.action_records()[ActionId("a-hold")]
        assert record.expected_end == ms(4900), "维持不延长预计结束"
        assert record.checkpoints == (ms(1900), ms(2900))
        # 行动在时限内实际开始（节点真实调度），后续维持轮不重排。
        starts = [
            node
            for adjudication in world.adjudications
            for node in adjudication["action_nodes"]
            if node.node is ActionNodeKind.START
        ]
        assert len(starts) == 1

    asyncio.run(scenario())


def test_omitting_pending_action_revokes_future_events_without_world_effects() -> None:
    """未开始行动被遗漏只撤回登记：撤销未来执行，不产生客观裁决。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-later", "稍后开始"),)),
                response("alice", "r-2", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(plans={"稍后开始": (4000, 1000, ())})
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-later")]
        assert record.phase is ActionPhase.WITHDRAWN
        assert all(
            event.action_id != ActionId("a-later") for event in scheduler.pending_events()
        ), "待开始行为的未来执行事件被撤销"
        assert not any(
            node.node is ActionNodeKind.WITHDRAWN
            for adjudication in world.adjudications
            for node in adjudication["action_nodes"]
        ), "未开始的撤回不需要世界裁决客观效果"
        assert not any("a-later" in fact for fact in world.facts)

    asyncio.run(scenario())


def test_omitting_active_action_adjudicates_withdrawal_and_keeps_passive_effects() -> None:
    """执行中撤回：身体停止由 World 裁决提交，被动过程保留。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-run", "向前跑"),)),
                response("alice", "r-2", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-run", 0.002)
        translator = ScriptedTranslator(plans={"向前跑": (900, 2000, ())})
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(5000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-run")]
        assert record.phase is ActionPhase.WITHDRAWN
        withdrawal = [
            adjudication
            for adjudication in world.adjudications
            if any(node.node is ActionNodeKind.WITHDRAWN for node in adjudication["action_nodes"])
        ]
        assert len(withdrawal) == 1, "执行中撤回触发客观效果裁决"
        assert "a-run 撤回主动维持：身体停止，惯性继续" in world.facts
        # 撤回后的被动过程保留：空集合响应不能清除它，惯性事件真实发生。
        assert any("惯性滑行结束" in fact for fact in world.facts)
        assert world.passive_effects, "被动影响不因声明缺席而清空"
        assert all(event.action_id != ActionId("a-run") for event in scheduler.pending_events()), (
            "撤回行动的剩余节点被移除"
        )
        # 撤回角色在后续周期收到撤回状态反馈，而不是行动被删除。
        later = [t for t in alice.perceived if t.simulation_time >= ms(1400)]
        assert later
        assert any(
            state.action_id == ActionId("a-run") and state.phase is ActionPhase.WITHDRAWN
            for turn in later
            for state in turn.known_action_states
        )

    asyncio.run(scenario())


def test_empty_set_is_legal_and_keeps_established_effects() -> None:
    """合法空集合撤回全部主动维持，不清空被动影响或既有效果。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-hold", "持续推动"),)),
                response("alice", "r-2", ()),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        translator = ScriptedTranslator(plans={"持续推动": (900, 500, ())})
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(2000), max_duration=ms(3000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert "a-hold 开始：持续推动" in world.facts, "已发生效果不被空集合抹除"
        assert scheduler.action_records()[ActionId("a-hold")].phase is ActionPhase.WITHDRAWN

    asyncio.run(scenario())


def test_modification_adjusts_only_future_nodes() -> None:
    """明确修改只调整未来节点：历史开始与已过节点不可改写。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [response("alice", "r-1", (new_action("a-run", "沿走廊跑"),))],
            fallback=maintaining("alice", "a-run"),
        )
        alice.rounds.insert(1, response("alice", "r-2", (modified("a-run", "跑得更久"),)))
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-run", 0.001)
        translator = ScriptedTranslator(
            plans={"沿走廊跑": (900, 2100, (1000,))},
            modifications={"a-run": (8100, (2900, 5900))},
        )
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        await scheduler.run()
        record = scheduler.action_records()[ActionId("a-run")]
        assert record.start == ms(900), "已开始的开始时刻是历史，不可改写"
        # 修改发生在 t=900（行动刚开始）：原 cp@1900 属于未来，被新节点替换。
        assert record.checkpoints == (ms(3800), ms(6800))
        assert record.expected_end == ms(8100)
        checkpoint_times = [
            adjudication["simulation_time"]
            for adjudication in world.adjudications
            for node in adjudication["action_nodes"]
            if node.node is ActionNodeKind.CHECKPOINT
        ]
        assert set(checkpoint_times) == {ms(3800), ms(6800)}, "新过程节点真实调度"
        assert record.phase is ActionPhase.COMPLETED, "在修改后的结束时刻按 World 结果结算"
        assert record.registered_cycle == "cycle-1", "修改不改变登记周期"

    asyncio.run(scenario())


def test_modification_cannot_rewrite_history_and_is_rejected() -> None:
    """试图改写历史的修改安排被打回角色重新决定，而不是运行错误。"""

    async def scenario() -> None:
        from sceneweave.domain.translation import ModifiedActionSchedule, TranslationResult

        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-walk", "散步"),)),
                response("alice", "r-2", (maintain("a-walk"),)),
                response("alice", "r-3", (modified("a-walk", "重写历史"),)),
                response("alice", "r-4", (maintain("a-walk"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        base = ScriptedTranslator(plans={"散步": (900, 2100, (1000,))})

        def override(request):
            if any(isinstance(i, type(modified("x"))) for i in request.response.actions):
                arrangement = request.arrangements[0]
                return TranslationResult(
                    cycle_id=request.response.cycle_id,
                    role_id=request.response.role_id,
                    response_id=request.response.response_id,
                    modified_schedules=(
                        ModifiedActionSchedule(
                            action_id=arrangement.action_id,
                            start=arrangement.start,
                            expected_end=SimulationTime(request.current_time + 5000),
                            checkpoints=(SimulationTime(950),),
                        ),
                    ),
                )
            return base.default_result(request)

        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=ScriptedTranslator(override=override),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert alice.rejections, "改写历史被打回角色重新决定"
        assert "历史过程节点不可改写" in alice.rejections[0].reason
        record = scheduler.action_records()[ActionId("a-walk")]
        assert record.checkpoints == (ms(1900),), "打回候选不生效，原安排保留"
        assert record.expected_end == ms(3000)

    asyncio.run(scenario())


def test_terminal_action_id_cannot_restart_or_be_maintained() -> None:
    """终态行动身份不可复用：不因重复旧 ID 重新启动或重新维持。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-x", "短行动"),)),
                response("alice", "r-2", (maintain("a-x"),)),
                response("alice", "r-3", ()),
                response("alice", "r-4", (new_action("a-x", "再试一次"),)),
            ],
            fallback=idle_response("alice"),
        )
        world = FakeSemanticWorld(["alice"])
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=ScriptedTranslator(plans={"短行动": (500, 500, ())}),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "不得复用既有行动身份" in (result.error or "")
        assert scheduler.action_records()[ActionId("a-x")].phase is ActionPhase.COMPLETED

        alice2 = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-y", "短行动"),)),
                response("alice", "r-2", (maintain("a-y"),)),
                response("alice", "r-3", ()),
                response("alice", "r-4", (maintain("a-y"),)),
            ],
            fallback=idle_response("alice"),
        )
        world2 = FakeSemanticWorld(["alice"])
        scheduler2 = Scheduler(
            roles=[alice2],
            world=world2,
            translator=ScriptedTranslator(plans={"短行动": (500, 500, ())}),
            config=RunConfig(introspection_interval=ms(1000), max_duration=ms(3000)),
            initial_events=[startup()],
        )
        result2 = await scheduler2.run()
        assert result2.reason is RunStopReason.ERROR
        assert "终态行动不能被重新维持" in (result2.error or "")

    asyncio.run(scenario())


def test_cross_role_ownership_is_runtime_validated() -> None:
    """行动归属跨角色由运行时验证拒绝。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-a1", (new_action("a-own", "我的行动"),)),
                response("alice", "r-a2", ()),
            ],
            fallback=idle_response("alice"),
        )
        bob = ScriptedRole(
            "bob",
            [response("bob", "r-b1", ()), response("bob", "r-b2", (maintain("a-own"),))],
            fallback=idle_response("bob"),
        )
        world = FakeSemanticWorld(["alice", "bob"])
        scheduler = Scheduler(
            roles=[alice, bob],
            world=world,
            translator=ScriptedTranslator(plans={"我的行动": (900, 30_000, ())}),
            config=RunConfig(introspection_interval=ms(5000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "不能维持其他角色的行动" in (result.error or "")

    asyncio.run(scenario())


def test_modification_takes_effect_in_world_with_position_continuity() -> None:
    """修改经 World 裁决实际生效：速度变化、位置连续、安排统一更新。"""

    async def scenario() -> None:
        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-1", (new_action("a-run", "沿走廊跑"),)),
                response("alice", "r-2", (modified("a-run", "跑得更快"),)),
                response("alice", "r-3", (maintain("a-run"),)),
            ],
            fallback=maintaining("alice", "a-run"),
        )
        world = FakeSemanticWorld(["alice"])
        world.set_speed("a-run", 0.001)
        translator = ScriptedTranslator(
            plans={"沿走廊跑": (900, 10_000, (1000,))},
            modifications={"a-run": (15_000, (3_000,))},
        )
        scheduler = Scheduler(
            roles=[alice],
            world=world,
            translator=translator,
            config=RunConfig(introspection_interval=ms(20_000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        # World 收到结构化修改输入：归属、原描述、明确变化与新旧安排。
        modification_calls = [
            entry["modifications"] for entry in world.adjudications if entry["modifications"]
        ]
        assert modification_calls, "修改送交 World 裁决"
        node = modification_calls[0][0]
        assert node.role_id == "alice", "归属来自节点结构，不靠旁路绑定"
        assert node.change == "跑得更快"
        assert node.description == "沿走廊跑"
        assert node.old_start == ms(900) and node.old_expected_end == ms(10_900)
        assert node.new_expected_end == ms(15_000)
        # World 实际生效：速度翻倍，位置按剧情时间连续推进到时限。
        assert world.speeds[ActionId("a-run")] == pytest.approx(0.002)
        assert world.positions["alice"] == pytest.approx(0.002 * (10_000 - 900))
        record = scheduler.action_records()[ActionId("a-run")]
        assert record.checkpoints == (ms(3_900),), "未来节点按新安排替换"
        assert record.expected_end == ms(15_000)
        checkpoint_times = [
            entry["simulation_time"]
            for entry in world.adjudications
            for node_ in entry["action_nodes"]
            if getattr(node_, "node", None) is ActionNodeKind.CHECKPOINT
        ]
        assert checkpoint_times == [ms(3_900)], "旧过程节点不再触发"
        assert "a-run 修改生效（alice）：跑得更快" in "；".join(world.facts)

    asyncio.run(scenario())


def test_cross_role_duplicate_new_action_from_translation_retry_is_plan_error() -> None:
    """翻译后段重试引入的新行动身份不得与其他角色的最终声明重复。"""

    async def scenario() -> None:
        from sceneweave.domain.translation import ActionSchedule, TranslationResult

        alice = ScriptedRole(
            "alice",
            [
                response("alice", "r-a1", (new_action("a-first", "先提一个"),)),
                response("alice", "r-a2", (new_action("a-dup", "重试引入重复"),)),
            ],
            fallback=idle_response("alice"),
        )
        bob = ScriptedRole(
            "bob",
            [response("bob", "r-b1", (new_action("a-dup", "别人的行动"),))],
            fallback=idle_response("bob"),
        )
        world = FakeSemanticWorld(["alice", "bob"])

        def override(request):
            new_items = [i for i in request.response.actions if isinstance(i, NewAction)]
            if not new_items:
                return TranslationResult(
                    cycle_id=request.response.cycle_id,
                    role_id=request.response.role_id,
                    response_id=request.response.response_id,
                )
            action = new_items[0]
            start = (
                request.current_time
                if request.response.response_id == "r-a1"
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

        scheduler = Scheduler(
            roles=[alice, bob],
            world=world,
            translator=ScriptedTranslator(override=override),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[startup()],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.ERROR
        assert "新行动身份跨角色重复" in (result.error or "")
        assert len(scheduler.action_records()) == 0, "不部分登记"

    asyncio.run(scenario())


def test_prepare_registration_is_independent_pure_validation_entry() -> None:
    """prepare_registration 可直接调用：伪造输入与关联错误都被拒绝。"""

    from sceneweave.engine.actions import ActionRecord, FinalizedDeclaration, prepare_registration

    good = response("alice", "r-1", (new_action("a-ok", "正常"),))
    good_translation = TranslationResult(good.cycle_id, good.role_id, good.response_id, ())
    # 周期不一致。
    plan = prepare_registration(
        CycleId("cycle-9"),
        ms(0),
        {},
        {"alice": FinalizedDeclaration(good, good_translation)},
    )
    assert any("周期或归属" in problem for problem in plan.problems)
    # 翻译关联不一致。
    other = TranslationResult(good.cycle_id, good.role_id, ResponseId("r-other"), ())
    plan = prepare_registration(
        good.cycle_id,
        ms(0),
        {},
        {"alice": FinalizedDeclaration(good, other)},
    )
    assert any("关联同一响应" in problem for problem in plan.problems)
    # 跨角色新行动重复（不经响应屏障直接构造；双方都带合法安排）。
    from sceneweave.domain.translation import ActionSchedule

    def with_schedule(item: RoleResponse) -> TranslationResult:
        action = item.actions[0]
        return TranslationResult(
            item.cycle_id,
            item.role_id,
            item.response_id,
            (
                ActionSchedule(
                    action_id=action.action_id,
                    commit_group_id=action.commit_group_id,
                    start=ms(100),
                    expected_end=ms(1000),
                ),
            ),
        )

    bob = response("bob", "r-b", (new_action("a-ok", "重复身份"),))
    plan = prepare_registration(
        good.cycle_id,
        ms(0),
        {},
        {
            "alice": FinalizedDeclaration(good, with_schedule(good)),
            "bob": FinalizedDeclaration(bob, with_schedule(bob)),
        },
    )
    assert any("跨角色重复" in problem for problem in plan.problems)
    # 伪造非 FinalizedDeclaration 输入。
    plan = prepare_registration(
        good.cycle_id,
        ms(0),
        {},
        {"alice": "not-a-declaration"},  # type: ignore[dict-item]
    )
    assert any("FinalizedDeclaration" in problem for problem in plan.problems)
    # 既有记录复用（登记表直接构造，不依赖运行过程）。
    record = ActionRecord(
        action_id=ActionId("a-old"),
        role_id="alice",
        description="旧行动",
        phase=ActionPhase.WITHDRAWN,
        start=ms(0),
        expected_end=ms(100),
        checkpoints=(),
        registered_cycle=CycleId("cycle-1"),
    )
    revived = response("alice", "r-2", (new_action("a-old", "复活"),))
    revived_translation = TranslationResult(
        revived.cycle_id, revived.role_id, revived.response_id, ()
    )
    plan = prepare_registration(
        good.cycle_id,
        ms(200),
        {ActionId("a-old"): record},
        {"alice": FinalizedDeclaration(revived, revived_translation)},
    )
    assert any("复用既有行动身份" in problem for problem in plan.problems)


def test_prepare_registration_reruns_full_schedule_validation() -> None:
    """纯入口复用 registration_issues：时间/引用/提交组/历史全部重跑。"""

    from sceneweave.domain.translation import ActionSchedule, TranslationResult
    from sceneweave.engine.actions import FinalizedDeclaration, prepare_registration

    # 时间约束：开始不晚于当前时刻（开始 0、结束 100、当前 50）。
    early = response("alice", "r-1", (new_action("a-early", "过早"),))
    early_translation = TranslationResult(
        early.cycle_id,
        early.role_id,
        early.response_id,
        (
            ActionSchedule(
                action_id=ActionId("a-early"),
                commit_group_id=early.actions[0].commit_group_id,
                start=ms(0),
                expected_end=ms(100),
            ),
        ),
    )
    plan = prepare_registration(
        early.cycle_id,
        ms(50),
        {},
        {"alice": FinalizedDeclaration(early, early_translation)},
    )
    assert not plan.ok
    assert any("晚于当前时刻" in problem for problem in plan.problems), (
        "纯入口必须重跑时间校验，不能只做归属检查"
    )

    # 附加多余安排：引用了非本次新行动。
    extra = response("alice", "r-2", (new_action("a-main", "正常"),))
    extra_translation = TranslationResult(
        extra.cycle_id,
        extra.role_id,
        extra.response_id,
        (
            ActionSchedule(
                action_id=ActionId("a-main"),
                commit_group_id=extra.actions[0].commit_group_id,
                start=ms(100),
                expected_end=ms(900),
            ),
            ActionSchedule(
                action_id=ActionId("a-ghost"),
                commit_group_id=extra.actions[0].commit_group_id,
                start=ms(100),
                expected_end=ms(900),
            ),
        ),
    )
    plan = prepare_registration(
        extra.cycle_id,
        ms(0),
        {},
        {"alice": FinalizedDeclaration(extra, extra_translation)},
    )
    assert not plan.ok
    assert any("非本次新行动" in problem for problem in plan.problems)

    # 提交组不一致。
    wrong_group = TranslationResult(
        extra.cycle_id,
        extra.role_id,
        extra.response_id,
        (
            ActionSchedule(
                action_id=ActionId("a-main"),
                commit_group_id=CommitGroupId("g-other"),
                start=ms(100),
                expected_end=ms(900),
            ),
        ),
    )
    plan = prepare_registration(
        extra.cycle_id,
        ms(0),
        {},
        {"alice": FinalizedDeclaration(extra, wrong_group)},
    )
    assert not plan.ok
    assert any("提交组" in problem for problem in plan.problems)

    # translation 类型伪造：报告问题而不是 AttributeError。
    plan = prepare_registration(  # type: ignore[arg-type]
        extra.cycle_id,
        ms(0),
        {},
        {"alice": FinalizedDeclaration(extra, object())},
    )
    assert not plan.ok
    assert any("TranslationResult" in problem for problem in plan.problems)
