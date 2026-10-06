"""验证一个处理周期的全角色响应屏障与检查/打回约束。

全部使用假角色与假世界，不调用真实模型，也不依赖墙钟睡眠。
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

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
    ActionPhase,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Observation,
    Rejection,
    RoleResponse,
    TriggerReason,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionArrangement
from sceneweave.engine.responses import ResponseRunner
from sceneweave.roles.base import Role
from sceneweave.world.base import World


def turn(
    role_id: str,
    *,
    cycle: str = "cycle-1",
    time: int = 10_000,
    events: tuple[str, ...] = ("event-1",),
    observations: tuple[str, ...] = (),
    introspection: bool = False,
) -> TurnInput:
    """构造某个角色的本轮输入；observations 为感知内容文本。"""
    return TurnInput(
        cycle_id=CycleId(cycle),
        role_id=RoleId(role_id),
        simulation_time=SimulationTime(time),
        trigger=TriggerReason(tuple(EventId(event) for event in events), introspection),
        observations=tuple(
            Observation(EventId(event), SimulationTime(time), content)
            for event, content in zip(events, observations, strict=False)
        ),
    )


def response(
    role_id: str,
    response_id: str,
    actions: tuple[Any, ...] = (),
    *,
    cycle: str = "cycle-1",
) -> RoleResponse:
    return RoleResponse(CycleId(cycle), RoleId(role_id), ResponseId(response_id), actions)


def maintain(action_id: str) -> MaintainAction:
    return MaintainAction(ActionId(action_id))


def new_action(
    action_id: str,
    description: str = "尝试行动",
    group: str = "group-1",
) -> NewAction:
    return NewAction(ActionId(action_id), description, CommitGroupId(group))


def modified(action_id: str, change: str = "改为更慢的速度") -> ModifiedAction:
    return ModifiedAction(ActionId(action_id), change)


def arrangement(action_id: str, *, description: str = "既有安排") -> ActionArrangement:
    return ActionArrangement(
        action_id=ActionId(action_id),
        phase=ActionPhase.REGISTERED,
        description=description,
        start=SimulationTime(100),
        expected_end=SimulationTime(900),
    )


class SceneRole(Role):
    """受控假角色：记录输入与打回反馈，按脚本返回完整声明。

    gate=True 时 perceive 阻塞在 release 事件上，用于验证屏障等待；
    默认直接完成，便于驱动正常流程与错误路径。
    """

    def __init__(
        self,
        role_id: str,
        first: RoleResponse,
        retries: Sequence[RoleResponse] = (),
        *,
        gate: bool = False,
    ) -> None:
        self._role_id = RoleId(role_id)
        self._first = first
        self._retries = list(retries)
        self.gate = gate
        self.perceived: list[TurnInput] = []
        self.rejections: list[Rejection] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.broken = False

    @property
    def role_id(self) -> RoleId:
        return self._role_id

    async def perceive(self, turn_input: TurnInput) -> RoleResponse:
        self.perceived.append(turn_input)
        self.started.set()
        if self.gate:
            await self.release.wait()
        self.completed.set()
        if self.broken:
            raise RuntimeError("角色内部错误")
        return self._first

    async def rejected(self, feedback: Rejection) -> RoleResponse:
        self.rejections.append(feedback)
        return self._retries.pop(0)


class ScriptedWorld(World):
    """脚本化假世界：按描述拒绝特定新行动，其余通过检查。"""

    def __init__(self, forbidden: tuple[str, ...] = ()) -> None:
        self._forbidden = forbidden
        self._version = WorldStateVersion(0)
        self.checked: list[RoleResponse] = []
        self.checked_arrangements: list[tuple[RoleId, tuple[ActionArrangement, ...]]] = []

    @property
    def state_version(self) -> WorldStateVersion:
        return self._version

    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        return ProgressUpdate(cycle_id=cycle_id, simulation_time=simulation_time)

    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        return ObjectiveContext(
            world_state_version=self._version, role_id=role_id, conditions=("无特殊客观条件",)
        )

    def diagnostic_snapshot(self) -> WorldDiagnostic:
        return WorldDiagnostic(world_state_version=self._version)

    async def check_response(
        self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
    ) -> ActionCheckResult:
        self.checked.append(candidate)
        self.checked_arrangements.append((candidate.role_id, tuple(arrangements)))
        related = tuple(
            item.action_id
            for item in candidate.actions
            if isinstance(item, NewAction) and item.description in self._forbidden
        )
        passed = not related
        return ActionCheckResult(
            cycle_id=candidate.cycle_id,
            role_id=candidate.role_id,
            response_id=candidate.response_id,
            world_state_version=self._version,
            passed=passed,
            reason="" if passed else "客观条件不允许该行动",
            action_ids=related,
        )

    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
    ) -> Any:
        raise AssertionError("响应屏障不得触发世界裁决")

    def validate_adjudication(self, candidate: Any) -> Any:
        raise AssertionError("响应屏障不得触发裁决校验")

    def commit_adjudication(self, candidate: Any) -> Any:
        raise AssertionError("响应屏障不得提交客观事实")

    def publish(self, commit: Any) -> tuple[Any, ...]:
        raise AssertionError("响应屏障不得发布感知")


def both_inputs() -> dict[RoleId, TurnInput]:
    return {
        RoleId("alice"): turn("alice", observations=("门被推开",)),
        RoleId("bob"): turn("bob"),
    }


def flatten(group: BaseExceptionGroup[BaseException]) -> list[BaseException]:
    flat: list[BaseException] = []
    stack: list[BaseExceptionGroup[BaseException]] = [group]
    while stack:
        current = stack.pop()
        for error in current.exceptions:
            if isinstance(error, BaseExceptionGroup):
                stack.append(error)
            else:
                flat.append(error)
    return flat


@pytest.mark.parametrize("first_released", ["alice", "bob"])
def test_barrier_waits_for_all_roles_and_order_does_not_change_result(
    first_released: str,
) -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (new_action("a-1"),)), gate=True)
        bob = SceneRole("bob", response("bob", "r-b", (maintain("b-1"),)), gate=True)
        runner = ResponseRunner([bob, alice], ScriptedWorld())
        task = asyncio.create_task(runner.run_cycle(both_inputs()))
        await alice.started.wait()
        await bob.started.wait()
        first, last = (alice, bob) if first_released == "alice" else (bob, alice)
        first.release.set()
        await first.completed.wait()
        assert not task.done()
        last.release.set()
        outcome = await task
        assert outcome.cycle_id == CycleId("cycle-1")
        assert outcome.simulation_time == SimulationTime(10_000)
        assert set(outcome.accepted) == {RoleId("alice"), RoleId("bob")}
        assert outcome.accepted[RoleId("alice")].response.actions == (new_action("a-1"),)
        assert outcome.accepted[RoleId("bob")].response.actions == (maintain("b-1"),)

    asyncio.run(scenario())


def test_role_without_perception_still_gets_explicit_input_and_completes() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (maintain("a-1"),)))
        bob = SceneRole("bob", response("bob", "r-b"))
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        outcome = await runner.run_cycle(both_inputs())
        assert bob.perceived == [turn("bob")]
        assert bob.perceived[0].observations == ()
        assert outcome.accepted[RoleId("bob")].response.actions == ()

    asyncio.run(scenario())


def test_same_time_events_are_merged_into_one_input() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        merged = turn(
            "alice",
            events=("event-1", "event-2"),
            observations=("门被推开", "有人说话"),
        )
        outcome = await ResponseRunner([alice], ScriptedWorld()).run_cycle(
            {RoleId("alice"): merged}
        )
        assert len(alice.perceived) == 1
        assert [item.content for item in alice.perceived[0].observations] == [
            "门被推开",
            "有人说话",
        ]
        assert outcome.accepted[RoleId("alice")].response.response_id == ResponseId("r-a")

    asyncio.run(scenario())


def test_introspection_arrives_via_trigger_not_fake_perception() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (maintain("a-1"),)))
        thinking = turn("alice", events=(), introspection=True)
        await ResponseRunner([alice], ScriptedWorld()).run_cycle({RoleId("alice"): thinking})
        assert alice.perceived[0].trigger.introspection is True
        assert alice.perceived[0].observations == ()

    asyncio.run(scenario())


def test_rejection_calls_rejected_with_current_response_and_accepts_full_replacement() -> None:
    async def scenario() -> None:
        forbidden = "隔着房间拉倒对方"
        retry = response(
            "alice",
            "r-a-2",
            (maintain("a-0"), new_action("a-2", "走近两步", "group-2")),
        )
        alice = SceneRole(
            "alice",
            response("alice", "r-a-1", (new_action("a-1", forbidden),)),
            [retry],
        )
        world = ScriptedWorld((forbidden,))
        outcome = await ResponseRunner([alice], world).run_cycle(
            {RoleId("alice"): turn("alice", observations=("门被推开",))}
        )
        assert len(alice.perceived) == 1, "打回不得重投原始感知"
        assert len(alice.rejections) == 1
        feedback = alice.rejections[0]
        assert feedback.rejected_response_id == ResponseId("r-a-1")
        assert feedback.cycle_id == CycleId("cycle-1")
        assert feedback.role_id == RoleId("alice")
        assert feedback.action_ids == (ActionId("a-1"),)
        assert feedback.reason == "客观条件不允许该行动"
        accepted = outcome.accepted[RoleId("alice")]
        assert accepted.response == retry, "接受的是新的完整候选，不合并增量"
        assert accepted.check.response_id == ResponseId("r-a-2")

    asyncio.run(scenario())


def test_each_rejection_targets_the_pending_response_not_older_ones() -> None:
    async def scenario() -> None:
        always_failing = new_action("a-1", "不可能的行动")
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (always_failing,)),
            [
                response("alice", "r-2", (always_failing,)),
                response("alice", "r-3", (always_failing,)),
            ],
        )
        world = ScriptedWorld(("不可能的行动",))
        runner = ResponseRunner([alice], world, max_rejections=2)
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        errors = flatten(info.value)
        assert any(isinstance(error, RuntimeError) and "打回次数" in str(error) for error in errors)
        assert [item.rejected_response_id for item in alice.rejections] == [
            ResponseId("r-1"),
            ResponseId("r-2"),
        ], "每次反馈只针对当前待检查响应"

    asyncio.run(scenario())


def test_role_error_propagates_without_partial_result() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        bob = SceneRole("bob", response("bob", "r-b"))
        alice.broken = True
        alice.release.set()
        bob.release.set()
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle(both_inputs())
        assert any(
            isinstance(error, RuntimeError) and "角色内部错误" in str(error)
            for error in flatten(info.value)
        )
        assert bob.completed.is_set()
        alice.broken = False
        alice.release.set()
        second = await runner.run_cycle(both_inputs())
        assert set(second.accepted) == {RoleId("alice"), RoleId("bob")}

    asyncio.run(scenario())


def test_runner_freezes_per_role_arrangements_for_all_checks() -> None:
    """周期开始时冻结本角色安排；初检与打回重查共用同一快照。"""

    async def scenario() -> None:
        fetched: list[RoleId] = []
        snapshot = {
            RoleId("alice"): (arrangement("a-1", description="爱丽丝的安排"),),
            RoleId("bob"): (arrangement("b-1", description="鲍勃的安排"),),
        }

        def arrangements_for(role_id: RoleId) -> tuple[ActionArrangement, ...]:
            fetched.append(role_id)
            return snapshot[role_id]

        world = ScriptedWorld(("不可能行动",))
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (maintain("a-1"), new_action("a-9", "不可能行动"))),
            [response("alice", "r-2", (maintain("a-1"),))],
        )
        bob = SceneRole("bob", response("bob", "r-b", (maintain("b-1"),)))
        runner = ResponseRunner([alice, bob], world, arrangements_for=arrangements_for)
        outcome = await runner.run_cycle(both_inputs())
        assert set(outcome.accepted) == {RoleId("alice"), RoleId("bob")}
        assert sorted(fetched, key=str) == [RoleId("alice"), RoleId("bob")], (
            "每角色周期开始只取一次"
        )
        alice_checks = [item for item in world.checked_arrangements if item[0] == RoleId("alice")]
        assert alice_checks == [
            (RoleId("alice"), snapshot[RoleId("alice")]),
            (RoleId("alice"), snapshot[RoleId("alice")]),
        ], "初检与打回重查共用同一冻结快照"
        bob_checks = [item for item in world.checked_arrangements if item[0] == RoleId("bob")]
        assert bob_checks == [(RoleId("bob"), snapshot[RoleId("bob")])], "角色作用域不混用"
        assert all(
            item.action_id != ActionId("a-9")
            for _, arrangements in world.checked_arrangements
            for item in arrangements
        ), "被打回的新候选不进入安排上下文"

    asyncio.run(scenario())


def test_runner_rejects_non_arrangement_context_before_role_processing() -> None:
    """非 ActionArrangement 的既有安排来源在角色处理前被拒绝。"""

    async def scenario() -> None:
        def arrangements_for(role_id: RoleId) -> tuple[Any, ...]:
            return ("坏安排",)

        alice = SceneRole("alice", response("alice", "r-a"))
        runner = ResponseRunner([alice], ScriptedWorld(), arrangements_for=arrangements_for)
        with pytest.raises(ValueError, match="ActionArrangement"):
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert alice.perceived == [], "安排结构错误在角色处理前拒绝"

    asyncio.run(scenario())


def test_check_result_must_match_the_checked_response() -> None:
    class MismatchedWorld(ScriptedWorld):
        async def check_response(
            self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
        ) -> ActionCheckResult:
            result = await super().check_response(candidate, arrangements=arrangements)
            return ActionCheckResult(
                cycle_id=result.cycle_id,
                role_id=result.role_id,
                response_id=ResponseId("其他响应"),
                world_state_version=result.world_state_version,
                passed=True,
            )

    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        runner = ResponseRunner([alice], MismatchedWorld())
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "不一致" in str(error)
            for error in flatten(info.value)
        )

    asyncio.run(scenario())


def test_check_result_based_on_newer_state_is_rejected() -> None:
    class SilentDriftWorld(ScriptedWorld):
        """在检查中途推进版本，但结果仍报告旧版本。"""

        def __init__(self) -> None:
            super().__init__(("不可能的行动",))
            self.checks_made = 0

        async def check_response(
            self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
        ) -> ActionCheckResult:
            self.checks_made += 1
            result = await super().check_response(candidate, arrangements=arrangements)
            if self.checks_made > 1:
                self._version = WorldStateVersion(self._version + 1)
            return result

    async def scenario() -> None:
        failing = new_action("a-1", "不可能的行动")
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (failing,)),
            [response("alice", "r-2", ())],
        )
        runner = ResponseRunner([alice], SilentDriftWorld(), max_rejections=1)
        with pytest.raises(RuntimeError, match="共同世界状态版本发生变化"):
            await runner.run_cycle({RoleId("alice"): turn("alice")})

    asyncio.run(scenario())


def test_check_result_claiming_newer_state_is_rejected() -> None:
    class ClaimingDriftWorld(ScriptedWorld):
        """检查结果直接报告更新后的版本。"""

        def __init__(self) -> None:
            super().__init__(("不可能的行动",))
            self.checks_made = 0

        async def check_response(
            self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
        ) -> ActionCheckResult:
            self.checks_made += 1
            if self.checks_made > 1:
                self._version = WorldStateVersion(self._version + 1)
            return await super().check_response(candidate, arrangements=arrangements)

    async def scenario() -> None:
        failing = new_action("a-1", "不可能的行动")
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (failing,)),
            [response("alice", "r-2", ())],
        )
        runner = ResponseRunner([alice], ClaimingDriftWorld(), max_rejections=1)
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "起始版本" in str(error)
            for error in flatten(info.value)
        )

    asyncio.run(scenario())


def test_commit_groups_and_declarations_are_preserved_in_outcome() -> None:
    async def scenario() -> None:
        actions = (
            maintain("a-0"),
            new_action("a-1", "伸手拉开抽屉", "group-7"),
            new_action("a-2", "同时说话", "group-7"),
            modified("a-3"),
        )
        alice = SceneRole("alice", response("alice", "r-a", actions))
        outcome = await ResponseRunner([alice], ScriptedWorld()).run_cycle(
            {RoleId("alice"): turn("alice")}
        )
        accepted = outcome.accepted[RoleId("alice")].response.actions
        assert accepted == actions, "不扁平化为增量行动"
        assert {item.commit_group_id for item in accepted if isinstance(item, NewAction)} == {
            CommitGroupId("group-7")
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda inputs: inputs | {RoleId("unknown"): turn("unknown")}, "未知角色"),
        (lambda inputs: {RoleId("alice"): inputs[RoleId("alice")]}, "缺少本轮输入"),
        (
            lambda inputs: inputs | {RoleId("bob"): turn("bob", cycle="cycle-2")},
            "同一个周期",
        ),
        (
            lambda inputs: inputs | {RoleId("bob"): turn("bob", time=20_000)},
            "同一剧情时刻",
        ),
        (
            lambda inputs: [
                turn("alice"),
                turn("alice", cycle="cycle-1"),
                turn("bob"),
            ],
            "多份本轮输入",
        ),
    ],
)
def test_invalid_cycle_inputs_are_rejected_before_any_role_runs(mutate: Any, message: str) -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        bob = SceneRole("bob", response("bob", "r-b"))
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        with pytest.raises(ValueError, match=message):
            await runner.run_cycle(mutate(both_inputs()))
        assert not alice.started.is_set()
        assert not bob.started.is_set()

    asyncio.run(scenario())


def test_response_from_another_cycle_or_role_is_rejected() -> None:
    async def scenario() -> None:
        wrong_cycle = SceneRole("alice", response("alice", "r-a", cycle="cycle-2"))
        with pytest.raises(BaseExceptionGroup) as cycle_info:
            await ResponseRunner([wrong_cycle], ScriptedWorld()).run_cycle(
                {RoleId("alice"): turn("alice")}
            )
        assert any(
            isinstance(error, ValueError) and "周期与输入不一致" in str(error)
            for error in flatten(cycle_info.value)
        )
        wrong_role = SceneRole("alice", response("bob", "r-b"))
        with pytest.raises(BaseExceptionGroup) as role_info:
            await ResponseRunner([wrong_role], ScriptedWorld()).run_cycle(
                {RoleId("alice"): turn("alice")}
            )
        assert any(
            isinstance(error, ValueError) and "角色身份不一致" in str(error)
            for error in flatten(role_info.value)
        )

    asyncio.run(scenario())


def test_overlapping_rounds_are_rejected() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"), gate=True)
        runner = ResponseRunner([alice], ScriptedWorld())
        task = asyncio.create_task(runner.run_cycle({RoleId("alice"): turn("alice")}))
        await alice.started.wait()
        with pytest.raises(RuntimeError, match="尚未结束"):
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        alice.release.set()
        assert (await task).cycle_id == CycleId("cycle-1")
        assert (await runner.run_cycle({RoleId("alice"): turn("alice")})).cycle_id == (
            CycleId("cycle-1")
        )

    asyncio.run(scenario())


def test_barrier_does_not_commit_or_advance_world_state() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (maintain("a-1"),)))
        world = ScriptedWorld()
        outcome = await ResponseRunner([alice], world).run_cycle({RoleId("alice"): turn("alice")})
        assert world.state_version == WorldStateVersion(0)
        assert outcome.world_state_version == WorldStateVersion(0)
        assert len(world.checked) == 1

    asyncio.run(scenario())


def test_duplicate_role_id_is_rejected_at_construction() -> None:
    alice = SceneRole("alice", response("alice", "r-a"))
    with pytest.raises(ValueError, match="重复角色身份"):
        ResponseRunner([alice, alice], ScriptedWorld())


def test_empty_scene_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="至少包含一个角色"):
        ResponseRunner([], ScriptedWorld())


def test_invalid_rejection_limit_is_rejected_at_construction() -> None:
    alice = SceneRole("alice", response("alice", "r-a"))
    with pytest.raises(ValueError, match="正整数"):
        ResponseRunner([alice], ScriptedWorld(), max_rejections=0)
    with pytest.raises(ValueError, match="正整数"):
        ResponseRunner([alice], ScriptedWorld(), max_rejections=True)  # type: ignore[arg-type]


def test_input_key_must_match_the_turn_role() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        bob = SceneRole("bob", response("bob", "r-b"))
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        with pytest.raises(ValueError, match="输入键与角色身份不一致"):
            await runner.run_cycle({RoleId("alice"): turn("alice"), RoleId("bob"): turn("alice")})
        assert not alice.started.is_set()
        assert not bob.started.is_set()

    asyncio.run(scenario())


def test_role_returning_non_response_is_rejected_cleanly() -> None:
    async def scenario() -> None:
        class NoneRole(SceneRole):
            async def perceive(self, turn_input: TurnInput) -> Any:
                return None

        class TextRole(SceneRole):
            async def perceive(self, turn_input: TurnInput) -> Any:
                return "空集合"

        for role in (
            NoneRole("alice", response("alice", "r-a")),
            TextRole("alice", response("alice", "r-a")),
        ):
            runner = ResponseRunner([role], ScriptedWorld())
            with pytest.raises(BaseExceptionGroup) as info:
                await runner.run_cycle({RoleId("alice"): turn("alice")})
            assert any(
                isinstance(error, ValueError) and "必须返回 RoleResponse" in str(error)
                for error in flatten(info.value)
            ), "非法返回值不得造成 AttributeError"

    asyncio.run(scenario())


def test_world_returning_non_check_result_is_rejected_cleanly() -> None:
    class BrokenWorld(ScriptedWorld):
        async def check_response(
            self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
        ) -> Any:
            return {"passed": True}

    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        runner = ResponseRunner([alice], BrokenWorld())
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "ActionCheckResult" in str(error)
            for error in flatten(info.value)
        )

    asyncio.run(scenario())


def test_check_may_only_reference_declared_actions() -> None:
    class OffTopicWorld(ScriptedWorld):
        async def check_response(
            self, candidate: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
        ) -> ActionCheckResult:
            result = await super().check_response(candidate, arrangements=arrangements)
            return ActionCheckResult(
                cycle_id=result.cycle_id,
                role_id=result.role_id,
                response_id=result.response_id,
                world_state_version=result.world_state_version,
                passed=False,
                reason="客观条件不允许",
                action_ids=(ActionId("a-别的行动"),),
            )

    async def scenario() -> None:
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (new_action("a-1"),)),
            [response("alice", "r-2", (new_action("a-2"),))],
        )
        runner = ResponseRunner([alice], OffTopicWorld(), max_rejections=1)
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "声明外的行动" in str(error)
            for error in flatten(info.value)
        )
        assert alice.rejections == [], "错误引用不得发给 rejected"

    asyncio.run(scenario())


def test_response_id_reuse_within_a_role_is_a_runtime_error() -> None:
    async def scenario() -> None:
        always_failing = new_action("a-1", "不可能的行动")
        alice = SceneRole(
            "alice",
            response("alice", "r-1", (always_failing,)),
            [
                response("alice", "r-2", (always_failing,)),
                response("alice", "r-1", (always_failing,)),
            ],
        )
        runner = ResponseRunner([alice], ScriptedWorld(("不可能的行动",)), max_rejections=3)
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, RuntimeError) and "不得重用" in str(error)
            for error in flatten(info.value)
        )

    asyncio.run(scenario())


def test_response_id_cannot_be_shared_across_roles_in_one_cycle() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "shared-id"))
        bob = SceneRole("bob", response("bob", "shared-id"))
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        with pytest.raises(BaseExceptionGroup) as info:
            await runner.run_cycle(both_inputs())
        assert any(
            isinstance(error, RuntimeError) and "跨角色共享" in str(error)
            for error in flatten(info.value)
        )

    asyncio.run(scenario())


def test_action_id_cannot_be_claimed_by_two_roles_after_barrier() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (new_action("shared-action"),)))
        bob = SceneRole("bob", response("bob", "r-b", (new_action("shared-action"),)))
        runner = ResponseRunner([alice, bob], ScriptedWorld())
        with pytest.raises(ValueError, match="不能同时指向多个角色"):
            await runner.run_cycle(both_inputs())

    asyncio.run(scenario())


def test_accepted_response_and_outcome_validate_association() -> None:
    good = response("alice", "r-a")
    good_check = ActionCheckResult(
        CycleId("cycle-1"),
        RoleId("alice"),
        ResponseId("r-a"),
        WorldStateVersion(0),
        True,
    )
    failed = ActionCheckResult(
        CycleId("cycle-1"),
        RoleId("alice"),
        ResponseId("r-a"),
        WorldStateVersion(0),
        False,
        "客观条件不允许",
    )
    from sceneweave.engine.responses import AcceptedResponse, CycleOutcome

    with pytest.raises(ValueError, match="必须已通过"):
        AcceptedResponse(good, failed)
    with pytest.raises(ValueError, match="身份不一致"):
        AcceptedResponse(
            good,
            ActionCheckResult(
                CycleId("cycle-1"), RoleId("alice"), ResponseId("其他"), WorldStateVersion(0), True
            ),
        )
    accepted = AcceptedResponse(good, good_check)
    with pytest.raises(ValueError, match="角色身份不一致"):
        CycleOutcome(
            CycleId("cycle-1"),
            SimulationTime(10_000),
            WorldStateVersion(0),
            {RoleId("bob"): accepted},
        )
    with pytest.raises(ValueError, match="世界状态版本与本轮不一致"):
        CycleOutcome(
            CycleId("cycle-1"),
            SimulationTime(10_000),
            WorldStateVersion(7),
            {RoleId("alice"): accepted},
        )
    with pytest.raises(ValueError, match="响应周期与本轮不一致"):
        CycleOutcome(
            CycleId("cycle-2"),
            SimulationTime(10_000),
            WorldStateVersion(0),
            {RoleId("alice"): accepted},
        )
    with pytest.raises(ValueError, match="结果值必须是 AcceptedResponse"):
        CycleOutcome(
            CycleId("cycle-1"),
            SimulationTime(10_000),
            WorldStateVersion(0),
            {RoleId("alice"): None},  # type: ignore[dict-item]
        )
