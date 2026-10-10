"""验证一个处理周期的全角色响应采集屏障。

屏障只并行投递输入并采集完整主动行为声明，不做检查、不登记、不提交；
整链翻译与 World 检查由调度器在屏障之后统一编排（见 test_scheduler 与
test_chain_check）。全部使用假角色，不调用真实模型，也不依赖墙钟睡眠。
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
)
from sceneweave.domain.messages import (
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
from sceneweave.engine.responses import CandidateResponse, CycleOutcome, ResponseRunner
from sceneweave.roles.base import Role


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


class SceneRole(Role):
    """受控假角色：记录输入与打回反馈，按脚本返回完整声明。"""

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
        runner = ResponseRunner([bob, alice])
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
        assert set(outcome.candidates) == {RoleId("alice"), RoleId("bob")}
        assert outcome.candidates[RoleId("alice")].response.actions == (new_action("a-1"),)
        assert outcome.candidates[RoleId("bob")].response.actions == (maintain("b-1"),)

    asyncio.run(scenario())


def test_role_without_perception_still_gets_explicit_input_and_completes() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (maintain("a-1"),)))
        bob = SceneRole("bob", response("bob", "r-b"))
        runner = ResponseRunner([alice, bob])
        outcome = await runner.run_cycle(both_inputs())
        assert bob.perceived == [turn("bob")]
        assert bob.perceived[0].observations == ()
        assert outcome.candidates[RoleId("bob")].response.actions == ()

    asyncio.run(scenario())


def test_same_time_events_are_merged_into_one_input() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        merged = turn(
            "alice",
            events=("event-1", "event-2"),
            observations=("门被推开", "有人说话"),
        )
        outcome = await ResponseRunner([alice]).run_cycle({RoleId("alice"): merged})
        assert len(alice.perceived) == 1
        assert [item.content for item in alice.perceived[0].observations] == [
            "门被推开",
            "有人说话",
        ]
        assert outcome.candidates[RoleId("alice")].response.response_id == ResponseId("r-a")

    asyncio.run(scenario())


def test_introspection_arrives_via_trigger_not_fake_perception() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a", (maintain("a-1"),)))
        thinking = turn("alice", events=(), introspection=True)
        await ResponseRunner([alice]).run_cycle({RoleId("alice"): thinking})
        assert alice.perceived[0].trigger.introspection is True
        assert alice.perceived[0].observations == ()

    asyncio.run(scenario())


def test_role_error_propagates_without_partial_result() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        bob = SceneRole("bob", response("bob", "r-b"))
        alice.broken = True
        alice.release.set()
        bob.release.set()
        runner = ResponseRunner([alice, bob])
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
        assert set(second.candidates) == {RoleId("alice"), RoleId("bob")}

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
        outcome = await ResponseRunner([alice]).run_cycle({RoleId("alice"): turn("alice")})
        declared = outcome.candidates[RoleId("alice")].response.actions
        assert declared == actions, "不扁平化为增量行动"
        assert {item.commit_group_id for item in declared if isinstance(item, NewAction)} == {
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
        runner = ResponseRunner([alice, bob])
        with pytest.raises(ValueError, match=message):
            await runner.run_cycle(mutate(both_inputs()))
        assert not alice.started.is_set()
        assert not bob.started.is_set()

    asyncio.run(scenario())


def test_response_from_another_cycle_or_role_is_rejected() -> None:
    async def scenario() -> None:
        wrong_cycle = SceneRole("alice", response("alice", "r-a", cycle="cycle-2"))
        with pytest.raises(BaseExceptionGroup) as cycle_info:
            await ResponseRunner([wrong_cycle]).run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "周期与输入不一致" in str(error)
            for error in flatten(cycle_info.value)
        )
        wrong_role = SceneRole("alice", response("bob", "r-b"))
        with pytest.raises(BaseExceptionGroup) as role_info:
            await ResponseRunner([wrong_role]).run_cycle({RoleId("alice"): turn("alice")})
        assert any(
            isinstance(error, ValueError) and "角色身份不一致" in str(error)
            for error in flatten(role_info.value)
        )

    asyncio.run(scenario())


def test_overlapping_rounds_are_rejected() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"), gate=True)
        runner = ResponseRunner([alice])
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


def test_duplicate_role_id_is_rejected_at_construction() -> None:
    alice = SceneRole("alice", response("alice", "r-a"))
    with pytest.raises(ValueError, match="重复角色身份"):
        ResponseRunner([alice, alice])


def test_empty_scene_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="至少包含一个角色"):
        ResponseRunner([])


def test_input_key_must_match_the_turn_role() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "r-a"))
        bob = SceneRole("bob", response("bob", "r-b"))
        runner = ResponseRunner([alice, bob])
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
            runner = ResponseRunner([role])
            with pytest.raises(BaseExceptionGroup) as info:
                await runner.run_cycle({RoleId("alice"): turn("alice")})
            assert any(
                isinstance(error, ValueError) and "必须返回 RoleResponse" in str(error)
                for error in flatten(info.value)
            ), "非法返回值不得造成 AttributeError"

    asyncio.run(scenario())


def test_response_id_cannot_be_shared_across_roles_in_one_cycle() -> None:
    async def scenario() -> None:
        alice = SceneRole("alice", response("alice", "shared-id"))
        bob = SceneRole("bob", response("bob", "shared-id"))
        runner = ResponseRunner([alice, bob])
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
        runner = ResponseRunner([alice, bob])
        with pytest.raises(ValueError, match="不能同时指向多个角色"):
            await runner.run_cycle(both_inputs())

    asyncio.run(scenario())


def test_candidate_and_outcome_validate_association() -> None:
    good = response("alice", "r-a")
    candidate = CandidateResponse(good)
    with pytest.raises(ValueError, match="角色身份不一致"):
        CycleOutcome(
            CycleId("cycle-1"),
            SimulationTime(10_000),
            {RoleId("bob"): candidate},
        )
    with pytest.raises(ValueError, match="响应周期与本轮不一致"):
        CycleOutcome(
            CycleId("cycle-2"),
            SimulationTime(10_000),
            {RoleId("alice"): candidate},
        )
    with pytest.raises(ValueError, match="结果值必须是 CandidateResponse"):
        CycleOutcome(
            CycleId("cycle-1"),
            SimulationTime(10_000),
            {RoleId("alice"): None},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="候选必须包含 RoleResponse"):
        CandidateResponse(None)  # type: ignore[arg-type]
