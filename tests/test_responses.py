"""验证响应屏障与身份约束，不调用真实模型或使用墙钟睡眠。"""

import asyncio

import pytest

from sceneweave.domain.messages import (
    ActionId,
    ActionProposal,
    Delivery,
    EventId,
    Observation,
    RoleId,
    RoleResponse,
    SimulationTime,
)
from sceneweave.engine.responses import ResponseRunner
from sceneweave.roles.base import Role


class ControlledRole(Role):
    def __init__(self, role_id: str, response: RoleResponse) -> None:
        self._role_id = RoleId(role_id)
        self.response = response
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.observations: tuple[Observation, ...] = ()

    @property
    def role_id(self) -> RoleId:
        return self._role_id

    async def respond(self, observations: tuple[Observation, ...]) -> RoleResponse:
        self.observations = observations
        self.started.set()
        await self.release.wait()
        self.completed.set()
        return self.response


def delivery(role_id: str, content: str = "可见内容") -> Delivery:
    return Delivery(
        RoleId(role_id),
        Observation(EventId("event-1"), SimulationTime(10), content),
    )


def action(role_id: str, action_id: str | None = None) -> ActionProposal:
    return ActionProposal(ActionId(action_id or f"action-{role_id}"), RoleId(role_id), "尝试行动")


@pytest.mark.parametrize("first_id", ["alice", "bob"])
def test_barrier_waits_and_completion_order_does_not_change_output(first_id: str) -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse((action("alice"),)))
        bob = ControlledRole("bob", RoleResponse((action("bob"),)))
        runner = ResponseRunner([bob, alice])
        task = asyncio.create_task(runner.collect((delivery("bob"), delivery("alice"))))
        await alice.started.wait()
        await bob.started.wait()
        first, last = (alice, bob) if first_id == "alice" else (bob, alice)
        first.release.set()
        await first.completed.wait()
        assert not task.done()
        last.release.set()
        result = await task
        assert result == (action("alice"), action("bob"))
        assert result[0].description == "尝试行动"

    asyncio.run(scenario())


def test_observations_are_grouped_and_only_delivered_to_recipients() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse())
        bob = ControlledRole("bob", RoleResponse())
        alice.release.set()
        runner = ResponseRunner([alice, bob])
        inputs = (delivery("alice", "看到门"), delivery("alice", "听到脚步"))
        assert await runner.collect(inputs) == ()
        assert alice.observations == tuple(item.observation for item in inputs)
        assert not bob.started.is_set()

    asyncio.run(scenario())


def test_unknown_recipient_is_rejected_before_any_role_runs() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse())
        runner = ResponseRunner([alice])
        with pytest.raises(ValueError, match="未知角色"):
            await runner.collect((delivery("alice"), delivery("missing")))
        assert not alice.started.is_set()

    asyncio.run(scenario())


def test_cross_event_inputs_are_rejected() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse())
        other = Delivery(
            alice.role_id, Observation(EventId("event-2"), SimulationTime(10), "另一个事件")
        )
        with pytest.raises(ValueError, match="同一个事件"):
            await ResponseRunner([alice]).collect((delivery("alice"), other))
        assert not alice.started.is_set()

    asyncio.run(scenario())


def test_role_cannot_submit_actions_for_another_role() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse((action("bob"),)))
        alice.release.set()
        with pytest.raises(ValueError, match="其他角色"):
            await ResponseRunner([alice]).collect((delivery("alice"),))

    asyncio.run(scenario())


def test_duplicate_action_ids_are_rejected() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse((action("alice", "same"),)))
        bob = ControlledRole("bob", RoleResponse((action("bob", "same"),)))
        alice.release.set()
        bob.release.set()
        with pytest.raises(ValueError, match="重复行动"):
            await ResponseRunner([alice, bob]).collect((delivery("alice"), delivery("bob")))

    asyncio.run(scenario())


def test_overlapping_rounds_are_rejected() -> None:
    async def scenario() -> None:
        alice = ControlledRole("alice", RoleResponse())
        runner = ResponseRunner([alice])
        task = asyncio.create_task(runner.collect((delivery("alice"),)))
        await alice.started.wait()
        with pytest.raises(RuntimeError, match="尚未结束"):
            await runner.collect((delivery("alice"),))
        alice.release.set()
        assert await task == ()
        assert await runner.collect((delivery("alice"),)) == ()

    asyncio.run(scenario())
