"""用测试用分阶段假世界验证 World 接口的边界语义。

MiniWorld 只是演示接口边界的测试假实现，不是运行时原子提交器的
实现；真实事实模型与语义判断由具体场景注入。
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from sceneweave.domain.adjudication import (
    AdjudicationCandidate,
    AdjudicationCommit,
    AdjudicationValidation,
    FactChangeProposal,
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
    Delivery,
    MaintainAction,
    NewAction,
    Observation,
    RoleResponse,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.world.base import World


class MiniWorld(World):
    """测试假世界：维护版本号、已提交对象身份与已提交事实文本。"""

    def __init__(self) -> None:
        self._version = WorldStateVersion(0)
        self.committed_facts: list[str] = []
        self.published: list[str] = []
        self._committed_objects: list[AdjudicationCommit] = []

    @property
    def state_version(self) -> WorldStateVersion:
        return self._version

    async def check_response(self, candidate: RoleResponse) -> ActionCheckResult:
        blocked = tuple(
            item.action_id
            for item in candidate.actions
            if isinstance(item, NewAction) and "不可能" in item.description
        )
        return ActionCheckResult(
            cycle_id=candidate.cycle_id,
            role_id=candidate.role_id,
            response_id=candidate.response_id,
            world_state_version=self._version,
            passed=not blocked,
            reason="" if not blocked else "客观条件不允许",
            action_ids=blocked,
        )

    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
    ) -> AdjudicationCandidate:
        changes = tuple(FactChangeProposal(f"事件 {event_id} 落地") for event_id in event_ids)
        return AdjudicationCandidate(
            cycle_id=cycle_id,
            simulation_time=simulation_time,
            event_ids=tuple(event_ids),
            action_ids=tuple(action_ids),
            proposed_changes=changes,
        )

    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        problems: list[str] = []
        if not candidate.proposed_changes:
            problems.append("候选没有可提交的事实变更")
        return AdjudicationValidation(
            candidate=candidate,
            world_state_version=self._version,
            passed=not problems,
            problems=tuple(problems),
        )

    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        validation = self.validate_adjudication(candidate)
        if validation.candidate != candidate:
            raise ValueError("校验结果与候选不一致，拒绝提交")
        if validation.world_state_version != self._version:
            raise ValueError("校验依据的世界状态已过期，拒绝提交")
        if not validation.passed:
            raise ValueError("候选未通过校验，拒绝提交")
        self._version = WorldStateVersion(self._version + 1)
        self.committed_facts.extend(item.description for item in candidate.proposed_changes)
        commit = AdjudicationCommit(
            candidate=candidate,
            committed_changes=candidate.proposed_changes,
            resulting_version=self._version,
        )
        self._committed_objects.append(commit)
        return commit

    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        if not any(item is commit for item in self._committed_objects):
            raise ValueError("只能发布已提交的裁决")
        self.published.extend(item.description for item in commit.committed_changes)
        return tuple(
            Delivery(
                RoleId("lin_zhou"),
                Observation(
                    EventId(f"publish-{index}"),
                    commit.candidate.simulation_time,
                    change.description,
                ),
            )
            for index, change in enumerate(commit.committed_changes)
        )


def check(coro: Any) -> Any:
    return asyncio.run(coro)


def test_check_response_is_side_effect_free() -> None:
    world = MiniWorld()
    passing = RoleResponse(
        CycleId("cycle-1"),
        RoleId("lin_zhou"),
        ResponseId("r-1"),
        (MaintainAction(ActionId("a-0")),),
    )
    result = check(world.check_response(passing))
    assert result.passed is True
    assert result.world_state_version == WorldStateVersion(0)
    blocked = RoleResponse(
        CycleId("cycle-1"),
        RoleId("lin_zhou"),
        ResponseId("r-2"),
        (NewAction(ActionId("a-9"), "不可能地拉开抽屉", CommitGroupId("g-1")),),
    )
    blocked_result = check(world.check_response(blocked))
    assert blocked_result.passed is False
    assert blocked_result.reason == "客观条件不允许"
    assert blocked_result.action_ids == (ActionId("a-9"),)
    assert world.state_version == WorldStateVersion(0), "检查不得改变共同状态"
    assert world.committed_facts == [], "检查不得提交事实"
    assert world.published == [], "检查不得发布感知"


def test_adjudicate_produces_candidate_without_committing() -> None:
    world = MiniWorld()
    candidate = check(
        world.adjudicate(
            CycleId("cycle-1"),
            SimulationTime(5_000),
            event_ids=(EventId("event-1"),),
            action_ids=(ActionId("a-1"),),
        )
    )
    assert candidate.event_ids == (EventId("event-1"),)
    assert candidate.proposed_changes == (FactChangeProposal("事件 event-1 落地"),)
    assert world.state_version == WorldStateVersion(0), "生成候选不得提交事实"
    assert world.committed_facts == []


def test_failed_validation_leaves_no_partial_commit() -> None:
    world = MiniWorld()
    empty = AdjudicationCandidate(
        cycle_id=CycleId("cycle-1"), simulation_time=SimulationTime(5_000)
    )
    validation = world.validate_adjudication(empty)
    assert validation.passed is False
    assert validation.problems == ("候选没有可提交的事实变更",)
    with pytest.raises(ValueError, match="拒绝提交"):
        world.commit_adjudication(empty)
    assert world.state_version == WorldStateVersion(0)
    assert world.committed_facts == []


def test_mismatched_validation_candidate_is_refused() -> None:
    class MismatchedValidationWorld(MiniWorld):
        def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
            other = AdjudicationCandidate(
                cycle_id=candidate.cycle_id,
                simulation_time=candidate.simulation_time,
                proposed_changes=(FactChangeProposal("另一个候选"),),
            )
            return AdjudicationValidation(other, self._version, True)

    world = MismatchedValidationWorld()
    candidate = check(
        world.adjudicate(CycleId("cycle-1"), SimulationTime(5_000), event_ids=(EventId("event-1"),))
    )
    with pytest.raises(ValueError, match="校验结果与候选不一致"):
        world.commit_adjudication(candidate)
    assert world.state_version == WorldStateVersion(0)
    assert world.committed_facts == []


def test_stale_validation_version_is_refused() -> None:
    class StaleValidationWorld(MiniWorld):
        def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
            return AdjudicationValidation(candidate, WorldStateVersion(99), True)

    world = StaleValidationWorld()
    candidate = check(
        world.adjudicate(CycleId("cycle-1"), SimulationTime(5_000), event_ids=(EventId("event-1"),))
    )
    with pytest.raises(ValueError, match="世界状态已过期"):
        world.commit_adjudication(candidate)


def test_commit_then_publish_boundary() -> None:
    world = MiniWorld()
    candidate = check(
        world.adjudicate(
            CycleId("cycle-1"),
            SimulationTime(5_000),
            event_ids=(EventId("event-1"), EventId("event-2")),
        )
    )
    commit = world.commit_adjudication(candidate)
    assert world.state_version == WorldStateVersion(1)
    assert world.committed_facts == ["事件 event-1 落地", "事件 event-2 落地"]
    assert commit.resulting_version == WorldStateVersion(1)
    deliveries = world.publish(commit)
    assert [item.observation.content for item in deliveries] == [
        "事件 event-1 落地",
        "事件 event-2 落地",
    ]
    assert deliveries[0].role_id == RoleId("lin_zhou")
    forged_future = AdjudicationCommit(
        candidate=candidate,
        committed_changes=(FactChangeProposal("未提交的变化"),),
        resulting_version=WorldStateVersion(99),
    )
    with pytest.raises(ValueError, match="只能发布已提交"):
        world.publish(forged_future)
    forged_identical = AdjudicationCommit(
        candidate=candidate,
        committed_changes=commit.committed_changes,
        resulting_version=commit.resulting_version,
    )
    # 内容相同的伪造提交也必须按已提交对象身份拒绝，不能仅凭字段相等放行。
    with pytest.raises(ValueError, match="只能发布已提交"):
        world.publish(forged_identical)
