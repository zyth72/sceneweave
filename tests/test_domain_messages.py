"""校验跨模块消息的身份、不可变性与毫秒时间约束。"""

from dataclasses import FrozenInstanceError
from types import MappingProxyType
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
    ActionPhase,
    Delivery,
    KnownActionState,
    MaintainAction,
    ModifiedAction,
    NewAction,
    Observation,
    Rejection,
    RoleResponse,
    TriggerReason,
    TurnInput,
    WorldEvent,
)
from sceneweave.domain.time import (
    SimulationTime,
    milliseconds_from_seconds,
    validate_simulation_time,
)
from sceneweave.domain.tracing import TraceRecord
from sceneweave.domain.translation import (
    ActionSchedule,
    TranslationRequest,
    TranslationResult,
)


def observation(event: str = "event-1", content: str = "门被推开") -> Observation:
    return Observation(EventId(event), SimulationTime(10_000), content)


def turn(**overrides: Any) -> TurnInput:
    values: dict[str, Any] = {
        "cycle_id": CycleId("cycle-1"),
        "role_id": RoleId("alice"),
        "simulation_time": SimulationTime(10_000),
        "trigger": TriggerReason((EventId("event-1"),)),
        "observations": (observation(),),
    }
    values.update(overrides)
    return TurnInput(**values)


def response(actions: tuple[Any, ...] | None = None) -> RoleResponse:
    return RoleResponse(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), actions or ())


def check_result(passed: bool, **overrides: Any) -> ActionCheckResult:
    values: dict[str, Any] = {
        "cycle_id": CycleId("cycle-1"),
        "role_id": RoleId("alice"),
        "response_id": ResponseId("r-1"),
        "world_state_version": WorldStateVersion(3),
        "passed": passed,
        "reason": "" if passed else "客观条件不允许",
    }
    values.update(overrides)
    return ActionCheckResult(**values)


class TestMilliseconds:
    def test_converts_integer_and_fractional_seconds(self) -> None:
        assert milliseconds_from_seconds(30) == SimulationTime(30_000)
        assert milliseconds_from_seconds(30.0) == SimulationTime(30_000)
        assert milliseconds_from_seconds(0.5) == SimulationTime(500)
        assert milliseconds_from_seconds(0) == SimulationTime(0)

    def test_large_values_convert_exactly_without_context_rounding(self) -> None:
        assert milliseconds_from_seconds(10**30) == SimulationTime(10**33)
        assert milliseconds_from_seconds(1e300) == SimulationTime(10**303)

    @pytest.mark.parametrize(
        "seconds", [-1, -0.5, 0.0005, float("nan"), float("inf"), True, "1.5", None, [3]]
    )
    def test_rejects_invalid_seconds(self, seconds: Any) -> None:
        with pytest.raises(ValueError, match="秒配置"):
            milliseconds_from_seconds(seconds)

    @pytest.mark.parametrize("value", [True, False, 10.5, "100", -1])
    def test_simulation_time_must_be_non_negative_int(self, value: Any) -> None:
        with pytest.raises(ValueError, match="非负整数毫秒"):
            validate_simulation_time(value, "剧情时刻")

    def test_bool_is_not_accepted_as_turn_time(self) -> None:
        with pytest.raises(ValueError, match="非负整数毫秒"):
            turn(simulation_time=True)


class TestIdentityValidation:
    def test_empty_identity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="周期身份"):
            turn(cycle_id=CycleId(""))
        with pytest.raises(ValueError, match="响应身份"):
            RoleResponse(CycleId("c"), RoleId("alice"), ResponseId(""), ())

    def test_missing_actions_is_a_construction_error_not_empty_set(self) -> None:
        with pytest.raises(TypeError):
            RoleResponse(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"))  # type: ignore[call-arg]

    @pytest.mark.parametrize("description", ["", ["假装描述"], 42, None])
    def test_text_fields_must_be_non_empty_str(self, description: Any) -> None:
        with pytest.raises(ValueError, match="新行动描述"):
            NewAction(ActionId("a-1"), description, CommitGroupId("g-1"))
        with pytest.raises(ValueError, match="事件描述"):
            WorldEvent(EventId("e"), SimulationTime(1), description)
        with pytest.raises(ValueError, match="感知内容"):
            Observation(EventId("e"), SimulationTime(1), description)
        with pytest.raises(ValueError, match="行动修改"):
            ModifiedAction(ActionId("a-1"), description)
        with pytest.raises(ValueError, match="行动状态反馈"):
            KnownActionState(ActionId("a-1"), ActionPhase.STARTED, description)

    def test_rejection_reason_must_be_non_empty_str(self) -> None:
        with pytest.raises(ValueError, match="打回原因"):
            Rejection(CycleId("c"), RoleId("a"), ResponseId("r"), ["原因"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="打回原因"):
            Rejection(CycleId("c"), RoleId("a"), ResponseId("r"), "")

    def test_check_reason_must_be_str(self) -> None:
        with pytest.raises(ValueError, match="检查原因"):
            check_result(False, reason=["原因"])  # type: ignore[arg-type]

    def test_duplicate_action_ids_are_rejected(self) -> None:
        actions = (
            NewAction(ActionId("a-1"), "伸手", CommitGroupId("g-1")),
            MaintainAction(ActionId("a-1")),
        )
        with pytest.raises(ValueError, match="重复身份"):
            response(actions)

    def test_new_actions_must_share_one_commit_group(self) -> None:
        actions = (
            NewAction(ActionId("a-1"), "伸手", CommitGroupId("g-1")),
            NewAction(ActionId("a-2"), "说话", CommitGroupId("g-2")),
        )
        with pytest.raises(ValueError, match="一个并行提交组"):
            response(actions)

    def test_empty_action_set_is_legal_when_explicit(self) -> None:
        assert response().actions == ()

    def test_list_is_rejected_to_avoid_shared_mutable_state(self) -> None:
        with pytest.raises(ValueError, match="行动声明元组"):
            response([MaintainAction(ActionId("a-1"))])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field", ["event_ids", "rejection_actions", "check_actions", "candidate_events"]
    )
    def test_identifier_tuples_reject_lists(self, field: str) -> None:
        listed = [EventId("event-1")]
        with pytest.raises(ValueError, match="元组"):
            if field == "event_ids":
                TriggerReason(listed)  # type: ignore[arg-type]
            elif field == "rejection_actions":
                Rejection(CycleId("c"), RoleId("a"), ResponseId("r"), "原因", listed)  # type: ignore[arg-type]
            elif field == "check_actions":
                check_result(True, action_ids=listed)  # type: ignore[arg-type]
            else:
                AdjudicationCandidate(
                    cycle_id=CycleId("cycle-1"),
                    simulation_time=SimulationTime(1),
                    event_ids=listed,  # type: ignore[arg-type]
                )

    def test_trace_action_ids_reject_lists(self) -> None:
        with pytest.raises(ValueError, match="元组"):
            TraceRecord(
                stage="world.check",
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                summary="检查",
                action_ids=[ActionId("a-1")],  # type: ignore[arg-type]
            )


class TestTurnInput:
    def test_observation_time_must_match_turn_time(self) -> None:
        with pytest.raises(ValueError, match="同一剧情时刻"):
            turn(observations=(Observation(EventId("event-1"), SimulationTime(9_999), "门"),))

    def test_observation_must_reference_a_triggered_event(self) -> None:
        with pytest.raises(ValueError, match="未触发的事件"):
            turn(observations=(Observation(EventId("event-9"), SimulationTime(10_000), "异响"),))

    def test_no_triggered_events_means_no_external_perception(self) -> None:
        with pytest.raises(ValueError, match="未触发的事件"):
            turn(
                trigger=TriggerReason(introspection=True),
                observations=(observation(),),
            )

    def test_same_event_may_carry_multiple_perceptions(self) -> None:
        merged = turn(
            observations=(
                Observation(EventId("event-1"), SimulationTime(10_000), "看到门"),
                Observation(EventId("event-1"), SimulationTime(10_000), "听到脚步"),
            )
        )
        assert [item.content for item in merged.observations] == ["看到门", "听到脚步"]

    def test_duplicate_trigger_events_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="触发事件身份"):
            TriggerReason((EventId("event-1"), EventId("event-1")))

    def test_no_external_perception_is_legal(self) -> None:
        empty = turn(observations=(), trigger=TriggerReason())
        assert empty.observations == ()
        assert empty.trigger.introspection is False

    def test_introspection_trigger_is_not_a_fake_perception(self) -> None:
        thinking = turn(observations=(), trigger=TriggerReason(introspection=True))
        assert thinking.trigger.introspection is True
        assert thinking.observations == ()

    def test_known_action_states_must_be_unique(self) -> None:
        states = (
            KnownActionState(ActionId("a-1"), ActionPhase.STARTED, "正在移动"),
            KnownActionState(ActionId("a-1"), ActionPhase.COMPLETED, "已到达"),
        )
        with pytest.raises(ValueError, match="已知行动身份"):
            turn(known_action_states=states)


class TestCheckAndRejection:
    def test_failed_check_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="必须提供原因"):
            check_result(False, reason="")

    def test_passed_check_may_omit_reason_and_records_version(self) -> None:
        check = check_result(True)
        assert check.world_state_version == WorldStateVersion(3)

    def test_bool_version_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="非负整数"):
            check_result(True, world_state_version=True)  # type: ignore[arg-type]


class TestImmutability:
    def test_messages_are_frozen(self) -> None:
        record = turn()
        with pytest.raises(FrozenInstanceError):
            record.cycle_id = CycleId("cycle-2")  # type: ignore[misc]

    def test_trace_details_are_copied_into_read_only_mapping(self) -> None:
        source = {"reason": "客观条件不允许"}
        record = TraceRecord(
            stage="world.check",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(1),
            summary="检查未通过",
            details=source,
        )
        assert isinstance(record.details, MappingProxyType)
        source["reason"] = "被外部改写"
        assert record.details["reason"] == "客观条件不允许"

    def test_trace_rejects_complex_detail_values_and_bad_summaries(self) -> None:
        with pytest.raises(ValueError, match="值类型不支持"):
            TraceRecord(
                stage="world.check",
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                summary="检查",
                details={"payload": object()},
            )
        with pytest.raises(ValueError, match="追踪摘要"):
            TraceRecord(
                stage="world.check",
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                summary=123,  # type: ignore[arg-type]
            )

    def test_trace_inputs_outputs_must_be_non_empty_str_tuples(self) -> None:
        with pytest.raises(ValueError, match="感知输入"):
            TraceRecord(
                stage="role.perceive",
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                summary="投递",
                inputs=(["门"],),  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="行动输出"):
            TraceRecord(
                stage="role.response",
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                summary="响应",
                outputs=("",),  # type: ignore[arg-type]
            )


class TestTranslationMessages:
    def test_schedule_requires_end_after_start(self) -> None:
        with pytest.raises(ValueError, match="晚于开始时刻"):
            ActionSchedule(
                ActionId("a-1"),
                CommitGroupId("g-1"),
                SimulationTime(2_000),
                SimulationTime(2_000),
            )

    def test_checkpoints_must_be_strictly_inside(self) -> None:
        with pytest.raises(ValueError, match="过程节点"):
            ActionSchedule(
                ActionId("a-1"),
                CommitGroupId("g-1"),
                SimulationTime(2_000),
                SimulationTime(5_000),
                (SimulationTime(2_000), SimulationTime(4_000)),
            )
        with pytest.raises(ValueError, match="过程节点"):
            ActionSchedule(
                ActionId("a-1"),
                CommitGroupId("g-1"),
                SimulationTime(2_000),
                SimulationTime(5_000),
                (SimulationTime(3_000), SimulationTime(3_000)),
            )

    def test_translation_result_allows_only_one_group(self) -> None:
        schedules = (
            ActionSchedule(
                ActionId("a-1"), CommitGroupId("g-1"), SimulationTime(2_000), SimulationTime(5_000)
            ),
            ActionSchedule(
                ActionId("a-2"), CommitGroupId("g-2"), SimulationTime(3_000), SimulationTime(6_000)
            ),
        )
        with pytest.raises(ValueError, match="一个提交组"):
            TranslationResult(CycleId("cycle-1"), RoleId("alice"), ResponseId("r-1"), schedules)

    def test_translation_request_keeps_response_and_state_reference(self) -> None:
        request = TranslationRequest(
            response=response(),
            current_time=SimulationTime(1_000),
            world_state_version=WorldStateVersion(2),
        )
        assert request.response.response_id == ResponseId("r-1")


class TestAdjudicationMessages:
    def test_follow_up_events_must_be_in_the_future(self) -> None:
        with pytest.raises(ValueError, match="晚于裁决时刻"):
            AdjudicationCandidate(
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(5_000),
                follow_up_events=(WorldEvent(EventId("later"), SimulationTime(5_000), "后续"),),
            )

    def test_duplicate_related_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="关联事件身份"):
            AdjudicationCandidate(
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(5_000),
                event_ids=(EventId("event-1"), EventId("event-1")),
            )

    def test_candidate_action_ids_reject_lists(self) -> None:
        with pytest.raises(ValueError, match="元组"):
            AdjudicationCandidate(
                cycle_id=CycleId("cycle-1"),
                simulation_time=SimulationTime(1),
                action_ids=[ActionId("a-1")],  # type: ignore[arg-type]
            )

    def test_validation_carries_candidate_and_version_association(self) -> None:
        candidate = AdjudicationCandidate(
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(1),
            proposed_changes=(FactChangeProposal("事件落地"),),
        )
        validation = AdjudicationValidation(candidate, WorldStateVersion(2), True)
        assert validation.candidate == candidate
        assert validation.world_state_version == WorldStateVersion(2)
        with pytest.raises(ValueError, match="一致"):
            AdjudicationValidation(candidate, WorldStateVersion(2), True, ("问题",))

    def test_commit_records_actual_changes_separately(self) -> None:
        candidate = AdjudicationCandidate(
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(5_000),
            proposed_changes=(FactChangeProposal("抽屉被拉开"),),
        )
        commit = AdjudicationCommit(
            candidate=candidate,
            committed_changes=(FactChangeProposal("抽屉被拉开"),),
            resulting_version=WorldStateVersion(4),
        )
        assert [item.description for item in commit.committed_changes] == ["抽屉被拉开"]
        assert commit.candidate is candidate
        assert commit.resulting_version == WorldStateVersion(4)


class TestDelivery:
    def test_delivery_requires_observation(self) -> None:
        with pytest.raises(ValueError, match="Observation"):
            Delivery(RoleId("alice"), "门被推开")  # type: ignore[arg-type]
