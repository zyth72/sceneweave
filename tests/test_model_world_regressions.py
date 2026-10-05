"""主代理复核中的事实进展、日志原子性和投影边界回归。"""

import asyncio
import json

import pytest

from sceneweave.domain.adjudication import ActionNodeKind
from sceneweave.domain.identity import ActionId, CycleId, EventId
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.world.model_world import ModelWorld
from sceneweave.world.scene import (
    SceneDataError,
    SceneObject,
    describe_scene,
    find_object,
    load_scene_card,
)

from .fakes import ScriptedSampler
from .test_model_world import NO_PROGRESS, SILENCE_CHANGE, Collector, node, study_card


async def ongoing_world(progress: dict, tracer=None, deliveries=None) -> ModelWorld:
    initial = {
        "changes": [],
        "outcomes": [{"action_id": "a", "result": "started", "detail": "已开始擦桌子"}],
        "follow_ups": [],
        "movements": [],
        "stop_movements": [],
        "passives": [
            {
                "id": "p",
                "description": "惯性",
                "object_id": "lin",
                "speed": 0.5,
                "direction": [1, 0],
            }
        ],
        "stop_passives": [],
    }
    sampler = ScriptedSampler(
        {
            "world-adjudicate": [json.dumps(initial), SILENCE_CHANGE],
            "world-progress": [json.dumps(progress)],
        }
    )
    world = ModelWorld(
        scene=load_scene_card(study_card()), sampler_for_scope=lambda scope: sampler, tracer=tracer
    )
    await world.update_progress(CycleId("c0"), SimulationTime(0))
    candidate = await world.adjudicate(
        CycleId("c1"),
        SimulationTime(0),
        event_ids=(EventId("e"),),
        action_ids=(ActionId("a"),),
        action_nodes=(node("a", "lin", ActionNodeKind.START, "擦桌子", time=0),),
    )
    commit = world.commit_adjudication(candidate)
    if deliveries is not None:
        deliveries.extend(world.publish(commit))
    return world


def test_nonmoving_progress_applies_facts_and_detail_without_completing() -> None:
    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress.update(
            changes=[
                {
                    "kind": "set_state",
                    "object_id": "desk",
                    "key": "clean_fraction",
                    "value": 0.5,
                    "description": "擦了一半",
                }
            ],
            action_progress=[{"action_id": "a", "detail": "已擦一半"}],
            updated_action_ids=["a"],
        )
        world = await ongoing_world(progress)
        before = world.scene
        update = await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert find_object(before, "desk").state.get("clean_fraction") is None
        assert find_object(world.scene, "desk").state["clean_fraction"] == 0.5
        [actual] = world.diagnostic_snapshot().action_states
        assert actual.state == "started"
        assert actual.detail == "已擦一半"
        assert update.updated_action_ids == (ActionId("a"),)
        assert world.state_version == 2

    asyncio.run(scenario())


def test_progress_invalid_later_patch_keeps_all_facts_and_baseline() -> None:
    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress["changes"] = [
            {
                "kind": "set_state",
                "object_id": "desk",
                "key": "clean_fraction",
                "value": 0.5,
                "description": "擦了一半",
            },
            {
                "kind": "set_state",
                "object_id": "missing",
                "key": "clean",
                "value": True,
                "description": "错误目标",
            },
        ]
        world = await ongoing_world(progress)
        before = world.scene
        with pytest.raises(SceneDataError, match="不存在"):
            await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert world.scene is before
        assert world.state_version == 1
        assert world._last_time == 0

    asyncio.run(scenario())


def test_finished_passive_keeps_last_movement_then_clears_metadata() -> None:
    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress.update(
            movements=[{"object_id": "lin", "position": [1.1, 0.6]}],
            passive_updates=[{"id": "p", "done": True, "note": "惯性结束"}],
        )
        world = await ongoing_world(progress)
        await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert find_object(world.scene, "lin").position == (1.1, 0.6)
        assert world.diagnostic_snapshot().passive_processes == ()
        assert not world._passive_movers and not world._passive_movements

    asyncio.run(scenario())


def test_identical_passive_note_does_not_change_world_version() -> None:
    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress["passive_updates"] = [{"id": "p", "note": "惯性", "done": False}]
        world = await ongoing_world(progress)
        before = world.scene
        await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert world.scene is before
        assert world.state_version == 1

    asyncio.run(scenario())


def test_success_log_failure_does_not_apply_progress() -> None:
    class FailingTrace(Collector):
        def trace(self, record: TraceRecord) -> None:
            if record.stage == "world.progress.applied":
                raise RuntimeError("日志失败")
            super().trace(record)

    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress["movements"] = [{"object_id": "lin", "position": [1.1, 0.6]}]
        trace = FailingTrace()
        world = await ongoing_world(progress, trace)
        before = world.scene
        with pytest.raises(RuntimeError, match="日志失败"):
            await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert world.scene is before and world.state_version == 1
        assert world._last_time == 0
        [error] = trace.by_stage("world.progress.error")
        assert error.outputs == (json.dumps(progress),)

    asyncio.run(scenario())


def test_projection_omits_filtered_parent_name() -> None:
    scene = load_scene_card(study_card())
    assert "书桌" not in describe_scene(scene, only_ids=("drawer",))
    assert "书桌" in describe_scene(scene, only_ids=("drawer", "desk"))


def test_direct_scene_outside_room_rejected() -> None:
    scene = SceneObject(
        "r",
        "房间",
        "房间",
        size=(4, 3),
        children=(SceneObject("lin", "林", "角色", position=(99, 99)),),
    )
    with pytest.raises(SceneDataError, match="房间范围"):
        ModelWorld(scene=scene, sampler_for_scope=lambda scope: ScriptedSampler())


def test_success_log_failure_does_not_commit_candidate() -> None:
    class FailingCommitTrace(Collector):
        def trace(self, record: TraceRecord) -> None:
            if record.stage == "world.commit":
                raise RuntimeError("提交日志失败")
            super().trace(record)

    async def scenario() -> None:
        sampler = ScriptedSampler({"world-adjudicate": [SILENCE_CHANGE]})
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: sampler,
            tracer=FailingCommitTrace(),
        )
        before = world.scene
        candidate = await world.adjudicate(
            CycleId("c"), SimulationTime(1000), event_ids=(EventId("e"),)
        )
        assert world._current_time == 0, "生成候选不能提交剧情时间现场"
        with pytest.raises(RuntimeError, match="提交日志失败"):
            world.commit_adjudication(candidate)
        assert world.scene is before and world.state_version == 0
        assert world.validate_adjudication(candidate).passed

    asyncio.run(scenario())


def test_progress_fact_change_reaches_observers_in_same_cycle() -> None:
    async def scenario() -> None:
        progress = json.loads(NO_PROGRESS)
        progress["changes"] = [
            {
                "kind": "set_state",
                "object_id": "desk",
                "key": "clean_fraction",
                "value": 0.5,
                "description": "擦了一半",
            }
        ]
        world = await ongoing_world(progress)
        await world.update_progress(CycleId("c2"), SimulationTime(1000))
        candidate = await world.adjudicate(
            CycleId("c2"), SimulationTime(1000), event_ids=(EventId("e2"),)
        )
        commit = world.commit_adjudication(candidate)
        deliveries = world.publish(commit)
        assert any("clean_fraction现在是0.5" in d.observation.content for d in deliveries)
        assert all(d.observation.event_id == EventId("e2") for d in deliveries)
        assert all(d.observation.simulation_time == 1000 for d in deliveries)

    asyncio.run(scenario())


def test_visible_actual_start_reaches_others_without_proposal_text() -> None:
    async def scenario() -> None:
        deliveries = []
        await ongoing_world(json.loads(NO_PROGRESS), deliveries=deliveries)
        assert any(
            d.role_id == "shen" and d.observation.content == "林开始行动" for d in deliveries
        )
        assert not any("擦桌子" in d.observation.content for d in deliveries)

    asyncio.run(scenario())
