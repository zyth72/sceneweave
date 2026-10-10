"""System1 接入复查：舍入边界、错误可追踪、日志不可变与可视化校验。"""

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest
from concordia.utils import structured_logging

from sceneweave.app.runner import _RunTracer
from sceneweave.app.ui_snapshot import RunSnapshotHistory, build_snapshot
from sceneweave.domain.identity import CycleId, ResponseId, RoleId
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.components import system1
from sceneweave.roles.components.support import TurnFacts
from sceneweave.roles.components.system1_policy import derive_route_rng
from sceneweave.roles.model_output import (
    ModelOutputError,
    System1Route,
    System1Tendency,
    parse_system1_tendency,
)


def _output(weights: dict[str, Any]) -> str:
    return json.dumps({"probabilities": weights, "note": "本轮影响"}, ensure_ascii=False)


@pytest.mark.parametrize("weights", [(0.98, 0.0, 0.0), (0.98, 0.04, 0.0)])
def test_sum_tolerance_includes_both_endpoints(weights: tuple[float, float, float]) -> None:
    tendency = parse_system1_tendency(
        _output(dict(zip((r.value for r in System1Route), weights, strict=True)))
    )
    assert sum(tendency.normalized().values()) == pytest.approx(1.0)
    assert tuple(tendency.probabilities.values()) == weights


def test_huge_integer_is_a_model_output_error() -> None:
    with pytest.raises(ModelOutputError):
        parse_system1_tendency(_output({"continue": 10**309, "replan": 0, "rethink": 0}))


def test_advertised_schema_example_is_valid_model_output() -> None:
    parse_system1_tendency(system1.SCHEMA)


def test_direct_component_call_uses_the_recorded_role_seed() -> None:
    facts = TurnFacts(
        CycleId("c-1"),
        RoleId("lin"),
        SimulationTime(0),
        (),
        False,
        ("模糊事件",),
        (),
    )

    class Sampler:
        async def sample(self, prompt: str, *, component: str) -> str:
            return _output({"continue": 0.45, "replan": 0.48, "rethink": 0.07})

    decision = asyncio.run(
        system1.run(
            Sampler(),
            facts,
            (),
            response_id=ResponseId("r-lin"),
            intention="继续观察",
            has_saved_plan=True,
            route_seed=9,
        )
    )
    expected_rng, expected_hash = derive_route_rng(9, "lin")
    assert decision.random_value == expected_rng.random()
    assert decision.seed_hash == expected_hash


@pytest.mark.parametrize("total", [0.0, 0.5, 1.01])
def test_tendency_cannot_be_constructed_with_inconsistent_total(total: float) -> None:
    with pytest.raises(ModelOutputError):
        System1Tendency(
            probabilities={r: float(r is System1Route.CONTINUE) for r in System1Route},
            note="无关事件",
            total=total,
        )


def test_nested_policy_trace_is_immutable_and_roundtrips_concordia() -> None:
    weights = {"continue": 0.82, "replan": 0.16, "rethink": 0.02}
    allowed = ["continue", "replan", "rethink"]
    record = TraceRecord(
        stage="system1.decide",
        cycle_id=CycleId("c-1"),
        simulation_time=SimulationTime(0),
        summary="程序路由",
        details={"legal_weights": weights, "allowed_routes": allowed},
    )
    weights["continue"] = 0.0
    allowed.clear()
    frozen = record.details["legal_weights"]
    assert isinstance(frozen, Mapping)
    assert frozen["continue"] == 0.82
    with pytest.raises(TypeError):
        frozen["continue"] = 0.0  # type: ignore[index]
    assert isinstance(record.details["allowed_routes"], tuple)
    log = structured_logging.SimulationLog()
    SimulationLogTracer(log).trace(record)
    loaded = structured_logging.SimulationLog.from_dict(json.loads(json.dumps(log.to_dict())))
    data = loaded.reconstruct_value(loaded.entries[0].deduplicated_data)
    assert data["details"]["legal_weights"] == {"continue": 0.82, "replan": 0.16, "rethink": 0.02}
    assert data["details"]["allowed_routes"] == ["continue", "replan", "rethink"]


@pytest.mark.parametrize(
    "raw",
    [
        _output({"continue": -1, "replan": 1, "rethink": 1}),
        _output({"continue": 0, "replan": 0, "rethink": 0}),
        _output({"continue": 1, "replan": 0, "rethink": 0, "extra": 0}),
        '{"probabilities":{"continue":0,"continue":1,"replan":0,"rethink":0},"note":"x"}',
    ],
)
def test_ui_does_not_draw_a_valid_distribution_for_invalid_output(raw: str) -> None:
    log = structured_logging.SimulationLog()
    log.add_entry(
        step=0,
        timestamp="2026-10-10T00:00:00Z",
        entity_name="role:lin",
        component_name="system1",
        entry_type="component",
        summary="System1",
        raw_data={"Component": "system1", "Key": "sample_text", "Response": raw},
    )
    [call] = build_snapshot(RunSnapshotHistory(), log)["calls"]
    assert call["distribution"] is None
    assert call["response"] == raw


def test_system1_request_identity_is_captured_before_sampler_await() -> None:
    tracer = _RunTracer()
    facts = TurnFacts(
        cycle_id=CycleId("c-1"),
        role_id=RoleId("lin"),
        simulation_time=SimulationTime(20),
        event_ids=(),
        introspection=False,
        observations=("门响了",),
        known_actions=(),
    )

    class Sampler:
        async def sample(self, prompt: str, *, component: str) -> str:
            channel = tracer.component_channel("role:lin", component)
            # 模拟请求在另一个角色完成之后返回；开始时捕获的身份仍属于林舟。
            tracer.trace(
                TraceRecord(
                    stage="system1.invoke",
                    cycle_id=CycleId("c-1"),
                    role_id=RoleId("shen"),
                    response_id=ResponseId("r-shen"),
                    simulation_time=SimulationTime(20),
                    summary="另一角色请求",
                )
            )
            raw = _output({"continue": 0.82, "replan": 0.16, "rethink": 0.02})
            channel(
                {"Component": component, "Key": "sample_text", "Prompt": prompt, "Response": raw}
            )
            return raw

    asyncio.run(
        system1.run(
            Sampler(),
            facts,
            (),
            response_id=ResponseId("r-lin"),
            intention="观察",
            has_saved_plan=True,
            tracer=tracer,
        )
    )
    entry = tracer.log.get_entries_by_component("system1")[0]
    data = tracer.log.reconstruct_value(entry.deduplicated_data)
    assert data["response_id"] == "r-lin"
    assert data["cycle_id"] == "c-1"
    calls = build_snapshot(RunSnapshotHistory(), tracer.log)["calls"]
    assert all(call["response_id"] == "r-lin" for call in calls)
