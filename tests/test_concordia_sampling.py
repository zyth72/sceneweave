"""Concordia 采样桥接与录制/重放测试（不依赖在途 model_trace.py）。

使用真实 ComponentWithLogging 组件与 SimulationLog 通道、真实
LanguageModel 子类桩（同步）验证异步桥；录制按精确请求重放，错配即
错，重放绝不重复真实调用。
"""

import asyncio
import json

import pytest
from concordia.language_model import language_model
from concordia.utils import structured_logging

from sceneweave.integrations.concordia.sampling import (
    LanguageModelSampler,
    RecordedModelError,
    RecordingDirectory,
    RecordingSampler,
    ReplayDirectory,
    ReplayMismatchError,
    ReplaySampler,
    SamplerComponent,
)
from sceneweave.integrations.concordia.tracing import component_logging_channel

from .fakes import ScriptedSampler, ScriptedTranslator, rid

RESPONSES = [
    '{"reconsider": true, "note": "有变化"}',
    '{"intention": "去书桌旁", "abandon": []}',
    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}',
]


class StubModel(language_model.LanguageModel):
    """同步语言模型桩：按序返回固定文本并计数。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators=language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)

    def sample_choice(self, prompt, responses, *, seed=None):  # type: ignore[no-untyped-def]
        raise AssertionError("本测试不使用选择采样")


def test_sampler_component_is_real_concordia_component_with_logging() -> None:
    """采样组件是真实 ComponentWithLogging：通道收到原始输入/输出。"""
    model = StubModel(["世界好"])
    component = SamplerComponent(model)
    log = structured_logging.SimulationLog()
    component.set_logging_channel(
        component_logging_channel(log, entity_name="lin", component_name="system1", step=lambda: 0)
    )
    assert component.sample("你好", component="system2") == "世界好"
    entries = log.get_entries_by_component("system1")
    assert len(entries) == 1
    data = dict(log.reconstruct_value(entries[0].deduplicated_data))
    assert data["Prompt"] == "你好"
    assert data["Response"] == "世界好"
    assert data["Component"] == "system2", "datum 标记实际阶段，而非固定通道名"
    # 组件协议完整：状态可导出/恢复（空状态）。
    assert component.get_state() == {}
    component.set_state({})


def test_language_model_sampler_bridges_async_without_blocking_loop() -> None:
    """异步桥可等待；事件循环在模型调用期间保持可用。"""

    async def scenario() -> None:
        model = StubModel(list(RESPONSES))
        sampler = LanguageModelSampler(model)
        marker = asyncio.Event()

        async def heartbeat() -> None:
            marker.set()

        task = asyncio.create_task(sampler.sample("p1", component="system1"))
        await asyncio.sleep(0)
        await heartbeat()
        assert await task == RESPONSES[0]
        assert model.calls == ["p1"]

    asyncio.run(scenario())


def test_recording_and_replay_round_trip(tmp_path) -> None:
    """录制精确请求与结果；重放严格匹配且不再触发真实采样。"""

    async def record() -> None:
        inner = ScriptedSampler(
            {
                "system1": [RESPONSES[0]],
                "system2": [RESPONSES[1]],
                "system3": [RESPONSES[2]],
            }
        )
        path = tmp_path / "record.jsonl"
        with RecordingSampler(inner, path, run_id="run-1") as recorder:
            for component in ("system1", "system2", "system3"):
                await recorder.sample(f"prompt-{component}", component=component)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["kind"] == "header"
        calls = [json.loads(line) for line in lines[1:]]
        assert [call["component"] for call in calls] == ["system1", "system2", "system3"]
        return path

    async def scenario() -> None:
        path = await record()
        replay = ReplaySampler(path)
        assert replay.header["run_id"] == "run-1"
        for component in ("system1", "system2", "system3"):
            assert (
                await replay.sample(f"prompt-{component}", component=component)
                == RESPONSES[["system1", "system2", "system3"].index(component)]
            )
        replay.assert_fully_consumed()
        with pytest.raises(ReplayMismatchError, match="调用次数超出录制"):
            await replay.sample("prompt-extra", component="system1")

    asyncio.run(scenario())


def test_replay_mismatches_are_refused(tmp_path) -> None:
    """请求与录制不一致（组件或提示）立即报错，不返回旧结果。"""

    async def scenario() -> None:
        inner = ScriptedSampler({"system1": [RESPONSES[0]]})
        path = tmp_path / "record.jsonl"
        with RecordingSampler(inner, path, run_id="run-2") as recorder:
            await recorder.sample("原始提示", component="system1")
        replay = ReplaySampler(path)
        with pytest.raises(ReplayMismatchError, match="请求与录制不一致"):
            await replay.sample("不同的提示", component="system1")
        replay2 = ReplaySampler(path)
        with pytest.raises(ReplayMismatchError, match="请求与录制不一致"):
            await replay2.sample("原始提示", component="system2")

    asyncio.run(scenario())


def test_recording_file_must_not_exist(tmp_path) -> None:
    path = tmp_path / "record.jsonl"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        RecordingSampler(ScriptedSampler(), path, run_id="run-3")


def test_role_run_record_then_replay_produces_identical_responses(tmp_path) -> None:
    """用角色跑一遍录制，再用重放采样器复现相同响应（无真实调用）。"""
    from sceneweave.domain.identity import CycleId
    from sceneweave.domain.messages import TriggerReason, TurnInput
    from sceneweave.domain.time import SimulationTime
    from sceneweave.roles.model_role import ModelRole

    async def run_with(sampler) -> object:
        role = ModelRole(rid("lin"), sampler)
        response = await role.perceive(
            TurnInput(
                cycle_id=CycleId("cycle-1"),
                role_id=rid("lin"),
                simulation_time=SimulationTime(0),
                trigger=TriggerReason((), False),
            )
        )
        return response

    async def scenario() -> None:
        inner = ScriptedSampler(
            {
                "system1": [RESPONSES[0]],
                "system2": [RESPONSES[1]],
                "system3": [RESPONSES[2]],
            }
        )
        path = tmp_path / "role-record.jsonl"
        with RecordingSampler(inner, path, run_id="role-run") as recorder:
            first = await run_with(recorder)
        replay = ReplaySampler(path)
        second = await run_with(replay)
        assert replay.assert_fully_consumed() is None
        assert second.response_id == first.response_id
        assert second.actions == first.actions

    asyncio.run(scenario())


def test_recording_captures_failures_and_replay_reproduces_them(tmp_path) -> None:
    """失败调用同样录制；重放按原错误身份重现，不重新采样。"""

    async def scenario() -> None:
        class FailingOnce(ScriptedSampler):
            def __init__(self) -> None:
                super().__init__({"system1": [RESPONSES[0]]})
                self.failed = False

            async def sample(self, prompt: str, *, component: str) -> str:
                if not self.failed:
                    self.failed = True
                    raise TimeoutError("模型超时")
                return await super().sample(prompt, component=component)

        path = tmp_path / "failure.jsonl"
        with RecordingSampler(FailingOnce(), path, run_id="fail-run") as recorder:
            with pytest.raises(TimeoutError):
                await recorder.sample("p-fail", component="system1")
            await recorder.sample("p-ok", component="system1")
        replay = ReplaySampler(path)
        with pytest.raises(RecordedModelError, match="TimeoutError.*模型超时"):
            await replay.sample("p-fail", component="system1")
        assert await replay.sample("p-ok", component="system1") == RESPONSES[0]
        replay.assert_fully_consumed()

    asyncio.run(scenario())


def test_replay_rejects_bad_headers_and_files(tmp_path) -> None:
    """头部/版本/运行身份/字段非法的录制文件明确拒绝。"""
    bad_files = {
        "no-header": '{"kind": "call", "component": "s", "prompt": "p", "response": "r"}\n',
        "bad-version": '{"kind": "header", "version": true, "run_id": "r"}\n',
        "empty-run": '{"kind": "header", "version": 1, "run_id": ""}\n',
        "missing-field": '{"kind": "header", "version": 1, "run_id": "r"}\n'
        '{"kind": "call", "component": "s", "prompt": "p"}\n',
        "unknown-kind": '{"kind": "header", "version": 1, "run_id": "r"}\n{"kind": "mystery"}\n',
        "not-json": "plain text\n",
    }
    for name, content in bad_files.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ReplayMismatchError):
            ReplaySampler(path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ReplayMismatchError):
        ReplaySampler(empty)


def test_request_mismatch_does_not_consume_record(tmp_path) -> None:
    """请求不匹配时不消费记录：修正后的请求仍能取到同一条。"""

    async def scenario() -> None:
        inner = ScriptedSampler({"system1": [RESPONSES[0]]})
        path = tmp_path / "order.jsonl"
        with RecordingSampler(inner, path, run_id="order") as recorder:
            await recorder.sample("正确提示", component="system1")
        replay = ReplaySampler(path)
        with pytest.raises(ReplayMismatchError):
            await replay.sample("错误提示", component="system1")
        assert await replay.sample("正确提示", component="system1") == RESPONSES[0]
        replay.assert_fully_consumed()

    asyncio.run(scenario())


def test_two_role_parallel_run_recorded_by_scope_and_replayed(tmp_path) -> None:
    """双角色并行运行：按角色作用域录制，重放零后端调用并复现行为。"""

    from sceneweave.domain.identity import EventId
    from sceneweave.domain.messages import WorldEvent
    from sceneweave.domain.time import SimulationTime
    from sceneweave.engine.config import RunConfig
    from sceneweave.engine.scheduler import RunStopReason, Scheduler
    from sceneweave.roles.model_role import ModelRole

    from .fakes import FakeSemanticWorld, ms

    async def run_once(samplers: dict) -> tuple:

        roles = [ModelRole(rid(name), samplers[name]) for name in ("alice", "bob")]
        world = FakeSemanticWorld(["alice", "bob"])
        scheduler = Scheduler(
            roles=roles,
            world=world,
            translator=ScriptedTranslator(plans={"走向窗户": (1000, 1000, ())}),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(10_000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        return result, world

    async def scenario() -> None:
        # 录制：两个作用域各自包裹一个脚本后端；用门强制交错顺序不确定。
        started = {name: asyncio.Event() for name in ("alice", "bob")}
        release = {name: asyncio.Event() for name in ("alice", "bob")}
        completed = {name: asyncio.Event() for name in ("alice", "bob")}

        def scripted_for(name: str) -> ScriptedSampler:
            class GatedSampler(ScriptedSampler):
                async def sample(self, prompt: str, *, component: str) -> str:
                    if component == "system1" and not started[name].is_set():
                        started[name].set()
                        await release[name].wait()
                        result = await super().sample(prompt, component=component)
                        completed[name].set()
                        return result
                    return await super().sample(prompt, component=component)

            return GatedSampler(
                {
                    "system1": [
                        '{"reconsider": true, "note": "' + name + ' 有变化"}',
                        '{"reconsider": false, "note": "照旧"}',
                        '{"reconsider": false, "note": "照旧"}',
                        '{"reconsider": false, "note": "照旧"}',
                    ],
                    "system2": ['{"intention": "看窗户", "abandon": []}'],
                    "system3": [
                        '{"feasible": true, "maintain": [], "modify": [],'
                        ' "new_actions": ["走向窗户"]}'
                    ],
                },
            )

        directory = tmp_path / "recordings"
        with RecordingDirectory(scripted_for, directory, run_id="dual") as recorder_dir:
            samplers = {name: recorder_dir.for_scope(name) for name in ("alice", "bob")}
            task = asyncio.create_task(run_once(samplers))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in started.values())), timeout=2
                )
                assert not task.done(), "两个角色均已开始采样，完整周期仍在等待"
                # 明确让 bob 先返回，不依赖墙钟延迟来制造并行顺序。
                release["bob"].set()
                await completed["bob"].wait()
                release["alice"].set()
                result, world = await task
            finally:
                for event in release.values():
                    event.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        assert result.reason is RunStopReason.TIME_LIMIT

        # 重放：每个作用域独立重放器；后端零调用（无后端可调）。
        replay_dir = ReplayDirectory(directory)
        replay_samplers = {name: replay_dir.for_scope(name) for name in ("alice", "bob")}
        replay_result, replay_world = await run_once(replay_samplers)
        replay_dir.assert_fully_consumed()
        assert replay_result.reason is result.reason
        assert replay_result.completed_cycles == result.completed_cycles
        assert replay_result.final_time == result.final_time
        assert replay_world.facts == world.facts, "客观事实一致"
        assert replay_world.adjudications == world.adjudications, "裁决输入一致"

    asyncio.run(scenario())


def test_language_model_sampler_channels_label_real_stage(tmp_path) -> None:
    """channel_for 按阶段提供通道：组件日志带真实阶段与因果身份。"""

    async def scenario() -> None:
        model = StubModel([RESPONSES[0], RESPONSES[1]])
        log = structured_logging.SimulationLog()
        steps = iter(range(100))
        sampler = LanguageModelSampler(
            model,
            channel_for=lambda component: component_logging_channel(
                log,
                entity_name="lin_zhou",
                component_name=component,
                step=lambda: next(steps),
            ),
        )
        await sampler.sample("提示一", component="system1")
        await sampler.sample("提示二", component="system2")
        assert [e.component_name for e in log.entries] == ["system1", "system2"]
        data1 = dict(log.reconstruct_value(log.entries[0].deduplicated_data))
        assert data1["Prompt"] == "提示一" and data1["Response"] == RESPONSES[0]

    asyncio.run(scenario())


def test_scopes_with_path_separators_record_inside_directory(tmp_path) -> None:
    async def scenario() -> None:
        directory = tmp_path / "recordings"
        with RecordingDirectory(
            lambda _: ScriptedSampler({"system1": [RESPONSES[0]]}), directory, run_id="scope"
        ) as recordings:
            await recordings.for_scope("角色/甲").sample("提示", component="system1")
        assert len(list(directory.iterdir())) == 1
        replay = ReplayDirectory(directory)
        assert await replay.for_scope("角色/甲").sample("提示", component="system1") == RESPONSES[0]
        replay.assert_fully_consumed()

    asyncio.run(scenario())


def test_replay_directory_refuses_mixed_runs(tmp_path) -> None:
    for scope, run_id in (("alice", "run-a"), ("bob", "run-b")):
        with RecordingDirectory(lambda _: ScriptedSampler(), tmp_path, run_id=run_id) as recordings:
            recordings.for_scope(scope)
    replay = ReplayDirectory(tmp_path)
    replay.for_scope("alice")
    with pytest.raises(ReplayMismatchError, match="同一运行身份"):
        replay.for_scope("bob")
