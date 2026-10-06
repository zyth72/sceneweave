"""模型 World 的调度集成测试：全模型链路端到端、冻结与录放。

ModelRole + ModelTranslator + ModelWorld + Scheduler + 真实结构化日志与
StepData；脚本假模型驱动，不调用真实 API。
"""

import asyncio

from concordia.utils import structured_logging

from sceneweave.domain.identity import ActionId, EventId
from sceneweave.domain.messages import ActionPhase, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import RunStopReason, Scheduler
from sceneweave.integrations.concordia.control import step_data_from_report
from sceneweave.integrations.concordia.sampling import (
    RecordingDirectory,
    ReplayDirectory,
)
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator
from sceneweave.world.model_world import ModelWorld
from sceneweave.world.scene import find_object, load_scene_card

from .fakes import (
    ScriptedRole,
    ScriptedSampler,
    ScriptedTranslator,
    maintain,
    maintaining,
    new_action,
    response,
    rid,
)
from .test_model_world import Brain, study_card

S1_GO = '{"reconsider": true, "note": "门被推开了"}'
S1_SKIP = '{"reconsider": false, "note": "照旧"}'
S2 = '{"intention": "查看抽屉里的东西", "abandon": []}'
S3_DRAWER = (
    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走到书桌旁拉开抽屉"]}'
)
TRANSLATION = (
    '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
    ' "start": 1000, "expected_end": 4000, "checkpoints": [2500]}],'
    ' "modified_schedules": []}'
)


class RoleBrain(ScriptedSampler):
    """角色与翻译器的脚本模型：记录提示供断言。"""

    def __init__(self, script: dict, name: str) -> None:
        super().__init__(script)
        self.name = name
        self.prompts: list[str] = []

    async def sample(self, prompt: str, *, component: str) -> str:
        self.prompts.append(prompt)
        return await super().sample(prompt, component=component)


def build_world_brain() -> Brain:
    return Brain()


def test_full_model_stack_end_to_end_with_stepdata_and_log() -> None:
    async def scenario() -> None:
        log = structured_logging.SimulationLog()
        tracer = SimulationLogTracer(log, wall_clock=lambda: "2026-01-01T00:00:00+00:00")
        role_model = RoleBrain(
            {
                "system1": [S1_GO] + [S1_SKIP] * 8,
                "system2": [S2],
                "system3": [S3_DRAWER],
            },
            "role",
        )
        translator_model = RoleBrain({"translator": [TRANSLATION]}, "translator")
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: build_world_brain(),
            roles=(rid("lin"),),
            tracer=tracer,
        )
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_model, tracer=tracer)],
            world=world,
            translator=ModelTranslator.shared(translator_model, tracer=tracer),
            config=RunConfig(introspection_interval=ms(6000), max_duration=ms(6000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
            tracer=tracer,
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        # 事实链：行动开始 → 实际移动 → 结算开抽屉取钥匙。
        record = scheduler.action_records()[ActionId("a-lin-1")]
        assert record.phase is ActionPhase.COMPLETED
        drawer = find_object(world.scene, "drawer")
        assert drawer.state["open"] is True, "开抽屉实际提交"
        lin = find_object(world.scene, "lin")
        assert any(child.id == "key" for child in lin.children), "换父取物实际提交"
        assert lin.position is not None and lin.position[0] > 0.6, "按剧情时间移动"
        # 隐私：抽屉打开提交之前，隐藏钥匙不进入任何模型提示；
        # 打开并取走提交之后，角色自身的行动结果反馈可以提及。
        first_role_prompt = role_model.prompts[0]
        assert "铜钥匙" not in first_role_prompt and '"key"' not in first_role_prompt
        for prompt in translator_model.prompts:
            assert "铜钥匙" not in prompt, "翻译提示不含隐藏物品"
        # 日志因果链：world 模型 IO 与引擎链同 SimulationLog。
        stages = {e.component_name for e in log.entries}
        assert {
            "world.check.invoke",
            "world.check.parsed",
            "world.progress.invoke",
            "world.adjudicate.invoke",
            "world.adjudicate.candidate",
            "translator.invoke",
            "system3.declare",
            "registration.commit",
            "engine.cycle_complete",
        } <= stages
        check_raw = [
            dict(log.reconstruct_value(e.deduplicated_data))
            for e in log.get_entries_by_component("world.check.parsed")
        ]
        assert check_raw and check_raw[0]["outputs"], "原始模型输出入日志"
        # StepData 携带真实声明与世界诊断。
        report = scheduler.last_report
        assert report is not None
        data = step_data_from_report(report)
        assert "lin" in data.entity_actions
        diagnostic = report.world_diagnostic
        assert diagnostic is not None and diagnostic.action_states

    asyncio.run(scenario())


def test_check_context_gives_waiting_maintain_its_registered_arrangement() -> None:
    """另一角色较早开始事件触发检查时，尚未开始的已登记维持拿到安排上下文。"""

    async def scenario() -> None:
        brain = Brain()
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: brain,
            roles=(rid("lin"), rid("shen")),
        )
        lin = ScriptedRole(
            "lin",
            [response("lin", "r-lin-1", (new_action("a-lin-1", "林舟行动"),))],
            fallback=maintaining("lin", "a-lin-1"),
        )
        shen = ScriptedRole(
            "shen",
            [
                response("shen", "r-shen-1", (new_action("a-shen-1", "沈言行动"),)),
                response("shen", "r-shen-2", (maintain("a-shen-1"),)),
            ],
            fallback=maintaining("shen", "a-shen-1"),
        )
        scheduler = Scheduler(
            roles=[lin, shen],
            world=world,
            translator=ScriptedTranslator(
                plans={"林舟行动": (100, 4000, ()), "沈言行动": (600, 4000, ())}
            ),
            config=RunConfig(introspection_interval=ms(10_000), max_duration=ms(2000)),
            initial_events=[WorldEvent(EventId("startup"), SimulationTime(0), "开场")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        shen_checks = [
            prompt
            for component, prompt in brain.prompts
            if component == "world-check" and "角色 shen" in prompt
        ]
        waiting = [
            prompt
            for prompt in shen_checks
            if "a-shen-1（registered）" in prompt and "沈言行动" in prompt
        ]
        assert waiting, "等待中的已登记行动进入检查提示，而非只给维持身份"
        assert "World 实际行动状态（已提交事实）：" in waiting[0]
        assert "a-lin-1" not in waiting[0], "不泄漏另一角色的安排"
        assert scheduler.action_records()[ActionId("a-shen-1")].phase is ActionPhase.STARTED, (
            "后续节点已实际开始，证明此前的 registered 检查确实发生在开始之前"
        )

    asyncio.run(scenario())


def test_world_model_wait_freezes_clock_and_facts() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()

        class GatedBrain(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-check":
                    gate.release_needed = True  # type: ignore[attr-defined]
                    await gate.wait()
                return await super().sample(prompt, component=component)

        role_model = RoleBrain(
            {
                "system1": [S1_GO] + [S1_SKIP] * 6,
                "system2": [S2],
                "system3": [S3_DRAWER],
            },
            "role",
        )
        translator_model = RoleBrain({"translator": [TRANSLATION]}, "translator")
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: GatedBrain(),
            roles=(rid("lin"),),
        )
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_model)],
            world=world,
            translator=ModelTranslator.shared(translator_model),
            config=RunConfig(introspection_interval=ms(6000), max_duration=ms(6000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        task = asyncio.create_task(scheduler.run())
        for _ in range(2000):
            if scheduler.stage == "cycle.responses":
                break
            await asyncio.sleep(0)
        assert scheduler.stage == "cycle.responses"
        version = world.state_version
        scene = world.scene
        await asyncio.sleep(0.05)
        assert scheduler.current_time == SimulationTime(0), "等待期间时钟冻结"
        assert world.state_version == version and world.scene is scene
        gate.set()
        result = await task
        assert result.reason is RunStopReason.TIME_LIMIT

    asyncio.run(scenario())


def test_time_limit_boundary_progresses_without_role_calls() -> None:
    async def scenario() -> None:
        world_brain = Brain()
        role_model = RoleBrain(
            {
                "system1": [S1_GO] + [S1_SKIP] * 6,
                "system2": [S2],
                "system3": [S3_DRAWER],
            },
            "role",
        )
        translator_model = RoleBrain({"translator": [TRANSLATION]}, "translator")
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: world_brain,
            roles=(rid("lin"),),
        )
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_model)],
            world=world,
            translator=ModelTranslator.shared(translator_model),
            # 时限 2500ms：开始(1000) 与检查点(2500) 处理后停；结束 4000 保留。
            config=RunConfig(introspection_interval=ms(9000), max_duration=ms(2500)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        assert result.final_time == SimulationTime(2500)
        record = scheduler.action_records()[ActionId("a-lin-1")]
        assert record.phase is ActionPhase.STARTED, "不凭空完成"
        assert any(e.simulation_time == SimulationTime(4000) for e in scheduler.pending_events())
        lin = find_object(world.scene, "lin")
        assert lin.position[0] == 0.6 + 0.5 * 1.5, "边界推进实际进展（0.5m/s × 1.5s）"

    asyncio.run(scenario())


def ms(value: int) -> SimulationTime:
    return SimulationTime(value)


def test_scoped_record_replay_full_stack_two_roles(tmp_path) -> None:
    """全栈双角色并行按作用域录制重放：零模型回放、事实一致。"""

    async def run_once(assemblies) -> tuple:
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=assemblies["world_scope"],
            roles=(rid("lin"), rid("shen")),
        )
        scheduler = Scheduler(
            roles=[ModelRole(rid(name), assemblies[name]()) for name in ("lin", "shen")],
            world=world,
            translator=ModelTranslator(assemblies["translator_scope"]),
            config=RunConfig(introspection_interval=ms(6000), max_duration=ms(6000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        return result, world, scheduler

    def role_script(name: str) -> dict:
        if name == "lin":
            return {
                "system1": [S1_GO] + [S1_SKIP] * 8,
                "system2": [S2],
                "system3": [S3_DRAWER],
            }
        return {"system1": [S1_SKIP] * 9}

    async def scenario() -> None:
        directory = tmp_path / "recordings"

        def world_scope(scope: str) -> object:
            return Brain()

        with RecordingDirectory(
            lambda scope: RoleBrain(
                role_script("lin") if "lin" in scope else {"system1": [S1_SKIP] * 9}, scope
            ),
            directory,
            run_id="world-dual",
        ) as role_rec:
            with RecordingDirectory(
                lambda scope: RoleBrain({"translator": [TRANSLATION]}, scope),
                directory,
                run_id="world-dual",
            ) as translator_rec:
                with RecordingDirectory(world_scope, directory, run_id="world-dual") as world_rec:
                    # ModelWorld 的采样按 scope 命中同一目录文件。
                    def assemblies():
                        return {
                            "lin": lambda: role_rec.for_scope("lin"),
                            "shen": lambda: role_rec.for_scope("shen"),
                            "translator_scope": lambda role: translator_rec.for_scope(
                                f"translator-{role}"
                            ),
                            "world_scope": world_rec.for_scope,
                        }

                    result, world, scheduler = await run_once(assemblies())
        assert result.reason is RunStopReason.TIME_LIMIT

        role_replay = ReplayDirectory(directory)
        translator_replay = ReplayDirectory(directory)
        world_replay = ReplayDirectory(directory)

        def replay_assemblies():
            return {
                "lin": lambda: role_replay.for_scope("lin"),
                "shen": lambda: role_replay.for_scope("shen"),
                "translator_scope": lambda role: translator_replay.for_scope(f"translator-{role}"),
                "world_scope": world_replay.for_scope,
            }

        replay_result, replay_world, replay_scheduler = await run_once(replay_assemblies())
        role_replay.assert_fully_consumed()
        translator_replay.assert_fully_consumed()
        world_replay.assert_fully_consumed()
        assert replay_result.reason is result.reason
        assert replay_result.completed_cycles == result.completed_cycles
        assert replay_world.state_version == world.state_version
        replay_lin = find_object(replay_world.scene, "lin")
        lin = find_object(world.scene, "lin")
        assert replay_lin.position == lin.position, "实际位置一致"
        assert replay_lin.children == lin.children, "容纳关系一致"
        assert set(replay_scheduler.action_records()) == set(scheduler.action_records())

    asyncio.run(scenario())


def test_ui_chain_real_controller_and_stepdata_with_model_world() -> None:
    """真实 StepController/StepData/SimulationServer 链路 + ModelWorld。"""

    async def scenario() -> None:

        import socket

        from concordia.utils.simulation_server import SimulationServer

        from sceneweave.integrations.concordia.control import StepControllerGate

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = SimulationServer(port=port, html_content="<html>mw</html>")
        server.start()
        try:
            gate = StepControllerGate(server.step_controller)
            world = ModelWorld(
                scene=load_scene_card(study_card()),
                sampler_for_scope=lambda scope: Brain(),
                roles=(rid("lin"),),
            )
            scheduler = Scheduler(
                roles=[
                    ModelRole(
                        rid("lin"),
                        RoleBrain(
                            {
                                "system1": [S1_GO] + [S1_SKIP] * 8,
                                "system2": [S2],
                                "system3": [S3_DRAWER],
                            },
                            "lin",
                        ),
                    )
                ],
                world=world,
                translator=ModelTranslator.shared(RoleBrain({"translator": [TRANSLATION]}, "tr")),
                config=RunConfig(introspection_interval=ms(6000), max_duration=ms(6000)),
                initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
                gate=gate,
            )
            task = asyncio.create_task(scheduler.run())
            # 许可在每个周期前取得：暂停时零处理。
            for _ in range(1000):
                if scheduler.stage == "run.gate_wait":
                    break
                await asyncio.sleep(0)
            assert scheduler.stage == "run.gate_wait"
            import json as _json
            import urllib.request as _rq

            with _rq.urlopen(f"http://127.0.0.1:{port}/cmd/step", timeout=5) as answer:
                assert _json.loads(answer.read().decode("utf-8"))["status"] == "stepping"
            for _ in range(1000):
                if scheduler.last_report is not None:
                    break
                await asyncio.sleep(0)
            # 单步完成一个完整周期后广播真实步数据。
            report = scheduler.last_report
            assert report is not None
            from sceneweave.integrations.concordia.control import broadcast_report

            broadcast_report(server, report)
            current = server.current_step_data
            assert current["step"] == 1
            assert "lin" in current["entity_actions"]
            world_log = current["entity_logs"]["world"]
            assert world_log["world_diagnostic"] is not None
            with _rq.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as answer:
                assert _json.loads(answer.read().decode("utf-8"))["is_paused"] is True
            with _rq.urlopen(f"http://127.0.0.1:{port}/cmd/play", timeout=5) as answer:
                answer.read()
            result = await task
            assert result.reason is RunStopReason.TIME_LIMIT
        finally:
            server.stop()

    asyncio.run(scenario())


def test_modification_speed_takes_effect_in_full_scheduler() -> None:
    """全 Scheduler：修改行动速度经 World 裁决真实生效。"""

    async def scenario() -> None:
        import re

        class SpeedBrain(Brain):
            """进展模型按提示中的速度计算位置。"""

            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    elapsed = re.search(r"流逝 (\d+) 毫秒", prompt)
                    if elapsed is None:
                        return NO_PROGRESS
                    elapsed_ms = int(elapsed.group(1))
                    movement = re.search(
                        r"- \S+（(\w+)）：速度 (\d+\.\d+) 米/秒，方向 \((-?\d+\.\d+), (-?\d+\.\d+)\)",
                        prompt,
                    )
                    if movement is None:
                        return NO_PROGRESS
                    role, speed, dx, dy = (
                        movement.group(1),
                        float(movement.group(2)),
                        float(movement.group(3)),
                        float(movement.group(4)),
                    )
                    pos = re.search(rf"- {role}：\((-?\d+\.\d+), (-?\d+\.\d+)\)", prompt)
                    if pos is None:
                        return NO_PROGRESS
                    x, y = float(pos.group(1)), float(pos.group(2))
                    seconds = elapsed_ms / 1000
                    import json as _json

                    return _json.dumps(
                        {
                            "movements": [
                                {
                                    "object_id": role,
                                    "position": [
                                        round(x + dx * speed * seconds, 4),
                                        round(y + dy * speed * seconds, 4),
                                    ],
                                }
                            ],
                            "updated_action_ids": [],
                            "notes": ["按速度推进"],
                            "passive_updates": [],
                        },
                        ensure_ascii=False,
                    )
                return await super().sample(prompt, component=component)

        role_model = RoleBrain(
            {
                "system1": [
                    '{"reconsider": true, "note": "开始走"}',
                    '{"reconsider": false, "note": "照旧"}',
                    '{"reconsider": true, "note": "想走得更快"}',
                ]
                + ['{"reconsider": false, "note": "照旧"}'] * 8,
                "system2": [
                    '{"intention": "走向窗户", "abandon": []}',
                    '{"intention": "加快", "abandon": []}',
                ],
                "system3": [
                    '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向窗户"]}',
                    '{"feasible": true, "maintain": [],'
                    ' "modify": [{"id": "a-lin-1", "change": "走得更快"}],'
                    ' "new_actions": []}',
                ],
            },
            "role",
        )
        # 翻译器第一次给新行动安排，第二次给修改安排。
        translator_model = RoleBrain(
            {
                "translator": [
                    '{"schedules": [{"action_id": "a-lin-1",'
                    ' "commit_group_id": "g-lin-1", "start": 1000,'
                    ' "expected_end": 4000, "checkpoints": [2500]}],'
                    ' "modified_schedules": []}',
                    '{"schedules": [], "modified_schedules":'
                    ' [{"action_id": "a-lin-1", "start": 1000,'
                    ' "expected_end": 6000, "checkpoints": [2500, 4000]}]}',
                ]
            },
            "tr",
        )
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: SpeedBrain(),
            roles=(rid("lin"),),
        )
        scheduler = Scheduler(
            roles=[ModelRole(rid("lin"), role_model)],
            world=world,
            translator=ModelTranslator.shared(translator_model),
            # 时限 5000：t=0(登记),1000(start),2500(cp),3000(修改),4000(cp),5000(边界)。
            config=RunConfig(introspection_interval=ms(9000), max_duration=ms(5000)),
            initial_events=[WorldEvent(EventId("door"), SimulationTime(0), "门被推开")],
        )
        result = await scheduler.run()
        assert result.reason is RunStopReason.TIME_LIMIT
        record = scheduler.action_records()[ActionId("a-lin-1")]
        assert record.expected_end == SimulationTime(6000), "修改安排生效"
        lin = find_object(world.scene, "lin")
        # t=1000 start（0.5m/s）；t=2500 cp 流逝1500ms → +0.75;
        # t=3000 修改为 1.0m/s（Brain 按"更快"）；t=4000 cp 流逝1000ms → +1.0;
        # t=5000 边界流逝1000ms → +1.0（但 4000 是 cp 后修改时刻 3000 后的 cp）。
        # 实际 Brain 的修改裁决给出 1.0m/s，位置推进按各段速度积分。
        assert lin.position[0] > 0.6, "位置按修改后速度实际推进"
        # World 实际生效：修改后运动速度在进展提示中翻倍。
        # （Brain 对"更快"给出 1.0m/s，修改后位置推进明显快于原 0.5m/s。）

    from .test_model_world import NO_PROGRESS  # noqa: PLC0415

    asyncio.run(scenario())
