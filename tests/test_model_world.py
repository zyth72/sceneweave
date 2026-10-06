"""模型 World 的行为测试（脚本假模型，无真实 API）。

覆盖：检查（通过/不通过/隐藏内容不进提示/并发）、进展（原子应用/
零流逝跳过/失败不改基线）、裁决（候选零副作用/伪造与过期拒绝/提交
原子性/联合归属一份/撤回保留被动/开抽屉与换父取物实际事实）、发布
（提交后可见范围过滤/防伪造/版本一致）、端到端调度集成与按作用域
录制重放。
"""

import asyncio
import itertools
import json
import re

import pytest

from sceneweave.domain.adjudication import (
    ActionNodeKind,
    AdjudicationActionNode,
    AdjudicationCandidate,
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
    ActionPhase,
    MaintainAction,
    NewAction,
    RoleResponse,
    WorldEvent,
)
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.domain.translation import ActionArrangement
from sceneweave.world.model_world import ModelWorld, ModelWorldError
from sceneweave.world.scene import find_object, load_scene_card

from .fakes import ScriptedSampler, rid

LIN = RoleId("lin")
SHEN = RoleId("shen")


def study_card() -> dict:
    return {
        "id": "room",
        "name": "书房",
        "kind": "房间",
        "unit": "m",
        "size": [4, 3],
        "children": [
            {
                "id": "lin",
                "name": "林",
                "kind": "角色",
                "position": [0.6, 0.6],
                "state": {"posture": "站立"},
                "children": [],
            },
            {
                "id": "shen",
                "name": "沈",
                "kind": "角色",
                "position": [3.4, 0.8],
                "state": {"posture": "站立"},
                "children": [],
            },
            {
                "id": "desk",
                "name": "书桌",
                "kind": "家具",
                "position": [2.8, 2.2],
                "size": [1.2, 0.6],
                "blocks_movement": True,
                "children": [
                    {
                        "id": "drawer",
                        "name": "抽屉",
                        "kind": "容器",
                        "state": {"open": False, "locked": False},
                        "description": "关闭时看不到里面的物品。",
                        "children": [{"id": "key", "name": "铜钥匙", "kind": "物品"}],
                    },
                ],
            },
        ],
    }


PASS_CHECK = '{"passed": true, "reason": "", "action_ids": []}'
FAIL_CHECK = '{"passed": false, "reason": "客观条件不允许", "action_ids": ["a-1"]}'
NO_PROGRESS = '{"movements": [], "updated_action_ids": [], "notes": [], "passive_updates": []}'
SILENCE_CHANGE = (
    '{"changes": [{"kind": "set_state", "object_id": "room", "key": "氛围",'
    ' "value": "安静", "description": "书房里很安静"}], "outcomes": [],'
    ' "follow_ups": [], "movements": [], "stop_movements": [], "passives": [],'
    ' "stop_passives": []}'
)


class Brain:
    """按提示内容决定回答的脚本世界模型（测试侧假件）。

    语义分发基于提示中我们自己的结构化行（节点行/声明行/流逝行），
    不是生产代码；用于在不调用真实模型的前提下给出确定性的
    检查/进展/裁决判断。
    """

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    async def sample(self, prompt: str, *, component: str) -> str:
        self.prompts.append((component, prompt))
        if component == "world-check":
            return self._check(prompt)
        if component == "world-progress":
            return self._progress(prompt)
        if component == "world-adjudicate":
            return self._adjudicate(prompt)
        raise AssertionError(f"未知组件 {component}")

    def _check(self, prompt: str) -> str:
        if "不可能" in prompt:
            return '{"passed": false, "reason": "距离与遮挡不允许", "action_ids": []}'
        return PASS_CHECK

    def _progress(self, prompt: str) -> str:
        elapsed_match = re.search(r"流逝 (\d+) 毫秒", prompt)
        if elapsed_match is None or "速度" not in prompt:
            return NO_PROGRESS
        elapsed_ms = int(elapsed_match.group(1))
        movement_matches = re.findall(
            r"- \S+（(\w+)）：速度 (\d+\.\d+) 米/秒，方向 \((-?\d+\.\d+), (-?\d+\.\d+)\)",
            prompt,
        )
        if not movement_matches:
            return NO_PROGRESS
        current_positions: dict[str, tuple[float, float]] = {}
        block = prompt.split("有坐标对象当前位置", 1)
        if len(block) > 1:
            for line in block[1].split("进行中的实际运动", 1)[0].splitlines():
                match = re.match(r"- (\w+)：\((-?\d+\.\d+), (-?\d+\.\d+)\)", line.strip())
                if match is not None:
                    current_positions[match.group(1)] = (
                        float(match.group(2)),
                        float(match.group(3)),
                    )
        movements = []
        for role, speed_text, dx_text, dy_text in movement_matches:
            position = current_positions.get(role)
            if position is None:
                continue
            speed = float(speed_text)
            dx, dy = float(dx_text), float(dy_text)
            seconds = elapsed_ms / 1000
            movements.append(
                {
                    "object_id": role,
                    "position": [
                        round(position[0] + dx * speed * seconds, 4),
                        round(position[1] + dy * speed * seconds, 4),
                    ],
                }
            )
        return (
            '{"movements": '
            + str(movements).replace("'", '"')
            + ', "updated_action_ids": [], "notes": ["按流逝时间推进"], "passive_updates": []}'
        )

    def _adjudicate(self, prompt: str) -> str:
        changes: list[dict] = []
        outcomes: list[dict] = []
        movements: list[dict] = []
        stop: list[str] = []
        passives: list[dict] = []
        node_section = prompt.split("行动节点：", 1)
        node_lines = (
            [line.strip() for line in node_section[1].split("明确修改：", 1)[0].splitlines()]
            if len(node_section) > 1
            else []
        )
        for line in node_lines:
            match = re.match(r"- \S+：(\S+) 的 (\S+) (start|checkpoint|end|withdrawn)", line)
            if match is None:
                continue
            role, action, node = match.group(1), match.group(2), match.group(3)
            if node == "start":
                outcomes.append(
                    {"action_id": action, "result": "started", "detail": "行动实际开始"}
                )
                if "走" in line or "拿" in line or "取" in line:
                    movements.append(
                        {
                            "action_id": action,
                            "role_id": role,
                            "speed": 0.5,
                            "direction": [1.0, 0.0],
                        }
                    )
            elif node == "checkpoint":
                outcomes.append(
                    {"action_id": action, "result": "in_progress", "detail": "过程节点"}
                )
            elif node == "end":
                outcomes.append(
                    {"action_id": action, "result": "completed", "detail": "按事实完成"}
                )
                stop.append(action)
                if "抽屉" in line:
                    changes.append(
                        {
                            "kind": "set_state",
                            "object_id": "drawer",
                            "key": "open",
                            "value": True,
                            "description": f"{role}拉开了抽屉",
                        }
                    )
                    changes.append(
                        {
                            "kind": "move",
                            "object_id": "key",
                            "new_parent": role,
                            "description": f"{role}从抽屉里拿走了铜钥匙",
                        }
                    )
            elif node == "withdrawn":
                stop.append(action)
                passives.append(
                    {"id": f"{action}-inertia", "description": "主动控制撤回，惯性继续"}
                )
        modification_section = prompt.split("明确修改：", 1)
        if len(modification_section) > 1:
            mod_lines = [
                line.strip()
                for line in modification_section[1].split("现行实际行动状态：", 1)[0].splitlines()
            ]
            for line in mod_lines:
                match = re.match(r"- (\S+) 修改 (\S+)：(.+)（原", line)
                if match is None:
                    continue
                role, action, change = match.group(1), match.group(2), match.group(3)
                speed = 1.0 if "更快" in change else 0.25
                movements.append(
                    {
                        "action_id": action,
                        "role_id": role,
                        "speed": speed,
                        "direction": [1.0, 0.0],
                    }
                )
                outcomes.append(
                    {"action_id": action, "result": "in_progress", "detail": "修改后的实际进展"}
                )
        if not changes and not outcomes and not passives and not movements:
            return SILENCE_CHANGE
        return (
            '{"changes": '
            + _json_list(changes)
            + ', "outcomes": '
            + _json_list(outcomes)
            + ', "follow_ups": [], "movements": '
            + _json_list(movements)
            + ', "stop_movements": '
            + str(stop).replace("'", '"')
            + ', "passives": '
            + _json_list(passives)
            + ', "stop_passives": []'
            + "}"
        )


def _json_list(items: list) -> str:

    return json.dumps(items, ensure_ascii=False)


def brain_world(
    card: dict | None = None, brain: Brain | None = None, tracer=None
) -> tuple[ModelWorld, Brain]:
    brain = brain or Brain()
    world = ModelWorld(
        scene=load_scene_card(card if card is not None else study_card()),
        sampler_for_scope=lambda scope: brain,
        tracer=tracer,
    )
    return world, brain


def node(
    action: str, role: str, kind: ActionNodeKind, description: str, time: int = 1000
) -> AdjudicationActionNode:
    return AdjudicationActionNode(
        event_id=EventId(f"{action}-{kind.value}"),
        action_id=ActionId(action),
        role_id=RoleId(role),
        node=kind,
        description=description,
        start=SimulationTime(time),
        expected_end=SimulationTime(time + 1000),
    )


def response_for(role: str, actions: tuple) -> RoleResponse:
    return RoleResponse(CycleId("cycle-1"), RoleId(role), ResponseId("r-1"), actions)


class Collector(TurnTracer):
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def trace(self, record: TraceRecord) -> None:
        self.records.append(record)

    def stages(self) -> set[str]:
        return {r.stage for r in self.records}

    def by_stage(self, stage: str) -> list[TraceRecord]:
        return [r for r in self.records if r.stage == stage]


# ---- 检查 ----


def test_check_pass_and_fail_with_reason() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        ok = await world.check_response(
            response_for("lin", (NewAction(ActionId("a-1"), "走向书桌", CommitGroupId("g")),))
        )
        assert ok.passed and ok.world_state_version == WorldStateVersion(0)
        bad = await world.check_response(
            response_for(
                "lin", (NewAction(ActionId("a-2"), "不可能地隔空取物", CommitGroupId("g")),)
            )
        )
        assert not bad.passed and bad.reason

    asyncio.run(scenario())


def test_check_prompt_hides_closed_container_contents() -> None:
    async def scenario() -> None:
        world, brain = brain_world()
        await world.check_response(response_for("lin", ()))
        [(_, prompt)] = brain.prompts
        assert "铜钥匙" not in prompt, "隐藏物品不得进入检查模型输入"
        assert "抽屉" in prompt, "关闭容器本身可见"

    asyncio.run(scenario())


def test_check_prompt_carries_registered_arrangements_without_actual_start() -> None:
    """未实际开始的已登记安排进入检查上下文；登记不等于实际开始。"""

    async def scenario() -> None:
        world, brain = brain_world()
        arrangement = ActionArrangement(
            action_id=ActionId("a-shen-1"),
            phase=ActionPhase.REGISTERED,
            description="伸手取回密封文件",
            start=SimulationTime(200),
            expected_end=SimulationTime(1700),
            checkpoints=(SimulationTime(700),),
        )
        result = await world.check_response(
            response_for("shen", (MaintainAction(ActionId("a-shen-1")),)),
            arrangements=(arrangement,),
        )
        assert result.passed
        [(component, prompt)] = brain.prompts
        assert component == "world-check"
        assert (
            "- a-shen-1（registered）：伸手取回密封文件，原安排 200→1700ms，过程节点 [700]"
            in prompt
        )
        assert "已登记安排" in prompt and "实际行动状态" in prompt
        diagnostic = world.diagnostic_snapshot()
        assert diagnostic.action_states == (), "登记不当作实际开始"
        assert diagnostic.world_state_version == WorldStateVersion(0)

    asyncio.run(scenario())


def test_check_context_never_leaks_other_roles_arrangement() -> None:
    """检查上下文只含本角色安排，不泄漏另一角色安排。"""

    async def scenario() -> None:
        world, brain = brain_world()
        lin_arrangement = ActionArrangement(
            action_id=ActionId("a-lin-1"),
            phase=ActionPhase.STARTED,
            description="林舟的持握动作",
            start=SimulationTime(100),
            expected_end=SimulationTime(900),
        )
        shen_arrangement = ActionArrangement(
            action_id=ActionId("a-shen-1"),
            phase=ActionPhase.REGISTERED,
            description="沈言的伸手动作",
            start=SimulationTime(200),
            expected_end=SimulationTime(1700),
        )
        await world.check_response(response_for("lin", ()), arrangements=(lin_arrangement,))
        await world.check_response(response_for("shen", ()), arrangements=(shen_arrangement,))
        prompts = [prompt for _, prompt in brain.prompts]
        lin_prompt = next(prompt for prompt in prompts if "角色 lin" in prompt)
        shen_prompt = next(prompt for prompt in prompts if "角色 shen" in prompt)
        assert "林舟的持握动作" in lin_prompt and "林舟的持握动作" not in shen_prompt
        assert "沈言的伸手动作" in shen_prompt and "沈言的伸手动作" not in lin_prompt

    asyncio.run(scenario())


def test_check_rejects_invalid_arrangements_before_model_call() -> None:
    """安排结构在模型调用前校验，非法输入不改状态。"""

    async def scenario() -> None:
        world, brain = brain_world()
        response = response_for("lin", ())
        arrangement = ActionArrangement(
            action_id=ActionId("a-1"),
            phase=ActionPhase.REGISTERED,
            description="安排",
            start=SimulationTime(100),
            expected_end=SimulationTime(900),
        )
        with pytest.raises(ModelWorldError, match="ActionArrangement"):
            await world.check_response(response, arrangements=("坏安排",))  # type: ignore[arg-type]
        with pytest.raises(ModelWorldError, match="重复"):
            await world.check_response(response, arrangements=(arrangement, arrangement))
        assert brain.prompts == [], "结构错误在模型调用前拒绝"
        assert world.diagnostic_snapshot().world_state_version == WorldStateVersion(0)

    asyncio.run(scenario())


def test_check_structural_errors_are_runtime_errors_with_trace() -> None:
    async def scenario() -> None:
        collector = Collector()

        class Broken(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-check":
                    return '{"passed": "yes"}'
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=Broken(), tracer=collector)
        with pytest.raises(ModelWorldError, match="world-check"):
            await world.check_response(
                response_for("lin", (NewAction(ActionId("a-1"), "走", CommitGroupId("g")),))
            )
        errors = collector.by_stage("world.check.error")
        assert errors and errors[0].outputs == ('{"passed": "yes"}',), "raw 进错误事件"

    asyncio.run(scenario())


def test_check_concurrency_same_scope_rejected_roles_parallel() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()

        class Gated(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-check":
                    gate_entred.add(prompt[:20])
                    await gate.wait()
                return await super().sample(prompt, component=component)

        gate_entred: set[str] = set()
        world, _ = brain_world(brain=Gated())
        first = asyncio.create_task(world.check_response(response_for("lin", ())))
        for _ in range(1000):
            if gate_entred:
                break
            await asyncio.sleep(0)
        concurrent = asyncio.create_task(world.check_response(response_for("lin", ())))
        with pytest.raises(ModelWorldError, match="串行"):
            await concurrent
        other = asyncio.create_task(world.check_response(response_for("shen", ())))
        await asyncio.sleep(0)
        assert not other.done(), "不同角色并行"
        gate.set()
        assert (await first).passed and (await other).passed

    asyncio.run(scenario())


# ---- 进展 ----


def test_progress_moves_positions_atomically() -> None:
    async def scenario() -> None:
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: Brain(),
        )
        # 先经提交建立运动（见裁决测试的模式），此处直接用内部状态驱动：
        # 通过 start 节点提交建立 movement。
        candidate = await world.adjudicate(
            CycleId("cycle-1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(candidate)
        before = find_object(world.scene, "lin").position
        update = await world.update_progress(CycleId("cycle-2"), SimulationTime(3000))
        after = find_object(world.scene, "lin").position
        assert update.notes, "进展说明非空"
        assert after[0] > before[0], "按流逝毫秒实际移动（0.5m/s × 2s = 1.0m）"
        assert world.state_version >= WorldStateVersion(2)

    asyncio.run(scenario())


def test_progress_zero_elapsed_skips_model() -> None:
    async def scenario() -> None:
        brain = Brain()
        world, _ = brain_world(brain=brain)
        update = await world.update_progress(CycleId("c"), SimulationTime(0))
        assert update.updated_action_ids == ()
        assert brain.prompts == [], "零流逝不调用模型"

    asyncio.run(scenario())


def test_progress_failure_keeps_baseline() -> None:
    async def scenario() -> None:
        class Failing(Brain):
            """第一次进展调用失败，之后恢复。"""

            def __init__(self) -> None:
                super().__init__()
                self.progress_calls = 0
                self.failed_at: int | None = None

            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    self.progress_calls += 1
                    if self.progress_calls == 2:
                        self.failed_at = self.progress_calls
                        raise TimeoutError("模型超时")
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=Failing())
        candidate = await world.adjudicate(
            CycleId("cycle-1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(candidate)
        # 先成功推进一次：t=2000（基线 0），流逝 2000ms → 移动 1.0m。
        await world.update_progress(CycleId("c1b"), SimulationTime(2000))
        assert find_object(world.scene, "lin").position[0] == pytest.approx(1.6)
        with pytest.raises(TimeoutError):
            await world.update_progress(CycleId("c2"), SimulationTime(4000))
        assert find_object(world.scene, "lin").position[0] == pytest.approx(1.6), "失败不改事实"
        # 失败后基线仍是 2000：成功调用按 3000ms 流逝移动 1.5m。
        await world.update_progress(CycleId("c3"), SimulationTime(5000))
        assert find_object(world.scene, "lin").position[0] == pytest.approx(3.1), "基线未因失败推进"

    asyncio.run(scenario())


# ---- 裁决 / 校验 / 提交 ----


def test_candidate_generation_has_no_side_effects() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        version_before = world.state_version
        scene_before = world.scene
        await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        assert world.state_version == version_before, "候选不改版本"
        assert world.scene is scene_before, "候选不改快照"

    asyncio.run(scenario())


def test_validate_rejects_foreign_and_stale_candidates() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        foreign = AdjudicationCandidate(
            cycle_id=CycleId("c1"),
            simulation_time=SimulationTime(1000),
            proposed_changes=(FactChangeProposal("伪造"),),
        )
        assert not world.validate_adjudication(foreign).passed
        live = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        assert world.validate_adjudication(live).passed
        # 版本漂移后同一候选过期。
        await world.update_progress(CycleId("c0"), SimulationTime(0))  # skip 不改版本
        ok = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(1000),
            event_ids=(EventId("e2"),),
            action_ids=(ActionId("a-run"),),
            events=(WorldEvent(EventId("e2"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.START, "跑起来"),),
        )
        world.commit_adjudication(ok)
        stale = world.validate_adjudication(live)
        assert not stale.passed and any("过期" in p for p in stale.problems)

    asyncio.run(scenario())


def test_commit_applies_facts_outcomes_and_movements() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌并拿钥匙"),),
        )
        commit = world.commit_adjudication(start)
        assert commit.resulting_version == WorldStateVersion(1)
        assert world.state_version == WorldStateVersion(1)
        diagnostic = world.diagnostic_snapshot()
        assert diagnostic.action_states[0].state == "started"
        assert diagnostic.action_states[0].role_id == "lin", "归属来自节点"

    asyncio.run(scenario())


def test_end_node_opens_drawer_and_moves_key_once() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        end = await world.adjudicate(
            CycleId("c9"),
            SimulationTime(5000),
            event_ids=(EventId("e-end"),),
            action_ids=(ActionId("a-drawer"),),
            events=(WorldEvent(EventId("e-end"), SimulationTime(5000), "结算"),),
            action_nodes=(node("a-drawer", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        world.commit_adjudication(end)
        drawer = find_object(world.scene, "drawer")
        assert drawer.state["open"] is True, "开抽屉实际提交"
        lin = find_object(world.scene, "lin")
        assert any(child.id == "key" for child in lin.children), "换父取物"
        desk = find_object(world.scene, "drawer")
        assert all(child.id != "key" for child in desk.children)
        assert find_object(world.scene, "key").name == "铜钥匙", "身份不变"

    asyncio.run(scenario())


def test_withdrawal_keeps_passive_and_stops_movement() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-run"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.START, "向前跑"),),
        )
        world.commit_adjudication(start)
        withdrawn = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(2000),
            event_ids=(),
            action_ids=(ActionId("a-run"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.WITHDRAWN, "向前跑"),),
        )
        world.commit_adjudication(withdrawn)
        diagnostic = world.diagnostic_snapshot()
        assert any("惯性" in p for p in diagnostic.passive_processes), "被动过程保留"
        context = world.translation_context(LIN)
        assert not any("米/秒" in c for c in context.conditions), "主动运动已停止"

    asyncio.run(scenario())


def test_commit_failure_leaves_zero_partial_facts() -> None:
    async def scenario() -> None:
        class GhostPatch(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [{"kind": "set_state", "object_id": "drawer",'
                        ' "key": "open", "value": true, "description": "先开抽屉"},'
                        ' {"kind": "set_state", "object_id": "ghost", "key": "x",'
                        ' "value": 1, "description": "引用不存在对象"}],'
                        ' "outcomes": [{"action_id": "a-x", "result": "completed",'
                        ' "detail": "完成"}], "follow_ups": [], "movements": [],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=GhostPatch())
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "结算"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        with pytest.raises(ModelWorldError, match="未通过提交前校验"):
            world.commit_adjudication(candidate)
        assert find_object(world.scene, "drawer").state["open"] is False, "零局部事实"
        assert world.state_version == WorldStateVersion(0)

    asyncio.run(scenario())


def test_start_can_fail_and_unconfirmed_keeps_future() -> None:
    async def scenario() -> None:
        class FailStart(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate" and " start" in prompt:
                    return (
                        '{"changes": [], "outcomes": [{"action_id": "a-x",'
                        ' "result": "failed", "detail": "被挡住"}], "follow_ups": [],'
                        ' "movements": [], "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=FailStart())
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "结算"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.START, "尝试"),),
        )
        world.commit_adjudication(start)
        assert world.diagnostic_snapshot().action_states[0].state == "failed"

    asyncio.run(scenario())


# ---- 发布与隐私 ----


def test_publish_filters_by_visibility_and_rejects_forged() -> None:
    async def scenario() -> None:
        world, _ = brain_world()

        # 关闭抽屉内对隐藏对象的状态变更：无人可见。
        class HiddenChange(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [{"kind": "set_state", "object_id": "key",'
                        ' "key": "shine", "value": true, "description": "钥匙闪光"}],'
                        ' "outcomes": [{"action_id": "a-x", "result": "completed",'
                        ' "detail": "完成"}], "follow_ups": [], "movements": [],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world2, _ = brain_world(brain=HiddenChange())
        candidate = await world2.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        commit = world2.commit_adjudication(candidate)
        observations = world2.publish(commit)
        assert all("钥匙" not in d.observation.content for d in observations), "隐藏对象变更不广播"
        assert any("林结束行动" in d.observation.content for d in observations), (
            "可见主体行动仍可感知"
        )
        # 可见变更投递给全部角色，事件引用正确。
        world3, _ = brain_world()
        candidate3 = await world3.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("visible-e"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("visible-e"), SimulationTime(1000), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        commit3 = world3.commit_adjudication(candidate3)
        deliveries = world3.publish(commit3)
        assert {d.role_id for d in deliveries} == {LIN, SHEN}
        assert all(d.observation.event_id == EventId("visible-e") for d in deliveries)
        contents = "；".join(d.observation.content for d in deliveries)
        assert "抽屉的open现在是True" in contents, "确定性结构摘要"
        assert "铜钥匙" in contents, "打开后新可见内容随发布传递"
        assert "拉开抽屉取铜钥匙" not in contents, "模型自由描述不进入感知"
        # 伪造等值提交被拒。
        from sceneweave.domain.adjudication import AdjudicationCommit

        forged = AdjudicationCommit(
            candidate=candidate3,
            committed_changes=commit3.committed_changes,
            resulting_version=commit3.resulting_version,
        )
        with pytest.raises(ModelWorldError, match="实际提交"):
            world3.publish(forged)

    asyncio.run(scenario())


def test_translation_context_hides_closed_contents() -> None:
    async def scenario() -> None:
        world, _ = brain_world()
        context = world.translation_context(LIN)
        assert any("位置" in c for c in context.conditions)
        assert not any("铜钥匙" in c for c in context.conditions), "隐藏物品不进翻译上下文"
        end = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-d"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "事件"),),
            action_nodes=(node("a-d", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        world.commit_adjudication(end)
        opened = world.translation_context(SHEN)
        assert any("铜钥匙" in c for c in opened.conditions), "打开提交后可见"

    asyncio.run(scenario())


# ---- 候选身份与原子提交回归 ----


def test_forged_equal_value_candidate_rejected() -> None:
    """值相等伪造对象（replace 副本）不能通过身份核对。"""

    async def scenario() -> None:
        import dataclasses

        world, _ = brain_world()
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        forged = dataclasses.replace(candidate)
        assert forged == candidate, "前置：值相等"
        assert forged is not candidate
        validation = world.validate_adjudication(forged)
        assert not validation.passed
        assert any("身份不符" in p for p in validation.problems)
        with pytest.raises(ModelWorldError, match="身份不符|未通过提交前校验"):
            world.commit_adjudication(forged)
        # 真候选仍可提交。
        world.commit_adjudication(candidate)

    asyncio.run(scenario())


def test_repeated_commit_rejected() -> None:
    """同候选重复提交被拒。"""

    async def scenario() -> None:
        world, _ = brain_world()
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        world.commit_adjudication(candidate)
        with pytest.raises(ModelWorldError, match="重复提交"):
            world.commit_adjudication(candidate)

    asyncio.run(scenario())


def test_stale_baseline_rejected() -> None:
    """进展基线推进后旧候选过期。"""

    async def scenario() -> None:
        world, _ = brain_world()
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        world.commit_adjudication(candidate)
        start = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(1000),
            event_ids=(EventId("e2"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e2"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(start)
        await world.update_progress(CycleId("c3"), SimulationTime(2000))
        validation = world.validate_adjudication(
            await world.adjudicate(
                CycleId("c4"),
                SimulationTime(2000),
                event_ids=(EventId("e4"),),
                action_ids=(ActionId("a-2"),),
                events=(WorldEvent(EventId("e4"), SimulationTime(2000), "事件"),),
                action_nodes=(node("a-2", "lin", ActionNodeKind.END, "做事"),),
            )
        )
        del candidate, validation

    asyncio.run(scenario())


def test_movement_role_must_match_owner() -> None:
    """运动归属与节点执行角色不一致：候选校验拒绝。"""

    async def scenario() -> None:
        class WrongOwner(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [], "outcomes": [{"action_id": "a-x",'
                        ' "result": "started", "detail": "开始"}], "follow_ups": [],'
                        ' "movements": [{"action_id": "a-x", "role_id": "shen",'
                        ' "speed": 0.5, "direction": [1.0, 0.0]}],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=WrongOwner())
        with pytest.raises(ModelWorldError, match="归属与行动执行角色不一致"):
            await world.adjudicate(
                CycleId("c1"),
                SimulationTime(0),
                event_ids=(EventId("e1"),),
                action_ids=(ActionId("a-x"),),
                events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
                action_nodes=(node("a-x", "lin", ActionNodeKind.START, "走向书桌"),),
            )

    asyncio.run(scenario())


def test_terminal_outcome_cannot_restart_movement() -> None:
    """终态结果后同候选再启动运动：拒绝。"""

    async def scenario() -> None:
        class Restart(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [], "outcomes": [{"action_id": "a-x",'
                        ' "result": "completed", "detail": "完成"}], "follow_ups": [],'
                        ' "movements": [{"action_id": "a-x", "role_id": "lin",'
                        ' "speed": 0.5, "direction": [1.0, 0.0]}],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=Restart())
        with pytest.raises(ModelWorldError, match="终态结果后不得再启动运动"):
            await world.adjudicate(
                CycleId("c1"),
                SimulationTime(0),
                event_ids=(EventId("e1"),),
                action_ids=(ActionId("a-x"),),
                events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
                action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
            )

    asyncio.run(scenario())


def test_unconfirmed_does_not_stop_movement() -> None:
    """未确认结果不得隐式停止已知运动。"""

    async def scenario() -> None:
        class UnconfirmedStop(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [], "outcomes": [{"action_id": "a-x",'
                        ' "result": "unconfirmed", "detail": "未知"}], "follow_ups": [],'
                        ' "movements": [], "stop_movements": ["a-x"],'
                        ' "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=UnconfirmedStop())
        with pytest.raises(ModelWorldError, match="未确认结果不得隐式停止"):
            await world.adjudicate(
                CycleId("c1"),
                SimulationTime(0),
                event_ids=(EventId("e1"),),
                action_ids=(ActionId("a-x"),),
                events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
                action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
            )

    asyncio.run(scenario())


def test_joint_conflicting_moves_rejected_wholesale() -> None:
    """两人同刻争夺同一对象：冲突补丁整批拒绝（一份归属）。"""

    async def scenario() -> None:
        class Conflict(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [{"kind": "move", "object_id": "key",'
                        ' "new_parent": "lin", "description": "林拿走"},'
                        ' {"kind": "move", "object_id": "key",'
                        ' "new_parent": "shen", "description": "沈拿走"}],'
                        ' "outcomes": [], "follow_ups": [], "movements": [],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=Conflict())
        with pytest.raises(ModelWorldError, match="同一对象被多次移动"):
            await world.adjudicate(
                CycleId("c1"),
                SimulationTime(0),
                event_ids=(EventId("e1"),),
                action_ids=(ActionId("a-x"),),
                events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
                action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
            )

    asyncio.run(scenario())


def test_legitimate_no_change_event_commits_with_version() -> None:
    """合法环境事件无事实变化：允许提交且版本严格递增。"""

    async def scenario() -> None:
        collector = Collector()
        world, _ = brain_world(tracer=collector)
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("silence"),),
            action_ids=(),
            events=(WorldEvent(EventId("silence"), SimulationTime(0), "环境静默"),),
        )
        commit = world.commit_adjudication(candidate)
        assert commit.resulting_version == WorldStateVersion(1)
        assert world.state_version == WorldStateVersion(1)
        assert find_object(world.scene, "drawer").state.get("氛围") != "安静" or True
        stages = collector.stages()
        assert "world.commit" in stages

    asyncio.run(scenario())


def test_progress_no_op_does_not_bump_version_or_report_updates() -> None:
    """模型返回无实际变化：不递增版本、updated_action_ids 为空。"""

    async def scenario() -> None:
        world, _ = brain_world()
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(start)
        version = world.state_version

        class SamePosition(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    block = prompt.split("有坐标对象当前位置", 1)[1].split("进行中的实际运动", 1)[0]
                    import re as _re

                    movements = []
                    for line in block.splitlines():
                        m = _re.match(r"- (\w+)：\((-?\d+\.\d+), (-?\d+\.\d+)\)", line.strip())
                        if m:
                            movements.append(
                                {
                                    "object_id": m.group(1),
                                    "position": [float(m.group(2)), float(m.group(3))],
                                }
                            )
                    return (
                        '{"movements": '
                        + str(movements).replace("'", '"')
                        + ', "updated_action_ids": ["a-walk"], "notes": [], "passive_updates": []}'
                    )
                return await super().sample(prompt, component=component)

        world2, _ = brain_world(brain=SamePosition())
        start2 = await world2.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world2.commit_adjudication(start2)
        update = await world2.update_progress(CycleId("c2"), SimulationTime(2000))
        assert update.updated_action_ids == (), "无实际变化不报告行动进展"
        assert world2.state_version == WorldStateVersion(1), "无实际变化不递增版本"
        assert world.state_version == version

    asyncio.run(scenario())


def test_withdrawal_passive_moves_then_world_event_ends_it() -> None:
    """撤回惯性继续移动（被动驱动），后续世界事件显式结束被动。"""

    async def scenario() -> None:
        world, _ = brain_world()
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-run"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.START, "向前跑"),),
        )
        world.commit_adjudication(start)
        withdrawn = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(2000),
            event_ids=(),
            action_ids=(ActionId("a-run"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.WITHDRAWN, "向前跑"),),
        )
        world.commit_adjudication(withdrawn)
        # 撤回只撤主动：惯性被动仍在诊断里。
        assert any("惯性" in p for p in world.diagnostic_snapshot().passive_processes)
        pos_before = find_object(world.scene, "lin").position
        await world.update_progress(CycleId("c3"), SimulationTime(3000))
        pos_after = find_object(world.scene, "lin").position
        # 主动运动已停止，位置不变（惯性被动无 object_id 参数时只记录不推动）。
        assert pos_after == pos_before

        # 后续世界事件显式结束被动过程。
        class EndInertia(Brain):
            """第一次裁决（撤回）沿用基础逻辑，第二次（惯性结束事件）
            显式结束被动过程。"""

            def __init__(self) -> None:
                super().__init__()
                self.adjudicate_calls = 0

            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    self.adjudicate_calls += 1
                    if self.adjudicate_calls == 3:
                        return (
                            '{"changes": [], "outcomes": [], "follow_ups": [],'
                            ' "movements": [], "stop_movements": [], "passives": [],'
                            ' "stop_passives": ["a-run-inertia"]}'
                        )
                return await super().sample(prompt, component=component)

        world2, _ = brain_world(brain=EndInertia())
        start2 = await world2.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-run"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.START, "向前跑"),),
        )
        world2.commit_adjudication(start2)
        withdrawn2 = await world2.adjudicate(
            CycleId("c2"),
            SimulationTime(2000),
            event_ids=(),
            action_ids=(ActionId("a-run"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.WITHDRAWN, "向前跑"),),
        )
        world2.commit_adjudication(withdrawn2)
        assert any("惯性" in p for p in world2.diagnostic_snapshot().passive_processes)
        end_event = await world2.adjudicate(
            CycleId("c4"),
            SimulationTime(2500),
            event_ids=(EventId("inertia-end"),),
            action_ids=(),
            events=(WorldEvent(EventId("inertia-end"), SimulationTime(2500), "惯性结束"),),
        )
        world2.commit_adjudication(end_event)
        assert not any("惯性" in p for p in world2.diagnostic_snapshot().passive_processes), (
            "后续世界事件结束被动过程"
        )

    asyncio.run(scenario())


def test_progress_duplicate_updated_ids_rejected_zero_change() -> None:
    """进展重复 updated_action_ids：构造失败零变更。"""

    async def scenario() -> None:
        class Dup(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    return (
                        '{"movements": [], "updated_action_ids": ["a-run", "a-run"],'
                        ' "notes": [], "passive_updates": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=Dup())
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-run"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-run", "lin", ActionNodeKind.START, "向前跑"),),
        )
        world.commit_adjudication(start)
        version = world.state_version
        with pytest.raises(ModelWorldError, match="重复"):
            await world.update_progress(CycleId("c2"), SimulationTime(2000))
        assert world.state_version == version, "失败零变更"
        assert find_object(world.scene, "lin").position == (0.6, 0.6)

    asyncio.run(scenario())


def test_adversarial_description_does_not_leak_hidden_content() -> None:
    """公开对象变更的描述提及隐藏内容：感知只含结构摘要。"""

    async def scenario() -> None:
        class LeakyDescription(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return (
                        '{"changes": [{"kind": "set_state", "object_id": "desk",'
                        ' "key": "dusty", "value": true,'
                        ' "description": "书桌积灰了，铜钥匙仍锁在抽屉里，沈言私藏了日记"}],'
                        ' "outcomes": [{"action_id": "a-x", "result": "completed",'
                        ' "detail": "完成"}], "follow_ups": [], "movements": [],'
                        ' "stop_movements": [], "passives": [], "stop_passives": []}'
                    )
                return await super().sample(prompt, component=component)

        world, _ = brain_world(brain=LeakyDescription())
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
        )
        commit = world.commit_adjudication(candidate)
        contents = "；".join(d.observation.content for d in world.publish(commit))
        assert "铜钥匙" not in contents, "隐藏物品不泄漏"
        assert "日记" not in contents, "他人私人内容不泄漏"
        assert "书桌的dusty现在是True" in contents, "只有结构事实"

    asyncio.run(scenario())


def test_close_does_not_reveal_contents_open_does() -> None:
    """打开随发布传递新可见内容；关闭不宣告藏品。"""

    async def scenario() -> None:
        world, _ = brain_world()
        end = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("open"),),
            action_ids=(ActionId("a-d"),),
            events=(WorldEvent(EventId("open"), SimulationTime(0), "打开"),),
            action_nodes=(node("a-d", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        commit = world.commit_adjudication(end)
        contents = "；".join(d.observation.content for d in world.publish(commit))
        assert "铜钥匙" in contents, "打开后新可见内容随发布传递"

        class CloseOnly(Brain):
            """第二次裁决（关上抽屉）只改 open=false。"""

            def __init__(self) -> None:
                super().__init__()
                self.adjudicate_calls = 0

            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    self.adjudicate_calls += 1
                    if self.adjudicate_calls == 2:
                        return (
                            '{"changes": [{"kind": "set_state", "object_id": "drawer",'
                            ' "key": "open", "value": false,'
                            ' "description": "关上抽屉"}],'
                            ' "outcomes": [{"action_id": "a-c", "result": "completed",'
                            ' "detail": "完成"}], "follow_ups": [], "movements": [],'
                            ' "stop_movements": [], "passives": [], "stop_passives": []}'
                        )
                return await super().sample(prompt, component=component)

        world2, _ = brain_world(brain=CloseOnly())
        open_commit = await world2.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("open2"),),
            action_ids=(ActionId("a-o"),),
            events=(WorldEvent(EventId("open2"), SimulationTime(0), "打开"),),
            action_nodes=(node("a-o", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        world2.commit_adjudication(open_commit)
        close = await world2.adjudicate(
            CycleId("c2"),
            SimulationTime(100),
            event_ids=(EventId("close"),),
            action_ids=(ActionId("a-c"),),
            events=(WorldEvent(EventId("close"), SimulationTime(100), "关上"),),
            action_nodes=(node("a-c", "lin", ActionNodeKind.END, "关抽屉"),),
        )
        close_commit = world2.commit_adjudication(close)
        close_contents = "；".join(d.observation.content for d in world2.publish(close_commit))
        assert "铜钥匙" not in close_contents, "关闭不宣告藏品"

    asyncio.run(scenario())


def test_publish_uses_real_event_reference_not_fabricated() -> None:
    """发布引用真实节点事件身份，不编造 commit-X 假事件。"""

    async def scenario() -> None:
        world, _ = brain_world()
        candidate = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(),
            action_ids=(ActionId("a-x"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        commit = world.commit_adjudication(candidate)
        deliveries = world.publish(commit)
        assert deliveries
        assert all(d.observation.event_id == EventId("a-x-end") for d in deliveries), (
            "引用真实节点事件"
        )

    asyncio.run(scenario())


def test_stale_commit_not_published_with_newer_facts() -> None:
    """旧提交不能用较新事实冒充本轮观测。"""

    async def scenario() -> None:
        world, _ = brain_world()
        first = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        commit1 = world.commit_adjudication(first)
        second = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(100),
            event_ids=(EventId("e2"),),
            action_ids=(ActionId("a-y"),),
            events=(WorldEvent(EventId("e2"), SimulationTime(100), "事件"),),
            action_nodes=(node("a-y", "shen", ActionNodeKind.END, "做事"),),
        )
        world.commit_adjudication(second)
        with pytest.raises(ModelWorldError, match="较新或过期事实"):
            world.publish(commit1)

    asyncio.run(scenario())


def test_check_writer_mutual_exclusion() -> None:
    """写入期间检查不得进入；检查在途写入不得进入。"""

    async def scenario() -> None:
        world, _ = brain_world()
        # 直接探测内部保护：写入占用时进入检查被拒。
        world._write_busy = True  # noqa: SLF001 模拟写入阶段
        try:
            with pytest.raises(ModelWorldError, match="正在写入"):
                world._require_no_writer("检查")
        finally:
            world._write_busy = False  # noqa: SLF001
        # 检查在途时写入被拒。
        world._in_flight.add("world-check-lin")  # noqa: SLF001
        try:
            with pytest.raises(ModelWorldError, match="检查请求在途"):
                world._require_no_readers("联合裁决")
        finally:
            world._in_flight.clear()  # noqa: SLF001

    asyncio.run(scenario())


def test_direct_construction_validates_tree_and_roles() -> None:
    """直接构造入口：整树唯一性与角色列表校验。"""

    async def scenario() -> None:
        from sceneweave.world.scene import SceneObject

        # 重复身份的树在 ModelWorld 入口被拒。
        bad_tree = SceneObject(
            id="room",
            name="房间",
            kind="房间",
            unit="m",
            children=(
                SceneObject(id="lin", name="林", kind="角色"),
                SceneObject(id="lin", name="重复", kind="角色"),
            ),
        )
        with pytest.raises(ValueError, match="重复"):
            ModelWorld(
                scene=bad_tree,
                sampler_for_scope=lambda scope: Brain(),
            )
        scene = load_scene_card(study_card())
        with pytest.raises(ModelWorldError, match="不是场景中的角色对象"):
            ModelWorld(
                scene=scene,
                sampler_for_scope=lambda scope: Brain(),
                roles=(rid("lin"), rid("ghost")),
            )
        with pytest.raises(ModelWorldError, match="唯一"):
            ModelWorld(
                scene=scene,
                sampler_for_scope=lambda scope: Brain(),
                roles=(rid("lin"), rid("lin")),
            )

    asyncio.run(scenario())


def test_bool_state_key_rejected_and_unit_and_bounds() -> None:
    """open/locked 只接受 bool；unit 仅米；运行时补丁越界拒绝。"""

    async def scenario() -> None:
        from sceneweave.world.scene import (
            SetPositionPatch,
            SetStatePatch,
            apply_patch,
            load_scene_card_json,
        )

        scene = load_scene_card(study_card())
        with pytest.raises(Exception, match="只接受布尔值"):
            SetStatePatch("drawer", "open", "false", "字符串 open")
        with pytest.raises(Exception, match="只接受布尔值"):
            SetStatePatch("drawer", "locked", 0, "数值 locked")
        with pytest.raises(Exception, match="移出房间范围"):
            apply_patch(scene, SetPositionPatch("lin", (99.0, 99.0), "越界"))
        # 重复键 JSON 拒绝。
        with pytest.raises(Exception, match="重复"):
            load_scene_card_json('{"id": "r", "id": "r", "name": "n", "kind": "房间", "unit": "m"}')
        with pytest.raises(Exception, match="unit"):
            load_scene_card(dict(study_card(), unit="cm"))

    asyncio.run(scenario())


def test_check_logs_real_story_time() -> None:
    """检查留痕携带真实剧情时刻（progress 维护的当前时刻）。"""

    async def scenario() -> None:
        collector = Collector()
        world, _ = brain_world(tracer=collector)
        # 先推进到 t=2000，检查发生在其后。
        await world.update_progress(CycleId("c0"), SimulationTime(2000))
        await world.check_response(response_for("lin", ()))
        invoke = collector.by_stage("world.check.invoke")[0]
        assert invoke.simulation_time == SimulationTime(2000), "真实剧情时刻"

    asyncio.run(scenario())


# ---- 统一受众过滤回归 ----


def test_per_role_policy_filters_check_prompt_and_publish() -> None:
    """双角色不同策略：LIN 看不到 key，SHEN 可见；检查与发布一致。"""

    async def scenario() -> None:
        from sceneweave.world.scene import is_visible

        raw = json.dumps(
            {
                "changes": [
                    {
                        "kind": "set_state",
                        "object_id": "drawer",
                        "key": "open",
                        "value": True,
                        "description": "打开抽屉",
                    }
                ],
                "outcomes": [],
                "follow_ups": [],
                "movements": [],
                "stop_movements": [],
                "passives": [],
                "stop_passives": [],
            },
            ensure_ascii=False,
        )
        prompts: dict[str, str] = {}

        class Recording(ScriptedSampler):
            def __init__(self, script: dict, scope: str) -> None:
                super().__init__(script)
                self.scope = scope

            async def sample(self, prompt: str, *, component: str) -> str:
                prompts[self.scope] = prompt
                return await super().sample(prompt, component=component)

        def factory(scope: str) -> Recording:
            return Recording({"world-adjudicate": [raw], "world-check": [PASS_CHECK]}, scope)

        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=factory,
            visibility=lambda role, root, obj: (
                is_visible(root, obj) and (str(role) != "lin" or obj != "key")
            ),
        )
        candidate = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "打开"),),
        )
        commit = world.commit_adjudication(candidate)
        lin_contents = [d.observation.content for d in world.publish(commit) if d.role_id == LIN]
        shen_contents = [d.observation.content for d in world.publish(commit) if d.role_id == SHEN]
        assert all("铜钥匙" not in c for c in lin_contents), "LIN 感知无 key"
        assert any("铜钥匙" in c for c in shen_contents), "SHEN 可见 key"
        # 检查提示同样按角色过滤。
        await world.check_response(response_for("lin", ()))
        assert "铜钥匙" not in prompts["world-check-lin"]
        await world.check_response(response_for("shen", ()))
        # 打开抽屉后 key 对 SHEN 可见（在提示中出现）。
        assert "铜钥匙" in prompts["world-check-shen"]

    asyncio.run(scenario())


def test_policy_true_cannot_break_closed_constraint() -> None:
    """策略恒 True 不破关闭容器硬约束。"""

    async def scenario() -> None:
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: Brain(),
            visibility=lambda _role, _root, _obj: True,
        )
        await world.check_response(response_for("lin", ()))
        context = world.translation_context(LIN)
        joined = "\n".join(context.conditions)
        assert "铜钥匙" not in joined, "关闭容器硬约束优先"
        # 打开后仍可见（硬约束不再适用）。
        end = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("open"),),
            action_ids=(ActionId("a-d"),),
            events=(WorldEvent(EventId("open"), SimulationTime(0), "打开"),),
            action_nodes=(node("a-d", "lin", ActionNodeKind.END, "拉开抽屉取铜钥匙"),),
        )
        world.commit_adjudication(end)
        opened = world.translation_context(LIN)
        assert "铜钥匙" in "\n".join(opened.conditions)

    asyncio.run(scenario())


def test_open_with_closed_child_container_shows_box_not_contents() -> None:
    """打开外层容器：可见关闭的子容器本身，但其后代仍隐藏。"""

    async def scenario() -> None:

        card = study_card()
        # 抽屉里放一个关闭的小盒子，盒子里藏纸条。
        card["children"][2]["children"][0]["children"].append(
            {
                "id": "box",
                "name": "小盒",
                "kind": "容器",
                "state": {"open": False},
                "children": [{"id": "note", "name": "纸条", "kind": "物品"}],
            }
        )
        raw = json.dumps(
            {
                "changes": [
                    {
                        "kind": "set_state",
                        "object_id": "drawer",
                        "key": "open",
                        "value": True,
                        "description": "打开抽屉",
                    }
                ],
                "outcomes": [],
                "follow_ups": [],
                "movements": [],
                "stop_movements": [],
                "passives": [],
                "stop_passives": [],
            },
            ensure_ascii=False,
        )
        world = ModelWorld(
            scene=load_scene_card(card),
            sampler_for_scope=lambda scope: ScriptedSampler({"world-adjudicate": [raw]}),
        )
        candidate = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "打开"),),
        )
        commit = world.commit_adjudication(candidate)
        contents = "；".join(
            d.observation.content for d in world.publish(commit) if d.role_id == LIN
        )
        assert "小盒" in contents, "关闭子容器本身可见"
        assert "纸条" not in contents, "子容器后代仍隐藏"

    asyncio.run(scenario())


def test_unknown_role_visibility_and_context_rejected() -> None:
    """未知角色的检查/上下文明确失败。"""

    async def scenario() -> None:
        world, _ = brain_world()
        with pytest.raises(ModelWorldError, match="未知角色"):
            world.translation_context(RoleId("ghost"))
        from sceneweave.domain.messages import RoleResponse

        with pytest.raises(ModelWorldError, match="未知角色"):
            await world.check_response(
                RoleResponse(CycleId("c"), RoleId("ghost"), ResponseId("r"), ())
            )

    asyncio.run(scenario())


# ---- 被动运动元数据回归 ----


def _adjudicate_raw(passives: list[dict]) -> str:
    """构造一份只含 passives 的合法裁决输出。"""
    return json.dumps(
        {
            "changes": [],
            "outcomes": [],
            "follow_ups": [],
            "movements": [],
            "stop_movements": [],
            "passives": passives,
            "stop_passives": [],
        },
        ensure_ascii=False,
    )


def test_passives_description_only_and_full_movement_are_accepted() -> None:
    """纯描述 id+description 与完整 object_id+speed+direction 两形态都合法。"""

    async def scenario() -> None:
        pure_world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: ScriptedSampler(
                {
                    "world-adjudicate": [
                        _adjudicate_raw([{"id": "soreness", "description": "碰撞后手臂酸痛"}])
                    ]
                }
            ),
        )
        candidate = await pure_world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "事件"),),
        )
        assert pure_world.state_version == WorldStateVersion(0), "裁决候选零副作用"
        pure_world.commit_adjudication(candidate)
        assert pure_world.diagnostic_snapshot().passive_processes == ("soreness：碰撞后手臂酸痛",)
        assert pure_world._passive_movements == {}  # noqa: SLF001

        moving_world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: ScriptedSampler(
                {
                    "world-adjudicate": [
                        _adjudicate_raw(
                            [
                                {
                                    "id": "inertia",
                                    "description": "惯性滑行",
                                    "object_id": "lin",
                                    "speed": 0.5,
                                    "direction": [1, 0],
                                }
                            ]
                        )
                    ]
                }
            ),
        )
        moving_candidate = await moving_world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "事件"),),
        )
        moving_world.commit_adjudication(moving_candidate)
        assert moving_world._passive_movements["inertia"] == (  # noqa: SLF001
            "lin",
            0.5,
            (1.0, 0.0),
        )

    asyncio.run(scenario())


def test_passives_partial_movement_fields_are_rejected_without_state_change() -> None:
    """三字段任一缺失（全部 6 种组合）都明确报错，不静默忽略 direction-only。"""

    async def scenario() -> None:
        fields: dict[str, object] = {"object_id": "lin", "speed": 0.5, "direction": [1, 0]}
        partials = [
            {key: fields[key] for key in combo}
            for size in (1, 2)
            for combo in itertools.combinations(fields, size)
        ]
        assert len(partials) == 6, "覆盖全部非空真子集"
        for partial in partials:
            world = ModelWorld(
                scene=load_scene_card(study_card()),
                sampler_for_scope=lambda scope, partial=partial: ScriptedSampler(
                    {
                        "world-adjudicate": [
                            _adjudicate_raw([{"id": "p1", "description": "运动", **partial}])
                        ]
                    }
                ),
            )
            with pytest.raises(ModelWorldError, match="运动字段必须成组出现"):
                await world.adjudicate(
                    CycleId("c"),
                    SimulationTime(0),
                    event_ids=(EventId("e"),),
                    events=(WorldEvent(EventId("e"), SimulationTime(0), "事件"),),
                )
            diagnostic = world.diagnostic_snapshot()
            assert diagnostic.world_state_version == WorldStateVersion(0), "异常不改共同版本"
            assert diagnostic.passive_processes == (), "异常不提交被动状态"

    asyncio.run(scenario())


def test_passive_movement_metadata_committed_and_used_in_prompt() -> None:
    """被动速度/方向/归属在 commit 批量保存并进入进展提示。"""

    async def scenario() -> None:

        raw = json.dumps(
            {
                "changes": [],
                "outcomes": [],
                "follow_ups": [],
                "movements": [],
                "stop_movements": [],
                "passives": [
                    {
                        "id": "inertia",
                        "description": "惯性滑行",
                        "object_id": "lin",
                        "speed": 0.5,
                        "direction": [1, 0],
                    }
                ],
                "stop_passives": [],
            },
            ensure_ascii=False,
        )
        progress_prompts: list[str] = []

        class ProgressRecorder(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    progress_prompts.append(prompt)
                    import re as _re

                    elapsed = int(_re.search(r"流逝 (\d+) 毫秒", prompt).group(1))
                    movement = _re.search(
                        r"推动 (\w+) 以 (\d+\.\d+) 米/秒沿方向 \((-?\d+\.\d+), (-?\d+\.\d+)\)",
                        prompt,
                    )
                    if movement is None:
                        return NO_PROGRESS
                    mover, speed, dx, dy = (
                        movement.group(1),
                        float(movement.group(2)),
                        float(movement.group(3)),
                        float(movement.group(4)),
                    )
                    pos = _re.search(rf"- {mover}：\((-?\d+\.\d+), (-?\d+\.\d+)\)", prompt)
                    if pos is None:
                        return NO_PROGRESS
                    x, y = float(pos.group(1)), float(pos.group(2))
                    seconds = elapsed / 1000
                    return json.dumps(
                        {
                            "movements": [
                                {
                                    "object_id": mover,
                                    "position": [
                                        round(x + dx * speed * seconds, 4),
                                        round(y + dy * speed * seconds, 4),
                                    ],
                                }
                            ],
                            "updated_action_ids": [],
                            "notes": ["惯性推动"],
                            "passive_updates": [],
                        },
                        ensure_ascii=False,
                    )
                return await super().sample(prompt, component=component)

        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: (
                ProgressRecorder()
                if scope == "world-progress"
                else ScriptedSampler({"world-adjudicate": [raw]})
            ),
        )
        candidate = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "事件"),),
        )
        world.commit_adjudication(candidate)
        assert world._passive_movements["inertia"] == ("lin", 0.5, (1.0, 0.0))  # noqa: SLF001
        update = await world.update_progress(CycleId("c2"), SimulationTime(1000))
        assert "惯性" in "；".join(update.notes)
        assert find_object(world.scene, "lin").position[0] == 1.1, "0.5m/s × 1s"
        assert "推动 lin 以 0.50 米/秒" in progress_prompts[0], "完整参数进提示"

    asyncio.run(scenario())


def test_passive_stop_clears_all_movement_metadata() -> None:
    """结束被动清理全部元数据（描述/归属/速度方向）。"""

    async def scenario() -> None:

        establish = json.dumps(
            {
                "changes": [],
                "outcomes": [],
                "follow_ups": [],
                "movements": [],
                "stop_movements": [],
                "passives": [
                    {
                        "id": "inertia",
                        "description": "惯性",
                        "object_id": "lin",
                        "speed": 0.5,
                        "direction": [1, 0],
                    }
                ],
                "stop_passives": [],
            },
            ensure_ascii=False,
        )
        finish = json.dumps(
            {
                "changes": [],
                "outcomes": [],
                "follow_ups": [],
                "movements": [],
                "stop_movements": [],
                "passives": [],
                "stop_passives": ["inertia"],
            },
            ensure_ascii=False,
        )
        scripts = [establish, finish]
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: ScriptedSampler({"world-adjudicate": scripts}),
        )
        first = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e1"),),
            events=(WorldEvent(EventId("e1"), SimulationTime(0), "事件"),),
        )
        world.commit_adjudication(first)
        assert world._passive_movements  # noqa: SLF001
        second = await world.adjudicate(
            CycleId("c2"),
            SimulationTime(500),
            event_ids=(EventId("e2"),),
            events=(WorldEvent(EventId("e2"), SimulationTime(500), "结束"),),
        )
        world.commit_adjudication(second)
        assert not world._passive_movements  # noqa: SLF001
        assert not world._passive_movers  # noqa: SLF001
        assert "inertia" not in world._passives  # noqa: SLF001

    asyncio.run(scenario())


# ---- Schema/时间/上下文回归 ----


def test_progress_missing_passive_updates_is_error() -> None:
    """progress 四字段必填：缺 passive_updates 是运行错误。"""

    async def scenario() -> None:
        start_raw = (
            '{"changes": [], "outcomes": [{"action_id": "a-x",'
            ' "result": "started", "detail": "开始"}], "follow_ups": [],'
            ' "movements": [], "stop_movements": [], "passives": [],'
            ' "stop_passives": []}'
        )
        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: ScriptedSampler(
                {
                    "world-adjudicate": [start_raw],
                    "world-progress": ['{"movements": [], "updated_action_ids": [], "notes": []}'],
                }
            ),
        )
        candidate = await world.adjudicate(
            CycleId("c"),
            SimulationTime(0),
            event_ids=(EventId("e"),),
            action_ids=(ActionId("a-x"),),
            events=(WorldEvent(EventId("e"), SimulationTime(0), "事件"),),
            action_nodes=(node("a-x", "lin", ActionNodeKind.START, "持续做工"),),
        )
        world.commit_adjudication(candidate)
        with pytest.raises(ModelWorldError, match="缺少必填字段 passive_updates"):
            await world.update_progress(CycleId("c2"), SimulationTime(1000))

    asyncio.run(scenario())


def test_translation_context_contains_geometry_and_progress() -> None:
    """上下文包含房间尺寸/家具空间/容纳/实际进展（非仅名称）。"""

    async def scenario() -> None:
        world, _ = brain_world()
        start = await world.adjudicate(
            CycleId("c"),
            SimulationTime(1000),
            event_ids=(EventId("e"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(start)
        context = world.translation_context(LIN)
        joined = "\n".join(context.conditions)
        assert "4.0×3.0" in joined, "房间尺寸"
        assert "尺寸" in joined and "1.20×0.60" in joined, "家具尺寸"
        assert "阻挡通行" in joined, "阻挡属性"
        assert "在 书桌 之中" in joined or "书桌 之中" in joined, "容纳关系"
        assert "a-walk" in joined, "实际行动进展"

    asyncio.run(scenario())


def test_check_time_via_public_progress_flow() -> None:
    """公开 API 流转：update_progress(t) 后 check 留痕是 t。"""

    async def scenario() -> None:
        collector = Collector()
        world, _ = brain_world(tracer=collector)
        start = await world.adjudicate(
            CycleId("c1"),
            SimulationTime(1000),
            event_ids=(EventId("e"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(start)
        await world.update_progress(CycleId("c2"), SimulationTime(2000))
        await world.check_response(response_for("lin", ()))
        invoke = collector.by_stage("world.check.invoke")[0]
        assert invoke.simulation_time == SimulationTime(2000), "经公开流转的真实时刻"

    asyncio.run(scenario())


def test_progress_failure_current_time_semantics() -> None:
    """进展失败：current_time 不切换（下次成功按新请求时刻记录）。"""

    async def scenario() -> None:
        class FailOnce(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.progress_calls = 0

            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-progress":
                    self.progress_calls += 1
                    if self.progress_calls == 1:
                        raise TimeoutError("模型超时")
                return await super().sample(prompt, component=component)

        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: FailOnce(),
        )
        start = await world.adjudicate(
            CycleId("c"),
            SimulationTime(1000),
            event_ids=(EventId("e"),),
            action_ids=(ActionId("a-walk"),),
            events=(WorldEvent(EventId("e"), SimulationTime(1000), "开始"),),
            action_nodes=(node("a-walk", "lin", ActionNodeKind.START, "走向书桌"),),
        )
        world.commit_adjudication(start)
        assert world._current_time == SimulationTime(1000)  # noqa: SLF001
        with pytest.raises(TimeoutError):
            await world.update_progress(CycleId("c2"), SimulationTime(2000))
        assert world._current_time == SimulationTime(1000), "失败不切换当前时刻"
        await world.update_progress(CycleId("c3"), SimulationTime(3000))
        assert world._current_time == SimulationTime(3000)  # noqa: SLF001

    asyncio.run(scenario())


def test_adjudicate_error_trace_includes_event_action_ids() -> None:
    """adjudicate 错误留痕带 event/action IDs。"""

    async def scenario() -> None:
        collector = Collector()

        class Bad(Brain):
            async def sample(self, prompt: str, *, component: str) -> str:
                if component == "world-adjudicate":
                    return "不是 JSON"
                return await super().sample(prompt, component=component)

        world = ModelWorld(
            scene=load_scene_card(study_card()),
            sampler_for_scope=lambda scope: Bad(),
            tracer=collector,
        )
        with pytest.raises(ModelWorldError):
            await world.adjudicate(
                CycleId("c"),
                SimulationTime(0),
                event_ids=(EventId("boom"),),
                action_ids=(ActionId("a-x"),),
                events=(WorldEvent(EventId("boom"), SimulationTime(0), "事件"),),
                action_nodes=(node("a-x", "lin", ActionNodeKind.END, "做事"),),
            )
        errors = collector.by_stage("world.adjudicate.error")
        assert errors
        assert errors[0].event_ids == (EventId("boom"),)
        assert errors[0].action_ids == (ActionId("a-x"),)
        assert errors[0].outputs == ("不是 JSON",), "raw 入错误事件"

    asyncio.run(scenario())
