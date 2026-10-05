"""模型 World 的离线生产装配演示（脚本假模型，无真实 API）。

演示完整生产链路：场景卡（内联同型数据，不依赖未提交场景文件）加载
为不可变事实模型；ModelWorld 驱动检查/进展/联合裁决/发布；ModelRole
与 ModelTranslator 经注入采样器工作；调度器串起整个核心循环；结构化
日志写入 SimulationLog 并导出 JSON。所有脚本采样器都可替换为
integrations/concordia/sampling.LanguageModelSampler 的真实模型桥
（接口一致）。感知物理是最小可见性策略（关闭容器隐藏后代），不是
精确视觉模拟。

运行：

    uv run --locked python examples/model_world_demo.py
"""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from concordia.utils import structured_logging

from sceneweave.domain.identity import EventId, RoleId
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TurnTracer
from sceneweave.engine.config import RunConfig
from sceneweave.engine.scheduler import Scheduler
from sceneweave.integrations.concordia.tracing import SimulationLogTracer
from sceneweave.roles.model_role import ModelRole
from sceneweave.translator.model_translator import ModelTranslator
from sceneweave.world.model_world import ModelWorld
from sceneweave.world.scene import find_object, load_scene_card

SCENE_CARD: Mapping[str, Any] = {
    "id": "room",
    "name": "书房",
    "kind": "房间",
    "unit": "m",
    "size": [4, 3],
    "description": "一间小书房。",
    "children": [
        {
            "id": "lin",
            "name": "林舟",
            "kind": "角色",
            "position": [0.6, 0.6],
            "state": {"posture": "站立"},
            "children": [],
        },
        {
            "id": "shen",
            "name": "沈言",
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
                }
            ],
        },
    ],
}


class ScriptedModel:
    """按组件名与提示内容决定回答的脚本假模型。"""

    def __init__(self, script: Mapping[str, list[str]]) -> None:
        self._script = {name: list(items) for name, items in script.items()}

    async def sample(self, prompt: str, *, component: str) -> str:
        if component == "world-check":
            return '{"passed": true, "reason": "", "action_ids": []}'
        if component == "world-adjudicate":
            result = self._adjudicate(prompt)
            if '"outcomes": []' in result and '"movements": []' in result:
                # 合法无变化：有参与事件即可提交（版本严格递增）。
                return result.replace(
                    '"changes": []',
                    '"changes": [{"kind": "set_state", "object_id": "room",'
                    ' "key": "氛围", "value": "安静", "description": "书房里很安静"}]',
                )
            return result
        responses = self._script.get(component)
        if responses is None or not responses:
            raise RuntimeError(f"组件 {component} 的脚本已耗尽")
        return responses.pop(0)

    def _adjudicate(self, prompt: str) -> str:
        import re

        outcomes: list[dict[str, Any]] = []
        movements: list[dict[str, Any]] = []
        stop: list[str] = []
        passives: list[dict[str, Any]] = []
        section = prompt.split("行动节点：", 1)
        lines = (
            [line.strip() for line in section[1].split("明确修改：", 1)[0].splitlines()]
            if len(section) > 1
            else []
        )
        for line in lines:
            match = re.match(r"- \S+：(\S+) 的 (\S+) (start|checkpoint|end|withdrawn)", line)
            if match is None:
                continue
            role, action, node_kind = match.group(1), match.group(2), match.group(3)
            if node_kind == "start":
                outcomes.append({"action_id": action, "result": "started", "detail": "开始"})
                movements.append(
                    {"action_id": action, "role_id": role, "speed": 0.5, "direction": [1.0, 0.0]}
                )
            elif node_kind == "checkpoint":
                outcomes.append(
                    {"action_id": action, "result": "in_progress", "detail": "过程节点"}
                )
            elif node_kind == "end":
                outcomes.append({"action_id": action, "result": "completed", "detail": "完成"})
                stop.append(action)
            else:
                stop.append(action)
                passives.append({"id": f"{action}-inertia", "description": "惯性继续"})
        return json.dumps(
            {
                "changes": [],
                "outcomes": outcomes,
                "follow_ups": [],
                "movements": movements,
                "stop_movements": stop,
                "passives": passives,
                "stop_passives": [],
            },
            ensure_ascii=False,
        )


class WorldProgressBrain:
    """按流逝毫秒与运动参数计算位置的进展假模型。"""

    def __init__(self) -> None:
        pass

    async def sample(self, prompt: str, *, component: str) -> str:
        import re

        elapsed = re.search(r"流逝 (\d+) 毫秒", prompt)
        movements = []
        if elapsed:
            elapsed_ms = int(elapsed.group(1))
            block = prompt.split("有坐标对象当前位置", 1)
            positions: dict[str, tuple[float, float]] = {}
            if len(block) > 1:
                for line in block[1].split("进行中的实际运动", 1)[0].splitlines():
                    match = re.match(r"- (\w+)：\((-?\d+\.\d+), (-?\d+\.\d+)\)", line.strip())
                    if match is not None:
                        positions[match.group(1)] = (
                            float(match.group(2)),
                            float(match.group(3)),
                        )
            for role, speed_text, dx_text, dy_text in re.findall(
                r"- \S+（(\w+)）：速度 (\d+\.\d+) 米/秒，方向 \((-?\d+\.\d+), (-?\d+\.\d+)\)",
                prompt,
            ):
                position = positions.get(role)
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
        return json.dumps(
            {
                "movements": movements,
                "updated_action_ids": [],
                "notes": ["按流逝时间推进"],
                "passive_updates": [],
            },
            ensure_ascii=False,
        )


class CompositeTracer(TurnTracer):
    """结构化日志 + 关键边界打印。"""

    INTERESTING = {
        "world.adjudicate.candidate",
        "world.progress.applied",
        "world.check.parsed",
        "registration.commit",
        "engine.stop",
    }

    def __init__(self, log: structured_logging.SimulationLog) -> None:
        self._log_tracer = SimulationLogTracer(log)

    def trace(self, record: TraceRecord) -> None:
        self._log_tracer.trace(record)
        if record.stage in self.INTERESTING:
            head = f"t={record.simulation_time:>6}ms [{record.stage}] {record.summary}"
            if record.outputs:
                head += "：" + "；".join(record.outputs[:2])
            print(head)


async def main() -> None:
    log = structured_logging.SimulationLog()
    tracer = CompositeTracer(log)
    scene = load_scene_card(SCENE_CARD)
    role_model = ScriptedModel(
        {
            "system1": ['{"reconsider": true, "note": "门被推开了"}']
            + ['{"reconsider": false, "note": "照旧"}'] * 10,
            "system2": ['{"intention": "走到书桌旁看看", "abandon": []}'],
            "system3": [
                '{"feasible": true, "maintain": [], "modify": [], "new_actions": ["走向书桌"]}'
            ],
        }
    )
    shen_model = ScriptedModel({"system1": ['{"reconsider": false, "note": "照旧"}'] * 11})
    translator_model = ScriptedModel(
        {
            "translator": [
                '{"schedules": [{"action_id": "a-lin-1", "commit_group_id": "g-lin-1",'
                ' "start": 1000, "expected_end": 4000, "checkpoints": [2500]}],'
                ' "modified_schedules": []}'
            ]
        }
    )

    def world_brain_for(scope: str) -> object:
        """按作用域提供 World 脚本模型（生产可替换为真实模型桥）。"""
        if scope == "world-progress":
            return WorldProgressBrain()
        return ScriptedModel({})

    world = ModelWorld(
        scene=scene,
        sampler_for_scope=world_brain_for,
        roles=(RoleId("lin"), RoleId("shen")),
        tracer=tracer,
    )
    scheduler = Scheduler(
        roles=[
            ModelRole(RoleId("lin"), role_model, tracer=tracer),
            ModelRole(RoleId("shen"), shen_model, tracer=tracer),
        ],
        world=world,
        translator=ModelTranslator.shared(translator_model, tracer=tracer),
        config=RunConfig.from_seconds(
            introspection_interval_seconds=5, max_simulation_duration_seconds=8
        ),
        initial_events=[WorldEvent(EventId("startup"), SimulationTime(0), "书房里很安静")],
        tracer=tracer,
    )
    result = await scheduler.run()
    print()
    lin = find_object(world.scene, "lin")
    print(
        f"运行结束：{result.reason.value}，完成周期 {result.completed_cycles}，"
        f"最终剧情时刻 {result.final_time}ms"
    )
    print(f"林舟最终位置：({lin.position[0]:.2f}, {lin.position[1]:.2f}) 米（按剧情时间推进）")
    print(f"共同世界状态版本：{world.state_version}")
    log_path = Path("model_world_demo_log.json")
    log_path.write_text(log.to_json(), encoding="utf-8")
    print(f"结构化日志已导出：{log_path}")
    if result.error:
        print(f"运行错误：{result.error}")


if __name__ == "__main__":
    asyncio.run(main())
