"""脚本假模型：按组件名与提示内容决定回答。

从 examples/model_world_demo 的脚本模型提取为可复用源码；不调用真实
模型、不读取配置。世界检查始终通过、联合裁决按行动节点类型给出结构
化结果（含移动/被动/结算进展的完整参数）、角色与翻译器从 JSON 队列
按序消费。可通过 scope 覆盖个别组件脚本。
"""

import json
import re
from collections.abc import Mapping
from typing import Any

PASS_CHECK = '{"passed": true, "reason": "", "action_ids": []}'
NO_PROGRESS = '{"movements": [], "updated_action_ids": [], "notes": [], "passive_updates": []}'


def empty_adjudicate(
    *,
    changes: list[dict[str, object]] | None = None,
    outcomes: list[dict[str, object]] | None = None,
    follow_ups: list[dict[str, object]] | None = None,
    movements: list[dict[str, object]] | None = None,
    stop_movements: list[str] | None = None,
    passives: list[dict[str, object]] | None = None,
    stop_passives: list[str] | None = None,
) -> str:
    """构造一份联合裁决 JSON（空字段给空数组）。"""
    return json.dumps(
        {
            "changes": changes or [],
            "outcomes": outcomes or [],
            "follow_ups": follow_ups or [],
            "movements": movements or [],
            "stop_movements": stop_movements or [],
            "passives": passives or [],
            "stop_passives": stop_passives or [],
        },
        ensure_ascii=False,
    )


class ScriptedModel:
    """按组件名分发回答的假采样器。

    世界检查/裁决由内置语义逻辑生成（确定性，适合装配演示与集成测试）；
    其他组件（system1/2/3、translator、world-progress）按 script 队列
    依次消费；队列耗尽或缺脚本即错误（不 fallback）。
    """

    def __init__(self, script: Mapping[str, list[str]] | None = None) -> None:
        self._script = {name: list(items) for name, items in (script or {}).items()}

    async def sample(self, prompt: str, *, component: str) -> str:
        # 显式脚本优先；空或耗尽队列不能被内置演示逻辑掩盖。
        if component in self._script:
            responses = self._script[component]
            if not responses:
                raise RuntimeError(f"组件 {component} 的脚本已耗尽或未提供")
            return responses.pop(0)
        if component == "world-check":
            return PASS_CHECK
        if component == "world-adjudicate":
            return self._adjudicate(prompt)
        if component == "world-progress":
            return self._progress(prompt)
        raise RuntimeError(f"组件 {component} 的脚本已耗尽或未提供")

    def _progress(self, prompt: str) -> str:
        elapsed_match = re.search(r"流逝 (\d+) 毫秒", prompt)
        if elapsed_match is None or "速度" not in prompt:
            return NO_PROGRESS
        elapsed_ms = int(elapsed_match.group(1))
        block = prompt.split("有坐标对象当前位置", 1)
        positions: dict[str, tuple[float, float]] = {}
        if len(block) > 1:
            for line in block[1].split("进行中的实际运动", 1)[0].splitlines():
                match = re.match(r"- (\w+)：\((-?\d+\.\d+), (-?\d+\.\d+)\)", line.strip())
                if match is not None:
                    positions[match.group(1)] = (float(match.group(2)), float(match.group(3)))
        movements = []
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

    def _adjudicate(self, prompt: str) -> str:
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
                    {
                        "action_id": action,
                        "role_id": role,
                        "speed": 0.5,
                        "direction": [1.0, 0.0],
                    }
                )
            elif node_kind == "checkpoint":
                outcomes.append(
                    {"action_id": action, "result": "in_progress", "detail": "过程节点"}
                )
            elif node_kind == "end":
                outcomes.append({"action_id": action, "result": "completed", "detail": "完成"})
                stop.append(action)
            else:  # withdrawn
                stop.append(action)
                passives.append({"id": f"{action}-inertia", "description": "惯性继续"})
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
                    {"action_id": action, "role_id": role, "speed": speed, "direction": [1.0, 0.0]}
                )
                outcomes.append(
                    {"action_id": action, "result": "in_progress", "detail": "修改后的实际进展"}
                )
        if not outcomes and not movements and not passives and not stop:
            # 无参与节点：合法无变化（有事件即可提交）。
            return empty_adjudicate()
        return empty_adjudicate(
            outcomes=outcomes,
            movements=movements,
            stop_movements=stop,
            passives=passives,
        )
