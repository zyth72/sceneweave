"""同一事件的角色响应屏障，不负责事件排序或世界结算。"""

import asyncio
from collections.abc import Iterable

from sceneweave.domain.messages import ActionId, ActionProposal, Delivery, Observation, RoleId
from sceneweave.roles.base import Role


class ResponseRunner:
    """并行收集本轮响应，全部完成并通过校验后才交回行动提议。

    返回顺序按角色身份固定，仅用于稳定呈现，不是同刻事件裁决顺序。
    单个实例不允许同时执行两轮，防止同一角色状态被并发写入。
    """

    def __init__(self, roles: Iterable[Role]) -> None:
        self._roles: dict[RoleId, Role] = {}
        for role in roles:
            if role.role_id in self._roles:
                raise ValueError(f"重复角色身份：{role.role_id}")
            self._roles[role.role_id] = role
        self._running = False

    async def collect(self, deliveries: tuple[Delivery, ...]) -> tuple[ActionProposal, ...]:
        """收集同一事件的响应；错误传播给调用者，不转换为剧情结果。

        未知角色、跨事件混合输入和非法响应抛出 ValueError。
        响应失败时不会返回部分行动，但已经更新的角色内部状态不会自动回滚。
        调用者须处理运行中止或恢复，不能盲目重放本轮。
        """
        if self._running:
            raise RuntimeError("上一轮角色响应尚未结束")
        grouped = self._group(deliveries)
        self._running = True
        try:
            async with asyncio.TaskGroup() as group:
                tasks = {
                    role_id: group.create_task(self._roles[role_id].respond(tuple(observations)))
                    for role_id, observations in grouped.items()
                }
            actions: list[ActionProposal] = []
            action_ids: set[ActionId] = set()
            for role_id in sorted(tasks):
                for action in tasks[role_id].result().actions:
                    if action.role_id != role_id:
                        raise ValueError("角色不能替其他角色提交行动")
                    if action.action_id in action_ids:
                        raise ValueError(f"本轮存在重复行动身份：{action.action_id}")
                    action_ids.add(action.action_id)
                    actions.append(action)
            return tuple(actions)
        finally:
            self._running = False

    def _group(self, deliveries: tuple[Delivery, ...]) -> dict[RoleId, list[Observation]]:
        grouped: dict[RoleId, list[Observation]] = {}
        event_keys = {
            (delivery.observation.event_id, delivery.observation.simulation_time)
            for delivery in deliveries
        }
        if len(event_keys) > 1:
            raise ValueError("一轮响应只能处理同一剧情时刻的同一个事件")
        for delivery in deliveries:
            if delivery.role_id not in self._roles:
                raise ValueError(f"未知角色：{delivery.role_id}")
            grouped.setdefault(delivery.role_id, []).append(delivery.observation)
        return grouped
