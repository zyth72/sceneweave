"""World 的职责接口，具体事实模型和裁决规则由场景实现。"""

from abc import ABC, abstractmethod

from sceneweave.domain.messages import ActionProposal, Delivery, SimulationTime, WorldEvent


class World(ABC):
    """客观事实的唯一写入入口，不参与角色内部决策。"""

    @abstractmethod
    def apply_event(self, event: WorldEvent) -> tuple[Delivery, ...]:
        """验证并提交事件对应的事实，再返回经过感知过滤的投递。

        发生错误时不得发布成功结果或留下半提交状态。
        """

    @abstractmethod
    def settle_action(
        self, action: ActionProposal, simulation_time: SimulationTime
    ) -> tuple[Delivery, ...]:
        """在调度器指定的结算时刻裁决并提交实际结果，再生成反馈。

        此入口只用于已登记、到达结算时刻的行动，不接受预测作为事实。
        """
