"""调度器能够调用的角色接口。"""

from abc import ABC, abstractmethod

from sceneweave.domain.messages import Observation, RoleId, RoleResponse


class Role(ABC):
    """内部实现负责 System1/2/3 的编排，引擎只调用 respond。"""

    @property
    @abstractmethod
    def role_id(self) -> RoleId:
        """角色的稳定身份，与显示名称分离。"""

    @abstractmethod
    async def respond(self, observations: tuple[Observation, ...]) -> RoleResponse:
        """根据本轮能够感知的信息响应，允许不提交新行动。

        响应只更新自身内部状态，不直接写入 World 或调度队列。
        """
