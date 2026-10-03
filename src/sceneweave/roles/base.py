"""调度器能够调用的角色交互契约。"""

from abc import ABC, abstractmethod

from sceneweave.domain.messages import Rejection, RoleId, RoleResponse, TurnInput


class Role(ABC):
    """角色交互契约；System1/2/3 的编排完全在角色实现内部。

    引擎只通过 perceive 与 rejected 驱动角色：先接收本轮输入，
    检查不通过时接收打回反馈并返回新的完整声明。两个方法只更新
    角色私有状态，不写入 World、调度队列或剧情时钟。
    """

    @property
    @abstractmethod
    def role_id(self) -> RoleId:
        """角色的稳定身份，与显示名称分离。"""

    @abstractmethod
    async def perceive(self, turn: TurnInput) -> RoleResponse:
        """接收本轮输入，返回从现在起要主动实施或维持的完整行为集合。

        即使没有新决定，也必须返回要继续主动维持的集合；合法空集合
        表示撤回全部主动维持。字段缺失、解析失败或运行错误不得被
        转换为空集合。
        """

    @abstractmethod
    async def rejected(self, feedback: Rejection) -> RoleResponse:
        """接收 World 的打回反馈，返回新的完整声明替代旧候选。

        反馈只针对当前周期、当前角色、当前待检查的响应；没有新事件
        落地，不重复投递原始感知。外部不直接调用 System3，也不指定
        角色内部阶段。
        """
