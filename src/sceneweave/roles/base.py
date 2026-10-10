"""调度器能够调用的角色交互契约。"""

from abc import ABC, abstractmethod

from sceneweave.domain.messages import Rejection, RoleId, RoleResponse, TurnInput
from sceneweave.domain.translation import ExecutionSubmission, PlanAcceptance


class Role(ABC):
    """角色交互契约；System1/2/3 的编排完全在角色实现内部。

    引擎只通过 perceive/rejected 驱动角色思考，并在整链检查通过后调用
    ``save_plan`` 让角色内部保存完整链与全部排程，再通过
    ``execution_submission`` 取得最近步骤执行提交——保存与执行提交是
    两个独立边界，不用一个返回值同时表达两种用途。所有方法只更新角色
    私有状态，不写入 World、调度队列或剧情时钟。
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
        转换为空集合。复用已保存的完整链与排程时把 ``reused`` 置真。
        """

    @abstractmethod
    async def rejected(self, feedback: Rejection) -> RoleResponse:
        """接收 World 的打回反馈，返回新的完整声明替代旧候选。

        反馈只针对当前周期、当前角色、当前待检查的响应；没有新事件
        落地，不重复投递原始感知。外部不直接调用 System3，也不指定
        角色内部阶段。
        """

    async def save_plan(self, acceptance: PlanAcceptance) -> None:
        """整链检查通过后把完整链与全部排程保存到角色私有缓存。

        默认实现表示角色没有需要缓存的内部计划（低层测试角色），仅
        接受保存消息；生产角色实现须原子保存并关联候选/检查/翻译身份。
        保存本身不登记执行事件、不产生世界事实。
        """
        return None

    def execution_submission(self) -> ExecutionSubmission | None:
        """返回本周期最近步骤执行提交。

        默认 None 表示角色没有内部完整链，引擎直接按本轮声明登记；
        生产角色返回仅含最近步骤与新增排程的提交，未来步骤留在私有缓存。
        """
        return None
