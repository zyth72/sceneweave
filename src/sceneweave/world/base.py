"""World 的职责接口，具体事实模型和裁决规则由场景实现。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sceneweave.domain.adjudication import (
    ActionModificationNode,
    AdjudicationActionNode,
    AdjudicationCandidate,
    AdjudicationCommit,
    AdjudicationValidation,
    ObjectiveContext,
    ProgressUpdate,
    WorldDiagnostic,
)
from sceneweave.domain.identity import ActionId, CycleId, EventId, RoleId, WorldStateVersion
from sceneweave.domain.messages import ActionCheckResult, Delivery, RoleResponse, WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionArrangement


class World(ABC):
    """客观事实的唯一写入入口，不参与角色内部决策。

    裁决候选、运行时校验、实际提交与发布是分开的边界：未经校验提交
    的候选不是客观事实；提交失败不得留下半提交事实或发布成功结果；
    只有已提交的结果才能发布为角色感知。检查与裁决中不递归调用角色，
    裁决本身不得改变共同状态（进展更新是独立入口）。

    进展更新是 World 自身的原子边界：update_progress 因位置或被动过程
    发生的实际变化必须原子应用并递增共同状态版本；notes 只是说明文本，
    不代表事实。所有写入路径（进展更新与事实提交）都递增版本，只读
    入口（检查、上下文、诊断）不得改变状态。
    """

    @property
    @abstractmethod
    def state_version(self) -> WorldStateVersion:
        """当前共同客观状态的版本；每次实际状态变化后递增，只增不改。"""

    @abstractmethod
    async def update_progress(
        self, cycle_id: CycleId, simulation_time: SimulationTime
    ) -> ProgressUpdate:
        """时钟进入本事件时刻后，按已流逝剧情时间更新未完成行动与被动过程。

        进展只由流逝的剧情时间与本方法内的语义判断决定；模型调用与
        现实等待期间引擎不会再次调用，也不会自行推进位置或进度。
        实际变化必须原子应用并递增共同状态版本；无变化时不递增。
        其后的裁决基于更新后的状态与版本。
        """

    @abstractmethod
    def translation_context(self, role_id: RoleId) -> ObjectiveContext:
        """为翻译该角色的请求提供只读客观上下文。

        携带当前共同状态版本与 World 认为必要的位置、姿态、障碍及
        实际进展等条件描述；只读、不改变状态，也不进入角色感知。
        """

    @abstractmethod
    def diagnostic_snapshot(self) -> WorldDiagnostic:
        """对外只读诊断快照：实际行动状态与被动过程。

        区别于已提交事实的增量文本；供界面与调试呈现客观现状，不得
        改变状态，也不可作为新的世界事实来源。
        """

    @abstractmethod
    async def check_response(
        self, response: RoleResponse, *, arrangements: Sequence[ActionArrangement] = ()
    ) -> ActionCheckResult:
        """对完整主动行为声明的无副作用检查。

        基于事件落地后的共同状态检查整个集合（含义、当前步骤边界、
        未确认结果依赖、客观条件及与已有行动的关系），不只检查新增
        项；只产生结果：不调用角色、不提交事实、不取消或预登记行动。
        结果记录所依据的世界状态版本。语义判断可由模型承担，运行时
        校验负责结构与一致性。

        arrangements 是该角色由引擎提供的只读既有安排快照（可为空
        元组，兼容独立新声明入口）：只供检查维持/修改与既有安排的
        关系，World 不得写入或缓存为已登记事实，也不得把登记当作
        实际开始；登记等待中的行动仍可被主动维持。
        """

    @abstractmethod
    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
        events: Sequence[WorldEvent] = (),
        action_nodes: Sequence[AdjudicationActionNode] = (),
        modifications: Sequence[ActionModificationNode] = (),
    ) -> AdjudicationCandidate:
        """对同刻事件与行动生成联合裁决候选；不提交任何事实、不改变状态。

        候选基于本时刻事件提交前、World 已更新相关未完成行动进展后
        的共同状态，同时考虑相互作用，不因队列插入顺序或角色编号
        制造优先级。events 与 action_nodes 携带完整事件内容与行动
        节点的事件身份、执行角色、类型和登记安排；action_nodes 的
        node 为 WITHDRAWN 时表示撤回主动维持后的客观效果裁决；
        modifications 携带统一登记中的明确修改（归属、变化与新旧
        未来安排），由 World 裁决实际作用后与登记统一生效。
        """

    @abstractmethod
    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        """运行时校验候选的结构与状态一致性；不改变状态。

        校验基于裁决前的共同状态版本；不通过走明确失败，不拆分
        候选、不部分接受。
        """

    @abstractmethod
    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        """实际提交通过校验的候选；未通过校验的候选必须拒绝。

        提交是原子边界：失败不得留下半提交事实，成功后共同状态版本
        严格递增且等于提交结果报告的版本。实际状态变化记录在提交
        结果中，与候选分开。
        """

    @abstractmethod
    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        """从已提交的裁决生成按角色感知范围过滤的投递。

        只能发布已提交的内容；不得发布未提交候选或与提交后事实
        矛盾的结果文本。观察内容须引用本轮触发的事件并使用当前
        剧情时刻，由引擎在构造角色输入前核对。
        """
