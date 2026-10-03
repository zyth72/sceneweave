"""World 的职责接口，具体事实模型和裁决规则由场景实现。"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sceneweave.domain.adjudication import (
    AdjudicationCandidate,
    AdjudicationCommit,
    AdjudicationValidation,
)
from sceneweave.domain.identity import ActionId, CycleId, EventId, WorldStateVersion
from sceneweave.domain.messages import ActionCheckResult, Delivery, RoleResponse
from sceneweave.domain.time import SimulationTime


class World(ABC):
    """客观事实的唯一写入入口，不参与角色内部决策。

    裁决候选、运行时校验、实际提交与发布是分开的边界：未经校验提交
    的候选不是客观事实；提交失败不得留下半提交事实或发布成功结果；
    只有已提交的结果才能发布为角色感知。检查与裁决中不递归调用角色。
    """

    @property
    @abstractmethod
    def state_version(self) -> WorldStateVersion:
        """当前共同客观状态的版本；事件落地后、本轮检查前捕获。"""

    @abstractmethod
    async def check_response(self, response: RoleResponse) -> ActionCheckResult:
        """对完整主动行为声明的无副作用检查。

        基于事件落地后的共同状态检查整个集合（含义、当前步骤边界、
        未确认结果依赖、客观条件及与已有行动的关系），不只检查新增
        项；只产生结果：不调用角色、不提交事实、不取消或预登记行动。
        结果记录所依据的世界状态版本。语义判断可由模型承担，运行时
        校验负责结构与一致性。
        """

    @abstractmethod
    async def adjudicate(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        *,
        event_ids: Sequence[EventId] = (),
        action_ids: Sequence[ActionId] = (),
    ) -> AdjudicationCandidate:
        """对同刻事件与行动生成联合裁决候选；不提交任何事实。

        候选基于本时刻事件提交前、World 已更新相关未完成行动进展后
        的共同状态，同时考虑相互作用，不因队列插入顺序或角色编号
        制造优先级。
        """

    @abstractmethod
    def validate_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationValidation:
        """运行时校验候选的结构与状态一致性；不改变状态。

        校验不通过走明确失败，不拆分候选、不部分接受。
        """

    @abstractmethod
    def commit_adjudication(self, candidate: AdjudicationCandidate) -> AdjudicationCommit:
        """实际提交通过校验的候选；未通过校验的候选必须拒绝。

        提交是原子边界：失败不得留下半提交事实，成功后递增状态版本。
        实际状态变化记录在提交结果中，与候选分开。
        """

    @abstractmethod
    def publish(self, commit: AdjudicationCommit) -> tuple[Delivery, ...]:
        """从已提交的裁决生成按角色感知范围过滤的投递。

        只能发布已提交的内容；不得发布未提交候选或与提交后事实
        矛盾的结果文本。
        """
