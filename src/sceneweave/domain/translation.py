"""行动翻译的输入输出消息。

翻译器把角色的完整一维动作链（尚未检查）转换为平坦、独立的逐动作时间
安排；这里只定义消息与结构校验，不实现翻译模型本身。时间安排是对尝试的
安排，不是已经发生的事实。整链翻译完成后才由 World 对整条链与完整排程
做无副作用检查；无依赖的动作可部分交叠或链式先后，不再要求整组共同
非零时间交集。旧三类声明（新增/修改）的安排约束仅为低层历史测试保留。
"""

from dataclasses import dataclass

from sceneweave.domain.action_plan import ActionPlan
from sceneweave.domain.adjudication import ObjectiveContext
from sceneweave.domain.identity import (
    ActionId,
    CommitGroupId,
    CycleId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_unique_identifiers,
    require_version,
)
from sceneweave.domain.messages import ActionCheckResult, ActionPhase, RoleResponse, StepAction
from sceneweave.domain.time import SimulationTime, validate_simulation_time


def _validate_window(
    start: SimulationTime, end: SimulationTime, checkpoints: tuple[SimulationTime, ...]
) -> None:
    """开始、过程节点与结束的通用结构约束；与当前时刻无关。"""
    if end <= start:
        raise ValueError("预计结束时刻必须晚于开始时刻")
    if not isinstance(checkpoints, tuple):
        raise ValueError("过程节点必须是毫秒时间元组")
    previous = start
    for checkpoint in checkpoints:
        validate_simulation_time(checkpoint, "过程节点")
        if checkpoint <= previous:
            raise ValueError("过程节点必须严格递增且晚于开始时刻")
        previous = checkpoint
    if end <= previous:
        raise ValueError("过程节点必须早于预计结束时刻")


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    """单个新行动的执行时间安排。

    开始、过程节点与结束时刻都是本次提交的安排，不代表状态变化
    已经发生；实际结果仍由 World 在对应剧情时刻结算。
    """

    action_id: ActionId
    commit_group_id: CommitGroupId
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        require_identifier(self.commit_group_id, "提交组身份")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class ModifiedActionSchedule:
    """对已有行动的修改后的完整时间安排。

    历史不可改写：已到或已过的开始与过程节点必须原样保留，只有
    未来节点可以调整。该约束与当前时刻相关，在登记校验时核对；
    这里只做结构性校验。修改不参与新行动的并行组交集约束。
    """

    action_id: ActionId
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class ActionArrangement:
    """翻译请求携带的既有行动安排与阶段；表达引擎已知的事实性安排。"""

    action_id: ActionId
    phase: ActionPhase
    description: str
    start: SimulationTime
    expected_end: SimulationTime
    checkpoints: tuple[SimulationTime, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "行动身份")
        if not isinstance(self.phase, ActionPhase):
            raise ValueError("行动阶段必须是 ActionPhase")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("行动描述必须是非空字符串")
        validate_simulation_time(self.start, "开始时刻")
        validate_simulation_time(self.expected_end, "预计结束时刻")
        _validate_window(self.start, self.expected_end, self.checkpoints)


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """交给行动翻译器的输入：角色的完整候选链与客观上下文。

    翻译发生在整链检查之前：response 是尚未通过检查的完整声明候选。
    arrangements 是该角色已登记行动的现行安排与阶段（既有进展的
    引擎侧事实）；objective_context 是 World 提供的只读客观条件
    （位置、姿态、障碍、实际进展，携带共同状态版本），只供翻译器
    使用，不得流入角色感知。
    """

    response: RoleResponse
    current_time: SimulationTime
    world_state_version: WorldStateVersion
    arrangements: tuple[ActionArrangement, ...] = ()
    objective_context: ObjectiveContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, RoleResponse):
            raise ValueError("翻译输入必须包含角色响应")
        validate_simulation_time(self.current_time, "当前剧情时刻")
        require_version(self.world_state_version, "共同世界状态版本")
        if not isinstance(self.arrangements, tuple) or not all(
            isinstance(item, ActionArrangement) for item in self.arrangements
        ):
            raise ValueError("既有安排必须是 ActionArrangement 元组")
        require_unique_identifiers(
            [item.action_id for item in self.arrangements], "既有安排行动身份"
        )
        if self.objective_context is not None and not isinstance(
            self.objective_context, ObjectiveContext
        ):
            raise ValueError("客观上下文必须是 ObjectiveContext")
        if (
            self.objective_context is not None
            and self.objective_context.role_id != self.response.role_id
        ):
            raise ValueError("客观上下文必须属于被翻译的角色")
        if (
            self.objective_context is not None
            and self.objective_context.world_state_version != self.world_state_version
        ):
            raise ValueError("客观上下文必须使用翻译请求的共同世界状态版本")


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """翻译输出：保留原响应及提交组关联，给出各声明的未来时间安排。

    schedules 只覆盖本次新行动；modified_schedules 覆盖明确修改的
    行动。维持声明不需要安排：同 ID 维持不重新翻译或重排。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    schedules: tuple[ActionSchedule, ...] = ()
    modified_schedules: tuple[ModifiedActionSchedule, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        if not isinstance(self.schedules, tuple) or not all(
            isinstance(item, ActionSchedule) for item in self.schedules
        ):
            raise ValueError("时间安排必须是 ActionSchedule 元组")
        require_unique_identifiers([item.action_id for item in self.schedules], "行动身份")
        if not isinstance(self.modified_schedules, tuple) or not all(
            isinstance(item, ModifiedActionSchedule) for item in self.modified_schedules
        ):
            raise ValueError("修改安排必须是 ModifiedActionSchedule 元组")
        require_unique_identifiers(
            [item.action_id for item in self.modified_schedules], "修改行动身份"
        )
        groups = {item.commit_group_id for item in self.schedules}
        if len(groups) > 1:
            raise ValueError("翻译结果只能包含一个提交组")
        overlap = {item.action_id for item in self.schedules} & {
            item.action_id for item in self.modified_schedules
        }
        if overlap:
            raise ValueError(f"同一行动不能同时是新行动安排与修改安排：{sorted(overlap)}")


@dataclass(frozen=True, slots=True)
class PlanAcceptance:
    """整链检查通过后交给角色的保存消息：完整链、完整排程与检查依据。

    这是「准备完成」到「角色内部保存」之间的独立边界，与执行提交
    （``ExecutionSubmission``）分开。角色收到后把完整链与全部排程原子
    保存（私有缓存），关联候选、检查与翻译版本与身份；保存原子完成，
    打回、格式错误或取消不得保存。保存成功本身不登记任何执行事件，
    也不产生世界事实。
    """

    declaration: RoleResponse
    plan: ActionPlan
    schedules: tuple[ActionSchedule, ...]
    check: ActionCheckResult

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, RoleResponse):
            raise ValueError("保存消息必须包含候选声明")
        if not isinstance(self.plan, ActionPlan):
            raise ValueError("保存消息必须包含完整动作链")
        if not isinstance(self.schedules, tuple) or not all(
            isinstance(item, ActionSchedule) for item in self.schedules
        ):
            raise ValueError("保存消息必须包含 ActionSchedule 元组")
        if not isinstance(self.check, ActionCheckResult):
            raise ValueError("保存消息必须包含整链检查结果")
        if not self.check.passed:
            raise ValueError("只有通过整链检查的声明才能保存")
        if self.declaration.plan is None or self.declaration.plan != self.plan:
            raise ValueError("保存消息的完整链必须与候选声明的 plan 完全一致")
        if (
            self.check.cycle_id,
            self.check.role_id,
            self.check.response_id,
        ) != (
            self.declaration.cycle_id,
            self.declaration.role_id,
            self.declaration.response_id,
        ):
            raise ValueError("保存消息的检查与候选身份不一致")
        plan_ids = [item.action_id for item in self.plan.actions]
        if any(item is None for item in plan_ids):
            raise ValueError("保存消息的完整链必须携带已分配的动作身份")
        schedule_ids = [item.action_id for item in self.schedules]
        require_unique_identifiers(schedule_ids, "保存排程行动身份")
        if set(schedule_ids) != set(plan_ids):
            raise ValueError("保存消息的排程必须一一对应完整链的每个动作")


@dataclass(frozen=True, slots=True)
class ExecutionSubmission:
    """最近步骤执行提交：只登记本周期要执行的原子动作及其「新增」排程。

    与完整链分开：未来步骤仍留在角色内部缓存，不进入登记。身份与
    候选响应一致；schedules 只覆盖本次新登记的动作（已有行动的现行
    安排不重复携带，避免被当作重排）。检查通过并保存之后才生成，是
    执行提交与保存两种用途的不同类型，不用一个语义掩盖。
    """

    cycle_id: CycleId
    role_id: RoleId
    response_id: ResponseId
    actions: tuple[StepAction, ...]
    schedules: tuple[ActionSchedule, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        require_identifier(self.role_id, "角色身份")
        require_identifier(self.response_id, "响应身份")
        if not isinstance(self.actions, tuple) or not all(
            isinstance(item, StepAction) for item in self.actions
        ):
            raise ValueError("执行提交动作必须是 StepAction 元组")
        if not isinstance(self.schedules, tuple) or not all(
            isinstance(item, ActionSchedule) for item in self.schedules
        ):
            raise ValueError("执行提交排程必须是 ActionSchedule 元组")
        require_unique_identifiers([item.action_id for item in self.actions], "提交动作身份")
        require_unique_identifiers([item.action_id for item in self.schedules], "提交排程身份")
        action_ids = {item.action_id for item in self.actions}
        if not {item.action_id for item in self.schedules} <= action_ids:
            raise ValueError("执行提交的排程只能覆盖本次提交的动作")
