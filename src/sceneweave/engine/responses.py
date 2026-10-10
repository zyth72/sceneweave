"""一次处理周期的角色响应采集屏障。

屏障只在同一周期内并行投递全部角色的输入并采集完整主动行为声明；
自身不拥有或推进剧情时钟，不检查、不登记、不撤回或提交客观事实。
整链翻译与 World 整链检查在屏障之后由调度器统一编排（见
``engine/scheduler.py``）。感知、响应与错误都通过注入的追踪接口留痕，
可沿周期/角色/响应/行动身份检索。
"""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    EventId,
    ResponseId,
    RoleId,
    WorldStateVersion,
    require_identifier,
    require_unique_identifiers,
)
from sceneweave.domain.messages import (
    MaintainAction,
    ModifiedAction,
    NewAction,
    RoleResponse,
    StepAction,
    TurnInput,
)
from sceneweave.domain.time import SimulationTime, validate_simulation_time
from sceneweave.domain.tracing import TraceRecord, TraceValue, TurnTracer
from sceneweave.roles.base import Role


@dataclass(frozen=True, slots=True)
class CandidateResponse:
    """某角色本轮的完整行为声明候选（尚未整链检查）。

    只校验结构、周期/角色/响应关联与声明内部一致性；World 整链检查
    与保存由调度器在翻译后统一执行。不得在这里声称检查已经通过。
    """

    response: RoleResponse

    def __post_init__(self) -> None:
        if not isinstance(self.response, RoleResponse):
            raise ValueError("候选必须包含 RoleResponse")


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """全部角色完成后整体返回的本轮候选；保留响应与提交组边界。

    键必须与各响应的角色身份一致。used_response_ids 记录本轮（含打回
    过程中）已使用的响应身份，供后段翻译重试延续同一周期内的唯一性检查。
    """

    cycle_id: CycleId
    simulation_time: SimulationTime
    candidates: Mapping[RoleId, CandidateResponse]
    used_response_ids: tuple[ResponseId, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.cycle_id, "周期身份")
        validate_simulation_time(self.simulation_time, "剧情时刻")
        if not isinstance(self.candidates, Mapping):
            raise ValueError("本轮结果必须是角色到候选响应的映射")
        for key, item in self.candidates.items():
            require_identifier(key, "角色身份")
            if not isinstance(item, CandidateResponse):
                raise ValueError("本轮结果值必须是 CandidateResponse")
            if key != item.response.role_id:
                raise ValueError(f"结果键与响应角色身份不一致：{key}")
            if item.response.cycle_id != self.cycle_id:
                raise ValueError(f"响应周期与本轮不一致：{key}")
        if not isinstance(self.used_response_ids, tuple):
            raise ValueError("已使用响应身份必须是元组")
        require_unique_identifiers(self.used_response_ids, "已使用响应身份")
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))


class ResponseRunner:
    """一个处理周期的并行响应采集屏障。

    场景中全部角色都获得明确输入：不同角色并行处理，同一角色的
    ``perceive`` 严格串行；完成先后不产生剧情优先级，返回顺序按角色
    身份固定，仅用于稳定呈现。响应身份在当前处理周期内全局唯一（含
    跨角色）。单个实例不允许同时执行两轮，防止同一角色状态被并发写入。
    """

    def __init__(
        self,
        roles: Iterable[Role],
        *,
        tracer: TurnTracer | None = None,
    ) -> None:
        self._roles: dict[RoleId, Role] = {}
        for role in roles:
            if role.role_id in self._roles:
                raise ValueError(f"重复角色身份：{role.role_id}")
            self._roles[role.role_id] = role
        if not self._roles:
            raise ValueError("场景必须至少包含一个角色；空场景没有周期上下文")
        self._tracer = tracer
        self._running = False

    async def run_cycle(
        self, inputs: Mapping[RoleId, TurnInput] | Iterable[TurnInput]
    ) -> CycleOutcome:
        """运行一个处理周期的响应采集；错误传播给调用者，不返回部分结果。

        未知角色、缺失角色输入、输入键与角色不一致、跨周期混合输入
        和非法响应抛出 ValueError。角色异常原样传播；已更新的角色
        内部状态不会自动回滚，调用者不能盲目重放本轮。
        """
        if self._running:
            raise RuntimeError("上一轮角色响应尚未结束")
        turn_inputs = self._collect_inputs(inputs)
        first = next(iter(turn_inputs.values()))
        cycle_id = first.cycle_id
        simulation_time = first.simulation_time
        used_response_ids: set[ResponseId] = set()
        self._running = True
        try:
            self._trace_cycle(
                "cycle.start",
                cycle_id,
                simulation_time,
                "处理周期开始",
                {"roles": len(self._roles)},
            )
            async with asyncio.TaskGroup() as group:
                tasks = {
                    role_id: group.create_task(
                        self._perceive_role(
                            self._roles[role_id],
                            turn_inputs[role_id],
                            used_response_ids,
                        )
                    )
                    for role_id in self._roles
                }
            accepted = {role_id: tasks[role_id].result() for role_id in sorted(tasks)}
            self._require_unique_action_ownership(accepted)
            self._trace_cycle(
                "cycle.complete",
                cycle_id,
                simulation_time,
                "处理周期完成",
                {"roles": len(accepted)},
            )
            return CycleOutcome(
                cycle_id=cycle_id,
                simulation_time=simulation_time,
                candidates=MappingProxyType(accepted),
                used_response_ids=tuple(sorted(used_response_ids, key=str)),
            )
        except asyncio.CancelledError:
            self._safe_trace_cycle(cycle_id, simulation_time, "cycle.cancelled", "周期被取消")
            raise
        except Exception as error:
            self._safe_trace_cycle(
                cycle_id,
                simulation_time,
                "cycle.error",
                "周期处理失败",
                error=f"{type(error).__name__}: {error!r}",
            )
            raise
        finally:
            self._running = False

    def _collect_inputs(
        self, inputs: Mapping[RoleId, TurnInput] | Iterable[TurnInput]
    ) -> dict[RoleId, TurnInput]:
        turn_inputs: dict[RoleId, TurnInput] = {}
        if isinstance(inputs, Mapping):
            for key, turn in inputs.items():
                if not isinstance(turn, TurnInput):
                    raise ValueError("周期输入必须是 TurnInput")
                if key != turn.role_id:
                    raise ValueError(f"输入键与角色身份不一致：{key} != {turn.role_id}")
                turn_inputs[turn.role_id] = turn
        else:
            for turn in inputs:
                if not isinstance(turn, TurnInput):
                    raise ValueError("周期输入必须是 TurnInput")
                if turn.role_id in turn_inputs:
                    raise ValueError(f"同一角色收到多份本轮输入：{turn.role_id}")
                turn_inputs[turn.role_id] = turn
        missing = self._roles.keys() - turn_inputs.keys()
        if missing:
            raise ValueError(f"场景角色缺少本轮输入：{sorted(missing)}")
        unknown = turn_inputs.keys() - self._roles.keys()
        if unknown:
            raise ValueError(f"未知角色：{sorted(unknown)}")
        if len({turn.cycle_id for turn in turn_inputs.values()}) > 1:
            raise ValueError("一轮只能处理同一个周期")
        if len({turn.simulation_time for turn in turn_inputs.values()}) > 1:
            raise ValueError("一轮只能处理同一剧情时刻")
        return turn_inputs

    async def _perceive_role(
        self,
        role: Role,
        turn: TurnInput,
        used_response_ids: set[ResponseId],
    ) -> CandidateResponse:
        current: RoleResponse | None = None
        try:
            self._trace(
                "role.perceive",
                turn,
                role.role_id,
                None,
                "投递本轮输入",
                {
                    "observations": len(turn.observations),
                    "introspection": turn.trigger.introspection,
                    "known_action_states": len(turn.known_action_states),
                },
                event_ids=turn.trigger.event_ids,
                inputs=tuple(item.content for item in turn.observations),
            )
            response = await role.perceive(turn)
            self._require_valid_response(response, turn, role, used_response_ids)
            current = response
            details = self._response_details(response)
            details["reused"] = response.reused
            self._trace(
                "role.response",
                turn,
                role.role_id,
                response.response_id,
                "角色返回本轮完整声明",
                details,
                tuple(item.action_id for item in response.actions),
                outputs=self._action_texts(response),
            )
            return CandidateResponse(response=response)
        except asyncio.CancelledError:
            self._safe_trace(
                "role.cancelled",
                turn,
                role.role_id,
                current,
                "角色处理被取消",
            )
            raise
        except Exception as error:
            self._safe_trace(
                "role.error",
                turn,
                role.role_id,
                current,
                f"角色 {role.role_id} 处理失败",
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def _require_valid_response(
        self,
        response: object,
        turn: TurnInput,
        role: Role,
        used_response_ids: set[ResponseId],
    ) -> None:
        """在访问字段或记录响应日志之前完成类型与关联校验。"""
        if not isinstance(response, RoleResponse):
            raise ValueError(
                f"角色 {role.role_id} 必须返回 RoleResponse，得到：{type(response).__name__}"
            )
        if response.cycle_id != turn.cycle_id:
            raise ValueError(f"响应周期与输入不一致：{response.response_id}")
        if response.role_id != role.role_id:
            raise ValueError(f"响应角色与角色身份不一致：{response.response_id}")
        if response.response_id in used_response_ids:
            raise RuntimeError(
                f"响应身份在当前周期已被使用，不得重用或跨角色共享：{response.response_id}"
            )
        used_response_ids.add(response.response_id)

    @staticmethod
    def _require_unique_action_ownership(
        accepted: Mapping[RoleId, CandidateResponse],
    ) -> None:
        """全部角色完成后检查行动身份不得跨角色重复。

        这里只做本轮声明的一致性检查，不创建行动注册表；已有行动的
        客观归属与终态校验仍由 World 负责。
        """
        owners: dict[ActionId, RoleId] = {}
        for role_id, item in accepted.items():
            for action in item.response.actions:
                owner = owners.get(action.action_id)
                if owner is not None and owner != role_id:
                    raise ValueError(f"同一行动身份不能同时指向多个角色：{action.action_id}")
                owners[action.action_id] = role_id

    @staticmethod
    def _action_texts(response: RoleResponse) -> tuple[str, ...]:
        texts: list[str] = []
        for item in response.actions:
            if isinstance(item, StepAction):
                texts.append(item.action.display)
            elif isinstance(item, NewAction):
                texts.append(item.description)
            elif isinstance(item, ModifiedAction):
                texts.append(item.change)
            else:
                texts.append(f"维持 {item.action_id}")
        return tuple(texts)

    @staticmethod
    def _response_details(response: RoleResponse) -> dict[str, TraceValue]:
        new_group: str | None = None
        for item in response.actions:
            if isinstance(item, NewAction):
                new_group = item.commit_group_id
                break
        return {
            "maintained": sum(isinstance(item, MaintainAction) for item in response.actions),
            "new_actions": sum(isinstance(item, NewAction) for item in response.actions),
            "modified": sum(isinstance(item, ModifiedAction) for item in response.actions),
            "step_actions": sum(isinstance(item, StepAction) for item in response.actions),
            "commit_group": new_group,
        }

    def _trace(
        self,
        stage: str,
        turn: TurnInput,
        role_id: RoleId,
        response_id: ResponseId | None,
        summary: str,
        details: Mapping[str, TraceValue],
        action_ids: tuple[ActionId, ...] = (),
        *,
        event_ids: tuple[EventId, ...] = (),
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
        related_response_id: ResponseId | None = None,
        world_state_version: WorldStateVersion | None = None,
        error: str | None = None,
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=turn.cycle_id,
                simulation_time=turn.simulation_time,
                summary=summary,
                role_id=role_id,
                response_id=response_id,
                related_response_id=related_response_id,
                action_ids=action_ids,
                event_ids=event_ids,
                inputs=inputs,
                outputs=outputs,
                world_state_version=world_state_version,
                details=details,
                error=error,
            )
        )

    def _safe_trace(
        self,
        stage: str,
        turn: TurnInput,
        role_id: RoleId,
        current: RoleResponse | None,
        summary: str,
        error: str | None = None,
    ) -> None:
        """错误路径的留痕；追踪器自身的失败不得掩盖原异常。"""
        if self._tracer is None:
            return
        try:
            self._tracer.trace(
                TraceRecord(
                    stage=stage,
                    cycle_id=turn.cycle_id,
                    simulation_time=turn.simulation_time,
                    summary=summary,
                    role_id=role_id,
                    response_id=current.response_id if current is not None else None,
                    action_ids=(
                        tuple(item.action_id for item in current.actions)
                        if current is not None
                        else ()
                    ),
                    error=error,
                )
            )
        except Exception:
            pass

    def _trace_cycle(
        self,
        stage: str,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        summary: str,
        details: Mapping[str, TraceValue],
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=cycle_id,
                simulation_time=simulation_time,
                summary=summary,
                details=details,
            )
        )

    def _safe_trace_cycle(
        self,
        cycle_id: CycleId,
        simulation_time: SimulationTime,
        stage: str,
        summary: str,
        error: str | None = None,
    ) -> None:
        if self._tracer is None:
            return
        try:
            self._tracer.trace(
                TraceRecord(
                    stage=stage,
                    cycle_id=cycle_id,
                    simulation_time=simulation_time,
                    summary=summary,
                    error=error,
                )
            )
        except Exception:
            pass
