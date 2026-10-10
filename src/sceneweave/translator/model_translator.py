"""生产可调用的模型驱动行动翻译器。

ModelTranslator 通过可注入的采样器调用真实模型文本，把角色的完整一维
动作链（尚未通过检查）转换为平坦、独立的逐动作时间安排：结构化链中
每个动作只约束自己的 start/expected_end/checkpoints，可交叠或链式，
已有现行安排的动作不重译。输入完整关联候选 RoleResponse 与只读客观
上下文（位置/姿态/障碍/实际进展），主观意向与客观条件在提示中明确
区分；缺关键上下文是明确错误，不靠猜或默认固定延迟。

结构校验在本入口完成：非法 JSON、重复/未知字段、缺数组、bool/浮点
毫秒、行动或提交组关联不一致、多余安排都是运行错误（不伪空、不拆组
部分接受、不自行改写绕过角色决定）。时间业务约束（新动作开始晚于当前
时刻、修改历史不可改写；旧三类声明的组内共同非零交集仅为低层历史测试
路径）仍由引擎的 registration_issues 与打回路径执行——本入口对合法
结构不判时间业务。仅维持或空声明直接合成空结果并显式跳过留痕，不调用
模型。

并发与录放：同一角色的并发请求（含跳过路径）是明确错误，try/finally
可靠清理；不同角色并行。sampler_for_role 按角色提供采样器，每个角色
只构造一次并缓存（构造失败不缓存），配合 RecordingDirectory/
ReplayDirectory 的作用域文件可确定性重放并行运行——共享单个录制器
无法重放。留痕：成功路径记录完整提示与原始模型输出；任何失败（预检、
工厂、采样、解析、构造）都记录本阶段错误事件后按原异常身份传播，
原始异常始终优先于追踪失败；取消释放占用并记录后保持 CancelledError
语义。错误事件内容不进入角色感知（打回只携带引擎的机械时间约束原因）。
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sceneweave.domain.identity import ActionId, CommitGroupId, RoleId
from sceneweave.domain.messages import (
    MaintainAction,
    ModifiedAction,
    NewAction,
    RoleResponse,
    StepAction,
)
from sceneweave.domain.model_json import (
    ModelJsonError,
    parse_model_json,
    require_json_int,
    require_json_int_list,
)
from sceneweave.domain.modeling import TextSampler
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord, TraceValue, TurnTracer
from sceneweave.domain.translation import (
    ActionSchedule,
    ModifiedActionSchedule,
    TranslationRequest,
    TranslationResult,
)

SCHEMA = (
    '{"schedules": [{"action_id": "新行动身份", "commit_group_id": "提交组",'
    ' "start": 整数毫秒, "expected_end": 整数毫秒, "checkpoints": [整数毫秒, ...]}, ...],'
    ' "modified_schedules": [{"action_id": "修改行动身份", "start": 整数毫秒,'
    ' "expected_end": 整数毫秒, "checkpoints": [整数毫秒, ...]}, ...]}'
)

# 结构化当前动作的平坦安排：每个动作只有自己的时间轴，无需提交组。
SCHEMA_STEP = (
    '{"schedules": [{"action_id": "新动作身份", "start": 整数毫秒,'
    ' "expected_end": 整数毫秒, "checkpoints": [整数毫秒, ...]}, ...]}'
)


class ModelTranslatorError(ValueError):
    """翻译器结构/关联错误；属于运行错误，不得转换为空声明。"""


@dataclass(frozen=True, slots=True)
class _StepSpec:
    """翻译器内部统一的结构化待排程动作（来自完整链或旧声明）。"""

    action_id: ActionId
    display: str
    depends_on: tuple[str, ...]


class ModelTranslator:
    """以注入采样器驱动的行动翻译器；实现 translator.base.Translator 契约。

    sampler_for_role 按角色提供采样器：每个角色只在首次使用时构造一次
    并缓存（构造失败不缓存、如实传播），不同角色相互独立、可并行；
    shared() 便于单采样器装配。构造不调用模型、不读配置。
    """

    def __init__(
        self,
        sampler_for_role: Callable[[RoleId], TextSampler],
        *,
        tracer: TurnTracer | None = None,
    ) -> None:
        if not callable(sampler_for_role):
            raise ValueError("sampler_for_role 必须是可调用对象")
        self._sampler_for_role = sampler_for_role
        self._tracer = tracer
        self._in_flight: set[RoleId] = set()
        self._samplers: dict[RoleId, TextSampler] = {}

    @classmethod
    def shared(cls, sampler: TextSampler, *, tracer: TurnTracer | None = None) -> ModelTranslator:
        """单一采样器装配（单作用域；并行录放请用作用域工厂）。"""
        return cls(lambda _role: sampler, tracer=tracer)

    @classmethod
    def scoped(
        cls,
        directory: Any,
        *,
        tracer: TurnTracer | None = None,
        prefix: str = "translator",
    ) -> ModelTranslator:
        """按 RecordingDirectory/ReplayDirectory 的作用域装配。

        每个角色获得独立文件（{prefix}-{角色}）且只构造一次；跨角色并行
        运行可确定性重放；共享单个录制器无法回放并行完成顺序。
        """

        def factory(role: RoleId) -> TextSampler:
            built: TextSampler = directory.for_scope(f"{prefix}-{role}")
            return built

        return cls(factory, tracer=tracer)

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """翻译完整声明；时间业务校验与打回仍由引擎负责。

        对结构化角色优先翻译 ``response.plan`` 的**完整动作链**（含尚未满足
        依赖的后续动作），使整链检查能拿到全部排程；已有现行安排的动作不
        重译。非结构化旧声明回退到 ``response.actions``。
        """
        self._require_request(request)
        response = request.response
        role_id = response.role_id
        if role_id in self._in_flight:
            error = ModelTranslatorError(f"同一角色 {role_id} 的翻译请求必须串行，检测到并发调用")
            self._trace_failure(request, error, prompt=repr(request), raw=None, error_kind="并发")
            raise error
        self._in_flight.add(role_id)
        prompt = repr(request)
        raw: str | None = None
        error_kind = "预检"
        try:
            new_actions = [item for item in response.actions if isinstance(item, NewAction)]
            modified_actions = [
                item for item in response.actions if isinstance(item, ModifiedAction)
            ]
            arrangement_ids = {item.action_id for item in request.arrangements}
            specs = self._step_specs(response)
            step_new = [item for item in specs if item.action_id not in arrangement_ids]
            if not new_actions and not modified_actions and not step_new:
                self._trace_stage(
                    "translator.skip",
                    request,
                    "仅维持或空声明，无需模型翻译",
                    inputs=(prompt,),
                    outputs=(),
                    details={"maintained": len(response.actions)},
                )
                return TranslationResult(response.cycle_id, role_id, response.response_id)
            if request.objective_context is None:
                raise ModelTranslatorError(
                    f"翻译请求缺少只读客观上下文：{role_id} 的新行动/修改需要 World 条件"
                )
            for action in modified_actions:
                if action.action_id not in arrangement_ids:
                    raise ModelTranslatorError(f"修改声明缺少既有安排：{action.action_id}")
            if step_new:
                return await self._translate_steps(request, step_new)
            prompt = self._build_prompt(request, new_actions, modified_actions)
            error_kind = "追踪"
            self._trace_stage(
                "translator.invoke",
                request,
                "翻译器确定时间安排",
                inputs=(prompt,),
                outputs=(),
                details={
                    "new_actions": len(new_actions),
                    "modified": len(modified_actions),
                    "maintained": sum(
                        isinstance(item, MaintainAction) for item in response.actions
                    ),
                },
            )
            error_kind = "工厂"
            sampler = self._sampler_for(role_id)
            error_kind = "采样"
            raw = await sampler.sample(prompt, component="translator")
            error_kind = "解析"
            schedules, modified_schedules = self._parse(raw, request, new_actions, modified_actions)
            error_kind = "构造"
            try:
                result = TranslationResult(
                    response.cycle_id,
                    role_id,
                    response.response_id,
                    schedules=schedules,
                    modified_schedules=modified_schedules,
                )
            except ValueError as error:
                raise ModelTranslatorError(f"翻译结果构造失败：{error}") from error
            error_kind = "追踪"
            self._trace_stage(
                "translator.parsed",
                request,
                "翻译完成",
                inputs=(prompt,),
                outputs=(raw, *self._schedule_texts(schedules, modified_schedules)),
                details={
                    "schedules": len(schedules),
                    "modified_schedules": len(modified_schedules),
                },
            )
            return result
        except (Exception, asyncio.CancelledError) as error:
            self._trace_failure(
                request,
                error,
                prompt=prompt,
                raw=raw,
                error_kind="取消" if isinstance(error, asyncio.CancelledError) else error_kind,
            )
            raise
        finally:
            self._in_flight.discard(role_id)

    async def _translate_steps(
        self, request: TranslationRequest, step_new: list[_StepSpec]
    ) -> TranslationResult:
        """结构化当前动作路径：平坦、独立的逐动作时间安排。

        每个动作只有自己的 start/expected_end/checkpoints；不要求整组共同
        时间交集，允许交叠或链式。依赖表示真实完成门控，由引擎处理。
        """
        response = request.response
        role_id = response.role_id
        prompt = self._build_step_prompt(request, step_new)
        self._trace_stage(
            "translator.invoke",
            request,
            "翻译器确定平坦的独立时间安排",
            inputs=(prompt,),
            outputs=(),
            details={"step_actions": len(step_new)},
        )
        sampler = self._sampler_for(role_id)
        raw = await sampler.sample(prompt, component="translator")
        schedules = self._parse_step_schedules(raw, step_new, request)
        try:
            result = TranslationResult(
                response.cycle_id,
                role_id,
                response.response_id,
                schedules=schedules,
            )
        except ValueError as error:
            raise ModelTranslatorError(f"翻译结果构造失败：{error}") from error
        self._trace_stage(
            "translator.parsed",
            request,
            "翻译完成",
            inputs=(prompt,),
            outputs=(raw, *self._schedule_texts(schedules, ())),
            details={"schedules": len(schedules)},
        )
        return result

    @staticmethod
    def _step_specs(response: RoleResponse) -> tuple[_StepSpec, ...]:
        """优先取完整链（plan）的全部动作；旧声明回退到结构化 actions。"""
        plan = response.plan
        if plan is not None and plan.actions:
            specs: list[_StepSpec] = []
            for item in plan.actions:
                action_id = item.action_id
                if action_id is None:  # pragma: no cover - 角色保证身份齐备
                    raise ModelTranslatorError("完整链动作缺少已分配身份")
                specs.append(_StepSpec(action_id, item.action.display, tuple(item.depends_on)))
            return tuple(specs)
        return tuple(
            _StepSpec(
                item.action_id,
                item.action.display,
                tuple(str(dep) for dep in item.depends_on),
            )
            for item in response.actions
            if isinstance(item, StepAction)
        )

    def _build_step_prompt(self, request: TranslationRequest, step_new: list[_StepSpec]) -> str:
        objective = request.objective_context
        assert objective is not None  # 预检已拒绝缺上下文
        conditions = [f"- {item}" for item in objective.conditions]
        existing_lines = [
            f"- {item.action_id}（{item.phase.value}）：{item.description}"
            for item in request.arrangements
        ] or ["- 无"]
        action_lines = []
        for item in step_new:
            suffix = (
                " 依赖[" + "、".join(str(dep) for dep in item.depends_on) + "]"
                if item.depends_on
                else ""
            )
            action_lines.append(f"- {item.action_id}：{item.display}{suffix}")
        lines = [
            "你是行动翻译器：把角色的结构化原子动作转换为可调度的平坦时间安排。",
            f"角色 {request.response.role_id}；当前剧情时刻 {request.current_time} 毫秒"
            "（整数毫秒；模型耗时不影响剧情时间）。",
            "【本次新动作】（每个动作独立安排）",
            *action_lines,
            "【World 客观上下文】（只读事实，翻译专用，不进入角色认知）",
            *conditions,
            "【既有安排】（引擎登记，含历史过程节点）",
            *existing_lines,
            "要求：",
            "- schedules 一一对应本次全部新动作，action_id 必须一致；",
            "- 每个动作只约束自己的 start、expected_end 与 checkpoints；"
            "start 晚于当前剧情时刻，start < expected_end，checkpoints 严格递增且在区间内；",
            "- 不要求各动作有共同时间交集：动作可以交叠、也可以链式先后，按物理常识安排即可；",
            "- 依赖（依赖[…]）表示“前置动作被 World 实际提交为完成，本动作才允许开始”，"
            "不是预计结束、也不是到点就成功；你给出合理预计，实际开始由运行时按真实完成门控；",
            "- 只为必要的过程检查点设置 checkpoints，无必要过程节点时用 []；",
            "- start/expected_end/checkpoints 只接受整数毫秒，不接受 bool 或浮点；",
            "- 时间是本次执行安排，不代表已经发生；不产生额外动作或未来行动链。",
            "只输出一个 JSON 对象，只含 schedules 一个字段，格式：",
            SCHEMA_STEP,
        ]
        return "\n".join(lines)

    def _parse_step_schedules(
        self,
        raw: str,
        step_new: list[_StepSpec],
        request: TranslationRequest,
    ) -> tuple[ActionSchedule, ...]:
        try:
            data = parse_model_json(raw, "translator", allowed_keys=("schedules",))
        except ModelJsonError as error:
            raise ModelTranslatorError(str(error)) from error
        if not isinstance(data.get("schedules"), list):
            raise ModelTranslatorError("translator 输出缺少 schedules 数组")
        expected = {item.action_id for item in step_new}
        group = CommitGroupId(f"g-{request.response.role_id}-{request.response.response_id}")
        parsed: list[ActionSchedule] = []
        seen: set[ActionId] = set()
        for item in data["schedules"]:
            if not isinstance(item, dict):
                raise ModelTranslatorError("schedules 每项必须是 JSON 对象")
            unknown = sorted(set(item) - {"action_id", "start", "expected_end", "checkpoints"})
            if unknown:
                raise ModelTranslatorError(f"schedules 项包含未知字段：{unknown}")
            missing = [key for key in ("action_id", "start", "expected_end") if key not in item]
            if missing:
                raise ModelTranslatorError(f"schedules 项缺少必填字段：{missing}")
            action_id = self._require_id(item["action_id"])
            if action_id not in expected:
                raise ModelTranslatorError(f"时间安排引用了非本次新动作：{action_id}")
            if action_id in seen:
                raise ModelTranslatorError(f"新动作安排身份重复：{action_id}")
            seen.add(action_id)
            try:
                schedule = ActionSchedule(
                    action_id=action_id,
                    commit_group_id=group,
                    start=SimulationTime(require_json_int(item["start"], "start")),
                    expected_end=SimulationTime(
                        require_json_int(item["expected_end"], "expected_end")
                    ),
                    checkpoints=tuple(
                        SimulationTime(value)
                        for value in require_json_int_list(
                            item.get("checkpoints", []), "checkpoints"
                        )
                    ),
                )
            except (ModelJsonError, ValueError) as error:
                raise ModelTranslatorError(f"新动作安排结构非法：{error}") from error
            parsed.append(schedule)
        missing_ids = expected - seen
        if missing_ids:
            raise ModelTranslatorError(
                f"新动作缺少时间安排：{sorted(str(item) for item in missing_ids)}"
            )
        return tuple(parsed)

    def _sampler_for(self, role_id: RoleId) -> TextSampler:
        """每个角色只构造一次采样器；构造失败如实传播且不缓存。"""
        cached = self._samplers.get(role_id)
        if cached is not None:
            return cached
        built: TextSampler = self._sampler_for_role(role_id)
        self._samplers[role_id] = built
        return built

    def _build_prompt(
        self,
        request: TranslationRequest,
        new_actions: list[NewAction],
        modified_actions: list[ModifiedAction],
    ) -> str:
        maintain = [item for item in request.response.actions if isinstance(item, MaintainAction)]
        objective = request.objective_context
        assert objective is not None  # 预检已拒绝缺上下文
        conditions = [f"- {item}" for item in objective.conditions]
        arrangement_lines = [
            f"- {item.action_id}（{item.phase.value}，{item.description}）："
            f"{item.start}→{item.expected_end}ms 节点[{', '.join(str(c) for c in item.checkpoints)}]"
            for item in request.arrangements
        ] or ["- 无"]
        lines = [
            "你是行动翻译器：把角色的主动行为声明转换为可调度的时间安排。",
            f"角色 {request.response.role_id}；"
            f"当前剧情时刻 {request.current_time} 毫秒（整数毫秒；模型耗时"
            "不影响剧情时间），"
            f"共同世界状态版本 {request.world_state_version}。",
            "【角色主观声明】（意向，尚不是事实）",
            *[
                f"- 新行动 {item.action_id}（组 {item.commit_group_id}）：{item.description}"
                for item in new_actions
            ],
            *[f"- 修改 {item.action_id}：{item.change}" for item in modified_actions],
            *(
                ["- 维持（无需重译）：" + "、".join(str(i.action_id) for i in maintain)]
                if maintain
                else []
            ),
            "【World 客观上下文】（只读事实，翻译专用，不进入角色认知）",
            *conditions,
            "【既有安排】（引擎登记，含历史过程节点）",
            *arrangement_lines,
            "要求：",
            "- schedules 一一对应本次全部新行动，action_id 与 commit_group_id 必须一致；",
            "- 新行动的 start 晚于当前剧情时刻；同组全部新行动满足"
            " max(start) < min(expected_end)（共同非零交集）；",
            "- modified_schedules 一一对应本次全部明确修改：已开始的行动保持原 start、"
            "已到达的过程节点保持不变，只调整未来节点，"
            "expected_end 晚于当前剧情时刻；",
            "- 每个安排满足 start < expected_end；checkpoints 严格递增，"
            "每个节点都满足 start < checkpoint < expected_end，不包含开始或结束时刻；",
            "- 只为必要的过程检查设置 checkpoints；无必要过程节点时用 []，"
            "不要机械地为简单动作添加固定比例检查点；",
            "- 维持行动不重新安排；不产生额外行动或未来行动链；",
            "- start/expected_end/checkpoints 是本次提交的执行安排（整数毫秒），不代表已经发生。",
            "只输出一个 JSON 对象，只含这两个字段，格式：",
            SCHEMA,
        ]
        return "\n".join(lines)

    def _parse(
        self,
        raw: str,
        request: TranslationRequest,
        new_actions: list[NewAction],
        modified_actions: list[ModifiedAction],
    ) -> tuple[tuple[ActionSchedule, ...], tuple[ModifiedActionSchedule, ...]]:
        """严格解析并核对与声明的一一对应；结构错误不伪空。"""
        try:
            data = parse_model_json(
                raw,
                "translator",
                allowed_keys=("schedules", "modified_schedules"),
            )
        except ModelJsonError as error:
            raise ModelTranslatorError(str(error)) from error
        for key in ("schedules", "modified_schedules"):
            if not isinstance(data.get(key), list):
                raise ModelTranslatorError(f"translator 输出缺少 {key} 数组")
        schedules = self._parse_schedules(data["schedules"])
        modified_schedules = self._parse_modified(data["modified_schedules"])
        self._require_correspondence(
            request, new_actions, modified_actions, schedules, modified_schedules
        )
        return schedules, modified_schedules

    def _parse_schedules(self, items: list[Any]) -> tuple[ActionSchedule, ...]:
        parsed: list[ActionSchedule] = []
        seen: set[ActionId] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ModelTranslatorError("schedules 每项必须是 JSON 对象")
            unknown = sorted(
                set(item) - {"action_id", "commit_group_id", "start", "expected_end", "checkpoints"}
            )
            if unknown:
                message = f"schedules 项包含未知字段：{unknown}"
                raise ModelTranslatorError(message)
            missing = [
                key
                for key in ("action_id", "commit_group_id", "start", "expected_end")
                if key not in item
            ]
            if missing:
                message = f"schedules 项缺少必填字段：{missing}"
                raise ModelTranslatorError(message)
            try:
                schedule = ActionSchedule(
                    action_id=self._require_id(item["action_id"]),
                    commit_group_id=self._require_group(item["commit_group_id"]),
                    start=SimulationTime(require_json_int(item["start"], "start")),
                    expected_end=SimulationTime(
                        require_json_int(item["expected_end"], "expected_end")
                    ),
                    checkpoints=tuple(
                        SimulationTime(value)
                        for value in require_json_int_list(
                            item.get("checkpoints", []), "checkpoints"
                        )
                    ),
                )
            except (ModelJsonError, ValueError) as error:
                raise ModelTranslatorError(f"新行动安排结构非法：{error}") from error
            if schedule.action_id in seen:
                raise ModelTranslatorError(f"新行动安排身份重复：{schedule.action_id}")
            seen.add(schedule.action_id)
            parsed.append(schedule)
        return tuple(parsed)

    def _parse_modified(self, items: list[Any]) -> tuple[ModifiedActionSchedule, ...]:
        parsed: list[ModifiedActionSchedule] = []
        seen: set[ActionId] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ModelTranslatorError("modified_schedules 每项必须是 JSON 对象")
            unknown = sorted(set(item) - {"action_id", "start", "expected_end", "checkpoints"})
            if unknown:
                message = f"modified_schedules 项包含未知字段：{unknown}"
                raise ModelTranslatorError(message)
            missing = [key for key in ("action_id", "start", "expected_end") if key not in item]
            if missing:
                message = f"modified_schedules 项缺少必填字段：{missing}"
                raise ModelTranslatorError(message)
            try:
                schedule = ModifiedActionSchedule(
                    action_id=self._require_id(item["action_id"]),
                    start=SimulationTime(require_json_int(item["start"], "start")),
                    expected_end=SimulationTime(
                        require_json_int(item["expected_end"], "expected_end")
                    ),
                    checkpoints=tuple(
                        SimulationTime(value)
                        for value in require_json_int_list(
                            item.get("checkpoints", []), "checkpoints"
                        )
                    ),
                )
            except (ModelJsonError, ValueError) as error:
                raise ModelTranslatorError(f"修改安排结构非法：{error}") from error
            if schedule.action_id in seen:
                raise ModelTranslatorError(f"修改安排身份重复：{schedule.action_id}")
            seen.add(schedule.action_id)
            parsed.append(schedule)
        return tuple(parsed)

    def _require_correspondence(
        self,
        request: TranslationRequest,
        new_actions: list[NewAction],
        modified_actions: list[ModifiedAction],
        schedules: tuple[ActionSchedule, ...],
        modified_schedules: tuple[ModifiedActionSchedule, ...],
    ) -> None:
        expected_new = {item.action_id: item.commit_group_id for item in new_actions}
        expected_modified = {item.action_id for item in modified_actions}
        got_new = {item.action_id: item.commit_group_id for item in schedules}
        got_modified = {item.action_id for item in modified_schedules}
        maintain_ids = {
            item.action_id for item in request.response.actions if isinstance(item, MaintainAction)
        }
        overlap = maintain_ids & (set(got_new) | got_modified)
        if overlap:
            raise ModelTranslatorError(f"维持行动不得重新安排：{sorted(overlap)}")
        if set(got_new) != set(expected_new):
            missing = sorted(set(expected_new) - set(got_new))
            extra = sorted(set(got_new) - set(expected_new))
            message = f"新行动安排与声明不一致：缺少 {missing}，多余 {extra}"
            raise ModelTranslatorError(message)
        for action_id, group in expected_new.items():
            if got_new[action_id] != group:
                message = f"新行动安排的提交组与声明不一致：{action_id}"
                raise ModelTranslatorError(message)
        if got_modified != expected_modified:
            missing = sorted(expected_modified - got_modified)
            extra = sorted(got_modified - expected_modified)
            message = f"修改安排与声明不一致：缺少 {missing}，多余 {extra}"
            raise ModelTranslatorError(message)

    def _require_request(self, request: object) -> None:
        if not isinstance(request, TranslationRequest):
            raise ModelTranslatorError(
                f"翻译请求必须是 TranslationRequest，得到：{type(request).__name__}"
            )
        if not isinstance(request.response, RoleResponse):
            raise ModelTranslatorError("翻译请求必须包含 RoleResponse")

    @staticmethod
    def _require_id(value: Any) -> ActionId:
        if not isinstance(value, str) or not value:
            raise ModelTranslatorError("action_id 必须是非空字符串")
        return ActionId(value)

    @staticmethod
    def _require_group(value: Any) -> CommitGroupId:
        if not isinstance(value, str) or not value:
            raise ModelTranslatorError("commit_group_id 必须是非空字符串")
        return CommitGroupId(value)

    @staticmethod
    def _schedule_texts(
        schedules: tuple[ActionSchedule, ...],
        modified_schedules: tuple[ModifiedActionSchedule, ...],
    ) -> tuple[str, ...]:
        """完整安排文本：身份、组、起止与全部过程节点。"""
        texts = [
            f"{s.action_id}（组 {s.commit_group_id}）：{s.start}→{s.expected_end}ms"
            f" 节点[{', '.join(str(c) for c in s.checkpoints)}]"
            for s in schedules
        ]
        texts.extend(
            f"{m.action_id}（修改）：{m.start}→{m.expected_end}ms"
            f" 节点[{', '.join(str(c) for c in m.checkpoints)}]"
            for m in modified_schedules
        )
        return tuple(texts)

    # ---- 留痕 ----

    def _trace_stage(
        self,
        stage: str,
        request: TranslationRequest,
        summary: str,
        *,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        details: dict[str, TraceValue] | None = None,
    ) -> None:
        """成功链留痕；追踪器自身的失败必须传播（不静默假成功）。"""
        if self._tracer is None:
            return
        response = request.response
        self._tracer.trace(
            TraceRecord(
                stage=stage,
                cycle_id=response.cycle_id,
                simulation_time=request.current_time,
                summary=summary,
                role_id=response.role_id,
                response_id=response.response_id,
                action_ids=tuple(item.action_id for item in response.actions),
                inputs=inputs,
                outputs=outputs,
                world_state_version=request.world_state_version,
                details=details or {},
            )
        )

    def _trace_failure(
        self,
        request: TranslationRequest,
        error: BaseException,
        *,
        prompt: str,
        raw: str | None,
        error_kind: str,
    ) -> None:
        """保留完整失败现场；日志故障可诊断且不替换原始异常。"""
        if self._tracer is None:
            return
        response = request.response
        message = f"{type(error).__name__}: {error}"
        try:
            self._tracer.trace(
                TraceRecord(
                    stage="translator.error",
                    cycle_id=response.cycle_id,
                    simulation_time=request.current_time,
                    summary=f"翻译失败（{error_kind}）",
                    role_id=response.role_id,
                    response_id=response.response_id,
                    action_ids=tuple(item.action_id for item in response.actions),
                    inputs=(prompt,),
                    outputs=(raw,) if raw else (),
                    world_state_version=request.world_state_version,
                    details={"error_kind": error_kind, "raw_output": raw},
                    error=message,
                )
            )
        except Exception as trace_error:
            note = f"行动翻译器错误日志写入失败：{trace_error}"
            error.add_note(note)
