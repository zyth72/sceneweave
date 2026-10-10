#!/usr/bin/env python3
"""独立翻译器探针：同动作不同角色身体条件下的生产 ModelTranslator 时间安排。

本探针独立于真实模拟：只调用 ``ModelTranslator.translate`` 的结构化
``ActionPlan`` 路径，复用生产提示（``_build_step_prompt``）与生产解析器
（``_parse_step_schedules``），不启动 World / 引擎 / 完整循环，也不替角色
选择动作、不传私有意向。角色身体条件全部放在 World 提供的
``ObjectiveContext.conditions``（只读客观条件）；模型只产生各动作的时间
安排，实际完成与否仍由 World 负责，本探针不做物理实测。

采样使用 AsyncOpenAI 官方兼容 Chat Completions 接口，逐样本捕获：
exact prompt、完整安全请求体（不含请求头）、原始响应、usage、模型返回
标识、finish_reason、API 墙钟耗时与异常。密钥只经 ``dotenv_values``
私密读取，值绝不写入输出；拒绝 ``OPENAI_CUSTOM_HEADERS``。

默认 deepseek / qwen 可配置 model、base_url、env-key 名（不暴露值）；
思考开关 ``--thinking`` 默认关闭，开启时 deepseek 发送
``thinking={"type": "enabled"}`` 且 ``reasoning_effort="high"``、qwen 发送
``enable_thinking=true``；``--max-tokens`` 默认 2048，可显式调整；两者都
JSON object、temperature=0、max_retries=0、timeout 60s、并发默认 6、
重复默认 2。全部样本落盘 samples.jsonl，另写 manifest.json、summary.json、
report.md；失败也留下可分析完整记录，联网不通不会静默假成功。

用法示例：
    uv run --locked python scripts/probe_translator.py --dry-run
    uv run --locked python scripts/probe_translator.py --provider both --repeats 2
    uv run --locked python scripts/probe_translator.py --thinking --max-tokens 8192 --timeout 120
    uv run --locked python scripts/probe_translator.py --case move-5m-1ms --case reach-tired
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    OpenAIError,
)

from sceneweave.domain.action_plan import (
    ActionPlan,
    PlanAction,
    StepAction,
    atomic_action_from_json,
)
from sceneweave.domain.adjudication import ObjectiveContext
from sceneweave.domain.identity import (
    ActionId,
    CycleId,
    ResponseId,
    RoleId,
    WorldStateVersion,
)
from sceneweave.domain.messages import ActionPhase, RoleResponse
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.translation import ActionArrangement, TranslationRequest
from sceneweave.engine.registration import RegistrationIssue, registration_issues
from sceneweave.translator.model_translator import ModelTranslator, ModelTranslatorError

PROBE_ID = "independent-translator-probe"
PROBE_CASES_PATH = Path(__file__).with_name("probe_translator_cases.json")
DEFAULT_OUTDIR = Path("sceneweave-runs/translator-probe")
MAX_TOKENS = 2048
TEMPERATURE = 0.0
DEFAULT_CONCURRENCY = 6
DEFAULT_REPEATS = 2
DEFAULT_TIMEOUT = 60.0
_CUSTOM_HEADERS_ENV = "OPENAI_CUSTOM_HEADERS"
_KNOWN_PARAMS = frozenset(
    {"model", "messages", "stream", "n", "temperature", "max_tokens", "response_format"}
)


class ProbeError(ValueError):
    """探针输入/配置错误；消息不包含密钥或文件值。"""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """供应商配置；只保存可公开的配置与环境变量名，不保存密钥值。"""

    name: str
    model: str
    base_url: str
    api_key_env: str
    extra_body: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "extra_body": self.extra_body,
        }


DEFAULT_PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    ProviderConfig(
        name="qwen",
        model="qwen3.7-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        extra_body={"enable_thinking": False},
    ),
)


@dataclass(frozen=True, slots=True)
class ProbeCase:
    """一个探针用例：固定 ActionId/RoleId 保证同 case 的所有模型/重复拿到同一提示。"""

    id: str
    title: str
    role_id: str
    cycle_id: str
    response_id: str
    current_time: int
    world_state_version: int
    scene: str
    conditions: tuple[str, ...]
    plan: tuple[dict[str, Any], ...]
    arrangements: tuple[dict[str, Any], ...]
    rules: dict[str, Any]
    scoring: str


# ---------------------------------------------------------------------------
# 用例加载
# ---------------------------------------------------------------------------


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{field} 必须是非空字符串")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbeError(f"{field} 必须是非负整数")
    return value


def _load_case(raw: dict[str, Any]) -> ProbeCase:
    case_id = _require_text(raw.get("id"), "id")
    title = _require_text(raw.get("title"), f"{case_id}.title")
    scene = _require_text(raw.get("scene"), f"{case_id}.scene")
    role_id = _require_text(raw.get("role_id"), f"{case_id}.role_id")
    cycle_id = _require_text(raw.get("cycle_id"), f"{case_id}.cycle_id")
    response_id = _require_text(raw.get("response_id"), f"{case_id}.response_id")
    current_time = _require_int(raw.get("current_time"), f"{case_id}.current_time")
    version = _require_int(raw.get("world_state_version"), f"{case_id}.world_state_version")
    conditions_raw = raw.get("conditions")
    if (
        not isinstance(conditions_raw, list)
        or not conditions_raw
        or not all(isinstance(item, str) and item for item in conditions_raw)
    ):
        raise ProbeError(f"{case_id}.conditions 必须是非空字符串数组")
    plan_raw = raw.get("plan")
    if not isinstance(plan_raw, list):
        raise ProbeError(f"{case_id}.plan 必须是数组")
    for index, entry in enumerate(plan_raw):
        if not isinstance(entry, dict):
            raise ProbeError(f"{case_id}.plan[{index}] 必须是对象")
        _require_text(entry.get("action_id"), f"{case_id}.plan[{index}].action_id")
        _require_text(entry.get("kind"), f"{case_id}.plan[{index}].kind")
        deps = entry.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) and d for d in deps):
            raise ProbeError(f"{case_id}.plan[{index}].depends_on 必须是非空字符串数组")
    arrangements_raw = raw.get("arrangements")
    if not isinstance(arrangements_raw, list):
        raise ProbeError(f"{case_id}.arrangements 必须是数组")
    for index, item in enumerate(arrangements_raw):
        if not isinstance(item, dict):
            raise ProbeError(f"{case_id}.arrangements[{index}] 必须是对象")
        for key in ("action_id", "phase", "description", "start", "expected_end"):
            if key not in item:
                raise ProbeError(f"{case_id}.arrangements[{index}] 缺少字段 {key}")
    rules = raw.get("rules")
    if not isinstance(rules, dict) or not _require_text(rules.get("type"), f"{case_id}.rules.type"):
        raise ProbeError(f"{case_id}.rules 必须包含非空 type")
    scoring = _require_text(raw.get("scoring"), f"{case_id}.scoring")
    return ProbeCase(
        id=case_id,
        title=title,
        role_id=role_id,
        cycle_id=cycle_id,
        response_id=response_id,
        current_time=current_time,
        world_state_version=version,
        scene=scene,
        conditions=tuple(conditions_raw),
        plan=tuple(plan_raw),
        arrangements=tuple(arrangements_raw),
        rules=rules,
        scoring=scoring,
    )


def load_cases(path: Path = PROBE_CASES_PATH) -> list[ProbeCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ProbeError(f"用例文件格式错误：{path}")
    cases = [_load_case(raw) for raw in data["cases"]]
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ProbeError("用例 id 必须唯一")
    if not cases:
        raise ProbeError("用例文件不得为空")
    return cases


def select_cases(cases: list[ProbeCase], case_ids: list[str] | None) -> list[ProbeCase]:
    if not case_ids:
        return cases
    wanted = list(dict.fromkeys(case_ids))
    known = {case.id for case in cases}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise ProbeError(f"未知用例：{sorted(unknown)}")
    by_id = {case.id: case for case in cases}
    return [by_id[item] for item in wanted]


def case_to_dict(case: ProbeCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "title": case.title,
        "role_id": case.role_id,
        "cycle_id": case.cycle_id,
        "response_id": case.response_id,
        "current_time": case.current_time,
        "world_state_version": case.world_state_version,
        "scene": case.scene,
        "conditions": list(case.conditions),
        "plan": [dict(item) for item in case.plan],
        "arrangements": [dict(item) for item in case.arrangements],
        "rules": dict(case.rules),
        "scoring": case.scoring,
    }


# ---------------------------------------------------------------------------
# 复用生产翻译器：构造 TranslationRequest
# ---------------------------------------------------------------------------


def build_arrangements(case: ProbeCase) -> tuple[ActionArrangement, ...]:
    result: list[ActionArrangement] = []
    for item in case.arrangements:
        result.append(
            ActionArrangement(
                action_id=ActionId(item["action_id"]),
                phase=ActionPhase(item["phase"]),
                description=item["description"],
                start=SimulationTime(_require_int(item["start"], f"{case.id}.arrangement.start")),
                expected_end=SimulationTime(
                    _require_int(item["expected_end"], f"{case.id}.arrangement.expected_end")
                ),
                checkpoints=tuple(
                    SimulationTime(_require_int(c, f"{case.id}.arrangement.checkpoint"))
                    for c in item.get("checkpoints", [])
                ),
            )
        )
    return tuple(result)


def build_plan_and_steps(case: ProbeCase) -> tuple[ActionPlan, tuple[StepAction, ...]]:
    plan_items: list[PlanAction] = []
    steps: list[StepAction] = []
    for index, entry in enumerate(case.plan):
        payload = {k: v for k, v in entry.items() if k not in ("action_id", "depends_on")}
        action = atomic_action_from_json(payload, f"{case.id}.plan[{index}]")
        action_id = ActionId(entry["action_id"])
        raw_deps = tuple(str(item) for item in entry.get("depends_on", []))
        plan_items.append(PlanAction(action=action, action_id=action_id, depends_on=raw_deps))
        steps.append(StepAction(action_id, action, tuple(ActionId(item) for item in raw_deps)))
    return ActionPlan(actions=tuple(plan_items)), tuple(steps)


def build_request(case: ProbeCase) -> TranslationRequest:
    """确定性构造与生产一致的结构化 ActionPlan 翻译请求（固定身份）。"""
    plan, steps = build_plan_and_steps(case)
    role_id = RoleId(case.role_id)
    version = WorldStateVersion(case.world_state_version)
    response = RoleResponse(
        cycle_id=CycleId(case.cycle_id),
        role_id=role_id,
        response_id=ResponseId(case.response_id),
        actions=steps,
        plan=plan,
    )
    return TranslationRequest(
        response=response,
        current_time=SimulationTime(case.current_time),
        world_state_version=version,
        arrangements=build_arrangements(case),
        objective_context=ObjectiveContext(version, role_id, case.conditions),
    )


class _NullSampler:
    """永不真正调用的占位采样器；只用于离线生成提示。"""

    async def sample(self, prompt: str, *, component: str) -> str:  # pragma: no cover
        raise AssertionError("占位采样器不应被调用")


def build_probe_prompt(case: ProbeCase) -> str:
    """用生产 ``ModelTranslator._build_step_prompt`` 离线生成完整提示，不调用模型。

    与真实运行走同一入口：相同 case 的所有模型/重复必然拿到逐字一致提示。
    """
    request = build_request(case)
    specs = ModelTranslator._step_specs(request.response)
    arrangement_ids = {item.action_id for item in request.arrangements}
    step_new = [item for item in specs if item.action_id not in arrangement_ids]
    if not step_new:
        return ""
    translator = ModelTranslator(lambda _role: _NullSampler())
    return translator._build_step_prompt(request, step_new)


# ---------------------------------------------------------------------------
# 探针采样器（AsyncOpenAI 官方兼容接口 + 全证据捕获）
# ---------------------------------------------------------------------------


def request_body(
    provider: ProviderConfig, prompt: str, *, max_tokens: int = MAX_TOKENS
) -> dict[str, Any]:
    """完整安全请求体：不含请求头、不含密钥值。"""
    return {
        "model": provider.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "n": 1,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        **provider.extra_body,
    }


def sdk_call_kwargs(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """把已保存请求体拆成 SDK 具名参数与 extra_body（供应商专用字段）。"""
    kwargs = {k: v for k, v in body.items() if k in _KNOWN_PARAMS}
    extra = {k: v for k, v in body.items() if k not in _KNOWN_PARAMS}
    return kwargs, extra


def usage_record(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    return dict(usage)


def cache_ratio(usage: dict[str, Any] | None) -> float | None:
    """cache 命中占 prompt 比例；缺失时返回 None（未知，不作 0）。"""
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    if not isinstance(prompt, int) or prompt <= 0 or not isinstance(hit, int) or hit < 0:
        return None
    return hit / prompt


class ProbeSampler:
    """探针采样器：AsyncOpenAI Chat Completions，逐字段捕获证据到 Sample。"""

    def __init__(
        self,
        client: AsyncOpenAI,
        provider: ProviderConfig,
        sample: Sample,
        *,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._client = client
        self._provider = provider
        self._sample = sample
        self._max_tokens = max_tokens

    async def sample(self, prompt: str, *, component: str) -> str:
        sample = self._sample
        sample.component = component
        sample.prompt = prompt
        sample.request_body = request_body(self._provider, prompt, max_tokens=self._max_tokens)
        sample.calls_made += 1
        started = time.monotonic()
        try:
            kwargs, extra = sdk_call_kwargs(sample.request_body)
            response = await self._client.chat.completions.create(
                **kwargs, extra_body=extra or None
            )
        except APITimeoutError:
            sample.timed_out = True
            sample.call_error = "APITimeoutError"
            sample.exit_code = 1
            raise
        except APIStatusError as error:
            sample.status_code = error.status_code
            sample.call_error = f"APIStatusError: HTTP {error.status_code}"
            sample.exit_code = 1
            raise
        except APIConnectionError:
            sample.call_error = "APIConnectionError"
            sample.exit_code = 1
            raise
        except OpenAIError as error:
            sample.call_error = f"OpenAIError:{type(error).__name__}"
            sample.exit_code = 1
            raise
        finally:
            sample.elapsed_seconds = time.monotonic() - started
        sample.raw_response = response.model_dump(mode="json")
        choice = response.choices[0]
        message = choice.message
        sample.content = message.content or ""
        sample.reasoning = str(getattr(message, "reasoning_content", "") or "")
        sample.finish_reason = choice.finish_reason
        sample.model_returned = getattr(response, "model", None)
        sample.response_id = getattr(response, "id", None)
        sample.usage = usage_record(
            response.usage.model_dump(mode="json") if response.usage is not None else None
        )
        sample.exit_code = 0
        return sample.content


# ---------------------------------------------------------------------------
# 样本记录与单次运行
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """一次 (case, provider, repeat) 的完整可分析记录。"""

    task_id: str
    case_id: str
    provider: str
    model: str
    repeat: int
    component: str | None = None
    prompt: str = ""
    request_body: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    model_returned: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    status_code: int | None = None
    call_error: str | None = None
    exit_code: int = 0
    calls_made: int = 0
    schema_ok: bool = True
    parse_error: str | None = None
    schedules: list[dict[str, Any]] = field(default_factory=list)
    registration_issues: list[dict[str, Any]] = field(default_factory=list)
    rule_type: str | None = None
    rule_passed: bool | None = None
    rule_checks: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "case_id": self.case_id,
            "provider": self.provider,
            "model": self.model,
            "repeat": self.repeat,
            "component": self.component,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "request_body": self.request_body,
            "raw_response": self.raw_response,
            "content": self.content,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "model_returned": self.model_returned,
            "response_id": self.response_id,
            "usage": self.usage,
            "cache_ratio": cache_ratio(self.usage),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "timed_out": self.timed_out,
            "status_code": self.status_code,
            "call_error": self.call_error,
            "exit_code": self.exit_code,
            "calls_made": self.calls_made,
            "schema_ok": self.schema_ok,
            "parse_error": self.parse_error,
            "schedules": self.schedules,
            "registration_issues": self.registration_issues,
            "rule_type": self.rule_type,
            "rule_passed": self.rule_passed,
            "rule_checks": self.rule_checks,
            "note": self.note,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Sample:
        return cls(
            task_id=record["task_id"],
            case_id=record["case_id"],
            provider=record["provider"],
            model=record["model"],
            repeat=record["repeat"],
            component=record.get("component"),
            prompt=record.get("prompt", ""),
            request_body=record.get("request_body"),
            raw_response=record.get("raw_response"),
            content=record.get("content", ""),
            reasoning=record.get("reasoning", ""),
            finish_reason=record.get("finish_reason"),
            model_returned=record.get("model_returned"),
            response_id=record.get("response_id"),
            usage=record.get("usage"),
            elapsed_seconds=record.get("elapsed_seconds", 0.0),
            timed_out=record.get("timed_out", False),
            status_code=record.get("status_code"),
            call_error=record.get("call_error"),
            exit_code=record.get("exit_code", 0),
            calls_made=record.get("calls_made", 0),
            schema_ok=record.get("schema_ok", True),
            parse_error=record.get("parse_error"),
            schedules=record.get("schedules", []),
            registration_issues=record.get("registration_issues", []),
            rule_type=record.get("rule_type"),
            rule_passed=record.get("rule_passed"),
            rule_checks=record.get("rule_checks", []),
            note=record.get("note"),
        )


def _schedule_record(schedule: Any) -> dict[str, Any]:
    return {
        "action_id": str(schedule.action_id),
        "commit_group_id": str(schedule.commit_group_id),
        "start": int(schedule.start),
        "expected_end": int(schedule.expected_end),
        "checkpoints": [int(item) for item in schedule.checkpoints],
    }


def _duration(schedule: dict[str, Any]) -> int:
    return int(schedule["expected_end"]) - int(schedule["start"])


def score_sample(sample: Sample, case: ProbeCase) -> None:
    """在解析与登记校验之外，独立执行 probe 专有评分规则。"""
    rules = case.rules
    rtype = str(rules.get("type", ""))
    sample.rule_type = rtype
    checks = sample.rule_checks

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    by_id: dict[str, dict[str, Any]] = {item["action_id"]: item for item in sample.schedules}

    if rtype == "no_calls":
        add("zero_calls", sample.calls_made == 0, f"calls_made={sample.calls_made}")
        add("empty_schedules", len(sample.schedules) == 0, f"schedules={len(sample.schedules)}")
    elif rtype == "exact_duration":
        expected = int(rules["expected_ms"])
        tolerance = int(rules.get("tolerance_ms", 0))
        if len(sample.schedules) != 1:
            add("single_schedule", False, f"期望 1 个安排，得到 {len(sample.schedules)}")
        else:
            schedule = sample.schedules[0]
            duration = _duration(schedule)
            delta = abs(duration - expected)
            add("single_schedule", True, f"action_id={schedule['action_id']}")
            add(
                "duration",
                delta <= tolerance,
                f"duration={duration}ms 期望 {expected}ms 误差 {delta}ms ≤ {tolerance}ms",
            )
            min_ms = rules.get("min_duration_ms")
            if min_ms is not None:
                add(
                    "min_duration",
                    duration >= int(min_ms),
                    f"duration={duration}ms ≥ {min_ms}ms（防止改回直线/缩短距离）",
                )
    elif rtype == "direction":
        add("single_schedule", len(sample.schedules) == 1, "只检查安排完整性；身体耗时无黄金值")
        sample.note = "物理耗时仅做成对诊断，不计入确定性时长规则满足率"
    elif rtype == "schedule_consistency":
        ok_all = True
        for entry in case.plan:
            deps = [str(item) for item in entry.get("depends_on", [])]
            if not deps:
                continue
            action_id = entry["action_id"]
            dependent_schedule = by_id.get(action_id)
            if dependent_schedule is None:
                add(f"ordering_{action_id}", False, "缺少安排")
                ok_all = False
                continue
            for dep in deps:
                dep_schedule = by_id.get(dep)
                if dep_schedule is None:
                    add(f"ordering_{action_id}->{dep}", False, f"依赖 {dep} 缺少安排")
                    ok_all = False
                    continue
                ok = int(dependent_schedule["start"]) >= int(dep_schedule["expected_end"])
                ok_all = ok_all and ok
                add(
                    f"ordering_{action_id}->{dep}",
                    ok,
                    f"后置 start={dependent_schedule['start']}ms ≥ 前置 expected_end={dep_schedule['expected_end']}ms",
                )
        if not checks:
            add("plan", True, "无依赖，无需顺序检查")
    elif rtype == "no_retranslate":
        only = ActionId(rules["only_action_id"])
        got = set(by_id)
        add("only_new_action", got == {only}, f"得到 {sorted(str(i) for i in got)}")
        arrangement_ids = {ActionId(item["action_id"]) for item in case.arrangements}
        overlap = got & arrangement_ids
        add(
            "no_retranslate",
            not overlap,
            f"既有安排被重排：{sorted(str(i) for i in overlap) or '无'}",
        )
    elif rtype == "independent":
        expected_ids = {ActionId(entry["action_id"]) for entry in case.plan}
        got = set(by_id)
        add(
            "covers_all_actions",
            got == expected_ids,
            f"期望 {sorted(str(i) for i in expected_ids)} 得到 {sorted(str(i) for i in got)}",
        )
        ids = sorted(got, key=str)
        if len(ids) == 2:
            first, second = by_id[ids[0]], by_id[ids[1]]
            chained = int(first["expected_end"]) <= int(second["start"]) or int(
                second["expected_end"]
            ) <= int(first["start"])
            sample.note = "串行" if chained else "交叠"
    elif rtype == "offset":
        if len(sample.schedules) != 1:
            add("single_schedule", False, f"期望 1 个安排，得到 {len(sample.schedules)}")
        else:
            schedule = sample.schedules[0]
            add(
                "start_after_now",
                int(schedule["start"]) > case.current_time,
                f"start={schedule['start']}ms > now={case.current_time}ms",
            )
    else:
        add("unknown_rule", False, f"未知规则类型：{rtype}")

    if checks:
        sample.rule_passed = all(item["passed"] for item in checks)


async def run_one(
    client: Any,
    provider: ProviderConfig,
    case: ProbeCase,
    repeat: int,
    *,
    sample_record: Sample | None = None,
    max_tokens: int = MAX_TOKENS,
) -> Sample:
    """一次样本运行：复用生产 ModelTranslator.translate，捕获全部证据。"""
    task_id = f"{case.id}/{provider.name}/r{repeat}"
    sample = sample_record or Sample(
        task_id=task_id,
        case_id=case.id,
        provider=provider.name,
        model=provider.model,
        repeat=repeat,
    )
    sampler = ProbeSampler(client, provider, sample, max_tokens=max_tokens)
    translator = ModelTranslator(lambda _role: sampler)
    request = build_request(case)
    try:
        result = await translator.translate(request)
    except ModelTranslatorError as error:
        sample.schema_ok = False
        sample.parse_error = str(error)
        if sample.exit_code == 0:
            sample.exit_code = 1
    except asyncio.CancelledError:
        raise
    except Exception as error:
        # 采样器已记录 call_error（API/连接/超时）；保留失败样本，不静默假成功。
        sample.schema_ok = False
        if not sample.call_error:
            sample.call_error = type(error).__name__
        if sample.exit_code == 0:
            sample.exit_code = 1
    else:
        sample.schedules = [_schedule_record(item) for item in result.schedules]
        issues: tuple[RegistrationIssue, ...] = registration_issues(
            request.response,
            result,
            current_time=request.current_time,
            arrangements=request.arrangements,
        )
        sample.registration_issues = [
            {"text": item.text, "structural": item.structural} for item in issues
        ]
    score_sample(sample, case)
    return sample


async def run_tasks(
    clients: dict[str, AsyncOpenAI],
    tasks: list[tuple[ProbeCase, ProviderConfig, int]],
    outdir: Path,
    concurrency: int,
    *,
    max_tokens: int = MAX_TOKENS,
) -> list[Sample]:
    """并发运行全部任务；逐样本落盘 samples.jsonl，失败也保留完整记录。"""
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    samples: list[Sample] = []
    samples_path = outdir / "samples.jsonl"
    started_batch = time.monotonic()

    async def run(case: ProbeCase, provider: ProviderConfig, repeat: int) -> None:
        client = clients[provider.name]
        sample = Sample(
            f"{case.id}/{provider.name}/r{repeat}", case.id, provider.name, provider.model, repeat
        )
        try:
            async with semaphore:
                await run_one(
                    client, provider, case, repeat, sample_record=sample, max_tokens=max_tokens
                )
        except asyncio.CancelledError:
            sample.call_error = "CancelledError"
            sample.schema_ok = False
            sample.exit_code = 1
            sample.note = "调用被取消；是否已由上游计费未知"
            samples.append(sample)
            # 文件追加为同步小操作，不在取消路径再等待异步锁。
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
            raise
        samples.append(sample)
        async with write_lock:
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
        status = (
            "ok"
            if sample.call_error is None and sample.schema_ok and not sample.registration_issues
            else ("api-error" if sample.call_error else "invalid")
        )
        print(
            f"[{len(samples)}/{len(tasks)}] {sample.task_id} {status} "
            f"{sample.elapsed_seconds:.1f}s",
            flush=True,
        )

    await asyncio.gather(*(run(case, provider, repeat) for case, provider, repeat in tasks))
    (outdir / "timing.json").write_text(
        json.dumps(
            {
                "batch_elapsed_seconds": time.monotonic() - started_batch,
                "concurrency_limit": concurrency,
                "samples": len(samples),
                "calls": sum(sample.calls_made for sample in samples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return samples


# ---------------------------------------------------------------------------
# 统计与聚合
# ---------------------------------------------------------------------------


def _paid(sample: Sample) -> bool:
    return sample.calls_made > 0


def _api_ok(sample: Sample) -> bool:
    return _paid(sample) and not sample.timed_out and sample.call_error is None


def _constraint_ok(sample: Sample) -> bool:
    return sample.schema_ok and not sample.registration_issues and sample.rule_passed is True


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _usage_tokens(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not usage:
        return {
            "input": None,
            "output": None,
            "reasoning": None,
            "cache_read": None,
            "cache_write": None,
        }
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "input": usage.get("prompt_tokens"),
        "output": usage.get("completion_tokens"),
        "reasoning": details.get("reasoning_tokens"),
        "cache_read": usage.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens")),
        "cache_write": usage.get(
            "prompt_cache_write_tokens", prompt_details.get("cache_write_tokens")
        ),
    }


def _sum_tokens(samples: list[Sample], key: str) -> int | None:
    values = [
        value for sample in samples if (value := _usage_tokens(sample.usage)[key]) is not None
    ]
    return sum(values) if values else None


def _cache_aggregate(samples: list[Sample]) -> dict[str, Any]:
    """cache 命中占 prompt 比例；缺失是未知，不作 0。"""
    pairs = []
    for sample in samples:
        ratio = cache_ratio(sample.usage)
        if ratio is None or sample.usage is None:
            continue
        prompt = sample.usage.get("prompt_tokens")
        details = sample.usage.get("prompt_tokens_details") or {}
        hit = sample.usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
        assert isinstance(prompt, int) and isinstance(hit, int)
        pairs.append((int(prompt), int(hit)))
    if not pairs:
        return {"known_samples": 0, "value": None, "unknown": True}
    total_prompt = sum(item[0] for item in pairs)
    total_hit = sum(item[1] for item in pairs)
    return {
        "known_samples": len(pairs),
        "value": total_hit / total_prompt if total_prompt > 0 else None,
        "unknown": False,
    }


def _schedule_signature(sample: Sample) -> tuple[Any, ...]:
    return tuple(
        (
            item["action_id"],
            item["start"],
            item["expected_end"],
            tuple(item.get("checkpoints", [])),
        )
        for item in sorted(sample.schedules, key=lambda item: item["action_id"])
    )


def _repeat_stability(samples: list[Sample]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[int, tuple[Any, ...]]] = {}
    for sample in samples:
        if not sample.schema_ok or not sample.schedules:
            continue
        groups.setdefault((sample.case_id, sample.provider), {})[sample.repeat] = (
            _schedule_signature(sample)
        )
    stable = 0
    groups_with_repeats = 0
    rows: dict[str, Any] = {}
    for (case_id, provider), repeats in sorted(groups.items()):
        if len(repeats) < 2:
            continue
        groups_with_repeats += 1
        consistent = len(set(repeats.values())) == 1
        if consistent:
            stable += 1
        rows[f"{case_id}/{provider}"] = {
            "repeats": sorted(repeats),
            "consistent": consistent,
        }
    return {
        "groups_with_>=2_repeats": groups_with_repeats,
        "stable_groups": stable,
        "stability_rate": _rate(stable, groups_with_repeats),
        "rows": rows,
    }


def _schedules_for(samples: list[Sample], case_id: str, provider: str) -> list[Sample]:
    return [
        sample
        for sample in samples
        if sample.case_id == case_id
        and sample.provider == provider
        and sample.schema_ok
        and _api_ok(sample)
        and not sample.registration_issues
    ]


def _median_duration(samples: list[Sample]) -> float | None:
    durations = [
        _duration(item)
        for sample in samples
        if sample.schema_ok and len(sample.schedules) == 1
        for item in sample.schedules
    ]
    return _median([float(item) for item in durations])


def _median_start(samples: list[Sample]) -> float | None:
    starts = [
        int(sample.schedules[0]["start"])
        for sample in samples
        if sample.schema_ok and len(sample.schedules) == 1
    ]
    return _median([float(item) for item in starts])


def pairwise_diagnostics(samples: list[Sample]) -> dict[str, Any]:
    """成对角色/距离/障碍/偏移方向诊断；只描述现象，不伪造准确率。"""
    result: dict[str, Any] = {}

    def pair(name: str, providers: list[str], rows: dict[str, Any]) -> None:
        result[name] = {
            "providers": providers,
            "rows": rows,
            "note": "方向诊断使用各样本执行段时长（expected_end−start）；普通物理耗时没有实测黄金值，只报告方向与重复方差。",
            "registration_valid_rule_failures_provider_total": {
                provider: sum(
                    sample.rule_passed is False
                    for sample in samples
                    if sample.provider == provider
                    and sample.schema_ok
                    and _api_ok(sample)
                    and not sample.registration_issues
                )
                for provider in providers
            },
        }

    providers = sorted({sample.provider for sample in samples})
    # 距离成对：1m / 5m / 10m 恒速 1m/s。
    distance_rows: dict[str, Any] = {}
    for provider in providers:
        distance_rows[provider] = {
            case_id: _median_duration(_schedules_for(samples, case_id, provider))
            for case_id in ("move-1m-1ms", "move-5m-1ms", "move-10m-1ms")
        }
    pair("distance_scale", providers, distance_rows)
    # 同路径速度成对：5m 2m/s vs 5m 1m/s。
    speed_rows: dict[str, Any] = {}
    for provider in providers:
        fast = _median_duration(_schedules_for(samples, "move-5m-2ms", provider))
        slow = _median_duration(_schedules_for(samples, "move-5m-1ms", provider))
        speed_rows[provider] = {
            "fast_2mps_median_ms": fast,
            "normal_1mps_median_ms": slow,
            "shorter_at_faster_speed": fast < slow
            if fast is not None and slow is not None
            else None,
        }
    pair("speed_same_path", providers, speed_rows)
    # 伤腿成对：伤腿 0.5m/s vs 正常 1m/s（同一 5 米路径）。
    injured_rows: dict[str, Any] = {}
    for provider in providers:
        injured = _median_duration(_schedules_for(samples, "move-5m-injured-0.5ms", provider))
        normal = _median_duration(_schedules_for(samples, "move-5m-1ms", provider))
        injured_rows[provider] = {
            "injured_0.5mps_median_ms": injured,
            "normal_1mps_median_ms": normal,
            "injured_takes_longer": injured > normal
            if injured is not None and normal is not None
            else None,
        }
    pair("injured_same_path", providers, injured_rows)
    # 障碍成对：已选绕行 6 米 vs 直线 2 米。
    obstacle_rows: dict[str, Any] = {}
    for provider in providers:
        detour = _median_duration(_schedules_for(samples, "move-obstacle-detour-6m", provider))
        obstacle_rows[provider] = {
            "detour_6m_median_ms": detour,
            "straight_line_2m_ms": _median_duration(
                _schedules_for(samples, "move-obstacle-straight-2m", provider)
            ),
            "uses_detour_not_shortcut": detour >= 4500 if detour is not None else None,
        }
    pair("obstacle_detour", providers, obstacle_rows)
    # 姿态/疲劳成对：疲劳伸手 vs 精力充沛伸手。
    reach_rows: dict[str, Any] = {}
    for provider in providers:
        tired = _median_duration(_schedules_for(samples, "reach-tired", provider))
        standing = _median_duration(_schedules_for(samples, "reach-standing", provider))
        reach_rows[provider] = {
            "tired_median_ms": tired,
            "standing_median_ms": standing,
            "tired_slower": tired > standing
            if tired is not None and standing is not None
            else None,
        }
    pair("reach_body_condition", providers, reach_rows)
    posture_rows = {}
    for provider in providers:
        seated = _median_duration(_schedules_for(samples, "reach-seated", provider))
        standing = _median_duration(_schedules_for(samples, "reach-standing", provider))
        posture_rows[provider] = {
            "seated_median_ms": seated,
            "standing_median_ms": standing,
            "difference_ms": seated - standing
            if seated is not None and standing is not None
            else None,
        }
    pair("reach_posture", providers, posture_rows)
    # 熟练/生疏成对：同一台词 4 字/秒 vs 1.5 字/秒。
    say_rows: dict[str, Any] = {}
    for provider in providers:
        novice = _median_duration(_schedules_for(samples, "say-novice-1.5cps", provider))
        proficient = _median_duration(_schedules_for(samples, "say-proficient-4cps", provider))
        say_rows[provider] = {
            "novice_1.5cps_median_ms": novice,
            "proficient_4cps_median_ms": proficient,
            "novice_slower": novice > proficient
            if novice is not None and proficient is not None
            else None,
        }
    pair("say_proficiency", providers, say_rows)
    # 当前时刻偏移成对：t0 vs t100000。
    offset_rows: dict[str, Any] = {}
    for provider in providers:
        t0_start = _median_start(_schedules_for(samples, "offset-same-scene-t0", provider))
        t100_start = _median_start(_schedules_for(samples, "offset-same-scene-t100000", provider))
        t0_duration = _median_duration(_schedules_for(samples, "offset-same-scene-t0", provider))
        t100_duration = _median_duration(
            _schedules_for(samples, "offset-same-scene-t100000", provider)
        )
        delta = (t100_start - t0_start) if t0_start is not None and t100_start is not None else None
        offset_rows[provider] = {
            "t0_start_median_ms": t0_start,
            "t100_start_median_ms": t100_start,
            "start_delta_ms": delta,
            "expected_delta_ms": 100000,
            "duration_delta_abs_ms": (
                abs(t0_duration - t100_duration)
                if t0_duration is not None and t100_duration is not None
                else None
            ),
            "start_moved_by_offset": abs(delta - 100000) <= 500 if delta is not None else None,
        }
    pair("time_offset", providers, offset_rows)
    return result


def aggregate(
    samples: list[Sample], cases: list[ProbeCase], providers: list[ProviderConfig]
) -> dict[str, Any]:
    paid = [sample for sample in samples if _paid(sample)]
    api_ok = [sample for sample in paid if _api_ok(sample)]
    schema_ok = [sample for sample in paid if sample.schema_ok]
    constraint_ok = [sample for sample in paid if _constraint_ok(sample)]
    latency = [sample.elapsed_seconds for sample in samples if sample.elapsed_seconds > 0]
    usage_known = [sample for sample in paid if sample.usage]
    result: dict[str, Any] = {
        "probe": PROBE_ID,
        "total_samples": len(samples),
        "paid_calls": len(paid),
        "skipped_no_call": len(samples) - len(paid),
        "api_success": len(api_ok),
        "schema_success": len(schema_ok),
        "constraint_satisfied": len(constraint_ok),
        "exact_duration_rules": {
            "samples": sum(sample.rule_type == "exact_duration" for sample in api_ok),
            "satisfied": sum(
                sample.rule_type == "exact_duration"
                and sample.schema_ok
                and sample.rule_passed is True
                for sample in api_ok
            ),
            "note": "只检查显式时间规则，排程是否合法另计；不是物理实测准确率",
        },
        "paid_pass_rate": _rate(len(constraint_ok), len(paid)),
        "api_success_rate": _rate(len(api_ok), len(paid)),
        "schema_success_rate": _rate(len(schema_ok), len(paid)),
        "latency_seconds": {"median": _median(latency), "p95": _p95(latency)},
        "tokens": {
            "input": _sum_tokens(paid, "input"),
            "output": _sum_tokens(paid, "output"),
            "reasoning": _sum_tokens(paid, "reasoning"),
            "cache_read": _sum_tokens(paid, "cache_read"),
            "cache_write": _sum_tokens(paid, "cache_write"),
        },
        "cache_ratio": _cache_aggregate(paid),
        "usage_known_samples": len(usage_known),
        "repeat_stability": _repeat_stability(samples),
        "by_provider": {},
        "by_case": {},
        "pairwise": pairwise_diagnostics(samples),
        "errors": [
            {
                "task_id": sample.task_id,
                "call_error": sample.call_error,
                "timed_out": sample.timed_out,
                "status_code": sample.status_code,
                "parse_error": sample.parse_error,
                "registration_issues": sample.registration_issues,
                "rule_checks": [item for item in sample.rule_checks if not item["passed"]],
            }
            for sample in samples
            if sample.call_error
            or sample.timed_out
            or not sample.schema_ok
            or sample.registration_issues
            or (sample.rule_checks and not sample.rule_passed)
        ],
    }
    for provider in providers:
        subset = [sample for sample in samples if sample.provider == provider.name]
        paid_subset = [sample for sample in subset if _paid(sample)]
        latency_subset = [sample.elapsed_seconds for sample in subset if sample.elapsed_seconds > 0]
        result["by_provider"][provider.name] = {
            "model": provider.model,
            "samples": len(subset),
            "paid_calls": len(paid_subset),
            "api_success": sum(1 for sample in paid_subset if _api_ok(sample)),
            "schema_success": sum(1 for sample in paid_subset if sample.schema_ok),
            "constraint_satisfied": sum(1 for sample in paid_subset if _constraint_ok(sample)),
            "registration_satisfied": sum(
                sample.schema_ok and _api_ok(sample) and not sample.registration_issues
                for sample in paid_subset
            ),
            "exact_duration_samples": sum(
                sample.rule_type == "exact_duration" and _api_ok(sample) for sample in paid_subset
            ),
            "exact_duration_satisfied": sum(
                sample.rule_type == "exact_duration"
                and _api_ok(sample)
                and sample.schema_ok
                and sample.rule_passed is True
                for sample in paid_subset
            ),
            "paid_pass_rate": _rate(
                sum(1 for sample in paid_subset if _constraint_ok(sample)), len(paid_subset)
            ),
            "latency_seconds": {
                "median": _median(latency_subset),
                "p95": _p95(latency_subset),
            },
            "tokens": {
                "input": _sum_tokens(paid_subset, "input"),
                "output": _sum_tokens(paid_subset, "output"),
                "reasoning": _sum_tokens(paid_subset, "reasoning"),
                "cache_read": _sum_tokens(paid_subset, "cache_read"),
            },
            "cache_ratio": _cache_aggregate(paid_subset),
        }
    for case in cases:
        subset = [sample for sample in samples if sample.case_id == case.id]
        rows: dict[str, Any] = {}
        for provider in providers:
            provider_samples = [sample for sample in subset if sample.provider == provider.name]
            paid_provider = [sample for sample in provider_samples if _paid(sample)]
            rows[provider.name] = {
                "samples": len(provider_samples),
                "paid": len(paid_provider),
                "api_ok": sum(1 for sample in paid_provider if _api_ok(sample)),
                "schema_ok": sum(1 for sample in paid_provider if sample.schema_ok),
                "constraint_ok": sum(1 for sample in paid_provider if _constraint_ok(sample)),
                "rule_passed": sum(1 for sample in paid_provider if sample.rule_passed is True),
                "registration_issues": sum(
                    1 for sample in paid_provider if sample.registration_issues
                ),
                "median_duration_ms": _median_duration(
                    _schedules_for(samples, case.id, provider.name)
                ),
                "raw_median_duration_ms": _median_duration(provider_samples),
                "repeats": [
                    {
                        "repeat": sample.repeat,
                        "schema_ok": sample.schema_ok,
                        "rule_passed": sample.rule_passed,
                        "schedules": sample.schedules,
                        "registration_issues": sample.registration_issues,
                        "call_error": sample.call_error,
                        "note": sample.note,
                    }
                    for sample in sorted(provider_samples, key=lambda item: item.repeat)
                ],
            }
        result["by_case"][case.id] = {
            "title": case.title,
            "rule_type": case.rules.get("type"),
            "by_provider": rows,
        }
    return result


def unsuccessful(summary: dict[str, Any]) -> bool:
    paid = summary["paid_calls"]
    return bool(
        summary["api_success"] != paid
        or summary["schema_success"] != paid
        or summary["constraint_satisfied"] != paid
        or any(
            sample.get("call_error")
            or not sample.get("schema_ok")
            or sample.get("registration_issues")
            for sample in summary.get("errors", [])
        )
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fenced(text: str, language: str = "text") -> str:
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _actual_max_tokens(samples: list[Sample]) -> str:
    """报告展示真实请求体里的 max_tokens 集合，不写死常量。"""
    values = sorted(
        {
            int(sample.request_body["max_tokens"])
            for sample in samples
            if sample.request_body and "max_tokens" in sample.request_body
        }
    )
    return ",".join(str(value) for value in values) if values else "n/a"


def _schedule_line(schedule: dict[str, Any]) -> str:
    nodes = ", ".join(str(item) for item in schedule.get("checkpoints", []))
    return (
        f"{schedule['action_id']}：{schedule['start']}→{schedule['expected_end']}ms 节点[{nodes}]"
    )


def _render_case_evidence(
    index: int,
    case: ProbeCase,
    samples: list[Sample],
    providers: list[ProviderConfig],
) -> list[str]:
    subset = [sample for sample in samples if sample.case_id == case.id]
    prompts = {sample.prompt for sample in subset if sample.prompt}
    lines: list[str] = [
        f"### 场景 {index}：{case.title}",
        "",
        f"- 编号：`{case.id}`；角色：`{case.role_id}`；当前剧情时刻："
        f"{case.current_time} 毫秒；共同世界状态版本：{case.world_state_version}。",
        f"- 规则类型：`{case.rules.get('type')}`。",
        "",
        "**完整场景（中文）**",
        "",
        _fenced(case.scene),
        "",
        "**全部角色客观条件**（进入 World 提供的 ObjectiveContext.conditions，翻译专用）",
        "",
    ]
    lines += [f"- {item}" for item in case.conditions]
    lines += ["", "**结构化动作全量输入（原子 plan）**", ""]
    for entry in case.plan:
        suffix = (
            f" 依赖[{', '.join(str(item) for item in entry.get('depends_on', []))}]"
            if entry.get("depends_on")
            else ""
        )
        payload = {k: v for k, v in entry.items() if k not in ("action_id", "depends_on")}
        lines.append(f"- {entry['action_id']}：{json.dumps(payload, ensure_ascii=False)}{suffix}")
    lines += ["", "**既有安排**", ""]
    if case.arrangements:
        for item in case.arrangements:
            lines.append(
                f"- {item['action_id']}（{item['phase']}）：{item['description']} "
                f"{item['start']}→{item['expected_end']}ms 节点[{', '.join(str(c) for c in item.get('checkpoints', []))}]"
            )
    else:
        lines.append("- 无")
    lines += ["", "**评分条件**", "", case.scoring, ""]
    if prompts:
        shared = len(prompts) == 1
        lines += [
            "**完整生产提示词**（相同输入下所有模型/重复逐字一致"
            + ("；此处只展示一次" if shared else "，但本次不一致")
            + "）",
            "",
            _fenced(sorted(prompts)[0]),
            "",
        ]
        if not shared:
            for sample in subset:
                lines += ["**本次提示（逐字）**", "", _fenced(sample.prompt), ""]
    lines += [
        "**各模型各重复结果**",
        "",
        "| provider | repeat | schema | 登记校验 | 规则 | 时长(ms) | finish | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sample in sorted(subset, key=lambda item: (item.provider, item.repeat)):
        duration = _duration(sample.schedules[0]) if len(sample.schedules) == 1 else None
        registration = (
            "通过"
            if not sample.registration_issues
            else "有" + str(len(sample.registration_issues))
        )
        lines.append(
            f"| {sample.provider} | r{sample.repeat} | "
            f"{'是' if sample.schema_ok else '否'} | {registration} | "
            f"{'通过' if sample.rule_passed is True else ('失败' if sample.rule_passed is False else 'n/a')} | "
            f"{_fmt(duration)} | {sample.finish_reason or 'n/a'} | {_fmt(sample.elapsed_seconds)} |"
        )
    lines.append("")
    for sample in sorted(subset, key=lambda item: (item.provider, item.repeat)):
        lines += [
            f"#### {sample.provider} r{sample.repeat}（task `{sample.task_id}`）",
            "",
            "**API 耗时与使用量**",
            "",
            _fenced(
                json.dumps(
                    {
                        "model": sample.model,
                        "model_returned": sample.model_returned,
                        "response_id": sample.response_id,
                        "finish_reason": sample.finish_reason,
                        "elapsed_seconds": round(sample.elapsed_seconds, 6),
                        "timed_out": sample.timed_out,
                        "status_code": sample.status_code,
                        "call_error": sample.call_error,
                        "calls_made": sample.calls_made,
                        "usage": sample.usage,
                        "cache_ratio": cache_ratio(sample.usage),
                        "exit_code": sample.exit_code,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "json",
            ),
            "",
            "**原始文本（content）**",
            "",
            _fenced(sample.content or "（无）"),
            "",
        ]
        if sample.reasoning:
            lines += [
                "<details>",
                "<summary>API 返回的推理字段 reasoning_content 原文</summary>",
                "",
                _fenced(sample.reasoning),
                "",
                "</details>",
                "",
            ]
        lines += [
            "**原始响应 JSON（raw_response）**",
            "",
            _fenced(json.dumps(sample.raw_response, ensure_ascii=False, indent=2), "json"),
            "",
        ]
        if sample.schedules:
            lines += [
                "**解析出的安排（经生产解析器）**",
                "",
                "\n".join(f"- {_schedule_line(item)}" for item in sample.schedules),
                "",
            ]
        if sample.registration_issues:
            lines += [
                "**登记前校验问题（引擎 registration_issues）**",
                "",
                "\n".join(
                    f"- [{'结构' if item['structural'] else '时间'}] {item['text']}"
                    for item in sample.registration_issues
                ),
                "",
            ]
        lines += ["**规则检查与评价**", ""]
        if not sample.rule_checks:
            lines.append("- 无规则检查")
        else:
            lines += [
                "- " + ("✔ " if item["passed"] else "✘ ") + f"{item['name']}：{item['detail']}"
                for item in sample.rule_checks
            ]
        if sample.parse_error:
            lines += ["", f"- **解析错误**：{sample.parse_error}", ""]
        if sample.call_error:
            lines += ["", f"- **调用错误**：{sample.call_error}", ""]
        verdict = (
            "通过"
            if sample.schema_ok and not sample.registration_issues and sample.rule_passed is True
            else "未通过"
        )
        lines += ["", f"**总体评价**：{verdict}。", ""]
    return lines


def _render_pairwise(summary: dict[str, Any]) -> list[str]:
    lines = ["## 成对方向诊断", ""]
    lines.append(
        "> 方向诊断使用各样本执行段时长（expected_end−start）与起点的中位数；"
        "普通物理耗时没有实测黄金值，只报告方向与重复方差，不伪造准确率。"
        "依赖链的“预计排程一致性”只表达顺序合理，不是声称前置已成功。"
    )
    labels: dict[str, str] = {
        "distance_scale": "距离缩放（1m/5m/10m 恒速 1m/s，期望 1000/5000/10000ms）",
        "speed_same_path": "同 5 米路径·速度（2m/s vs 1m/s）",
        "injured_same_path": "同 5 米路径·伤腿（0.5m/s vs 1m/s）",
        "obstacle_detour": "障碍·已选绕行 6 米 vs 直线 2 米",
        "reach_body_condition": "身体条件·疲劳 vs 精力充沛伸手",
        "reach_posture": "姿态·坐姿 vs 站姿（不预设快慢，仅记录差异）",
        "say_proficiency": "熟练度·同一台词 1.5 字/秒 vs 4 字/秒",
        "time_offset": "当前时刻偏移·t0 vs t100000",
    }
    for key, label in labels.items():
        data = summary.get("pairwise", {}).get(key)
        if not data:
            continue
        lines += [f"### {label}", ""]
        rows = data.get("rows", {})
        for provider in sorted(rows):
            row = rows[provider]
            parts = [f"{name}={_fmt(value)}" for name, value in row.items() if name != "note"]
            lines.append(f"- **{provider}**：" + "；".join(parts))
        lines.append("")
    lines.append("")
    return lines


def _render_markdown(
    summary: dict[str, Any],
    cases: list[ProbeCase],
    samples: list[Sample],
    providers: list[ProviderConfig],
) -> str:
    lines = [
        "# 独立翻译器探针报告",
        "",
        f"- 探针：`{summary['probe']}`；样本：{summary['total_samples']}（"
        f"模型调用 {summary['paid_calls']}，跳过零调用 {summary['skipped_no_call']}）。",
        f"- API 成功 {summary['api_success']}；schema 成功 {summary['schema_success']}；"
        f"声明约束满足 {summary['constraint_satisfied']}；"
        f"模型调用约束满足率：{_fmt(summary['paid_pass_rate'])}。",
        f"- 延迟 p50/p95（秒）：{_fmt(summary['latency_seconds']['median'])} / "
        f"{_fmt(summary['latency_seconds']['p95'])}。",
        f"- tokens（仅已知）：input={_fmt(summary['tokens']['input'])} "
        f"output={_fmt(summary['tokens']['output'])} "
        f"reasoning={_fmt(summary['tokens']['reasoning'])} "
        f"cache_read={_fmt(summary['tokens']['cache_read'])}。",
        f"- cache 命中占 prompt 比例：{_fmt(summary['cache_ratio']['value'])}"
        f"（已知样本 {summary['cache_ratio']['known_samples']}；缺失为未知，不当作 0）。",
        f"- 重复稳定性：{summary['repeat_stability']['stable_groups']}/"
        f"{summary['repeat_stability']['groups_with_>=2_repeats']} 组一致"
        f"（稳定率 {_fmt(summary['repeat_stability']['stability_rate'])}）。",
        "- 上述整体约束满足率混合结构、时间和场景规则，仅作运行诊断，不等于耗时准确率。",
        f"- 明确耗时规则单独统计：{summary['exact_duration_rules']['satisfied']}/"
        f"{summary['exact_duration_rules']['samples']}；只评价给定距离/速度/语速，不作物理实测结论。",
        "",
        "## 方法 / 边界",
        "",
        "- 本探针独立于真实模拟：复用生产 `ModelTranslator.translate` 的结构化 "
        "`ActionPlan` 路径、生产提示与生产解析器，但**不启动完整 World / 引擎循环**。",
        "- 角色身体条件全部放在 World 提供的 `ObjectiveContext.conditions`（只读客观条件），"
        "不传入私有意向，也不替角色选择动作；模型只产生时间安排，实际完成与否由 World 负责。",
        "- 本次场景与角色身体条件为合成夹具，模拟 World 的只读上下文；并非真实角色实测或正式场景快照。",
        "- **无物理实测、无完整 World**：显式恒速/字速用例可比较 `expected_end−start` 与约定误差；"
        "普通物理耗时（伸手、疲劳）没有实测黄金值，只做方向诊断与重复方差，不伪造准确率。",
        "- 依赖链的前后顺序比较是**“预计排程一致性”**，不是声称前置已经成功。",
        "- 成对方向诊断仅使用 API、结构及登记时间校验都通过的排程；缺少任一对照时标记 n/a。",
        "- 方向诊断保留明确规则失败但登记合法的输出，并单列失败数量，避免只筛正确答案；每例另存原始时长中位数。",
        "- 重复稳定性比较所有结构合法的原始排程，包括登记失败的输出；稳定不代表正确。",
        "- 本探针不构成训练收益或生产语义的验证；`empty-plan` 与维持跳过（零调用）不计入模型调用约束满足率。",
        "- 请求只含单条 user 消息、`stream=false`、`n=1`、`max_retries=0`、禁止重定向；"
        "思考开关由 `--thinking` 控制，实际供应商 `extra_body` 见下表。",
        "",
        "## 模型配置",
        "",
        "| provider | model | base_url | env-key 名 | extra_body |",
        "| --- | --- | --- | --- | --- |",
    ]
    for provider in providers:
        lines.append(
            f"| {provider.name} | `{provider.model}` | `{provider.base_url}` | "
            f"`{provider.api_key_env}` | `{json.dumps(provider.extra_body, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        f"采样参数：`temperature={TEMPERATURE}`、`max_tokens={_actual_max_tokens(samples)}`"
        f"（取自实际请求体）、`max_retries=0`、`response_format=json_object`；密钥只按 `api_key_env` "
        "指定的环境变量名从 `.env` 私密读取，值与请求头不出现在任何输出。",
        "",
        "## 汇总",
        "",
        "| provider | 调用 | API/schema成功 | 登记合法 | 明确耗时规则 | 全部约束满足 | 延迟中位/p95(s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for provider in providers:
        data = summary["by_provider"].get(provider.name, {})
        latency = data.get("latency_seconds", {})
        lines.append(
            f"| {provider.name} | {data.get('paid_calls')} | "
            f"{data.get('api_success')}/{data.get('schema_success')} | {data.get('registration_satisfied')} | "
            f"{data.get('exact_duration_satisfied')}/{data.get('exact_duration_samples')} | "
            f"{data.get('constraint_satisfied')} | "
            f"{_fmt(latency.get('median'))} / {_fmt(latency.get('p95'))} |"
        )
    lines += [
        "",
        "### 按用例",
        "",
        "| case | 规则类型 | provider | 调用 | schema | 约束 | 时长中位(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        data = summary["by_case"].get(case.id, {})
        for provider in providers:
            rows = data.get("by_provider", {}).get(provider.name, {})
            lines.append(
                f"| `{case.id}` | {data.get('rule_type')} | {provider.name} | "
                f"{rows.get('paid')} | {rows.get('schema_ok')} | {rows.get('constraint_ok')} | "
                f"{_fmt(rows.get('median_duration_ms'))} |"
            )
    lines += ["", "## 逐条错误", ""]
    errors = summary.get("errors", [])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- 无")
    lines += ["", "## 重复稳定性（同一 case+provider 各 repeat 的排程是否一致）", ""]
    rows = summary["repeat_stability"]["rows"]
    if rows:
        for key, data in rows.items():
            lines.append(f"- {key}：repeats={data['repeats']}；一致={data['consistent']}")
    else:
        lines.append("- 无可用重复对。")
    lines += ["", *_render_pairwise(summary)]
    lines += ["## 逐例完整证据", ""]
    for index, case in enumerate(cases, 1):
        lines += _render_case_evidence(index, case, samples, providers)
    return "\n".join(lines) + "\n"


def write_report(
    *,
    outdir: Path,
    samples: list[Sample],
    cases: list[ProbeCase],
    providers: list[ProviderConfig],
) -> dict[str, Any]:
    summary = aggregate(samples, cases, providers)
    summary["report_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _render_markdown(summary, cases, samples, providers)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return summary


def regenerate_report(outdir: Path) -> dict[str, Any]:
    """只用保存的输入输出重导出，不访问模型、不读密钥、不用当前用例文件。"""
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["probe"] != PROBE_ID:
        raise ProbeError(f"该目录不是独立翻译器探针记录：{outdir}")
    cases = [_load_case(item) for item in manifest["cases"]]
    providers = [
        ProviderConfig(
            name=item["name"],
            model=item["model"],
            base_url=item["base_url"],
            api_key_env=item["api_key_env"],
            extra_body=item["extra_body"],
        )
        for item in manifest["providers"]
    ]
    samples: list[Sample] = []
    for line in (outdir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(Sample.from_record(json.loads(line)))
    return write_report(outdir=outdir, samples=samples, cases=cases, providers=providers)


# ---------------------------------------------------------------------------
# 密钥与配置
# ---------------------------------------------------------------------------


def private_env() -> dict[str, str]:
    """私密读取当前目录 .env（interpolate=False），合并后返回；值绝不打印/落盘。"""
    values = dotenv_values(Path.cwd() / ".env", interpolate=False)
    env = {str(key): str(value) for key, value in values.items() if value is not None}
    # override=False 语义：终端已设置的环境变量优先于 .env。
    env.update({key: value for key, value in os.environ.items()})
    return env


def _extra_body_for(provider: ProviderConfig, thinking: bool) -> dict[str, Any]:
    """返回供应商实际 extra_body；启用思考时用新字典，不修改 DEFAULT_PROVIDERS 嵌套字典。"""
    if not thinking:
        return dict(provider.extra_body)
    if provider.name == "deepseek":
        return {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    if provider.name == "qwen":
        return {"enable_thinking": True}
    raise ProbeError(f"未知供应商：{provider.name}")


def _thinking_enabled(providers: list[ProviderConfig]) -> bool:
    """从实际 providers.extra_body 判断思考是否开启，不依赖硬编码开关字符串。"""
    for provider in providers:
        if provider.name == "deepseek":
            if provider.extra_body.get("thinking", {}).get("type") == "enabled":
                return True
        elif provider.name == "qwen":
            if provider.extra_body.get("enable_thinking") is True:
                return True
    return False


def configured_providers(args: argparse.Namespace) -> list[ProviderConfig]:
    thinking = bool(getattr(args, "thinking", False))
    names = {"deepseek", "qwen"} if args.provider == "both" else {args.provider}
    overrides = {
        "deepseek": (args.deepseek_model, args.deepseek_base_url),
        "qwen": (args.qwen_model, args.qwen_base_url),
    }
    result: list[ProviderConfig] = []
    for provider in DEFAULT_PROVIDERS:
        if provider.name not in names:
            continue
        model, base_url = overrides[provider.name]
        result.append(
            ProviderConfig(
                name=provider.name,
                model=model if model else provider.model,
                base_url=base_url if base_url else provider.base_url,
                api_key_env=provider.api_key_env,
                extra_body=_extra_body_for(provider, thinking),
            )
        )
    return result


def build_manifest(
    *,
    cases: list[ProbeCase],
    providers: list[ProviderConfig],
    repeats: int,
    concurrency: int,
    timeout: float,
    max_tokens: int = MAX_TOKENS,
    outdir: Path,
    generated_utc: str,
    status: str,
) -> dict[str, Any]:
    return {
        "probe": PROBE_ID,
        "generated_utc": generated_utc,
        "outdir": str(outdir),
        "semantics": (
            "只翻译结构化动作链的时间安排，不跑完整循环；角色条件放在 World 的 "
            "ObjectiveContext.conditions；不传私有意向、不替角色选动作；模型只产生时间，"
            "实际完成由 World 负责。依赖前后顺序是预计排程一致性，不是前置已成功。"
        ),
        "providers": [provider.to_dict() for provider in providers],
        "sampling": {
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "timeout_seconds": timeout,
            "response_format": {"type": "json_object"},
            "thinking": {
                "enabled": _thinking_enabled(providers),
                "note": "由 --thinking 决定；实际供应商 extra_body 见 providers 节",
            },
            "concurrency": concurrency,
            "repeats": repeats,
        },
        "cases": [case_to_dict(case) for case in cases],
        "source_hashes": {
            "cases_file": str(PROBE_CASES_PATH),
            "cases_file_sha256": hashlib.sha256(PROBE_CASES_PATH.read_bytes()).hexdigest(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "status": status,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只产出 case/完整生产prompt/manifest，不读密钥、不调模型",
    )
    parser.add_argument("--provider", choices=("deepseek", "qwen", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="单次采样输出最大 token 数（正整数）",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="开启思考：deepseek thinking=enabled 且 reasoning_effort=high，qwen enable_thinking=true",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None, help="只跑指定 case id，可重复")
    parser.add_argument("--deepseek-model", default=None)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument(
        "--report-from", type=Path, default=None, help="从已保存目录离线重导出，不调模型"
    )
    args = parser.parse_args(argv)

    if args.report_from is not None:
        summary = regenerate_report(args.report_from)
        print(f"离线报告：{args.report_from / 'report.md'}")
        return int(unsuccessful(summary))
    if args.repeats < 1:
        raise SystemExit("--repeats 必须是正整数")
    if args.concurrency < 1:
        raise SystemExit("--concurrency 必须是正整数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须是正整数")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout 必须是有限正数")

    cases = load_cases()
    selected = select_cases(cases, args.case)
    providers = configured_providers(args)

    if args.dry_run:
        print(
            f"用例数：{len(selected)}；provider：{[p.name for p in providers]}；"
            f"重复：{args.repeats}；预计模型调用数："
            f"{sum(1 for case in selected if case.rules.get('type') != 'no_calls') * len(providers) * args.repeats}"
        )
        print(f"任务数（含零调用跳过）：{len(selected) * len(providers) * args.repeats}")
        for case in selected:
            print("=" * 72)
            print(f"case {case.id}：{case.title}（规则 {case.rules.get('type')}）")
            print(f"角色 {case.role_id}；当前时刻 {case.current_time}ms")
            print("完整生产提示词：")
            print(build_probe_prompt(case) or "（无需翻译，跳过模型，无发送提示词）")
        manifest = build_manifest(
            cases=selected,
            providers=providers,
            repeats=args.repeats,
            concurrency=args.concurrency,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            outdir=Path("(dry-run)"),
            generated_utc=_dt.datetime.now(_dt.UTC).isoformat(),
            status="dry-run",
        )
        print("=" * 72)
        print("manifest（不含密钥值）：")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    env = private_env()
    missing = sorted({p.api_key_env for p in providers if not env.get(p.api_key_env)})
    if missing:
        raise SystemExit(
            f"缺少密钥环境变量（只报变量名，不报值）：{missing}；"
            "请在当前目录 .env 配置或设置环境变量"
        )
    if env.get(_CUSTOM_HEADERS_ENV):
        raise SystemExit("检测到 OPENAI_CUSTOM_HEADERS，拒绝引入未声明的全局请求头")

    outdir = args.outdir or DEFAULT_OUTDIR / _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆写混入旧样本：{outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    generated_utc = _dt.datetime.now(_dt.UTC).isoformat()
    manifest = build_manifest(
        cases=selected,
        providers=providers,
        repeats=args.repeats,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        outdir=outdir,
        generated_utc=generated_utc,
        status="running",
    )
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出目录：{outdir}")

    tasks = [
        (case, provider, repeat)
        for repeat in range(args.repeats)
        for case in selected
        for provider in providers
    ]

    async def execute() -> list[Sample]:
        clients: dict[str, AsyncOpenAI] = {}
        try:
            for provider in providers:
                key = env[provider.api_key_env]
                clients[provider.name] = AsyncOpenAI(
                    api_key=key,
                    base_url=provider.base_url,
                    admin_api_key="",
                    organization="",
                    project="",
                    webhook_secret="",
                    timeout=args.timeout,
                    max_retries=0,
                    http_client=DefaultAsyncHttpxClient(
                        follow_redirects=False,
                        limits=httpx.Limits(
                            max_connections=args.concurrency,
                            max_keepalive_connections=args.concurrency,
                        ),
                    ),
                )
            return await run_tasks(
                clients=clients,
                tasks=tasks,
                outdir=outdir,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
            )
        finally:
            await asyncio.gather(
                *(client.close() for client in clients.values()), return_exceptions=True
            )

    samples = asyncio.run(execute())
    summary = write_report(outdir=outdir, samples=samples, cases=selected, providers=providers)
    manifest["status"] = "done"
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告：{outdir / 'report.md'}")
    print(
        f"完成：样本 {summary['total_samples']}，模型调用 {summary['paid_calls']}，"
        f"API 成功 {summary['api_success']}，schema 成功 {summary['schema_success']}，"
        f"约束满足 {summary['constraint_satisfied']}"
    )
    return int(unsuccessful(summary))


if __name__ == "__main__":
    raise SystemExit(main())
