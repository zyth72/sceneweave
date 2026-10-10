"""独立翻译器探针的离线测试。

默认测试绝不进行真实模型调用、绝不读取真实 .env 或密钥；只用 fake API
（httpx.MockTransport + AsyncOpenAI）与生产解析器/登记校验/报告验证探针
自身的纯逻辑与证据落盘口径。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe_translator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_translator", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def test_invalid_registration_is_not_a_valid_median_but_keeps_raw_estimate() -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")
    sample = probe.Sample("invalid", case.id, "deepseek", "fake-model", 0)
    sample.calls_made = 1
    sample.schedules = [{"action_id": "a-move", "start": 0, "expected_end": 5000}]
    sample.registration_issues = [{"text": "start 不晚于当前时刻", "structural": False}]
    probe.score_sample(sample, case)
    summary = probe.aggregate([sample], [case], [probe.DEFAULT_PROVIDERS[0]])
    row = summary["by_case"][case.id]["by_provider"]["deepseek"]
    assert row["median_duration_ms"] is None
    assert row["raw_median_duration_ms"] == 5000
    assert (
        summary["pairwise"]["speed_same_path"]["rows"]["deepseek"]["shorter_at_faster_speed"]
        is None
    )
    report = probe._render_markdown(summary, [case], [sample], [probe.DEFAULT_PROVIDERS[0]])
    assert "明确耗时规则单独统计：1/1" in report


def test_cancellation_is_recorded_and_propagated(tmp_path: Path) -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("等待应由取消中断")

        case = _by_id(probe.load_cases(), "move-5m-1ms")
        client = _client(handler)
        task = asyncio.create_task(
            probe.run_tasks(
                {"deepseek": client},
                [(case, probe.DEFAULT_PROVIDERS[0], 0)],
                tmp_path,
                1,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.close()
        records = (tmp_path / "samples.jsonl").read_text().splitlines()
        assert len(records) == 1
        record = json.loads(records[0])
        assert record["call_error"] == "CancelledError"
        assert record["calls_made"] == 1
        assert record["schema_ok"] is False

    asyncio.run(scenario())


def test_role_condition_comparison_keeps_the_action_and_target_identical() -> None:
    cases = probe.load_cases()
    baseline = _by_id(cases, "move-5m-1ms")
    for case_id in ("move-5m-2ms", "move-5m-injured-0.5ms"):
        other = _by_id(cases, case_id)
        assert other.role_id == baseline.role_id
        assert other.plan == baseline.plan


_CASE_IDS = {
    "move-1m-1ms",
    "move-5m-1ms",
    "move-10m-1ms",
    "move-5m-2ms",
    "move-5m-injured-0.5ms",
    "move-obstacle-detour-6m",
    "reach-standing",
    "reach-tired",
    "reach-seated",
    "maintain-only",
    "move-obstacle-straight-2m",
    "say-proficient-4cps",
    "say-novice-1.5cps",
    "reach-grasp-chain",
    "move-reach-grasp-chain",
    "move-say-independent",
    "maintain-reach-new-say",
    "empty-plan",
    "offset-same-scene-t0",
    "offset-same-scene-t100000",
}


def _response(
    body: dict[str, Any],
    *,
    finish: str = "stop",
    content: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_usage = {
        "prompt_tokens": 120,
        "completion_tokens": 25,
        "total_tokens": 145,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    return {
        "id": "completion-test",
        "object": "chat.completion",
        "created": 1,
        "model": body.get("model", "fake-model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content or '{"schedules": []}',
                },
            }
        ],
        "usage": usage if usage is not None else default_usage,
    }


def _client(handler: Any) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="fake-not-a-secret",
        base_url=probe.DEFAULT_PROVIDERS[0].base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _ok_handler(content: str) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(json.loads(request.content), content=content))

    return handler


def _by_id(cases: list[Any], case_id: str) -> Any:
    return next(case for case in cases if case.id == case_id)


# ---------------------------------------------------------------------------
# 用例文件与提示
# ---------------------------------------------------------------------------


def test_cases_file_has_16_to_20_cases_with_required_fields() -> None:
    cases = probe.load_cases()
    assert 16 <= len(cases) <= 20
    ids = {case.id for case in cases}
    assert ids == _CASE_IDS, "覆盖任务要求的全部场景类型"
    for case in cases:
        assert case.scene and case.conditions and case.plan is not None
        assert case.rules.get("type")
        assert case.scoring
        assert case.role_id and case.current_time >= 0


def test_rule_types_cover_required_scenarios() -> None:
    cases = probe.load_cases()
    by_type: dict[str, list[str]] = {}
    for case in cases:
        by_type.setdefault(str(case.rules["type"]), []).append(case.id)
    assert set(by_type) >= {
        "exact_duration",
        "direction",
        "schedule_consistency",
        "no_retranslate",
        "no_calls",
        "offset",
        "independent",
    }, "规则类型应覆盖任务要求"
    assert set(by_type["no_calls"]) == {"empty-plan", "maintain-only"}
    assert by_type["exact_duration"], "应有显式恒速/字速的确定性用例"


def test_prompt_is_deterministic_and_identical_for_same_case() -> None:
    cases = probe.load_cases()
    for case in cases[:6]:
        first = probe.build_probe_prompt(case)
        second = probe.build_probe_prompt(case)
        assert first == second, "相同 case 的提示必须逐字一致"
        assert f"角色 {case.role_id}" in first
        assert f"当前剧情时刻 {case.current_time} 毫秒" in first
        for condition in case.conditions:
            assert condition in first, "角色客观条件应进入提示"


def test_prompt_does_not_leak_offline_rules_or_scoring() -> None:
    cases = probe.load_cases()
    for case in cases:
        prompt = probe.build_probe_prompt(case)
        assert case.scoring not in prompt, "评分规则不得进入模型提示"
        assert "expected_ms" not in prompt
        assert case.rules.get("type") not in prompt
        # 结构化路径没有主观意向段：不传私有意向、不替角色选择动作。
        assert "意向" not in prompt


def test_prompt_identical_across_providers_and_repeats_via_real_run() -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")

    async def scenario() -> None:
        handler = _ok_handler(
            '{"schedules": [{"action_id": "a-move", "start": 1000, '
            '"expected_end": 6000, "checkpoints": []}]}'
        )
        client = _client(handler)
        first = await probe.run_one(client, probe.DEFAULT_PROVIDERS[0], case, 0)
        second = await probe.run_one(client, probe.DEFAULT_PROVIDERS[0], case, 1)
        third = await probe.run_one(client, probe.DEFAULT_PROVIDERS[1], case, 0)
        assert first.prompt == second.prompt == third.prompt == probe.build_probe_prompt(case)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 生产解析器与登记校验
# ---------------------------------------------------------------------------


def test_production_parser_valid_output() -> None:
    case = _by_id(probe.load_cases(), "move-1m-1ms")
    raw = (
        '{"schedules": [{"action_id": "a-move", "start": 300, '
        '"expected_end": 1300, "checkpoints": [800]}]}'
    )

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.schema_ok, sample.parse_error
        assert sample.parse_error is None
        [schedule] = sample.schedules
        assert schedule["action_id"] == "a-move"
        assert schedule["start"] == 300 and schedule["expected_end"] == 1300
        assert schedule["checkpoints"] == [800]
        assert not sample.registration_issues
        assert sample.rule_passed is True, sample.rule_checks

    asyncio.run(scenario())


def test_production_parser_rejects_invalid_action_id_and_preserves_raw() -> None:
    case = _by_id(probe.load_cases(), "move-1m-1ms")
    raw = '{"schedules": [{"action_id": "a-wrong", "start": 300, "expected_end": 1300}]}'

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.schema_ok is False
        assert sample.parse_error and "非本次新动作" in sample.parse_error
        assert sample.call_error is None, "这是解析错误不是调用错误"
        assert (
            sample.raw_response is not None
            and sample.raw_response["choices"][0]["message"]["content"] == raw
        )
        assert not sample.schedules
        assert sample.rule_passed is False

    asyncio.run(scenario())


def test_production_parser_rejects_duplicate_and_unknown_fields() -> None:
    case = _by_id(probe.load_cases(), "move-1m-1ms")

    async def scenario() -> None:
        for raw in (
            '{"schedules": [{"action_id": "a-move", "start": 300, "expected_end": 1300, "extra": 1}]}',
            '{"schedules": [{"action_id": "a-move", "start": true, "expected_end": 1300}]}',
            '{"schedules": [{"action_id": "a-move", "start": 2000.5, "expected_end": 1300}]}',
            '{"schedules": [], "schedules": []}',
        ):
            sample = await probe.run_one(
                _client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0
            )
            assert sample.schema_ok is False, raw

    asyncio.run(scenario())


def test_registration_business_constraint_start_must_be_after_now() -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")
    # start=0 不晚于当前时刻 0 → 时间业务约束问题（不是结构错误）。
    raw = '{"schedules": [{"action_id": "a-move", "start": 0, "expected_end": 5000}]}'

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.schema_ok is True, "生产解析器只做结构校验"
        assert sample.registration_issues, "业务时间约束应由登记校验逐条报告"
        assert any(
            "开始时刻必须晚于当前时刻" in issue["text"] and not issue["structural"]
            for issue in sample.registration_issues
        )

    asyncio.run(scenario())


def test_dependency_schedule_consistency_ordering() -> None:
    case = _by_id(probe.load_cases(), "reach-grasp-chain")
    # grasp 的 start 早于 reach 的 expected_end → 预计排程不一致（探针专有规则）。
    raw = (
        '{"schedules": [{"action_id": "a-reach", "start": 100, "expected_end": 1200},'
        ' {"action_id": "a-grasp", "start": 800, "expected_end": 1800}]}'
    )

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.schema_ok
        assert sample.rule_passed is False
        assert any(
            "ordering_a-grasp->a-reach" in check["name"] and not check["passed"]
            for check in sample.rule_checks
        )
        # 登记校验不把依赖顺序当结构/时间问题（那是 World 真实完成门控）。

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 零调用
# ---------------------------------------------------------------------------


def _maintain_only_case() -> Any:
    return probe.ProbeCase(
        id="test-maintain-only",
        title="维持既有动作（零调用）",
        role_id="测试角色",
        cycle_id="probe-cycle",
        response_id="r-keep",
        current_time=0,
        world_state_version=0,
        scene="只维持既有伸手动作，没有新增动作。",
        conditions=("a-keep 已在进行中",),
        plan=(
            {
                "action_id": "a-keep",
                "kind": "reach",
                "target": "桌上文件",
                "effector": "right_hand",
            },
        ),
        arrangements=(
            {
                "action_id": "a-keep",
                "phase": "started",
                "description": "伸手触及 桌上文件（right_hand）",
                "start": 0,
                "expected_end": 3000,
                "checkpoints": [],
            },
        ),
        rules={"type": "no_calls"},
        scoring="只维持既有动作：零模型调用、空安排。",
    )


def test_empty_plan_zero_calls() -> None:
    case = _by_id(probe.load_cases(), "empty-plan")

    async def scenario() -> None:
        sample = await probe.run_one(
            _client(_ok_handler("{}")), probe.DEFAULT_PROVIDERS[0], case, 0
        )
        assert sample.calls_made == 0, "合法空计划必须零调用"
        assert sample.schedules == []
        assert sample.schema_ok and not sample.registration_issues
        assert sample.rule_passed is True, sample.rule_checks

    asyncio.run(scenario())


def test_maintain_only_zero_calls() -> None:
    case = _maintain_only_case()

    async def scenario() -> None:
        sample = await probe.run_one(
            _client(_ok_handler("{}")), probe.DEFAULT_PROVIDERS[0], case, 0
        )
        assert sample.calls_made == 0, "仅维持既有动作必须零调用"
        assert sample.schedules == []
        assert sample.rule_passed is True, sample.rule_checks

    asyncio.run(scenario())


def test_no_retranslate_only_new_say_scheduled() -> None:
    case = _by_id(probe.load_cases(), "maintain-reach-new-say")
    raw = '{"schedules": [{"action_id": "a-new-say", "start": 1500, "expected_end": 5000}]}'

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.calls_made == 1
        assert [item["action_id"] for item in sample.schedules] == ["a-new-say"]
        assert sample.rule_passed is True, sample.rule_checks

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# API 错误落盘与密钥不外泄
# ---------------------------------------------------------------------------


def test_api_error_recorded_and_not_counted_as_success(tmp_path: Path) -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")
    seen: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(500, json={"error": {"message": "SECRET_ERROR_BODY"}})

    async def scenario() -> None:
        sample = await probe.run_one(_client(handler), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.call_error and "500" in sample.call_error
        assert sample.exit_code == 1
        assert sample.schema_ok is False
        record = json.dumps(sample.to_record(), ensure_ascii=False)
        assert "SECRET_ERROR_BODY" not in record, "上游错误正文不得写入输出"

    asyncio.run(scenario())
    assert len(seen) == 1, "max_retries=0，不做 SDK 自动重试"


def test_auth_failure_sets_credentials_error() -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "SECRET_ERROR_BODY"}})

    async def scenario() -> None:
        sample = await probe.run_one(_client(handler), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.call_error and "401" in sample.call_error
        assert "SECRET_ERROR_BODY" not in json.dumps(sample.to_record(), ensure_ascii=False)

    asyncio.run(scenario())


def test_api_key_value_never_in_record(tmp_path: Path) -> None:
    case = _by_id(probe.load_cases(), "move-5m-1ms")
    secret = "super-secret-key-xyz-12345"
    raw = '{"schedules": [{"action_id": "a-move", "start": 1000, "expected_end": 6000}]}'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(json.loads(request.content), content=raw))

    async def scenario() -> None:
        client = AsyncOpenAI(
            api_key=secret,
            base_url=probe.DEFAULT_PROVIDERS[0].base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        sample = await probe.run_one(client, probe.DEFAULT_PROVIDERS[0], case, 0)
        record = json.dumps(sample.to_record(), ensure_ascii=False)
        assert secret not in record, "密钥值不得进入样本记录"
        assert secret not in json.dumps(sample.request_body, ensure_ascii=False)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 报告完整性、usage 与 cache 口径
# ---------------------------------------------------------------------------


def test_usage_unknown_and_cache_ratio_not_faked() -> None:
    # 没有 cache 字段 → cache_ratio 是 None（未知），不是 0。
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert probe.cache_ratio(usage) is None
    assert probe.cache_ratio(None) is None
    assert (
        probe.cache_ratio({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}})
        == 0.0
    )
    aggregate = probe._cache_aggregate([])
    assert (
        aggregate["value"] is None
        and aggregate["unknown"] is True
        and aggregate["known_samples"] == 0
    )


def test_aggregate_does_not_fake_pass_rate_with_zero_paid() -> None:
    case = _by_id(probe.load_cases(), "empty-plan")
    samples = []
    summary = probe.aggregate(
        samples + [probe.Sample("x", case.id, "deepseek", "m", 0)],
        [case],
        list(probe.DEFAULT_PROVIDERS[:1]),
    )
    assert summary["paid_calls"] == 0
    assert summary["paid_pass_rate"] is None


def test_report_contains_full_scene_prompt_and_raw(tmp_path: Path) -> None:
    cases = probe.load_cases()
    case = _by_id(cases, "move-5m-1ms")
    raw = '{"schedules": [{"action_id": "a-move", "start": 1000, "expected_end": 6000, "checkpoints": [3000]}]}'

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        summary = probe.write_report(
            outdir=tmp_path,
            samples=[sample],
            cases=[case],
            providers=[probe.DEFAULT_PROVIDERS[0]],
        )
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert case.scene in report, "报告必须包含完整场景"
        for condition in case.conditions:
            assert condition in report, "报告必须包含全部角色条件"
        assert case.scoring in report, "报告必须包含评分条件"
        assert sample.prompt in report, "报告必须包含完整 exact 生产 prompt"
        assert raw in report, "报告必须包含完整原始输出"
        assert "1000→6000ms" in report
        assert "cache 命中占 prompt 比例" in report
        assert summary["by_case"][case.id]["by_provider"]["deepseek"]["schema_ok"] == 1

    asyncio.run(scenario())


def test_manifest_and_samples_records_exist_without_secrets(tmp_path: Path) -> None:
    cases = probe.load_cases()
    case = _by_id(cases, "move-5m-1ms")
    raw = '{"schedules": [{"action_id": "a-move", "start": 1000, "expected_end": 6000}]}'

    async def scenario() -> None:
        sample = await probe.run_one(_client(_ok_handler(raw)), probe.DEFAULT_PROVIDERS[0], case, 0)
        manifest = probe.build_manifest(
            cases=[case],
            providers=[probe.DEFAULT_PROVIDERS[0]],
            repeats=1,
            concurrency=2,
            timeout=60.0,
            outdir=tmp_path,
            generated_utc="2026-01-01T00:00:00+00:00",
            status="done",
        )
        serialized = json.dumps(manifest, ensure_ascii=False)
        assert "DEEPSEEK_API_KEY" in serialized, "manifest 记录的是环境变量名而非值"
        for forbidden in ("sk-", "api_key_=", "Authorization"):
            assert forbidden not in serialized, f"manifest 不得包含密钥形态：{forbidden}"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        (tmp_path / "samples.jsonl").write_text(
            json.dumps(sample.to_record(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # 离线重导出不调模型。
        summary = probe.regenerate_report(tmp_path)
        assert summary["total_samples"] == 1
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert case.scene in report and sample.prompt in report

    asyncio.run(scenario())


def test_report_from_regenerate_returns_failure_flag(tmp_path: Path) -> None:
    # 失败样本离线重导出应返回非零退出（unsuccessful）。
    case = _by_id(probe.load_cases(), "move-5m-1ms")

    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {}})

        sample = await probe.run_one(_client(handler), probe.DEFAULT_PROVIDERS[0], case, 0)
        assert sample.call_error
        summary = probe.write_report(
            outdir=tmp_path,
            samples=[sample],
            cases=[case],
            providers=[probe.DEFAULT_PROVIDERS[0]],
        )
        assert probe.unsuccessful(summary)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 配置与 dry-run
# ---------------------------------------------------------------------------


def test_configured_providers_overrides_and_filter() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(
        provider="deepseek",
        deepseek_model="custom-model",
        deepseek_base_url="https://example.com/v1",
        qwen_model=None,
        qwen_base_url=None,
    )
    providers = probe.configured_providers(args)
    assert [p.name for p in providers] == ["deepseek"]
    assert providers[0].model == "custom-model"
    assert providers[0].base_url == "https://example.com/v1"

    args_both = SimpleNamespace(
        provider="both",
        deepseek_model=None,
        deepseek_base_url=None,
        qwen_model="qwen-custom",
        qwen_base_url=None,
    )
    both = probe.configured_providers(args_both)
    assert [p.name for p in both] == ["deepseek", "qwen"]
    assert both[1].model == "qwen-custom"


def test_select_cases_filters_and_rejects_unknown() -> None:
    cases = probe.load_cases()
    selected = probe.select_cases(cases, ["move-1m-1ms", "reach-tired"])
    assert [case.id for case in selected] == ["move-1m-1ms", "reach-tired"]
    with pytest.raises(probe.ProbeError, match="未知用例"):
        probe.select_cases(cases, ["no-such-case"])


def test_dry_run_does_not_read_credentials_or_call_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("dry-run 不应读取密钥或构造 API 客户端")

    monkeypatch.setattr(probe, "private_env", forbidden)
    monkeypatch.setattr(probe, "AsyncOpenAI", forbidden)
    assert probe.main(["--dry-run", "--case", "move-1m-1ms", "--case", "reach-grasp-chain"]) == 0


def test_main_rejects_bad_arguments() -> None:
    with pytest.raises(SystemExit):
        probe.main(["--repeats", "0"])
    with pytest.raises(SystemExit):
        probe.main(["--timeout", "-1"])
