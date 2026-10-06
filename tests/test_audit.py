"""运行后审计：严格配置、证据报告、机械检查、多模型隔离与 CLI 接线。

全部使用假采样器，不调用真实 API、不读 .env、不写入既有录制目录。
"""

import asyncio
import json
from pathlib import Path

import pytest

import sceneweave.__main__ as cli_main
from sceneweave.app.assembly import build_scripted_factory
from sceneweave.app.runner import run_headless
from sceneweave.audit.config import AuditConfig, AuditModelSpec, load_audit_config
from sceneweave.audit.models import AuditOutputError, parse_audit_report
from sceneweave.audit.report import (
    AuditSummary,
    build_evidence_report,
    run_mechanical_checks,
)
from sceneweave.audit.session import AuditSession
from sceneweave.config.loader import ConfigError, load_assembly_config
from sceneweave.domain.identity import CycleId
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.world.scene import load_scene_card

from .test_cli_headless import SCRIPTS, setup_config

VALID_AUDIT = json.dumps(
    {
        "summary": "报告整体稳定",
        "findings": [
            {
                "severity": "suspicion",
                "certainty": "uncertain",
                "title": "信息不足",
                "explanation": "缺少通信正文证据，只作为疑点。",
                "evidence_refs": ["E-0001"],
                "suggestion": "补充通信日志后再判断。",
            }
        ],
    },
    ensure_ascii=False,
)


class FakeAuditSampler:
    """按脚本返回文本或抛出指定异常；记录调用次数。"""

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls = 0

    async def sample(self, prompt: str, *, component: str) -> str:
        assert component == "audit"
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, str)
        return item


def _write_audit_config(tmp_path: Path, payload: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _records() -> list[TraceRecord]:
    cycle = CycleId("cycle-1")
    return [
        TraceRecord(
            stage="engine.cycle_start",
            cycle_id=cycle,
            simulation_time=SimulationTime(0),
            summary="进入事件时刻",
        ),
        TraceRecord(
            stage="world.adjudicate.candidate",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="裁决候选",
            outputs=('{"outcomes": [{"action_id": "a-1", "result": "failed", "detail": "失败"}]}',),
        ),
        TraceRecord(
            stage="world.validate",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="校验通过",
            world_state_version=1,
        ),
        # 引擎与 World 两层对同一次提交留痕：只应算一笔。
        TraceRecord(
            stage="world.commit",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="事实已提交",
            details={"changes": 0, "outcomes": 1, "version": 1},
            world_state_version=1,
        ),
        TraceRecord(
            stage="world.commit",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="事实已原子提交",
            details={"changes": 0},
            world_state_version=1,
        ),
        TraceRecord(
            stage="world.progress.applied",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="实际进展已原子应用",
            details={"moved": False, "version": 1},
            world_state_version=1,
        ),
        # 角色屏障完成：不是完整周期边界。
        TraceRecord(
            stage="cycle.complete",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="处理周期完成",
        ),
        TraceRecord(
            stage="engine.cycle_complete",
            cycle_id=cycle,
            simulation_time=SimulationTime(100),
            summary="完整周期完成",
            details={"completed_cycles": 1, "outcomes": 1, "registry_actions": 0},
            world_state_version=1,
        ),
    ]


def _summary(**overrides: object) -> AuditSummary:
    base: dict[str, object] = {
        "stop_reason": "time_limit",
        "final_time_ms": 100,
        "completed_cycles": 1,
        "world_state_version": 1,
        "error": None,
    }
    base.update(overrides)
    return AuditSummary(**base)  # type: ignore[arg-type]


def _scene():
    return load_scene_card(
        {
            "id": "room",
            "name": "房间",
            "kind": "房间",
            "unit": "m",
            "size": [4, 3],
            "children": [
                {"id": "lin", "name": "林", "kind": "角色", "position": [0.5, 0.5], "children": []},
                {
                    "id": "box",
                    "name": "盒子",
                    "kind": "容器",
                    "state": {"open": False},
                    "children": [],
                },
            ],
        }
    )


def test_audit_config_accepts_single_and_multiple(tmp_path: Path) -> None:
    single = load_audit_config(
        _write_audit_config(
            tmp_path,
            {
                "models": [
                    {
                        "id": "deepseek-flash",
                        "options": {
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-flash",
                            "api_key_env": "DEEPSEEK_API_KEY",
                        },
                    }
                ]
            },
        )
    )
    assert [item.model_id for item in single.models] == ["deepseek-flash"]
    assert single.models[0].model_name == "deepseek-flash"
    multi = load_audit_config(
        _write_audit_config(
            tmp_path / "m",
            {
                "models": [
                    {"id": "a", "options": {"model": "m1"}},
                    {
                        "id": "b",
                        "factory": "tests.fakes:create_model",
                        "options": {"model": "m2"},
                    },
                ]
            },
        )
    )
    assert [item.model_id for item in multi.models] == ["a", "b"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"models": []},
        {"models": "x"},
        {"models": [{"id": "a"}, {"id": "a"}]},
        {"models": [{"id": "a", "unknown": 1}]},
        {"models": [{"id": "a", "options": {"api_key": "sk-plain"}}]},
        {"models": [{"id": "a", "factory": "not-a-factory"}]},
        {"models": [{"id": "a", "options": []}]},
        {"models": [{"id": "a"}], "extra": 1},
    ],
)
def test_audit_config_rejects_shape_errors(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ConfigError):
        load_audit_config(_write_audit_config(tmp_path, payload))


def test_evidence_report_keeps_failures_and_reports_findings() -> None:
    report = build_evidence_report(_scene(), _records(), _summary(), incomplete=False)
    text = report.markdown()
    assert "失败" in text, "失败结算进入报告，不做成功筛选"
    assert report.evidence_ids
    assert all(item.evidence_id.startswith("E-") for item in report.items)
    assert "机械检查" in text

    broken = [record for record in _records()]
    broken.append(
        TraceRecord(
            stage="world.commit",
            cycle_id=CycleId("cycle-2"),
            simulation_time=SimulationTime(200),
            summary="事实已提交",
            world_state_version=1,
        )
    )
    checks = run_mechanical_checks(report.items, broken, _summary())
    monotonic = next(check for check in checks if check.check_id == "commit-version-monotonic")
    assert monotonic.status == "finding", "提交版本未严格递增必须报确定问题"


def test_evidence_report_flags_missing_traces_as_not_covered() -> None:
    records = [record for record in _records() if record.stage == "cycle.complete"]
    report = build_evidence_report(_scene(), records, _summary(), incomplete=False)
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["trace-coverage:world.commit"] == "not_covered"
    assert statuses["commit-version-monotonic"] == "not_covered"
    assert statuses["state-version-consistency"] == "not_covered"
    # 角色屏障完成不能冒充完整周期边界。
    assert statuses["summary-cycle-consistency"] == "not_covered"


def test_incomplete_report_marks_scope_without_faking_cycles() -> None:
    report = build_evidence_report(
        _scene(),
        _records(),
        _summary(stop_reason="error", error="模拟内部错误", completed_cycles=0),
        incomplete=True,
    )
    assert "未正常结束" in report.scope_note
    assert "不伪造" in report.scope_note
    assert any(check.check_id == "run-completion" for check in report.checks)


def test_parse_audit_report_strict_reference_and_value_validation() -> None:
    ids = frozenset({"E-0001", "E-0002"})
    parsed = parse_audit_report(VALID_AUDIT, model_id="a", model="m", evidence_ids=ids)
    assert parsed.findings[0].evidence_refs == ("E-0001",)

    bad_ref = json.dumps(
        {
            "summary": "s",
            "findings": [
                {
                    "severity": "info",
                    "certainty": "certain",
                    "title": "t",
                    "explanation": "e",
                    "evidence_refs": ["E-9999"],
                    "suggestion": "g",
                }
            ],
        },
        ensure_ascii=False,
    )
    with pytest.raises(AuditOutputError, match="报告之外"):
        parse_audit_report(bad_ref, model_id="a", model="m", evidence_ids=ids)
    bad_severity = VALID_AUDIT.replace('"suspicion"', '"critical"')
    with pytest.raises(AuditOutputError, match="severity"):
        parse_audit_report(bad_severity, model_id="a", model="m", evidence_ids=ids)
    missing = json.dumps({"summary": "s"}, ensure_ascii=False)
    with pytest.raises(AuditOutputError):
        parse_audit_report(missing, model_id="a", model="m", evidence_ids=ids)
    extra = json.dumps({"summary": "s", "findings": [], "extra": 1}, ensure_ascii=False)
    with pytest.raises(AuditOutputError, match="未知字段"):
        parse_audit_report(extra, model_id="a", model="m", evidence_ids=ids)


def test_session_isolates_model_failure_and_writes_outputs(tmp_path: Path) -> None:
    config = AuditConfig(
        models=(
            AuditModelSpec("bad", "tests.fakes:create_model", {"model": "m"}),
            AuditModelSpec("good", "tests.fakes:create_model", {"model": "m"}),
        )
    )
    samplers = {
        "bad": FakeAuditSampler([RuntimeError("外部正文 sk-secret")]),
        "good": FakeAuditSampler([VALID_AUDIT]),
    }
    session = AuditSession(config, lambda spec: samplers[spec.model_id])

    async def scenario() -> None:
        await session.run(
            scene=_scene(),
            records=_records(),
            summary=_summary(),
            output_directory=tmp_path,
            incomplete=False,
        )

    asyncio.run(scenario())
    state = session.state.snapshot()
    assert state["status"] == "done"
    assert state["failed_models"] == 1
    by_id = {model["id"]: model for model in state["models"]}
    assert by_id["bad"]["status"] == "failed"
    assert "sk-secret" not in str(by_id["bad"]["error"]), "外部正文不得进入审计输出"
    assert by_id["good"]["status"] == "ok"
    payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert [model["id"] for model in payload["models"]] == ["bad", "good"]
    for filename in ("report.md", "audit.md", "audit.json"):
        assert (tmp_path / filename).is_file()
    assert (tmp_path / "audit-good.response.txt").read_text(encoding="utf-8") == VALID_AUDIT


def test_headless_audit_runs_once_and_keeps_simulation_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_assembly_config(setup_config(tmp_path))
    factory = build_scripted_factory(SCRIPTS)
    audit_config = AuditConfig(
        models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),)
    )
    sampler = FakeAuditSampler([VALID_AUDIT])
    session = AuditSession(audit_config, lambda spec: sampler)

    async def run() -> tuple:
        return await run_headless(
            config,
            factory,
            output_directory=tmp_path / "with-audit",
            audit=session,
        )

    summary, out_dir = asyncio.run(run())
    assert sampler.calls == 1, "审计只触发一次"
    assert session.state.snapshot()["status"] == "done"
    assert (out_dir / "report.md").is_file() and (out_dir / "audit.json").is_file()
    summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert json.loads(summary_text)["world_state_version"] == summary.world_state_version

    # 关闭审计：不产生任何审计文件，也不需要审计模型。
    async def run_without() -> tuple:
        return await run_headless(
            config, build_scripted_factory(SCRIPTS), output_directory=tmp_path / "no-audit"
        )

    _, plain_dir = asyncio.run(run_without())
    assert not (plain_dir / "audit.json").exists()
    assert not (plain_dir / "report.md").exists()


def test_cli_requires_audit_config_before_simulation(tmp_path: Path) -> None:
    config = setup_config(tmp_path)
    audit_path = _write_audit_config(tmp_path, {"models": [{"id": "a", "options": {}}]})
    exit_code = cli_main.main(
        [
            str(config),
            "--scripted",
            str(tmp_path / "scripted.json"),
            "--audit-config",
            str(audit_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "out").exists(), "审计配置错误不得先跑模拟"


def test_cli_audit_disabled_does_not_load_dotenv_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = setup_config(tmp_path)
    script_path = tmp_path / "scripted.json"
    script_path.write_text(json.dumps(SCRIPTS, ensure_ascii=False), encoding="utf-8")

    def forbidden() -> None:
        raise AssertionError("未开启审计不得读取 .env")

    monkeypatch.setattr(cli_main, "_load_live_dotenv", forbidden)
    exit_code = cli_main.main(
        [
            str(config),
            "--scripted",
            str(script_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    assert not (tmp_path / "out" / "audit.json").exists()


def test_cli_audit_failure_keeps_simulation_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = setup_config(tmp_path)
    script_path = tmp_path / "scripted.json"
    script_path.write_text(json.dumps(SCRIPTS, ensure_ascii=False), encoding="utf-8")
    audit_path = _write_audit_config(
        tmp_path,
        {
            "models": [
                {"id": "bad", "options": {"model": "m"}},
                {"id": "good", "options": {"model": "m"}},
            ]
        },
    )
    samplers = {
        "bad": FakeAuditSampler(["不是 JSON"]),
        "good": FakeAuditSampler([VALID_AUDIT]),
    }
    monkeypatch.setattr(cli_main, "_load_live_dotenv", lambda: None)
    monkeypatch.setattr(cli_main, "_default_audit_factory", lambda spec: samplers[spec.model_id])
    exit_code = cli_main.main(
        [
            str(config),
            "--scripted",
            str(script_path),
            "--audit-config",
            str(audit_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1, "一个审计模型失败应保持非零退出"
    captured = capsys.readouterr()
    assert "停止原因" in captured.out, "模拟成功概要仍打印"
    assert "bad" in captured.out and "good" in captured.out
    payload = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    statuses = {model["id"]: model["status"] for model in payload["models"]}
    assert statuses == {"bad": "failed", "good": "ok"}


def test_audit_outputs_stay_out_of_recording_directory(tmp_path: Path) -> None:
    """审计只写输出目录，不改录制目录内容或清单。"""
    config = load_assembly_config(setup_config(tmp_path))
    audit_config = AuditConfig(
        models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),)
    )
    session = AuditSession(audit_config, lambda spec: FakeAuditSampler([VALID_AUDIT]))

    async def run() -> tuple:
        return await run_headless(
            config,
            build_scripted_factory(SCRIPTS),
            output_directory=tmp_path / "out",
            record_directory=tmp_path / "rec",
            audit=session,
        )

    _, out_dir = asyncio.run(run())
    record_files = {path.name for path in (tmp_path / "rec").iterdir()}
    assert "manifest.json" in record_files
    assert "audit.json" not in record_files and "report.md" not in record_files
    assert (out_dir / "audit.json").is_file()


class CancelAuditSampler:
    """审计模型调用期间取消；用于验证取消不回写已导出的成功概要。"""

    async def sample(self, prompt: str, *, component: str) -> str:
        raise asyncio.CancelledError()


def test_two_layer_commit_records_count_as_one_and_duplicate_version_fails() -> None:
    """引擎/World 同一次提交只算一笔；真正重复的同版本仍必须报错。"""
    report = build_evidence_report(_scene(), _records(), _summary(), incomplete=False)
    check = next(c for c in report.checks if c.check_id == "commit-version-monotonic")
    assert check.status == "passed", check.detail
    assert "1 笔提交" in check.detail

    duplicate = list(_records())
    duplicate.append(
        TraceRecord(
            stage="world.commit",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="同版本再次提交",
            details={"changes": 0},
            world_state_version=1,
        )
    )
    checks = run_mechanical_checks(report.items, duplicate, _summary())
    duplicate_check = next(c for c in checks if c.check_id == "commit-version-monotonic")
    assert duplicate_check.status == "finding", "真正重复同版本仍必须报错"


def test_state_version_consistency_accepts_progress_beyond_last_commit() -> None:
    progress = TraceRecord(
        stage="world.progress.applied",
        cycle_id=CycleId("cycle-2"),
        simulation_time=SimulationTime(200),
        summary="实际进展已原子应用",
        details={"moved": True, "version": 2},
        world_state_version=2,
    )
    report = build_evidence_report(
        _scene(),
        [*_records(), progress],
        _summary(final_time_ms=200, world_state_version=2),
        incomplete=False,
    )
    check = next(c for c in report.checks if c.check_id == "state-version-consistency")
    assert check.status == "passed", check.detail
    assert "进展" in check.detail


def test_commit_pairing_does_not_hide_other_cycles_or_same_time_rollbacks() -> None:
    """两层合并必须同周期同刻；同刻版本回退不能通过排序被隐藏。"""
    records = [
        TraceRecord(
            stage="world.commit",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="组件提交",
            details={"version": 2},
            world_state_version=2,
        ),
        TraceRecord(
            stage="world.commit",
            cycle_id=CycleId("cycle-2"),
            simulation_time=SimulationTime(100),
            summary="另一个周期同版本提交",
            world_state_version=2,
        ),
        TraceRecord(
            stage="world.commit",
            cycle_id=CycleId("cycle-2"),
            simulation_time=SimulationTime(100),
            summary="同刻版本回退",
            world_state_version=1,
        ),
    ]
    checks = run_mechanical_checks((), records, _summary(world_state_version=1))
    by_id = {check.check_id: check for check in checks}
    assert by_id["commit-version-monotonic"].status == "finding"
    assert "(2, 2)" in by_id["commit-version-monotonic"].detail
    assert "(2, 1)" in by_id["commit-version-monotonic"].detail
    assert by_id["state-version-consistency"].status == "finding"


def test_report_omits_repeated_role_prompts_but_retains_actual_observations_and_answers() -> None:
    records = [
        TraceRecord(
            stage="role.perceive",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="实际投递",
            inputs=("实际收到的观测",),
        ),
        TraceRecord(
            stage="system1.decide",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="判断完成",
            inputs=("重复角色提示词",),
            outputs=('{"reconsider":false,"note":"完整认知说明"}',),
        ),
    ]
    report = build_evidence_report(_scene(), records, _summary(), incomplete=False)
    text = report.markdown()
    assert "重复角色提示词" not in text
    assert "实际收到的观测" in text
    assert "完整认知说明" in text


def test_report_omits_invoke_prompts_and_keeps_candidate_structures() -> None:
    huge_prompt = "重复指令" * 100 + "SECRET_PROMPT_MARKER"
    records = [
        TraceRecord(
            stage="world.adjudicate.invoke",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="World 联合裁决模型调用",
            inputs=(huge_prompt,),
        ),
        TraceRecord(
            stage="world.adjudicate.candidate",
            cycle_id=CycleId("cycle-1"),
            simulation_time=SimulationTime(100),
            summary="联合裁决候选已生成",
            outputs=('{"outcomes": [{"action_id": "a-1", "result": "unconfirmed"}]}',),
            world_state_version=1,
        ),
    ]
    report = build_evidence_report(_scene(), records, _summary(), incomplete=False)
    text = report.markdown()
    assert "SECRET_PROMPT_MARKER" not in text
    assert "原始提示与请求已省略" in text
    assert "outcomes" in text and "unconfirmed" in text and "a-1" in text
    assert "世界版本=1" in text


def test_findings_require_valid_evidence_reference() -> None:
    ids = frozenset({"E-0001"})
    empty_refs = json.dumps(
        {
            "summary": "s",
            "findings": [
                {
                    "severity": "info",
                    "certainty": "uncertain",
                    "title": "t",
                    "explanation": "e",
                    "evidence_refs": [],
                    "suggestion": "g",
                }
            ],
        },
        ensure_ascii=False,
    )
    with pytest.raises(AuditOutputError, match="至少需要一个有效证据引用"):
        parse_audit_report(empty_refs, model_id="a", model="m", evidence_ids=ids)


def test_illegal_output_error_does_not_echo_model_content() -> None:
    secret = "模型原文密钥样式 SK-SECRET-12345"
    raw = json.dumps(
        {
            "summary": "s",
            "findings": [
                {
                    "severity": secret,
                    "certainty": "certain",
                    "title": "t",
                    "explanation": "e",
                    "evidence_refs": ["E-0001"],
                    "suggestion": "g",
                }
            ],
        },
        ensure_ascii=False,
    )
    with pytest.raises(AuditOutputError) as info:
        parse_audit_report(raw, model_id="a", model="m", evidence_ids=frozenset({"E-0001"}))
    assert secret not in str(info.value)


def test_session_run_is_single_use_and_guard_blocks_repeat(tmp_path: Path) -> None:
    config = AuditConfig(models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),))
    sampler = FakeAuditSampler([VALID_AUDIT])
    session = AuditSession(config, lambda spec: sampler)

    async def run_once() -> None:
        await session.run(
            scene=_scene(),
            records=_records(),
            summary=_summary(),
            output_directory=tmp_path,
            incomplete=False,
        )

    asyncio.run(run_once())
    assert sampler.calls == 1
    with pytest.raises(RuntimeError, match="只能执行一次"):
        asyncio.run(run_once())
    assert sampler.calls == 1


def test_session_concurrent_run_only_sends_once(tmp_path: Path) -> None:
    config = AuditConfig(models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),))
    sampler = FakeAuditSampler([VALID_AUDIT])
    session = AuditSession(config, lambda spec: sampler)

    async def call() -> str:
        try:
            await session.run(
                scene=_scene(),
                records=_records(),
                summary=_summary(),
                output_directory=tmp_path,
                incomplete=False,
            )
            return "ok"
        except RuntimeError as error:
            return f"rejected:{error}"

    async def both() -> list[str]:
        return list(await asyncio.gather(call(), call()))

    results = asyncio.run(both())
    assert sampler.calls == 1
    assert any(result.startswith("rejected") for result in results)
    assert "ok" in results


def test_mark_failed_keeps_report_scope_and_separate_error(tmp_path: Path) -> None:
    config = AuditConfig(models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),))
    session = AuditSession(config, lambda spec: FakeAuditSampler([VALID_AUDIT]))
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file", encoding="utf-8")
    asyncio.run(
        session.run(
            scene=_scene(),
            records=_records(),
            summary=_summary(),
            output_directory=blocker,
            incomplete=False,
        )
    )
    state = session.state.snapshot()
    assert state["status"] == "failed"
    assert state["scope"], "已取得的报告范围保留"
    assert state["error"], "真实错误说明单独保存"


def test_audit_cancel_after_export_preserves_success_summary(tmp_path: Path) -> None:
    config = load_assembly_config(setup_config(tmp_path))
    audit_config = AuditConfig(
        models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),)
    )
    session = AuditSession(audit_config, lambda spec: CancelAuditSampler())

    async def run() -> tuple:
        return await run_headless(
            config,
            build_scripted_factory(SCRIPTS),
            output_directory=tmp_path / "out",
            audit=session,
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "time_limit"
    assert summary["error"] is None
    state = session.state.snapshot()
    assert state["status"] == "cancelled"
    assert "审计被取消" in (state["error"] or "")


def test_run_result_error_marks_audit_partial(tmp_path: Path) -> None:
    """scheduler.run 返回 RunResult(error) 时审计必须标 incomplete。"""
    config = load_assembly_config(setup_config(tmp_path))
    audit_config = AuditConfig(
        models=(AuditModelSpec("a", "tests.fakes:create_model", {"model": "m"}),)
    )
    session = AuditSession(audit_config, lambda spec: FakeAuditSampler([VALID_AUDIT]))

    async def run() -> tuple:
        return await run_headless(
            config,
            build_scripted_factory({}),
            output_directory=tmp_path / "out",
            audit=session,
        )

    summary, out_dir = asyncio.run(run())
    assert summary.error, "无脚本导致组件错误，RunResult 携带真实错误"
    payload = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["incomplete"] is True
    assert "未正常结束" in payload["scope"]


def test_cli_error_run_still_prints_audit_and_merges_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模拟 RunResult(error) 时 CLI 仍打印模拟概要并打印审计状态。"""
    config = setup_config(tmp_path)
    script_path = tmp_path / "scripted.json"
    script_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
    audit_path = _write_audit_config(tmp_path, {"models": [{"id": "a", "options": {"model": "m"}}]})
    monkeypatch.setattr(cli_main, "_load_live_dotenv", lambda: None)
    monkeypatch.setattr(
        cli_main, "_default_audit_factory", lambda spec: FakeAuditSampler([VALID_AUDIT])
    )
    exit_code = cli_main.main(
        [
            str(config),
            "--scripted",
            str(script_path),
            "--audit-config",
            str(audit_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "停止原因：error" in captured.out
    assert "审计状态：done" in captured.out
    payload = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    assert payload["incomplete"] is True
