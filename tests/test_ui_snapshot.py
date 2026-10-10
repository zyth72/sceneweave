"""UI 只读快照：历史归属、版本递增、去重解引用、摘要与运行结束首次访问。

用真实 SimulationServer 子类与真实 SimulationLog，不调用真实模型、
不读取密钥；端到端用真实（假后端）LanguageModel 走完整组件日志通道。
HTTP 全部走本机回环并有界等待，测试失败也会可靠停止控制器。
"""

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from concordia.utils import structured_logging

from sceneweave.app.assembly import build_live_factory
from sceneweave.app.page import PAGE
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import run_headless
from sceneweave.app.ui import run_ui
from sceneweave.app.ui_snapshot import (
    RunSnapshotHistory,
    SnapshotSimulationServer,
    build_snapshot,
)
from sceneweave.config.loader import load_assembly_config

from .test_cli_headless import setup_config
from .test_ui_and_components import _install_fake_model


def _http_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 本机回环
        return json.loads(response.read().decode("utf-8"))


def _http_post(url: str) -> None:
    request = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        assert response.status == 200


def _wait_snapshot(
    url: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = _http_get(url + "/run-snapshot")
        if predicate(latest):
            return latest
        time.sleep(0.05)
    raise AssertionError(f"等待快照条件超时：{latest}")


def _long_text(marker: str) -> str:
    return (marker + "，") * 40


def _add_component(
    log: structured_logging.SimulationLog,
    *,
    step: int,
    scope: str,
    role_id: str,
    component: str,
    raw: dict[str, Any],
    summary: str = "sample_text",
) -> None:
    payload: dict[str, Any] = {
        "Key": "sample_text",
        "Component": component,
        "scope": scope,
        "role_id": role_id,
        "cycle_id": f"cycle-{step + 1}",
        "simulation_time_ms": step * 100,
        "Prompt": _long_text("提示"),
        "Response": '{"route": "rethink", "note": "更新认识"}',
    }
    payload.update(raw)
    log.add_entry(
        step=step,
        timestamp=f"2026-01-01T00:00:0{step}+00:00",
        entity_name=scope,
        component_name=component,
        entry_type="component",
        summary=summary,
        raw_data=payload,
    )


def test_history_separates_cycles_from_final_and_ignores_reads() -> None:
    """周期按广播顺序保留；完成时最后一帧转为最终状态；读取不改变历史。"""
    history = RunSnapshotHistory()
    first = {"step": 1, "action": "周期一"}
    second = {"step": 2, "action": "周期二"}
    history.record_step(first)
    history.record_step(second)
    first["action"] = "被外部修改"
    revision, cycles, final, finished = history.view()
    assert [item["action"] for item in cycles] == ["周期一", "周期二"], "记录时复制快照"
    assert final is None and finished is False and revision == 2
    # 重复读取不产生任何变化。
    assert history.view() == (revision, cycles, None, False)
    history.mark_completed()
    revision2, cycles, final, finished = history.view()
    assert [item["action"] for item in cycles] == ["周期一"]
    assert final is not None and final["action"] == "周期二"
    assert finished is True and revision2 == revision + 1
    history.mark_completed()  # 幂等：再次完成不重复弹栈
    assert history.view() == (revision2 + 1, cycles, final, True)


def test_build_snapshot_resolves_refs_and_records_skip_and_error() -> None:
    """快照解开内容引用；跳过与采样错误如实出现，且不出现哈希引用。"""
    history = RunSnapshotHistory()
    history.record_step({"step": 1, "entity_logs": {"world": {"simulation_time_ms": 0}}})
    log = structured_logging.SimulationLog()
    prompt = _long_text("提示原文")
    response = _long_text("模型回答")
    log.add_entry(
        step=0,
        timestamp="2026-01-01T00:00:00+00:00",
        entity_name="role:lin",
        component_name="system1",
        entry_type="component",
        summary="sample_text",
        raw_data={
            "Key": "sample_text",
            "Component": "system1",
            "Prompt": prompt,
            "Response": response,
            "scope": "role:lin",
            "role_id": "lin",
            "cycle_id": "cycle-1",
            "simulation_time_ms": 0,
        },
    )
    log.add_entry(
        step=1,
        timestamp="2026-01-01T00:00:01+00:00",
        entity_name="role:shen",
        component_name="system1",
        entry_type="component",
        summary="sample_text_error",
        raw_data={
            "Key": "sample_text_error",
            "Component": "system1",
            "Prompt": _long_text("失败提示"),
            "ErrorType": "RuntimeError",
            "ErrorMessage": "示例模型错误",
            "scope": "role:shen",
            "role_id": "shen",
            "cycle_id": "cycle-2",
            "simulation_time_ms": 1000,
        },
    )
    log.add_entry(
        step=0,
        timestamp="2026-01-01T00:00:02+00:00",
        entity_name="role:shen",
        component_name="system2.skipped",
        entry_type="sceneweave",
        summary="System1 判断无需重新考虑，跳过模型调用并维持既有行动",
        raw_data={
            "cycle_id": "cycle-1",
            "simulation_time_ms": 0,
            "role_id": "shen",
            "inputs": ["（跳过模型调用）"],
            "outputs": [],
            "details": {"attempt_stage": "system2.skipped"},
            "error": None,
        },
    )
    payload = build_snapshot(history, log)
    assert set(payload) == {"version", "finished", "cycles", "final", "calls", "audit"}
    assert payload["version"] == [history.revision, 3, 0, 0]
    assert payload["audit"]["enabled"] is False
    assert payload["audit"]["status"] == "disabled"
    assert payload["finished"] is False
    assert payload["cycles"] == [{"step": 1, "entity_logs": {"world": {"simulation_time_ms": 0}}}]
    assert payload["final"] is None
    calls = payload["calls"]
    assert [item["kind"] for item in calls] == ["call", "call", "note"]
    first = calls[0]
    assert first["prompt"] == prompt and first["response"] == response, "引用必须解出原文"
    assert first["group"] == "lin" and first["phase"] == "system1"
    assert first["cycle_id"] == "cycle-1" and first["simulation_time_ms"] == 0
    assert first["summary_text"] == "回答不是合法 JSON；展开可查看原始回答。"
    error_call = calls[1]
    assert error_call["is_error"] is True
    assert error_call["error_type"] == "RuntimeError"
    assert error_call["error_message"] == "示例模型错误"
    assert error_call["response"] is None
    assert error_call["summary_text"] == "调用失败：RuntimeError: 示例模型错误"
    skip = calls[2]
    assert skip["is_skip"] is True and skip["is_error"] is False
    assert skip["phase"] == "system2.skipped" and skip["group"] == "shen"
    assert skip["summary_text"] == "System1 判断无需重新考虑，跳过模型调用并维持既有行动"
    text = json.dumps(payload, ensure_ascii=False)
    assert "_ref" not in text, "浏览器不得看到哈希引用"


def test_new_model_log_changes_version_and_appears_without_new_steps() -> None:
    """同一周期内新增模型日志也改变版本标识，页面应能看到新记录。"""
    history = RunSnapshotHistory()
    history.record_step({"step": 1, "entity_logs": {"world": {"simulation_time_ms": 0}}})
    log = structured_logging.SimulationLog()
    _add_component(log, step=0, scope="role:lin", role_id="lin", component="system1", raw={})
    before = build_snapshot(history, log)
    _add_component(
        log,
        step=0,
        scope="role:shen",
        role_id="shen",
        component="system1",
        raw={"Response": '{"route": "continue", "note": "保持"}'},
    )
    after = build_snapshot(history, log)
    assert after["version"] != before["version"], "日志增长必须改变版本"
    assert after["version"][0] == before["version"][0], "没有新周期时修订号不变"
    assert after["version"][1] == before["version"][1] + 1
    assert len(after["calls"]) == len(before["calls"]) + 1
    assert after["calls"][-1]["group"] == "shen"


def test_completion_during_log_capture_keeps_polling_until_all_calls_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """快照读取期间运行收尾：不能以完成态返回尚未包含末次调用的日志。"""
    history = RunSnapshotHistory()
    log = structured_logging.SimulationLog()
    original_snapshot = log.to_dict

    def finish_during_capture() -> dict[str, Any]:
        captured = original_snapshot()
        _add_component(
            log,
            step=0,
            scope="role:lin",
            role_id="lin",
            component="system1",
            raw={"Response": '{"route": "continue", "note": "最后一次判断"}'},
        )
        history.record_step(
            {"step": 1, "entity_logs": {"diagnostics": {"stop_reason": "time_limit"}}}
        )
        history.mark_completed()
        return captured

    monkeypatch.setattr(log, "to_dict", finish_during_capture)
    racing = build_snapshot(history, log)
    assert racing["finished"] is False, "日志尚未完整时页面必须继续轮询"
    assert racing["calls"] == []

    monkeypatch.setattr(log, "to_dict", original_snapshot)
    completed = build_snapshot(history, log)
    assert completed["finished"] is True
    assert len(completed["calls"]) == 1
    assert "最后一次判断" in completed["calls"][0]["response"]


def test_call_summaries_from_real_answer_fields() -> None:
    """摘要从真实字段提炼；缺字段与非 JSON 不编造事实。"""
    history = RunSnapshotHistory()
    log = structured_logging.SimulationLog()
    cases = (
        (
            "system1",
            '{"route": "continue", "note": "无需改变"}',
            "分流：沿用已保存计划；说明：无需改变",
        ),
        ("system2", '{"intention": "走过去"}', "意向：走过去"),
        (
            "system3",
            '[{"kind": "wait"}]',
            "结构化动作计划：1 个动作",
        ),
        (
            "world-check",
            '{"passed": false, "reason": "距离太远", "action_ids": ["a1"]}',
            "检查未通过；原因：距离太远",
        ),
        (
            "world-progress",
            '{"movements": [], "updated_action_ids": [], "notes": ["无进展"], "passive_updates": []}',
            "运动 0 项；更新 0 项；被动 0 项；说明：无进展",
        ),
        (
            "translator",
            '{"schedules": []}',
            "新安排 0 项",
        ),
        ("system1", '{"route": "rethink"}', "回答为 JSON 对象；展开可查看原始回答。"),
        ("system2", "不是JSON", "回答不是合法 JSON；展开可查看原始回答。"),
    )
    for index, (component, response, expected) in enumerate(cases):
        _add_component(
            log,
            step=0,
            scope="role:lin",
            role_id="lin",
            component=component,
            raw={"Response": response},
        )
        payload = build_snapshot(history, log)
        assert payload["calls"][index]["summary_text"] == expected, component


def test_snapshot_server_fixed_readonly_endpoint_and_capture() -> None:
    """固定只读端点：仅精确路径 GET；广播被真实捕获；其它路径 404。"""
    history = RunSnapshotHistory()
    log = structured_logging.SimulationLog()
    server = SnapshotSimulationServer(
        port=0,
        html_content=PAGE,
        history=history,
        log_provider=lambda: log,
    )
    server.start()
    try:
        address = server._server.server_address  # noqa: SLF001 测试读取实际端口
        base = f"http://127.0.0.1:{address[1]}"
        initial = _http_get(base + "/run-snapshot")
        assert initial == {
            "version": [0, 0, 0, 0],
            "finished": False,
            "cycles": [],
            "final": None,
            "calls": [],
            "audit": {
                "enabled": False,
                "status": "disabled",
                "revision": 0,
                "scope": "",
                "error": None,
                "evidence_count": 0,
                "checks": [],
                "models": [],
                "failed_models": 0,
            },
        }
        # 固定路径：查询串、其它文件路径与 POST 都不是端点。
        for path in ("/run-snapshot?path=/etc/passwd", "/etc/passwd", "/status?x=1"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path, timeout=5)  # noqa: S310
            assert error.value.code == 404
        request = urllib.request.Request(base + "/run-snapshot", method="POST", data=b"")
        with pytest.raises(urllib.error.HTTPError) as post_error:
            urllib.request.urlopen(request, timeout=5)  # noqa: S310
        assert post_error.value.code == 404
        # HTML 页面可访问，且不拼接 innerHTML。
        with urllib.request.urlopen(base + "/", timeout=5) as response:  # noqa: S310
            html = response.read().decode("utf-8")
        assert "/run-snapshot" in html
        assert "innerHTML" not in html
    finally:
        server.stop()


def test_ui_end_to_end_snapshot_history_calls_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 run_ui：初始暂停无调用、单步完整周期、结束后首次请求有全历史。"""
    _install_fake_model(tmp_path)
    release = threading.Event()
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: release.wait(60))
    ready = threading.Event()
    holder: dict[str, str] = {}

    def on_ready(address: str, port: int) -> None:
        holder["url"] = address
        ready.set()

    config = load_assembly_config(setup_config(tmp_path))
    factory = build_live_factory("fake_live_for_tests:factory", {})
    result_holder: dict[str, object] = {}
    errors_holder: list[BaseException] = []

    def runner() -> None:
        async def scenario() -> tuple[Any, Path]:
            return await run_ui(
                config,
                factory,
                port=0,
                output_directory=tmp_path / "snapshot-out",
                wait_after_finish=True,
                on_server_ready=on_ready,
            )

        try:
            result_holder["result"] = asyncio.run(scenario())
        except BaseException as error:  # noqa: BLE001 测试需要收集线程内异常
            errors_holder.append(error)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    url = ""
    stepped: dict[str, Any] = {}
    final_snapshot: dict[str, Any] = {}
    try:
        assert ready.wait(15), "服务器未在时限内就绪"
        url = holder["url"]
        # 初始暂停：没有模型调用，也没有周期历史。
        initial = _http_get(url + "/run-snapshot")
        assert initial["cycles"] == []
        assert initial["calls"] == []
        assert initial["finished"] is False
        # 单步：一个完整周期的历史与调用记录都在同一快照里。
        _http_post(url + "/step")
        stepped = _wait_snapshot(url, lambda snapshot: len(snapshot["cycles"]) >= 1)
        assert len(stepped["cycles"]) == 1, "单步恰好完成一个完整周期"
        calls = stepped["calls"]
        assert calls, "单步后应有真实组件模型调用记录"
        assert any(
            item["kind"] == "call" and item["prompt"] and item["response"] for item in calls
        ), "调用记录必须解出提示与回答原文"
        assert any(item["kind"] == "note" and item["is_skip"] for item in calls), "跳过也保留"
        assert any(item.get("summary_text") for item in calls), "可见回答摘要"
        assert "_ref" not in json.dumps(stepped, ensure_ascii=False)
        # 播放到时限；运行结束后的请求（新连接）应看到完整历史与最终状态。
        _http_post(url + "/play")
        final_snapshot = _wait_snapshot(url, lambda snapshot: snapshot["finished"] is True)
        diagnostics = final_snapshot["final"]["entity_logs"]["diagnostics"]
        assert diagnostics["stop_reason"] == "time_limit"
        assert final_snapshot["final"]["entity_logs"]["world"]["simulation_time_ms"] == 8000
        # 周期与最终收尾状态保持区分：历史周期不含停止原因，收尾帧不含周期身份。
        assert all(
            "stop_reason" not in cycle["entity_logs"]["diagnostics"]
            for cycle in final_snapshot["cycles"]
        )
        assert "cycle_id" not in diagnostics
        # 重复同步与陈旧 /status 不改变快照，也不产生重复历史。
        _http_get(url + "/status")
        again = _http_get(url + "/run-snapshot")
        assert again == final_snapshot
    finally:
        release.set()
        if url:
            try:
                _http_post(url + "/stop")
            except OSError:
                pass  # 运行已结束、服务器已关闭属正常
        thread.join(timeout=30)
    assert not thread.is_alive(), "run_ui 线程必须在测试结束前退出"
    assert not errors_holder, f"run_ui 线程异常：{errors_holder}"
    summary, out_dir = result_holder["result"]  # type: ignore[misc]
    assert summary.stop_reason == "time_limit"
    # 快照的真实周期数与线程结果、收尾帧三者一致。
    assert len(final_snapshot["cycles"]) == summary.completed_cycles
    assert diagnostics["completed_cycles"] == summary.completed_cycles
    assert (out_dir / "simulation_log.json").is_file()


def test_ui_replayed_component_calls_in_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重放运行：快照仍显示解引用的真实 Prompt/Response，并标记 replay。"""
    _install_fake_model(tmp_path)
    config = load_assembly_config(setup_config(tmp_path))

    async def record() -> None:
        await run_headless(
            config,
            build_live_factory("fake_live_for_tests:factory", {}),
            output_directory=tmp_path / "record-run",
            record_directory=tmp_path / "record",
        )

    asyncio.run(record())
    replay = RecordedRun(tmp_path / "record", config)
    release = threading.Event()
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: release.wait(60))
    ready = threading.Event()
    holder: dict[str, str] = {}
    errors: list[BaseException] = []

    def on_ready(address: str, port: int) -> None:
        holder["url"] = address
        ready.set()

    def runner() -> None:
        async def scenario() -> None:
            await run_ui(
                config,
                replay.for_scope,
                port=0,
                output_directory=tmp_path / "replay-out",
                replay_run=replay,
                wait_after_finish=True,
                on_server_ready=on_ready,
            )

        try:
            asyncio.run(scenario())
        except BaseException as error:  # noqa: BLE001 测试需要收集线程内异常
            errors.append(error)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    url = ""
    try:
        assert ready.wait(15), "重放服务器未在时限内就绪"
        url = holder["url"]
        _http_post(url + "/play")
        snapshot = _wait_snapshot(url, lambda item: item["finished"] is True)
        replay_calls = [item for item in snapshot["calls"] if item.get("replay")]
        assert replay_calls, "重放也写入可查询的组件调用"
        assert any(item["prompt"] and item["response"] for item in replay_calls)
        assert "_ref" not in json.dumps(snapshot, ensure_ascii=False)
        replay.assert_fully_consumed()
    finally:
        release.set()
        if url:
            try:
                _http_post(url + "/stop")
            except OSError:
                pass  # 运行已结束、服务器已关闭属正常
        thread.join(timeout=30)
    assert not thread.is_alive(), "run_ui 线程必须在测试结束前退出"
    assert not errors, f"replay run_ui 线程异常：{errors}"


def test_ui_snapshot_moves_final_frame_out_of_history() -> None:
    """收尾帧从周期历史中撤出：周期数等于完成周期数，不与最终状态重复。"""
    history = RunSnapshotHistory()
    for step in (1, 2):
        history.record_step({"step": step, "entity_logs": {"world": {"simulation_time_ms": step}}})
    assert len(history.view()[1]) == 2
    history.mark_completed()
    _, cycles, final, finished = history.view()
    assert len(cycles) == 1 and final is not None and final["step"] == 2 and finished


def test_snapshot_endpoint_closed_after_server_stop() -> None:
    """服务器关闭后端口不再接受连接；快照只在服务器存活期间可读。"""
    history = RunSnapshotHistory()
    log = structured_logging.SimulationLog()
    server = SnapshotSimulationServer(
        port=0,
        html_content=PAGE,
        history=history,
        log_provider=lambda: log,
    )
    server.start()
    address = server._server.server_address  # noqa: SLF001
    server.stop()
    probe = socket.socket()
    try:
        with pytest.raises(OSError):
            probe.settimeout(2)
            probe.connect(("127.0.0.1", address[1]))
    finally:
        probe.close()


def test_snapshot_audit_state_revision_and_disabled_compatibility() -> None:
    """审计状态独立进快照；未开启时保持 enabled=False 且旧字段不变。"""
    from sceneweave.audit.session import AuditState

    history = RunSnapshotHistory()
    log = structured_logging.SimulationLog()
    state = AuditState(["a", "b"], {"a": "m1", "b": "m2"})
    state.mark_checking()
    first = build_snapshot(history, log, state)
    assert first["audit"]["enabled"] is True
    assert first["audit"]["status"] == "checking"
    assert first["audit"]["revision"] == state.snapshot()["revision"]
    assert first["version"][-1] == first["audit"]["revision"]
    state.mark_running("a")
    second = build_snapshot(history, log, state)
    assert second["version"] != first["version"], "审计进度变化必须改变版本"
    assert second["audit"]["models"][0]["status"] == "running"
    disabled = build_snapshot(history, log)
    assert disabled["audit"] == {
        "enabled": False,
        "status": "disabled",
        "revision": 0,
        "scope": "",
        "error": None,
        "evidence_count": 0,
        "checks": [],
        "models": [],
        "failed_models": 0,
    }
