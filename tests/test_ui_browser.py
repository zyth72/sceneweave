"""页面 DOM 的真实浏览器行为验证（Playwright + 本机 Chromium）。

验证：解引用的记录以 textContent 渲染、恶意文本不执行、回答摘要可见、
角色/阶段筛选、完成态锁定按钮、刷新后历史一致，以及晚到的旧快照不会
覆盖新完成态。环境没有 node/chromium/playwright-core 时如实跳过。
"""

import http.server
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from sceneweave.app.page import PAGE

# 浏览器验证使用调用方提供的本机依赖路径，不绑定开发者的用户目录。
PLAYWRIGHT_CORE = os.environ.get("SCENEWEAVE_TEST_PLAYWRIGHT_CORE", "")
CHROMIUM = shutil.which("chromium") or "chromium"

_JS_DRIVER = r"""
'use strict';
const { chromium } = require(process.argv[2]);
const url = process.argv[3];
const payloads = JSON.parse(require('node:fs').readFileSync(process.argv[4], 'utf8'));

(async () => {
  const browser = await chromium.launch({
    executablePath: process.argv[5],
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  let snapshotCalls = 0;
  await page.route('**/run-snapshot', async (route) => {
    snapshotCalls += 1;
    if (snapshotCalls === 1) {
      await route.fulfill({ json: payloads.intermediate });
    } else if (snapshotCalls === 2) {
      await new Promise((resolve) => setTimeout(resolve, 1800));
      await route.fulfill({ json: payloads.stale });
    } else if (snapshotCalls === 3) {
      await route.fulfill({ json: payloads.grown });
    } else {
      await route.fulfill({ json: payloads.final });
    }
  });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.querySelectorAll('.cycle').length === 1 &&
          document.querySelectorAll('.call').length === 1,
    null,
    { timeout: 15000 },
  );
  // 同时触发两次刷新：先发的旧快照被延迟，后发的新快照（无新 StepData、
  // 只新增模型日志）先返回。
  await page.evaluate(() => { window.refreshSnapshot(); window.refreshSnapshot(); });
  await page.waitForFunction(
    () => document.querySelectorAll('.cycle').length === 1 &&
          document.querySelectorAll('.call').length === 2,
    null,
    { timeout: 15000 },
  );
  const logGrowthApplied = await page.evaluate(() => ({
    cycles: document.querySelectorAll('.cycle').length,
    calls: document.querySelectorAll('.call').length,
    finished: document.getElementById('status').textContent.indexOf('运行结束') >= 0,
  }));
  // 触发最终态；随后等被延迟的旧快照真正返回，确认没有回退。
  await page.evaluate(() => { window.refreshSnapshot(); });
  await page.waitForFunction(
    () => document.querySelectorAll('.cycle').length === 2 &&
          document.querySelectorAll('.call').length === 3 &&
          Array.prototype.every.call(
            document.querySelectorAll('button:not(#btn-export)'),
            (button) => button.disabled),
    null,
    { timeout: 15000 },
  );
  await page.waitForTimeout(2300);
  const result = await page.evaluate(() => {
    const text = document.body.textContent;
    const detail = {
      cycles: document.querySelectorAll('.cycle').length,
      calls: document.querySelectorAll('.call').length,
      errors: document.querySelectorAll('.call.error').length,
      skips: document.querySelectorAll('.call.skip').length,
      buttonsDisabled: Array.prototype.every.call(
        document.querySelectorAll('button:not(#btn-export)'), (button) => button.disabled),
      exportEnabled: !document.getElementById('btn-export').disabled,
      statusFinished: document.getElementById('status').textContent.indexOf('运行结束') >= 0,
      finalVisible: text.indexOf('停止原因：time_limit') >= 0,
      noActionVisible: text.indexOf('（无主动行为声明）') >= 0,
      errorSummaryVisible: text.indexOf('调用失败：RuntimeError') >= 0,
      skipSummaryVisible: text.indexOf('未调用模型（跳过）') >= 0,
      system1SummaryVisible: text.indexOf('重新考虑：否') >= 0,
      rawPromptVisible: text.indexOf('<script>window.__xss=2</script>提示原文') >= 0,
      rawResponseVisible: text.indexOf('<img src=x onerror=window.__xss=1>') >= 0,
      xssExecuted: typeof window.__xss !== 'undefined',
      injectedNodes: document.querySelectorAll(
        '#cycles img, #calls img, #calls script, #cycles script').length,
    };
    const roleSelect = document.getElementById('filter-role');
    roleSelect.value = 'lin';
    roleSelect.dispatchEvent(new Event('change'));
    detail.filteredLin = document.querySelectorAll('.call').length;
    roleSelect.value = '';
    roleSelect.dispatchEvent(new Event('change'));
    const phaseSelect = document.getElementById('filter-phase');
    phaseSelect.value = 'system2.skipped';
    phaseSelect.dispatchEvent(new Event('change'));
    detail.filteredSkipPhase = document.querySelectorAll('.call').length;
    phaseSelect.value = '';
    phaseSelect.dispatchEvent(new Event('change'));
    detail.filteredAll = document.querySelectorAll('.call').length;
    return detail;
  });
  // 刷新后一致：最终态与完整历史不重复、不丢失。
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(
    () => document.querySelectorAll('.cycle').length === 2,
    null,
    { timeout: 15000 },
  );
  const afterReload = await page.evaluate(() => ({
    cycles: document.querySelectorAll('.cycle').length,
    calls: document.querySelectorAll('.call').length,
    finalVisible: document.body.textContent.indexOf('停止原因：time_limit') >= 0,
  }));
  await browser.close();
  console.log('RESULT ' + JSON.stringify(
    { snapshotCalls, logGrowthApplied, result, afterReload, pageErrors }));
})().catch((error) => {
  console.error('BROWSER_FAILED ' + (error && error.stack ? error.stack : error));
  process.exit(1);
});
"""


class _PageServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _PageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 标准库命名
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            body = json.dumps({"is_running": False, "is_paused": True, "current_step": 99}).encode(
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args: Any) -> None:
        del args  # 静默测试服务器日志


def _cycle(step: int, cycle_id: str, time_ms: int) -> dict[str, Any]:
    return {
        "step": step,
        "acting_entity": "scheduler",
        "action": f"剧情时刻 {time_ms} 毫秒（周期 {cycle_id}）",
        "entity_actions": {"lin": "（无主动行为声明）"},
        "entity_logs": {
            "world": {
                "simulation_time_ms": time_ms,
                "events": [
                    {
                        "event_id": f"event-{step}",
                        "kind": "world",
                        "description": "<svg onload=window.__xss=3>" if step == 1 else "安静",
                    }
                ],
                "committed_facts": [],
                "action_outcomes": [],
                "registry_actions": [],
                "world_diagnostic": None,
            },
            "diagnostics": {"cycle_id": cycle_id, "change_reasons": {}},
            "lin": {"perceptions": [], "action_states": []},
        },
        "game_master": "world",
    }


def _call(
    entry_index: int,
    *,
    group: str,
    phase: str,
    summary_text: str,
    prompt: str,
    response: str | None = None,
    is_error: bool = False,
    is_skip: bool = False,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "call" if not is_skip else "note",
        "entry_index": entry_index,
        "step": 0,
        "cycle_id": "cycle-1",
        "simulation_time_ms": 0,
        "scope": f"role:{group}" if not is_skip else None,
        "role_id": group,
        "group": group,
        "phase": phase,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "summary": summary_text,
        "summary_text": summary_text,
        "is_error": is_error,
        "is_skip": is_skip,
        "replay": False,
        "prompt": prompt,
        "response": response,
        "error_type": error_type,
        "error_message": error_message,
    }


def _final() -> dict[str, Any]:
    return {
        "step": 2,
        "acting_entity": "scheduler",
        "action": "运行结束：time_limit，最终剧情时刻 8000 毫秒",
        "entity_actions": {},
        "entity_logs": {
            "world": {
                "simulation_time_ms": 8000,
                "events": [],
                "committed_facts": [],
                "action_outcomes": [],
                "registry_actions": [],
                "world_diagnostic": None,
            },
            "diagnostics": {"stop_reason": "time_limit", "completed_cycles": 2, "error": None},
        },
        "game_master": "world",
    }


def _payloads() -> dict[str, Any]:
    good_call = _call(
        0,
        group="lin",
        phase="system1",
        summary_text="重新考虑：否；说明：<img src=x onerror=window.__xss=1>",
        prompt="<script>window.__xss=2</script>提示原文",
        response='{"reconsider": false, "note": "<img src=x onerror=window.__xss=1>"}',
    )
    error_call = _call(
        1,
        group="shen",
        phase="system1",
        summary_text="调用失败：RuntimeError: 示例错误：模型不可用",
        prompt="失败提示",
        is_error=True,
        error_type="RuntimeError",
        error_message="示例错误：模型不可用",
    )
    skip_note = _call(
        2,
        group="shen",
        phase="system2.skipped",
        summary_text="System1 判断无需重新考虑，跳过模型调用并维持既有行动",
        prompt="（跳过模型调用）",
        is_skip=True,
    )
    return {
        "intermediate": {
            "version": [1, 1, 0],
            "finished": False,
            "cycles": [_cycle(1, "cycle-1", 0)],
            "final": None,
            "calls": [good_call],
        },
        "stale": {
            "version": [0, 0, 0],
            "finished": False,
            "cycles": [],
            "final": None,
            "calls": [],
        },
        # 同一周期内只新增模型日志：周期与完成态不变，版本必须变化。
        "grown": {
            "version": [1, 2, 0],
            "finished": False,
            "cycles": [_cycle(1, "cycle-1", 0)],
            "final": None,
            "calls": [
                good_call,
                _call(
                    3,
                    group="lin",
                    phase="system2",
                    summary_text="意向：走过去；放弃：0 项",
                    prompt="追加提示",
                    response='{"intention": "走过去", "abandon": []}',
                ),
            ],
        },
        "final": {
            "version": [2, 3, 1],
            "finished": True,
            "cycles": [_cycle(1, "cycle-1", 0), _cycle(2, "cycle-2", 1000)],
            "final": _final(),
            "calls": [good_call, error_call, skip_note],
        },
    }


@pytest.mark.skipif(
    shutil.which("node") is None
    or shutil.which("chromium") is None
    or not PLAYWRIGHT_CORE
    or not Path(PLAYWRIGHT_CORE).is_dir(),
    reason="需要 node、chromium，以及 SCENEWEAVE_TEST_PLAYWRIGHT_CORE 指定的依赖",
)
def test_page_dom_behavior_in_real_browser(tmp_path: Path) -> None:
    """真实 Chromium 中执行页面脚本，验证渲染、安全与旧快照保护。"""
    server = _PageServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        payload_file = tmp_path / "payloads.json"
        payload_file.write_text(json.dumps(_payloads(), ensure_ascii=False), encoding="utf-8")
        driver = tmp_path / "browser_probe.js"
        driver.write_text(_JS_DRIVER, encoding="utf-8")
        completed = subprocess.run(
            [
                shutil.which("node") or "node",
                str(driver),
                PLAYWRIGHT_CORE,
                url,
                str(payload_file),
                CHROMIUM,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=tmp_path,
        )
        assert completed.returncode == 0, (
            f"浏览器探针失败：\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
        result_lines = [
            line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
        ]
        assert result_lines, f"浏览器探针没有输出结果：{completed.stdout}"
        outcome = json.loads(result_lines[-1].removeprefix("RESULT "))
        result = outcome["result"]
        assert outcome["pageErrors"] == [], f"页面脚本异常：{outcome['pageErrors']}"
        assert result["cycles"] == 2, "周期历史完整"
        assert result["calls"] == 3, "调用/跳过/错误记录都在"
        assert result["errors"] == 1 and result["skips"] == 1
        assert result["buttonsDisabled"] is True, "完成态停用控制按钮"
        assert result["exportEnabled"] is True, "完成态导出按钮仍可用"
        assert result["statusFinished"] is True, "陈旧 /status 不覆盖完成态"
        assert result["finalVisible"] is True
        assert result["noActionVisible"] is True, "无行动声明如实显示"
        assert result["errorSummaryVisible"] is True, "错误摘要可见"
        assert result["skipSummaryVisible"] is True, "跳过如实说明"
        assert result["system1SummaryVisible"] is True, "真实回答摘要默认可见"
        assert result["rawPromptVisible"] is True, "原始提示折叠内容仍在 DOM 文本中"
        assert result["rawResponseVisible"] is True
        assert result["xssExecuted"] is False, "恶意文本不得执行"
        assert result["injectedNodes"] == 0, "不得注入 img/script 节点"
        assert outcome["logGrowthApplied"] == {
            "cycles": 1,
            "calls": 2,
            "finished": False,
        }, "无新 StepData、只新增模型日志时页面必须显示新记录"
        assert result["filteredLin"] == 1, "角色筛选只显示该角色记录"
        assert result["filteredSkipPhase"] == 1, "阶段筛选只看跳过记录"
        assert result["filteredAll"] == 3, "清除筛选后恢复全部记录"
        assert outcome["afterReload"] == {
            "cycles": 2,
            "calls": 3,
            "finalVisible": True,
        }, "刷新后历史与最终态一致且不重复"
        assert outcome["snapshotCalls"] >= 4, "确实发生了重叠刷新"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


_VIEW_JS_DRIVER = r"""
'use strict';
const { chromium } = require(process.argv[2]);
const fs = require('node:fs');
const url = process.argv[3];
const payloads = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));

(async () => {
  const browser = await chromium.launch({
    executablePath: process.argv[5],
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  let current = payloads.empty;
  await page.route('**/run-snapshot', async (route) => {
    await route.fulfill({ json: JSON.parse(JSON.stringify(current)) });
  });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.querySelectorAll('.cycle').length === 2,
    null,
    { timeout: 15000 },
  );

  const initial = await page.evaluate(() => ({
    cycles: document.querySelectorAll('.cycle').length,
    completedHidden: document.getElementById('completed-section').style.display === 'none',
    callsVisible: document.getElementById('calls-section').style.display !== 'none',
    exportEnabled: !document.getElementById('btn-export').disabled,
  }));

  await page.selectOption('#view-mode', 'completed');
  const emptyState = await page.evaluate(() => ({
    rows: document.querySelectorAll('.completed-row').length,
    completedVisible: document.getElementById('completed-section').style.display !== 'none',
    cyclesHidden: document.getElementById('cycles-section').style.display === 'none',
    callsHidden: document.getElementById('calls-section').style.display === 'none',
    emptyHint: document.getElementById('completed').textContent.indexOf(
      '暂无 World 已确认完成的行动') >= 0,
    finalNote: document.getElementById('final-state').textContent.indexOf('运行尚未结束') >= 0,
    timeText: document.getElementById('time').textContent,
  }));

  const [emptyDownload] = await Promise.all([
    page.waitForEvent('download'),
    page.click('#btn-export'),
  ]);
  const emptyMd = fs.readFileSync(await emptyDownload.path(), 'utf8');
  const emptyExport = {
    filename: emptyDownload.suggestedFilename(),
    hasEmptyNote: emptyMd.indexOf('暂无 World 已确认完成的行动') >= 0,
    hasNoFinal: emptyMd.indexOf('运行尚未结束') >= 0,
    hasDisclaimer: emptyMd.indexOf('不等于全局目标必然达成') >= 0,
  };

  current = payloads.full;
  await page.evaluate(() => { window.refreshSnapshot(); });
  await page.waitForFunction(
    () => document.getElementById('status').textContent.indexOf('运行结束') >= 0,
    null,
    { timeout: 15000 },
  );
  const afterFull = await page.evaluate(() => {
    const text = document.getElementById('completed').textContent;
    return {
      mode: document.getElementById('view-mode').value,
      rows: document.querySelectorAll('.completed-row').length,
      completedVisible: document.getElementById('completed-section').style.display !== 'none',
      cyclesHidden: document.getElementById('cycles-section').style.display === 'none',
      callsHidden: document.getElementById('calls-section').style.display === 'none',
      hasCompleted: text.indexOf('a-1') >= 0 && text.indexOf('行动甲完成') >= 0,
      hasFailed: text.indexOf('a-2') >= 0,
      hasUnconfirmed: text.indexOf('a-3') >= 0,
      hasCandidate: text.indexOf('a-candidate') >= 0,
      countText: document.getElementById('completed-count').textContent,
      timeText: document.getElementById('time').textContent,
      finalVisible: document.getElementById('final-state').textContent.indexOf('time_limit') >= 0,
      playDisabled: document.getElementById('btn-play').disabled,
      exportEnabled: !document.getElementById('btn-export').disabled,
      xssExecuted: typeof window.__xss !== 'undefined',
      injectedNodes: document.querySelectorAll('#completed img, #completed script').length,
    };
  });

  const [completedDownload] = await Promise.all([
    page.waitForEvent('download'),
    page.click('#btn-export'),
  ]);
  const completedMd = fs.readFileSync(await completedDownload.path(), 'utf8');
  const completedExport = {
    filename: completedDownload.suggestedFilename(),
    hasCompleted: completedMd.indexOf('a-1') >= 0 && completedMd.indexOf('行动甲完成') >= 0,
    hasMultiline: completedMd.indexOf('\n  第二行说明') >= 0,
    safeHtml: completedMd.indexOf('<img') < 0 && completedMd.indexOf('&lt;img') >= 0,
    safeHeading: completedMd.includes(String.raw`\#\# 假标题`),
    safeLink: completedMd.includes(String.raw`\[伪链接\]\(javascript:alert\(1\)\)`),
    noFailed: completedMd.indexOf('a-2') < 0,
    noUnconfirmed: completedMd.indexOf('a-3') < 0,
    noCandidate: completedMd.indexOf('a-candidate') < 0,
    hasCount: completedMd.indexOf('1 条完成结果，涉及 1 个周期') >= 0,
    hasFinal: completedMd.indexOf(String.raw`停止原因：time\_limit`) >= 0 &&
      completedMd.indexOf('8000 ms') >= 0,
    hasDisclaimer: completedMd.indexOf('不等于全局目标必然达成') >= 0,
  };

  await page.selectOption('#view-mode', 'all');
  await page.selectOption('#filter-role', 'lin');
  const allView = await page.evaluate(() => ({
    mode: document.getElementById('view-mode').value,
    cyclesVisible: document.getElementById('cycles-section').style.display !== 'none',
    callsVisible: document.getElementById('calls-section').style.display !== 'none',
    calls: document.querySelectorAll('.call').length,
  }));
  const [allDownload] = await Promise.all([
    page.waitForEvent('download'),
    page.click('#btn-export'),
  ]);
  const allMd = fs.readFileSync(await allDownload.path(), 'utf8');
  const allExport = {
    filename: allDownload.suggestedFilename(),
    hasCycle: allMd.indexOf('## 周期 1') >= 0,
    hasEvents: allMd.indexOf('### 同刻事件') >= 0 && allMd.indexOf('event-1') >= 0,
    hasLinSummary: allMd.indexOf('林舟摘要') >= 0,
    filteredShen: allMd.indexOf('沈言摘要') < 0,
    noRawPrompt: allMd.indexOf('SECRET_PROMPT_LIN') < 0 &&
      allMd.indexOf('SECRET_PROMPT_WORLD') < 0,
    hasCallsSection: allMd.indexOf('## 模型调用摘要') >= 0,
  };

  await page.selectOption('#filter-role', '');
  const allCallsAfterClear = await page.evaluate(
    () => document.querySelectorAll('.call').length);
  await page.selectOption('#view-mode', 'completed');
  await page.evaluate(() => { window.refreshSnapshot(); });
  await page.waitForTimeout(150);
  const modePersisted = await page.evaluate(() => ({
    mode: document.getElementById('view-mode').value,
    callsHidden: document.getElementById('calls-section').style.display === 'none',
    rows: document.querySelectorAll('.completed-row').length,
  }));

  await browser.close();
  console.log('RESULT ' + JSON.stringify({
    initial, emptyState, emptyExport, afterFull, completedExport,
    allView, allExport, allCallsAfterClear, modePersisted, pageErrors,
  }));
})().catch((error) => {
  console.error('BROWSER_FAILED ' + (error && error.stack ? error.stack : error));
  process.exit(1);
});
"""


def _view_cycle(
    step: int, cycle_id: str, time_ms: int, outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    cycle = _cycle(step, cycle_id, time_ms)
    world = cycle["entity_logs"]["world"]
    world["action_outcomes"] = outcomes
    world["committed_facts"] = [f"周期 {step} 已提交事实"]
    world["registry_actions"] = [
        {
            "action_id": f"a-{step}",
            "role_id": "lin",
            "phase": (
                "completed"
                if any(item["result"] == "completed" for item in outcomes)
                else "started"
            ),
            "start_ms": time_ms,
            "expected_end_ms": time_ms + 10,
            "description": f"周期 {step} 行动",
        }
    ]
    return cycle


def _view_payloads() -> dict[str, Any]:
    empty = {
        "version": [1, 0, 0],
        "finished": False,
        "cycles": [
            _view_cycle(
                1, "cycle-1", 500, [{"action_id": "a-failed", "result": "failed", "detail": "失败"}]
            ),
            _view_cycle(
                2,
                "cycle-2",
                900,
                [{"action_id": "a-open", "result": "unconfirmed", "detail": "未确认"}],
            ),
        ],
        "final": None,
        "calls": [],
    }
    full = {
        "version": [2, 3, 1],
        "finished": True,
        "cycles": [
            _view_cycle(
                1,
                "cycle-1",
                500,
                [
                    {
                        "action_id": "a-1",
                        "result": "completed",
                        "detail": (
                            "行动甲完成 <img src=x onerror=window.__xss=9>\n第二行说明"
                            "\n## 假标题\n[伪链接](javascript:alert(1))"
                        ),
                    },
                    {"action_id": "a-2", "result": "failed", "detail": "行动乙失败"},
                ],
            ),
            _view_cycle(
                2,
                "cycle-2",
                2500,
                [{"action_id": "a-3", "result": "unconfirmed", "detail": "行动丙未确认"}],
            ),
        ],
        "final": _final(),
        "calls": [
            _call(
                0,
                group="lin",
                phase="system1",
                summary_text="林舟摘要",
                prompt="SECRET_PROMPT_LIN",
                response='{"reconsider": false, "note": "x"}',
            ),
            _call(
                1,
                group="shen",
                phase="system1",
                summary_text="沈言摘要",
                prompt="SECRET_PROMPT_SHEN",
                response='{"reconsider": true, "note": "y"}',
            ),
            _call(
                2,
                group="World",
                phase="world-adjudicate",
                summary_text="裁决候选摘要",
                prompt="SECRET_PROMPT_WORLD",
                response=(
                    '{"outcomes": [{"action_id": "a-candidate", "result": "completed",'
                    ' "detail": "候选完成"}]}'
                ),
            ),
        ],
    }
    return {"empty": empty, "full": full}


@pytest.mark.skipif(
    shutil.which("node") is None
    or shutil.which("chromium") is None
    or not PLAYWRIGHT_CORE
    or not Path(PLAYWRIGHT_CORE).is_dir(),
    reason="需要 node、chromium，以及 SCENEWEAVE_TEST_PLAYWRIGHT_CORE 指定的依赖",
)
def test_page_completed_view_and_markdown_export_in_real_browser(tmp_path: Path) -> None:
    """真实 Chromium 中验证完成视图、Markdown 导出与模式稳定。"""
    server = _PageServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        payload_file = tmp_path / "view_payloads.json"
        payload_file.write_text(json.dumps(_view_payloads(), ensure_ascii=False), encoding="utf-8")
        driver = tmp_path / "view_probe.js"
        driver.write_text(_VIEW_JS_DRIVER, encoding="utf-8")
        completed = subprocess.run(
            [
                shutil.which("node") or "node",
                str(driver),
                PLAYWRIGHT_CORE,
                url,
                str(payload_file),
                CHROMIUM,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=tmp_path,
        )
        assert completed.returncode == 0, (
            f"浏览器探针失败：\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
        result_lines = [
            line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
        ]
        assert result_lines, f"浏览器探针没有输出结果：{completed.stdout}"
        outcome = json.loads(result_lines[-1].removeprefix("RESULT "))
        assert outcome["pageErrors"] == [], f"页面脚本异常：{outcome['pageErrors']}"
        assert outcome["initial"] == {
            "cycles": 2,
            "completedHidden": True,
            "callsVisible": True,
            "exportEnabled": True,
        }
        empty = outcome["emptyState"]
        assert empty["rows"] == 0
        assert empty["completedVisible"] and empty["cyclesHidden"] and empty["callsHidden"]
        assert empty["emptyHint"] and empty["finalNote"]
        assert "已完成 2 个周期" in empty["timeText"], "顶部周期总数不因筛选改变"
        empty_export = outcome["emptyExport"]
        assert empty_export["filename"] == "sceneweave-completed.md"
        assert empty_export["hasEmptyNote"]
        assert empty_export["hasNoFinal"]
        assert empty_export["hasDisclaimer"]
        full = outcome["afterFull"]
        assert full["mode"] == "completed", "新快照到达后保持所选模式"
        assert full["rows"] == 1
        assert full["completedVisible"] and full["cyclesHidden"] and full["callsHidden"]
        assert full["hasCompleted"], "只显示 completed 结算"
        assert not full["hasFailed"] and not full["hasUnconfirmed"]
        assert not full["hasCandidate"], "模型候选里的 completed 不计入成功"
        assert "1 条" in full["countText"] and "1 个周期" in full["countText"]
        assert "已完成 2 个周期" in full["timeText"]
        assert full["finalVisible"], "最终收尾状态独立保留"
        assert full["playDisabled"] and full["exportEnabled"]
        assert full["xssExecuted"] is False and full["injectedNodes"] == 0
        completed_export = outcome["completedExport"]
        assert completed_export["filename"] == "sceneweave-completed.md"
        assert completed_export["hasCompleted"]
        assert completed_export["hasMultiline"], "多行说明保留且不破坏结构"
        assert completed_export["safeHtml"], "导出将 HTML 当作文字"
        assert completed_export["safeHeading"] and completed_export["safeLink"]
        assert completed_export["noFailed"] and completed_export["noUnconfirmed"]
        assert completed_export["noCandidate"]
        assert completed_export["hasCount"] and completed_export["hasFinal"]
        assert completed_export["hasDisclaimer"]
        all_view = outcome["allView"]
        assert all_view == {
            "mode": "all",
            "cyclesVisible": True,
            "callsVisible": True,
            "calls": 1,
        }
        all_export = outcome["allExport"]
        assert all_export["filename"] == "sceneweave-process.md"
        assert all_export["hasCycle"] and all_export["hasEvents"]
        assert all_export["hasLinSummary"] and all_export["filteredShen"]
        assert all_export["noRawPrompt"], "导出省略原始 Prompt/Response"
        assert all_export["hasCallsSection"]
        assert outcome["allCallsAfterClear"] == 3, "清除筛选后恢复全部调用"
        assert outcome["modePersisted"] == {
            "mode": "completed",
            "callsHidden": True,
            "rows": 1,
        }, "再次刷新快照后模式稳定"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
