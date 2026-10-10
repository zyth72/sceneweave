"""用真实 Chromium 验证 System1 分布、程序最终路由和 Markdown 导出。"""

import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from concordia.utils import structured_logging

from sceneweave.app.ui_snapshot import RunSnapshotHistory, build_snapshot
from sceneweave.domain.identity import CycleId, ResponseId, RoleId
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.integrations.concordia.tracing import SimulationLogTracer

from .test_ui_browser import CHROMIUM, PLAYWRIGHT_CORE, _PageHandler, _PageServer

_DRIVER = r"""
const { chromium } = require(process.argv[2]);
const payload = JSON.parse(require('node:fs').readFileSync(process.argv[4], 'utf8'));
(async () => {
  const browser = await chromium.launch({executablePath: process.argv[5], headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.route('**/run-snapshot', route => route.fulfill({json: payload}));
  await page.goto(process.argv[3]);
  await page.waitForFunction(() => document.querySelectorAll('.s1-route').length === 1);
  const result = await page.evaluate(() => ({
    route: document.querySelector('.s1-route').textContent,
    headers: Array.from(document.querySelectorAll('.call-head'), n => n.textContent),
    percentages: Array.from(document.querySelectorAll('.s1-row .pct'), n => n.textContent),
    widths: Array.from(document.querySelectorAll('.s1-fill'), n => n.style.width),
    markdown: buildProcessMarkdown(),
    injection: typeof window.__xss !== 'undefined' ||
      document.querySelectorAll('#calls img, #calls script').length !== 0,
  }));
  result.errors = errors;
  await browser.close();
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });
"""


@pytest.mark.skipif(
    not PLAYWRIGHT_CORE
    or not shutil.which("node")
    or not shutil.which(CHROMIUM)
    or not Path(PLAYWRIGHT_CORE).is_dir(),
    reason="需要本机 node/chromium 与 SCENEWEAVE_TEST_PLAYWRIGHT_CORE",
)
def test_system1_bars_use_real_trace_and_export_actual_forced_route(tmp_path: Path) -> None:
    log = structured_logging.SimulationLog()
    note = '<img src=x onerror="window.__xss=1">本轮影响'
    raw_weights = {"continue": 0.82, "replan": 0.16, "rethink": 0.02}
    raw = json.dumps({"probabilities": raw_weights, "note": note}, ensure_ascii=False)
    log.add_entry(
        step=0,
        timestamp="2026-10-10T00:00:00Z",
        entity_name="role:lin",
        component_name="system1",
        entry_type="component",
        summary="System1 输出倾向",
        raw_data={
            "Component": "system1",
            "Key": "sample_text",
            "Response": raw,
            "role_id": "lin",
            "cycle_id": "c-1",
            "response_id": "r-lin",
        },
    )
    SimulationLogTracer(log).trace(
        TraceRecord(
            stage="system1.decide",
            cycle_id=CycleId("c-1"),
            role_id=RoleId("lin"),
            response_id=ResponseId("r-lin"),
            simulation_time=SimulationTime(0),
            summary="程序强制 rethink",
            outputs=(raw,),
            details={
                "route": "rethink",
                "method": "forced",
                "reason": "no_intention",
                "raw_probabilities": raw_weights,
                "normalized_probabilities": raw_weights,
                "legal_weights": {"rethink": 1.0},
                "allowed_routes": ["rethink"],
                "note": note,
                "max_weight": 1.0,
                "margin": 0.0,
                "certainty_threshold": 0.75,
                "margin_threshold": 0.25,
                "route_seed": 0,
                "seed_hash": "test",
                "random_value": None,
            },
        )
    )
    payload = tmp_path / "snapshot.json"
    payload.write_text(json.dumps(build_snapshot(RunSnapshotHistory(), log)), encoding="utf-8")
    driver = tmp_path / "browser.cjs"
    driver.write_text(_DRIVER, encoding="utf-8")
    server = _PageServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = subprocess.run(
            [
                "node",
                str(driver),
                PLAYWRIGHT_CORE,
                f"http://127.0.0.1:{server.server_port}/",
                str(payload),
                CHROMIUM,
            ],
            text=True,
            capture_output=True,
            timeout=25,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    result = json.loads(run.stdout)
    assert "rethink" in result["route"] and "forced" in result["route"]
    assert any("程序路由决策" in title for title in result["headers"])
    assert all("阶段错误" not in title for title in result["headers"])
    assert "82%" in result["percentages"] and "100%" in result["percentages"]
    assert "82%" in result["widths"] and "100%" in result["widths"]
    assert "rethink" in result["markdown"] and "no\\_intention" in result["markdown"]
    assert "82%" in result["markdown"] and "100%" in result["markdown"]
    assert not result["injection"]
    assert result["errors"] == []
