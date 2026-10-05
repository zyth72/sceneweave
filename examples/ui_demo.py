"""本地 UI 演示：Concordia SimulationServer + 真实 StepData 的薄 HTML 视图。

复用 offline_demo 的脚本假件（不调用真实模型、不读取密钥），把每个
完整事件周期转换为 Concordia StepData 广播给浏览器。页面区分：完整
主动声明、角色原始感知与自身行动状态、统一登记后的行动记录、World
实际行动与被动过程的客观诊断、事实增量与同刻全部事件（含行动节点
的类型与身份）。运行结束广播真实最终状态（实际剧情时刻与停止原因），
不复播旧周期报告。所有动态内容经 textContent 构造，不拼接
innerHTML。按钮经服务器端点驱动真实 StepController 的运行、暂停、
单步与停止；许可在每个完整周期前取得，暂停不打断提交。

注意：SimulationServer 按框架默认绑定全部网络接口，请在可信环境
运行（或用防火墙阻断外部访问），不要公开部署。

运行期间可随时用页面的「停止」按钮或 Ctrl+C 中止；两者都会执行清理。
回车退出提示只在运行结束（或中止）后出现：引擎完成后页面保持可查看，
按回车即退出并清理服务器与控制器。运行日志写入结构化 SimulationLog
并导出 JSON 文件（见结束时打印的路径），可离线检索引擎/世界/角色/
翻译的因果链记录。

运行：

    uv run --locked python examples/ui_demo.py [端口]

然后浏览器打开 http://127.0.0.1:8080（或指定端口）。
"""

import asyncio
import sys
from pathlib import Path

from concordia.utils import structured_logging
from concordia.utils.simulation_server import SimulationServer
from offline_demo import LIN, SHEN, DemoRole, DemoTranslator, DemoWorld, PrintingTracer, declare

from sceneweave.domain.identity import EventId
from sceneweave.domain.messages import WorldEvent
from sceneweave.domain.time import SimulationTime
from sceneweave.domain.tracing import TraceRecord
from sceneweave.engine.control import RunGate
from sceneweave.engine.scheduler import RunResult, Scheduler
from sceneweave.integrations.concordia.control import (
    StepControllerGate,
    broadcast_final,
    broadcast_report,
    records_tuple,
)
from sceneweave.integrations.concordia.tracing import SimulationLogTracer

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>sceneweave 运行调试视图</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; color: #1c2430;
         background: #f6f7f9; }
  h1 { font-size: 1.2rem; }
  header { display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap; }
  #time { font-variant-numeric: tabular-nums; font-weight: 700; }
  button { margin-right: .5rem; padding: .35rem 1rem; cursor: pointer; }
  main { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit,
         minmax(320px, 1fr)); margin-top: 1rem; }
  section { background: white; border: 1px solid #d8dde5; border-radius: 8px;
            padding: .8rem 1rem; }
  section h2 { font-size: .95rem; margin: 0 0 .5rem; color: #35507a; }
  ul { margin: 0; padding-left: 1.1rem; }
  li { margin: .15rem 0; }
  .muted { color: #6b7686; }
  #facts li { border-left: 3px solid #7a9a35; padding-left: .5rem; list-style: none; }
  #registry li, #worlddiag li { list-style: none; }
</style>
</head>
<body>
<header>
  <h1>sceneweave 运行调试视图</h1>
  <span id="time">剧情时刻 — ms</span>
  <span id="events" class="muted"></span>
</header>
<p>
  <button onclick="cmd('/play')">运行</button>
  <button onclick="cmd('/pause')">暂停</button>
  <button onclick="cmd('/step')">单步</button>
  <button onclick="cmd('/stop')">停止</button>
  <span id="status" class="muted"></span>
</p>
<main>
  <section id="roles"></section>
  <section id="registry-sec"><h2>统一登记后的行动记录</h2><ul id="registry"></ul></section>
  <section id="worlddiag-sec"><h2>World 实际行动与被动过程（客观诊断）</h2><ul id="worlddiag"></ul></section>
  <section><h2>事实增量与同刻事件</h2><ul id="facts"></ul></section>
  <section><h2>诊断：变更原因 / 停止原因</h2><ul id="reasons"></ul></section>
</main>
<script>
let finished = false;
function cmd(path) { fetch(path, {method: 'POST'}).then(refreshStatus); }
function refreshStatus() {
  if (finished) return;  // 完成态不被状态轮询覆盖
  fetch('/status').then(r => r.json()).then(s => {
    document.getElementById('status').textContent =
      s.is_paused ? '（已暂停）' : (s.is_running ? '（运行中）' : '');
  }).catch(() => {});
}
function li(parent, text, className) {
  const item = document.createElement('li');
  if (className) item.className = className;
  item.textContent = text;  // 只用 textContent，不拼接 innerHTML
  parent.appendChild(item);
}
function render(data) {
  const worldLog = data.entity_logs.world;
  document.getElementById('time').textContent =
    '剧情时刻 ' + worldLog.simulation_time_ms + ' ms（步 ' + data.step + '）';
  const events = document.getElementById('events');
  events.textContent = (worldLog.events || [])
    .map(e => e.kind + ' ' + e.event_id).join('；');
  const roles = document.getElementById('roles');
  roles.innerHTML = '';
  for (const name of Object.keys(data.entity_actions)) {
    const sec = document.createElement('section');
    const h = document.createElement('h2');
    h.textContent = '角色 ' + name;
    sec.appendChild(h);
    const decl = document.createElement('p');
    decl.textContent = '主动声明：' + data.entity_actions[name];
    sec.appendChild(decl);
    const log = data.entity_logs[name] || {};
    const ul = document.createElement('ul');
    for (const state of (log.action_states || [])) {
      li(ul, '[' + state.phase + '] ' + state.action_id + ' — ' + state.detail);
    }
    for (const p of (log.perceptions || []).slice(0, 3)) {
      li(ul, '感知：' + p, 'muted');
    }
    if (!ul.childNodes.length) li(ul, '本轮无自身行动状态与感知', 'muted');
    sec.appendChild(ul);
    roles.appendChild(sec);  }
  const registry = document.getElementById('registry');
  registry.innerHTML = '';
  for (const record of (worldLog.registry_actions || [])) {
    li(registry, '[' + record.phase + '] ' + record.action_id +
      '（' + record.role_id + '，' + record.start_ms + '→' +
      record.expected_end_ms + 'ms）' + record.description);
  }
  if (!registry.childNodes.length) li(registry, '暂无登记行动', 'muted');
  const diag = document.getElementById('worlddiag');
  diag.innerHTML = '';
  const diagnostic = worldLog.world_diagnostic;
  if (diagnostic) {
    for (const state of (diagnostic.action_states || [])) {
      li(diag, '[实际 ' + state.state + '] ' + state.action_id +
        '（' + state.role_id + '）' + state.detail);
    }
    for (const passive of (diagnostic.passive_processes || [])) {
      li(diag, '[被动] ' + passive, 'muted');
    }
  }
  if (!diag.childNodes.length) li(diag, '暂无实际行动与被动过程', 'muted');
  const facts = document.getElementById('facts');
  facts.innerHTML = '';
  for (const fact of (worldLog.committed_facts || [])) li(facts, fact);
  for (const o of (worldLog.action_outcomes || [])) {
    li(facts, '结算：' + o.action_id + ' → ' + o.result + '（' + o.detail + '）');
  }
  const reasons = document.getElementById('reasons');
  reasons.innerHTML = '';
  const diagnostics = data.entity_logs.diagnostics || {};
  for (const [k, v] of Object.entries(diagnostics.change_reasons || {})) {
    li(reasons, k + '：' + v);
  }
  if (diagnostics.stop_reason) {
    li(reasons, '停止原因：' + diagnostics.stop_reason +
      '（完成周期 ' + diagnostics.completed_cycles + '）');
    finished = true;
    document.getElementById('status').textContent =
      '（运行结束：' + diagnostics.stop_reason + '）';
  }
  if (diagnostics.error) li(reasons, '错误：' + diagnostics.error);
}
const source = new EventSource('/events');
source.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.completion) {
    finished = true;  // 最终步数据已渲染，只锁定完成态
    return;
  }
  if (!data.entity_logs) return;
  render(data);
  refreshStatus();
};
refreshStatus();
</script>
</body>
</html>
"""


class BroadcastingGate(RunGate):
    """周期边界：先把刚完成的周期广播给服务器，再等待运行许可。"""

    def __init__(self, server: SimulationServer, scheduler_ref: dict[str, Scheduler | None]):
        self._server = server
        self._ref = scheduler_ref
        self._inner = StepControllerGate(server.step_controller)

    def should_stop(self) -> bool:
        return self._inner.should_stop()

    async def wait_for_permission(self) -> bool:
        scheduler = self._ref["scheduler"]
        assert scheduler is not None
        if scheduler.last_report is not None:
            broadcast_report(self._server, scheduler.last_report)
        return await self._inner.wait_for_permission()


def build_scripts() -> tuple[list, list]:
    from sceneweave.domain.identity import ActionId, CommitGroupId
    from sceneweave.domain.messages import MaintainAction, NewAction

    keep_walk = (MaintainAction(ActionId("a-walk")),)
    lin_script = [
        declare(
            LIN,
            "lin-1",
            (NewAction(ActionId("a-reach"), "隔空拿到桌上钥匙", CommitGroupId("g1")),),
            "cycle-x",
        ),
        declare(
            LIN,
            "lin-2",
            (NewAction(ActionId("a-walk"), "走向书桌", CommitGroupId("g2")),),
            "cycle-x",
        ),
        declare(LIN, "lin-3", keep_walk, "cycle-x"),
        declare(LIN, "lin-4", keep_walk, "cycle-x"),
        declare(LIN, "lin-5", keep_walk, "cycle-x"),
        declare(LIN, "lin-6", (), "cycle-x"),
        declare(
            LIN,
            "lin-7",
            (NewAction(ActionId("a-drawer"), "拉开抽屉", CommitGroupId("g3")),),
            "cycle-x",
        ),
        declare(LIN, "lin-8", (MaintainAction(ActionId("a-drawer")),), "cycle-x"),
        declare(LIN, "lin-9", (), "cycle-x"),
        declare(LIN, "lin-10", (), "cycle-x"),
        declare(LIN, "lin-11", (), "cycle-x"),
    ]
    shen_script = [
        declare(SHEN, "shen-1", (), "cycle-x"),
        declare(SHEN, "shen-2", (), "cycle-x"),
        declare(
            SHEN,
            "shen-3",
            (NewAction(ActionId("a-follow"), "跟随林舟", CommitGroupId("g4")),),
            "cycle-x",
        ),
        declare(SHEN, "shen-4", (MaintainAction(ActionId("a-follow")),), "cycle-x"),
        declare(SHEN, "shen-5", (), "cycle-x"),
        declare(SHEN, "shen-6", (), "cycle-x"),
        declare(SHEN, "shen-7", (), "cycle-x"),
        declare(SHEN, "shen-8", (), "cycle-x"),
        declare(SHEN, "shen-9", (), "cycle-x"),
        declare(SHEN, "shen-10", (), "cycle-x"),
    ]
    return lin_script, shen_script


async def wait_for_quit() -> None:
    """运行结束后等待用户退出；Ctrl+C 或输入流结束同样触发清理。"""
    try:
        await asyncio.to_thread(input, "页面保持可查看，按回车退出并清理服务器...")
    except EOFError:
        pass  # 无交互输入流（重定向运行）时视为立即退出，仍执行清理


class CompositeTracer:
    """组合追踪器：写入 Concordia 结构化日志并打印关键边界。"""

    def __init__(self, log: structured_logging.SimulationLog) -> None:
        self._log_tracer = SimulationLogTracer(log)
        self._printer = PrintingTracer()

    def trace(self, record: TraceRecord) -> None:
        self._log_tracer.trace(record)
        self._printer.trace(record)


def export_log(log: structured_logging.SimulationLog, path: Path) -> None:
    """把结构化日志导出为 JSON 文件，供离线检索因果链。"""
    path.write_text(log.to_json(), encoding="utf-8")


async def main(port: int) -> None:
    from sceneweave.engine.config import RunConfig

    server = SimulationServer(port=port, html_content=PAGE)
    server.start()
    log = structured_logging.SimulationLog()
    log_path = Path(f"ui_demo_log_{port}.json")
    try:
        holder: dict[str, Scheduler | None] = {"scheduler": None}
        lin_script, shen_script = build_scripts()
        world = DemoWorld()
        scheduler = Scheduler(
            roles=[DemoRole(LIN, lin_script), DemoRole(SHEN, shen_script)],
            world=world,
            translator=DemoTranslator(),
            config=RunConfig.from_seconds(
                introspection_interval_seconds=8,
                max_simulation_duration_seconds=30,
            ),
            initial_events=[
                WorldEvent(EventId("startup"), SimulationTime(0), "书房里很安静，午后光线斜照")
            ],
            gate=BroadcastingGate(server, holder),
            tracer=CompositeTracer(log),
        )
        holder["scheduler"] = scheduler
        print(f"UI 演示运行中：http://127.0.0.1:{port}")
        result: RunResult = await scheduler.run()
        # 运行结束广播真实最终状态：实际最终时刻与停止原因。
        broadcast_final(
            server,
            result,
            scheduler.last_report,
            records=records_tuple(scheduler.action_records()),
            diagnostic=world.diagnostic_snapshot(),
        )
        print(
            f"运行结束：{result.reason.value}，完成周期 {result.completed_cycles}，"
            f"最终剧情时刻 {result.final_time}ms"
        )
        if result.error:
            print(f"运行错误：{result.error}")
        export_log(log, log_path)
        print(f"结构化日志已导出：{log_path}")
        await wait_for_quit()
    finally:
        # 先释放许可等待线程，再关闭 HTTP 服务器；两者都必须执行。
        server.step_controller.stop()
        server.stop()
        export_log(log, log_path)
        print("服务器与控制器已清理")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    try:
        asyncio.run(main(port))
    except KeyboardInterrupt:
        pass
