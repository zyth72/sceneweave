"""提取的 UI 页面资源（textContent 渲染，无 innerHTML 注入）。"""

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
    h.textContent = '角色 ' + ((data.entity_logs[name] || {}).role_id || name);
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
