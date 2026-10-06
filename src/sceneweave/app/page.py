"""提取的 UI 页面资源（textContent 渲染，无 innerHTML 注入）。

页面只从两处固定端点取数据：``/run-snapshot``（本次运行的真实周期
历史、最终状态与解引用的模型调用/跳过记录）与 ``/status``（运行/暂停
标签）。SSE 只作为变化提醒，触发重新拉取快照；刷新页面后仍能从快照
恢复完整历史。所有模型、角色与事件文本一律经 textContent 写入 DOM，
不拼 HTML、不写 script，恶意文本不会执行。
"""

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
  button:disabled { cursor: not-allowed; opacity: .5; }
  main { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit,
         minmax(360px, 1fr)); margin-top: 1rem; align-items: start; }
  section { background: white; border: 1px solid #d8dde5; border-radius: 8px;
            padding: .8rem 1rem; }
  section h2 { font-size: .95rem; margin: 0 0 .5rem; color: #35507a; }
  h4 { font-size: .85rem; margin: .6rem 0 .2rem; color: #35507a; }
  ul { margin: 0; padding-left: 1.1rem; }
  li { margin: .15rem 0; }
  .muted { color: #6b7686; }
  .empty { color: #6b7686; margin: .3rem 0; }
  .full { grid-column: 1 / -1; }
  .cycle { border: 1px solid #e2e6ec; border-radius: 6px; margin: .5rem 0;
           padding: .4rem .6rem; }
  .cycle > summary { cursor: pointer; font-weight: 600; }
  .role { border-left: 3px solid #7a9a35; padding-left: .6rem; margin: .5rem 0; }
  .role h3 { font-size: .9rem; margin: .2rem 0; }
  .facts li { border-left: 3px solid #7a9a35; padding-left: .5rem; list-style: none; }
  .registry li, .worlddiag li { list-style: none; }
  .call { border: 1px solid #e2e6ec; border-radius: 6px; margin: .5rem 0;
          padding: .4rem .6rem; }
  .call.error { border-color: #b06; background: #fff5f8; }
  .call.skip { background: #f7f8fa; }
  .call-head { font-size: .88rem; font-weight: 600; }
  .call-summary { font-size: .85rem; margin-top: .2rem; }
  .call-meta { color: #6b7686; font-size: .8rem; }
  .filters { display: flex; gap: .8rem; flex-wrap: wrap; align-items: center;
             margin-bottom: .5rem; }
  .filters label { font-size: .85rem; color: #35507a; }
  pre { white-space: pre-wrap; word-break: break-word; background: #f2f4f7;
        padding: .5rem; border-radius: 4px; max-height: 24rem; overflow: auto; }
</style>
</head>
<body>
<header>
  <h1>sceneweave 运行调试视图</h1>
  <span id="time">剧情时刻 — ms</span>
  <span id="events" class="muted"></span>
</header>
<p>
  <button id="btn-play" type="button">运行</button>
  <button id="btn-pause" type="button">暂停</button>
  <button id="btn-step" type="button">单步</button>
  <button id="btn-stop" type="button">停止</button>
  <span id="status" class="muted"></span>
</p>
<main>
  <section class="full">
    <h2>运行过程（每个完整周期）</h2>
    <div id="cycles"></div>
  </section>
  <section class="full">
    <h2>模型调用（含跳过 / 未调用说明）</h2>
    <div class="filters">
      <label for="filter-role">角色 / 作用域</label>
      <select id="filter-role"></select>
      <label for="filter-phase">阶段</label>
      <select id="filter-phase"></select>
      <span id="callcount" class="muted"></span>
    </div>
    <div id="calls"></div>
  </section>
  <section class="full">
    <h2>最终收尾状态</h2>
    <div id="final-state"></div>
  </section>
</main>
<script>
"use strict";
let finished = false;
let lastVersion = null;
let refreshSeq = 0;
let appliedSeq = 0;
let callItems = [];
let refreshTimer = null;
const buttons = {
  play: document.getElementById("btn-play"),
  pause: document.getElementById("btn-pause"),
  step: document.getElementById("btn-step"),
  stop: document.getElementById("btn-stop"),
};
const statusEl = document.getElementById("status");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function displayMs(value) {
  return typeof value === "number" ? value + " ms" : "—";
}

function listItem(parent, text, className) {
  const node = element("li", className, text);
  parent.appendChild(node);
  return node;
}

function listBlock(title, texts, emptyText) {
  const block = element("div");
  block.appendChild(element("h4", null, title));
  const list = element("ul");
  if (texts.length) {
    for (const text of texts) listItem(list, text);
  } else {
    listItem(list, emptyText, "muted");
  }
  block.appendChild(list);
  return block;
}

function postCommand(path) {
  fetch(path, { method: "POST" })
    .then(function () { refreshStatus(); scheduleRefresh(); })
    .catch(function () {});
}

function refreshStatus() {
  fetch("/status", { cache: "no-store" })
    .then(function (response) { return response.json(); })
    .then(function (state) {
      if (finished) return;  // 完成态只由快照的完成信号决定
      if (state.is_running) statusEl.textContent = "（运行中）";
      else if (state.is_paused) statusEl.textContent = "（已暂停）";
      else statusEl.textContent = "（待开始）";
    })
    .catch(function () {});
}

function scheduleRefresh() {
  if (refreshTimer !== null) return;
  refreshTimer = window.setTimeout(function () {
    refreshTimer = null;
    refreshSnapshot();
  }, 80);
}

function versionGreater(left, right) {
  // 服务端版本是递增的整数数组；晚到的旧快照不能覆盖新状态。
  if (right === null || right === undefined) return true;
  const a = Array.isArray(left) ? left : [left];
  const b = Array.isArray(right) ? right : [right];
  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const x = a[index] === undefined ? 0 : Number(a[index]);
    const y = b[index] === undefined ? 0 : Number(b[index]);
    if (x !== y) return x > y;
  }
  return false;
}

function refreshSnapshot() {
  const seq = (refreshSeq += 1);
  fetch("/run-snapshot", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("snapshot " + response.status);
      return response.json();
    })
    .then(function (data) {
      if (seq < appliedSeq) return;  // 已有更新的响应先到，丢弃
      if (finished && data.finished !== true) return;  // 完成态单调，不被旧快照覆盖
      if (!versionGreater(data.version, lastVersion)) return;
      appliedSeq = seq;
      applySnapshot(data);
    })
    .catch(function () {
      const host = document.getElementById("cycles");
      if (!host.childNodes.length) {
        host.appendChild(element("p", "empty", "无法读取运行快照：服务器可能已关闭。"));
      }
    });
}

function applySnapshot(data) {
  lastVersion = data.version;
  finished = data.finished === true;
  for (const key of Object.keys(buttons)) buttons[key].disabled = finished;
  if (finished) {
    const finalLogs = (data.final || {}).entity_logs || {};
    const diagnostics = finalLogs.diagnostics || {};
    statusEl.textContent = "（运行结束：" + (diagnostics.stop_reason || "—") + "）";
  }
  renderHeader(data);
  renderCycles(data.cycles || []);
  renderCalls(data.calls || []);
  renderFinal(data.final || null);
}

function renderHeader(data) {
  const cycles = data.cycles || [];
  const finalWorld = (((data.final || {}).entity_logs || {}).world) || null;
  const lastCycle = cycles.length ? cycles[cycles.length - 1] : null;
  const lastWorld = lastCycle ? ((lastCycle.entity_logs || {}).world || {}) : {};
  const timeEl = document.getElementById("time");
  if (finished && finalWorld) {
    timeEl.textContent = "最终剧情时刻 " + displayMs(finalWorld.simulation_time_ms) +
      "（已完成 " + cycles.length + " 个周期）";
  } else if (cycles.length) {
    timeEl.textContent = "剧情时刻 " + displayMs(lastWorld.simulation_time_ms) +
      "（已完成 " + cycles.length + " 个周期）";
  } else {
    timeEl.textContent = "剧情时刻 — ms（尚未完成周期）";
  }
  const events = lastWorld.events || [];
  document.getElementById("events").textContent =
    events.map(function (event) { return event.kind + " " + event.event_id; }).join("；");
}

function renderCycles(cycles) {
  const host = document.getElementById("cycles");
  const openSteps = {};
  for (const node of host.querySelectorAll("details.cycle[open]")) {
    openSteps[node.dataset.step] = true;  // 新增日志不收起已展开的周期
  }
  host.replaceChildren();
  if (!cycles.length) {
    host.appendChild(element("p", "empty",
      "暂无已完成周期。初始为暂停；点击「单步」或「运行」后，每个完整的事件周期都会按时间顺序保留在这里。"));
    return;
  }
  cycles.forEach(function (data, index) {
    const node = cycleNode(data, index === cycles.length - 1 || openSteps[String(data.step)]);
    node.dataset.step = String(data.step);
    host.appendChild(node);
  });
}

function cycleNode(data, open) {
  const logs = data.entity_logs || {};
  const world = logs.world || {};
  const diagnostics = logs.diagnostics || {};
  const details = element("details", "cycle");
  details.open = !!open;
  const summary = element("summary");
  const cycleId = diagnostics.cycle_id ? " · " + diagnostics.cycle_id : "";
  summary.textContent = "周期 " + data.step + cycleId + " · 剧情时刻 " + displayMs(world.simulation_time_ms);
  details.appendChild(summary);

  const events = world.events || [];
  details.appendChild(listBlock("同刻事件",
    events.map(function (event) {
      return "[" + event.kind + "] " + event.event_id + "：" + event.description;
    }),
    "本轮无同刻事件。"));

  const entityActions = data.entity_actions || {};
  const roleKeys = Object.keys(entityActions);
  for (const key of roleKeys) {
    const roleLog = logs[key] || {};
    const role = element("div", "role");
    role.appendChild(element("h3", null, "角色 " + (roleLog.role_id || key)));
    role.appendChild(element("p", null, "主动声明：" + entityActions[key]));
    const list = element("ul");
    const states = roleLog.action_states || [];
    for (const state of states) {
      listItem(list, "[" + state.phase + "] " + state.action_id + " — " + state.detail);
    }
    const perceptions = roleLog.perceptions || [];
    for (const text of perceptions.slice(0, 5)) listItem(list, "感知：" + text, "muted");
    if (!states.length && !perceptions.length) {
      listItem(list, "本轮无自身行动状态与感知", "muted");
    }
    role.appendChild(list);
    details.appendChild(role);
  }
  if (!roleKeys.length) {
    details.appendChild(element("p", "empty", "本周期没有角色声明记录。"));
  }

  const registry = world.registry_actions || [];
  details.appendChild(listBlock("统一登记后的行动记录",
    registry.map(function (record) {
      return "[" + record.phase + "] " + record.action_id + "（" + record.role_id + "，" +
        record.start_ms + "→" + record.expected_end_ms + "ms）" + record.description;
    }),
    "暂无登记行动"));

  const diagnostic = world.world_diagnostic;
  const diagnosticTexts = [];
  if (diagnostic) {
    for (const state of diagnostic.action_states || []) {
      diagnosticTexts.push("[实际 " + state.state + "] " + state.action_id +
        "（" + state.role_id + "）" + state.detail);
    }
    for (const passive of diagnostic.passive_processes || []) {
      diagnosticTexts.push("[被动] " + passive);
    }
  }
  const diagnosticBlock = listBlock("World 实际行动与被动过程（客观诊断）",
    diagnosticTexts, "暂无实际行动与被动过程");
  diagnosticBlock.className = "worlddiag";
  details.appendChild(diagnosticBlock);

  const factTexts = [];
  for (const fact of world.committed_facts || []) factTexts.push(fact);
  for (const outcome of world.action_outcomes || []) {
    factTexts.push("结算：" + outcome.action_id + " → " + outcome.result +
      "（" + outcome.detail + "）");
  }
  const factBlock = listBlock("事实增量与结算结果", factTexts, "暂无事实增量与结算结果");
  factBlock.className = "facts";
  details.appendChild(factBlock);

  const changes = diagnostics.change_reasons || {};
  details.appendChild(listBlock("诊断：行动变化原因",
    Object.keys(changes).map(function (key) { return key + "：" + changes[key]; }),
    "本周期无行动变化诊断"));
  return details;
}

function fillFilter(selectId, values) {
  const select = document.getElementById(selectId);
  const previous = select.value;
  select.replaceChildren();
  const all = element("option", null, "全部");
  all.value = "";
  select.appendChild(all);
  const unique = Array.from(new Set(values.filter(function (value) {
    return typeof value === "string" && value;
  }))).sort();
  for (const value of unique) {
    const option = element("option", null, value);
    option.value = value;
    select.appendChild(option);
  }
  if (previous && unique.indexOf(previous) >= 0) select.value = previous;
}

function renderCalls(items) {
  callItems = items;
  fillFilter("filter-role", items.map(function (item) { return item.group; }));
  fillFilter("filter-phase", items.map(function (item) { return item.phase; }));
  applyCallFilters();
}

function applyCallFilters() {
  const host = document.getElementById("calls");
  const openEntries = {};
  for (const node of host.querySelectorAll("details[open]")) {
    if (node.dataset.entryIndex !== undefined) openEntries[node.dataset.entryIndex] = true;
  }
  const role = document.getElementById("filter-role").value;
  const phase = document.getElementById("filter-phase").value;
  host.replaceChildren();
  let shown = 0;
  for (const item of callItems) {
    if (role && item.group !== role) continue;
    if (phase && item.phase !== phase) continue;
    shown += 1;
    host.appendChild(callNode(item, openEntries[String(item.entry_index)] === true));
  }
  document.getElementById("callcount").textContent =
    "共 " + callItems.length + " 条，当前显示 " + shown + " 条";
  if (!shown) {
    host.appendChild(element("p", "empty", callItems.length
      ? "没有符合当前筛选的调用记录。"
      : "暂无模型调用记录。初始暂停不会调用模型；点击「单步」查看一个完整周期的模型输入输出与跳过记录。"));
  }
}

function callNode(item, open) {
  const classes = "call" + (item.is_error ? " error" : (item.is_skip ? " skip" : ""));
  const node = element("div", classes);
  const head = element("div", "call-head");
  if (item.kind === "call") {
    head.textContent = (item.is_error ? "模型调用失败" : "模型调用") +
      " · " + item.phase + " · " + item.group;
  } else {
    head.textContent = (item.is_skip ? "未调用模型（跳过）" : "阶段错误") +
      " · " + item.phase + " · " + item.group;
  }
  node.appendChild(head);
  const parts = [];
  if (item.scope) parts.push("作用域 " + item.scope);
  if (item.role_id) parts.push("角色 " + item.role_id);
  if (item.cycle_id) parts.push("周期 " + item.cycle_id);
  if (typeof item.simulation_time_ms === "number") {
    parts.push("剧情时刻 " + item.simulation_time_ms + " ms");
  }
  node.appendChild(element("div", "call-meta", parts.join(" · ")));
  if (item.summary_text) {
    node.appendChild(element("div", "call-summary", item.summary_text));
  }
  const details = element("details");
  details.dataset.entryIndex = String(item.entry_index);
  details.open = open === true;
  details.appendChild(element("summary", null,
    item.kind === "call" ? "展开完整提示与回答" : "查看阶段说明"));
  if (item.kind === "call") {
    details.appendChild(preBlock("提示（Prompt）", item.prompt));
    if (item.is_error) {
      details.appendChild(preBlock("错误（Error）",
        (item.error_type || "错误") + ": " + (item.error_message || "")));
    } else {
      details.appendChild(preBlock("回答（Response）", item.response));
    }
  } else {
    details.appendChild(preBlock("说明", item.summary || ""));
    const inputs = (item.inputs || []).filter(function (text) {
      return text && text !== "（跳过模型调用）";
    });
    if (inputs.length) details.appendChild(preBlock("阶段输入", inputs.join("\\n\\n")));
    const detailKeys = Object.keys(item.details || {});
    if (detailKeys.length) {
      details.appendChild(preBlock("阶段细节", detailKeys.map(function (key) {
        return key + ": " + item.details[key];
      }).join("\\n")));
    }
    if (item.error) details.appendChild(preBlock("错误", item.error));
  }
  node.appendChild(details);
  return node;
}

function preBlock(label, text) {
  const wrap = element("div");
  wrap.appendChild(element("div", "muted", label));
  wrap.appendChild(element("pre", null, text === null || text === undefined ? "（无）" : text));
  return wrap;
}

function renderFinal(finalData) {
  const host = document.getElementById("final-state");
  host.replaceChildren();
  if (!finalData) {
    host.appendChild(element("p", "empty", finished
      ? "运行已结束，但没有收尾步数据。"
      : "运行尚未结束。完成后这里显示真实停止原因、最终剧情时刻与完成周期数。"));
    return;
  }
  const logs = finalData.entity_logs || {};
  const world = logs.world || {};
  const diagnostics = logs.diagnostics || {};
  const list = element("ul");
  listItem(list, "停止原因：" + (diagnostics.stop_reason || "—"));
  listItem(list, "完成周期：" + (diagnostics.completed_cycles === undefined ? "—" : diagnostics.completed_cycles));
  listItem(list, "最终剧情时刻：" + displayMs(world.simulation_time_ms));
  if (finalData.action) listItem(list, "收尾说明：" + finalData.action);
  if (diagnostics.error) listItem(list, "错误：" + diagnostics.error);
  for (const record of world.registry_actions || []) {
    listItem(list, "[最终登记 " + record.phase + "] " + record.action_id +
      "（" + record.role_id + "）" + record.description);
  }
  host.appendChild(list);
}

document.getElementById("btn-play").addEventListener("click", function () { postCommand("/play"); });
document.getElementById("btn-pause").addEventListener("click", function () { postCommand("/pause"); });
document.getElementById("btn-step").addEventListener("click", function () { postCommand("/step"); });
document.getElementById("btn-stop").addEventListener("click", function () { postCommand("/stop"); });
document.getElementById("filter-role").addEventListener("change", applyCallFilters);
document.getElementById("filter-phase").addEventListener("change", applyCallFilters);

const source = new EventSource("/events");
source.onmessage = function (event) {
  let data = null;
  try { data = JSON.parse(event.data); } catch (error) { return; }
  if (data && data.completion) {
    refreshSnapshot();
    refreshStatus();
    return;
  }
  scheduleRefresh();
};
source.onerror = function () {};

refreshStatus();
refreshSnapshot();
window.setInterval(function () { if (!finished) refreshSnapshot(); }, 1500);
window.setInterval(function () { if (!finished) refreshStatus(); }, 1500);
</script>
</body>
</html>
"""
