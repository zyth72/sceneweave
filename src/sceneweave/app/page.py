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
  :root {
    --ink: #1c2430; --muted: #5b6675; --faint: #7b8694;
    --line: #e0e5eb; --line-strong: #c9d1db;
    --accent: #35507a; --blue: #2d5f9e; --green: #2f7d4f;
    --amber: #7d5800; --red: #a3344f; --bg: #eef1f4; --card: #ffffff;
  }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body { max-width: 1160px; margin: 0 auto; padding: 1.1rem 1.3rem 3rem;
         font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
         color: var(--ink); background: transparent; font-size: 15px; line-height: 1.55; }
  header.top { display: flex; justify-content: space-between; align-items: flex-start;
               gap: 1rem; flex-wrap: wrap; margin-bottom: .85rem; }
  h1 { font-size: 1.15rem; margin: 0; letter-spacing: .01em; }
  .status-strip { display: flex; flex-direction: column; align-items: flex-end;
                  gap: .1rem; text-align: right; min-width: 0; }
  #time { font-variant-numeric: tabular-nums; font-weight: 700; }
  #events { font-size: .82rem; max-width: 52ch; overflow-wrap: anywhere; }
  .toolbar { display: flex; gap: 1.4rem; flex-wrap: wrap; align-items: center;
             background: var(--card); border: 1px solid var(--line); border-radius: 10px;
             padding: .55rem .8rem; margin-bottom: 1rem; }
  .control-group { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  .group-label { font-size: .76rem; color: var(--faint); letter-spacing: .04em; }
  button { padding: .36rem .9rem; border: 1px solid var(--line-strong); border-radius: 7px;
           background: #fff; color: var(--ink); font-size: .87rem; cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  button:disabled { cursor: not-allowed; opacity: .45; }
  select { padding: .3rem .45rem; border: 1px solid var(--line-strong); border-radius: 7px;
           background: #fff; color: var(--ink); font-size: .87rem; max-width: 100%; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #status { font-size: .85rem; }
  #view-note { font-size: .8rem; }
  main { display: block; }
  section, details.fold-section { background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: .85rem 1rem; margin: 0 0 1rem; }
  section h2 { font-size: .95rem; margin: 0 0 .55rem; color: var(--accent); }
  h4, .subhead { font-size: .83rem; margin: .55rem 0 .2rem; color: var(--accent); }
  ul { margin: 0; padding-left: 1.05rem; }
  li { margin: .18rem 0; overflow-wrap: anywhere; }
  .muted { color: var(--muted); }
  .empty { color: var(--muted); margin: .3rem 0; }
  .cycle { border: 1px solid var(--line); border-left: 3px solid #a9b8cb; border-radius: 8px;
           margin: .5rem 0; background: #fbfcfd; }
  .cycle > summary { cursor: pointer; padding: .5rem .7rem; font-weight: 600;
                     display: flex; gap: .5rem; flex-wrap: wrap; align-items: baseline;
                     list-style-position: inside; }
  .cycle-index { font-weight: 700; color: var(--accent); }
  .cycle-meta { color: var(--muted); font-size: .8rem; font-weight: 400; }
  .cycle-body { padding: 0 .75rem .7rem; }
  .chips { display: inline-flex; gap: .3rem; flex-wrap: wrap; }
  .chip { display: inline-block; padding: .04rem .45rem; border-radius: 999px;
          font-size: .74rem; border: 1px solid var(--line-strong); color: var(--muted);
          background: #fff; white-space: nowrap; vertical-align: baseline; }
  .chip.completed { color: #1f6b43; border-color: #a9d5bb; background: #eefaf3; }
  .chip.failed { color: #93213c; border-color: #e6b6c2; background: #fdf0f3; }
  .chip.unconfirmed { color: var(--amber); border-color: #e6d29a; background: #fcf7e8; }
  .chip.started, .chip.in_progress { color: #24518c; border-color: #b9cce6; background: #eef4fb; }
  .chip.withdrawn, .chip.submitted, .chip.registered { color: #55606e; background: #f2f4f7; }
  .roles { display: grid; gap: .55rem; }
  .role { border: 1px solid var(--line); border-left: 3px solid #7a9a35; border-radius: 8px;
          padding: .5rem .65rem; background: #fff; }
  .role h3 { font-size: .9rem; margin: 0 0 .25rem; }
  .role .declaration { margin: 0 0 .3rem; }
  .more, .tech { border: 1px solid var(--line); border-radius: 8px; margin: .45rem 0;
                 padding: .3rem .55rem; background: #fafbfc; }
  .tech > summary, .more > summary { cursor: pointer; font-size: .83rem; font-weight: 600;
                                     color: var(--accent); }
  .facts { border-left: 3px solid #7a9a35; padding-left: .55rem; margin-top: .5rem; }
  .facts > h4 { margin-top: .2rem; }
  .facts li, .worlddiag li, .registry li { list-style: none; }
  .registry, .worlddiag { padding-left: .1rem; }
  .call { border: 1px solid var(--line); border-radius: 8px; margin: .45rem 0;
          padding: .4rem .6rem; background: #fff; }
  .call.error { border-color: #e6b6c2; background: #fdf0f3; }
  .call.skip { background: #f7f8fa; }
  .call-head { font-size: .87rem; font-weight: 600; overflow-wrap: anywhere; }
  .call-summary { font-size: .85rem; margin-top: .18rem; overflow-wrap: anywhere; }
  .s1-bars { display: grid; gap: .25rem; margin: .4rem 0; }
  .s1-row { display: grid; grid-template-columns: 5.5rem 1fr 3rem; align-items: center;
            gap: .45rem; font-size: .8rem; }
  .s1-row .track { background: #e8ecf1; border-radius: 999px; height: .72rem;
                   overflow: hidden; border: 1px solid var(--line); }
  .s1-row .s1-fill { height: 100%; background: var(--blue); }
  .s1-row .s1-fill.replan { background: #b7791f; }
  .s1-row .s1-fill.rethink { background: var(--red); }
  .s1-row .pct { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .s1-route { font-weight: 600; margin: .35rem 0 .1rem; }
  .call-meta { color: var(--muted); font-size: .78rem; overflow-wrap: anywhere; }
  .filters { display: flex; gap: .8rem; flex-wrap: wrap; align-items: center;
             margin-bottom: .5rem; }
  .filters label { font-size: .85rem; color: var(--accent); }
  .completed-row { border: 1px solid var(--line); border-radius: 8px; margin: .5rem 0;
                   padding: .4rem .6rem; }
  .transcript { border-left: 3px solid #7a9a35; padding-left: .85rem; }
  .transcript-line { margin: .5rem 0; }
  .transcript-text { margin: 0; line-height: 1.6; overflow-wrap: anywhere; }
  .settlement-meta { margin-top: .15rem; font-size: .8rem; color: var(--muted); }
  details.fold-section > summary { cursor: pointer; font-weight: 600; color: var(--accent); }
  .audit-model { border: 1px solid var(--line); border-radius: 8px; margin: .5rem 0;
                 padding: .4rem .6rem; background: #fff; }
  .audit-model.error { border-color: #e6b6c2; background: #fdf0f3; }
  .audit-model .call-head { overflow-wrap: anywhere; }
  pre { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
        background: #f2f4f7; padding: .5rem; border-radius: 6px; max-height: 24rem;
        overflow: auto; max-width: 100%; }
  @media (max-width: 600px) {
    body { padding: .8rem .75rem 2rem; }
    header.top { flex-direction: column; align-items: stretch; }
    .status-strip { align-items: flex-start; text-align: left; }
    .toolbar { gap: .8rem; padding: .5rem .6rem; }
    .control-group { width: 100%; }
    .cycle > summary { padding: .45rem .55rem; }
    .cycle-body { padding: 0 .5rem .55rem; }
    select { max-width: 100%; }
  }
</style>
</head>
<body>
<header class="top">
  <h1>sceneweave 运行调试视图</h1>
  <div class="status-strip">
    <span id="time">剧情时刻 — ms</span>
    <span id="events" class="muted"></span>
  </div>
</header>
<div class="toolbar">
  <div class="control-group">
    <span class="group-label">运行控制</span>
    <button id="btn-play" type="button">运行</button>
    <button id="btn-pause" type="button">暂停</button>
    <button id="btn-step" type="button">单步</button>
    <button id="btn-stop" type="button">停止</button>
    <span id="status" class="muted"></span>
  </div>
  <div class="control-group">
    <span class="group-label">阅读与导出</span>
    <label for="view-mode">视图</label>
    <select id="view-mode">
      <option value="all">全部过程</option>
      <option value="completed">成功动作转录（只看已完成）</option>
    </select>
    <button id="btn-export" type="button">导出 Markdown</button>
    <span id="view-note" class="muted"></span>
  </div>
</div>
<main>
  <section class="full" id="cycles-section">
    <h2>运行过程（每个完整周期）</h2>
    <div id="cycles"></div>
  </section>
  <section class="full" id="completed-section" style="display: none">
    <h2>World 已确认完成的行动</h2>
    <p id="completed-count" class="muted"></p>
    <div id="completed"></div>
  </section>
  <section class="full" id="calls-section">
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
  <details class="full fold-section" id="final-section" open>
    <summary>最终收尾状态（独立于视图筛选）</summary>
    <div id="final-state"></div>
  </details>
  <details class="full fold-section" id="audit-section" open>
    <summary>运行后审计（独立于视图筛选；不代表成功动作）</summary>
    <div id="audit"></div>
  </details>
</main>
<script>
"use strict";
let finished = false;
let lastVersion = null;
let refreshSeq = 0;
let appliedSeq = 0;
let callItems = [];
let refreshTimer = null;
let lastSnapshot = null;
let auditData = null;
let lastAppliedView = null;
let viewMode = "all";
const NL = String.fromCharCode(10);
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

function transcriptTime(value) {
  // 剧情钟 mm:ss[.毫秒最多3位去尾零]；整数毫秒精确保留，分钟不限制两位。
  if (typeof value !== "number" || !isFinite(value) || value < 0) return "—";
  const ms = Math.floor(value);
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.floor(ms / 1000) % 60;
  const fraction = ms % 1000;
  let text = String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
  if (fraction !== 0) {
    text += "." + String(fraction).padStart(3, "0").replace(/0+$/, "");
  }
  return text;
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

const PHASE_LABELS = {
  submitted: "已提交", registered: "已登记等待", started: "已开始", completed: "已完成",
  failed: "失败", unconfirmed: "未确认（结果未知）", withdrawn: "已撤回",
};
const RESULT_LABELS = {
  started: "已开始", in_progress: "进行中", completed: "已完成", failed: "失败",
  unconfirmed: "未确认", withdrawn: "已撤回",
};

function stateChip(code, labels) {
  const key = String(code === undefined || code === null ? "" : code);
  return element("span", "chip " + key, (labels[key] || key) + "（" + key + "）");
}

function phaseChip(phase) {
  return stateChip(phase, PHASE_LABELS);
}

function resultChip(result) {
  return stateChip(result, RESULT_LABELS);
}

function countChip(code, count) {
  const key = String(code || "");
  return element("span", "chip " + key, (RESULT_LABELS[key] || key) + " " + count);
}

function roleCard(roleLog, key, declaration) {
  const role = element("div", "role");
  role.appendChild(element("h3", null, "角色 " + (roleLog.role_id || key)));
  role.appendChild(element("p", "declaration",
    "主动声明：" + (declaration || "（无主动行为声明）")));
  const states = roleLog.action_states || [];
  if (states.length) {
    role.appendChild(element("div", "subhead", "自身行动状态"));
    const list = element("ul");
    for (const state of states) {
      const item = element("li");
      item.appendChild(phaseChip(state.phase));
      item.appendChild(element("span", null, " " + state.action_id + " — " + state.detail));
      list.appendChild(item);
    }
    role.appendChild(list);
  }
  const perceptions = roleLog.perceptions || [];
  if (perceptions.length) {
    role.appendChild(element("div", "subhead", "自身感知（" + perceptions.length + " 条）"));
    const list = element("ul");
    for (const text of perceptions.slice(0, 5)) listItem(list, "感知：" + text, "muted");
    role.appendChild(list);
    if (perceptions.length > 5) {
      const more = element("details", "more");
      more.appendChild(element("summary", null,
        "展开其余 " + (perceptions.length - 5) + " 条感知（完整保留）"));
      const rest = element("ul");
      for (const text of perceptions.slice(5)) listItem(rest, "感知：" + text, "muted");
      more.appendChild(rest);
      role.appendChild(more);
    }
  }
  if (!states.length && !perceptions.length) {
    role.appendChild(element("p", "muted", "本轮无自身行动状态与感知"));
  }
  return role;
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
  lastSnapshot = data;
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
  auditData = data.audit || null;
  renderAudit(auditData);
  renderCompleted(collectCompleted(data.cycles || []));
  applyView();
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
  const remembered = new Map();
  for (const node of host.querySelectorAll("details.cycle")) {
    remembered.set(node.dataset.step, {
      open: node.open,
      nested: Array.from(node.querySelectorAll("details"), function (child) {
        return child.open;
      }),
    });
  }
  host.replaceChildren();
  if (!cycles.length) {
    host.appendChild(element("p", "empty",
      "暂无已完成周期。初始为暂停；点击「单步」或「运行」后，每个完整的事件周期都会按时间顺序保留在这里。"));
    return;
  }
  cycles.forEach(function (data, index) {
    const previous = remembered.get(String(data.step));
    const node = cycleNode(data, previous ? previous.open : index === cycles.length - 1);
    node.dataset.step = String(data.step);
    // 已完成周期内容固定；审计或新周期更新时，保留用户的展开/收起选择。
    if (previous) {
      Array.from(node.querySelectorAll("details")).forEach(function (child, childIndex) {
        if (childIndex < previous.nested.length) child.open = previous.nested[childIndex];
      });
    }
    host.appendChild(node);
  });
}

function cycleNode(data, open) {
  const logs = data.entity_logs || {};
  const world = logs.world || {};
  const diagnostics = logs.diagnostics || {};
  const details = element("details", "cycle");
  details.open = !!open;

  const events = world.events || [];
  const entityActions = data.entity_actions || {};
  const roleKeys = Object.keys(entityActions);
  const registry = world.registry_actions || [];
  const outcomes = world.action_outcomes || [];
  const outcomeCounts = {};
  for (const outcome of outcomes) {
    const code = String((outcome && outcome.result) || "");
    outcomeCounts[code] = (outcomeCounts[code] || 0) + 1;
  }

  const summary = element("summary");
  summary.appendChild(element("span", "cycle-index", "周期 " + data.step));
  summary.appendChild(element("span", "cycle-meta",
    "剧情时刻 " + displayMs(world.simulation_time_ms) +
    (diagnostics.cycle_id ? " · " + diagnostics.cycle_id : "")));
  const chips = element("span", "chips");
  chips.appendChild(element("span", "chip", "事件 " + events.length));
  chips.appendChild(element("span", "chip", "角色 " + roleKeys.length));
  chips.appendChild(element("span", "chip", "登记 " + registry.length));
  for (const code of Object.keys(outcomeCounts)) {
    chips.appendChild(countChip(code, outcomeCounts[code]));
  }
  summary.appendChild(chips);
  details.appendChild(summary);

  const body = element("div", "cycle-body");
  body.appendChild(listBlock("同刻事件",
    events.map(function (event) {
      return "[" + event.kind + "] " + event.event_id + "：" + event.description;
    }),
    "本轮无同刻事件。"));

  if (roleKeys.length) {
    body.appendChild(element("h4", null, "角色主动声明与自身感知"));
    const roles = element("div", "roles");
    for (const key of roleKeys) {
      roles.appendChild(roleCard(logs[key] || {}, key, entityActions[key]));
    }
    body.appendChild(roles);
  } else {
    body.appendChild(element("p", "empty", "本周期没有角色声明记录。"));
  }

  if (registry.length) {
    const block = element("details", "tech registry");
    block.appendChild(element("summary", null,
      "统一登记后的行动记录（" + registry.length + " 条）"));
    const list = element("ul");
    for (const record of registry) {
      const item = element("li");
      item.appendChild(phaseChip(record.phase));
      item.appendChild(element("span", null,
        " " + record.action_id + "（" + record.role_id + "，" +
        record.start_ms + "→" + record.expected_end_ms + "ms）" + record.description));
      list.appendChild(item);
    }
    block.appendChild(list);
    body.appendChild(block);
  } else {
    body.appendChild(listBlock("统一登记后的行动记录", [], "暂无登记行动"));
  }

  const diagnosticTexts = [];
  const diagnostic = world.world_diagnostic;
  if (diagnostic) {
    for (const state of diagnostic.action_states || []) {
      diagnosticTexts.push("[实际 " + state.state + "] " + state.action_id +
        "（" + state.role_id + "）" + state.detail);
    }
    for (const passive of diagnostic.passive_processes || []) {
      diagnosticTexts.push("[被动] " + passive);
    }
  }
  if (diagnosticTexts.length) {
    const block = element("details", "tech worlddiag");
    block.appendChild(element("summary", null,
      "World 实际行动与被动过程（" + diagnosticTexts.length + " 条）"));
    const list = element("ul");
    for (const text of diagnosticTexts) listItem(list, text);
    block.appendChild(list);
    body.appendChild(block);
  } else {
    body.appendChild(listBlock("World 实际行动与被动过程（客观诊断）", [],
      "暂无实际行动与被动过程"));
  }

  const facts = element("div", "facts");
  facts.appendChild(element("h4", null, "事实增量与结算结果"));
  const factList = element("ul");
  for (const fact of world.committed_facts || []) listItem(factList, fact);
  for (const outcome of outcomes) {
    const item = element("li");
    item.appendChild(resultChip(outcome.result));
    item.appendChild(element("span", null, " " + outcome.action_id + "：" + outcome.detail));
    factList.appendChild(item);
  }
  if (!(world.committed_facts || []).length && !outcomes.length) {
    listItem(factList, "暂无事实增量与结算结果", "muted");
  }
  facts.appendChild(factList);
  body.appendChild(facts);

  const changes = diagnostics.change_reasons || {};
  const changeKeys = Object.keys(changes);
  if (changeKeys.length) {
    const block = element("details", "tech");
    block.appendChild(element("summary", null,
      "诊断：行动变化原因（" + changeKeys.length + " 条）"));
    const list = element("ul");
    for (const key of changeKeys) listItem(list, key + "：" + changes[key]);
    block.appendChild(list);
    body.appendChild(block);
  } else {
    body.appendChild(listBlock("诊断：行动变化原因", [], "本周期无行动变化诊断"));
  }

  details.appendChild(body);
  return details;
}

const S1_ROUTE_ORDER = ["continue", "replan", "rethink"];

function pctText(value) {
  // 数值百分比：整数去尾零，供条宽、标签与导出使用；纯数值来自服务端。
  if (typeof value !== "number" || !isFinite(value)) return "—";
  const text = (value * 100).toFixed(2).replace(/(\\.\\d+?)0+$/, "$1").replace(/\\.0+$/, "");
  return text + "%";
}

function system1Bars(weights, caption) {
  // 三路百分比条：数值只来自服务端已校验的结构字段，宽度按数值设置，
  // 全程 textContent / style，不注入 HTML。
  const wrap = element("div", "s1-bars");
  if (caption) wrap.appendChild(element("div", "muted", caption));
  for (const route of S1_ROUTE_ORDER) {
    const raw = weights ? weights[route] : undefined;
    const value = (typeof raw === "number" && isFinite(raw)) ? Math.max(0, Math.min(1, raw)) : 0;
    const row = element("div", "s1-row");
    row.appendChild(element("span", null, route));
    const track = element("div", "track");
    const fill = element("div", "s1-fill");
    fill.classList.add(route);
    fill.style.width = pctText(value);
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(element("span", "pct", pctText(value)));
    wrap.appendChild(row);
  }
  return wrap;
}

function system1PolicySummary(item) {
  // 程序 trace 才是实际路由与选择依据的来源；绝不从分布反推一个 route。
  const data = item.details || {};
  const parts = [
    "路由 " + (typeof data.route === "string" ? data.route : "—"),
    "方式 " + (typeof data.method === "string" ? data.method : "—"),
    "原因 " + (typeof data.reason === "string" ? data.reason : "—"),
  ];
  const legal = data.legal_weights;
  if (legal && typeof legal === "object") {
    parts.push("程序实际权重 " + S1_ROUTE_ORDER.map(function (route) {
      return route + " " + pctText(legal[route]);
    }).join(" / "));
  }
  const raw = data.raw_probabilities;
  if (raw && typeof raw === "object") {
    parts.push("模型原始倾向 " + S1_ROUTE_ORDER.map(function (route) {
      return route + " " + pctText(raw[route]);
    }).join(" / "));
  }
  if (typeof data.certainty_threshold === "number") {
    parts.push("确定阈值 " + pctText(data.certainty_threshold));
  }
  if (typeof data.margin_threshold === "number") parts.push("边际阈值 " + pctText(data.margin_threshold));
  if (typeof data.random_value === "number") {
    parts.push("随机抽值 " + data.random_value + "（种子 " + data.route_seed + "）");
  }
  if (typeof data.note === "string" && data.note) parts.push("说明 " + data.note);
  return parts.join("；");
}

function system1PolicyBlock(item) {
  const data = item.details || {};
  const block = element("div");
  const legal = data.legal_weights;
  if (legal && typeof legal === "object") {
    block.appendChild(system1Bars(legal, "程序实际使用的合法归一化权重"));
  }
  const route = typeof data.route === "string" ? data.route : "—";
  const method = typeof data.method === "string" ? data.method : "—";
  const reason = typeof data.reason === "string" ? data.reason : "—";
  block.appendChild(element("p", "s1-route",
    "最终路由：" + route + "（选择方式 " + method + "；原因 " + reason + "）"));
  const bits = [];
  if (Array.isArray(data.allowed_routes)) bits.push("允许路径 " + data.allowed_routes.join("/"));
  if (typeof data.max_weight === "number") bits.push("最大权重 " + pctText(data.max_weight));
  if (typeof data.margin === "number") bits.push("领先幅度 " + pctText(data.margin));
  if (typeof data.random_value === "number") bits.push("随机抽值 " + data.random_value);
  if (typeof data.route_seed === "number") bits.push("route_seed " + data.route_seed);
  if (bits.length) block.appendChild(element("p", "muted", bits.join(" · ")));
  return block;
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
  } else if (item.is_policy) {
    head.textContent = "程序路由决策 · " + item.phase + " · " + item.group;
  } else {
    head.textContent = (item.is_skip ? "未调用模型（跳过）" : "阶段错误") +
      " · " + item.phase + " · " + item.group;
  }
  node.appendChild(head);
  const parts = [];
  if (item.scope) parts.push("作用域 " + item.scope);
  if (item.role_id) parts.push("角色 " + item.role_id);
  if (item.response_id) parts.push("响应 " + item.response_id);
  if (item.cycle_id) parts.push("周期 " + item.cycle_id);
  if (typeof item.simulation_time_ms === "number") {
    parts.push("剧情时刻 " + item.simulation_time_ms + " ms");
  }
  node.appendChild(element("div", "call-meta", parts.join(" · ")));
  if (item.summary_text) {
    node.appendChild(element("div", "call-summary", item.summary_text));
  }
  if (item.kind === "call" && item.phase === "system1" && item.distribution) {
    // 模型调用的三条百分比条只展示模型自报倾向，不据此推断 route。
    node.appendChild(system1Bars(item.distribution, "模型倾向（原始，未校准）"));
  }
  if (item.kind === "note" && item.is_policy) {
    node.appendChild(system1PolicyBlock(item));
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
        const value = item.details[key];
        return key + ": " + (typeof value === "string" ? value : JSON.stringify(value));
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

function auditStatusText(audit) {
  if (!audit || audit.enabled !== true) {
    return "未开启审计（启动时未提供 --audit-config）。";
  }
  const status = audit.status || "waiting";
  if (status === "waiting") return "等待模拟结束…";
  if (status === "checking") return "模拟已结束，正在构建证据报告并执行机械检查…";
  if (status === "running") return "审计模型运行中…";
  if (status === "done") {
    return "审计完成（模型 " + (audit.models || []).length + " 个，失败 " +
      (audit.failed_models || 0) + " 个）。";
  }
  if (status === "cancelled") {
    return "审计被取消（部分模型可能已执行）；模拟概要/世界事实未受影响，不代表审计通过。";
  }
  if (status === "failed") return "审计内部失败：" + (audit.error || audit.scope || "（无说明）");
  return status;
}

function auditCompactStatus(audit) {
  if (!audit || audit.enabled !== true) return "未开启";
  const status = audit.status || "waiting";
  if (status === "waiting") return "等待模拟结束";
  if (status === "checking") return "机械检查中";
  if (status === "running") return "模型运行中";
  if (status === "done") return "完成（失败模型 " + (audit.failed_models || 0) + " 个）";
  if (status === "cancelled") return "已取消（部分模型可能已执行）";
  if (status === "failed") return "内部失败";
  return status;
}

function renderAudit(audit) {
  const host = document.getElementById("audit");
  const summary = document.querySelector("#audit-section > summary");
  if (summary) {
    summary.textContent = "运行后审计：" + auditCompactStatus(audit) +
      "（独立于视图筛选，不代表成功动作）";
  }
  host.replaceChildren();
  host.appendChild(element("p", "muted", auditStatusText(audit)));
  if (!audit || audit.enabled !== true) return;
  if (audit.scope) host.appendChild(element("p", "muted", audit.scope));
  if (audit.error) host.appendChild(element("p", "muted", "错误：" + audit.error));
  if (typeof audit.evidence_count === "number" && audit.evidence_count > 0) {
    host.appendChild(element("p", "muted", "证据条数：" + audit.evidence_count));
  }
  const checks = audit.checks || [];
  if (checks.length) {
    host.appendChild(element("h4", null, "机械检查（确定性）"));
    const list = element("ul");
    for (const check of checks) {
      listItem(list, "[" + check.status + "] " + check.check_id + "：" + check.detail);
    }
    host.appendChild(list);
  }
  for (const model of audit.models || []) {
    const block = element("div", "audit-model" + (model.status === "failed" ? " error" : ""));
    block.appendChild(element("div", "call-head",
      "模型 " + model.id + "（" + model.model + "）— " + model.status));
    if (model.error) block.appendChild(element("div", "call-summary", "错误：" + model.error));
    if (model.summary) block.appendChild(element("div", "call-summary", "总结：" + model.summary));
    for (const finding of model.findings || []) {
      const refs = (finding.evidence_refs || []).join("、");
      const list = element("ul");
      listItem(list, "[" + finding.severity + "/" + finding.certainty + "] " +
        finding.title + "：" + finding.explanation +
        (refs ? "（证据：" + refs + "）" : "") + " 建议：" + finding.suggestion);
      block.appendChild(list);
    }
    host.appendChild(block);
  }
}

function auditMarkdownLines(audit) {
  const lines = ["", "## 运行后审计", ""];
  if (!audit || audit.enabled !== true) {
    lines.push("- 未开启审计。");
    return lines;
  }
  lines.push(markdownLines(auditStatusText(audit), "- "));
  if (audit.scope) lines.push(markdownLines(audit.scope, "- "));
  if (audit.error) lines.push(markdownLines("错误：" + audit.error, "- "));
  for (const check of audit.checks || []) {
    lines.push(markdownLines(
      "[" + check.status + "] " + check.check_id + "：" + check.detail, "- "));
  }
  for (const model of audit.models || []) {
    lines.push("", "### " + markdownText(
      "模型 " + model.id + "（" + model.model + "）— " + model.status));
    if (model.error) lines.push(markdownLines("错误：" + model.error, "- "));
    if (model.summary) lines.push(markdownLines("总结：" + model.summary, "- "));
    for (const finding of model.findings || []) {
      const refs = (finding.evidence_refs || []).join("、");
      lines.push(markdownLines("[" + finding.severity + "/" + finding.certainty + "] " +
        finding.title + "：" + finding.explanation +
        (refs ? "（证据：" + refs + "）" : "") + " 建议：" + finding.suggestion, "- "));
    }
  }
  return lines;
}

function collectCompleted(cycles) {
  const rows = [];
  const steps = new Set();
  for (const cycle of cycles) {
    const world = ((cycle.entity_logs || {}).world) || {};
    const diagnostics = ((cycle.entity_logs || {}).diagnostics) || {};
    const outcomes = Array.isArray(world.action_outcomes) ? world.action_outcomes : [];
    for (const outcome of outcomes) {
      if (!outcome || outcome.result !== "completed") continue;
      rows.push({
        step: cycle.step,
        cycleId: diagnostics.cycle_id || null,
        timeMs: world.simulation_time_ms,
        actionId: outcome.action_id,
        detail: outcome.detail,
      });
      steps.add(String(cycle.step));
    }
  }
  return { rows: rows, cycleCount: steps.size };
}

function renderCompleted(result) {
  const host = document.getElementById("completed");
  const count = document.getElementById("completed-count");
  const openRows = new Set(Array.from(host.querySelectorAll(".settlement-meta[open]"),
    function (node) { return node.dataset.rowKey; }));
  host.replaceChildren();
  count.textContent = result.rows.length
    ? "已确认完成结果 " + result.rows.length + " 条，涉及 " + result.cycleCount + " 个周期。"
    : "已确认完成结果 0 条。";
  if (!result.rows.length) {
    host.appendChild(element("p", "empty",
      "暂无 World 已确认完成的行动。这里只统计已提交周期中 result=completed 的结算；" +
      "候选模型输出、检查通过与普通调用都不会计入。"));
    return;
  }
  const transcript = element("div", "transcript");
  for (const row of result.rows) {
    const line = element("div", "transcript-line");
    const detail = row.detail === undefined || row.detail === null ||
      String(row.detail).trim() === "" ? "（World 已确认完成，无结算说明）" : String(row.detail);
    line.appendChild(element("p", "transcript-text",
      "[" + transcriptTime(row.timeMs) + "] " + detail));
    const meta = element("details", "settlement-meta");
    meta.dataset.rowKey = JSON.stringify([row.step, row.actionId, row.timeMs]);
    meta.open = openRows.has(meta.dataset.rowKey);
    meta.appendChild(element("summary", null, "查看结算详情"));
    const list = element("ul");
    listItem(list, "结果：completed");
    listItem(list, "行动身份：" + (row.actionId || "—"));
    listItem(list, "剧情时刻：" + displayMs(row.timeMs));
    listItem(list, "周期：" + row.step + (row.cycleId ? "（" + row.cycleId + "）" : ""));
    meta.appendChild(list);
    line.appendChild(meta);
    transcript.appendChild(line);
  }
  host.appendChild(transcript);
}

function applyView() {
  const completed = viewMode === "completed";
  document.getElementById("cycles-section").style.display = completed ? "none" : "";
  document.getElementById("calls-section").style.display = completed ? "none" : "";
  document.getElementById("completed-section").style.display = completed ? "" : "none";
  document.getElementById("view-note").textContent = completed
    ? "成功动作转录：只显示 World 已提交的 result=completed 结算；最终状态与审计折叠保留，与筛选无关。"
    : "完整调试视图；最终收尾状态与审计始终独立显示。";
  if (lastAppliedView !== viewMode) {
    // 只在切换模式时调整折叠默认；轮询不重新收起/展开，也不影响审计更新。
    document.getElementById("final-section").open = !completed;
    document.getElementById("audit-section").open = !completed;
    lastAppliedView = viewMode;
  }
}

function markdownText(value) {
  const text = String(value === null || value === undefined ? "" : value);
  // 动态内容作为文字导出，不能生成额外标题、链接或 HTML。
  const normalized = text.split(String.fromCharCode(13)).join("");
  const special = new Set([92, 96, 42, 95, 123, 125, 91, 93, 40, 41, 35, 33, 124]);
  return normalized.split(NL).map(function (line) {
    return Array.from(line).map(function (char, index) {
      if (char === "&") return "&amp;";
      if (char === "<") return "&lt;";
      if (char === ">") return "&gt;";
      const startMarker = (char === "-" || char === "+") && !line.slice(0, index).trim();
      const numberMarker = char === "." && /^[ ]*[0-9]+$/.test(line.slice(0, index));
      return special.has(char.charCodeAt(0)) || startMarker || numberMarker
        ? String.fromCharCode(92) + char : char;
    }).join("");
  }).join(NL);
}

function markdownLines(value, prefix) {
  const parts = markdownText(value).split(NL);
  return [prefix + parts[0]].concat(parts.slice(1).map(function (line) {
    return "  " + line;
  })).join(NL);
}

function pushMarkdownBlock(lines, texts, emptyText) {
  if (!texts.length) {
    lines.push(markdownLines(emptyText, "- "));
    return;
  }
  for (const text of texts) lines.push(markdownLines(text, "- "));
}

function finalStateMarkdown(finalData) {
  const state = lastSnapshot || {};
  if (!finalData) {
    return [state.finished === true
      ? "运行已结束，但没有收尾步数据。"
      : "运行尚未结束；本导出只包含当前已确认内容。"];
  }
  const world = ((finalData.entity_logs || {}).world) || {};
  const diagnostics = ((finalData.entity_logs || {}).diagnostics) || {};
  const out = [
    "停止原因：" + (diagnostics.stop_reason || "—"),
    "完成周期：" +
      (diagnostics.completed_cycles === undefined ? "—" : diagnostics.completed_cycles),
    "最终剧情时刻：" + displayMs(world.simulation_time_ms),
  ];
  if (finalData.action) out.push("收尾说明：" + finalData.action);
  if (diagnostics.error) out.push("错误：" + diagnostics.error);
  return out;
}

function buildCompletedMarkdown() {
  const result = collectCompleted((lastSnapshot || {}).cycles || []);
  const lines = [
    "# sceneweave 已确认完成行动",
    "",
    "> 范围：仅包含已提交周期中 World 结算 result=completed 的行动；失败、未确认或混合周期" +
      "不会整体标为成功，候选模型输出、检查通过与普通模型调用不计入。",
    "> completed 表示该行动完成，不等于全局目标必然达成。",
    "",
    "共 " + result.rows.length + " 条完成结果，涉及 " + result.cycleCount + " 个周期。",
    "",
    "## 成功动作转录",
    "",
  ];
  if (!result.rows.length) {
    lines.push("暂无 World 已确认完成的行动。");
  }
  for (const row of result.rows) {
    const detail = row.detail === undefined || row.detail === null ||
      String(row.detail).trim() === "" ? "（World 已确认完成，无结算说明）" : row.detail;
    const body = markdownLines(detail, "");
    lines.push("- [" + transcriptTime(row.timeMs) + "] " + body);
  }
  lines.push("", "## 最终状态", "");
  for (const line of finalStateMarkdown((lastSnapshot || {}).final)) {
    lines.push(markdownLines(line, "- "));
  }
  for (const line of auditMarkdownLines((lastSnapshot || {}).audit)) lines.push(line);
  return lines.join(NL) + NL;
}

function buildProcessMarkdown() {
  const snapshot = lastSnapshot || {};
  const cycles = snapshot.cycles || [];
  const lines = [
    "# sceneweave 运行过程导出",
    "",
    "> 范围：全部已完成周期的事件、角色完整主动声明、登记行动与 World 已提交事实/结算；" +
      "模型调用摘要按当前角色/阶段筛选，省略原始 Prompt/Response 原文。",
    "",
  ];
  if (!cycles.length) {
    lines.push("尚未完成任何周期。", "");
  }
  for (const cycle of cycles) {
    const logs = cycle.entity_logs || {};
    const world = logs.world || {};
    const diagnostics = logs.diagnostics || {};
    lines.push("## 周期 " + cycle.step +
      (diagnostics.cycle_id ? "（" + markdownText(diagnostics.cycle_id) + "）" : "") +
      " · " + displayMs(world.simulation_time_ms), "");
    lines.push("### 同刻事件");
    pushMarkdownBlock(lines, (world.events || []).map(function (event) {
      return "[" + event.kind + "] " + event.event_id + "：" + event.description;
    }), "本轮无同刻事件。");
    lines.push("", "### 角色主动声明与可见状态");
    const entityActions = cycle.entity_actions || {};
    const roleKeys = Object.keys(entityActions);
    if (!roleKeys.length) lines.push("- 本周期没有角色声明记录。");
    for (const key of roleKeys) {
      const roleLog = logs[key] || {};
      lines.push("", "#### 角色 " + markdownText(roleLog.role_id || key));
      pushMarkdownBlock(lines, ["主动声明：" + entityActions[key]], "（无主动行为声明）");
      pushMarkdownBlock(lines, (roleLog.action_states || []).map(function (state) {
        return "[" + state.phase + "] " + state.action_id + " — " + state.detail;
      }), "本轮无自身行动状态。");
      pushMarkdownBlock(lines, (roleLog.perceptions || []).map(function (text) {
        return "感知：" + text;
      }), "本轮无可用感知。");
    }
    lines.push("", "### 统一登记后的行动记录");
    pushMarkdownBlock(lines, (world.registry_actions || []).map(function (record) {
      return "[" + record.phase + "] " + record.action_id + "（" + record.role_id + "，" +
        record.start_ms + "→" + record.expected_end_ms + "ms）" + record.description;
    }), "暂无登记行动。");
    lines.push("", "### World 实际行动与被动过程");
    const diagnosticTexts = [];
    const diagnostic = world.world_diagnostic;
    if (diagnostic) {
      for (const state of diagnostic.action_states || []) {
        diagnosticTexts.push("[实际 " + state.state + "] " + state.action_id +
          "（" + state.role_id + "）" + state.detail);
      }
      for (const passive of diagnostic.passive_processes || []) {
        diagnosticTexts.push("[被动] " + passive);
      }
    }
    pushMarkdownBlock(lines, diagnosticTexts, "暂无实际行动与被动过程。");
    lines.push("", "### 事实增量与结算结果");
    const outcomeTexts = (world.action_outcomes || []).map(function (outcome) {
      return "结算：" + outcome.action_id + " → " + outcome.result +
        "（" + outcome.detail + "）";
    });
    pushMarkdownBlock(lines, (world.committed_facts || []).concat(outcomeTexts),
      "暂无事实增量与结算结果。");
    lines.push("", "### 诊断：行动变化原因");
    const changes = diagnostics.change_reasons || {};
    pushMarkdownBlock(lines, Object.keys(changes).map(function (key) {
      return key + "：" + changes[key];
    }), "本周期无行动变化诊断。");
    lines.push("");
  }
  const role = document.getElementById("filter-role").value;
  const phase = document.getElementById("filter-phase").value;
  const filtered = callItems.filter(function (item) {
    if (role && item.group !== role) return false;
    if (phase && item.phase !== phase) return false;
    return true;
  });
  lines.push("## 模型调用摘要（当前筛选：角色/作用域 " + markdownText(role || "全部") +
    "；阶段 " + markdownText(phase || "全部") + "）", "");
  if (!filtered.length) {
    lines.push(callItems.length ? "当前筛选下没有模型调用记录。" : "暂无模型调用记录。");
  }
  for (const item of filtered) {
    let label;
    let body;
    if (item.kind === "call") {
      label = "模型调用";
      body = item.summary_text || "";
    } else if (item.is_policy) {
      // 程序 trace 的路由与依据进入导出，让审计看到实际选择而非只模型倾向。
      label = "程序路由决策";
      body = system1PolicySummary(item);
    } else {
      label = item.is_skip ? "未调用（跳过）" : "阶段错误";
      body = item.summary_text || "";
    }
    lines.push(markdownLines(label + " · " + item.phase + " · " + (item.group || "—") +
      " · " + (typeof item.simulation_time_ms === "number"
        ? item.simulation_time_ms + " ms" : "—") + "：" + body, "- "));
  }
  lines.push("", "## 最终状态", "");
  for (const line of finalStateMarkdown(snapshot.final)) {
    lines.push(markdownLines(line, "- "));
  }
  for (const line of auditMarkdownLines(snapshot.audit)) lines.push(line);
  return lines.join(NL) + NL;
}

function buildExportMarkdown() {
  return viewMode === "completed" ? buildCompletedMarkdown() : buildProcessMarkdown();
}

function downloadMarkdown(text, filename) {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(function () { URL.revokeObjectURL(url); }, 0);
}

function exportMarkdown() {
  downloadMarkdown(buildExportMarkdown(), "sceneweave-" +
    (viewMode === "completed" ? "completed" : "process") + ".md");
}

document.getElementById("btn-play").addEventListener("click", function () { postCommand("/play"); });
document.getElementById("btn-pause").addEventListener("click", function () { postCommand("/pause"); });
document.getElementById("btn-step").addEventListener("click", function () { postCommand("/step"); });
document.getElementById("btn-stop").addEventListener("click", function () { postCommand("/stop"); });
document.getElementById("btn-export").addEventListener("click", exportMarkdown);
document.getElementById("view-mode").addEventListener("change", function (event) {
  viewMode = event.target.value;
  applyView();
});
document.getElementById("filter-role").addEventListener("change", applyCallFilters);
document.getElementById("filter-phase").addEventListener("change", applyCallFilters);

const source = new EventSource("/events");
function auditPending() {
  return !!(auditData && auditData.enabled === true &&
    ["waiting", "checking", "running"].indexOf(auditData.status) >= 0);
}

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
window.setInterval(function () {
  if (!finished || auditPending()) refreshSnapshot();
}, 1500);
window.setInterval(function () { if (!finished) refreshStatus(); }, 1500);
</script>
</body>
</html>
"""
