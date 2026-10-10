"""UI 只读快照：真实 StepData 历史与 SimulationLog 调用记录的可读适配。

数据来源只有两处：``SimulationServer`` 实际广播过的 ``StepData``（经
``SnapshotSimulationServer`` 捕获），以及运行层创建的同一
``SimulationLog``。本模块不建立第二套业务日志，不读取环境变量、密钥、
全局配置或磁盘路径；快照只存在内存中，服务器关闭即消失，运行结束后的
离线查看仍以输出目录的 ``simulation_log.json`` 为准。

``SnapshotSimulationServer`` 只是轻量子类：广播行为沿用父类，仅额外
保留历史，并在 ``_create_handler`` 提供的固定路径上增加一个只读 JSON
端点。原有控制、状态与 SSE 端点完全继承，不修改 site-packages 或框架
语义。所有原始文本都作为 JSON 字符串返回，由页面用 textContent 渲染。
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from typing import Any

from concordia.environment.step_controller import StepData
from concordia.utils import structured_logging
from concordia.utils.simulation_server import SimulationServer

from sceneweave.audit.session import AuditState
from sceneweave.roles.model_output import ModelOutputError, parse_system1_tendency

SNAPSHOT_PATH = "/run-snapshot"

# 跳过模型调用的阶段标记：这些阶段如实说明“未调用”，不伪造成调用记录。
_SKIP_SUFFIXES = (".skip", ".skipped")


def _is_skip_stage(stage: str) -> bool:
    return stage.endswith(_SKIP_SUFFIXES)


# System1 成功的程序策略留痕也需要在界面可读；其余成功阶段不收录。
_POLICY_STAGE = "system1.decide"

_ROUTE_LABELS = {
    "continue": "沿用已保存计划",
    "replan": "保留意向重新落实",
    "rethink": "重选意向",
}
_SYSTEM1_ROUTES = ("continue", "replan", "rethink")


def _safe_json_value(value: Any) -> Any:
    """递归保留安全 JSON 值（含 float 与嵌套分布），丢弃不支持的标量。

    模型文本不在此列；这里只用于程序写入的阶段细节，保证分布、阈值与
    随机抽值等数值字段在快照中不被丢弃。
    """
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key:
                kept[key] = _safe_json_value(item)
        return kept
    return None


def _system1_distribution(raw: Any) -> dict[str, float] | None:
    """复用正式解析器，坏输出只展示原文，不画成合法分布。"""
    if not isinstance(raw, str):
        return None
    try:
        tendency = parse_system1_tendency(raw)
    except ModelOutputError:
        return None
    return {route.value: value for route, value in tendency.probabilities.items()}


def _percent(value: float) -> str:
    text = f"{value * 100:.2f}".rstrip("0").rstrip(".")
    return text + "%"


def _text(value: Any) -> str | None:
    """只保留真实字符串；非字符串一律视为缺失，不猜测、不转换。"""
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _group_of(role_id: str | None, scope: str | None, stage: str) -> str:
    """筛选用角色/作用域标签：优先真实角色身份，其次作用域，最后阶段前缀。"""
    if role_id:
        return role_id
    if scope:
        return scope
    if stage.startswith("world"):
        return "World"
    if stage.startswith("translator"):
        return "翻译器"
    return "调度器"


class RunSnapshotHistory:
    """已广播步数据的线程安全历史；完成时把最后一帧划为收尾状态。

    每个周期由运行编排在周期边界广播一次；历史按广播顺序追加，不做
    去重（广播本身每个周期恰好一次）。完成信号到达时把最后一帧从
    周期历史中取出作为最终状态，使“整轮周期”与“收尾帧”保持区分。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: list[dict[str, Any]] = []
        self._last: dict[str, Any] | None = None
        self._final: dict[str, Any] | None = None
        self._finished = False
        self._revision = 0

    def record_step(self, data: dict[str, Any]) -> None:
        """记录一次真实广播的步数据；调用方已复制为独立字典。"""
        with self._lock:
            stored = dict(data)
            self._cycles.append(stored)
            self._last = stored
            self._revision += 1

    def mark_completed(self) -> None:
        """完成信号：最后一帧转为最终状态，不再出现在周期历史中。"""
        with self._lock:
            if self._last is not None:
                self._final = self._last
                if self._cycles and self._cycles[-1] is self._last:
                    self._cycles.pop()
            self._finished = True
            self._revision += 1

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def view(self) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, bool]:
        """只读副本：(修订号, 周期历史, 最终状态, 是否完成)。

        修订号与数据在同一次加锁内读取，避免版本元数据与内容错位。
        """
        with self._lock:
            return (self._revision, list(self._cycles), self._final, self._finished)


def _short(text: str, limit: int = 72) -> str:
    """折叠空白并限长；摘要只截断展示，不修改原始回答。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _call_summary(component: str, item: dict[str, Any]) -> str:
    """从真实回答 JSON 提炼简短摘要；缺字段或非 JSON 如实说明。"""
    if item["is_error"]:
        error_type = item["error_type"] or "错误"
        message = item["error_message"] or ""
        return f"调用失败：{error_type}: {_short(message)}"
    raw = item["response"]
    if raw is None:
        return "没有回答文本；展开可查看提示。"
    try:
        data = json.loads(raw)
    except TypeError, ValueError:
        return "回答不是合法 JSON；展开可查看原始回答。"
    if not isinstance(data, dict):
        if component == "system3" and isinstance(data, list):
            return f"结构化动作计划：{len(data)} 个动作"
        return "回答不是 JSON 对象；展开可查看原始回答。"
    if component == "system1":
        distribution = _system1_distribution(raw)
        note = data.get("note")
        if distribution is not None and isinstance(note, str):
            trend = " / ".join(
                f"{route} {_percent(distribution[route])}" for route in _SYSTEM1_ROUTES
            )
            return f"倾向：{trend}；说明：{_short(note)}"
        if isinstance(data.get("route"), str) and isinstance(data.get("note"), str):
            # 旧日志兼容：历史记录可能仍是 route 字段，如实展示，不猜分布。
            route = _ROUTE_LABELS.get(data["route"], data["route"])
            return f"分流：{route}；说明：{_short(data['note'])}"
    elif component == "system2":
        if isinstance(data.get("intention"), str):
            return f"意向：{_short(data['intention'])}"
    elif component == "system3":
        if data.get("error") == "intention_unfulfillable" and isinstance(data.get("reason"), str):
            return f"意向不可落实：{_short(data['reason'])}"
    elif component == "world-check":
        if isinstance(data.get("passed"), bool) and isinstance(data.get("reason"), str):
            verdict = "通过" if data["passed"] else "未通过"
            reason = _short(data["reason"]) if data["reason"] else "无"
            return f"检查{verdict}；原因：{reason}"
    elif component == "world-progress":
        progress_keys = ("movements", "updated_action_ids", "notes", "passive_updates")
        if all(isinstance(data.get(key), list) for key in progress_keys):
            notes = "；".join(_short(str(note)) for note in data["notes"][:2])
            summary = (
                f"运动 {len(data['movements'])} 项；更新 {len(data['updated_action_ids'])} 项；"
                f"被动 {len(data['passive_updates'])} 项"
            )
            return f"{summary}；说明：{notes}" if notes else summary
    elif component == "world-adjudicate":
        known = (
            ("changes", "事实变更"),
            ("outcomes", "结算"),
            ("follow_ups", "后续"),
            ("movements", "运动"),
            ("stop_movements", "停止运动"),
            ("passives", "被动"),
            ("stop_passives", "停止被动"),
        )
        present = [(label, data[key]) for key, label in known if isinstance(data.get(key), list)]
        if present:
            return "；".join(f"{label} {len(items)} 项" for label, items in present)
    elif component == "translator":
        if isinstance(data.get("schedules"), list):
            parts = [f"新安排 {len(data['schedules'])} 项"]
            if isinstance(data.get("modified_schedules"), list):
                parts.append(f"修改安排 {len(data['modified_schedules'])} 项")
            return "；".join(parts)
    return "回答为 JSON 对象；展开可查看原始回答。"


def _call_item(
    entry: structured_logging.StructuredLogEntry, data: Any, index: int
) -> dict[str, Any] | None:
    """把一条组件日志转成模型调用条目；结构缺失如实返回 None。"""
    if not isinstance(data, dict):
        return None
    component = _text(data.get("Component")) or entry.component_name
    key = _text(data.get("Key"))
    error_type = _text(data.get("ErrorType"))
    error_message = _text(data.get("ErrorMessage"))
    role_id = _text(data.get("role_id"))
    scope = _text(data.get("scope")) or entry.entity_name
    item: dict[str, Any] = {
        "kind": "call",
        "entry_index": index,
        "step": entry.step,
        "cycle_id": _text(data.get("cycle_id")),
        "response_id": _text(data.get("response_id")),
        "simulation_time_ms": _optional_int(data.get("simulation_time_ms")),
        "scope": scope,
        "role_id": role_id,
        "group": _group_of(role_id, scope, component),
        "phase": component,
        "timestamp": entry.timestamp,
        "summary": entry.summary,
        "is_error": key == "sample_text_error",
        "replay": data.get("Replay") is True,
        "prompt": _text(data.get("Prompt")),
        "response": _text(data.get("Response")),
        "error_type": error_type,
        "error_message": error_message,
    }
    # 三路倾向分布保留为结构化字段（服务端已校验有限数值），供页面画百分比条。
    item["distribution"] = (
        _system1_distribution(item["response"]) if component == "system1" else None
    )
    item["summary_text"] = _call_summary(component, item)
    return item


def _stage_item(
    entry: structured_logging.StructuredLogEntry, data: Any, index: int
) -> dict[str, Any] | None:
    """把跳过/错误阶段留痕转成未调用或错误条目；其它阶段不收录。"""
    if not isinstance(data, dict):
        return None
    stage = _text(data.get("stage")) or entry.component_name
    error = _text(data.get("error"))
    skipped = _is_skip_stage(stage)
    is_policy = stage == _POLICY_STAGE and error is None and not skipped
    if not skipped and error is None and not is_policy:
        return None
    role_id = _text(data.get("role_id"))
    raw_inputs = data.get("inputs")
    raw_outputs = data.get("outputs")
    inputs = (
        [item for item in raw_inputs if isinstance(item, str)]
        if isinstance(raw_inputs, list)
        else []
    )
    outputs = (
        [item for item in raw_outputs if isinstance(item, str)]
        if isinstance(raw_outputs, list)
        else []
    )
    raw_details = data.get("details")
    details = _safe_json_value(raw_details) if isinstance(raw_details, dict) else {}
    summary_text = entry.summary
    if error is not None:
        summary_text = f"{entry.summary}：{_short(error)}"
    return {
        "kind": "note",
        "entry_index": index,
        "step": entry.step,
        "cycle_id": _text(data.get("cycle_id")),
        "response_id": _text(data.get("response_id")),
        "simulation_time_ms": _optional_int(data.get("simulation_time_ms")),
        "scope": None,
        "role_id": role_id,
        "group": _group_of(role_id, None, stage),
        "phase": stage,
        "timestamp": entry.timestamp,
        "summary": entry.summary,
        "summary_text": summary_text,
        "is_error": error is not None,
        "is_skip": skipped,
        "is_policy": is_policy,
        "inputs": inputs,
        "outputs": outputs,
        "details": details,
        "prompt": None,
        "response": None,
        "error_type": None,
        "error_message": None,
        "error": error,
    }


def build_snapshot(
    history: RunSnapshotHistory,
    log: structured_logging.SimulationLog,
    audit_state: AuditState | None = None,
) -> dict[str, Any]:
    """构造只读快照；所有去重引用都在服务端解开，浏览器只见原文。

    先在一次加锁内读取历史修订号与内容，再带锁读取日志快照；版本
    由实际捕获的修订号、日志条目数与完成标记组成，只有内容真的不同
    时才递增。完成信号在日志写入之后，因此先读历史保证完成快照包含
    收尾之前的全部日志，避免页面停下轮询后遗漏最后一次调用。
    重建的日志对象只用于解引用，不回写原日志。
    """
    revision, cycles, final, finished = history.view()
    raw_log = log.to_dict()
    raw_entries = raw_log.get("entries")
    entry_count = len(raw_entries) if isinstance(raw_entries, list) else 0
    snapshot_log = structured_logging.SimulationLog.from_dict(raw_log)
    calls: list[dict[str, Any]] = []
    for index, entry in enumerate(snapshot_log.entries):
        data = snapshot_log.reconstruct_value(entry.deduplicated_data)
        if entry.entry_type == "component":
            item = _call_item(entry, data, index)
        elif entry.entry_type == "sceneweave":
            item = _stage_item(entry, data, index)
        else:
            item = None
        if item is not None:
            calls.append(item)
    audit = (
        audit_state.snapshot()
        if audit_state is not None
        else {
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
    )
    return {
        "version": [revision, entry_count, 1 if finished else 0, audit["revision"]],
        "finished": finished,
        "cycles": cycles,
        "final": final,
        "calls": calls,
        "audit": audit,
    }


class SnapshotSimulationServer(SimulationServer):
    """``SimulationServer`` 的只读快照扩展；控制/状态/SSE 全部继承。

    ``log_provider`` 由运行层提供，返回同一 ``SimulationLog``；端点不
    接受任何参数、路径或查询条件，也不读取环境与磁盘。
    """

    def __init__(
        self,
        *,
        port: int = 8080,
        html_content: str = "",
        history: RunSnapshotHistory,
        log_provider: Callable[[], structured_logging.SimulationLog],
        audit_state: AuditState | None = None,
    ) -> None:
        super().__init__(port=port, html_content=html_content)
        self._history = history
        self._log_provider = log_provider
        self._audit_state = audit_state

    def broadcast_step(self, step_data: StepData) -> None:
        super().broadcast_step(step_data)
        self._history.record_step(dict(self._current_step_data))

    def broadcast_completion(self) -> None:
        super().broadcast_completion()
        self._history.mark_completed()

    def _create_handler(self) -> Any:
        base: Any = super()._create_handler()  # type: ignore[no-untyped-call]
        history = self._history
        log_provider = self._log_provider
        audit_state = self._audit_state

        class Handler(base):  # type: ignore[misc]
            def do_GET(self) -> None:
                if self.path == SNAPSHOT_PATH:
                    payload = build_snapshot(history, log_provider(), audit_state)
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

        return Handler
