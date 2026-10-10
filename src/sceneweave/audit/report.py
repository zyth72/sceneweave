"""固定证据报告与确定性机械检查：审计模型只读取本报告。

报告从已导出的引擎追踪记录、场景事实与真实运行概要以确定顺序构造，
每条证据有稳定 evidence_id（E-0001…），供模型 findings 引用并在校验
时核对。报告覆盖全部已完成周期（含失败/未确认/撤回），不做成功筛选；
原始组件调用提示不整段复制，只保留可审计结构并注明原文位置。

机械检查只基于明确结构报确定错误；缺少足够追踪时标为“未覆盖”而不是
通过，且对通信正文、语义终态与同刻批次语义明确说明未覆盖范围，避免
暗示整体已通过。规则语义的权威归 docs/，报告只做投影并标注来源。
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sceneweave.domain.tracing import TraceRecord, trace_details_to_json
from sceneweave.world.scene import SceneObject

_PROMPT_INVOKE_SUFFIX = ".invoke"
_MD_SPECIAL = frozenset("\\`*_{}[]()#!|")


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """审计可见的运行概要；字段与模拟导出 summary 一致，不新增事实。"""

    stop_reason: str
    final_time_ms: int
    completed_cycles: int
    world_state_version: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    text: str
    cycle_id: str | None = None
    simulation_time_ms: int | None = None
    role_id: str | None = None
    action_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MechanicalCheck:
    """一条确定性结构检查：passed / finding / not_covered 三态。"""

    check_id: str
    status: str
    detail: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    items: tuple[EvidenceItem, ...]
    checks: tuple[MechanicalCheck, ...]
    incomplete: bool
    scope_note: str

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.items)

    def markdown(self) -> str:
        lines = [
            "# sceneweave 运行后审计证据报告",
            "",
            "> " + escape_markdown(self.scope_note),
            "> 报告是投影：原始组件调用提示与重复指令不整段复制，完整原文可在输出目录的 "
            "`simulation_log.json` 按 stage/周期/角色检索；模型原始回答与候选不是已提交事实。",
            "",
            "## 规则语义（仅投影，权威见 docs/）",
            "",
        ]
        for item in self.items:
            if item.kind == "rule":
                lines.append("- " + escape_markdown(f"{item.evidence_id}：{item.text}"))
        lines += ["", "## 机械检查", ""]
        if not self.checks:
            lines.append("- 未执行机械检查。")
        for check in self.checks:
            refs = ("；证据：" + "、".join(check.evidence_refs)) if check.evidence_refs else ""
            lines.append(
                "- " + escape_markdown(f"[{check.status}] {check.check_id}：{check.detail}{refs}")
            )
        lines += ["", "## 证据清单", ""]
        for item in self.items:
            if item.kind == "rule":
                continue
            head = f"- {item.evidence_id}（{item.kind}）"
            meta = []
            if item.cycle_id:
                meta.append(f"周期 {item.cycle_id}")
            if item.simulation_time_ms is not None:
                meta.append(f"{item.simulation_time_ms}ms")
            if item.role_id:
                meta.append(f"角色 {item.role_id}")
            if meta:
                head += " [" + " · ".join(meta) + "]"
            lines.append(escape_markdown(head + "：" + item.text))
        return "\n".join(lines) + "\n"


def escape_markdown(value: object) -> str:
    """动态文本作为文字导出：转义标题/链接/HTML，保留多行结构。"""
    text = "" if value is None else str(value)
    normalized = text.split("\r")
    text = "".join(normalized)
    rendered_lines: list[str] = []
    for line in text.split("\n"):
        chars: list[str] = []
        for index, char in enumerate(line):
            if char == "&":
                chars.append("&amp;")
            elif char == "<":
                chars.append("&lt;")
            elif char == ">":
                chars.append("&gt;")
            elif char in _MD_SPECIAL:
                chars.append("\\" + char)
            elif char in "-+" and not line[:index].strip():
                chars.append("\\" + char)
            elif char == "." and re.fullmatch(r" *\d+", line[:index]):
                chars.append("\\" + char)
            else:
                chars.append(char)
        rendered_lines.append("".join(chars))
    return "\n".join(rendered_lines)


def markdown_lines(value: object, prefix: str) -> str:
    """把可能多行的值渲染为列表项：续行缩进，不产生新标题/链接。"""
    parts = escape_markdown(value).split("\n")
    return "\n".join([prefix + parts[0]] + ["  " + line for line in parts[1:]])


_RULES: tuple[str, ...] = (
    "来源 docs/核心循环与角色职责说明.md「调度语义」：同刻事件与行动整批联合裁决；"
    "“同刻整批”指同一时刻一次裁决与提交，不表示每个周期只能有一次提交——"
    "撤回、明确修改与后续结算可以是同周期内的另行裁决。",
    "来源 docs/核心循环与角色职责说明.md「World 与结果」与 docs/模块交互契约.md："
    "裁决候选提交前必须经过运行时校验；提交版本严格递增且等于 World 当前实际版本。",
    "来源 docs/核心循环与角色职责说明.md：「完成/失败/未确认保持区分」；"
    "unconfirmed 表示结果未知，不是已知终止，不能当成功。",
    "来源 docs/核心循环与角色职责说明.md「中断与取消」与「行动粒度与依赖」："
    "行动终态身份不可复用；已终止的行动不能重新开始或重新维持。",
    "来源 docs/核心循环与角色职责说明.md「通信与信息」：信息的产生、发送、送达、获得与"
    "理解是不同阶段；只有角色实际获得的信息才进入其认知，私有认知与审计意见不进入"
    "其他角色感知或世界事实。",
    "来源 docs/模块交互契约.md「World 检查与事实提交」：只有已提交的结构事实与结算结果"
    "才是客观事实；模型自由描述、检查通过与候选输出不构成事实。",
    "来源 docs/核心循环与角色职责说明.md「全量主动行为声明与被动影响」："
    "撤回主动控制不等于身体立即停止；已发生效果与被动过程由 World 维护。",
)


def _scene_items(scene: SceneObject) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for obj in scene.walk():
        parts = [f"{obj.id}（{obj.name}，{obj.kind}）"]
        if obj.position is not None:
            parts.append(f"position=({obj.position[0]:.2f}, {obj.position[1]:.2f})")
        if obj.size is not None:
            parts.append(f"size=({obj.size[0]:.2f}×{obj.size[1]:.2f})")
        if obj.state:
            parts.append(
                "state=" + ", ".join(f"{key}={obj.state[key]}" for key in sorted(obj.state))
            )
        if obj.blocks_movement:
            parts.append("blocks_movement")
        if obj.blocks_sight:
            parts.append("blocks_sight")
        child_ids = [child.id for child in obj.children]
        if child_ids:
            parts.append("children=" + ",".join(child_ids))
        items.append(EvidenceItem(evidence_id="", kind="scene", text="；".join(parts)))
    return items


def _record_kind(stage: str) -> str:
    if stage.startswith("engine.cycle") or stage.startswith("cycle"):
        return "cycle"
    if stage.startswith("role") or stage.startswith("world.check"):
        return "role"
    if stage.startswith("world"):
        return "world"
    if stage.startswith("translation") or stage.startswith("registration"):
        return "translation"
    if stage.startswith("engine"):
        return "engine"
    return "trace"


def _record_version(record: TraceRecord) -> int | None:
    detail = record.details.get("version")
    if isinstance(detail, int) and not isinstance(detail, bool):
        return detail
    if record.world_state_version is not None:
        return int(record.world_state_version)
    return None


def _record_text(record: TraceRecord) -> str:
    parts = [f"[{record.stage}]", record.summary]
    version = _record_version(record)
    if version is not None:
        parts.append(f"世界版本={version}")
    identity = []
    if record.role_id:
        identity.append(f"角色 {record.role_id}")
    if record.response_id:
        identity.append(f"响应 {record.response_id}")
    if record.related_response_id:
        identity.append(f"关联响应 {record.related_response_id}")
    if record.action_ids:
        identity.append("行动 " + ",".join(str(item) for item in record.action_ids))
    if record.event_ids:
        identity.append("事件 " + ",".join(str(item) for item in record.event_ids))
    if identity:
        parts.append("；".join(identity))
    if record.stage.endswith(_PROMPT_INVOKE_SUFFIX) or record.stage.startswith(
        ("system1.", "system2.", "system3.")
    ):
        parts.append("原始提示与请求已省略（完整原文见 simulation_log.json）")
    else:
        if record.inputs:
            parts.append("输入：" + " | ".join(record.inputs))
    if record.outputs:
        parts.append("输出：" + " | ".join(record.outputs))
    if record.details:
        details = trace_details_to_json(record.details)
        parts.append("细节：" + "；".join(f"{key}={details[key]}" for key in sorted(details)))
    if record.error:
        parts.append("错误：" + record.error)
    return "。".join(parts)


def _coverage_checks(records: Sequence[TraceRecord]) -> list[MechanicalCheck]:
    stages = {record.stage for record in records}
    required = {
        "world.adjudicate.candidate": "World 联合裁决候选",
        "world.validate": "裁决运行时校验",
        "world.commit": "事实提交",
        "registration.commit": "统一登记生效",
        "role.response": "角色完整声明",
        "world.progress.applied": "进展实际变化",
    }
    checks = []
    for stage, label in required.items():
        if stage in stages:
            checks.append(MechanicalCheck(f"trace-coverage:{stage}", "passed", f"存在{label}追踪"))
        else:
            checks.append(
                MechanicalCheck(
                    f"trace-coverage:{stage}",
                    "not_covered",
                    f"缺少{label}追踪；该方面未被本报告覆盖，不能视为通过",
                )
            )
    checks.append(
        MechanicalCheck(
            "batch-order-semantics",
            "not_covered",
            "同刻整批联合裁决的语义先后是否合理，机械检查不判断；"
            "“同刻整批”不表示每周期只能一次提交（撤回/修改另有裁决），需要结合候选与提交证据阅读。",
        )
    )
    checks.append(
        MechanicalCheck(
            "communication-content",
            "not_covered",
            "通信正文/发送/送达/理解没有专门结构证据；机械检查不按关键字硬断言，"
            "通信类问题应标信息不足或疑点。",
        )
    )
    checks.append(
        MechanicalCheck(
            "terminal-lifecycle",
            "not_covered",
            "终态是否被错误重启/复用需要逐行动比对登记与节点证据；"
            "当前报告不做关键字推断，未覆盖部分不得视为通过。",
        )
    )
    return checks


def _commit_records(
    records: Sequence[TraceRecord],
) -> list[tuple[TraceRecord, int]]:
    """两层 world.commit 留痕按真实提交身份合并；返回每笔真实提交。

    引擎与 World 各自为同一次提交记录一条 world.commit：World 组件记录带
    details.version，引擎边界记录带 world_state_version。按
    (周期, 时刻, 版本) 计数配对，同一层重复出现（例如同版本两次提交）
    保留为多笔，交给严格递增检查报错。
    """
    pending: Counter[tuple[str, int, int, bool]] = Counter()
    merged: list[tuple[TraceRecord, int]] = []
    for record in records:
        if record.stage != "world.commit":
            continue
        version = _record_version(record)
        if version is None:
            continue
        origin = "version" in record.details
        identity = (str(record.cycle_id), int(record.simulation_time), version)
        counterpart = (*identity, not origin)
        if pending[counterpart]:
            pending[counterpart] -= 1
            continue
        pending[(*identity, origin)] += 1
        # 保持真实留痕顺序，不能排序后把版本回退掩盖成递增。
        merged.append((record, version))
    return merged


def _state_versions(
    records: Sequence[TraceRecord], commits: Sequence[tuple[TraceRecord, int]]
) -> list[int]:
    commit_ids = {id(record) for record, _ in commits}
    versions: list[int] = []
    for record in records:
        if id(record) not in commit_ids and record.stage != "world.progress.applied":
            continue
        version = _record_version(record)
        if version is not None:
            versions.append(version)
    return versions


def run_mechanical_checks(
    items: Sequence[EvidenceItem],
    records: Sequence[TraceRecord],
    summary: AuditSummary,
    record_refs: Mapping[int, str] | None = None,
) -> tuple[MechanicalCheck, ...]:
    """确定性结构检查；只对明确可判定的结构报找出问题。"""
    checks = _coverage_checks(records)

    def refs_for(selected: Sequence[TraceRecord]) -> tuple[str, ...]:
        if not record_refs:
            return ()
        return tuple(record_refs[id(record)] for record in selected if id(record) in record_refs)

    completes = [record for record in records if record.stage == "engine.cycle_complete"]
    if not completes:
        checks.append(
            MechanicalCheck(
                "summary-cycle-consistency",
                "not_covered",
                "缺少 engine.cycle_complete 完整周期边界；ResponseRunner 的 cycle.complete "
                "只表示角色屏障完成，不证明翻译与统一登记完成。",
            )
        )
    else:
        refs = refs_for(completes)
        cycle_ids = {record.cycle_id for record in completes}
        reported_counts: list[int] = []
        for record in completes:
            count = record.details.get("completed_cycles")
            if isinstance(count, int) and not isinstance(count, bool):
                reported_counts.append(count)
        if len(cycle_ids) != summary.completed_cycles:
            checks.append(
                MechanicalCheck(
                    "summary-cycle-consistency",
                    "finding",
                    f"完整周期边界 {len(cycle_ids)} 个，但概要报告完成周期 "
                    f"{summary.completed_cycles}",
                    refs,
                )
            )
        elif reported_counts and max(reported_counts) != summary.completed_cycles:
            checks.append(
                MechanicalCheck(
                    "summary-cycle-consistency",
                    "finding",
                    f"周期完成追踪计数 {max(reported_counts)} 与概要 "
                    f"{summary.completed_cycles} 不一致",
                    refs,
                )
            )
        else:
            checks.append(
                MechanicalCheck(
                    "summary-cycle-consistency", "passed", "完整周期边界与概要一致", refs
                )
            )

    commits = _commit_records(records)
    if not commits:
        checks.append(
            MechanicalCheck(
                "commit-version-monotonic",
                "not_covered",
                "没有可识别的 world.commit 结构证据，无法核对提交版本",
            )
        )
    else:
        versions = [version for _, version in commits]
        problems = [
            (versions[index - 1], versions[index])
            for index in range(1, len(versions))
            if versions[index] <= versions[index - 1]
        ]
        commit_records_only = [record for record, _ in commits]
        if problems:
            checks.append(
                MechanicalCheck(
                    "commit-version-monotonic",
                    "finding",
                    f"提交版本未严格递增（可能同版本重复提交或版本回退）：{problems}",
                    refs_for(commit_records_only),
                )
            )
        else:
            checks.append(
                MechanicalCheck(
                    "commit-version-monotonic",
                    "passed",
                    f"合并引擎/World 两层留痕后 {len(versions)} 笔提交版本严格递增",
                    refs_for(commit_records_only),
                )
            )

    state_versions = _state_versions(records, commits)
    if not state_versions:
        checks.append(
            MechanicalCheck(
                "state-version-consistency",
                "not_covered",
                "没有提交或进展的结构版本证据，无法核对最终世界版本",
            )
        )
    elif any(
        after < before for before, after in zip(state_versions, state_versions[1:], strict=False)
    ):
        checks.append(
            MechanicalCheck("state-version-consistency", "finding", "提交或进展版本出现回退")
        )
    elif state_versions[-1] != summary.world_state_version:
        checks.append(
            MechanicalCheck(
                "state-version-consistency",
                "finding",
                f"提交与进展中的最后结构版本 {state_versions[-1]} 与概要世界版本 "
                f"{summary.world_state_version} 不一致",
            )
        )
    else:
        checks.append(
            MechanicalCheck(
                "state-version-consistency",
                "passed",
                f"提交与进展结构版本单调，最后版本 {state_versions[-1]} 与概要一致"
                "（最后版本可来自进展更新而非提交）",
            )
        )

    times = [
        record.simulation_time
        for record in records
        if record.stage in ("engine.cycle_start", "engine.cycle_complete")
    ]
    if not times:
        checks.append(
            MechanicalCheck("cycle-time-monotonic", "not_covered", "没有周期时刻追踪，无法核对时间")
        )
    elif any(times[index] < times[index - 1] for index in range(1, len(times))):
        checks.append(MechanicalCheck("cycle-time-monotonic", "finding", "周期时刻出现回退"))
    elif times[-1] > summary.final_time_ms:
        checks.append(
            MechanicalCheck(
                "cycle-time-monotonic",
                "finding",
                f"最后周期时刻 {times[-1]}ms 超过概要最终时刻 {summary.final_time_ms}ms",
            )
        )
    else:
        checks.append(
            MechanicalCheck("cycle-time-monotonic", "passed", "周期时刻单调且不超过最终时刻")
        )

    if summary.error:
        checks.append(
            MechanicalCheck(
                "run-completion",
                "not_covered",
                f"运行未正常结束（{summary.stop_reason}）：{summary.error}；"
                f"报告只覆盖已完成的 {summary.completed_cycles} 个周期",
            )
        )
    return tuple(checks)


def build_evidence_report(
    scene: SceneObject,
    records: Sequence[TraceRecord],
    summary: AuditSummary,
    *,
    incomplete: bool,
) -> EvidenceReport:
    """构造固定报告：场景、规则投影、概要、逐条追踪与机械检查。"""
    items: list[EvidenceItem] = []
    items.extend(_scene_items(scene))
    items.extend(EvidenceItem(evidence_id="", kind="rule", text=text) for text in _RULES)
    items.append(
        EvidenceItem(
            evidence_id="",
            kind="summary",
            text=(
                f"停止原因={summary.stop_reason}；最终时刻={summary.final_time_ms}ms；"
                f"完成周期={summary.completed_cycles}；世界版本={summary.world_state_version}；"
                f"错误={summary.error or '无'}"
            ),
        )
    )
    record_refs: dict[int, str] = {}
    numbered: list[EvidenceItem] = []
    for index, item in enumerate(items, start=1):
        numbered.append(
            EvidenceItem(
                evidence_id=f"E-{index:04d}",
                kind=item.kind,
                text=item.text,
                cycle_id=item.cycle_id,
                simulation_time_ms=item.simulation_time_ms,
                role_id=item.role_id,
                action_ids=item.action_ids,
            )
        )
    for record in records:
        evidence_id = f"E-{len(numbered) + 1:04d}"
        numbered.append(
            EvidenceItem(
                evidence_id=evidence_id,
                kind=_record_kind(record.stage),
                text=_record_text(record),
                cycle_id=str(record.cycle_id),
                simulation_time_ms=int(record.simulation_time),
                role_id=str(record.role_id) if record.role_id else None,
                action_ids=tuple(str(item) for item in record.action_ids),
            )
        )
        record_refs[id(record)] = evidence_id
    checks = run_mechanical_checks(tuple(numbered), records, summary, record_refs)
    if incomplete:
        note = (
            f"运行未正常结束（{summary.stop_reason}）：报告只覆盖截至 {summary.final_time_ms}ms 的"
            f"已完成 {summary.completed_cycles} 个周期与对应追踪，不伪造完整周期或最终事实。"
        )
    else:
        note = (
            f"运行结束（{summary.stop_reason}）：覆盖全部 {summary.completed_cycles} 个已完成周期，"
            "包含失败、未确认与撤回证据，不做成功筛选。"
        )
    return EvidenceReport(
        items=tuple(numbered), checks=checks, incomplete=incomplete, scope_note=note
    )
