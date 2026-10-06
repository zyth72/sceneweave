"""审计模型输出契约：严格 JSON 校验与安全错误分类。

每份输出只接受 summary 与 findings；findings 的 severity/certainty 使用
固定枚举，evidence_refs 必须落在报告的 evidence_id 范围内，虚构引用整体
拒绝。外部异常正文可能泄密，这里按异常类别映射为安全文本。
"""

import hashlib
from dataclasses import dataclass

from sceneweave.domain.model_json import ModelJsonError, parse_model_json

SEVERITIES = ("info", "suspicion", "error")
CERTAINTIES = ("certain", "likely", "uncertain")
_FINDING_KEYS = ("severity", "certainty", "title", "explanation", "evidence_refs", "suggestion")


class AuditOutputError(ValueError):
    """审计模型输出未通过结构校验；不得当作通过或空结论。"""


@dataclass(frozen=True, slots=True)
class AuditFinding:
    severity: str
    certainty: str
    title: str
    explanation: str
    evidence_refs: tuple[str, ...]
    suggestion: str


@dataclass(frozen=True, slots=True)
class AuditModelReport:
    model_id: str
    model: str
    summary: str
    findings: tuple[AuditFinding, ...]


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_audit_prompt(report_markdown: str, evidence_ids: frozenset[str], model_id: str) -> str:
    refs = "、".join(sorted(evidence_ids)) or "（无）"
    return (
        "你是 sceneweave 的独立观察者审计员，只审计下面的固定证据报告，不参与模拟。\n"
        f"本次审计模型标识：{model_id}。\n"
        "机械检查与模型意见分开：机械检查只给确定结构结论；你不得把它当作已通过的整体。\n"
        "报告包含全部已完成周期（含失败/未确认/撤回），没有做成功筛选。\n"
        "只能基于报告中的证据作结论；证据不足时明确写信息不足/疑点，不要凭自然语言关键词硬断言因果。\n"
        f"evidence_refs 只能使用报告中的证据 id（{refs}），不得虚构。\n"
        "只输出一个 JSON 对象，只含 summary 与 findings 两个字段：\n"
        '{"summary": "非空总结", "findings": [{"severity": "info|suspicion|error",'
        ' "certainty": "certain|likely|uncertain", "title": "非空标题",'
        ' "explanation": "非空解释", "evidence_refs": ["E-0001"],'
        ' "suggestion": "非空建议"}]}\n'
        "findings 可以为空数组，但不能省略字段；不要输出 JSON 以外的任何文字。\n"
        "以下是固定证据报告：\n\n" + report_markdown
    )


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditOutputError(f"audit 输出 {field} 必须是非空字符串")
    return value


def parse_audit_report(
    raw: str,
    *,
    model_id: str,
    model: str,
    evidence_ids: frozenset[str],
) -> AuditModelReport:
    """严格解析一份审计输出；任何结构问题整体拒绝，不做修复。"""
    try:
        data = parse_model_json(raw, "audit", allowed_keys=("summary", "findings"))
    except ModelJsonError as error:
        message = str(error)
        if "不是合法 JSON" in message or "重复键" in message or "非有限" in message:
            raise AuditOutputError("audit 输出不是合法 JSON 对象") from error
        raise AuditOutputError(message) from error
    if "summary" not in data or "findings" not in data:
        missing = [key for key in ("summary", "findings") if key not in data]
        raise AuditOutputError(f"audit 输出缺少必填字段：{missing}")
    summary = _require_text(data["summary"], "summary")
    raw_findings = data["findings"]
    if not isinstance(raw_findings, list):
        raise AuditOutputError("audit 输出 findings 必须是数组")
    findings: list[AuditFinding] = []
    for index, item in enumerate(raw_findings):
        at = f"findings[{index}]"
        if not isinstance(item, dict):
            raise AuditOutputError(f"audit 输出 {at} 必须是对象")
        unknown = sorted(set(item) - set(_FINDING_KEYS))
        if unknown:
            raise AuditOutputError(f"audit 输出 {at} 包含未知字段：{unknown}")
        missing = [key for key in _FINDING_KEYS if key not in item]
        if missing:
            raise AuditOutputError(f"audit 输出 {at} 缺少必填字段：{missing}")
        severity = item["severity"]
        if severity not in SEVERITIES:
            raise AuditOutputError(f"audit 输出 {at}.severity 不是允许的类别")
        certainty = item["certainty"]
        if certainty not in CERTAINTIES:
            raise AuditOutputError(f"audit 输出 {at}.certainty 不是允许的类别")
        raw_refs = item["evidence_refs"]
        if not isinstance(raw_refs, list):
            raise AuditOutputError(f"audit 输出 {at}.evidence_refs 必须是数组")
        if not raw_refs:
            raise AuditOutputError(f"audit 输出 {at}.evidence_refs 至少需要一个有效证据引用")
        refs: list[str] = []
        for ref in raw_refs:
            if not isinstance(ref, str) or ref not in evidence_ids:
                raise AuditOutputError(f"audit 输出 {at}.evidence_refs 引用了报告之外的证据")
            refs.append(ref)
        if len(set(refs)) != len(refs):
            raise AuditOutputError(f"audit 输出 {at}.evidence_refs 重复")
        findings.append(
            AuditFinding(
                severity=severity,
                certainty=certainty,
                title=_require_text(item["title"], f"{at}.title"),
                explanation=_require_text(item["explanation"], f"{at}.explanation"),
                evidence_refs=tuple(refs),
                suggestion=_require_text(item["suggestion"], f"{at}.suggestion"),
            )
        )
    return AuditModelReport(
        model_id=model_id, model=model, summary=summary, findings=tuple(findings)
    )


def safe_error_text(error: BaseException) -> str:
    """异常类别映射：只保留项目内可控错误正文，外部正文不入审计输出。"""
    module = type(error).__module__ or ""
    if module.startswith("sceneweave"):
        return f"{type(error).__name__}: {error}"
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return f"{type(error).__name__}（上游状态码 {status}）"
    return f"{type(error).__name__}（外部异常正文不记录）"
