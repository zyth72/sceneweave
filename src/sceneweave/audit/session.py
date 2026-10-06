"""审计会话：固定报告 → 机械检查 → 多模型独立审计 → 结果落盘与状态。

一个模型失败不阻止其他模型；审计异常不回写模拟 summary 或世界事实。
会话状态线程安全，供 UI 快照读取；审计只触发一次，刷新页面只读状态。
"""

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sceneweave.audit.config import AuditConfig, AuditModelSpec
from sceneweave.audit.models import (
    AuditOutputError,
    build_audit_prompt,
    parse_audit_report,
    prompt_fingerprint,
    safe_error_text,
)
from sceneweave.audit.report import (
    AuditSummary,
    EvidenceReport,
    build_evidence_report,
    escape_markdown,
    markdown_lines,
)
from sceneweave.domain.modeling import TextSampler
from sceneweave.domain.tracing import TraceRecord
from sceneweave.world.scene import SceneObject


class AuditState:
    """只读快照使用的审计状态；revision 随每次真实进度递增。"""

    def __init__(self, model_ids: Sequence[str], model_names: dict[str, str]) -> None:
        self._lock = threading.Lock()
        self._revision = 0
        self._status = "waiting"
        self._scope = ""
        self._error: str | None = None
        self._evidence_count = 0
        self._checks: list[dict[str, object]] = []
        self._models: list[dict[str, object]] = [
            {"id": model_id, "model": model_names.get(model_id, model_id), "status": "pending"}
            for model_id in model_ids
        ]

    def _bump(self) -> None:
        self._revision += 1

    def _model(self, model_id: str) -> dict[str, object]:
        for item in self._models:
            if item["id"] == model_id:
                return item
        raise KeyError(model_id)

    def mark_checking(self) -> None:
        with self._lock:
            self._status = "checking"
            self._bump()

    def set_report(self, report: EvidenceReport) -> None:
        with self._lock:
            self._scope = report.scope_note
            self._evidence_count = len(report.items)
            self._checks = [
                {
                    "check_id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                    "evidence_refs": list(check.evidence_refs),
                }
                for check in report.checks
            ]
            self._bump()

    def mark_running(self, model_id: str) -> None:
        with self._lock:
            self._status = "running"
            self._model(model_id)["status"] = "running"
            self._bump()

    def set_model_result(self, model_id: str, payload: dict[str, object]) -> None:
        with self._lock:
            item = self._model(model_id)
            item.update(payload)
            self._bump()

    def mark_done(self) -> None:
        with self._lock:
            self._status = "done"
            self._bump()

    def mark_failed(self, error: str) -> None:
        """记录真实错误说明，不改写已取得的报告范围 scope。"""
        with self._lock:
            self._status = "failed"
            self._error = error
            self._bump()

    def mark_cancelled(self) -> None:
        with self._lock:
            self._status = "cancelled"
            self._error = (
                "审计被取消；部分模型可能已执行，结果只覆盖已完成部分，不代表模拟失败或审计通过。"
            )
            self._bump()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            failed = sum(1 for item in self._models if item.get("status") == "failed")
            return {
                "enabled": True,
                "status": self._status,
                "revision": self._revision,
                "scope": self._scope,
                "error": self._error,
                "evidence_count": self._evidence_count,
                "checks": [dict(item) for item in self._checks],
                "models": [dict(item) for item in self._models],
                "failed_models": failed,
            }


class AuditSession:
    """一次运行的审计；prepare 校验模型配置，run 执行并落盘。"""

    def __init__(
        self,
        config: AuditConfig,
        model_factory: Callable[[AuditModelSpec], TextSampler],
        *,
        state: AuditState | None = None,
    ) -> None:
        self._config = config
        self._model_factory = model_factory
        self._samplers: dict[str, TextSampler] = {}
        self._run_lock = threading.Lock()
        self._started = False
        self.state = state or AuditState(
            [spec.model_id for spec in config.models],
            {spec.model_id: spec.model_name for spec in config.models},
        )

    def prepare(self) -> None:
        """预构造各模型（不读环境变量、不联网）；配置错误在模拟前暴露。"""
        for spec in self._config.models:
            if spec.model_id in self._samplers:
                continue
            self._samplers[spec.model_id] = self._model_factory(spec)

    def _write_text(self, directory: Path, filename: str, text: str) -> None:
        (directory / filename).write_text(text, encoding="utf-8")

    def _audit_markdown(self, report: EvidenceReport, results: list[dict[str, object]]) -> str:
        lines = [
            "# sceneweave 运行后审计",
            "",
            "> " + escape_markdown(report.scope_note),
            "",
            "## 机械检查（确定性）",
            "",
        ]
        for check in report.checks:
            lines.append(
                "- " + escape_markdown(f"[{check.status}] {check.check_id}：{check.detail}")
            )
        lines += ["", "## 模型结论（独立，不投票改事实）", ""]
        for item in results:
            lines.append(
                "### " + escape_markdown(f"{item['id']}（{item['model']}）— {item['status']}")
            )
            lines.append("")
            if item.get("error"):
                lines.append(markdown_lines("错误：" + str(item["error"]), "- "))
            if item.get("summary"):
                lines.append(markdown_lines("总结：" + str(item["summary"]), "- "))
            findings = item.get("findings")
            if isinstance(findings, list) and findings:
                for finding in findings:
                    refs = "、".join(finding.get("evidence_refs", []))
                    lines.append(
                        markdown_lines(
                            f"[{finding.get('severity')}/{finding.get('certainty')}] "
                            f"{finding.get('title')}：{finding.get('explanation')}"
                            + (f"（证据：{refs}）" if refs else "")
                            + f" 建议：{finding.get('suggestion')}",
                            "- ",
                        )
                    )
            elif item["status"] == "ok":
                lines.append("- 没有 findings。")
            lines.append("")
        return "\n".join(lines) + "\n"

    async def run(
        self,
        *,
        scene: SceneObject,
        records: Sequence[TraceRecord],
        summary: AuditSummary,
        output_directory: Path,
        incomplete: bool,
    ) -> None:
        """执行审计并落盘；内部异常只标记状态，不向调用方抛出。"""
        with self._run_lock:
            if self._started:
                raise RuntimeError("审计会话只能执行一次；重复或并发 run 被拒绝")
            self._started = True
        try:
            self.state.mark_checking()
            report = build_evidence_report(scene, records, summary, incomplete=incomplete)
            self.state.set_report(report)
            self._write_text(output_directory, "report.md", report.markdown())
            results: list[dict[str, object]] = []
            for spec in self._config.models:
                self.state.mark_running(spec.model_id)
                item: dict[str, object] = {
                    "id": spec.model_id,
                    "model": spec.model_name,
                    "status": "failed",
                    "error": None,
                    "summary": None,
                    "findings": [],
                }
                model_prompt = build_audit_prompt(
                    report.markdown(), report.evidence_ids, spec.model_id
                )
                prompt_file = f"audit-{spec.model_id}.prompt.txt"
                response_file = f"audit-{spec.model_id}.response.txt"
                item["prompt_file"] = prompt_file
                item["response_file"] = response_file
                item["prompt_sha256"] = prompt_fingerprint(model_prompt)
                self._write_text(output_directory, prompt_file, model_prompt)
                raw: str | None = None
                try:
                    sampler = self._samplers.get(spec.model_id)
                    if sampler is None:
                        sampler = self._model_factory(spec)
                        self._samplers[spec.model_id] = sampler
                    raw = await sampler.sample(model_prompt, component="audit")
                    self._write_text(output_directory, response_file, raw)
                    parsed = parse_audit_report(
                        raw,
                        model_id=spec.model_id,
                        model=spec.model_name,
                        evidence_ids=report.evidence_ids,
                    )
                    item.update(
                        {
                            "status": "ok",
                            "summary": parsed.summary,
                            "findings": [
                                {
                                    "severity": finding.severity,
                                    "certainty": finding.certainty,
                                    "title": finding.title,
                                    "explanation": finding.explanation,
                                    "evidence_refs": list(finding.evidence_refs),
                                    "suggestion": finding.suggestion,
                                }
                                for finding in parsed.findings
                            ],
                        }
                    )
                except AuditOutputError as error:
                    item["error"] = f"输出未通过校验：{error}"
                except Exception as error:  # noqa: BLE001 模型失败必须隔离
                    item["error"] = safe_error_text(error)
                results.append(item)
                self.state.set_model_result(spec.model_id, dict(item))
            audit_json = {
                "version": 1,
                "incomplete": incomplete,
                "scope": report.scope_note,
                "report_file": "report.md",
                "checks": [
                    {
                        "check_id": check.check_id,
                        "status": check.status,
                        "detail": check.detail,
                        "evidence_refs": list(check.evidence_refs),
                    }
                    for check in report.checks
                ],
                "models": results,
            }
            self._write_text(
                output_directory,
                "audit.json",
                json.dumps(audit_json, ensure_ascii=False, indent=2, allow_nan=False),
            )
            self._write_text(output_directory, "audit.md", self._audit_markdown(report, results))
            self.state.mark_done()
        except Exception as error:  # noqa: BLE001 审计内部错误不掩盖模拟结果
            self.state.mark_failed(safe_error_text(error))
