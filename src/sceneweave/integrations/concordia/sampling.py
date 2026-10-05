"""Concordia 语言模型接入角色的采样桥接与录制/重放。

LanguageModelSampler 把锁定的 Concordia LanguageModel 同步调用放到
线程中异步等待（事件循环保持可用，剧情时刻由引擎冻结）。注意：线程
中的同步模型调用无法被硬取消——取消/超时只放弃等待结果，底层调用会
继续运行到自然返回；这是如实限制，不宣称已停止后端。

录制按角色作用域隔离：跨角色并行采样的完成顺序不确定，共享一个录制
文件无法重放；每个角色使用独立的 RecordingSampler/ReplaySampler（或
RecordingDirectory/ReplayDirectory 的 for_scope 入口）即可确定性重放
整个双角色运行。失败调用同样记录（kind=error），重放时重现原异常。
请求与录制必须逐条严格一致，错配不消费记录、立即报错。本模块不依赖
在途的 model_trace.py。
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import quote

from concordia.language_model import language_model
from concordia.typing import logging as logging_lib
from concordia.typing.entity_component import ComponentWithLogging
from concordia.typing.logging import LoggingChannel

_RECORD_VERSION = 1


@runtime_checkable
class SupportsChannelBinding(Protocol):
    """采样器可选的公开通道绑定接口；运行层据此注入组件日志通道。

    channel_for 以组件阶段名（system1/world-progress 等）为参数，
    返回写入同一结构化日志的通道；未绑定的采样器照常工作（无日志）。
    """

    def bind_channel_for(self, channel_for: Callable[[str], LoggingChannel]) -> None: ...


class SamplerComponent(ComponentWithLogging):
    """真实的 Concordia 组件：执行一次采样并经日志通道记录原始内容。

    基类未发布类型存根，这里显式实现等价的通道初始化与设置；组件协议
    （get_state/set_state）与基类一致。datum 携带本次调用所属的阶段
    名（component），通道可按阶段建因果身份。
    """

    def __init__(self, model: language_model.LanguageModel) -> None:
        # 以默认参数调用 sample_text；窄类型保持类型检查可用。
        self._sample_text = cast(Callable[[str], str], model.sample_text)
        self._logging_channel: LoggingChannel = logging_lib.NoOpLoggingChannel

    def set_logging_channel(self, logging_channel: LoggingChannel) -> None:
        self._logging_channel = logging_channel

    def get_state(self) -> dict[str, str]:
        return {}

    def set_state(self, state: Any) -> None:
        del state  # 采样组件无可变状态

    def sample(self, prompt: str, *, component: str) -> str:
        response = self._sample_text(prompt)
        if not isinstance(response, str):  # 模型后端违约时如实暴露
            raise TypeError("语言模型 sample_text 必须返回字符串")
        self._logging_channel(
            {
                "Key": "sample_text",
                "Component": component,
                "Prompt": prompt,
                "Response": response,
            }
        )
        return response


class LanguageModelSampler:
    """以真实语言模型为后端的采样器；异步桥保持事件循环可用。

    channel_for 可按阶段（system1/system2/system3）提供各自的日志通道，
    使组件日志携带真实阶段与因果身份；未提供时共用桥接通道，datum 的
    Component 字段仍标记实际阶段。取消只放弃等待，见模块说明。
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        *,
        channel_for: Callable[[str], LoggingChannel] | None = None,
    ) -> None:
        self._component = SamplerComponent(model)
        self._channel_for = channel_for

    def bind_channel_for(self, channel_for: Callable[[str], LoggingChannel]) -> None:
        """公开绑定组件日志通道（组件阶段名 → 通道）；供运行层注入。

        不通过私有字段赋值；绑定后每次采样都会经通道写出 Prompt 与
        Response 原文。重复绑定以最后一次为准。
        """
        self._channel_for = channel_for

    @property
    def component(self) -> SamplerComponent:
        return self._component

    async def sample(self, prompt: str, *, component: str) -> str:
        if self._channel_for is None:
            return await asyncio.to_thread(self._component.sample, prompt, component=component)
        channel = self._channel_for(component)
        try:
            response = await asyncio.to_thread(self._component.sample, prompt, component=component)
        except BaseException as error:
            try:
                channel(
                    {
                        "Key": "sample_text_error",
                        "Component": component,
                        "Prompt": prompt,
                        "ErrorType": f"{type(error).__module__}.{type(error).__qualname__}",
                        "ErrorMessage": str(error),
                    }
                )
            except Exception as trace_error:
                error.add_note(f"组件错误日志写入失败：{trace_error}")
            raise
        channel(
            {
                "Key": "sample_text",
                "Component": component,
                "Prompt": prompt,
                "Response": response,
            }
        )
        return response


class RecordedModelError(RuntimeError):
    """重放到录制时记录的失败；保留错误身份，不动态创建异常类。"""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.original_message = message
        super().__init__(f"录制的模型调用失败：{error_type}: {message}")


class RecordingSampler:
    """包装任一采样器，把精确请求与结果（含失败）写入 JSONL 文件。

    文件必须不存在，防止覆盖旧故障现场。跨角色并行时每个角色使用独立
    的录制文件（见 RecordingDirectory），同一作用域内的调用顺序由同
    角色串行保证，可确定性重放。
    """

    def __init__(self, inner: Any, path: Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("运行身份必须是非空字符串")
        self._inner = inner
        self._path = path
        self._file = path.open("x", encoding="utf-8")
        self._file.write(
            json.dumps(
                {"kind": "header", "version": _RECORD_VERSION, "run_id": run_id},
                ensure_ascii=False,
            )
            + "\n"
        )
        self._file.flush()
        self.count = 0

    async def sample(self, prompt: str, *, component: str) -> str:
        try:
            response = await self._inner.sample(prompt, component=component)
        except Exception as error:
            self._write(
                {
                    "kind": "error",
                    "component": component,
                    "prompt": prompt,
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                }
            )
            raise
        if not isinstance(response, str):
            raise TypeError("被包装的采样器必须返回字符串")
        self._write(
            {
                "kind": "call",
                "component": component,
                "prompt": prompt,
                "response": response,
            }
        )
        return response

    def _write(self, record: dict[str, object]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RecordingSampler:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ReplayMismatchError(RuntimeError):
    """重放请求与录制不一致；不能继续使用旧结果。"""


class _RecordFile:
    """共享的录制文件解析：严格头部与字段校验。"""

    def __init__(self, path: Path) -> None:
        self.calls: list[dict[str, str]] = []
        self.header: dict[str, Any] = {}
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReplayMismatchError(
                        f"录制文件第 {line_number} 行不是合法 JSON：{error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ReplayMismatchError(f"录制文件第 {line_number} 行不是对象")
                if line_number == 1:
                    if (
                        record.get("kind") != "header"
                        or type(record.get("version")) is not int
                        or record["version"] != _RECORD_VERSION
                        or not isinstance(record.get("run_id"), str)
                        or not record["run_id"]
                    ):
                        raise ReplayMismatchError("录制文件头部缺失或版本/运行身份非法")
                    self.header = record
                    continue
                kind = record.get("kind")
                if kind not in ("call", "error"):
                    raise ReplayMismatchError(f"录制文件第 {line_number} 行类型未知")
                for field in ("component", "prompt"):
                    if not isinstance(record.get(field), str):
                        raise ReplayMismatchError(f"录制文件第 {line_number} 行缺少字段 {field}")
                if kind == "call" and not isinstance(record.get("response"), str):
                    raise ReplayMismatchError(f"录制文件第 {line_number} 行缺少字段 response")
                if kind == "error" and not (
                    isinstance(record.get("error_type"), str)
                    and isinstance(record.get("message"), str)
                ):
                    raise ReplayMismatchError(f"录制文件第 {line_number} 行错误记录缺少类型或消息")
                self.calls.append(record)
        if not self.header:
            raise ReplayMismatchError("录制文件为空或缺少头部")


class ReplaySampler:
    """按录制顺序严格匹配请求并返回录制结果；不持有真实模型。

    请求（component 与 prompt 逐字一致）不匹配时不消费记录并立即报错；
    录制的失败以 RecordedModelError 重现，不重新采样。每个作用域（角色）
    使用独立录制文件。
    """

    def __init__(self, path: Path) -> None:
        record = _RecordFile(path)
        self._calls = record.calls
        self._index = 0
        self.header: dict[str, Any] = record.header
        self._channel_for: Callable[[str], LoggingChannel] | None = None

    def bind_channel_for(self, channel_for: Callable[[str], LoggingChannel]) -> None:
        """公开绑定组件日志通道；重放不建真实模型也能查询组件日志。

        每次回放一条录制（含录制的失败）都经通道写出实际服务的
        Prompt 与 Response（或错误身份），与真实采样同一格式。
        """
        self._channel_for = channel_for

    def _emit_channel(self, prompt: str, component: str, record: dict[str, str]) -> None:
        if self._channel_for is None:
            return
        datum: dict[str, object] = {
            "Key": "sample_text",
            "Component": component,
            "Prompt": prompt,
            "Response": record.get("response", ""),
            "Replay": True,
        }
        if record["kind"] == "error":
            datum["Key"] = "sample_text_error"
            datum["ErrorType"] = record["error_type"]
            datum["ErrorMessage"] = record["message"]
        self._channel_for(component)(datum)

    async def sample(self, prompt: str, *, component: str) -> str:
        if self._index >= len(self._calls):
            raise ReplayMismatchError("缺少录制记录：调用次数超出录制")
        record = self._calls[self._index]
        if record["component"] != component or record["prompt"] != prompt:
            raise ReplayMismatchError(
                f"请求与录制不一致（第 {self._index + 1} 条）："
                f"component={component!r}, prompt 前 80 字符={prompt[:80]!r}"
            )
        self._index += 1
        if record["kind"] == "error":
            error = RecordedModelError(record["error_type"], record["message"])
            try:
                self._emit_channel(prompt, component, record)
            except Exception as trace_error:
                error.add_note(f"组件错误日志写入失败：{trace_error}")
            raise error
        self._emit_channel(prompt, component, record)
        return record["response"]

    def assert_fully_consumed(self) -> None:
        """检查是否完整回放；残留说明运行偏离了录制。"""
        if self._index != len(self._calls):
            raise ReplayMismatchError(f"还有未回放的录制调用：{len(self._calls) - self._index} 条")


def scope_file(directory: Path, scope: str, suffix: str = ".jsonl") -> Path:
    """作用域录制文件路径：每个角色（或明确作用域）一个文件。"""
    if not isinstance(scope, str) or not scope:
        raise ValueError("作用域名必须是非空字符串")
    return directory / f"model-{quote(scope, safe='')}{suffix}"


class RecordingDirectory:
    """按作用域创建录制器的入口：跨角色并行运行的可重放编组。

    inner_factory 收到作用域名（通常是角色名），为每个作用域创建各自的
    后端采样器；每个作用域一个录制文件，重放时顺序确定。
    """

    def __init__(
        self, inner_factory: Callable[[str], Any], directory: Path, *, run_id: str
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("运行身份必须是非空字符串")
        self._inner_factory = inner_factory
        self._directory = directory
        self._run_id = run_id
        self._recorders: list[RecordingSampler] = []
        directory.mkdir(parents=True, exist_ok=True)

    def for_scope(self, scope: str) -> RecordingSampler:
        recorder = RecordingSampler(
            self._inner_factory(scope), scope_file(self._directory, scope), run_id=self._run_id
        )
        self._recorders.append(recorder)
        return recorder

    def close(self) -> None:
        for recorder in self._recorders:
            recorder.close()

    def __enter__(self) -> RecordingDirectory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ReplayDirectory:
    """按作用域创建重放器的入口；与 RecordingDirectory 的文件对应。"""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._replayers: list[ReplaySampler] = []
        self._run_id: str | None = None

    def for_scope(self, scope: str) -> ReplaySampler:
        replayer = ReplaySampler(scope_file(self._directory, scope))
        run_id = replayer.header["run_id"]
        if self._run_id is not None and run_id != self._run_id:
            raise ReplayMismatchError("同一重放目录的作用域必须属于同一运行身份")
        self._run_id = run_id
        self._replayers.append(replayer)
        return replayer

    def assert_fully_consumed(self) -> None:
        for replayer in self._replayers:
            replayer.assert_fully_consumed()


def component_channel(
    simulation_log: Any,
    *,
    entity_name: str,
    component_name: str,
    step: Callable[[], int],
    timestamp: Callable[[], str] | None = None,
) -> LoggingChannel:
    """便捷转发到既有的 component_logging_channel（同一 SimulationLog）。"""
    from sceneweave.integrations.concordia.tracing import component_logging_channel

    return component_logging_channel(
        simulation_log,
        entity_name=entity_name,
        component_name=component_name,
        step=step,
        timestamp=timestamp,
    )
