"""真实 openai SDK + 本地假 HTTP 端点的 Chat Completions 适配器测试。

所有用例都通过真实 SDK 走真实 HTTP（127.0.0.1 假端点），不 monkeypatch
SDK 返回值，也不访问任何真实 API 或真实凭据。覆盖：请求编码/鉴权/
URL 拼接、参数策略与冲突、状态码/超时/连接/重定向/无重试、坏响应拒绝、
异常不泄密、构造无副作用、作用域隔离，以及完整装配 + runner + 组件
日志 + 录制/重放的集成链路（重放阶段服务器已关闭，证明不联网）。
"""

import asyncio
import json
import os
import socket
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from sceneweave.app.assembly import build_live_factory
from sceneweave.app.recording import RecordedRun
from sceneweave.app.runner import run_headless
from sceneweave.config.loader import load_assembly_config
from sceneweave.integrations.concordia.chat_completions import (
    ChatCompletionsConfigError,
    ChatCompletionsError,
    ChatCompletionsLanguageModel,
    ModelConnectionError,
    ModelCredentialsError,
    ModelRequestTimeoutError,
    ModelResponseError,
    ModelStatusError,
    ParameterConflictError,
    UnsupportedParameterError,
    create_model,
)
from sceneweave.integrations.concordia.sampling import LanguageModelSampler

from .test_model_world import Brain, study_card

_KEY_ENV = "SCENEWEAVE_TEST_API_KEY"
_KEY = "sk-sceneweave-test-secret"
_FACTORY = "sceneweave.integrations.concordia.chat_completions:create_model"

_Responder = Callable[[dict[str, Any]], "tuple[int, dict[str, str], bytes]"]
_Reply = tuple[int, dict[str, str], bytes]


@pytest.fixture(autouse=True)
def _clean_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离宿主环境：任何 OPENAI_* 全局配置都不影响用例。"""
    for name in list(os.environ):
        if name.startswith("OPENAI_"):
            monkeypatch.delenv(name, raising=False)


def _chat_payload(
    content: Any,
    *,
    finish_reason: str | None = "stop",
    choice_count: int = 1,
    tool_calls: list[dict[str, Any]] | None = None,
) -> bytes:
    choices = [
        {
            "index": index,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }
        for index in range(choice_count)
    ]
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": choices,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _completion_payload(choices: list[Any]) -> bytes:
    """允许构造缺失/异常字段的原始响应，验证稳定拒绝而非 AttributeError。"""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": choices,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _ok(content: str) -> _Reply:
    return 200, {"Content-Type": "application/json"}, _chat_payload(content)


def _echo(body: dict[str, Any]) -> _Reply:
    prompt = body["messages"][0]["content"]
    return _ok(f"回：{prompt}")


class _FakeChatServer:
    """真实监听 127.0.0.1 的假 Chat Completions 端点。

    POST/GET 都记录请求；responder 返回 (status, headers, payload)。
    daemon 线程随上下文关闭，不留下 socket 或线程。
    """

    def __init__(self, responder: _Responder) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _record(self, method: str) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    body = {}
                record = {
                    "method": method,
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
                owner.requests.append(record)
                return body

            def do_POST(self) -> None:
                body = self._record("POST")
                status, headers, payload = owner.responder(body)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:
                    pass  # 超时用例里客户端可能已断开

            def do_GET(self) -> None:
                self._record("GET")
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        address = self._server.server_address
        assert isinstance(address, tuple)
        return int(address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _FakeChatServer:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@contextmanager
def _server(responder: _Responder) -> Iterator[_FakeChatServer]:
    with _FakeChatServer(responder) as running:
        yield running


def _options(server: _FakeChatServer, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "base_url": f"{server.base_url}/v1",
        "model": "test-model",
        "api_key_env": _KEY_ENV,
    }
    options.update(overrides)
    return options


def test_create_model_rejects_unknown_and_invalid_options() -> None:
    valid: dict[str, Any] = {
        "base_url": "https://example.test/v1",
        "model": "model-x",
        "api_key_env": "MODEL_KEY",
    }
    cases: list[tuple[dict[str, Any], str]] = [
        ({}, "缺少必填字段"),
        ({**valid, "model": ""}, "model 必须是非空字符串"),
        ({**valid, "api_key_env": "BAD NAME"}, "合法环境变量名"),
        ({**valid, "api_key_env": ""}, "api_key_env 必须是非空字符串"),
        ({**valid, "base_url": "ftp://example.test"}, "仅支持 http/https"),
        ({**valid, "base_url": "https://user:pass@example.test"}, "不得包含用户名或密码"),
        ({**valid, "base_url": "https://example.test/v1?a=1"}, "不得包含 query 或 fragment"),
        ({**valid, "base_url": "https://example.test/v1#x"}, "不得包含 query 或 fragment"),
        ({**valid, "base_url": "http://[::1"}, "不是合法的 URL"),
        ({**valid, "base_url": "http://example.test:notaport"}, "不是合法的 URL"),
        ({**valid, "base_url": "http://example.test:99999999"}, "不是合法的 URL"),
        ({**valid, "base_url": "http://:8080"}, "缺少有效主机"),
        ({**valid, "base_url": "http://exa mple.test/v1"}, "空白、控制字符或反斜杠"),
        ({**valid, "base_url": "http://example.test\\v1"}, "空白、控制字符或反斜杠"),
        ({**valid, "base_url": "http://example.test/v1\x01"}, "空白、控制字符或反斜杠"),
        ({**valid, "timeout_seconds": True}, "timeout_seconds 必须是正数"),
        ({**valid, "timeout_seconds": None}, "显式 null"),
        ({**valid, "timeout_seconds": 0}, "正的有限数值"),
        ({**valid, "timeout_seconds": float("inf")}, "正的有限数值"),
        ({**valid, "max_tokens": True}, "max_tokens 必须是正整数"),
        ({**valid, "max_tokens": None}, "显式 null"),
        ({**valid, "max_tokens": 0}, "max_tokens 必须是正整数"),
        ({**valid, "temperature": True}, "temperature 必须是"),
        ({**valid, "temperature": None}, "显式 null"),
        ({**valid, "temperature": 3.0}, "temperature 必须是"),
        ({**valid, "temperature": float("nan")}, "temperature 必须是"),
        ({**valid, "api_key": "plain-secret"}, "未知字段"),
        (
            {**valid, "temperature": 0.25, "extra_body": {"temperature": 1.7}},
            "不得覆盖适配器拥有的字段",
        ),
        ({**valid, "extra_body": None}, "显式 null"),
        ({**valid, "extra_body": {"model": "override"}}, "不得覆盖适配器拥有的字段"),
        ({**valid, "extra_body": {"Stream": True}}, "不得覆盖适配器拥有的字段"),
        ({**valid, "extra_body": {"seed": 1}}, "不得覆盖适配器拥有的字段"),
        ({**valid, "extra_body": {"top_p": 1.0}}, "不得覆盖适配器拥有的字段"),
        ({**valid, "extra_body": {"top_k": 5}}, "不得覆盖适配器拥有的字段"),
        ({**valid, "extra_body": {"thinking": float("nan")}}, "JSON 可序列化"),
        ({**valid, "extra_body": [1, 2]}, "extra_body 必须是 JSON 对象"),
        ({**valid, "extra_body": {1: "x"}}, "键必须是非空字符串"),
    ]
    for options, message in cases:
        with pytest.raises(ChatCompletionsConfigError, match=message):
            create_model("role:lin", options)
    with pytest.raises(ChatCompletionsConfigError, match="scope 必须是非空字符串"):
        create_model("", valid)
    with pytest.raises(ChatCompletionsConfigError, match="options 键必须是字符串"):
        create_model("role:lin", {1: "x"})  # type: ignore[dict-item]


def test_base_url_errors_do_not_echo_credentials_or_query() -> None:
    valid: dict[str, Any] = {
        "base_url": "https://example.test/v1",
        "model": "model-x",
        "api_key_env": "MODEL_KEY",
    }
    for bad in (
        "http://user:secret@example.test/v1",
        "http://example.test/v1?token=secret",
        "http://example.test/v1#secret",
        "http://[::1",
        "http://example.test:notaport",
    ):
        with pytest.raises(ChatCompletionsConfigError) as info:
            create_model("role:lin", {**valid, "base_url": bad})
        assert "secret" not in str(info.value)


def test_create_model_does_not_read_env_or_touch_network() -> None:
    """构造只校验配置：不需要密钥、不建连接。"""
    model = create_model("role:lin", _options_for_unused_host())
    assert model.scope == "role:lin"
    assert model.model == "test-model"


def _options_for_unused_host() -> dict[str, Any]:
    return {
        "base_url": "http://127.0.0.1:9/v1",
        "model": "test-model",
        "api_key_env": _KEY_ENV,
    }


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/", "/gateway/v1"])
def test_request_body_headers_and_url_join(monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        model = create_model("role:lin", _options(server, base_url=f"{server.base_url}{suffix}"))
        assert model.sample_text("你好，世界") == "回：你好，世界"
        assert len(server.requests) == 1
        request = server.requests[0]
        expected_path = f"{suffix.rstrip('/')}/chat/completions"
        assert request["path"] == expected_path
        assert request["headers"]["authorization"] == f"Bearer {_KEY}"
        body = request["body"]
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "你好，世界"}]
        assert body["stream"] is False
        assert body["n"] == 1
        assert body["max_tokens"] == 5000  # Concordia 默认值
        assert body["temperature"] == 1.0
        assert body["top_p"] == 0.95, "默认 top_p 必须明确发送，不静默丢弃"
        assert "api_key" not in body and "authorization" not in body


def test_factory_caches_per_scope_and_requests_share_no_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        factory = build_live_factory(_FACTORY, _options(server))
        assert server.requests == [], "工厂构造不联网"
        lin = factory("role:lin")
        assert factory("role:lin") is lin
        shen = factory("role:shen")
        assert shen is not lin

        async def scenario() -> None:
            assert await lin.sample("甲", component="system1") == "回：甲"
            assert await shen.sample("乙", component="system1") == "回：乙"

        asyncio.run(scenario())
        assert [item["body"]["messages"] for item in server.requests] == [
            [{"role": "user", "content": "甲"}],
            [{"role": "user", "content": "乙"}],
        ]


def test_options_override_defaults_and_extra_body_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        model = create_model(
            "role:lin",
            _options(
                server,
                max_tokens=321,
                temperature=0.25,
                extra_body={"thinking": {"type": "enabled"}},
            ),
        )
        model.sample_text("提示")
        body = server.requests[0]["body"]
        assert body["max_tokens"] == 321
        assert body["temperature"] == 0.25
        assert body["thinking"] == {"type": "enabled"}


def test_call_parameters_override_when_unset_and_conflict_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        free = create_model("role:lin", _options(server))
        free.sample_text("p", max_tokens=42, temperature=0.5, timeout=12.5)
        body = server.requests[-1]["body"]
        assert body["max_tokens"] == 42 and body["temperature"] == 0.5

        fixed = create_model("role:lin", _options(server, max_tokens=321, temperature=0.25))
        fixed.sample_text("p", max_tokens=321, temperature=0.25)
        assert server.requests[-1]["body"]["max_tokens"] == 321

        before = len(server.requests)
        with pytest.raises(ParameterConflictError, match="max_tokens"):
            fixed.sample_text("p", max_tokens=99)
        with pytest.raises(ParameterConflictError, match="temperature"):
            fixed.sample_text("p", temperature=0.9)
        with pytest.raises(ParameterConflictError, match="timeout"):
            create_model("role:lin", _options(server, timeout_seconds=30)).sample_text(
                "p", timeout=5
            )
        assert len(server.requests) == before, "冲突不发出请求"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"top_p": 0.5}, UnsupportedParameterError),
        ({"top_p": True}, UnsupportedParameterError),
        ({"top_k": 1}, UnsupportedParameterError),
        ({"top_k": True}, UnsupportedParameterError),
        ({"seed": 7}, UnsupportedParameterError),
        ({"terminators": ("。",)}, UnsupportedParameterError),
        ({"max_tokens": 0}, UnsupportedParameterError),
        ({"max_tokens": True}, UnsupportedParameterError),
        ({"temperature": 3.0}, UnsupportedParameterError),
        ({"temperature": True}, UnsupportedParameterError),
        ({"timeout": -1}, UnsupportedParameterError),
        ({"timeout": float("nan")}, UnsupportedParameterError),
    ],
)
def test_unsupported_or_invalid_call_parameters_do_not_send_requests(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(error):
            model.sample_text("p", **kwargs)
        assert server.requests == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_status_errors_are_mapped_without_leaking_upstream_body(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    leak = "SECRET-UPSTREAM-BODY"

    def responder(body: dict[str, Any]) -> _Reply:
        payload = json.dumps({"error": {"message": leak}}).encode()
        return status, {"Content-Type": "application/json"}, payload

    with _server(responder) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ModelStatusError) as info:
            model.sample_text(f"含密钥的提示-{_KEY}")
        assert info.value.status_code == status
        assert info.value.category == "status"
        assert leak not in str(info.value)
        assert _KEY not in str(info.value)
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__ is True


_TOOL_CALLS = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
_BROKEN_JSON = (200, {"Content-Type": "application/json"}, b"{broken")
_NON_JSON = (200, {"Content-Type": "text/plain"}, b"not json")


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (_NON_JSON, "not_json"),
        (_BROKEN_JSON, "not_json"),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _chat_payload("部分", finish_reason="length"),
            ),
            "finish_reason",
        ),
        ((200, {"Content-Type": "application/json"}, _chat_payload("")), "empty_content"),
        ((200, {"Content-Type": "application/json"}, _chat_payload(None)), "empty_content"),
        ((200, {"Content-Type": "application/json"}, _chat_payload("   ")), "empty_content"),
        (
            (200, {"Content-Type": "application/json"}, _chat_payload("x", choice_count=0)),
            "choice_count",
        ),
        (
            (200, {"Content-Type": "application/json"}, _chat_payload("x", choice_count=2)),
            "choice_count",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _chat_payload("x", finish_reason="tool_calls"),
            ),
            "finish_reason",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _chat_payload(None, tool_calls=_TOOL_CALLS),
            ),
            "tool_calls",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _completion_payload(
                    [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "user", "content": "x"},
                        }
                    ]
                ),
            ),
            "message_role",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _completion_payload(
                    [{"index": 0, "finish_reason": "stop", "message": {"content": "x"}}]
                ),
            ),
            "message_role",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _completion_payload([{"index": 0, "finish_reason": "stop"}]),
            ),
            "message_shape",
        ),
        (
            (
                200,
                {"Content-Type": "application/json"},
                _completion_payload(
                    [{"index": 0, "message": {"role": "assistant", "content": "x"}}]
                ),
            ),
            "finish_reason",
        ),
        ((200, {"Content-Type": "application/json"}, _completion_payload([])), "choice_count"),
        ((200, {"Content-Type": "application/json"}, b"{}"), "choice_count"),
    ],
)
def test_bad_responses_are_rejected(
    monkeypatch: pytest.MonkeyPatch, reply: _Reply, reason: str
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(lambda body: reply) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ModelResponseError) as info:
            model.sample_text("提示")
        assert info.value.reason == reason
        assert info.value.category == "response_invalid"


def test_bad_response_errors_and_component_logs_do_not_leak_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """choices 数量异常与任意 finish_reason 不得把上游字段值带入异常/日志。"""
    monkeypatch.setenv(_KEY_ENV, _KEY)
    secret = "UPSTREAM-SECRET-BODY"
    replies = [
        (200, {"Content-Type": "application/json"}, _chat_payload(secret, choice_count=2)),
        (200, {"Content-Type": "application/json"}, _chat_payload("正常", finish_reason=secret)),
    ]
    for reply in replies:
        with _server(lambda body, reply=reply: reply) as server:
            sampler = LanguageModelSampler(create_model("role:lin", _options(server)))
            logged: list[dict[str, Any]] = []
            sampler.bind_channel_for(lambda component, sink=logged: sink.append)
            with pytest.raises(ModelResponseError) as info:
                asyncio.run(sampler.sample("提示", component="system1"))
            assert secret not in str(info.value)
            assert secret not in "".join(traceback.format_exception(info.value))
            assert _KEY not in str(info.value)
            assert _KEY not in "".join(traceback.format_exception(info.value))
            assert logged and logged[-1]["Key"] == "sample_text_error"
            assert secret not in json.dumps(logged, ensure_ascii=False)
            assert _KEY not in json.dumps(logged, ensure_ascii=False)


def test_no_retry_on_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(lambda body: (500, {"Content-Type": "application/json"}, b"{}")) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ModelStatusError):
            model.sample_text("提示")
        assert len(server.requests) == 1, "max_retries=0，不自动重试"


@pytest.mark.parametrize("request_fails", [False, True])
def test_client_close_failure_is_reported_without_masking_request(
    monkeypatch: pytest.MonkeyPatch, request_fails: bool
) -> None:
    """真实 HTTP 请求后模拟关闭异常：成功不吞错，失败保留原错误且不泄漏正文。"""
    from openai import OpenAI

    monkeypatch.setenv(_KEY_ENV, _KEY)
    original_close = OpenAI.close

    def broken_close(client: OpenAI) -> None:
        original_close(client)
        raise RuntimeError("CLOSE-SECRET-BODY")

    monkeypatch.setattr(OpenAI, "close", broken_close)
    reply = (500, {"Content-Type": "application/json"}, b"{}") if request_fails else _ok("成功")
    with _server(lambda body: reply) as server:
        with pytest.raises(ChatCompletionsError) as info:
            create_model("role:lin", _options(server)).sample_text("提示")
        if request_fails:
            assert isinstance(info.value, ModelStatusError)
            assert info.value.status_code == 500
            assert any("客户端关闭失败" in note for note in info.value.__notes__)
        else:
            assert info.value.category == "cleanup"
        formatted = "".join(traceback.format_exception(info.value))
        assert "CLOSE-SECRET-BODY" not in formatted
        assert _KEY not in formatted
        assert len(server.requests) == 1


def test_redirect_is_refused_without_forwarding_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(lambda body: (302, {"Location": "/stolen"}, b"")) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ModelStatusError) as info:
            model.sample_text("提示")
        assert info.value.status_code == 302
        assert [item["method"] for item in server.requests] == ["POST"]
        assert [item["path"] for item in server.requests] == ["/v1/chat/completions"]


def test_timeout_maps_to_timeout_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)

    def responder(body: dict[str, Any]) -> _Reply:
        time.sleep(0.5)
        return _ok("迟到的回复")

    with _server(responder) as server:
        model = create_model("role:lin", _options(server, timeout_seconds=0.1))
        started = time.monotonic()
        with pytest.raises(ModelRequestTimeoutError) as info:
            model.sample_text("提示")
        assert time.monotonic() - started < 0.5
        assert info.value.category == "timeout"
        assert isinstance(info.value, TimeoutError)
        assert info.value.__cause__ is None


def test_connection_failure_maps_to_connection_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    model = create_model(
        "role:lin",
        {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model": "test-model",
            "api_key_env": _KEY_ENV,
        },
    )
    with pytest.raises(ModelConnectionError) as info:
        model.sample_text("提示")
    assert info.value.category == "connection"


def test_missing_api_key_env_is_rejected_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_KEY_ENV, raising=False)
    with _server(_echo) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ModelCredentialsError, match="未设置或为空"):
            model.sample_text("提示")
        assert server.requests == []


def test_global_openai_env_does_not_override_explicit_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "global-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "global-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "global-project")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "global-admin")
    with _server(_echo) as server:
        model = create_model("role:lin", _options(server))
        assert model.sample_text("环境测试") == "回：环境测试"
        request = server.requests[0]
        assert request["headers"]["authorization"] == f"Bearer {_KEY}"
        assert request["headers"].get("openai-organization", "") == ""
        assert request["headers"].get("openai-project", "") == ""
        serialized = json.dumps(request["headers"], ensure_ascii=False)
        assert "global-openai-key" not in serialized
        assert "global-admin" not in serialized


def test_global_custom_headers_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer injected")
    with _server(_echo) as server:
        model = create_model("role:lin", _options(server))
        with pytest.raises(ChatCompletionsConfigError, match="OPENAI_CUSTOM_HEADERS"):
            model.sample_text("提示")
        assert server.requests == []


def test_sample_choice_is_not_implemented() -> None:
    model = create_model("role:lin", _options_for_unused_host())
    with pytest.raises(NotImplementedError, match="sample_choice"):
        model.sample_choice("提示", ["甲", "乙"])
    assert isinstance(model, ChatCompletionsLanguageModel)


def test_cli_builtin_factory_config_error_is_clean_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """内置工厂配置失败应经 CLI 已有错误处理：非 0 退出、无 Traceback、不联网。"""
    from sceneweave.__main__ import main

    config = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run.json"
    monkeypatch.chdir(tmp_path)  # live 入口只能读取临时目录，避免加载项目真实 .env。
    monkeypatch.setenv(_KEY_ENV, _KEY)
    with _server(_echo) as server:
        code = main(
            [
                str(config),
                "--live",
                _FACTORY,
                "--live-options",
                json.dumps({"base_url": server.base_url}),
                "--output",
                str(tmp_path / "cli-out"),
            ]
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "缺少必填字段" in captured.err
        assert "Traceback" not in captured.err
        assert server.requests == [], "配置校验失败不得发起网络请求"


# ---- 完整装配 + runner + 组件日志 + 录制/重放 ----


_S1_GO = '{"route": "rethink", "note": "门被推开了"}'
_S1_SKIP = '{"route": "continue", "note": "照旧"}'
_S2 = '{"intention": "查看抽屉里的东西"}'
_S3 = '[{"kind": "move", "destination": {"entity": "drawer"}}]'
_TRANSLATION = (
    '{"schedules": [{"action_id": "a-lin-1",'
    ' "start": 1000, "expected_end": 4000, "checkpoints": [2500]}]}'
)


class _FullStackResponder:
    """按提示语义分发的假端点：世界用测试假件，角色/翻译用固定脚本。"""

    def __init__(self) -> None:
        self._world = Brain()
        self._system1_calls = 0

    def __call__(self, body: dict[str, Any]) -> _Reply:
        messages = body["messages"]
        prompt = messages[0]["content"]
        if "你是场景 World 的客观检查者" in prompt:
            answer = self._world._check(prompt)  # noqa: SLF001 测试侧假件
        elif "你是场景 World 的进展裁决者" in prompt:
            answer = self._world._progress(prompt)  # noqa: SLF001
        elif "你是场景 World 的联合裁决者" in prompt:
            answer = self._world._adjudicate(prompt)  # noqa: SLF001
        elif "你是行动翻译器" in prompt:
            answer = _TRANSLATION
        elif "你是角色 shen" in prompt:
            answer = _S1_SKIP
        elif "你是角色 lin" in prompt and "第一步" in prompt:
            self._system1_calls += 1
            answer = _S1_GO if self._system1_calls == 1 else _S1_SKIP
        elif "你是角色 lin" in prompt and "第二步" in prompt:
            answer = _S2
        elif "你是角色 lin" in prompt and "第三步" in prompt:
            answer = _S3
        else:
            raise AssertionError(f"假端点未识别的提示：{prompt[:60]!r}")
        return _ok(answer)


def _write_full_config(tmp_path: Path) -> Path:
    (tmp_path / "scene.json").write_text(
        json.dumps(study_card(), ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "scene_path": "scene.json",
        "roles": [
            {"id": "lin", "background": "测试角色林"},
            {"id": "shen", "background": "测试角色沈"},
        ],
        "initial_events": [{"id": "door", "time_seconds": 0, "description": "门被推开"}],
        "run": {"introspection_interval_seconds": 6, "max_simulation_duration_seconds": 6},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def _component_raw(log_text: str) -> list[dict[str, Any]]:
    from concordia.utils import structured_logging

    log = structured_logging.SimulationLog.from_json(log_text)
    return [
        dict(log.reconstruct_value(entry.deduplicated_data))
        for entry in log.entries
        if entry.entry_type == "component"
    ]


def test_full_stack_record_then_replay_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_KEY_ENV, _KEY)
    config = load_assembly_config(_write_full_config(tmp_path))
    responder = _FullStackResponder()
    server = _FakeChatServer(responder)
    with server:
        factory = build_live_factory(_FACTORY, _options(server, timeout_seconds=30))
        assert server.requests == []

        async def record_run() -> tuple[Any, Path]:
            return await run_headless(
                config,
                factory,
                output_directory=tmp_path / "record-out",
                record_directory=tmp_path / "record",
            )

        summary, out_dir = asyncio.run(record_run())
        assert summary.stop_reason == "time_limit", summary.error
        recorded_requests = len(server.requests)
        assert recorded_requests > 0
        record_text = (out_dir / "simulation_log.json").read_text(encoding="utf-8")
        record_raw = _component_raw(record_text)
        assert any("你是角色" in str(item.get("Prompt", "")) for item in record_raw), (
            "组件日志保存 Prompt 原文"
        )
        assert any(str(item.get("Response", "")) == _S1_GO for item in record_raw), (
            "组件日志保存 Response 原文"
        )
        assert any(str(item.get("scope", "")).startswith("role:") for item in record_raw)
        assert _KEY not in record_text, "密钥不进入组件日志"
        assert (tmp_path / "record" / "manifest.json").is_file()
    # 服务器已关闭：重放若联网会立即连接失败。
    with socket.socket() as probe:
        probe.settimeout(0.5)
        assert probe.connect_ex(("127.0.0.1", server.port)) != 0

    replay = RecordedRun(tmp_path / "record", config)

    async def replay_run() -> tuple[Any, Path]:
        return await run_headless(
            config,
            replay.for_scope,
            replay_run=replay,
            output_directory=tmp_path / "replay-out",
        )

    replay_summary, replay_out = asyncio.run(replay_run())
    assert replay_summary.stop_reason == summary.stop_reason
    assert replay_summary.final_time_ms == summary.final_time_ms
    assert replay_summary.completed_cycles == summary.completed_cycles
    assert replay_summary.world_state_version == summary.world_state_version
    replay_text = (replay_out / "simulation_log.json").read_text(encoding="utf-8")
    replay_raw = _component_raw(replay_text)
    assert any("你是角色" in str(item.get("Prompt", "")) for item in replay_raw)
    assert any(item.get("Replay") is True for item in replay_raw), "重放仍写组件日志"
    assert _KEY not in replay_text
    assert len(server.requests) == recorded_requests, "重放不产生网络请求"
