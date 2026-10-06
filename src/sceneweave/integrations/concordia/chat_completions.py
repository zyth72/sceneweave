"""OpenAI 兼容 Chat Completions 的真实 Concordia LanguageModel 工厂。

协议编码、传输和响应反序列化由锁定版本的 openai SDK 完成（仅
``chat.completions.create`` 非流式调用）；本模块只做工厂选项与采样
参数校验、明确的响应终止判定和安全异常映射，不实现第二套 HTTP
客户端，也不叠加代理或供应商路由。作为统一入口的 ``--live`` 内置
后端使用：

    --live sceneweave.integrations.concordia.chat_completions:create_model \\
    --live-options '{"base_url": "...", "model": "...", "api_key_env": "..."}'

工厂选项：
- 必填 ``base_url``（仅 http/https，禁止用户信息、query、fragment，
  SDK 负责在根路径或 ``/v1`` 末尾斜杠后追加 ``/chat/completions``）、
  ``model``（显式模型名，不自动选择供应商或默认模型）、
  ``api_key_env``（环境变量名；不接受明文密钥 options）。
- 可选 ``timeout_seconds``、``max_tokens``、``temperature``；这三个字段
  要么省略使用默认，要么给有效值，显式 JSON ``null`` 明确拒绝。供应商
  专用选项只能放进严格受控的 ``extra_body``，且不得覆盖适配器设置或
  保留的字段（model/messages/stream/n/stop/seed/max_tokens/temperature/
  top_p/top_k/timeout/认证及工具调用字段）。

采样参数策略：``temperature``/``max_tokens`` 取工厂配置或调用参数
（同时显式给出且不一致时拒绝）；``top_p`` 固定发送 Concordia 默认值
0.95；非默认 ``top_p``、``top_k``、``terminators``、``seed`` 当前不
支持并显式拒绝，不静默忽略。

运行语义：
- 导入无网络与环境凭据读取；``create_model`` 只校验配置并构造对象。
- 每个 ``sample_text`` 新建并关闭独立 ``OpenAI`` 客户端，``max_retries=0``，
  ``follow_redirects=False``（禁止把密钥转发到其它 host）；只发送单条
  user 消息，不共享对话历史，scope 只标识实例；close 失败明确报告，
  已有请求错误时只附加安全说明，保留原始错误。
- 只接受 ``finish_reason == "stop"``、``message.role == "assistant"``
  且 content 为非空字符串的单个 assistant 文本响应；截断、工具调用、
  空内容、choice 数量异常、非 JSON 响应都是明确错误，坏输出不会被
  当作成功文本。
- 超时/连接/HTTP 状态/响应无效按类别映射为项目异常；异常消息不含
  密钥、Authorization、上游正文或提示原文，原始异常链以 ``from None``
  切断。SDK 自身的 ``OPENAI_*`` 环境读取被显式空值选项压制；
  若是 ``OPENAI_CUSTOM_HEADERS`` 已设置则明确拒绝，避免全局请求头
  静默注入。
- ``timeout_seconds`` 是 SDK/HTTP 客户端的连接与读写等待超时，不是
  整次调用的绝对墙钟 deadline；本适配器不声称实现调用总预算。
- ``sample_choice`` 不在本项目运行链中，显式抛 ``NotImplementedError``，
  不伪造概率或默认选项。
"""

import json
import math
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit

from concordia.language_model import language_model
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    DefaultHttpxClient,
    OpenAI,
    OpenAIError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice

__all__ = [
    "ChatCompletionsConfigError",
    "ChatCompletionsError",
    "ChatCompletionsLanguageModel",
    "ModelConnectionError",
    "ModelCredentialsError",
    "ModelRequestTimeoutError",
    "ModelResponseError",
    "ModelStatusError",
    "ParameterConflictError",
    "UnsupportedParameterError",
    "create_model",
]

_ALLOWED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "base_url",
        "model",
        "api_key_env",
        "timeout_seconds",
        "max_tokens",
        "temperature",
        "extra_body",
    }
)
_REQUIRED_OPTION_KEYS: Final[tuple[str, ...]] = ("base_url", "model", "api_key_env")
# 适配器自己设置或保留的字段；extra_body 覆盖它们会破坏项目语义或认证。
_PROTECTED_EXTRA_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "n",
        "stop",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "timeout",
        "api_key",
        "authorization",
        "base_url",
        "tools",
        "tool_choice",
        "function_call",
        "functions",
        "parallel_tool_calls",
    }
)
_API_KEY_ENV_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CUSTOM_HEADERS_ENV: Final = "OPENAI_CUSTOM_HEADERS"
_MISSING: Final = object()


class ChatCompletionsError(RuntimeError):
    """Chat Completions 适配器的可诊断错误基类。"""

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


class ChatCompletionsConfigError(ChatCompletionsError, ValueError):
    """工厂 options 或采样参数配置错误（消息不含密钥）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="config")


class UnsupportedParameterError(ChatCompletionsConfigError):
    """调用参数非默认且适配器不支持；不静默忽略。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.category = "unsupported_parameter"


class ParameterConflictError(ChatCompletionsConfigError):
    """工厂 options 与调用参数同时给出且不一致。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.category = "parameter_conflict"


class ModelCredentialsError(ChatCompletionsError):
    """指定环境变量没有提供 API key；不读 .env、全局凭据或默认 key。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="credentials")


class ModelRequestTimeoutError(ChatCompletionsError, TimeoutError):
    """连接或读写等待超时；不是整次调用的绝对墙钟 deadline。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="timeout")


class ModelConnectionError(ChatCompletionsError):
    """无法建立连接或传输失败。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="connection")


class ModelStatusError(ChatCompletionsError):
    """上游返回非 2xx 状态；只保留状态码与类别，不保留正文。"""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.status_kind = _status_kind(status_code)
        super().__init__(
            f"模型服务返回 HTTP {status_code}（{self.status_kind}）",
            category="status",
        )


class ModelResponseError(ChatCompletionsError):
    """响应结构或终止状态无效；reason 是稳定子类别。"""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message, category="response_invalid")


def _status_kind(status_code: int) -> str:
    if 300 <= status_code < 400:
        return "重定向被拒绝"
    if status_code == 400:
        return "请求无效"
    if status_code == 401:
        return "认证失败"
    if status_code == 403:
        return "无权限"
    if status_code == 404:
        return "接口不存在"
    if status_code == 408:
        return "请求超时"
    if status_code == 409:
        return "请求冲突"
    if status_code == 422:
        return "请求无法处理"
    if status_code == 429:
        return "触发限流"
    if status_code >= 500:
        return "上游服务错误"
    return "客户端错误"


@dataclass(frozen=True, slots=True)
class _Config:
    """已校验的工厂配置；不持有密钥对象。"""

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float | None
    max_tokens: int | None
    temperature: float | None
    extra_body: Mapping[str, Any]


def _validated_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChatCompletionsConfigError(f"{name} 必须是非空字符串，得到：{value!r}")
    return value


def _validated_base_url(value: object) -> str:
    """校验 base_url；错误不回报用户名、密码、query 或原始串。"""
    if not isinstance(value, str) or not value.strip():
        raise ChatCompletionsConfigError("base_url 必须是非空字符串")
    base_url = value
    if base_url != base_url.strip():
        raise ChatCompletionsConfigError("base_url 不得包含首尾空白")
    if any(ch <= " " or ch == "\x7f" or ch == "\\" for ch in base_url):
        raise ChatCompletionsConfigError("base_url 不得包含空白、控制字符或反斜杠")
    try:
        parsed = urlsplit(base_url)
        _ = parsed.port  # 非法端口/非法 IPv6 在这里明确失败
    except ValueError:
        raise ChatCompletionsConfigError("base_url 不是合法的 URL（主机或端口无效）") from None
    if parsed.scheme not in ("http", "https"):
        raise ChatCompletionsConfigError("base_url 仅支持 http/https")
    if not parsed.netloc or not parsed.hostname:
        raise ChatCompletionsConfigError("base_url 缺少有效主机")
    if parsed.username is not None or parsed.password is not None:
        raise ChatCompletionsConfigError("base_url 不得包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise ChatCompletionsConfigError("base_url 不得包含 query 或 fragment")
    return base_url


def _validated_api_key_env(value: object) -> str:
    name = _validated_str("api_key_env", value)
    if _API_KEY_ENV_PATTERN.fullmatch(name) is None:
        raise ChatCompletionsConfigError(
            f"api_key_env 必须是合法环境变量名（字母/下划线开头，仅字母数字下划线）：{name!r}"
        )
    return name


def _validated_timeout(value: object) -> float | None:
    if value is _MISSING:
        return None
    if value is None:
        raise ChatCompletionsConfigError("timeout_seconds 显式 null 无效；请省略或给正的有限数值")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatCompletionsConfigError(f"timeout_seconds 必须是正数，得到：{value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ChatCompletionsConfigError(f"timeout_seconds 必须是正的有限数值，得到：{value!r}")
    return number


def _validated_max_tokens(value: object) -> int | None:
    if value is _MISSING:
        return None
    if value is None:
        raise ChatCompletionsConfigError("max_tokens 显式 null 无效；请省略或给正整数")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChatCompletionsConfigError(f"max_tokens 必须是正整数，得到：{value!r}")
    return value


def _validated_temperature(value: object) -> float | None:
    if value is _MISSING:
        return None
    if value is None:
        raise ChatCompletionsConfigError("temperature 显式 null 无效；请省略或给 [0, 2] 内的数值")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatCompletionsConfigError(f"temperature 必须是 [0, 2] 内的数值，得到：{value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 2.0:
        raise ChatCompletionsConfigError(f"temperature 必须是 [0, 2] 内的有限数值，得到：{value!r}")
    return number


def _validated_extra_body(value: object) -> Mapping[str, Any]:
    if value is _MISSING:
        return {}
    if value is None:
        raise ChatCompletionsConfigError("extra_body 显式 null 无效；请省略或给 JSON 对象")
    if not isinstance(value, Mapping):
        raise ChatCompletionsConfigError(f"extra_body 必须是 JSON 对象，得到：{value!r}")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ChatCompletionsConfigError("extra_body 的键必须是非空字符串")
        if key.lower() in _PROTECTED_EXTRA_BODY_KEYS:
            raise ChatCompletionsConfigError(f"extra_body 不得覆盖适配器拥有的字段：{key!r}")
        result[key] = item
    try:
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ChatCompletionsConfigError(
            "extra_body 必须是 JSON 可序列化对象（不允许 NaN/Infinity 或任意对象）"
        ) from error
    copied = json.loads(serialized)
    if not isinstance(copied, dict):
        raise ChatCompletionsConfigError("extra_body 必须是 JSON 对象")
    return copied


def _parse_options(options: Mapping[str, Any]) -> _Config:
    if not isinstance(options, Mapping):
        raise ChatCompletionsConfigError(f"options 必须是映射对象，得到：{type(options).__name__}")
    for key in options:
        if not isinstance(key, str):
            raise ChatCompletionsConfigError(f"options 键必须是字符串，得到：{type(key).__name__}")
    unknown = sorted(set(options) - _ALLOWED_OPTION_KEYS)
    if unknown:
        raise ChatCompletionsConfigError(
            f"options 含未知字段 {unknown}；允许字段：{sorted(_ALLOWED_OPTION_KEYS)}"
        )
    missing = [key for key in _REQUIRED_OPTION_KEYS if key not in options]
    if missing:
        raise ChatCompletionsConfigError(f"options 缺少必填字段：{missing}")
    return _Config(
        base_url=_validated_base_url(options["base_url"]),
        model=_validated_str("model", options["model"]),
        api_key_env=_validated_api_key_env(options["api_key_env"]),
        timeout_seconds=_validated_timeout(options.get("timeout_seconds", _MISSING)),
        max_tokens=_validated_max_tokens(options.get("max_tokens", _MISSING)),
        temperature=_validated_temperature(options.get("temperature", _MISSING)),
        extra_body=_validated_extra_body(options.get("extra_body", _MISSING)),
    )


def _reject_non_default(name: str, value: object, default: object) -> None:
    if value != default:
        raise UnsupportedParameterError(
            f"{name} 非默认值 {value!r}：本适配器只支持项目默认采样策略下的调用参数"
        )


def _resolve_max_tokens(call_value: int, option_value: int | None) -> int:
    if isinstance(call_value, bool) or not isinstance(call_value, int) or call_value <= 0:
        raise UnsupportedParameterError(f"max_tokens 必须是正整数，得到：{call_value!r}")
    if option_value is None:
        return call_value
    if call_value != language_model.DEFAULT_MAX_TOKENS and call_value != option_value:
        raise ParameterConflictError(
            f"max_tokens 同时给出 options={option_value} 与调用参数={call_value} 且不一致"
        )
    return option_value


def _resolve_temperature(call_value: float, option_value: float | None) -> float:
    if isinstance(call_value, bool) or not isinstance(call_value, (int, float)):
        raise UnsupportedParameterError(f"temperature 必须是 [0, 2] 内的数值，得到：{call_value!r}")
    number = float(call_value)
    if not math.isfinite(number) or not 0.0 <= number <= 2.0:
        raise UnsupportedParameterError(
            f"temperature 必须是 [0, 2] 内的有限数值，得到：{call_value!r}"
        )
    if option_value is None:
        return number
    if number != language_model.DEFAULT_TEMPERATURE and number != option_value:
        raise ParameterConflictError(
            f"temperature 同时给出 options={option_value} 与调用参数={number} 且不一致"
        )
    return option_value


def _resolve_timeout(call_value: float, option_value: float | None) -> float:
    if isinstance(call_value, bool) or not isinstance(call_value, (int, float)):
        raise UnsupportedParameterError(f"timeout 必须是正的有限数值，得到：{call_value!r}")
    number = float(call_value)
    if not math.isfinite(number) or number <= 0:
        raise UnsupportedParameterError(f"timeout 必须是正的有限数值，得到：{call_value!r}")
    if option_value is None:
        return number
    if number != language_model.DEFAULT_TIMEOUT_SECONDS and number != option_value:
        raise ParameterConflictError(
            f"timeout 同时给出 options={option_value} 与调用参数={number} 且不一致"
        )
    return option_value


def _resolve_top_p(call_value: float) -> float:
    """只允许 Concordia 默认 top_p，并明确发送该默认值而非静默丢弃。"""
    if isinstance(call_value, bool) or not isinstance(call_value, (int, float)):
        raise UnsupportedParameterError(
            f"top_p 只支持 Concordia 默认值 {language_model.DEFAULT_TOP_P}，得到非数值"
        )
    number = float(call_value)
    if not math.isfinite(number) or number != language_model.DEFAULT_TOP_P:
        raise UnsupportedParameterError(
            f"top_p 只支持 Concordia 默认值 {language_model.DEFAULT_TOP_P}；非默认值不静默忽略"
        )
    return language_model.DEFAULT_TOP_P


def _extract_content(completion: object) -> str:
    """只接受单个 stop 结束、assistant 角色、含非空文本且无工具调用的响应。

    只检查项目实际依赖的少数字段；错误消息不包含上游字段值。
    """
    if not isinstance(completion, ChatCompletion):
        raise ModelResponseError(
            "not_json",
            "模型响应不是有效的 JSON Chat Completion（可能 Content-Type 非 application/json）",
        )
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        found = f"{len(choices)} 条" if isinstance(choices, list) else type(choices).__name__
        raise ModelResponseError(
            "choice_count", f"模型响应 choices 数量必须为 1（n=1），得到：{found}"
        )
    choice = choices[0]
    if not isinstance(choice, Choice):
        raise ModelResponseError("choice_shape", "模型响应 choice 结构无效")
    if getattr(choice, "finish_reason", None) != "stop":
        raise ModelResponseError(
            "finish_reason",
            "模型未以 stop 正常结束；拒绝截断、内容过滤或工具调用等异常结束",
        )
    message = getattr(choice, "message", None)
    if not isinstance(message, ChatCompletionMessage):
        raise ModelResponseError("message_shape", "模型响应 message 结构无效")
    if getattr(message, "role", None) != "assistant":
        raise ModelResponseError("message_role", "模型响应 message.role 不是 assistant")
    if getattr(message, "tool_calls", None) or getattr(message, "function_call", None) is not None:
        raise ModelResponseError("tool_calls", "模型返回了工具调用，本适配器只接受纯文本")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError("empty_content", "模型响应 content 为空或不是字符串")
    return content


class ChatCompletionsLanguageModel(language_model.LanguageModel):
    """锁定的 Concordia ``LanguageModel`` 子类：非流式文本采样。

    实例只保存已校验配置；每次 ``sample_text`` 独立建连、独立关闭。
    作用域只在实例标识层面保留，不会被注入请求或提示。
    """

    def __init__(self, scope: str, config: _Config) -> None:
        self._scope = scope
        self._config = config

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def model(self) -> str:
        return self._config.model

    def _api_key(self) -> str:
        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise ModelCredentialsError(
                f"环境变量 {self._config.api_key_env!r} 未设置或为空；"
                "本适配器不读取 .env、全局凭据或默认 key"
            )
        return api_key

    def _reject_global_custom_headers(self) -> None:
        # SDK 版本会无条件合并 OPENAI_CUSTOM_HEADERS；这里明确拒绝而不是静默注入。
        if os.environ.get(_CUSTOM_HEADERS_ENV):
            raise ChatCompletionsConfigError(
                f"检测到全局 {_CUSTOM_HEADERS_ENV}；本适配器要求请求头完全由显式配置决定，"
                "请清除该环境变量后重试"
            )

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ChatCompletionsConfigError("prompt 必须是非空字符串")
        _reject_non_default(
            "terminators", tuple(terminators), tuple(language_model.DEFAULT_TERMINATORS)
        )
        _reject_non_default("top_k", top_k, language_model.DEFAULT_TOP_K)
        if seed is not None:
            raise UnsupportedParameterError("seed 非 None：本适配器未提供种子策略，不静默忽略")
        effective_max_tokens = _resolve_max_tokens(max_tokens, self._config.max_tokens)
        effective_temperature = _resolve_temperature(temperature, self._config.temperature)
        effective_timeout = _resolve_timeout(timeout, self._config.timeout_seconds)
        effective_top_p = _resolve_top_p(top_p)
        api_key = self._api_key()
        self._reject_global_custom_headers()

        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        try:
            client = OpenAI(
                api_key=api_key,
                # 显式空值避免 SDK 回落到 OPENAI_ADMIN_KEY / OPENAI_ORG_ID /
                # OPENAI_PROJECT_ID / OPENAI_WEBHOOK_SECRET。
                admin_api_key="",
                organization="",
                project="",
                webhook_secret="",
                base_url=self._config.base_url,
                max_retries=0,
                timeout=effective_timeout,
                http_client=DefaultHttpxClient(follow_redirects=False),
            )
        except Exception as error:
            # 构造期配置/传输错误只报稳定类别，不回传原始异常正文。
            raise ChatCompletionsConfigError(
                f"模型客户端构造失败（{type(error).__name__}）；不包含原始异常正文"
            ) from None
        request_error: BaseException | None = None
        try:
            try:
                completion = client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    stream=False,
                    n=1,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    extra_body=self._config.extra_body or None,
                )
            except APITimeoutError:
                raise ModelRequestTimeoutError(
                    f"模型请求超时（等待 {effective_timeout:g} 秒；这是连接/读写等待超时，"
                    "不是整次调用的绝对墙钟上限）"
                ) from None
            except APIStatusError as error:
                raise ModelStatusError(error.status_code) from None
            except APIConnectionError:
                raise ModelConnectionError("模型服务连接失败（未建立连接或传输中断）") from None
            except OpenAIError as error:
                raise ChatCompletionsError(
                    f"模型请求失败（SDK 错误类型 {type(error).__name__}）",
                    category="sdk_error",
                ) from None
            except json.JSONDecodeError:
                raise ModelResponseError("not_json", "模型响应正文不是合法 JSON") from None
            return _extract_content(completion)
        except BaseException as error:
            request_error = error
            raise
        finally:
            try:
                client.close()
            except Exception as close_error:
                detail = f"模型客户端关闭失败（{type(close_error).__name__}）"
                if request_error is not None:
                    request_error.add_note(detail)
                else:
                    raise ChatCompletionsError(detail, category="cleanup") from None

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        raise NotImplementedError(
            "ChatCompletionsLanguageModel 只实现 sample_text；"
            "本项目运行链不使用 sample_choice，不伪造概率或默认选项"
        )


def create_model(scope: str, options: Mapping[str, Any]) -> language_model.LanguageModel:
    """``--live`` 工厂入口：校验显式配置并返回独立 LanguageModel 实例。

    构造阶段不读环境变量、不建连接；密钥只在 ``sample_text`` 时从
    ``api_key_env`` 指定的环境变量读取。
    """
    if not isinstance(scope, str) or not scope:
        raise ChatCompletionsConfigError(f"scope 必须是非空字符串，得到：{scope!r}")
    return ChatCompletionsLanguageModel(scope, _parse_options(options))
