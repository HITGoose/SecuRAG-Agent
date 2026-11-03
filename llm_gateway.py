"""
统一的模型调用入口。
- 阶段1：按 provider 建客户端、调用、整理返回格式、记录 token 和耗时。
- 阶段5：重试 + DeepSeek → 本地 Ollama 降级 + 每次调用的记录（给 trace 用）+ 测试用的故障注入。

重试规则：超时、连接失败、429、5xx 最多再试 2 次（指数退避）；认证失败（401/403）不重试，直接降级；
参数错误等其他 4xx 不重试也不降级（换个模型也一样错）。只从云端降到本地，本地模式不会被降级到云端。
"""
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, AuthenticationError, InternalServerError, OpenAI,
                    PermissionDeniedError, RateLimitError)

import config

load_dotenv()

RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)  # 临时故障：值得再试
FALLBACK_ON = RETRYABLE + (AuthenticationError, PermissionDeniedError)                  # 这些失败才降级


@dataclass
class LLMResult:
    message: dict      # OpenAI 格式的 assistant 消息：{"role", "content", "tool_calls"?}，可以直接追加进 messages
    input_tokens: int  # 接口返回的 usage；拿不到时为 None，不估算
    output_tokens: int
    latency_ms: int
    provider: str      # 实际完成这次调用的后端（降级后是 ollama）
    model: str
    retries: int = 0
    fallback_reason: str = None


class _Backend:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "deepseek":
            self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=config.BASE_URL, timeout=60,
                                 max_retries=0)  # 重试由网关自己做，关掉 SDK 自带的，免得两层叠加
            self.model = config.MODEL_NAME
        elif provider == "ollama":
            # 不用 OLLAMA_HOST：它是 Ollama 服务端的监听地址（本机 .env 里是 0.0.0.0），不是客户端能连的 URL
            host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # trust_env=False：不走系统代理，否则本地地址可能被代理劫持
            self.client = OpenAI(base_url=f"{host}/v1", api_key="ollama", timeout=120, max_retries=0,
                                 http_client=httpx.Client(trust_env=False))
            self.model = config.LOCAL_MODEL_NAME
        else:
            raise ValueError(f"未知的 provider：{provider}")


class LLMGateway:
    def __init__(self, provider: str = "deepseek", fallback: bool = True, max_retries: int = 2,
                 backoff_s: float = 0.5, fault=None):
        self.provider = provider
        self.primary = _Backend(provider)
        self.backup = _Backend("ollama") if provider == "deepseek" and fallback else None
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.fault = fault  # 测试用：fault(provider) 在每次真正请求前调用，可以抛出模拟的异常
        self.log = []       # 调用记录；ShopAgent 每个请求开始时清空，结束时写进 trace

    def chat(self, messages: list, tools: list = None, temperature: float = config.SHOP_TEMPERATURE,
             max_tokens: int = 800, purpose: str = "loop") -> LLMResult:
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        try:
            result = self._with_retry(self.primary, kwargs)
        except FALLBACK_ON as e:
            if self.backup is None:
                self._log_failure(purpose, self.primary, e)
                raise
            # 云端重试后仍失败（或认证失败）：换本地模型。已经执行过的工具不会重做，只重做这一次模型调用
            try:
                result = self._with_retry(self.backup, kwargs)
            except Exception as e2:
                self._log_failure(purpose, self.backup, e2, fallback_reason=type(e).__name__)
                raise
            result.fallback_reason = type(e).__name__
        except Exception as e:
            self._log_failure(purpose, self.primary, e)
            raise
        self.log.append({"purpose": purpose, "provider": result.provider, "model": result.model,
                         "latency_ms": result.latency_ms, "input_tokens": result.input_tokens,
                         "output_tokens": result.output_tokens, "retries": result.retries,
                         "fallback_reason": result.fallback_reason, "error": None})
        return result

    def _with_retry(self, backend: _Backend, kwargs: dict) -> LLMResult:
        for attempt in range(self.max_retries + 1):
            try:
                result = self._call(backend, kwargs)
                result.retries = attempt
                return result
            except RETRYABLE:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_s * (2 ** attempt))  # 0.5s、1s ……

    def _call(self, backend: _Backend, kwargs: dict) -> LLMResult:
        if self.fault:
            self.fault(backend.provider)
        start = time.perf_counter()
        resp = backend.client.chat.completions.create(model=backend.model, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        msg = resp.choices[0].message
        message = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            message["tool_calls"] = [{"id": tc.id, "type": "function",
                                      "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                     for tc in msg.tool_calls]
        usage = resp.usage
        return LLMResult(message=message,
                         input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                         output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                         latency_ms=latency_ms, provider=backend.provider, model=backend.model)

    def _log_failure(self, purpose, backend, error, fallback_reason=None) -> None:
        self.log.append({"purpose": purpose, "provider": backend.provider, "model": backend.model, "latency_ms": None,
                         "input_tokens": None, "output_tokens": None, "retries": None,
                         "fallback_reason": fallback_reason, "error": type(error).__name__})
