"""
统一的模型调用入口。阶段1：只负责按 provider 建客户端、调用、整理返回格式、记录 token 和耗时。
阶段5 在这里加重试和 DeepSeek → Ollama 降级，调用方不用改。
"""
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()


@dataclass
class LLMResult:
    message: dict      # OpenAI 格式的 assistant 消息：{"role", "content", "tool_calls"?}，可以直接追加进 messages
    input_tokens: int  # 接口返回的 usage；拿不到时为 None，不估算
    output_tokens: int
    latency_ms: int
    provider: str
    model: str


class LLMGateway:
    def __init__(self, provider: str = "deepseek"):
        self.provider = provider
        if provider == "deepseek":
            self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=config.BASE_URL, timeout=60)
            self.model = config.MODEL_NAME
        elif provider == "ollama":
            # 不用 OLLAMA_HOST：它是 Ollama 服务端的监听地址（本机 .env 里是 0.0.0.0），不是客户端能连的 URL
            host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # trust_env=False：不走系统代理，否则本地地址可能被代理劫持
            self.client = OpenAI(base_url=f"{host}/v1", api_key="ollama", timeout=120,
                                 http_client=httpx.Client(trust_env=False))
            self.model = config.LOCAL_MODEL_NAME
        else:
            raise ValueError(f"未知的 provider：{provider}")

    def chat(self, messages: list, tools: list = None, temperature: float = config.SHOP_TEMPERATURE,
             max_tokens: int = 800) -> LLMResult:
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        start = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
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
                         latency_ms=latency_ms, provider=self.provider, model=self.model)
