"""
阶段5 单元测试：网关的重试 / 降级规则、trace 记录。用假客户端，不发网络请求。
运行：python -m unittest tests.test_stage5 -v
"""
import tempfile
import unittest
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, RateLimitError

from llm_gateway import LLMGateway
from tracing import TraceWriter, build_trace, read_traces

REQ = httpx.Request("POST", "http://x")


def error(cls, status):
    return cls("injected", response=httpx.Response(status, request=REQ), body=None)


class FakeClient:
    """按剧本依次抛错或返回；数被调用了几次。"""

    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        item = self.script.pop(0) if self.script else "ok"
        if isinstance(item, Exception):
            raise item
        msg = SimpleNamespace(content=f"回复:{item}", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


def gateway(primary_script, backup_script=None, provider="deepseek"):
    gw = LLMGateway.__new__(LLMGateway)  # 跳过 __init__，不建真实客户端
    gw.provider, gw.max_retries, gw.backoff_s, gw.fault, gw.log = provider, 2, 0, None, []
    gw.primary = SimpleNamespace(provider=provider, model="m1", client=FakeClient(primary_script))
    gw.backup = (SimpleNamespace(provider="ollama", model="m2", client=FakeClient(backup_script or []))
                 if backup_script is not None else None)
    return gw


class TestGateway(unittest.TestCase):
    def test_retry_then_success(self):
        gw = gateway([error(RateLimitError, 429), "ok"], [])
        r = gw.chat([], purpose="t")
        self.assertEqual((r.provider, r.retries, r.fallback_reason), ("deepseek", 1, None))
        self.assertEqual((gw.primary.client.calls, gw.backup.client.calls), (2, 0))

    def test_retries_exhausted_then_fallback(self):
        gw = gateway([error(InternalServerError, 503)] * 3, [])
        r = gw.chat([], purpose="t")
        self.assertEqual((r.provider, r.fallback_reason), ("ollama", "InternalServerError"))
        self.assertEqual(gw.primary.client.calls, 3)  # 1 次 + 重试 2 次
        self.assertEqual(gw.log[-1]["fallback_reason"], "InternalServerError")

    def test_timeout_fallback(self):
        gw = gateway([APITimeoutError(request=REQ)] * 3, [])
        self.assertEqual(gw.chat([]).provider, "ollama")

    def test_auth_error_falls_back_without_retry(self):
        gw = gateway([error(AuthenticationError, 401)], [])
        self.assertEqual(gw.chat([]).provider, "ollama")
        self.assertEqual(gw.primary.client.calls, 1)

    def test_bad_request_neither_retries_nor_falls_back(self):
        gw = gateway([error(BadRequestError, 400)], [])
        with self.assertRaises(BadRequestError):
            gw.chat([])
        self.assertEqual((gw.primary.client.calls, gw.backup.client.calls), (1, 0))
        self.assertEqual(gw.log[-1]["error"], "BadRequestError")

    def test_both_fail_raises(self):
        gw = gateway([error(InternalServerError, 503)] * 3, [error(InternalServerError, 503)] * 3)
        with self.assertRaises(InternalServerError):
            gw.chat([])
        self.assertEqual(gw.backup.client.calls, 3)

    def test_local_mode_never_falls_back_to_cloud(self):
        gw = gateway([error(InternalServerError, 503)] * 3, provider="ollama")
        with self.assertRaises(InternalServerError):
            gw.chat([])
        self.assertIsNone(gw.backup)


class TestTracing(unittest.TestCase):
    def test_write_and_read(self):
        out = {"route": "agent", "intent": "LOGISTICS", "stop_reason": "answered", "fact_check": "passed",
               "latency_ms": 10, "input_tokens": 5, "output_tokens": 2, "tokens_complete": True,
               "steps": [{"step": 1, "tool": "track_shipment", "args": {"order_id": "ORD-DEMO-0009"},
                          "status": "ok", "error_code": None, "via": "mcp", "latency_ms": 4}],
               "answer": "运输中"}
        llm = [{"purpose": "loop", "provider": "ollama", "fallback_reason": "InternalServerError"}]
        with tempfile.TemporaryDirectory() as tmp:
            TraceWriter(tmp).write(build_trace("abc", "session-1", "USR-DEMO-01", "到哪了", out, llm))
            t = read_traces(tmp)[0]
        self.assertEqual((t["trace_id"], t["fallback_used"], t["tools"][0]["via"]), ("abc", True, "mcp"))
        self.assertNotEqual(t["session"], "session-1")  # 会话 id 做了哈希


if __name__ == "__main__":
    unittest.main()
