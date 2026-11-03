"""
阶段4 单元测试：事实依据检查、出口防泄露。用假模型和假工具，不加载真实模型。
运行：python -m unittest tests.test_stage4 -v
"""
import time
import unittest
from types import SimpleNamespace

import shop_agent
from shop_agent import FACT_FALLBACK, LEAK_BLOCKED_ANSWER, ShopAgent
from shop_tools import ToolContext


class FakeLLM:
    def __init__(self, text="订单正在运输中，最新到达演示转运中心A。"):
        self.text, self.calls = text, 0

    def chat(self, messages, tools=None, **kw):
        self.calls += 1
        return SimpleNamespace(message={"role": "assistant", "content": self.text}, input_tokens=5, output_tokens=5)


class FakeTools:
    def __init__(self, result):
        self.result, self.calls = result, []

    def execute(self, name, args, ctx):
        self.calls.append((name, args))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_agent(tool_result, llm=None):
    agent = ShopAgent.__new__(ShopAgent)  # 跳过 __init__，不加载模型
    agent.fact_check, agent.mcp, agent.skills_mode = True, None, "off"
    agent.tools, agent.llm = FakeTools(tool_result), llm or FakeLLM()
    return agent


def run_check(agent, intent, steps, answer="编造的物流：已签收"):
    out = {"stop_reason": "answered", "steps": list(steps), "llm_calls": 0, "input_tokens": 0,
           "output_tokens": 0, "tokens_complete": True}
    session = {"current_order": "ORD-DEMO-0009", "history": []}
    final = agent._check_facts(intent, answer, session, "那它现在到哪了", ToolContext("USR-DEMO-01", "r1"), out)
    return final, out


OK_SHIP = {"ok": True, "data": {"最新轨迹": {"location": "演示转运中心A"}}, "error": None}


class TestFactCheck(unittest.TestCase):
    def test_passes_when_tool_succeeded(self):
        agent = make_agent(OK_SHIP)
        final, out = run_check(agent, "LOGISTICS", [{"tool": "track_shipment", "status": "ok"}], "模型回答")
        self.assertEqual((final, out["fact_check"], agent.tools.calls), ("模型回答", "passed", []))

    def test_not_found_counts_as_valid(self):
        agent = make_agent(OK_SHIP)
        steps = [{"tool": "get_order", "status": "tool_failed", "error_code": "NOT_FOUND"}]
        final, out = run_check(agent, "ORDER", steps, "没有找到这个订单")
        self.assertEqual(final, "没有找到这个订单")

    def test_a_no_tool_call_forces_tool_and_reanswers(self):
        agent = make_agent(OK_SHIP)
        final, out = run_check(agent, "LOGISTICS", [])
        self.assertEqual(out["fact_check"], "forced_tool")
        self.assertEqual(agent.tools.calls, [("track_shipment", {"order_id": "ORD-DEMO-0009"})])
        self.assertIn("演示转运中心A", final)
        self.assertEqual(out["steps"][-1]["step"], "forced")

    def test_b_tool_error_and_forced_retry_fails_gives_fixed_reply(self):
        agent = make_agent(ConnectionError("mcp down"))
        steps = [{"tool": "track_shipment", "status": "exception", "error_code": "TOOL_ERROR"}]
        final, out = run_check(agent, "LOGISTICS", steps)
        self.assertEqual((final, out["fact_check"]), (FACT_FALLBACK, "fallback"))
        self.assertEqual(agent.llm.calls, 0)  # 工具失败时不再让模型生成，避免它编

    def test_wrong_tool_does_not_count(self):
        # 问物流却只查了订单：没有物流依据，同样要拦
        agent = make_agent(OK_SHIP)
        final, out = run_check(agent, "LOGISTICS", [{"tool": "get_order", "status": "ok"}])
        self.assertEqual(out["fact_check"], "forced_tool")

    def test_other_intents_and_switch_off_untouched(self):
        agent = make_agent(OK_SHIP)
        self.assertEqual(run_check(agent, "PRODUCT", [])[0], "编造的物流：已签收")
        agent.fact_check = False
        self.assertEqual(run_check(agent, "LOGISTICS", [])[0], "编造的物流：已签收")
        self.assertEqual(agent.tools.calls, [])


class TestOutputGuard(unittest.TestCase):
    def reply(self, answer):
        out, session = {}, {"history": []}
        res = ShopAgent._reply(out, session, "q", answer, "agent", time.perf_counter())
        return res, session

    def test_verbatim_rule_is_blocked_and_not_remembered(self):
        res, session = self.reply("根据规则，第二条是：“订单和物流只能查当前登录用户自己的。”")
        self.assertEqual((res["answer"], res["output_guard"]), (LEAK_BLOCKED_ANSWER, "prompt_leak"))
        self.assertEqual(session["history"], [])

    def test_canary_blocked(self):
        self.assertEqual(self.reply(f"内部标记是 {shop_agent.config.SHOP_CANARY}")[0]["answer"], LEAK_BLOCKED_ANSWER)

    def test_normal_answer_passes(self):
        res, session = self.reply("您的订单正在运输中，最新到达演示转运中心A。")
        self.assertNotIn("output_guard", res)
        self.assertEqual(len(session["history"]), 2)


if __name__ == "__main__":
    unittest.main()
