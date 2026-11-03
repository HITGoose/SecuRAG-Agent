"""
阶段1 单元测试：不调用真实模型，用"按剧本回复"的假模型驱动循环。
运行：python -m unittest tests.test_shop_stage1 -v
"""
import json
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

from agent_loop import run_agent_loop
from security_guard import redact_pii
from shop_repository import ShopRepository
from shop_tools import TOOL_SCHEMAS, prepare_args


# ---------- 假模型：按顺序返回预先写好的回复 ----------

def call(name, args, cid="c1"):
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_reply(*calls):
    return SimpleNamespace(message={"role": "assistant", "content": "", "tool_calls": list(calls)},
                           input_tokens=10, output_tokens=5)


def text_reply(text):
    return SimpleNamespace(message={"role": "assistant", "content": text}, input_tokens=10, output_tokens=5)


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []

    def chat(self, messages, tools=None, **kw):
        self.seen_messages.append(list(messages))
        return self.script.pop(0)


def run(script, execute, **kw):
    llm = FakeLLM(script)
    msgs = [{"role": "system", "content": "test"}, {"role": "user", "content": "q"}]
    return run_agent_loop(llm, msgs, TOOL_SCHEMAS, prepare_args, execute, **kw), llm, msgs


def ok_tool(name, args):
    return {"ok": True, "data": {"echo": args}, "error": None}


class TestRedaction(unittest.TestCase):
    def test_order_and_tracking_ids_survive_redaction(self):
        repo = ShopRepository()
        ids = list(repo.orders) + [s["tracking_no"] for s in repo.shipments.values()]
        text = "；".join(ids)
        self.assertEqual(redact_pii(text), text)  # 订单号、运单号一个都不能被误伤

    def test_real_pii_still_redacted_next_to_ids(self):
        text = "订单 ORD-DEMO-0001 运单 SFX-DEMO-12345678 手机13812345678 身份证110101199003071234"
        out = redact_pii(text)
        self.assertIn("ORD-DEMO-0001", out)
        self.assertIn("SFX-DEMO-12345678", out)
        self.assertIn("[PHONE_REDACTED]", out)
        self.assertIn("[ID_REDACTED]", out)
        self.assertNotIn("13812345678", out)


class TestRepository(unittest.TestCase):
    def setUp(self):
        self.repo = ShopRepository(runtime_dir=tempfile.mkdtemp())

    def test_other_users_order_looks_like_not_found(self):
        order = next(o for o in self.repo.orders.values() if o["user_id"] == "USR-DEMO-02")
        self.assertIsNone(self.repo.get_order("USR-DEMO-01", order["order_id"]))
        self.assertIsNone(self.repo.get_order("USR-DEMO-01", "ORD-DEMO-9999"))
        self.assertEqual(self.repo.get_shipment("USR-DEMO-01", order["order_id"]), (None, None))

    def test_seven_day_boundary(self):
        # 签收距今 7 天（10-27 签收）：第 7 天当天 24 点前仍可退；8 天（10-26）已过期
        by_day = {o["signed_at"][:10]: o for o in self.repo.orders.values() if o["status"] == "已签收"}
        self.assertTrue(self.repo.after_sales_windows(by_day["2026-10-27"])["七天无理由退货"]["是否在期限内"])
        self.assertFalse(self.repo.after_sales_windows(by_day["2026-10-26"])["七天无理由退货"]["是否在期限内"])
        self.assertTrue(self.repo.after_sales_windows(by_day["2026-10-19"])["质量问题换货"]["是否在期限内"])   # 15 天
        self.assertFalse(self.repo.after_sales_windows(by_day["2026-10-18"])["质量问题换货"]["是否在期限内"])  # 16 天

    def test_ticket_is_idempotent(self):
        t1 = self.repo.create_ticket("USR-DEMO-01", "k1", "USER_REQUEST", "a")
        t2 = self.repo.create_ticket("USR-DEMO-01", "k1", "USER_REQUEST", "a")
        self.assertEqual(t1["ticket_id"], t2["ticket_id"])
        self.assertTrue(t2["duplicate"])

    def test_data_consistency(self):
        self.assertEqual(len(self.repo.orders), 40)
        for o in self.repo.orders.values():
            self.assertIn(o["user_id"], self.repo.users)
            for it in o["items"]:
                self.assertIn(it["product_id"], self.repo.products)
            self.assertEqual(o["total_cents"], o["goods_cents"] + o["shipping_cents"])
            times = [o[k] for k in ("created_at", "paid_at", "shipped_at", "signed_at") if o[k]]
            self.assertEqual(times, sorted(times))
        for oid, s in self.repo.shipments.items():
            self.assertIsNotNone(self.repo.orders[oid]["shipped_at"])
            self.assertRegex(s["tracking_no"], r"^[A-Z]{3}-DEMO-\d{8}$")
        stalled = [s for s in self.repo.shipments.values() if self.repo.is_stalled(s)]
        self.assertEqual(len(stalled), 3)


class TestPrepareArgs(unittest.TestCase):
    def test_normalize_and_reject(self):
        self.assertEqual(prepare_args("get_order", {"order_id": " ord-demo-0001 "}), ({"order_id": "ORD-DEMO-0001"}, None))
        self.assertEqual(prepare_args("get_order", {"order_id": ""}), ({}, None))
        self.assertIsNotNone(prepare_args("get_order", {"order_id": "12345"})[1])
        self.assertIsNotNone(prepare_args("get_order", {"user_id": "USR-DEMO-02"})[1])  # 模型不能传 user_id
        self.assertIsNotNone(prepare_args("delete_order", {})[1])
        self.assertIsNotNone(prepare_args("transfer_to_human", {"reason_code": "REFUND_NOW", "summary": "x"})[1])


class TestAgentLoop(unittest.TestCase):
    def test_tool_then_answer(self):
        res, llm, msgs = run([tool_reply(call("get_order", {"order_id": "ORD-DEMO-0001"})), text_reply("好的")], ok_tool)
        self.assertEqual((res.answer, res.stop_reason, res.llm_calls), ("好的", "answered", 2))
        self.assertEqual([m["role"] for m in msgs[2:]], ["assistant", "tool"])  # 合法的消息序列
        self.assertEqual(msgs[3]["tool_call_id"], "c1")

    def test_bad_json_becomes_observation_and_loop_continues(self):
        res, llm, msgs = run([tool_reply(call("get_order", "{not json")), text_reply("请提供订单号")], ok_tool)
        self.assertEqual(res.steps[0]["status"], "bad_args")
        self.assertEqual(res.stop_reason, "answered")
        self.assertIn("BAD_ARGUMENTS", msgs[3]["content"])

    def test_unknown_tool(self):
        res, _, _ = run([tool_reply(call("refund_now", {})), text_reply("x")], ok_tool)
        self.assertEqual(res.steps[0]["status"], "bad_args")

    def test_tool_exception_is_caught(self):
        def boom(name, args):
            raise RuntimeError("C:/secret/path db down")
        res, _, msgs = run([tool_reply(call("get_order", {})), text_reply("系统忙")], boom)
        self.assertEqual(res.steps[0]["status"], "exception")
        self.assertNotIn("secret", msgs[3]["content"])  # 不把内部异常信息给模型

    def test_tool_timeout(self):
        def slow(name, args):
            time.sleep(0.5)
            return ok_tool(name, args)
        res, _, _ = run([tool_reply(call("get_order", {})), text_reply("x")], slow, tool_timeout_s=0.1)
        self.assertEqual(res.steps[0]["status"], "timeout")

    def test_repeated_call_is_not_executed_and_stops_after_two(self):
        executed = []
        def counting(name, args):
            executed.append(args)
            return ok_tool(name, args)
        same = {"order_id": "ord-demo-0001"}  # 大小写不同，规范化后仍算重复
        script = [tool_reply(call("get_order", {"order_id": "ORD-DEMO-0001"})),
                  tool_reply(call("get_order", same)), tool_reply(call("get_order", same)), text_reply("不会走到")]
        res, _, _ = run(script, counting)
        self.assertEqual(len(executed), 1)
        self.assertEqual([s["status"] for s in res.steps], ["ok", "duplicate", "duplicate"])
        self.assertEqual(res.stop_reason, "repeated")

    def test_max_steps(self):
        script = [tool_reply(call("search_policy", {"query": f"q{i}"})) for i in range(5)]
        res, _, _ = run(script, ok_tool, max_steps=5, max_tool_calls=10)
        self.assertEqual((res.stop_reason, res.llm_calls), ("max_steps", 5))

    def test_tool_budget(self):
        script = [tool_reply(call("search_policy", {"query": f"q{i}"})) for i in range(3)]
        res, _, _ = run(script, ok_tool, max_steps=5, max_tool_calls=2)
        self.assertEqual(res.stop_reason, "tool_budget")
        self.assertEqual([s["status"] for s in res.steps], ["ok", "ok", "budget"])

    def test_too_many_calls_in_one_response(self):
        calls = [call("search_policy", {"query": f"q{i}"}, cid=f"c{i}") for i in range(5)]
        res, _, msgs = run([tool_reply(*calls), text_reply("x")], ok_tool, max_calls_per_response=3)
        self.assertEqual([s["status"] for s in res.steps], ["ok"] * 3 + ["skipped"] * 2)
        self.assertEqual(sum(1 for m in msgs if m["role"] == "tool"), 5)  # 每个 tool_call_id 都有回应

    def test_llm_error_and_empty_answer(self):
        class Broken:
            def chat(self, *a, **k):
                raise ConnectionError()
        res = run_agent_loop(Broken(), [], TOOL_SCHEMAS, prepare_args, ok_tool)
        self.assertEqual(res.stop_reason, "llm_error")
        res, _, _ = run([text_reply("   ")], ok_tool)
        self.assertEqual(res.stop_reason, "empty_answer")


if __name__ == "__main__":
    unittest.main()
