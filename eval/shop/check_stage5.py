"""
阶段5 验收：模型重试与降级、降级后的可用率、MCP 超时（只读兜底 / 转人工幂等）。
运行（项目根目录）：python -m eval.shop.check_stage5

故障都是注入的：模型侧用 LLMGateway.fault 在真正请求前抛出模拟的异常（不是用错误 key 代替所有故障）；
MCP 侧用 Server 的 SHOP_MCP_FAULT_DELAY 环境变量让响应延迟。
"""
import json
import statistics
from pathlib import Path

import httpx
from openai import (APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
                    InternalServerError, RateLimitError)

import config
from eval.shop.run_dev_dialogs import DIALOGS, check, moving, repo
from llm_gateway import LLMGateway
from shop_agent import FACT_FALLBACK, UNAVAILABLE_ANSWER, ShopAgent

REQ = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def make_error(kind):
    if kind == "timeout":
        return APITimeoutError(request=REQ)
    if kind == "conn":
        return APIConnectionError(request=REQ)
    status, cls = {"429": (429, RateLimitError), "503": (503, InternalServerError),
                   "401": (401, AuthenticationError), "400": (400, BadRequestError)}[kind]
    return cls(f"injected {status}", response=httpx.Response(status, request=REQ), body=None)


class Fault:
    """按后端注入故障：deepseek / ollama 各自"前 N 次失败"，N=-1 表示一直失败。同时数每个后端的尝试次数。"""

    def __init__(self, deepseek=("", 0), ollama=("", 0)):
        self.plan = {"deepseek": list(deepseek), "ollama": list(ollama)}
        self.attempts = {"deepseek": 0, "ollama": 0}

    def __call__(self, provider):
        self.attempts[provider] += 1
        kind, times = self.plan[provider]
        if kind and (times == -1 or self.attempts[provider] <= times):
            raise make_error(kind)


def gateway_cases():
    rows = []
    cases = [("429 一次后恢复", ("429", 1), ("", 0)), ("503 持续", ("503", -1), ("", 0)),
             ("超时持续", ("timeout", -1), ("", 0)), ("401 认证失败", ("401", -1), ("", 0)),
             ("400 参数错误", ("400", -1), ("", 0)), ("云端和本地都失败", ("503", -1), ("conn", -1))]
    for name, ds, ol in cases:
        fault = Fault(ds, ol)
        gw = LLMGateway("deepseek", backoff_s=0.05, fault=fault)
        try:
            r = gw.chat([{"role": "user", "content": "只回复：好的"}], max_tokens=10, purpose="test")
            outcome = {"结果": "成功", "实际后端": r.provider, "重试": r.retries, "降级原因": r.fallback_reason,
                       "回复": r.message["content"][:20]}
        except Exception as e:
            outcome = {"结果": f"抛出 {type(e).__name__}"}
        rows.append({"场景": name, **outcome, "deepseek尝试次数": fault.attempts["deepseek"],
                     "ollama尝试次数": fault.attempts["ollama"]})
        print(rows[-1])
    return rows


def availability(fault_factory, label, dialogs):
    agent = ShopAgent(provider="deepseek", trace=True)
    agent.llm.backoff_s = 0.05
    agent.llm.fault = fault_factory()
    turns, passed = [], 0
    try:
        for d in dialogs:
            sid = f"{label}-{d['id']}"
            agent.login(sid, d["user"])
            res = [agent.handle(sid, t) for t in d["turns"]]
            passed += not check(d, res[-1])
            turns += res
    finally:
        agent.close()
    failed = [t for t in turns if t["route"] == "safety_unavailable" or t["stop_reason"] == "llm_error"]
    providers = {}
    for t in turns:
        for c in t["llm"]:
            key = c["provider"] + ("(失败)" if c["error"] else "")
            providers[key] = providers.get(key, 0) + 1
    out = {"场景": label, "用户轮": len(turns), "正常返回业务回复": len(turns) - len(failed),
           "可用率": f"{len(turns) - len(failed)}/{len(turns)}", "任务通过": f"{passed}/{len(dialogs)}",
           "模型调用按实际后端": providers, "延迟中位数ms": statistics.median(t["latency_ms"] for t in turns),
           "固定失败话术示例": failed[0]["answer"] if failed else None}
    print(out)
    return out


def mcp_timeouts():
    from eval.shop.check_mcp import server_processes
    results = {}
    agent = ShopAgent(provider="deepseek", tool_backend="mcp")
    try:
        def restart_server(fault):
            agent.mcp.params = agent.mcp.params.model_copy(update={"env": {"PYTHONIOENCODING": "utf-8",
                                                                           "SHOP_MCP_FAULT_DELAY": fault}})
            agent.mcp.timeout_s = 3  # 客户端超时 3 秒，Server 故意延迟 5 秒
            for p in server_processes():
                p.kill()
            agent.mcp._disconnect()
            agent.mcp._connect()

        def summarize(r):
            return {"route": r["route"], "fact_check": r["fact_check"], "latency_ms": r["latency_ms"],
                    "tools": [f"{s['tool']}:{s['status']}:{s.get('error_code')}" + (":forced" if s.get("step") == "forced" else "")
                              for s in r["steps"]], "answer": r["answer"]}

        uid, oid = moving["user_id"], moving["order_id"]
        true_loc = repo.shipments[oid]["events"][-1]["location"]
        # 1) 物流查询第 1 次响应超时，之后恢复：循环里超时 → 事实检查补查 → 正确回答
        restart_server("track_shipment:5:1")
        agent.login("t1", uid)
        r = agent.handle("t1", f"{oid}现在到哪了")
        results["物流超时一次后恢复"] = {**summarize(r), "回答含真实位置": true_loc in r["answer"]}
        # 2) 物流查询一直超时：补查也超时 → 固定话术，不给物流状态
        restart_server("track_shipment:5:10")
        agent.login("t2", uid)
        r = agent.handle("t2", f"{oid}现在到哪了")
        results["物流一直超时"] = {**summarize(r), "是固定话术": r["answer"] == FACT_FALLBACK}
        # 3) 转人工：Server 已建单但响应超时 → 客户端用同一 request_id 重试 → 拿到同一张单，不重复建单
        restart_server("transfer_to_human:5:1")
        before = json.loads(Path(config.SHOP_RUNTIME_DIR, "tickets.json").read_text(encoding="utf-8"))
        agent.login("t3", uid)
        r = agent.handle("t3", "我要转人工客服")
        after = json.loads(Path(config.SHOP_RUNTIME_DIR, "tickets.json").read_text(encoding="utf-8"))
        new = [t for t in after if t["ticket_id"] not in {x["ticket_id"] for x in before}]
        results["转人工响应超时"] = {**summarize(r), "新增工单数": len(new), "新增工单": [t["ticket_id"] for t in new],
                               "客户端超时重试次数": agent.mcp.timeout_retries}
        # 4) 超时之后会话还能正常用
        restart_server("")
        agent.login("t4", uid)
        r = agent.handle("t4", f"{oid}现在到哪了")
        results["故障解除后"] = {**summarize(r), "回答含真实位置": true_loc in r["answer"]}
    finally:
        agent.close()
    for k, v in results.items():
        print(k, v)
    return results


def main():
    report = {"模型故障注入": gateway_cases()}
    report["降级可用率_DeepSeek一直503"] = availability(lambda: Fault(("503", -1), ("", 0)), "ds-down", DIALOGS)
    report["降级可用率_云端本地都失败"] = availability(lambda: Fault(("503", -1), ("conn", -1)), "both-down", DIALOGS[:3])
    report["MCP超时"] = mcp_timeouts()
    out = Path("eval_results/shop/stage5_faults.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("结果：", out)


if __name__ == "__main__":
    main()
