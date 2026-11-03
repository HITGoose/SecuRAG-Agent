"""
阶段2 验收：MCP 调用与本地调用等价、越权仍被拒绝、断连可兜底、工单重试不重复、调用开销。
运行（项目根目录）：python -m eval.shop.check_mcp
不调用大模型（断连那一项用"按剧本回复"的假模型驱动 agent_loop）。
"""
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from agent_loop import run_agent_loop
from mcp_client import MCP_TOOLS, ShopMCPClient
from shop_repository import ShopRepository
from shop_tools import TOOL_SCHEMAS, ShopTools, ToolContext, prepare_args

report = {}


class _Proc:
    def __init__(self, pid):
        self.pid = pid

    def kill(self):
        subprocess.run(["taskkill", "/F", "/PID", str(self.pid)], capture_output=True)


def server_processes():
    """找本进程拉起的 shop_mcp_server.py 子进程（用 Windows 自带的 CIM 查询，不额外装 psutil）。"""
    cmd = (f"Get-CimInstance Win32_Process -Filter \"ParentProcessId={os.getpid()}\" | "
           "Where-Object { $_.CommandLine -like '*shop_mcp_server.py*' } | ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
    return [_Proc(int(x)) for x in out.split()]


def main():
    repo = ShopRepository()
    local = ShopTools(repo)
    t = time.perf_counter()
    client = ShopMCPClient()
    client.start()
    report["启动+握手+工具核对耗时ms"] = int((time.perf_counter() - t) * 1000)
    print("MCP Server 已启动，工具核对通过：", sorted(MCP_TOOLS), report["启动+握手+工具核对耗时ms"], "ms")
    try:
        run_checks(repo, local, client)
    finally:
        client.close()  # 出异常也要关闭：否则后台事件循环线程会让进程一直退不出去
    out = Path("eval_results/shop/stage2_mcp_checks.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("结果：", out)


def run_checks(repo, local, client):

    # 1. 等价性：每个用户的每笔订单 + 别人的订单 + 不存在的订单，本地和 MCP 结果逐字段比较
    cases = []
    for uid in repo.users:
        cases.append((uid, "get_order", {}))
        for o in repo.orders.values():
            cases.append((uid, "get_order", {"order_id": o["order_id"]}))      # 自己的 + 别人的
            cases.append((uid, "track_shipment", {"order_id": o["order_id"]}))
        cases.append((uid, "get_order", {"order_id": "ORD-DEMO-9999"}))
    diffs, not_found = [], 0
    for uid, name, args in cases:
        ctx = ToolContext(uid, "eq-test")
        a, b = local.execute(name, args, ctx), client.call(name, args, ctx)
        if a != b:
            diffs.append((uid, name, args))
        if not b["ok"] and b["error"]["code"] == "NOT_FOUND":
            not_found += 1
    others = sum(1 for uid, name, args in cases if args.get("order_id") in repo.orders
                 and repo.orders[args["order_id"]]["user_id"] != uid)
    report["等价性"] = {"比较次数": len(cases), "不一致": len(diffs), "NOT_FOUND次数": not_found,
                       "其中查他人订单的次数": others}
    print("1. 等价性：", report["等价性"], diffs[:3])

    # 2. 越权：模型传 user_id 会被 prepare_args 拒绝；Server 收到未知用户直接拒绝；查他人订单 NOT_FOUND
    other = next(o for o in repo.orders.values() if o["user_id"] == "USR-DEMO-02")
    r_model = prepare_args("get_order", {"order_id": other["order_id"], "user_id": "USR-DEMO-02"})
    r_other = client.call("get_order", {"order_id": other["order_id"]}, ToolContext("USR-DEMO-01", "x"))
    r_fake = client.call("get_order", {}, ToolContext("USR-DEMO-99", "x"))
    report["越权"] = {"模型传user_id": r_model[1], "查他人订单": r_other["error"], "未知用户": r_fake["error"]}
    print("2. 越权：", report["越权"])

    # 3. 工单幂等：同一 request_id 调两次（中间重启一次 Server）只建一张单
    ctx = ToolContext("USR-DEMO-03", f"idem-{time.time_ns()}")
    t1 = client.call("transfer_to_human", {"reason_code": "COMPLAINT", "summary": "幂等测试"}, ctx)
    for p in server_processes():
        p.kill()
    t2 = client.call("transfer_to_human", {"reason_code": "COMPLAINT", "summary": "幂等测试"}, ctx)
    report["工单幂等"] = {"第一次": t1["data"], "重启后第二次": t2["data"], "客户端重连次数": client.reconnects}
    print("3. 工单幂等：", report["工单幂等"])

    # 4. 断连：杀掉 Server，下一次调用自动重连成功
    pids_before = [p.pid for p in server_processes()]
    for p in server_processes():
        p.kill()
    r = client.call("track_shipment", {"order_id": other["order_id"]}, ToolContext("USR-DEMO-02", "x"))
    report["断连自动重连"] = {"被杀的PID": pids_before, "重连后调用成功": r["ok"], "新PID": [p.pid for p in server_processes()],
                        "累计重连次数": client.reconnects}
    print("4. 断连自动重连：", report["断连自动重连"])

    # 5. 重连也失败：Server 脚本换成不存在的路径后再杀掉，agent_loop 应把异常转成错误观察，不崩
    client.params = client.params.model_copy(update={"args": [str(Path(tempfile.gettempdir()) / "no_such_server.py")]})
    for p in server_processes():
        p.kill()
    script = [SimpleNamespace(message={"role": "assistant", "content": "", "tool_calls": [
                  {"id": "c1", "type": "function", "function": {"name": "get_order", "arguments": "{}"}}]},
                  input_tokens=1, output_tokens=1),
              SimpleNamespace(message={"role": "assistant", "content": "抱歉，订单系统暂时不可用。"}, input_tokens=1, output_tokens=1)]
    llm = SimpleNamespace(chat=lambda messages, tools=None, **kw: script.pop(0))
    loop = run_agent_loop(llm, [{"role": "user", "content": "我的订单"}], TOOL_SCHEMAS, prepare_args,
                          lambda name, args: client.call(name, args, ToolContext("USR-DEMO-01", "x")))
    report["Server彻底不可用"] = {"步骤": loop.steps, "停止原因": loop.stop_reason, "回答": loop.answer}
    print("5. Server 彻底不可用：", report["Server彻底不可用"])
    client.params = client.params.model_copy(update={"args": [str(Path(__file__).parents[2] / "shop_mcp_server.py")]})
    client._connect()

    # 6. 调用开销：同一查询本地 vs MCP 各 100 次
    ctx = ToolContext("USR-DEMO-01", "perf")
    oid = next(o["order_id"] for o in repo.orders.values() if o["user_id"] == "USR-DEMO-01")
    timing = {}
    for label, fn in (("本地", lambda: local.execute("get_order", {"order_id": oid}, ctx)),
                      ("MCP", lambda: client.call("get_order", {"order_id": oid}, ctx))):
        xs = []
        for _ in range(100):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1000)
        timing[label] = {"中位数ms": round(statistics.median(xs), 2), "P90ms": round(statistics.quantiles(xs, n=10, method="inclusive")[-1], 2)}
    report["单次调用耗时"] = timing
    print("6. 单次调用耗时：", timing)



if __name__ == "__main__":
    main()
