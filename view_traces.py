"""
命令行查看本地 trace（阶段5）。

    python view_traces.py              # 最近 20 个请求的摘要
    python view_traces.py -n 50        # 最近 50 个
    python view_traces.py <trace_id>   # 展开一个请求的每一步
    python view_traces.py --stats      # 汇总：路由分布、降级次数、每轮模型调用、延迟中位数/P90
"""
import argparse
import statistics

from tracing import read_traces


def summary_line(t):
    flags = []
    if t["fallback_used"]:
        flags.append("降级")
    if t.get("fact_check") in ("forced_tool", "fallback"):
        flags.append(f"事实检查:{t['fact_check']}")
    tokens = f"{t['input_tokens']}+{t['output_tokens']}" + ("" if t["tokens_complete"] else "(不完整)")
    return (f"{t['time'][5:]}  {t['trace_id']}  {t['route']:<9} {str(t['intent']):<15} "
            f"工具{len(t['tools'])} 模型{t['llm_calls']} token {tokens:<12} {t['latency_ms']:>6}ms  "
            f"{' '.join(flags)}  {t['message'][:30]}")


def show_detail(t):
    print(f"trace {t['trace_id']}  {t['time']}  会话 {t['session']}  用户 {t['user']}")
    print(f"消息：{t['message']}")
    print(f"意图 {t['intent']} → 路由 {t['route']}，停止原因 {t['stop_reason']}，事实检查 {t.get('fact_check')}")
    print("\n模型调用：")
    for i, c in enumerate(t["llm"], 1):
        extra = f" 重试{c['retries']}次" if c.get("retries") else ""
        extra += f" 降级({c['fallback_reason']})" if c.get("fallback_reason") else ""
        print(f"  {i}. {c['purpose']:<8} {c['provider']}/{c['model']}  {c['latency_ms']}ms  "
              f"token {c['input_tokens']}+{c['output_tokens']}{extra}")
    print("\n工具调用：")
    for s in t["tools"]:
        print(f"  [{s['step']}] {s['tool']}({s['args']}) → {s['status']}"
              + (f" {s['error_code']}" if s["error_code"] else "") + f"  via {s.get('via')}  {s['latency_ms']}ms")
    print(f"\n回答：{t['answer']}")
    print(f"端到端 {t['latency_ms']}ms")


def show_stats(traces):
    lat = [t["latency_ms"] for t in traces]
    routes = {}
    for t in traces:
        routes[t["route"]] = routes.get(t["route"], 0) + 1
    print(f"请求数 {len(traces)}；路由分布 {routes}")
    print(f"降级 {sum(t['fallback_used'] for t in traces)} 次；事实检查介入 "
          f"{sum(t.get('fact_check') in ('forced_tool', 'fallback') for t in traces)} 次")
    print(f"每轮模型调用平均 {statistics.mean(t['llm_calls'] for t in traces):.2f}")
    if len(lat) >= 2:
        print(f"延迟中位数 {statistics.median(lat)}ms，P90 {statistics.quantiles(lat, n=10, method='inclusive')[-1]:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="查看 SecuRAG 电商客服的本地 trace")
    parser.add_argument("trace_id", nargs="?")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    traces = read_traces()
    if not traces:
        print("logs/ 里还没有 trace")
        return
    if args.stats:
        show_stats(traces)
    elif args.trace_id:
        match = [t for t in traces if t["trace_id"].startswith(args.trace_id)]
        show_detail(match[-1]) if match else print("没有找到这个 trace_id")
    else:
        for t in traces[-args.n:]:
            print(summary_line(t))


if __name__ == "__main__":
    main()
