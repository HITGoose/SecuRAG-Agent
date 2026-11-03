"""
端到端延迟测量：开发集 15 条对话重复跑 N 遍，Agent 和写死流程基线交替执行，分别统计每个用户轮的延迟。
运行（项目根目录）：python -m eval.shop.measure_latency ollama 3

- 端到端延迟 = shop_agent.handle 从收到消息到返回回复的时间（含安全审查、意图识别、检索/重排、所有模型调用）
- 先跑一次预热请求（不计入），避免把模型加载时间算进去
- P90 用 statistics.quantiles(n=10, method="inclusive") 计算
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

from eval.shop.run_dev_dialogs import DIALOGS
from shop_agent import ShopAgent


def summarize(values):
    return {"n": len(values), "median_ms": round(statistics.median(values)),
            "p90_ms": round(statistics.quantiles(values, n=10, method="inclusive")[-1]),
            "min_ms": min(values), "max_ms": max(values)}


def main(provider, repeats):
    agent = ShopAgent(provider=provider)
    agent.login("warmup", "USR-DEMO-01")
    agent.handle("warmup", "你好")  # 预热：加载模型，不计入
    ollama_ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout if provider == "ollama" else ""
    print(ollama_ps)

    turns = {"agent": [], "baseline": []}
    for r in range(repeats):
        for d in DIALOGS:
            for mode in ("agent", "baseline"):  # 交替执行，减少机器状态变化对某一方的影响
                sid = f"{mode}-{r}-{d['id']}"
                agent.login(sid, d["user"])
                for t in d["turns"]:
                    res = agent.handle(sid, t, mode=mode)
                    turns[mode].append({"repeat": r, "id": d["id"], "route": res["route"],
                                        "latency_ms": res["latency_ms"], "llm_calls": res["llm_calls"]})
        print(f"第 {r + 1}/{repeats} 遍完成")

    report = {"provider": provider, "repeats": repeats, "ollama_ps": ollama_ps, "summary": {}, "turns": turns}
    for mode, rows in turns.items():
        loop_rows = [x for x in rows if x["route"] in ("agent", "baseline")]
        report["summary"][mode] = {
            "全部用户轮": summarize([x["latency_ms"] for x in rows]),
            "进入循环/流程的轮": summarize([x["latency_ms"] for x in loop_rows]),
            "平均模型调用": round(statistics.mean(x["llm_calls"] for x in rows), 2),
        }
        print(mode, json.dumps(report["summary"][mode], ensure_ascii=False))
    out = Path(f"eval_results/shop/latency_{provider}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"逐条结果：{out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ollama", int(sys.argv[2]) if len(sys.argv) > 2 else 3)
