"""
阶段3：比较技能的三种用法——不用技能（off）、全文常驻（full）、描述 + 按需加载（ondemand）。
运行（项目根目录）：python -m eval.shop.compare_skills deepseek   或   ... ollama

开发集 = 阶段1 的 15 条 + 下面 5 条技能相关对话，只跑 Agent 模式（基线不使用技能）。
记录：完整 token（每个用户轮所有模型调用的输入+输出）、通过数、load_skill 实际调用率、
业务工具调用是否积极（需要工具的对话里，最后一轮有没有成功调用业务工具）。
"""
import json
import statistics
import sys
from pathlib import Path

from eval.shop.run_dev_dialogs import DIALOGS, check, repo, signed_day
from shop_agent import ShopAgent

BUSINESS_TOOLS = {"search_products", "search_policy", "get_order", "track_shipment", "transfer_to_human"}


def find(pred):
    return next(o for o in sorted(repo.orders.values(), key=lambda o: o["order_id"]) if pred(o))


custom = find(lambda o: o["status"] == "已签收" and any(it["product_id"] == "PRD-DEMO-008" for it in o["items"]))
single = find(lambda o: o["user_id"] == "USR-DEMO-05")
stalled = repo.orders[next(s for s in repo.shipments.values() if repo.is_stalled(s))["order_id"]]
fresh = find(lambda o: signed_day(o, "2026-11-02") and all(repo.products[it["product_id"]]["no_reason_return"]
                                                          for it in o["items"]))
day12 = find(lambda o: signed_day(o, "2026-10-22"))

SKILL_DIALOGS = [
    {"id": "s01", "user": custom["user_id"], "turns": [f"{custom['order_id']}这单我不想要了，想退掉"],
     "tools_any": ["get_order"], "must_any": ["不支持", "不适用", "不能无理由", "不可以"]},
    {"id": "s02", "user": single["user_id"], "turns": ["我买的耳机有杂音，想退货"],
     "tools_any": ["get_order"], "must_any": ["人工", "照片", "视频"]},
    {"id": "s03", "user": stalled["user_id"], "turns": [f"{stalled['order_id']}物流一个星期都没动了，什么破快递，我要投诉"],
     "tools_any": ["transfer_to_human"], "must_any": ["TKT-DEMO-"]},
    {"id": "s04", "user": fresh["user_id"], "turns": [f"{fresh['order_id']}不想要了，能退吗"],
     "tools_any": ["get_order"], "must_any": ["完好", "拆封", "二次销售", "配件"]},
    {"id": "s05", "user": day12["user_id"], "turns": [f"{day12['order_id']}用了几天就坏了，想换个新的"],
     "tools_any": ["get_order"], "must_any": ["换货", "换新", "更换"]},
]
# 与技能有关的对话：新加的 5 条 + 阶段1 里属于退换货/投诉的 4 条
SKILL_RELATED = {"s01", "s02", "s03", "s04", "s05", "d05", "d06", "d11", "d12"}


def run_mode(provider, mode):
    agent = ShopAgent(provider=provider, skills=mode)
    rows = []
    try:
        for d in DIALOGS + SKILL_DIALOGS:
            sid = f"{mode}-{d['id']}"
            agent.login(sid, d["user"])
            turns = [agent.handle(sid, t) for t in d["turns"]]
            last = turns[-1]
            fails = check(d, last)
            loaded = [s["args"]["name"] for t in turns for s in t["steps"] if s["tool"] == "load_skill" and s["status"] == "ok"]
            business_ok = {s["tool"] for s in last["steps"] if s["tool"] in BUSINESS_TOOLS and s["status"] in ("ok", "tool_failed")}
            rows.append({"id": d["id"], "pass": not fails, "fails": fails, "loaded_skills": loaded,
                         "business_tools": sorted(business_ok), "needs_tool": bool(d.get("tools_any")), "turns": turns})
            print(f"[{provider}/{mode:8}] {d['id']} {'PASS' if not fails else 'FAIL ' + str(fails)} | "
                  f"skills={loaded} tools={[s['tool'] + ':' + s['status'] for s in last['steps']]} "
                  f"tok={last['input_tokens']}+{last['output_tokens']} | {last['answer'][:70]!r}")
    finally:
        agent.close()
    return rows


def summarize(rows):
    turns = [t for r in rows for t in r["turns"]]
    loop_turns = [t for t in turns if t["route"] == "agent"]
    related = [r for r in rows if r["id"] in SKILL_RELATED]
    unrelated = [r for r in rows if r["id"] not in SKILL_RELATED]
    needs = [r for r in rows if r["needs_tool"]]
    return {
        "通过": f"{sum(r['pass'] for r in rows)}/{len(rows)}",
        "技能相关对话通过": f"{sum(r['pass'] for r in related)}/{len(related)}",
        "其他对话通过": f"{sum(r['pass'] for r in unrelated)}/{len(unrelated)}",
        "load_skill调用率_技能相关": f"{sum(bool(r['loaded_skills']) for r in related)}/{len(related)}",
        "load_skill调用率_其他": f"{sum(bool(r['loaded_skills']) for r in unrelated)}/{len(unrelated)}",
        "需要工具的对话里调用了业务工具": f"{sum(bool(r['business_tools']) for r in needs)}/{len(needs)}",
        "每轮平均token_全部轮": round(statistics.mean(t["input_tokens"] + t["output_tokens"] for t in turns)),
        "每轮平均token_进入循环的轮": round(statistics.mean(t["input_tokens"] + t["output_tokens"] for t in loop_turns)),
        "每轮平均输入token_进入循环的轮": round(statistics.mean(t["input_tokens"] for t in loop_turns)),
        "token总计": sum(t["input_tokens"] + t["output_tokens"] for t in turns),
        "token记录完整": f"{sum(t['tokens_complete'] for t in turns)}/{len(turns)}",
        "每轮平均模型调用_进入循环的轮": round(statistics.mean(t["llm_calls"] for t in loop_turns), 2),
        "延迟中位数ms_进入循环的轮": statistics.median(t["latency_ms"] for t in loop_turns),
    }


def main(provider):
    report = {}
    for mode in ("off", "full", "ondemand"):
        rows = run_mode(provider, mode)
        report[mode] = {"summary": summarize(rows), "rows": rows}
    print()
    for mode, r in report.items():
        print(provider, mode, json.dumps(r["summary"], ensure_ascii=False))
    out = Path(f"eval_results/shop/stage3_skills_{provider}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("逐条结果：", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ollama")
