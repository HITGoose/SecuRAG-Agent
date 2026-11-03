"""
阶段4：在开发集上选"低相关"阈值——重排第 1 名的分数低于它，就认为知识库里没有可靠依据。
运行（项目根目录）：python -m eval.shop.pick_threshold

知识库内的问题：阶段1 检索开发集的 21 条（都有 gold）；
知识库外的问题：下面 16 条，商品或政策里都没有对应内容。
不用阶段6 的测试集；阈值选定后写进 config.SHOP_LOW_RELEVANCE_THRESHOLD。
"""
import json
from pathlib import Path

from shop_tools import ShopKnowledge

HERE = Path(__file__).parent
OUT_OF_KB = [
    ("product", "有没有卖笔记本电脑"), ("product", "iPhone 16 Pro 多少钱"), ("product", "你们的扫地机器人吸力多大"),
    ("product", "有没有适合跑马拉松的运动手表"), ("product", "电动自行车续航多少公里"), ("product", "有没有卖猫粮"),
    ("product", "游戏手柄支持哪些主机"), ("product", "投影仪的亮度是多少流明"),
    ("policy", "支持花呗分期付款吗"), ("policy", "能寄到海外吗"), ("policy", "会员积分怎么兑换"),
    ("policy", "可以货到付款吗"), ("policy", "生日有没有优惠券"), ("policy", "能不能用数字人民币支付"),
    ("policy", "门店自提怎么预约"), ("policy", "以旧换新怎么折价"),
]


def main():
    kb = ShopKnowledge()
    in_kb = json.loads((HERE / "dev_retrieval.json").read_text(encoding="utf-8"))["queries"]
    rows = []
    for q in in_kb:
        top = kb.search(q["kind"], q["query"], top_k=1)[0]
        rows.append({"query": q["query"], "kind": q["kind"], "in_kb": True, "top1": round(top["rerank_score"], 4)})
    for kind, query in OUT_OF_KB:
        top = kb.search(kind, query, top_k=1)[0]
        rows.append({"query": query, "kind": kind, "in_kb": False, "top1": round(top["rerank_score"], 4),
                     "top1_chunk": top["id"]})

    ins = sorted(r["top1"] for r in rows if r["in_kb"])
    outs = sorted(r["top1"] for r in rows if not r["in_kb"])
    print("知识库内 top1：", ins)
    print("知识库外 top1：", outs)
    # 在所有分数的中点上找让"内判内、外判外"总正确数最多的阈值；并列时取更低的（宁可少拦正常问题）
    scores = sorted(set(ins + outs))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best = max(candidates, key=lambda t: (sum(s >= t for s in ins) + sum(s < t for s in outs), -t))
    result = {"threshold": round(best, 4),
              "知识库内判为低相关(误拦)": f"{sum(s < best for s in ins)}/{len(ins)}",
              "知识库外判为低相关(拦住)": f"{sum(s < best for s in outs)}/{len(outs)}",
              "rows": rows}
    print({k: v for k, v in result.items() if k != "rows"})
    out = Path("eval_results/shop/stage4_threshold_dev.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
