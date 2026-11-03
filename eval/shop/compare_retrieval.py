"""
阶段1：在开发集上比较检索配置（英文模型 vs 中文模型，纯向量 / 混合 / 混合+重排）。
运行（在项目根目录）：python -m eval.shop.compare_retrieval
需要先用两种嵌入模型各建一次库：python build_shop_kb.py 和 python build_shop_kb.py default

指标：
- Hit@k：前 k 条里至少有一条 gold
- Recall@k：前 k 条命中的 gold 数 / gold 总数，再对查询取平均（多个 gold 时和 Hit@k 不同）
- MRR@5：第一条 gold 的名次倒数，前 5 没有记 0
"""
import json
import time
from pathlib import Path

import config
from shop_tools import ShopKnowledge

HERE = Path(__file__).parent
CONFIGS = [
    ("英文嵌入 + 英文重排（v1 原配置）", "default", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("中文嵌入 + 英文重排", config.SHOP_EMBED_MODEL, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("中文嵌入 + 中文重排（v2 配置）", config.SHOP_EMBED_MODEL, config.SHOP_RERANK_MODEL),
]
MODES = [("纯向量", "vector", False), ("BM25", "bm25", False), ("混合RRF", "hybrid", False), ("混合+重排", "hybrid", True)]


def score(ranked_ids, gold):
    first = next((i for i, cid in enumerate(ranked_ids[:5], 1) if cid in gold), None)
    return {"hit@1": float(bool(set(ranked_ids[:1]) & gold)), "hit@3": float(bool(set(ranked_ids[:3]) & gold)),
            "recall@3": len(set(ranked_ids[:3]) & gold) / len(gold), "mrr@5": 1 / first if first else 0.0}


def main():
    queries = json.loads((HERE / "dev_retrieval.json").read_text(encoding="utf-8"))["queries"]
    report = []
    for cfg_name, embed, rerank in CONFIGS:
        kb = ShopKnowledge(embed, rerank)
        for mode_name, mode, use_rerank in MODES:
            if mode == "bm25" and embed == "default":
                pass  # BM25 与嵌入模型无关，两种配置结果相同，照样跑一遍便于核对
            rows, latencies = [], []
            for q in queries:
                t = time.perf_counter()
                hits = kb.search(q["kind"], q["query"], top_k=5, mode=mode, rerank=use_rerank)
                latencies.append((time.perf_counter() - t) * 1000)
                rows.append({"id": q["id"], "ranked": [h["id"] for h in hits], **score([h["id"] for h in hits], set(q["gold"]))})
            n = len(rows)
            summary = {k: round(sum(r[k] for r in rows) / n, 3) for k in ("hit@1", "hit@3", "recall@3", "mrr@5")}
            summary["avg_ms"] = round(sum(latencies) / n, 1)
            report.append({"config": cfg_name, "mode": mode_name, "n": n, **summary,
                           "misses@3": [r["id"] for r in rows if r["hit@3"] == 0], "rows": rows})
            print(f"{cfg_name:<22} {mode_name:<6} n={n} Hit@1={summary['hit@1']:.3f} Hit@3={summary['hit@3']:.3f} "
                  f"Recall@3={summary['recall@3']:.3f} MRR@5={summary['mrr@5']:.3f} {summary['avg_ms']}ms "
                  f"未命中@3={report[-1]['misses@3']}")
    out = Path("eval_results/shop/stage1_retrieval_dev.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"逐条结果：{out}")


if __name__ == "__main__":
    main()
