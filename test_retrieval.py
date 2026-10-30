"""
检索评测：对比 纯向量 / 混合检索 / 混合检索+重排 的 Recall@k 和 MRR@5

用法:
    python test_retrieval.py --validate          # 只检查测试集的标准答案片段能否在库里找到
    python test_retrieval.py --tag baseline      # 跑评测，结果存到 eval_results/retrieval_baseline.json
    python test_retrieval.py --tag rrf --compare eval_results/retrieval_baseline.json

指标说明:
    - 命中：检索结果中某个片段包含任意一个 gold span（NFKC 归一化 + 去空白后比较）
    - Recall@k：前 k 条里有命中的问题占比（每题通常只有 1 个标准片段，等价于 Hit@k）
    - MRR@5：第一个命中片段排名的倒数，前 5 条都没命中记 0
    - 引擎实际只把重排后的 top3 交给 LLM，所以 Recall@3 最接近线上效果
    - 混合+重排(5候选)：混合检索取 5 条再重排；混合+重排(20候选)：混合检索取 20 条，重排后取前 5
"""
import argparse
import contextlib
import io
import json
import os
import re
import unicodedata

import chromadb
import jieba

from hybrid_retriever import HybridRetriever
from reranker import Reranker

K = 5
RERANK_CANDIDATES = 20
TESTSET = "retrieval_testset.json"
RESULT_DIR = "eval_results"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def first_hit_rank(docs: list, spans: list) -> int:
    """返回第一个命中片段的排名 (从 1 开始)，没命中返回 0"""
    norm_spans = [norm(s) for s in spans]
    for rank, doc in enumerate(docs, 1):
        nd = norm(doc)
        if any(s in nd for s in norm_spans):
            return rank
    return 0


@contextlib.contextmanager
def quiet():
    """检索器/重排器每次调用都会 print，评测时屏蔽掉"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def bm25_topk(retriever: HybridRetriever, query: str, k: int) -> list:
    """纯 BM25 结果 (参考用)，逻辑与 HybridRetriever.retrieve 里的 BM25 分支一致"""
    if retriever.bm25 is None:
        retriever._build_bm25()
    scores = retriever.bm25.get_scores(list(jieba.cut(query)))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [retriever.documents[i] for i in top]


def validate(questions: list, all_docs: list) -> bool:
    unique_docs = list(dict.fromkeys(all_docs))
    ok = True
    for q in questions:
        n = sum(1 for d in unique_docs if first_hit_rank([d], q["gold_spans"]))
        if n == 0:
            ok = False
            print(f"❌ {q['id']}: 标准答案片段在库里找不到 -> {q['gold_spans']}")
        elif n > 3:
            print(f"⚠️ {q['id']}: 匹配到 {n} 个不同片段，span 可能太宽泛")
    print(f"{'✅' if ok else '❌'} 校验完成：{len(questions)} 题，库内 {len(all_docs)} 个片段（去重后 {len(unique_docs)} 个）")
    return ok


def summarize(rows: list, method: str) -> dict:
    ranks = [r["ranks"][method] for r in rows]
    n = len(ranks)
    return {
        "n": n,
        "R@1": sum(1 for x in ranks if x == 1) / n,
        "R@3": sum(1 for x in ranks if 0 < x <= 3) / n,
        "R@5": sum(1 for x in ranks if 0 < x <= 5) / n,
        "MRR@5": sum(1 / x for x in ranks if x) / n,
    }


def print_table(title: str, rows: list, methods: list, baseline: dict = None):
    print(f"\n### {title} (n={len(rows)})")
    print(f"{'方法':<14}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR@5':>8}")
    for m in methods:
        s = summarize(rows, m)
        line = f"{m:<14}" + "".join(f"{s[k]:>8.3f}" for k in ("R@1", "R@3", "R@5", "MRR@5"))
        if baseline and m in baseline:
            line += "   Δ R@5 {:+.3f}  Δ MRR {:+.3f}".format(s["R@5"] - baseline[m]["R@5"], s["MRR@5"] - baseline[m]["MRR@5"])
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compare", help="之前保存的结果 json，用来打印差值")
    args = parser.parse_args()

    with open(TESTSET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    collection = chromadb.PersistentClient(path="./my_local_db").get_collection("secure_knowledge_base")
    all_docs = collection.get()["documents"]
    if not validate(questions, all_docs) or args.validate:
        return

    retriever = HybridRetriever(collection)
    with quiet():
        reranker = Reranker()

    methods = ["纯向量", "纯BM25(参考)", "混合检索", "混合+重排(5候选)", f"混合+重排({RERANK_CANDIDATES}候选)"]
    rows, bm25_kept = [], []
    for q in questions:
        query = q["question"]
        with quiet():
            vector = collection.query(query_texts=[query], n_results=K)["documents"][0]
            bm25 = bm25_topk(retriever, query, K)
            hybrid = retriever.retrieve(query, n_results=K)
            # 对混合检索的候选重排 (引擎取 top3，这里保留 5 条以便算 R@5)
            reranked = reranker.rerank(query, hybrid, top_k=K)
            reranked_wide = reranker.rerank(query, retriever.retrieve(query, n_results=RERANK_CANDIDATES), top_k=K)
        results = dict(zip(methods, [vector, bm25, hybrid, reranked, reranked_wide]))
        rows.append({
            **{k: q[k] for k in ("id", "paper", "lang", "type")},
            "ranks": {m: first_hit_rank(docs, q["gold_spans"]) for m, docs in results.items()},
        })
        # 混合结果里有几条是向量检索没给出、只由 BM25 贡献的
        bm25_kept.append(sum(1 for d in hybrid if d not in vector))

    baseline = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            baseline = json.load(f)["summary"]["全部"]

    print_table("全部", rows, methods, baseline)
    for key, values in (("lang", ["en", "zh"]), ("paper", ["RAG", "AMOGEL"]), ("type", ["keyword", "semantic"])):
        for v in values:
            print_table(f"{key}={v}", [r for r in rows if r[key] == v], methods)

    print(f"\n混合结果中由 BM25 贡献 (不在向量 top{K} 里) 的片段：平均 {sum(bm25_kept) / len(bm25_kept):.2f} 条/题，"
          f"{sum(1 for x in bm25_kept if x)} / {len(bm25_kept)} 题至少有 1 条")
    miss = [r["id"] for r in rows if not any(r["ranks"].values())]
    print(f"所有方法 top{K} 都没命中的题 ({len(miss)})：{', '.join(miss) or '无'}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, f"retrieval_{args.tag}.json")
    summary = {"全部": {m: summarize(rows, m) for m in methods}}
    for key, values in (("lang", ["en", "zh"]), ("paper", ["RAG", "AMOGEL"]), ("type", ["keyword", "semantic"])):
        for v in values:
            sub = [r for r in rows if r[key] == v]
            summary[f"{key}={v}"] = {m: summarize(sub, m) for m in methods}
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k": K, "summary": summary, "per_question": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
