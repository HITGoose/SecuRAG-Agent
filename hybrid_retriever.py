from rank_bm25 import BM25Okapi
import jieba

class HybridRetriever:
    def __init__(self, collection, candidate_k: int = 20, rrf_k: int = 60, query_prefix: str = ""):
        self.collection = collection
        self.bm25 = None
        self.documents = []
        self.ids = []
        self.metadatas = []
        self.candidate_k = candidate_k  # 每一路召回的候选数
        self.rrf_k = rrf_k              # RRF 平滑常数，论文常用 60
        self.query_prefix = query_prefix  # 只加在向量检索的查询前（bge 中文模型的检索指令），BM25 用原始查询

    def _build_bm25(self):
        """从ChromaDB加载所有文档构建BM25索引"""
        results = self.collection.get()
        if not results['documents']:
            return
        self.documents = results['documents']
        self.ids = results['ids']
        self.metadatas = results['metadatas'] or [{} for _ in self.ids]
        # 用jieba分词（支持中文）
        tokenized = [list(jieba.cut(doc)) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ BM25索引构建完成，共{len(self.documents)}个文档")

    def _vector_search(self, query: str, k: int) -> list:
        results = self.collection.query(query_texts=[query], n_results=k)
        return results['documents'][0] if results['documents'] else []

    def _bm25_search(self, query: str, k: int) -> list:
        if self.bm25 is None:
            self._build_bm25()
        if not (self.bm25 and self.documents):
            return []
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]

    def retrieve(self, query: str, n_results: int = 3) -> list:
        """
        混合检索：向量和 BM25 各召回 candidate_k 条，用 RRF (Reciprocal Rank Fusion) 融合后取前 n_results 条。
        RRF: score(d) = Σ 1 / (rrf_k + rank_i(d))，只看名次不看原始分数，
        所以不用把 BM25 分数和向量距离归一化到同一尺度；两路都靠前的文档得分最高。
        """
        k = max(self.candidate_k, n_results)
        vector_docs = self._vector_search(query, k)
        bm25_docs = self._bm25_search(query, k)

        scores = {}
        for docs in (vector_docs, bm25_docs):
            # 同一段文本可能被重复入库，先在这一路内部去重，名次按去重后的顺序计
            for rank, doc in enumerate(dict.fromkeys(docs), 1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rank)
        combined = sorted(scores, key=scores.get, reverse=True)[:n_results]
        print(f"🔍 混合检索完成：向量{len(vector_docs)}条 + BM25{len(bm25_docs)}条 → RRF融合取{len(combined)}条")

        return combined

    # ---------- v2：带 chunk_id 的结构化接口（上面 v1 的 retrieve 保持原样）----------

    def retrieve_with_ids(self, query: str, n_results: int = 20, mode: str = "hybrid") -> list:
        """
        返回 [{"id", "text", "metadata", "score"}]，按名次排好。
        mode: "hybrid" = 向量 + BM25 用 RRF 融合；"vector" / "bm25" = 只用一路（做对照实验用）。
        和 v1 的 retrieve 算法相同，区别是按 chunk_id 计分，结果能追溯到具体片段。
        """
        if self.bm25 is None:
            self._build_bm25()
        if not self.ids:
            return []
        k = min(max(self.candidate_k, n_results), len(self.ids))

        routes = []
        if mode in ("hybrid", "vector"):
            res = self.collection.query(query_texts=[self.query_prefix + query], n_results=k)
            routes.append(res['ids'][0])
        if mode in ("hybrid", "bm25"):
            scores = self.bm25.get_scores(list(jieba.cut(query)))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            routes.append([self.ids[i] for i in top])

        rrf = {}
        for ranked_ids in routes:
            for rank, cid in enumerate(ranked_ids, 1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        top_ids = sorted(rrf, key=rrf.get, reverse=True)[:n_results]

        pos = {cid: i for i, cid in enumerate(self.ids)}
        return [{"id": cid, "text": self.documents[pos[cid]], "metadata": self.metadatas[pos[cid]] or {},
                 "score": rrf[cid]} for cid in top_ids]
