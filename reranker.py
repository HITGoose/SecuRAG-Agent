from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"🔄 加载Reranker模型 {model_name} ...")
        # 默认是 v1 用的 ms-marco-MiniLM（只在英文 MS MARCO 上训练）；v2 中文数据传入 bge-reranker-base
        self.model = CrossEncoder(model_name)
        print("✅ Reranker加载完成")
    
    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        """
        对检索结果重新排序
        输入: 查询 + 文档列表
        输出: 重排后的top_k个文档
        """
        if not documents:
            return documents
            
        # 构建(query, doc)对
        pairs = [(query, doc) for doc in documents]
        
        # 打分
        scores = self.model.predict(pairs)
        
        # 按分数排序
        scored_docs = sorted(zip(scores, documents), reverse=True)
        
        top_docs = [doc for _, doc in scored_docs[:top_k]]
        print(f"✅ Reranking完成，从{len(documents)}个文档中选出top{top_k}")
        
        return top_docs

    def rerank_with_scores(self, query: str, items: list, top_k: int = 3) -> list:
        """
        v2 接口：items 是 retrieve_with_ids 的结果（带 id/text/metadata），
        返回前 top_k 条，每条加上 "rerank_score"（模型原始输出，不是概率，只用于排序和阈值实验）。
        """
        if not items:
            return []
        scores = self.model.predict([(query, it["text"]) for it in items])
        ranked = sorted(zip(scores, items), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return [{**it, "rerank_score": float(s)} for s, it in ranked]
