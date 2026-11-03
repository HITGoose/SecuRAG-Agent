"""
v2 知识库构建：把商品说明和售后政策切块，存进两个独立的 ChromaDB 集合。

运行：python build_shop_kb.py            # 用 config 里的中文嵌入模型
      python build_shop_kb.py default    # 用 ChromaDB 默认英文模型，做对照实验

- 只入库公开信息（商品说明、FAQ、政策）；订单、用户、物流不进向量库，只能通过带权限检查的工具查询。
- 价格、库存不写进文本块，回答时从 products.json 实时读取，避免模型引用过期数字。
- chunk_id 由源ID和小节位置决定（如 PRD-DEMO-001#faq2、POL-RETURN-7D#s3），重建后不变，评测按它判断命中。
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

import config

DATA_DIR = Path(config.SHOP_DATA_DIR)


def product_chunks() -> list:
    data = json.loads((DATA_DIR / "products.json").read_text(encoding="utf-8"))
    chunks = []
    for p in data["products"]:
        specs = "；".join(f"{k}：{v}" for k, v in p["specs"].items())
        meta = {"kind": "product", "source_id": p["product_id"], "version": data["version"], "title": p["name"]}
        chunks.append({"id": f'{p["product_id"]}#main', "metadata": {**meta, "section": "说明"},
                       "text": f'{p["name"]}（{p["category"]}）。{p["description"]} 规格：{specs}。'})
        for i, qa in enumerate(p["faq"], 1):
            chunks.append({"id": f'{p["product_id"]}#faq{i}', "metadata": {**meta, "section": f"FAQ{i}"},
                           "text": f'{p["name"]} 常见问题：{qa["q"]} {qa["a"]}'})
    return chunks


def policy_chunks() -> list:
    chunks = []
    for path in sorted((DATA_DIR / "policies").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        policy_id = re.search(r"^policy_id: (\S+)", text, re.M).group(1)
        version = re.search(r"^version: (\S+)", text, re.M).group(1)
        # 按 "## 小节" 切块，每块前面带上政策标题，单独看也知道是哪条政策
        sections = re.split(r"^## ", text, flags=re.M)[1:]
        for i, sec in enumerate(sections, 1):
            heading, _, body = sec.partition("\n")
            chunks.append({"id": f"{policy_id}#s{i}",
                           "metadata": {"kind": "policy", "source_id": policy_id, "version": version,
                                        "title": title, "section": heading.strip()},
                           "text": f"{title}｜{heading.strip()}：{body.strip()}"})
    return chunks


def embedding_function(model: str):
    if model == "default":
        return embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2（英文）
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model, device="cpu", normalize_embeddings=True)


def model_tag(model: str) -> str:
    return "minilm_en" if model == "default" else re.sub(r"[^a-z0-9]+", "_", Path(model).name.lower())


def collection_names(model: str) -> dict:
    tag = model_tag(model)
    return {"product": f"shop_products__{tag}", "policy": f"shop_policies__{tag}"}


def build(model: str = config.SHOP_EMBED_MODEL) -> dict:
    client = chromadb.PersistentClient(path=config.SHOP_DB_PATH)
    ef = embedding_function(model)
    counts = {}
    for kind, chunks in (("product", product_chunks()), ("policy", policy_chunks())):
        name = collection_names(model)[kind]
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)  # 整个重建，保证和源数据一致
        col = client.create_collection(name, embedding_function=ef, metadata={"hnsw:space": "cosine"})
        col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks])
        counts[name] = col.count()
    return counts


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else config.SHOP_EMBED_MODEL
    for name, n in build(model).items():
        print(f"{name}: {n} 块")
