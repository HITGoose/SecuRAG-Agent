# 🔬 SecuRAG: Academic Research Agent with Hybrid RAG Pipeline

> A production-style RAG Agent for academic literature research, featuring multi-stage retrieval, agentic routing, and automated evaluation.

![System Architecture](./SecuRAG_Retrieval_Pipeline-2026-02-06-094332.png)

## 📖 Introduction

**SecuRAG** is an intelligent research assistant built on a full agentic architecture. Unlike simple RAG systems, SecuRAG implements a **multi-stage retrieval pipeline** and an **autonomous intent router** that decides whether to retrieve from the knowledge base or respond directly.

### Key Features

* **Agentic Intent Router**: Autonomously classifies user queries into SEARCH or CHAT.
* **Hybrid Retrieval Pipeline**: BM25 keyword retrieval and dense vector search (top-20 each), fused with Reciprocal Rank Fusion (RRF).
* **CrossEncoder Reranking**: Two-stage retrieval — the CrossEncoder re-scores 20 fused candidates and keeps the top 3 for generation.
* **Query Rewriting**: LLM-based coreference resolution to handle multi-turn ambiguous queries.
* **Multi-turn Memory**: Session-based conversation history management.
* **LLM-as-a-Judge Eval**: Automated evaluation of answer relevance and completeness.
* **Local Privacy Mode**: Fully offline using Ollama + ChromaDB. Zero data egress.
* **Microservices Architecture**: Decoupled FastAPI backend + Streamlit frontend, Docker-ready.

---

## 📊 Performance

| Component | Setup |
|--------|--------|
| Retrieval | BM25 + Vector hybrid (top-20 each), RRF fusion (k=60) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2, 20 candidates → top 3 |
| Eval | Retrieval: Recall@k / MRR on a 74-question test set; Answers: LLM-as-a-Judge (relevance + completeness) |

### Retrieval Evaluation

Test set: `retrieval_testset.json`, 74 questions written from the two papers in the knowledge base (RAG, Lewis et al. 2020; AMOGEL, Tan et al. 2025) — 58 English / 16 Chinese. Each question's gold answer is a verbatim span of the source chunk; a retrieved chunk counts as a hit if it contains the span. Recall@k = share of questions with a hit in the top k; the engine passes the top 3 to the LLM, so **R@3 is closest to production behaviour**.

| Configuration | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| Baseline: concatenation fusion, rerank 5 candidates | 0.405 | 0.541 | 0.554 | 0.470 |
| A: RRF fusion, rerank 5 candidates | 0.392 | 0.568 | 0.581 | 0.470 |
| **B: RRF fusion, rerank 20 candidates (current)** | **0.459** | **0.662** | **0.743** | **0.563** |

* **RRF alone helps little** (top-5: 5 questions gained, 3 lost). The gain comes from **widening the first-stage candidate pool and letting the reranker choose**: with only 5 candidates the reranker can merely reorder them, and correct chunks ranked 6–20 never reach it. B vs A: top-3 gained 10 / lost 3, top-5 gained 12 / lost 0; English R@3 0.690 → 0.828.
* Cost: CrossEncoder latency on CPU rises from 79 ms to 259 ms (median) per query.
* **Known limitation — Chinese queries**: R@5 stays at 0.188 in every configuration. The embedding model (all-MiniLM-L6-v2) and the reranker are English-only, so Chinese questions against English papers barely match. Multilingual models (e.g. bge-m3 / bge-reranker-v2-m3) are the planned fix.
* Run: `python test_retrieval.py --tag <name> [--compare eval_results/retrieval_baseline.json]`; results are saved in `eval_results/`.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Ollama installed and running
- Docker (optional)

### 1. Clone & Install

git clone https://github.com/HITGoose/SecuRAG-Agent.git
cd SecuRAG-Agent
pip install -r requirements.txt

### 2. Configure Environment

APP_MODE=local
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b   # optional, this is the default
DEEPSEEK_API_KEY=your_key   # cloud mode only (APP_MODE=cloud, model deepseek-chat)

### 3. Pull Local Model

ollama pull qwen2.5:7b

The local model name comes from `OLLAMA_MODEL` (default `qwen2.5:7b`, ~4.7 GB) and must match a model you have pulled. On machines with less memory, `qwen2.5:3b` (~1.9 GB) also works: pull it and set `OLLAMA_MODEL=qwen2.5:3b`.

### 4. Run

Option A: Docker
docker-compose up

Option B: Manual
Terminal 1: python api_server.py
Terminal 2: streamlit run frontend.py

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | DeepSeek API (cloud) / Qwen2.5 via Ollama (local) |
| Vector DB | ChromaDB |
| Retrieval | BM25 + Dense Vector |
| Reranking | CrossEncoder |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker |

---

## 📁 Project Structure

securag_engine.py    - Core RAG Agent engine
hybrid_retriever.py  - BM25 + Vector hybrid retrieval (RRF fusion)
reranker.py          - CrossEncoder reranking
security_guard.py    - Input sanitization
pdf_loader.py        - PDF ingestion pipeline
api_server.py        - FastAPI backend
frontend.py          - Streamlit UI
test_intent.py       - Intent router evaluation
test_retrieval.py    - Retrieval evaluation (Recall@k / MRR)
retrieval_testset.json - 74-question retrieval test set
eval_results/        - Saved retrieval evaluation results
test_eval.py         - LLM-as-a-Judge evaluation
Dockerfile
docker-compose.yml
