# Reranker-Centric RAG

A highly performant and production-grade implementation of the **Reranker-Centric Retrieval-Augmented Generation** pattern using a cross-encoder neural reranker.

---

## 📖 What is Reranker-Centric RAG?

Reranker-Centric RAG addresses a critical weakness in standard vector-based retrieval: **approximate matching produces noisy context**.

Vector search is fast but mathematically approximate. Embedding-based retrievers map queries and documents into a shared vector space, prioritizing overall semantic proximity over exact, deep semantic matching. Consequently, the top-k retrieved documents often contain noise, are only partially relevant, or place the critical answer snippet too deep inside the context — triggering the well-documented **"Lost in the Middle"** LLM phenomenon.

**Reranker-Centric RAG** solves this by implementing a **two-stage pipeline**:

```text
Stage 1: Retrieve Many Candidates (Top-10 via Vector Search)
Stage 2: Cross-Encoder Reranking (Select Top-3 Best Documents)
```

Cross-Encoder Rerankers process the **query** and **document text** *together* inside the transformer encoder, computing deep bidirectional attention scores across all tokens in both strings. Unlike bi-encoders that encode query and document independently, cross-encoders produce highly precise relevance scores because they see both texts simultaneously.

This strategy dramatically improves retrieval precision, context quality, and hallucination reduction. In practice, modern RAG systems often realize larger accuracy improvements by adding a high-quality reranker than by upgrading their embedding model.

---

## 🏗️ Architecture & State Workflow

Below is the state graph workflow orchestrated by **LangGraph**:

```mermaid
graph TD
    Start([User Query]) --> RetrieveNode["1. Retrieve Candidates<br/>(Retrieve Top-10 Chunks)"]
    RetrieveNode --> RerankNode["2. Cross-Encoder Reranking<br/>(BAAI/bge-reranker-base)<br/>Selects Top-3"]
    RerankNode --> GenerateNode["3. LLM Generation<br/>(llama-3.3-70b-versatile via Groq)"]
    GenerateNode --> End([Final Answer])

    style Start fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style RetrieveNode fill:#0EA5E9,stroke:#0369A1,stroke-width:2px,color:#fff
    style RerankNode fill:#8B5CF6,stroke:#5B21B6,stroke-width:2px,color:#fff
    style GenerateNode fill:#10B981,stroke:#065F46,stroke-width:2px,color:#fff
    style End fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
```

---

## ⚙️ Key Components

| Component | File | Role |
| :--- | :--- | :--- |
| **State Schema** | `src/state.py` | Defines `GraphState` TypedDict carrying question, context, and answer through the workflow |
| **Document Ingestion** | `src/ingestion.py` | Loads and chunks documents, builds the ChromaDB vector index with `BAAI/bge-small-en-v1.5` embeddings |
| **Retriever** | `src/retriever.py` | First-stage retrieval — fetches the top-10 candidate chunks from ChromaDB via semantic similarity |
| **Cross-Encoder Reranker** | `src/reranker.py` | Loads `BAAI/bge-reranker-base` cross-encoder model and scores each (query, document) pair, selecting the top-3 most relevant chunks |
| **Prompt Templates** | `src/prompts.py` | Fact-grounded system prompts constraining the LLM to answer from reranked context |
| **Workflow Graph** | `src/graph.py` | Builds and compiles the LangGraph StateGraph: Retrieve → Rerank → Generate |
| **Application Entry** | `app.py` | Interactive CLI loop for querying the reranker-centric pipeline |

---

## 🔄 How It Works

1. **Document Ingestion** — Documents are loaded, chunked, and indexed into ChromaDB with dense `BAAI/bge-small-en-v1.5` embeddings.

2. **Broad Retrieval (Stage 1)** — The user's query is searched against ChromaDB, returning a large candidate set (top-10 chunks) to maximize recall. At this stage, precision is intentionally traded for coverage.

3. **Cross-Encoder Reranking (Stage 2)** — Each of the 10 candidate chunks is paired with the original query and fed through the `BAAI/bge-reranker-base` cross-encoder model. The cross-encoder computes deep bidirectional attention between all query and document tokens, producing a precise relevance score for each pair.

4. **Top-K Selection** — Candidates are sorted by their cross-encoder scores. Only the top-3 highest-scoring documents are selected, dramatically improving the signal-to-noise ratio of the context.

5. **LLM Generation** — The top-3 reranked documents are compiled into a structured prompt and sent to Groq's `llama-3.3-70b-versatile` for precise, fact-grounded answer generation.

---

## 📁 Project Structure

```bash
07_Reranker_Centric_RAG/
│
├── app.py               # Main CLI interactive loop entrypoint
├── requirements.txt     # Local project packages
│
│
└── src/
    ├── __init__.py      # Package initialization
    ├── state.py         # GraphState schema using TypedDict
    ├── prompts.py       # Fact-grounded system prompts
    ├── ingestion.py     # Document parser and Chroma indexer
    ├── retriever.py     # ChromaDB first-stage retriever (Top-10)
    ├── reranker.py      # BAAI/bge-reranker-base cross-encoder scorer
    └── graph.py         # LangGraph workflow builder and compiler
```

---

## ✅ Advantages

- **Dramatically Higher Precision**: Cross-encoder scoring produces far more accurate relevance assessments than vector cosine similarity alone.
- **Solves "Lost in the Middle"**: By placing the most relevant documents in the top context slots, the reranker prevents the LLM from missing critical information buried in the middle of a long context.
- **Reduced Hallucination**: Cleaner, higher-quality context directly translates to more factually grounded LLM responses.
- **Model-Agnostic Improvement**: Adding a reranker improves results regardless of which embedding model or LLM is used.
- **Runs Locally**: The `BAAI/bge-reranker-base` model runs on local hardware — no external API calls required for reranking.

## ⚠️ Limitations

- **Higher Latency**: The cross-encoder must score each (query, document) pair individually, adding inference time proportional to the number of candidates.
- **Increased Compute**: Running a transformer model for every candidate document requires GPU or significant CPU resources.
- **Model Download Size**: The `BAAI/bge-reranker-base` model must be downloaded on first use (~400MB), adding initial setup time.
- **Not Parallelizable**: Cross-encoder scoring is inherently sequential (or batch-sequential), unlike bi-encoder retrieval which is fully parallelizable.
- **Single-Pass Pipeline**: No self-correction loop — if the initial retrieval misses a document entirely, the reranker cannot recover it.

---

## 🎯 Ideal Use Cases

- **High-Precision QA Systems** — Applications where answer accuracy is more important than response speed (legal, medical, financial).
- **Long Document Corpora** — Knowledge bases with many similar documents where first-pass retrieval frequently returns borderline-relevant results.
- **Enterprise Search** — Production systems where reducing hallucination rates is a critical business requirement.
- **Retrieval Pipeline Optimization** — Drop-in addition to any existing RAG pipeline to boost precision without changing the embedding model.
- **Evaluation & Benchmarking** — Research scenarios measuring the maximum achievable retrieval quality on a given corpus.

---

## ⚖️ Comparison with Standard RAG

| Metric | Standard Vector Retrieval | Reranker-Centric Pipeline |
| :--- | :---: | :---: |
| **Precision** | 6 / 10 | **10 / 10** |
| **Context Quality** | 5 / 10 | **9 / 10** |
| **Hallucination Reduction** | 4 / 10 | **9 / 10** |
| **Semantic Accuracy** | 6 / 10 | **10 / 10** |
| **Latency** | Fast | Moderate (reranker inference overhead) |
| **Compute Requirements** | Low | Higher (cross-encoder model) |
