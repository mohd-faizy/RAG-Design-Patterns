# Hybrid RAG with LangGraph + Groq + BM25 + Vector Search

A stateful, zero-cost, and production-structured implementation of a **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** pipeline.

---

## 📖 What is Hybrid RAG?

While standard RAG excels at understanding conceptual, semantic queries, it frequently fails when matching exact keyword phrases, product IDs, technical error codes, acronyms, or rare terminology. 

**Hybrid RAG** resolves this limitation by running dual-engine retrieval:
1.  **Lexical Search (BM25)**: Performs high-precision, token-based exact word matching.
2.  **Dense Semantic Search (Vector DB)**: Captures abstract intent and conceptual similarity.

Both results are merged using **Reciprocal Rank Fusion (RRF)** to isolate the top factual matches.

---

## 🏗️ Architecture & State Workflow

```mermaid
graph TD
    Query([User Query]) --> Search[Hybrid Search]
    Search --> BM25["BM25 Keyword Search {rank-bm25}"]
    Search --> Vector["Vector DB Semantic Search {ChromaDB}"]
    BM25 --> RRF["RRF Fusion {Reciprocal Rank Fusion}"]
    Vector --> RRF
    RRF --> TopContext[Top Context]
    TopContext --> Groq["Groq LLM {Llama-3.3}"]
    Groq --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style RRF fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### Flow Breakdown
1.  **Dual Search**: The query is split to query a BM25 Okapi model and a local ChromaDB index simultaneously.
2.  **RRF Fusion**: Document ranking positions from both sources are fused using the formula:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    *(where $M$ represents search engines, $r_m(d)$ is the rank of document $d$, and $k = 60$ is a smoothing constant).*
3.  **Generation**: The fused top 3 documents are sent to Groq's high-capacity `llama-3.3-70b-versatile` LLM to compile the final grounded response.

---

## 📁 Project Structure

The codebase is highly modularized and clean:

```bash
02_Hybrid_RAG/
│
├── app.py               # Main CLI interactive loop entrypoint
├── requirements.txt     # Local project packages
│
│
└── src/
    ├── __init__.py      # Package initialization
    ├── state.py         # GraphState schema using TypedDict
    ├── prompts.py       # Fact-grounded prompt templates
    ├── ingestion.py     # Document parser and Chroma indexer
    ├── fusion.py        # Reciprocal Rank Fusion (RRF) logic
    ├── retriever.py     # BM25 + Vector hybrid coordinator
    └── graph.py         # LangGraph workflow builder
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have configured the **centralized `.env`** file in the root folder of the repository workspace:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 2. Install Dependencies
Navigate to this directory and install the required modules:
```bash
pip install -r requirements.txt
```

### 3. Run the Sandbox
Boot the interactive application:
```bash
python app.py
```

---

## ⚖️ Retrieval Strategy Matrix

| Retrieval Type | Strength | Weakness |
| :--- | :--- | :--- |
| **BM25 Lexical** | Exact keywords, codes, names, error tags | Fails on synonyms, typos, or abstract concepts |
| **Vector DB Semantic** | Handles vocabulary mismatches, parses abstract intent | Can hallucinate or omit specific keyword tokens |
| **Hybrid (BM25 + Vector)** | **Best of both worlds: conceptual and lexical accuracy** | Slightly higher complexity |
