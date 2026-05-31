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
    Start([User Question]) --> Retrieve[Node: Retrieve Documents]
    
    subgraph Hybrid Search Engine
    Retrieve --> BM25[BM25 Lexical Search]
    Retrieve --> VectorDB[Vector Dense Search]
    BM25 --> RRF[Reciprocal Rank Fusion Fusion]
    VectorDB --> RRF
    end
    
    RRF --> Generate[Node: Generate Answer]
    Generate --> End([Formulate Final Answer])

    style BM25 fill:#4B5563,stroke:#fff,color:#fff
    style VectorDB fill:#4B5563,stroke:#fff,color:#fff
    style RRF fill:#10B981,stroke:#fff,color:#fff
    style Retrieve fill:#2563EB,stroke:#fff,color:#fff
    style Generate fill:#D97706,stroke:#fff,color:#fff
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
├── data/
│   └── sample.txt       # Seed raw data files
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
