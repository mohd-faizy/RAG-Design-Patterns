# Multi-Source RAG

A stateful, zero-cost, and production-structured implementation of the **Multi-Source Retrieval-Augmented Generation (Multi-Source RAG)** pattern.

---

## 📖 What is Multi-Source RAG?

Multi-Source RAG addresses the reality that in production enterprise environments, **knowledge is rarely confined to a single database**. Critical data is dispersed across heterogeneous sources — local file indexes, SQL servers, APIs, public websites, and shared document directories.

Relying on a single search index leads to incomplete context coverage, fragmented reasoning, and poor factual answers. A question about current market trends, for example, requires both internal company data and real-time public information.

**Multi-Source RAG** solves this by querying **multiple heterogeneous sources concurrently** and fusing the results:
1.  **Semantic Vector Index (ChromaDB)**: Matches conceptual context in local document records.
2.  **Lexical BM25 Index**: Pulls high-precision exact keyword matches from the same corpus.
3.  **External Web Search (DuckDuckGo)**: Fetches real-time, up-to-date public facts from the open web.

All candidate segments are fused using **Reciprocal Rank Fusion (RRF)**, giving the generator a highly grounded, multi-perspective context block that no single source could provide alone.

---

## 🏗️ Architecture & State Workflow

The workflow executes query fanout, retrieves documents concurrently from three distinct channels, and fuses them into a structured output state:

```mermaid
graph TD
    Query([User Query]) --> Router[Query Router / Fanout]
    
    Router --> Vector["Vector DB Semantic Search {ChromaDB}"]
    Router --> BM25["BM25 Keyword Search {rank-bm25}"]
    Router --> WebSearch["Web Search {DuckDuckGo}"]
    
    Vector --> Fusion["Fusion Layer {Reciprocal Rank Fusion}"]
    BM25 --> Fusion
    WebSearch --> Fusion
    
    Fusion --> Rerank["Reranking {Optional Cross-Encoder}"]
    Rerank --> Groq["Groq LLM {Llama-3.3}"]
    Groq --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Fusion fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### Flow Breakdown
1.  **Query Router / Fanout**: User query triggers concurrent invocations to the Vector Retriever, BM25 Lexical Retriever, and DDG Web Search.
2.  **Fusion Layer (RRF)**: Merges distinct rankings into a unified order based on their rank reciprocals:
    $$RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$
3.  **LLM Generation**: The top 5 fused document blocks are passed as structured prompt context to Groq's high-efficiency `llama-3.3-70b-versatile` LLM to compile the final response.

---

## ⚙️ Key Components

| Component | File | Role |
| :--- | :--- | :--- |
| **State Schema** | `src/state.py` | Defines `GraphState` TypedDict carrying question, context, and answer through the workflow |
| **Document Ingestion** | `src/ingestion.py` | Loads and chunks documents, builds the ChromaDB vector index |
| **Vector Retriever** | `src/vector_retriever.py` | Semantic vector database interface — performs dense similarity search against ChromaDB |
| **BM25 Retriever** | `src/bm25_retriever.py` | Lexical keyword index interface — performs BM25 scoring over tokenized document chunks |
| **Web Retriever** | `src/web_retriever.py` | DuckDuckGo web snippet crawler — fetches real-time public web results for the query |
| **RRF Fusion Engine** | `src/fusion.py` | Reciprocal Rank Fusion algorithm that merges ranked results from all three sources |
| **Prompt Templates** | `src/prompts.py` | Fact-grounded system prompts for answer generation |
| **Workflow Graph** | `src/graph.py` | LangGraph workflow builder coordinating fanout → parallel retrieval → fusion → generation |
| **Application Entry** | `app.py` | Interactive CLI loop for querying the multi-source pipeline |

---

## 🔄 How It Works

1. **Document Ingestion** — Documents are loaded, chunked, and indexed into both ChromaDB (for vector search) and an in-memory BM25 index (for keyword search).

2. **Query Fanout** — When a user submits a question, the query is dispatched simultaneously to three independent retrieval channels:
   - **ChromaDB Vector Search**: Semantic embedding-based similarity.
   - **BM25 Keyword Search**: Lexical token-matching.
   - **DuckDuckGo Web Search**: Real-time public web results.

3. **Source-Specific Retrieval** — Each channel independently returns its ranked results, potentially containing unique documents not found by the other channels.

4. **Reciprocal Rank Fusion** — All three ranked lists are merged using RRF. Documents appearing highly across multiple sources receive the strongest fused scores.

5. **Context Assembly** — The top 5 fused documents are selected and formatted as structured context.

6. **LLM Generation** — The multi-source context is compiled with the user query into a prompt and sent to Groq's `llama-3.3-70b-versatile` for comprehensive answer generation.

---

## 📁 Project Structure

```bash
06_MultiSource_RAG/
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
    ├── vector_retriever.py# Semantic vector database interface
    ├── bm25_retriever.py# Lexical BM25 keyword index interface
    ├── web_retriever.py # DuckDuckGo web snippet crawler
    ├── fusion.py        # Reciprocal Rank Fusion (RRF) algorithm
    └── graph.py         # LangGraph workflow builder and compiler
```

---

## ✅ Advantages

- **Comprehensive Coverage**: Combines internal knowledge (vector + BM25) with external knowledge (web search), covering a far wider information space than any single source.
- **Real-Time Information**: Web search integration ensures the system can answer questions about current events or recently published information.
- **Redundancy & Resilience**: If one source fails to find relevant results, the other sources can compensate, improving system reliability.
- **RRF Quality Boost**: Documents appearing across multiple diverse sources are strong candidates for relevance, producing high-quality context.
- **Lexical + Semantic + Live**: Three fundamentally different retrieval strategies ensure coverage across exact terms, conceptual matches, and fresh information.

## ⚠️ Limitations

- **External Network Dependency**: Web search requires internet access, introducing latency and potential failure points.
- **Higher Latency**: Three parallel retrieval channels plus fusion add processing time compared to single-source retrieval.
- **Web Content Quality**: DuckDuckGo results may contain irrelevant, low-quality, or misleading web snippets.
- **No Source Attribution**: The fused context doesn't clearly distinguish which facts came from which source, reducing explainability.
- **Rate Limiting Risks**: Heavy usage of the DuckDuckGo search API may encounter rate limits in production scenarios.

---

## 🎯 Ideal Use Cases

- **Enterprise Knowledge Management** — Questions that span internal documentation, company policies, and current market information.
- **Competitive Intelligence** — Combining internal product knowledge with real-time public competitor information.
- **Research & Analysis** — Academic or business research where internal papers must be supplemented with current publications.
- **Customer Support** — Answering queries that mix product-specific knowledge with general troubleshooting from public forums.
- **Hybrid Knowledge Domains** — Any scenario where static internal data alone is insufficient and must be augmented with live information.

---

## ⚖️ Comparison with Standard RAG

| Capability | Single-Source RAG | Multi-Source RAG |
| :--- | :--- | :--- |
| **Search Coverage** | ❌ Limited (Vector DB only) | **✅ Dynamic (Vector DB + Web + BM25)** |
| **Keyword Matching** | ❌ Weak | **✅ BM25 high-precision lexical matching** |
| **Knowledge Freshness** | ❌ Out-of-date static database | **✅ Live web search for fresh facts** |
| **Context Breadth** | ❌ Isolated single-source context | **✅ RRF-fused comprehensive grounding** |
| **Complexity** | Low | Moderate (multi-channel coordination) |
