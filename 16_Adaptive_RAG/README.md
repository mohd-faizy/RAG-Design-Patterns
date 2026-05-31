# Phase 16: Adaptive RAG (Dynamic Retrieval Routing)

Adaptive RAG is a production-grade RAG architecture that intelligently selects the most appropriate retrieval strategy based on the complexity and nature of the incoming query. Instead of blindly executing vector search for every question, Adaptive RAG classifies the query complexity using an LLM router and dynamically dispatches to the optimal retrieval path.

---

## 🏗️ Architecture & State Workflow

```mermaid
graph TD
    Query([User Query]) --> Classifier["Complexity Classifier {Groq Llama-3.3}"]
    
    Classifier -- no_retrieval --> Generate["Generate {Groq Llama-3.3}"]
    Classifier -- vector --> VectorRetrieve["Vector Retrieve {ChromaDB}"]
    Classifier -- hybrid --> HybridRetrieve["Hybrid Retrieve {ChromaDB + BM25}"]
    Classifier -- web --> WebRetrieve["Web Retrieve {DuckDuckGo}"]
    
    VectorRetrieve --> Generate
    HybridRetrieve --> Generate
    WebRetrieve --> Generate
    
    Generate --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Classifier fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

---

## ⚡ Why Adaptive RAG Matters

Traditional RAG systems retrieve context for **every** question — even simple ones like `"What is 2 + 2?"`. This wastes compute, increases latency, and risks injecting irrelevant context into the generation prompt.

Adaptive RAG solves this by learning *what kind of answer* is needed before *how to retrieve it*:

1. **`no_retrieval`** — LLM answers from internal parametric knowledge directly.
2. **`vector`** — Dense semantic search against the local ChromaDB index.
3. **`hybrid`** — Combined vector + BM25 keyword retrieval for comprehensive coverage.
4. **`web`** — Real-time DuckDuckGo search for fresh, current information.

---

## 📊 Capability Comparison

| Feature | Traditional RAG | Adaptive RAG |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Always vector search | Dynamically selected per query |
| **Simple Query Handling** | Over-retrieves (wastes compute) | Skips retrieval entirely |
| **Real-Time Queries** | Misses current information | Routes to live web search |
| **Latency** | Higher (always retrieves) | Lower (skips when unnecessary) |
| **Cost Efficiency** | Fixed pipeline cost | Optimized per-query cost |
| **Retrieval Coverage** | Single strategy | Multi-strategy adaptive routing |

---

## 🔀 Query Routing Examples

| Query | Route | Strategy |
| :--- | :--- | :--- |
| `"What is 2 + 2?"` | `no_retrieval` | LLM answers directly |
| `"What is LangGraph?"` | `vector` | ChromaDB semantic search |
| `"How does ReAct differ from Agentic RAG?"` | `hybrid` | Vector + BM25 combined |
| `"Latest Groq model release?"` | `web` | DuckDuckGo live search |

---

## 📁 Project Structure

```bash
16_Adaptive_RAG/
├── app.py              # CLI Entrypoint loop
├── requirements.txt    # Phase dependencies
└── src/
    ├── __init__.py     # Package marker
    ├── ingestion.py    # Vector database builder (ChromaDB)
    ├── router.py       # LLM-based query complexity classifier
    ├── retrievers.py   # Vector / Hybrid BM25 / Web DuckDuckGo retrievers
    ├── prompts.py      # Prompt templates
    ├── state.py        # LangGraph State Schema (TypedDict)
    └── graph.py        # LangGraph node routing & compilation
```

---

## 🚀 Quick Start Guide

### 1. Install Phase Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Ensure you have a `.env` file at the root of the **entire repository** (`RAG-Design-Patterns/.env`) containing your credentials:
```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the Application
```bash
python app.py
```

### 4. Sample Queries to Test
* `"What is 2 + 2?"` → routes to `no_retrieval`
* `"What is LangGraph?"` → routes to `vector`
* `"Latest AI news?"` → routes to `web`
