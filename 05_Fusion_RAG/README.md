# Phase 05: Fusion RAG (Multi-Query Retrieval Fusion)

Fusion RAG improves retrieval quality by automatically generating multiple semantic variations of a user's query, executing parallel retrievals for all query variations, and combining the results using the **Reciprocal Rank Fusion (RRF)** algorithm to select the highest-quality context documents.

This pattern is highly effective in production environments because it overcomes ambiguous, incomplete, or poorly formulated user queries.

---

## 👁️ Visual Architecture

```mermaid
graph TD
    Query([User Query]) --> Generator["Query Generator {Groq Llama-3.3}"]
    
    subgraph "Query Expansion Layer"
        Generator --> Q1["Query Variation 1"]
        Generator --> Q2["Query Variation 2"]
        Generator --> Q3["Query Variation 3"]
        Generator --> Q4["Query Variation 4"]
    end
    
    subgraph "Parallel Retrieval Layer"
        Q1 --> Retrieve1["Retrieve {ChromaDB}"]
        Q2 --> Retrieve2["Retrieve {ChromaDB}"]
        Q3 --> Retrieve3["Retrieve {ChromaDB}"]
        Q4 --> Retrieve4["Retrieve {ChromaDB}"]
    end
    
    Retrieve1 --> Fusion["Reciprocal Rank Fusion {RRF}"]
    Retrieve2 --> Fusion
    Retrieve3 --> Fusion
    Retrieve4 --> Fusion
    
    Fusion --> Groq["Groq LLM {Llama-3.3-70B}"]
    Groq --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Fusion fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

---

## ⚡ Why Fusion RAG Matters

Users often submit search terms that are too generic, ambiguous, or miss key semantic synonyms.

* **Traditional Single-Query Retrieval:** Often misses crucial documents due to lexical gaps or minor phrasing differences.
* **Fusion RAG Query Expansion:** Automatically generates diverse perspectives (e.g. asking for "ReAct" prompts variations for "reasoning agent", "LLM planning", "tool-using agents").
* **Reciprocal Rank Fusion (RRF):** Fuses these hits mathematically by giving heavier weight to documents ranked highly across multiple variations.

### The Reciprocal Rank Fusion (RRF) Formula

$$RRF(d) = \sum_{q \in Q} \frac{1}{k + r_q(d)}$$

Where:
* $Q$ is the set of query variations.
* $r_q(d)$ is the rank position of document $d$ in retrieval query $q$ (starting at 0).
* $k$ is a constant smoothing parameter (typically set to $60$).

---

## 📊 Capability Comparison

| Capability | Traditional RAG | Fusion RAG |
| :--- | :--- | :--- |
| **Query Sensitivity** | High (Highly dependent on user phrasing) | Low (Extremely robust to query phrasing) |
| **Semantic Coverage** | Limited to single search vector | Expanded (Covers multiple angles & synonyms) |
| **Retrieval Recall** | Moderate / Poor | High / Superior |
| **Parallel Processing** | None | Concurrent Vector Retrievals |
| **Re-ranking Mechanics** | None | Reciprocal Rank Fusion (RRF) |

---

## 📁 Project Structure

```bash
05_Fusion_RAG/
├── app.py              # CLI Entrypoint loop
├── requirements.txt    # Phase-specific dependencies
└── src/
    ├── __init__.py     # Package initialization marker
    ├── ingestion.py    # Vector database builder (ChromaDB)
    ├── fusion.py       # Reciprocal Rank Fusion (RRF) core algorithm
    ├── retriever.py    # Multi-query variations generator & parallel retriever
    ├── prompts.py      # System instructions template
    ├── state.py        # LangGraph State Schema (TypedDict)
    └── graph.py        # LangGraph node configurations & compilation
```

---

## 🚀 Quick Start Guide

### 1. Install Phase Dependencies
Make sure you have your dependencies installed locally in your python virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Ensure you have a `.env` file at the root of the **entire repository** (`RAG-Design-Patterns/.env`) containing your credentials:
```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the Application
Execute the interactive console application:
```bash
python app.py
```

### 4. Sample Queries to Test
Try asking the engine:
* `"What is ReAct?"` (Watch it generate alternative query variations and retrieve fusions)
* `"How does Groq help LLMs?"`
