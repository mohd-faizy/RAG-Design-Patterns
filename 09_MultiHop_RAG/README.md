# Phase 09: Multi-Hop RAG (Iterative Reasoning Retrieval)

Multi-Hop RAG is a research-grade retrieval architecture designed for questions that cannot be answered from a single document chunk. It performs **iterative retrieval hops** — using the evidence from the first retrieval to generate a bridging follow-up query for the second retrieval — then fuses all gathered evidence to synthesize a deep, multi-document answer.

---

## 🏗️ Architecture & State Workflow

```mermaid
graph TD
    Query([User Question]) --> Hop1["Hop 1 Retrieval {ChromaDB}"]
    Hop1 --> Reasoning["Intermediate Reasoning {Groq Llama-3.3}"]
    Reasoning --> FollowUp["Follow-up Query Generation {Groq Llama-3.3}"]
    FollowUp --> Hop2["Hop 2 Retrieval {ChromaDB}"]
    Hop2 --> Fusion["Evidence Fusion {All Hops Combined}"]
    Fusion --> Generate["Final Answer Generation {Groq Llama-3.3}"]
    Generate --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Fusion fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

---

## ⚡ Why Multi-Hop RAG Matters

Some questions require chaining multiple pieces of evidence:

```text
"Which database is commonly used for Graph RAG systems?"

Hop 1: Graph RAG uses knowledge graphs for retrieval
Hop 2: Neo4j is one of the most popular graph databases
Final:  Graph RAG commonly uses Neo4j
```

Standard RAG retrieves once and misses the connection. Multi-Hop RAG bridges the gap by generating targeted follow-up queries based on what was already found.

---

## 📊 Capability Comparison

| Feature | Traditional RAG | Multi-Hop RAG |
| :--- | :---: | :---: |
| **Reasoning Depth** | 3/10 | 9/10 |
| **Complex QA** | 4/10 | 10/10 |
| **Multi-document Reasoning** | 3/10 | 10/10 |
| **Research Capability** | 4/10 | 9/10 |
| **Evidence Synthesis** | 5/10 | 10/10 |

| Feature | Traditional RAG | Multi-Hop RAG |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Single pass | Iterative 2-hop chain |
| **Reasoning** | Weak (flat context) | Deep (chained evidence) |
| **Follow-up Queries** | None | Auto-generated per hop |
| **Evidence Fusion** | None | All hops merged |
| **Production Use Cases** | Basic QA | Research, Enterprise, Scientific AI |

---

## 📁 Project Structure

```bash
09_MultiHop_RAG/
├── app.py              # CLI Entrypoint loop
├── requirements.txt    # Phase dependencies
└── src/
    ├── __init__.py     # Package marker
    ├── ingestion.py    # Vector database builder (ChromaDB)
    ├── retriever.py    # ChromaDB dense retriever
    ├── reasoning.py    # Follow-up query generator & multi-hop answer synthesizer
    ├── prompts.py      # Prompt templates
    ├── state.py        # LangGraph State Schema (TypedDict)
    └── graph.py        # LangGraph 4-node sequential multi-hop workflow
```

---

## 🚀 Quick Start Guide

### 1. Install Phase Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Ensure you have a `.env` file at the root of the **entire repository** (`RAG-Design-Patterns/.env`):
```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the Application
```bash
python app.py
```

### 4. Sample Queries to Test
* `"What database is commonly used for Graph RAG systems?"` — requires 2-hop reasoning across Graph RAG → Neo4j chunks.
* `"How does ReAct work with Groq?"` — connects ReAct concepts to Groq inference evidence.
