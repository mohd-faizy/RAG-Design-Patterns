# Standard RAG (Fundamentals Phase)

A stateful, zero-cost, and production-structured implementation of the **Standard Retrieval-Augmented Generation (Standard RAG)** pattern.

---

## 📖 What is Standard RAG?

Standard RAG (sometimes called Naive RAG) is the foundational architecture of retrieval-augmented generation. It solves the static knowledge limitations of Large Language Models (LLMs) by retrieving external context in real-time to generate factually grounded answers.

### What it Solves
*   **Hallucination Reduction**: Grounding model responses in verified source materials.
*   **Real-time Knowledge Update**: Allowing LLMs to answer questions using private or newly updated documents without requiring fine-tuning.
*   **Domain-Specific Expertise**: Providing dynamic context matching specific internal data sources.

---

## 🏗️ Architecture & State Workflow

Unlike linear pipelines, this implementation models RAG as a state-based workflow using **LangGraph**. The workflow propagates state variables (user question, retrieved document context, and generated answer) across discrete execution nodes.

```mermaid
graph TD
    Query([User Question]) --> Retrieve["Retrieve {ChromaDB}"]
    Retrieve --> Generate["Generate {Groq Llama-3.3}"]
    Generate --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
```

### Flow Breakdown
1.  **Retrieve Node**: Semantic vector query searches the local **ChromaDB** index to find the top 3 (`k=3`) closest document chunks using the user's question.
2.  **Generate Node**: Compiles the retrieved document contents and the user query into a clean, system-instructed prompt and invokes **Groq's** fast-tier `llama-3.3-70b-versatile` to produce a fact-grounded response.

---

## 📁 Project Structure

The project code is fully modularized and clean:

```bash
01_Standard_RAG/
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
    ├── prompts.py       # Fact-grounded system prompts
    ├── ingestion.py     # Document splitter and DB builder
    ├── retriever.py     # Chroma vector search retriever interface
    └── graph.py         # LangGraph workflow builder and compiler
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

## 🔬 Core Components & Choices

*   **Orchestration**: `LangGraph` stategraph compiler provides a stateful execution canvas.
*   **Embeddings**: Local open-source `BAAI/bge-small-en-v1.5` embeddings run completely free on your local hardware.
*   **Vector Database**: `ChromaDB` (configured inside `src/ingestion.py` without the deprecated `.persist()` method) automatically maintains the indexed documents in a local `chroma_db` cache.
*   **LLM Engine**: Groq's high-efficiency `llama-3.3-70b-versatile` runs deterministic factual generation (temperature: `0`).
