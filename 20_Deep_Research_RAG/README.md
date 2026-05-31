# Deep Research RAG using LangGraph + Groq + Multi-Step Autonomous Research

A highly stateful, production-structured, and zero-cost implementation of the **Deep Research Retrieval-Augmented Generation (Deep Research RAG)** pattern.

---

## 📖 What is Deep Research RAG?

Standard RAG architectures execute a linear, single-hop lookup to answer user queries:
```
Retrieve → Answer
```

However, complex analysis queries require multi-step reasoning, evidence synthesis across disparate sources, planning, knowledge gap detection, and iterative searches. 

**Deep Research RAG** is the foundation behind modern autonomous AI research agents (such as OpenAI Deep Research). Instead of a simple lookup, it performs a structured, multi-step investigation loop:
1.  **Plan Research**: A central coordinator breaks the complex target task down into discrete, highly targeted sub-questions.
2.  **Autonomous Investigation**: A multi-threaded researcher concurrently queries internal vector storage (ChromaDB) and the live web (DuckDuckGo Search) for each sub-question.
3.  **Knowledge Synthesis**: Combines all pieces of evidence into a highly formatted, multi-perspective analyst report.

---

## 🏗️ Architecture & State Workflow

### 1. Agentic Decision Flow

```mermaid
graph TD
    Query([User Query]) --> Planner["Research Planner {Planner Agent}"]
    
    subgraph Parallel Search Engine
        Planner --> Vector["Vector DB {ChromaDB}"]
        Planner --> Web["Web Search {DuckDuckGo}"]
        Planner --> APIs["APIs / Docs"]
    end
    
    Vector --> Evidence["Evidence Collection"]
    Web --> Evidence
    APIs --> Evidence
    
    Evidence --> Synthesis["Knowledge Synthesis {Synthesizer Agent}"]
    Synthesis --> Gap["Gap Identification"]
    Gap --> Additional["Additional Research"]
    Additional --> Report([Final Report])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style Report fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Synthesis fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### 2. State-Based Graph Schema

```
                      +-------------------+
                      |    planning_node  |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   research_node   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   synthesis_node  |
                      +---------+---------+
                                |
                                v
                            [  END  ]
```

---

## 📁 Project Structure

The project has a modular, scalable design:

```bash
20_Deep_Research_RAG/
│
├── app.py               # Main CLI interactive loop entrypoint
├── requirements.txt     # Local project packages
├── .env                 # Environment variables configuration
│
│
└── src/
    ├── __init__.py      # Package initialization
    ├── state.py         # GraphState schema using TypedDict
    ├── prompts.py       # Modularized prompts (Planner & Synthesizer)
    ├── ingestion.py     # Document loaders and Chroma vector database setup
    ├── retriever.py     # Vector retrieval and Live DuckDuckGo search wrappers
    ├── planner.py       # Orchestrates LLM research plan creation
    ├── researcher.py    # Merges vector DB evidence and web search results
    └── synthesizer.py   # Compiles accumulated insights into structured reports
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have configured the **centralized `.env`** file in the root folder of the repository workspace with your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 2. Install Dependencies
Navigate to this directory and install the required modules:
```bash
pip install -r requirements.txt
```

### 3. Run the Sandbox
Boot the interactive deep research console:
```bash
python app.py
```

---

## ⚖️ Strategic Advantage

| Feature | Traditional RAG | Deep Research RAG |
| :--- | :--- | :--- |
| **Search Depth** | Single query | **Multi-step, targeted sub-queries** |
| **Search Sources** | Vector Database only | **Hybrid (Vector DB + Live Web)** |
| **Reasoning Model** | Direct generation | **Structured planner & synthesizer agents** |
| **Report Structure** | Simple text response | **Formatted Analyst Report (Overview, Comparisons, Insights, Conclusions)** |
| **Failure Tolerance** | Fails on out-of-vocabulary terms | **Web search covers missing local context dynamically** |
