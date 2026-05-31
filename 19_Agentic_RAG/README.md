# Agentic RAG

A stateful, zero-cost, and production-structured implementation of the **Autonomous Agentic Retrieval-Augmented Generation (Agentic RAG)** pattern.

---

## 📖 What is Agentic RAG?

Agentic RAG represents the evolution of RAG from a passive retrieval pipeline into a **team of collaborative, specialized agents** coordinated by a central planner — the most autonomous RAG architecture before full deep research systems.

Standard RAG pipelines query a static retriever, inject results into a prompt, and generate an answer in a linear, blind execution. They lack the capacity to make planning choices, dynamically route queries based on domain, reflect on generation quality, or self-correct errors.

**Agentic RAG** transforms this by introducing multiple specialized agents, each with a distinct role:

1.  **Planner Agent**: Analyzes user questions to determine the optimal retrieval strategy and which specialized agents to deploy.
2.  **Specialized Retrieval Agents**: Individual agents dedicated to querying a single source — Vector DB, BM25 keyword index, or Web search.
3.  **Fusion Agent**: Merges results from multiple retrieval agents using Reciprocal Rank Fusion (RRF).
4.  **Reflection Agent**: Critiques the generated answer's quality and relevance.
5.  **Hallucination Agent**: Verifies that the answer is supported by retrieved evidence. If verification fails, it triggers a retry loop.

The key difference from ReAct RAG is that Agentic RAG uses **multiple specialized agents working as a team**, while ReAct uses a single agent with multiple tools.

---

## 🏗️ Architecture & State Workflow

### 1. Agentic Decision Flow
The system processes questions through planning, routing, tool execution, and reflective critique:

```mermaid
graph TD
    Question([Question]) --> Planning["Planning {Planner Agent}"]
    Planning --> ToolSelection["Tool Selection {Vector / BM25 / Web}"]
    ToolSelection --> Retrieval[Retrieval]
    Retrieval --> Reasoning["Reasoning {Groq Llama-3.3}"]
    Reasoning --> Reflection["Reflection {Reflection Agent}"]
    Reflection --> Correction[Correction / Retry]

    style Question fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style Reflection fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### 2. Multi-Agent RAG Architecture
Heterogeneous resources are coordinated under a fanned-out team of specialized agents:

```mermaid
graph TD
    Query([User Query]) --> Planner["Planner Agent {Groq Llama-3.3}"]
    
    Planner --> Vector["Vector Agent {ChromaDB}"]
    Planner --> Web["Web Agent {DuckDuckGo}"]
    Planner --> BM25["BM25 Agent {rank-bm25}"]
    
    Vector --> Fusion["Fusion Agent {Reciprocal Rank Fusion}"]
    Web --> Fusion
    BM25 --> Fusion
    
    Fusion --> Reflection["Reflection Agent {Groq Llama-3.3}"]
    Reflection --> Hallucination["Hallucination Agent {Groq Llama-3.3}"]
    Hallucination --> FinalAnswer([Final Answer])

    style Query fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalAnswer fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Fusion fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

---

## ⚙️ Key Components

| Component | File | Role |
| :--- | :--- | :--- |
| **State Schema** | `src/state.py` | Defines `GraphState` TypedDict carrying question, plan, agent outputs, fused context, and answer |
| **Document Ingestion** | `src/ingestion.py` | Loads and chunks documents, builds ChromaDB vector index and BM25 in-memory index |
| **Tool Definitions** | `src/tools.py` | Defines three LangChain-compatible tools: Vector Search (ChromaDB), BM25 Search (rank-bm25), and Web Search (DuckDuckGo) |
| **Agent Definitions** | `src/agents.py` | Implements the multi-agent system: **Planner Agent** (strategy selection), **Specialized Retrieval Agents** (one per search type), and **Reflection Agent** (quality critique and hallucination detection) |
| **Prompt Templates** | `src/prompts.py` | System prompts for the planner, retrieval agents, and reflection/hallucination agents |
| **Workflow Graph** | `src/graph.py` | LangGraph state-routing workflow compiler coordinating planning → fanout retrieval → fusion → reflection → output |
| **Application Entry** | `app.py` | Interactive CLI loop for querying the multi-agent pipeline |

---

## 🔄 How It Works

1. **Document Ingestion** — Documents are loaded, chunked, and indexed into ChromaDB (vector search) and an in-memory BM25 index (keyword search).

2. **Planning** — The Planner Agent analyzes the user's question and formulates a retrieval strategy: which agents to deploy, which sources to query, and in what priority order.

3. **Agent Dispatch** — Based on the plan, specialized retrieval agents are dispatched:
   - **Vector Agent**: Queries ChromaDB for semantically similar documents.
   - **BM25 Agent**: Queries the keyword index for exact term matches.
   - **Web Agent**: Queries DuckDuckGo for real-time public information.

4. **Result Fusion** — The Fusion Agent merges results from all dispatched agents using Reciprocal Rank Fusion (RRF), producing a unified, quality-ranked context set.

5. **Answer Generation** — The fused context is sent to Groq's `llama-3.3-70b-versatile` for answer synthesis.

6. **Reflection & Verification** — The Reflection Agent critiques the generated answer for quality and completeness. The Hallucination Agent verifies that all claims are supported by the retrieved evidence.

7. **Correction Loop** — If the answer fails reflection or hallucination checks, the system can retry with adjusted parameters (modified query, different agent combination).

---

## 📁 Project Structure

```bash
19_Agentic_RAG/
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
    ├── tools.py         # Vector, BM25, and Web search tool definitions
    ├── agents.py        # Planner, specialized vector/BM25/web, and reflection agents
    └── graph.py         # LangGraph state-routing workflow compiler
```

---

## ✅ Advantages

- **Multi-Agent Coordination**: Specialized agents focus on their strengths, while the planner orchestrates the overall strategy.
- **Quality Assurance Pipeline**: Reflection and hallucination agents provide multi-layered quality control before answers reach the user.
- **Dynamic Strategy Selection**: The planner adapts the retrieval strategy based on the query, not a fixed pipeline.
- **Self-Correcting**: Failed verification triggers retry loops, improving answer quality through iteration.
- **Scalable Architecture**: New agent types (e.g., SQL agent, API agent) can be added without restructuring the existing pipeline.

## ⚠️ Limitations

- **High Token Consumption**: Multiple agents (planner + retrievers + reflection + hallucination) each require LLM calls, significantly increasing token usage per query.
- **Increased Latency**: The multi-agent pipeline with reflection loops is substantially slower than simple RAG patterns.
- **Planning Accuracy**: The quality of the entire pipeline depends on the planner agent's ability to correctly analyze the query and select the right agents.
- **Complex Debugging**: Multi-agent interactions are harder to trace and debug than single-path pipelines.
- **Coordination Overhead**: Inter-agent communication and state management add engineering complexity.

---

## 🎯 Ideal Use Cases

- **Enterprise AI Platforms** — Large-scale knowledge systems requiring multiple search strategies with quality guarantees.
- **Autonomous Research Assistants** — Tools that independently plan and execute multi-step information gathering.
- **Mission-Critical QA** — Applications where answer quality must be verified before delivery (medical, legal, financial).
- **Multi-Source Intelligence** — Scenarios requiring simultaneous querying of internal databases, keyword indexes, and live web sources.
- **AI Copilots** — Developer or analyst copilots that need to dynamically decide how to best answer diverse questions.

---

## ⚖️ Comparison with Standard RAG

| Feature | Standard RAG | Agentic RAG |
| :--- | :--- | :--- |
| **Pipeline Type** | Linear, pre-defined | **Dynamic, self-directed** |
| **Routing Decisions** | Static | **LLM Planner Agent routing** |
| **Task Delegation** | Monolithic retrieve | **Specialized agent delegation** |
| **Grounding Checks** | None | **Stateful Reflection + Hallucination agents** |
| **Failsafe Correction** | None | **Self-guided correction retry logic** |
| **Scalability** | Fixed components | **Extensible agent team** |
