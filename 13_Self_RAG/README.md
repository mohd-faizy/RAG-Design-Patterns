# Self-RAG

A stateful, zero-cost, and production-structured implementation of the **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** pattern.

---

## 📖 What is Self-RAG?

Self-RAG introduces the concept of **self-reflection and self-correction** into the RAG pipeline, transforming it from a blind "retrieve-and-generate" system into one that critically evaluates its own retrieval quality and generation accuracy.

Standard RAG architectures execute a fixed pipeline: **Retrieve → Generate**. If the initial retrieval is noisy, irrelevant, or empty, the model produces poor answers or hallucinations — with no mechanism to detect or correct the problem.

**Self-RAG** addresses this by adding two critical reflection gates:
1.  **Relevance Assessment**: Grades whether retrieved documents are semantically relevant to the user query. If the context is graded as insufficient, the pipeline automatically **rewrites the query** and retrieves fresh context.
2.  **Hallucination Check**: After generation, verifies whether the response is strictly supported by facts inside the matched documents. If hallucination is detected, the pipeline can loop back for additional retrieval.

This creates a **closed-loop feedback system** where retrieval and generation are continuously refined until quality thresholds are met:

```text
Retrieve → Grade → [Poor? Rewrite & Retry] → Generate → Verify → [Hallucinated? Retry] → Answer
```

---

## 🏗️ Architecture & State Workflow

```mermaid
graph TD
    Retrieve["Retrieve {Hybrid BM25 + ChromaDB}"] --> Grade["Grade Docs {Groq Llama-3.3 Grader}"]
    Grade --> Rewrite["Rewrite Query {Groq Llama-3.3 Rewriter}"]
    Rewrite --> RetrieveAgain["Retrieve Again {Hybrid BM25 + ChromaDB}"]
    RetrieveAgain --> Generate["Generate {Groq Llama-3.3 Generator}"]
    Generate --> Verify["Hallucination Check {Groq Llama-3.3 Checker}"]
    Verify --> FinalResponse([Final Response])

    style Retrieve fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style FinalResponse fill:#F43F5E,stroke:#9F1239,stroke-width:2px,color:#fff
    style Grade fill:#B45309,stroke:#78350F,stroke-width:2px,color:#fff
```

### Flow Breakdown
1.  **Retrieve**: Semantic vector + BM25 hybrid search retrieves relevant chunks.
2.  **Grade**: Groq `llama-3.3-70b-versatile` grades the text quality. If zero items are relevant and retries are under the threshold, it triggers **Query Rewriting** and fetches new documents.
3.  **Generate**: Generates a factual grounded response once sufficient context is located.
4.  **Verify**: Performs a self-hallucination check. If not supported, it rewrites the query and loops back to retrieve better sources.

---

## ⚙️ Key Components

| Component | File | Role |
| :--- | :--- | :--- |
| **State Schema** | `src/state.py` | Defines `GraphState` TypedDict carrying question, context, answer, grading results, and retry count |
| **Document Ingestion** | `src/ingestion.py` | Loads and chunks documents, builds the ChromaDB vector index |
| **Hybrid Retriever** | `src/retriever.py` | Combines BM25 keyword search and ChromaDB vector search for dual-engine retrieval |
| **Graders** | `src/graders.py` | LLM-powered reflection nodes: **Relevance Grader** (assesses document quality) and **Hallucination Grader** (verifies answer factual support) |
| **Query Rewriter** | `src/query_rewriter.py` | LLM agent that reformulates the user query to improve retrieval results when initial context is graded as insufficient |
| **Prompt Templates** | `src/prompts.py` | Evaluation prompt templates for relevance grading, hallucination checking, and fact-grounded generation |
| **Workflow Graph** | `src/graph.py` | LangGraph stategraph compiler with conditional routing for grading and hallucination check loops |
| **Application Entry** | `app.py` | Interactive CLI loop for querying the Self-RAG pipeline |

---

## 🔄 How It Works

1. **Document Ingestion** — Documents are loaded, chunked, and indexed into both ChromaDB and an in-memory BM25 index for hybrid retrieval.

2. **Hybrid Retrieval** — The user's query triggers both BM25 keyword search and vector semantic search, returning a combined set of candidate chunks.

3. **Relevance Grading** — Each retrieved document is evaluated by Groq LLM acting as a relevance grader. The grader outputs `yes` or `no` for each document based on semantic match quality.

4. **Conditional Rewrite** — If too few documents pass the relevance threshold (and retry limit hasn't been reached), the query is sent to the Query Rewriter. The LLM reformulates the question to be more specific or use alternative phrasing, and retrieval is attempted again.

5. **LLM Generation** — Once sufficient relevant context is available, the documents and query are compiled into a prompt for Groq's `llama-3.3-70b-versatile`.

6. **Hallucination Verification** — The generated answer is checked against the retrieved context by the Hallucination Grader. It verifies that every assertion in the answer is supported by the source documents.

7. **Response Delivery** — If the answer passes the hallucination check, it is delivered. If not, the pipeline can loop back for additional retrieval and regeneration.

---

## 📁 Project Structure

```bash
13_Self_RAG/
│
├── app.py               # Main CLI interactive loop entrypoint
├── requirements.txt     # Local project packages
│
│
└── src/
    ├── __init__.py      # Package initialization
    ├── state.py         # GraphState schema using TypedDict
    ├── prompts.py       # Evaluation prompt templates
    ├── ingestion.py     # Document parser and Chroma indexer
    ├── retriever.py     # Hybrid BM25 + Vector retriever
    ├── graders.py       # LLM relevance and hallucination reflection nodes
    ├── query_rewriter.py# Query rewriting agent node
    └── graph.py         # Stategraph compiler and router
```

---

## ✅ Advantages

- **Self-Healing Retrieval**: Automatically detects and corrects poor retrieval results by rewriting queries and retrying.
- **Hallucination Protection**: Post-generation verification catches unsupported claims before they reach the user.
- **Hybrid Search Foundation**: BM25 + Vector dual-engine retrieval provides robust initial coverage.
- **Bounded Retries**: Retry limits prevent infinite loops while allowing multiple correction attempts.
- **Transparent Reflection**: Grading decisions are visible in the output, making the self-correction process explainable.

## ⚠️ Limitations

- **Higher Latency**: Multiple LLM calls for grading, rewriting, and hallucination checking significantly increase response time.
- **Increased API Usage**: Each question may trigger 3-5+ LLM calls (grading + rewriting + generation + hallucination check), consuming more tokens.
- **Grader Accuracy**: The quality of self-correction depends on the LLM's ability to accurately assess relevance and detect hallucinations.
- **Retry Limit Trade-Off**: Too few retries may miss recoverable queries; too many retries waste API calls on fundamentally unanswerable questions.
- **Complexity**: The conditional routing logic is significantly more complex than linear RAG pipelines, making debugging harder.

---

## 🎯 Ideal Use Cases

- **High-Stakes QA Systems** — Medical, legal, or financial applications where hallucinated answers could cause harm.
- **Noisy Knowledge Bases** — Corpora with mixed-quality documents where initial retrieval frequently returns irrelevant results.
- **Mission-Critical Enterprise Search** — Production systems where answer accuracy is more important than response speed.
- **Compliance & Audit Systems** — Applications where every generated claim must be traceable to source evidence.
- **Conversational AI Assistants** — User-facing chatbots where incorrect answers directly impact user trust.

---

## ⚖️ Comparison with Standard RAG

| Feature | Standard RAG | Self-RAG |
| :--- | :--- | :--- |
| **Retrieval Validation** | ❌ None | **✅ LLM-based relevance grading** |
| **Query Rewriting** | ❌ None | **✅ Automatic reformulation on poor results** |
| **Hallucination Check** | ❌ None | **✅ Post-generation factual verification** |
| **Retrieval Strategy** | Single-pass | **Multi-pass with retry loops** |
| **Pipeline Type** | Fixed linear | **Adaptive conditional routing** |
| **Latency** | Low | Higher (multiple LLM reflection calls) |

### Core Grader Prompts

Self-RAG depends on two critical logical reflection gates:
1.  **Retrieval Sufficiency Grader**: Evaluates whether documents match the semantic context of the query (outputs `yes` or `no`).
2.  **Hallucination Grader**: Checks if the answer's assertions are backed by raw context values (outputs `yes` or `no`).
