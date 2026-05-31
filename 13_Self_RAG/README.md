# Self-RAG with LangGraph + Groq + Hybrid Retrieval

Self-RAG adds **reflection and self-correction** to standard RAG.

The model:

1. Retrieves documents
2. Grades relevance
3. Detects hallucinations
4. Rewrites queries if retrieval is poor
5. Generates improved answers

This architecture is inspired by the Self-RAG paper:

[Self-RAG Paper](https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com)

and production patterns from:

* [LangGraph RAG Tutorials](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/?utm_source=chatgpt.com)
* [LangGraph Docs](https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com)

---

# What Self-RAG Adds

Standard RAG:

```text
Query → Retrieve → Generate
```

Self-RAG:

```text
Query
  ↓
Retrieve
  ↓
Grade Documents
  ↓
Good?
 ├── YES → Generate
 └── NO
        ↓
   Rewrite Query
        ↓
    Retrieve Again
        ↓
     Generate
        ↓
 Hallucination Check
        ↓
  Final Response
```

---

# PROJECT STRUCTURE

```bash
self-rag/
│
├── app.py
├── requirements.txt
├── .env
│
├── data/
│   └── sample.txt
│
└── src/
    ├── ingestion.py
    ├── retriever.py
    ├── prompts.py
    ├── state.py
    ├── graders.py
    ├── query_rewriter.py
    └── graph.py
```

---

# 1. Install Dependencies

## requirements.txt

```txt
langchain
langgraph
langchain-community
langchain-huggingface
langchain-groq
chromadb
sentence-transformers
rank-bm25
python-dotenv
```

Install:

```bash
pip install -r requirements.txt
```

---

# 2. Environment Variables

## .env

```env
GROQ_API_KEY=your_groq_api_key
```

Get free key:

[Groq Console](https://console.groq.com/keys?utm_source=chatgpt.com)

---

# 3. Sample Data

## data/sample.txt

```txt
LangGraph is a framework for building stateful AI workflows.

Self-RAG improves retrieval quality using reflection.

BM25 is useful for keyword matching.

Embeddings capture semantic relationships.

Groq provides fast inference for open-source LLMs.
```

---

# 4. Ingestion Pipeline

## src/ingestion.py

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"


def load_documents():

    loader = TextLoader("data/sample.txt")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


def create_vectorstore(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    # In modern versions of Chroma DB, persist is handled automatically and calling .persist() is deprecated/raises error.
    # vectorstore.persist()
```

---

# 5. Hybrid Retriever

## src/retriever.py

```python
from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.ingestion import load_documents

CHROMA_PATH = "chroma_db"


class HybridRetriever:

    def __init__(self):

        self.docs = load_documents()

        texts = [
            doc.page_content
            for doc in self.docs
        ]

        tokenized_docs = [
            text.split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.vector_retriever = (
            vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        )

    def retrieve(self, query):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:2]

        bm25_docs = [
            self.docs[i]
            for i in top_indices
        ]

        vector_docs = self.vector_retriever.invoke(
            query
        )

        combined = bm25_docs + vector_docs

        unique_docs = []

        seen = set()

        for doc in combined:

            if doc.page_content not in seen:

                unique_docs.append(doc)

                seen.add(doc.page_content)

        return unique_docs[:4]
```

---

# 6. Prompts

## src/prompts.py

```python
RAG_PROMPT = """
You are a helpful AI assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

DOC_GRADER_PROMPT = """
You are a grader.

Evaluate whether the document is relevant to the user question.

Question:
{question}

Document:
{document}

Answer only:
yes or no
"""

HALLUCINATION_PROMPT = """
You are checking whether an answer is grounded in facts.

Context:
{context}

Answer:
{answer}

Is the answer supported by the context?

Answer only:
yes or no
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query to improve retrieval quality.

Original Query:
{question}

Improved Query:
"""
```

---

# 7. State

## src/state.py

```python
from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):

    question: str

    rewritten_question: str

    context: List[Document]

    answer: str

    retries: int
```

---

# 8. Graders

## src/graders.py

```python
from langchain_groq import ChatGroq

from src.prompts import (
    DOC_GRADER_PROMPT,
    HALLUCINATION_PROMPT
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def grade_documents(question, docs):

    filtered_docs = []

    for doc in docs:

        prompt = DOC_GRADER_PROMPT.format(
            question=question,
            document=doc.page_content
        )

        response = llm.invoke(prompt)

        result = response.content.strip().lower()

        if "yes" in result:

            filtered_docs.append(doc)

    return filtered_docs


def check_hallucination(context, answer):

    context_text = "\n\n".join([
        doc.page_content
        for doc in context
    ])

    prompt = HALLUCINATION_PROMPT.format(
        context=context_text,
        answer=answer
    )

    response = llm.invoke(prompt)

    return response.content.strip().lower()
```

---

# 9. Query Rewriter

## src/query_rewriter.py

```python
from langchain_groq import ChatGroq

from src.prompts import QUERY_REWRITE_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def rewrite_query(question):

    prompt = QUERY_REWRITE_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    return response.content.strip()
```

---

# 10. LangGraph Workflow

## src/graph.py

```python
from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.state import GraphState

from src.retriever import HybridRetriever

from src.prompts import RAG_PROMPT

from src.graders import (
    grade_documents,
    check_hallucination
)

from src.query_rewriter import rewrite_query

retriever = HybridRetriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# RETRIEVE
# --------------------------------
def retrieve(state: GraphState) -> Dict:

    question = (
        state.get("rewritten_question")
        or state["question"]
    )

    docs = retriever.retrieve(question)

    return {
        "context": docs
    }


# --------------------------------
# GRADE DOCUMENTS
# --------------------------------
def grade(state: GraphState) -> Dict:

    question = state["question"]

    docs = state["context"]

    filtered_docs = grade_documents(
        question,
        docs
    )

    return {
        "context": filtered_docs
    }


# --------------------------------
# DECISION
# --------------------------------
def decide_to_generate(state):

    docs = state["context"]

    retries = state.get("retries", 0)

    if len(docs) == 0 and retries < 2:

        return "rewrite"

    return "generate"


# --------------------------------
# REWRITE QUERY
# --------------------------------
def rewrite(state: GraphState) -> Dict:

    rewritten = rewrite_query(
        state["question"]
    )

    retries = state.get("retries", 0)

    return {
        "rewritten_question": rewritten,
        "retries": retries + 1
    }


# --------------------------------
# GENERATE
# --------------------------------
def generate(state: GraphState) -> Dict:

    question = state["question"]

    docs = state["context"]

    context_text = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = RAG_PROMPT.format(
        context=context_text,
        question=question
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------------------
# CHECK HALLUCINATION
# --------------------------------
def check_answer(state):

    result = check_hallucination(
        state["context"],
        state["answer"]
    )

    if "yes" in result:

        return "useful"

    return "not_useful"


# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph():

    workflow = StateGraph(
        GraphState
    )

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "grade",
        grade
    )

    workflow.add_node(
        "rewrite",
        rewrite
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "grade"
    )

    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )

    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    workflow.add_conditional_edges(
        "generate",
        check_answer,
        {
            "useful": END,
            "not_useful": "rewrite"
        }
    )

    return workflow.compile()
```

---

# 11. Main App

## app.py

```python
from dotenv import load_dotenv

from src.ingestion import (
    load_documents,
    create_vectorstore
)

from src.graph import build_graph

load_dotenv()


def main():

    docs = load_documents()

    create_vectorstore(docs)

    app = build_graph()

    while True:

        question = input(
            "\nAsk Question: "
        )

        if question.lower() == "exit":

            break

        result = app.invoke({
            "question": question,
            "retries": 0
        })

        print("\nAnswer:\n")

        print(result["answer"])


if __name__ == "__main__":
    main()
```

---

# Run Project

```bash
python app.py
```

---

# Why Self-RAG Is Powerful

| Capability              | Standard RAG | Self-RAG |
| ----------------------- | ------------ | -------- |
| Semantic Retrieval      | ✅            | ✅        |
| Hybrid Retrieval        | ❌            | ✅        |
| Query Rewriting         | ❌            | ✅        |
| Reflection              | ❌            | ✅        |
| Hallucination Detection | ❌            | ✅        |
| Self-Correction         | ❌            | ✅        |

---

# Self-RAG Loop

```text
Retrieve
   ↓
Grade Docs
   ↓
Rewrite Query
   ↓
Retrieve Again
   ↓
Generate
   ↓
Hallucination Check
```

---

# Recommended Next Upgrades

After Self-RAG:

1. Corrective RAG (CRAG)
2. Adaptive RAG
3. Agentic RAG
4. Graph RAG
5. Multi-Agent RAG
6. Deep Research Agents
7. Tool-Calling Agents

---

# Advanced Production Production Features

## Add Reranker

Use:

* BAAI/bge-reranker-base
* Cohere Rerank

---

## Add Memory

Use:

* Redis
* Postgres
* LangGraph Memory

---

## Add Web Search

Use:

* Tavily
* DuckDuckGo
* SerpAPI

---

# Useful Resources

* [Self-RAG Paper](https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com)
* [LangGraph Docs](https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com)
* [Groq Docs](https://console.groq.com/docs?utm_source=chatgpt.com)
* [Sentence Transformers](https://www.sbert.net/?utm_source=chatgpt.com)
* [ChromaDB Docs](https://docs.trychroma.com/?utm_source=chatgpt.com)
