# Hybrid RAG with LangGraph + Groq + BM25 + Vector Search

This implementation combines:

* Semantic Search (Embeddings)
* Keyword Search (BM25)
* Reciprocal Rank Fusion (RRF)
* LangGraph orchestration
* Groq LLM
* HuggingFace embeddings
* ChromaDB vector store

This is a proper **Hybrid RAG architecture** used in production systems.

---

# What Hybrid RAG Solves

Pure vector search often fails for:

* Exact keywords
* Error codes
* IDs
* Acronyms
* Rare terms

BM25 solves lexical matching while embeddings solve semantic similarity.

Hybrid retrieval combines both.

---

# Architecture

```text
User Query
    │
    ▼
┌──────────────┐
│ Hybrid Search│
└──────┬───────┘
       │
 ┌─────┴─────┐
 ▼           ▼
BM25     Vector DB
 ▼           ▼
Keyword   Semantic
Results    Results
 └─────┬─────┘
       ▼
   RRF Fusion
       ▼
   Top Context
       ▼
      Groq
       ▼
 Final Answer
```

---

# PROJECT STRUCTURE

```bash
hybrid-rag/
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
    ├── fusion.py
    ├── graph.py
    ├── prompts.py
    └── state.py
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
numpy
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
LangGraph is used for orchestrating agentic workflows.

Groq provides ultra-fast inference for LLM applications.

Hybrid RAG combines vector search with keyword search.

BM25 is a lexical retrieval algorithm.

Embeddings capture semantic similarity between documents.
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


def load_and_split_documents():

    loader = TextLoader("data/sample.txt")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    return docs


def create_vector_store(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    # In modern versions of Chroma DB, persist is handled automatically and calling .persist() is deprecated/raises error.
    # vectorstore.persist()

    return vectorstore
```

---

# 5. Reciprocal Rank Fusion (RRF)

## src/fusion.py

```python
from collections import defaultdict


def reciprocal_rank_fusion(results, k=60):

    fused_scores = defaultdict(float)
    doc_map = {}

    for docs in results:

        for rank, doc in enumerate(docs):
            content = doc.page_content
            fused_scores[content] += 1 / (rank + k)
            if content not in doc_map:
                doc_map[content] = doc

    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return Document objects instead of raw strings to avoid downstream crashes in generator nodes
    return [doc_map[content] for content, _ in reranked]
```

RRF is commonly used in modern retrieval systems.

Formula:

RRF(d)=\sum_{r\in R}\frac{1}{k+r(d)}

Where:

* (R) = ranking systems
* (r(d)) = rank of document
* (k) = smoothing constant

---

# 6. Hybrid Retriever

## src/retriever.py

```python
from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.ingestion import load_and_split_documents
from src.fusion import reciprocal_rank_fusion

CHROMA_PATH = "chroma_db"


class HybridRetriever:

    def __init__(self):

        self.docs = load_and_split_documents()

        self.texts = [doc.page_content for doc in self.docs]

        tokenized_docs = [
            text.split() for text in self.texts
        ]

        # BM25
        self.bm25 = BM25Okapi(tokenized_docs)

        # VECTOR DB
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    def retrieve(self, query):

        # ---------------------
        # BM25 SEARCH
        # ---------------------
        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:3]

        bm25_docs = [
            self.docs[i]
            for i in bm25_top_indices
        ]

        # ---------------------
        # VECTOR SEARCH
        # ---------------------
        vector_docs = self.vector_retriever.invoke(query)

        # ---------------------
        # RRF FUSION
        # ---------------------
        fused = reciprocal_rank_fusion([
            bm25_docs,
            vector_docs
        ])

        return fused[:3]
```

---

# 7. Prompt Template

## src/prompts.py

```python
RAG_PROMPT = """
You are a helpful AI assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
```

---

# 8. Graph State

## src/state.py

```python
from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):

    question: str

    context: List[Document]

    answer: str
```

---

# 9. LangGraph Workflow

## src/graph.py

```python
from typing import Dict

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq

from src.state import GraphState
from src.retriever import HybridRetriever
from src.prompts import RAG_PROMPT

retriever = HybridRetriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# -------------------------
# RETRIEVE NODE
# -------------------------
def retrieve(state: GraphState) -> Dict:

    question = state["question"]

    docs = retriever.retrieve(question)

    return {
        "context": docs
    }


# -------------------------
# GENERATE NODE
# -------------------------
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


# -------------------------
# BUILD GRAPH
# -------------------------
def build_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "retrieve",
        retrieve
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
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()
```

---

# 10. Main Application

## app.py

```python
from dotenv import load_dotenv

from src.ingestion import (
    load_and_split_documents,
    create_vector_store
)

from src.graph import build_graph

load_dotenv()


def main():

    # -----------------------
    # CREATE VECTOR STORE
    # -----------------------
    docs = load_and_split_documents()

    create_vector_store(docs)

    # -----------------------
    # BUILD GRAPH
    # -----------------------
    app = build_graph()

    while True:

        question = input(
            "\nAsk Question: "
        )

        if question.lower() == "exit":
            break

        result = app.invoke({
            "question": question
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

# Example Queries

```text
Ask Question:
What is BM25?
```

```text
Answer:
BM25 is a lexical retrieval algorithm.
```

---

# Why Hybrid RAG Is Better

| Retrieval Type | Strength               |
| -------------- | ---------------------- |
| BM25           | Exact keyword match    |
| Vector Search  | Semantic understanding |
| Hybrid         | Best of both           |

---

# Production Improvements

## Add Cross Encoder Reranking

Use:

* BAAI/bge-reranker-base
* Cohere Rerank

Improves retrieval quality significantly.

---

# Better Embeddings

| Model                        | Quality   |
| ---------------------------- | --------- |
| BAAI/bge-small-en-v1.5       | Very Good |
| intfloat/e5-base-v2          | Excellent |
| nomic-ai/nomic-embed-text-v1 | Excellent |

---

# Production Vector DBs

| Vector DB | Best For               |
| --------- | ---------------------- |
| Chroma    | Local dev              |
| Qdrant    | Open-source production |
| Weaviate  | Hybrid search          |
| Pinecone  | Managed cloud          |

---

# Recommended Next Steps

After Hybrid RAG:

1. Parent Document Retrieval
2. Multi-Query Retrieval
3. Self-RAG
4. Corrective RAG (CRAG)
5. Agentic RAG
6. Graph RAG
7. Multi-Agent Systems

---

# Useful Resources

* [LangGraph Docs](https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com)
* [Groq Docs](https://console.groq.com/docs?utm_source=chatgpt.com)
* [Sentence Transformers](https://www.sbert.net/?utm_source=chatgpt.com)
* [ChromaDB Docs](https://docs.trychroma.com/?utm_source=chatgpt.com)
* [BM25 Paper](https://en.wikipedia.org/wiki/Okapi_BM25?utm_source=chatgpt.com)
