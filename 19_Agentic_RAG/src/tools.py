import re
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS

from src.ingestion import load_documents
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")

# --------------------------------
# VECTOR SEARCH
# --------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------
# BM25
# --------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercases and strips all non-alphanumeric characters for clean indexing."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()

docs = load_documents()
texts = [
    doc.page_content
    for doc in docs
]

tokenized_docs = [
    tokenize(text)
    for text in texts
]

bm25 = BM25Okapi(
    tokenized_docs
)


# --------------------------------
# VECTOR TOOL
# --------------------------------
@tool
def vector_search(query: str):
    """
    Semantic vector retrieval tool. Queries the local vector database.
    """
    results = vector_retriever.invoke(query)

    return "\n\n".join([
        doc.page_content
        for doc in results
    ])


# --------------------------------
# BM25 TOOL
# --------------------------------
@tool
def bm25_search(query: str):
    """
    Keyword BM25 retrieval tool. Performs lexical keyword search on local database.
    """
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    results = [
        docs[i].page_content
        for i in top_indices
    ]

    return "\n\n".join(results)


# --------------------------------
# WEB SEARCH TOOL
# --------------------------------
@tool
def web_search(query: str):
    """
    Web search retrieval tool. Searches DuckDuckGo for public and real-time facts.
    """
    outputs = []

    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=3
        )

        for r in results:
            body = r.get("body", "")
            if body:
                outputs.append(body)

    return "\n\n".join(outputs)
