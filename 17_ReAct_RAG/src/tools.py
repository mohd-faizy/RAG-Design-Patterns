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
# VECTOR RETRIEVER
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
docs = load_documents()
texts = [
    doc.page_content
    for doc in docs
]

tokenized_docs = [
    text.split()
    for text in texts
]

bm25 = BM25Okapi(
    tokenized_docs
)


# --------------------------------
# VECTOR SEARCH TOOL
# --------------------------------
@tool
def vector_search(query: str):
    """
    Semantic vector retrieval tool.
    """
    results = vector_retriever.invoke(query)

    return "\n\n".join([
        doc.page_content
        for doc in results
    ])


# --------------------------------
# BM25 SEARCH TOOL
# --------------------------------
@tool
def bm25_search(query: str):
    """
    Keyword BM25 retrieval tool.
    """
    tokenized_query = query.split()

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
    Web search retrieval tool.
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


TOOLS = [
    vector_search,
    bm25_search,
    web_search
]
