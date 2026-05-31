from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.ingestion import load_documents
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")

# Initialize dense vector store retriever
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Pre-load raw docs and build BM25 index
_docs = load_documents()
_texts = [doc.page_content for doc in _docs]
_tokenized = [t.split() for t in _texts]
bm25 = BM25Okapi(_tokenized)

def vector_retrieve(query: str) -> list[Document]:
    """Performs dense semantic vector retrieval from ChromaDB."""
    return retriever.invoke(query)

def hybrid_retrieve(query: str) -> list[Document]:
    """Combines dense vector retrieval with BM25 keyword matching."""
    vector_docs = retriever.invoke(query)
    
    # BM25 keyword retrieval
    scores = bm25.get_scores(query.split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    bm25_docs = [_docs[i] for i in top_indices]
    
    # Deduplicate by content and merge
    seen = set()
    merged = []
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged

def web_retrieve(query: str) -> list[str]:
    """Performs web search via DuckDuckGo and returns raw text snippets."""
    print(f"\n[Adaptive RAG] Running web search for: '{query}'...")
    outputs = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            if results:
                for r in results:
                    body = r.get("body", "")
                    if body:
                        outputs.append(body)
    except Exception as e:
        print(f"[Adaptive RAG] Web search error: {e}")
    return outputs
