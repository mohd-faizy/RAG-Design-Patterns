from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    # Reranker RAG retrieves a larger chunk (k=10) to let the cross-encoder filter
    return vectorstore.as_retriever(search_kwargs={"k": 10})

def retrieve_docs(query: str, retriever) -> list:
    """Retrieves initial candidate documents (Top-K) from vector store."""
    return retriever.invoke(query)
