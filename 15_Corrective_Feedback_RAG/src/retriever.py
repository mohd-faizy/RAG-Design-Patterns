from duckduckgo_search import DDGS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve_docs(query: str, retriever) -> list[Document]:
    return retriever.invoke(query)

def web_search(query: str) -> list[Document]:
    print(f"\n[CRAG] Executing Web Search Fallback for query: '{query}'...")
    docs = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            if results:
                for r in results:
                    body = r.get("body", "")
                    if body:
                        docs.append(
                            Document(
                                page_content=body,
                                metadata={
                                    "source": "duckduckgo",
                                    "title": r.get("title", "Web search result")
                                }
                            )
                        )
    except Exception as e:
        print(f"[CRAG] Error during web search: {e}")
    return docs
