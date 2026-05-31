from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------
# VECTOR RETRIEVAL
# --------------------------------
def retrieve_docs(query):
    try:
        docs = retriever.invoke(query)
        return [
            d.page_content
            for d in docs
        ]
    except Exception as e:
        print(f"[Vector Search Warning] {e}")
        return []


# --------------------------------
# WEB SEARCH
# --------------------------------
def web_search(query):
    outputs = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=3
            )
            if results:
                for r in results:
                    body = r.get("body", "")
                    if body:
                        outputs.append(body)
    except Exception as e:
        print(f"[Web Search Warning] DuckDuckGo search failed: {e}")
    return outputs
