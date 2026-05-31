from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_docs(query: str, retriever) -> list:
    """Retrieves the top-k most relevant documents from the knowledge base."""
    return retriever.invoke(query)
