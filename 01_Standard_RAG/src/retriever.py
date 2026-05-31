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
    
    # Retrieve the top 3 closest chunks matching the user query
    return vectorstore.as_retriever(search_kwargs={"k": 3})
