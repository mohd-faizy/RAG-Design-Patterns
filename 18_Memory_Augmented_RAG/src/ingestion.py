import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")

def load_documents():
    loader = TextLoader(str(BASE_DIR.parent / "_data" / "source.txt"))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    # Check if DB already exists and has documents to avoid duplication
    if os.path.exists(CHROMA_PATH):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
            count = vectorstore._collection.count()
            if count > 0:
                print(f"Vector database already exists and contains {count} documents. Skipping recreation.")
                return
            print("Vector database exists but is empty. Loading documents...")
        except Exception:
            print("Vector database exists but failed to load. Recreating...")

    print("Initializing ingestion pipeline...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    # Auto-persisted in modern Chroma — no persist() needed
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector DB created and saved locally successfully!")
