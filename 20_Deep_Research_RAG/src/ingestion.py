import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")


def load_documents():
    loader = TextLoader(str(Path(__file__).resolve().parent.parent.parent / "_data" / "source.txt"))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(docs)


def create_vectorstore(docs):
    if os.path.exists(CHROMA_PATH):
        print("Vector database directory already exists. Skipping recreation.")
        return

    print("Initializing ingestion pipeline...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector DB created and saved locally successfully!")
