import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def load_documents():
    loader = TextLoader("data/sample.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    # Only build it if it doesn't already exist to save compute time
    if os.path.exists(CHROMA_PATH):
        print("Vector database directory already exists. Skipping recreation.")
        return

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
