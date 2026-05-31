import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def create_vector_store():
    # Only build it if it doesn't already exist to save compute time
    if os.path.exists(CHROMA_PATH):
        print("Vector database directory already exists. Skipping recreation.")
        return

    print("Initializing ingestion pipeline...")
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    # Split long text documents into manageable pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Free local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Initialize and save locally
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector DB created and saved locally successfully!")
