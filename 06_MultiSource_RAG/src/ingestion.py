import json
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.abspath(os.path.join(current_dir, "..", "chroma_db"))


def load_documents():
    docs = []

    # Dynamically resolve data directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))

    # 1. Load Text Source
    txt_path = os.path.join(data_dir, "source_txt.txt")
    if os.path.exists(txt_path):
        print("Loading text source...")
        loader = TextLoader(txt_path, encoding="utf-8")
        docs.extend(loader.load())

    # 2. Load PDF Source
    pdf_path = os.path.join(data_dir, "source_pdf.pdf")
    if os.path.exists(pdf_path):
        try:
            print("Loading PDF source...")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[PDF Loader Warning] Failed to parse PDF: {e}")

    # 3. Load JSON Source
    json_path = os.path.join(data_dir, "source_json.json")
    if os.path.exists(json_path):
        try:
            print("Loading JSON source...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    text = item.get("text", "")
                    meta = item.get("metadata", {})
                    if text:
                        docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"[JSON Loader Warning] Failed to parse JSON: {e}")

    print(f"Total raw source documents loaded: {len(docs)}")

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
