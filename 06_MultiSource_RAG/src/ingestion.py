from pathlib import Path
import json
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")


def load_documents():
    docs = []

    # Dynamically resolve data directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = str(Path(__file__).resolve().parent.parent.parent / "_data")

    # 1. Load Text Source
    txt_path = os.path.join(data_dir, "source.txt")
    if os.path.exists(txt_path):
        print("Loading text source...")
        loader = TextLoader(txt_path, encoding="utf-8")
        docs.extend(loader.load())

    # 2. Load PDF Source
    pdf_path = os.path.join(data_dir, "source.pdf")
    if os.path.exists(pdf_path):
        try:
            print("Loading PDF source...")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[PDF Loader Warning] Failed to parse PDF: {e}")

    # 3. Load JSON Source
    json_path = os.path.join(data_dir, "source.json")
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

    # 4. Load Markdown Source
    md_path = os.path.join(data_dir, "source.md")
    if os.path.exists(md_path):
        try:
            print("Loading Markdown source...")
            loader = TextLoader(md_path, encoding="utf-8")
            md_docs = loader.load()
            for doc in md_docs:
                doc.metadata["source_type"] = "markdown"
            docs.extend(md_docs)
        except Exception as e:
            print(f"[Markdown Loader Warning] Failed to parse Markdown: {e}")

    # 5. Load CSV Source
    csv_path = os.path.join(data_dir, "source.csv")
    if os.path.exists(csv_path):
        try:
            print("Loading CSV source...")
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = f"Concept: {row.get('Concept', '')}. Topic: {row.get('Topic', '')}. Description: {row.get('Description', '')}."
                    meta = {"topic": row.get("Topic", ""), "source_type": "csv", "year": row.get("Year", "")}
                    docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"[CSV Loader Warning] Failed to parse CSV: {e}")

    # 6. Load HTML Source
    html_path = os.path.join(data_dir, "source.html")
    if os.path.exists(html_path):
        try:
            print("Loading HTML source...")
            import re
            with open(html_path, "r", encoding="utf-8") as f:
                html_text = f.read()
            # Basic tag stripping for lightweight loading
            clean_text = re.sub(r'<[^>]+>', ' ', html_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            docs.append(Document(page_content=clean_text, metadata={"source_type": "html"}))
        except Exception as e:
            print(f"[HTML Loader Warning] Failed to parse HTML: {e}")

    # 7. Load YAML Source
    yaml_path = os.path.join(data_dir, "source.yaml")
    if os.path.exists(yaml_path):
        try:
            print("Loading YAML source...")
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_text = f.read()
            docs.append(Document(page_content=yaml_text, metadata={"source_type": "yaml"}))
        except Exception as e:
            print(f"[YAML Loader Warning] Failed to parse YAML: {e}")

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
