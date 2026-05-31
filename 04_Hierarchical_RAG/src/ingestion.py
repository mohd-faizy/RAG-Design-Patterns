import uuid

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")


# -----------------------------------
# LOAD DOCUMENTS
# -----------------------------------
def load_documents():
    loader = TextLoader(str(BASE_DIR / "data" / "sample.txt"))
    return loader.load()


# -----------------------------------
# CREATE HIERARCHICAL CHUNKS
# -----------------------------------
def create_hierarchical_chunks():
    documents = load_documents()

    # -----------------------------
    # PARENT SPLITTER
    # -----------------------------
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # -----------------------------
    # CHILD SPLITTER
    # -----------------------------
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    parent_docs = parent_splitter.split_documents(documents)
    child_docs = []
    parent_map = {}

    # -----------------------------
    # CREATE PARENT-CHILD RELATION
    # -----------------------------
    for parent_doc in parent_docs:
        parent_id = str(uuid.uuid4())
        parent_map[parent_id] = parent_doc.page_content

        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child_docs.append(child)

    return child_docs, parent_map


# -----------------------------------
# CREATE VECTORSTORE
# -----------------------------------
def create_vectorstore(child_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    # vectorstore.persist()  # Omitted as modern versions of Chroma DB automatically persist data
