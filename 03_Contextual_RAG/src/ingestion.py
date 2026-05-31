from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.contextualizer import generate_document_context

CHROMA_PATH = "chroma_db"


def load_and_contextualize():
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    full_document_text = "\n".join([
        doc.page_content
        for doc in documents
    ])

    # -------------------------
    # DOCUMENT CONTEXT
    # -------------------------
    document_context = generate_document_context(full_document_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)
    contextualized_docs = []

    # -------------------------
    # ADD CONTEXT TO EACH CHUNK
    # -------------------------
    for doc in split_docs:
        enriched_text = f"""
        Document Context:
        {document_context}

        Chunk:
        {doc.page_content}
        """

        contextualized_docs.append(
            Document(
                page_content=enriched_text,
                metadata=doc.metadata
            )
        )

    return contextualized_docs


def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    # vectorstore.persist()  # Omitted as modern versions of Chroma DB automatically persist data

    return vectorstore
