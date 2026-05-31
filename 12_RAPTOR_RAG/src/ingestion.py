from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.tree_builder import build_recursive_tree


def build_raptor_tree():
    loader = TextLoader("data/sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(
        documents
    )

    texts = [
        doc.page_content
        for doc in docs
    ]

    embeddings_model = (
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
    )

    tree = build_recursive_tree(
        texts,
        embeddings_model
    )

    return tree
