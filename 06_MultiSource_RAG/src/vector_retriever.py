from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")


class VectorRetriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.retriever = (
            vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        )

    def retrieve(self, query):
        return self.retriever.invoke(query)
