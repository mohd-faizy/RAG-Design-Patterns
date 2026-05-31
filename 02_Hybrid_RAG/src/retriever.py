from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.ingestion import load_and_split_documents
from src.fusion import reciprocal_rank_fusion
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # repo module root

CHROMA_PATH = str(BASE_DIR / "chroma_db")


class HybridRetriever:
    def __init__(self):
        self.docs = load_and_split_documents()
        self.texts = [doc.page_content for doc in self.docs]

        tokenized_docs = [
            text.split() for text in self.texts
        ]

        # BM25
        self.bm25 = BM25Okapi(tokenized_docs)

        # VECTOR DB
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    def retrieve(self, query):
        # ---------------------
        # BM25 SEARCH
        # ---------------------
        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:3]

        bm25_docs = [
            self.docs[i]
            for i in bm25_top_indices
        ]

        # ---------------------
        # VECTOR SEARCH
        # ---------------------
        vector_docs = self.vector_retriever.invoke(query)

        # ---------------------
        # RRF FUSION
        # ---------------------
        fused = reciprocal_rank_fusion([
            bm25_docs,
            vector_docs
        ])

        return fused[:3]
