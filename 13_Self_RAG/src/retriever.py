from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.ingestion import load_documents

CHROMA_PATH = "chroma_db"


class HybridRetriever:
    def __init__(self):
        self.docs = load_documents()
        texts = [
            doc.page_content
            for doc in self.docs
        ]

        tokenized_docs = [
            text.split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        self.vector_retriever = (
            vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        )

    def retrieve(self, query):
        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:2]

        bm25_docs = [
            self.docs[i]
            for i in top_indices
        ]

        vector_docs = self.vector_retriever.invoke(
            query
        )

        combined = bm25_docs + vector_docs
        unique_docs = []
        seen = set()

        for doc in combined:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs[:4]
