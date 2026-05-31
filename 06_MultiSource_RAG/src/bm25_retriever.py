from rank_bm25 import BM25Okapi

from src.ingestion import load_documents


class BM25Retriever:
    def __init__(self):
        self.docs = load_documents()
        self.texts = [
            doc.page_content
            for doc in self.docs
        ]

        tokenized_docs = [
            text.split()
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
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
        )[:3]

        return [
            self.docs[i]
            for i in top_indices
        ]
