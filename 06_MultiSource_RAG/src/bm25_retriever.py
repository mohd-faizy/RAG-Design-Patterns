from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, docs):
        """
        Initialize BM25 index from pre-loaded document chunks.

        Args:
            docs: List of LangChain Document objects (already chunked).
        """
        self.docs = docs
        self.texts = [doc.page_content for doc in self.docs]

        tokenized_docs = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query):
        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:3]

        return [self.docs[i] for i in top_indices]
