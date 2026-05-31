from collections import defaultdict
from langchain_core.documents import Document

def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    """
    Applies Reciprocal Rank Fusion (RRF) algorithm to rank documents from multiple queries.
    Formula: RRF(d) = sum_{r in R} 1 / (k + r(d))
    """
    fused_scores = defaultdict(float)
    doc_map = {}

    for query_docs in results:
        for rank, doc in enumerate(query_docs):
            text = doc.page_content
            # Keep the first/most relevant instance of the doc object
            if text not in doc_map:
                doc_map[text] = doc
            fused_scores[text] += 1 / (rank + k)

    # Sort documents by fused score in descending order
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Reconstruct original Document structures sorted by rank
    return [doc_map[text] for text, _ in reranked]
