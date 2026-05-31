from collections import defaultdict


def reciprocal_rank_fusion(results, k=60):
    fused_scores = defaultdict(float)
    doc_map = {}  # Map to keep track of content to actual Document objects

    for docs in results:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            fused_scores[content] += 1 / (rank + k)
            if content not in doc_map:
                doc_map[content] = doc

    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return the Document objects instead of raw strings to avoid downstream crashes
    return [doc_map[content] for content, _ in reranked]
