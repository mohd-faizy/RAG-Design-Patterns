from collections import defaultdict


def reciprocal_rank_fusion(
    results,
    k=60
):
    fused_scores = defaultdict(float)
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            text = doc.page_content
            doc_map[text] = doc
            fused_scores[text] += (
                1 / (rank + k)
            )

    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_map[text]
        for text, _ in reranked
    ]
