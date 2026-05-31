from sklearn.cluster import KMeans
from langchain_groq import ChatGroq
import numpy as np

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# SUMMARIZE CLUSTER
# --------------------------------
def summarize_cluster(texts):
    joined_text = "\n\n".join(texts)

    prompt = f"""
    Summarize the following texts.

    Text:
    {joined_text}

    Summary:
    """

    response = llm.invoke(prompt)
    return response.content


# --------------------------------
# CLUSTER EMBEDDINGS
# --------------------------------
def cluster_embeddings(
    embeddings,
    n_clusters=2
):
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=42
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    return labels


# --------------------------------
# BUILD TREE RECURSIVELY
# --------------------------------
def build_recursive_tree(
    texts,
    embeddings_model,
    level=0,
    max_levels=3
):
    if (
        len(texts) <= 2
        or level >= max_levels
    ):
        return {
            "level": level,
            "texts": texts,
            "children": []
        }

    # -------------------------
    # EMBEDDINGS
    # -------------------------
    embeddings = embeddings_model.embed_documents(
        texts
    )

    embeddings = np.array(embeddings)

    # -------------------------
    # CLUSTER
    # -------------------------
    n_clusters = min(2, len(texts))

    labels = cluster_embeddings(
        embeddings,
        n_clusters=n_clusters
    )

    clustered_texts = {}

    for idx, label in enumerate(labels):
        clustered_texts.setdefault(
            label,
            []
        ).append(texts[idx])

    children = []
    summaries = []

    # -------------------------
    # SUMMARIZE CLUSTERS
    # -------------------------
    for cluster_docs in clustered_texts.values():
        summary = summarize_cluster(
            cluster_docs
        )

        summaries.append(summary)

        child_tree = build_recursive_tree(
            cluster_docs,
            embeddings_model,
            level + 1,
            max_levels
        )

        children.append(child_tree)

    return {
        "level": level,
        "summaries": summaries,
        "children": children
    }
