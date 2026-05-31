def collect_summaries(
    tree,
    summaries=None
):
    if summaries is None:
        summaries = []

    if "summaries" in tree:
        summaries.extend(
            tree["summaries"]
        )

    for child in tree.get(
        "children",
        []
    ):
        collect_summaries(
            child,
            summaries
        )

    return summaries


def retrieve_from_tree(
    tree,
    query
):
    summaries = collect_summaries(
        tree
    )

    # Simplified retrieval
    relevant = []

    for summary in summaries:
        if any(
            word.lower() in summary.lower()
            for word in query.split()
        ):
            relevant.append(summary)

    return relevant[:5]
