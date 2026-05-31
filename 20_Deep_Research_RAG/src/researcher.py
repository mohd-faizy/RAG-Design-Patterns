from src.retriever import (
    retrieve_docs,
    web_search
)


# --------------------------------
# RESEARCH A SINGLE TASK
# --------------------------------
def research_task(task):
    vector_results = retrieve_docs(task)
    web_results = web_search(task)

    combined = []
    combined.extend(vector_results)
    combined.extend(web_results)

    return combined
