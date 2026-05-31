from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    retrieved_docs: List
    reranked_docs: List
    answer: str
