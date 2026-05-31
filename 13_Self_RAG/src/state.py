from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    context: List[Document]
    answer: str
    retries: int
