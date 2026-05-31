from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    first_context: List[Document]
    followup_query: str
    second_context: List[Document]
    answer: str
