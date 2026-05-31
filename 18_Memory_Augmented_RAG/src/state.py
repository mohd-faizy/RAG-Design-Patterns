from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    knowledge_context: List[Document]
    memory_context: List[str]
    answer: str
