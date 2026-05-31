from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    route: str
    context: List[str]
    answer: str
