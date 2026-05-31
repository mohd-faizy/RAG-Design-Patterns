from typing import TypedDict, List


class GraphState(TypedDict):
    question: str
    tasks: List[str]
    evidence: List[str]
    answer: str
