from typing import TypedDict


class GraphState(TypedDict):
    question: str
    route: str
    context: str
    answer: str
    retries: int
