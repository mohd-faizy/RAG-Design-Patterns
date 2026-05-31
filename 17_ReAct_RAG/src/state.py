from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_step: str
