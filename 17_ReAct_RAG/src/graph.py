from langgraph.prebuilt import (
    create_react_agent
)
from langchain_groq import ChatGroq

from src.tools import TOOLS

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def build_graph():
    agent = create_react_agent(
        llm,
        TOOLS
    )

    return agent
