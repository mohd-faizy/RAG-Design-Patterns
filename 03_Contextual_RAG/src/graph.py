from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.state import GraphState
from src.retriever import get_retriever
from src.prompts import RAG_PROMPT

retriever = get_retriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# -------------------------
# RETRIEVE
# -------------------------
def retrieve(state: GraphState) -> Dict:
    docs = retriever.invoke(
        state["question"]
    )

    return {
        "context": docs
    }


# -------------------------
# GENERATE
# -------------------------
def generate(state: GraphState) -> Dict:
    context_text = "\n\n".join([
        doc.page_content
        for doc in state["context"]
    ])

    prompt = RAG_PROMPT.format(
        context=context_text,
        question=state["question"]
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -------------------------
# BUILD GRAPH
# -------------------------
def build_graph():
    workflow = StateGraph(
        GraphState
    )

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()
