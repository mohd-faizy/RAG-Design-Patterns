from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.state import GraphState
from src.retriever import retrieve_from_tree
from src.prompts import RAG_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def retrieve(
    state,
    tree
):
    docs = retrieve_from_tree(
        tree,
        state["question"]
    )

    return {
        "context": docs
    }


def generate(state: GraphState):
    context_text = "\n\n".join(
        state["context"]
    )

    prompt = RAG_PROMPT.format(
        context=context_text,
        question=state["question"]
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def build_graph(tree):
    workflow = StateGraph(
        GraphState
    )

    workflow.add_node(
        "retrieve",
        lambda state: retrieve(
            state,
            tree
        )
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
