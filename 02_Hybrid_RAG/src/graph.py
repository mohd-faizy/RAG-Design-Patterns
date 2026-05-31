from typing import Dict

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq

from src.state import GraphState
from src.retriever import HybridRetriever
from src.prompts import RAG_PROMPT

retriever = HybridRetriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# -------------------------
# RETRIEVE NODE
# -------------------------
def retrieve(state: GraphState) -> Dict:
    question = state["question"]
    docs = retriever.retrieve(question)
    return {
        "context": docs
    }


# -------------------------
# GENERATE NODE
# -------------------------
def generate(state: GraphState) -> Dict:
    question = state["question"]
    docs = state["context"]

    context_text = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = RAG_PROMPT.format(
        context=context_text,
        question=question
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -------------------------
# BUILD GRAPH
# -------------------------
def build_graph():
    workflow = StateGraph(GraphState)

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
