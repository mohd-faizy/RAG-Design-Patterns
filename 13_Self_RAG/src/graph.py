from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.state import GraphState
from src.retriever import HybridRetriever
from src.prompts import RAG_PROMPT
from src.graders import (
    grade_documents,
    check_hallucination
)
from src.query_rewriter import rewrite_query

retriever = HybridRetriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# RETRIEVE
# --------------------------------
def retrieve(state: GraphState) -> Dict:
    question = (
        state.get("rewritten_question")
        or state["question"]
    )

    docs = retriever.retrieve(question)

    return {
        "context": docs
    }


# --------------------------------
# GRADE DOCUMENTS
# --------------------------------
def grade(state: GraphState) -> Dict:
    question = state["question"]
    docs = state["context"]

    filtered_docs = grade_documents(
        question,
        docs
    )

    return {
        "context": filtered_docs
    }


# --------------------------------
# DECISION
# --------------------------------
def decide_to_generate(state):
    docs = state["context"]
    retries = state.get("retries", 0)

    if len(docs) == 0 and retries < 2:
        return "rewrite"

    return "generate"


# --------------------------------
# REWRITE QUERY
# --------------------------------
def rewrite(state: GraphState) -> Dict:
    rewritten = rewrite_query(
        state["question"]
    )

    retries = state.get("retries", 0)

    return {
        "rewritten_question": rewritten,
        "retries": retries + 1
    }


# --------------------------------
# GENERATE
# --------------------------------
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


# --------------------------------
# CHECK HALLUCINATION
# --------------------------------
def check_answer(state):
    result = check_hallucination(
        state["context"],
        state["answer"]
    )

    if "yes" in result:
        return "useful"

    return "not_useful"


# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph():
    workflow = StateGraph(
        GraphState
    )

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "grade",
        grade
    )

    workflow.add_node(
        "rewrite",
        rewrite
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
        "grade"
    )

    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )

    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    workflow.add_conditional_edges(
        "generate",
        check_answer,
        {
            "useful": END,
            "not_useful": "rewrite"
        }
    )

    return workflow.compile()
