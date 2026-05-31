from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from src.state import GraphState
from src.retriever import HybridRetriever
from src.graders import (
    evaluate_retrieval,
    check_hallucination
)
from src.query_rewriter import rewrite_query
from src.web_search import web_search
from src.prompts import RAG_PROMPT

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
# EVALUATE RETRIEVAL
# --------------------------------
def evaluate(state):
    result = evaluate_retrieval(
        state["question"],
        state["context"]
    )

    if "yes" in result:
        return "good"

    return "bad"


# --------------------------------
# QUERY REWRITE
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
# WEB SEARCH
# --------------------------------
def search_web(state: GraphState) -> Dict:
    query = (
        state.get("rewritten_question")
        or state["question"]
    )

    results = web_search(query)
    current_context = state["context"]

    # Wrap the strings into proper Document structures for state consistency
    web_docs = [
        Document(page_content=content, metadata={"source": "web_search"})
        for content in results
    ]
    current_context.extend(web_docs)

    return {
        "context": current_context
    }


# --------------------------------
# GENERATE
# --------------------------------
def generate(state: GraphState) -> Dict:
    question = state["question"]
    docs = state["context"]

    context_text = "\n\n".join([
        doc.page_content
        if hasattr(doc, "page_content")
        else str(doc)
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
# HALLUCINATION CHECK
# --------------------------------
def verify(state):
    result = check_hallucination(
        state["context"],
        state["answer"]
    )

    retries = state.get("retries", 0)

    if "yes" in result:
        return "useful"

    if retries >= 2:
        return "useful"

    return "retry"


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
        "rewrite",
        rewrite
    )

    workflow.add_node(
        "web_search",
        search_web
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_conditional_edges(
        "retrieve",
        evaluate,
        {
            "good": "generate",
            "bad": "rewrite"
        }
    )

    workflow.add_edge(
        "rewrite",
        "web_search"
    )

    workflow.add_edge(
        "web_search",
        "generate"
    )

    workflow.add_conditional_edges(
        "generate",
        verify,
        {
            "useful": END,
            "retry": "rewrite"
        }
    )

    return workflow.compile()
