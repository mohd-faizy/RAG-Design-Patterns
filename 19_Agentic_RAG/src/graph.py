from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.state import GraphState
from src.agents import (
    planner_agent,
    vector_agent,
    bm25_agent,
    web_agent,
    hybrid_agent,
    reflection_agent
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# PLAN
# --------------------------------
def plan(state: GraphState):
    route = planner_agent(
        state["question"]
    )

    return {
        "route": route
    }


# --------------------------------
# RETRIEVE
# --------------------------------
def retrieve(state: GraphState):
    question = state["question"]
    route = state["route"]

    if "vector" in route:
        context = vector_agent(question)
    elif "bm25" in route:
        context = bm25_agent(question)
    elif "web" in route:
        context = web_agent(question)
    else:
        context = hybrid_agent(question)

    return {
        "context": context
    }


# --------------------------------
# GENERATE
# --------------------------------
def generate(state: GraphState):
    prompt = f"""
    Answer the question using context.

    Context:
    {state["context"]}

    Question:
    {state["question"]}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------------------
# REFLECT
# --------------------------------
def reflect(state: GraphState):
    result = reflection_agent(
        state["question"],
        state["answer"]
    )

    retries = state.get(
        "retries",
        0
    )

    if "yes" in result:
        return "good"

    if retries >= 2:
        return "good"

    return "retry"


# --------------------------------
# RETRY
# --------------------------------
def retry(state: GraphState):
    retries = state.get(
        "retries",
        0
    )

    return {
        "retries": retries + 1,
        "route": "hybrid"
    }


# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph():
    workflow = StateGraph(
        GraphState
    )

    workflow.add_node(
        "plan",
        plan
    )

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.add_node(
        "retry",
        retry
    )

    workflow.set_entry_point(
        "plan"
    )

    workflow.add_edge(
        "plan",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_conditional_edges(
        "generate",
        reflect,
        {
            "good": END,
            "retry": "retry"
        }
    )

    workflow.add_edge(
        "retry",
        "retrieve"
    )

    return workflow.compile()
