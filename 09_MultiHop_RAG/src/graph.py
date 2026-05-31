from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever, retrieve
from src.reasoning import generate_followup_query, generate_answer

# Single shared LLM and retriever instances
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
retriever = get_retriever()

# --- WORKFLOW NODES ---

def first_retrieval(state: GraphState) -> Dict:
    """Hop 1: Retrieves initial documents for the original question."""
    print(f"\n[Multi-Hop RAG] Hop 1 — Retrieving for: '{state['question']}'...")
    docs = retrieve(state["question"], retriever)
    return {"first_context": docs}

def create_followup_query(state: GraphState) -> Dict:
    """Generates a bridging follow-up query based on first hop evidence."""
    context = "\n\n".join([d.page_content for d in state["first_context"]])
    followup = generate_followup_query(state["question"], context, llm)
    return {"followup_query": followup}

def second_retrieval(state: GraphState) -> Dict:
    """Hop 2: Retrieves additional documents using the follow-up query."""
    print(f"[Multi-Hop RAG] Hop 2 — Retrieving for: '{state['followup_query']}'...")
    docs = retrieve(state["followup_query"], retriever)
    return {"second_context": docs}

def generate(state: GraphState) -> Dict:
    """Fuses all multi-hop evidence and generates the final answer."""
    # Combine all evidence from both retrieval hops
    contexts = (
        [d.page_content for d in state["first_context"]] +
        [d.page_content for d in state["second_context"]]
    )
    answer = generate_answer(state["question"], contexts, llm)
    return {"answer": answer}

# --- GRAPH BUILDER ---

def build_graph():
    workflow = StateGraph(GraphState)

    # 1. Define nodes
    workflow.add_node("first_retrieval", first_retrieval)
    workflow.add_node("create_followup_query", create_followup_query)
    workflow.add_node("second_retrieval", second_retrieval)
    workflow.add_node("generate", generate)

    # 2. Wire sequential multi-hop execution path
    workflow.set_entry_point("first_retrieval")
    workflow.add_edge("first_retrieval", "create_followup_query")
    workflow.add_edge("create_followup_query", "second_retrieval")
    workflow.add_edge("second_retrieval", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
