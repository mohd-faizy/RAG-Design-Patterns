from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.router import classify_question
from src.retrievers import vector_retrieve, hybrid_retrieve, web_retrieve
from src.prompts import RAG_PROMPT

# Single shared LLM instance
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# --- WORKFLOW NODES ---

def route_question(state: GraphState) -> Dict:
    """Classifies query complexity to determine retrieval strategy."""
    route = classify_question(state["question"], llm)
    print(f"\n[Adaptive RAG] Query classified as: '{route}'")
    return {"route": route}

def retrieve(state: GraphState) -> Dict:
    """Executes retrieval based on the classified route."""
    route = state["route"]
    question = state["question"]
    context = []
    
    if route == "vector":
        docs = vector_retrieve(question)
        context = [d.page_content for d in docs]
        
    elif route == "hybrid":
        docs = hybrid_retrieve(question)
        context = [d.page_content for d in docs]
        
    elif route == "web":
        context = web_retrieve(question)
        
    print(f"[Adaptive RAG] Retrieved {len(context)} context segment(s).")
    return {"context": context}

def generate(state: GraphState) -> Dict:
    """Generates contextual answer with or without retrieved context."""
    context_text = "\n\n".join(state.get("context", []))
    
    prompt = RAG_PROMPT.format(
        context=context_text if context_text else "No additional context available.",
        question=state["question"]
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- CONDITIONAL ROUTER ---

def should_retrieve(state: GraphState) -> str:
    """Decides whether to retrieve or skip directly to generate."""
    if state["route"] == "no_retrieval":
        print("[Adaptive RAG] No retrieval needed — skipping directly to generate.")
        return "skip"
    return "retrieve"

# --- GRAPH BUILDER ---

def build_graph():
    workflow = StateGraph(GraphState)
    
    # 1. Define nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # 2. Wire the operational execution path
    workflow.set_entry_point("route_question")
    
    # 3. Conditional routing: skip retrieval or perform it
    workflow.add_conditional_edges(
        "route_question",
        should_retrieve,
        {
            "skip": "generate",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
