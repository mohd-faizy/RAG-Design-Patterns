from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever, fusion_retrieve
from src.prompts import RAG_PROMPT

# Initialize model and database retriever hooks
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
retriever = get_retriever()

# --- WORKFLOW NODES ---

def retrieve(state: GraphState) -> Dict:
    """Generates multi-query variations, retrieves docs for each, and fuses them."""
    question = state["question"]
    docs = fusion_retrieve(question, retriever, llm)
    return {"context": docs}

def generate(state: GraphState) -> Dict:
    """Generates final answer using fused reciprocal ranked documents."""
    question = state["question"]
    docs = state["context"]
    
    # Concatenate the fused page contents
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = RAG_PROMPT.format(
        context=context_text,
        question=question
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- GRAPH BUILDER ---

def build_graph():
    workflow = StateGraph(GraphState)
    
    # 1. Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # 2. Wire the operational execution path
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
