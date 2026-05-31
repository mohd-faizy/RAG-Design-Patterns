from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.graph_retriever import GraphRetriever
from src.prompts import RAG_PROMPT

# Initialize model hook
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# --- WORKFLOW NODES ---

def retrieve(state: GraphState) -> Dict:
    """Queries Neo4j database to find matching subgraphs."""
    question = state["question"]
    print(f"\n[Graph RAG] Retrieving context relationships for: '{question}'...")
    
    retriever = GraphRetriever()
    try:
        context = retriever.retrieve(question)
    finally:
        retriever.close()
        
    print(f"[Graph RAG] Retrieved subgraphs:\n" + "\n".join(f"  - {c}" for c in context))
    return {"context": context}

def generate(state: GraphState) -> Dict:
    """Generates final answer using Neo4j subgraph contexts."""
    question = state["question"]
    context = state["context"]
    
    context_text = "\n".join(context)
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
    
    # 2. Wire operational path
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
