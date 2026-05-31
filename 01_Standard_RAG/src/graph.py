from typing import Dict
from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever
from src.prompts import RAG_PROMPT

# Instantiate the tools inside our pipeline graph
retriever = get_retriever()
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# --- WORKFLOW NODES ---

def retrieve(state: GraphState) -> Dict:
    """Fetches relevant documents from ChromaDB."""
    question = state["question"]
    docs = retriever.invoke(question)
    return {"context": docs}

def generate(state: GraphState) -> Dict:
    """Generates an answer using the retrieved context."""
    question = state["question"]
    docs = state["context"]
    
    # Format document contents into a clean text block
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
