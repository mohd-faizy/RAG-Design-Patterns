from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever, retrieve_docs
from src.memory_manager import retrieve_memory, extract_memory, update_memory
from src.prompts import RAG_PROMPT

# Single shared LLM and retriever instances
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
retriever = get_retriever()

# --- WORKFLOW NODES ---

def knowledge_retrieval(state: GraphState) -> Dict:
    """Retrieves relevant documents from the vector knowledge base."""
    print(f"\n[Memory RAG] Retrieving knowledge for: '{state['question']}'...")
    docs = retrieve_docs(state["question"], retriever)
    return {"knowledge_context": docs}

def memory_retrieval(state: GraphState) -> Dict:
    """Retrieves relevant long-term memories for the question."""
    memories = retrieve_memory(state["question"])
    return {"memory_context": memories}

def generate(state: GraphState) -> Dict:
    """Generates a personalized answer fusing knowledge and memory contexts."""
    knowledge_text = "\n\n".join([
        d.page_content for d in state["knowledge_context"]
    ])
    memory_text = "\n\n".join(state["memory_context"]) if state["memory_context"] else "No prior memory available."

    prompt = RAG_PROMPT.format(
        knowledge_context=knowledge_text,
        memory_context=memory_text,
        question=state["question"]
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def store_memory(state: GraphState) -> Dict:
    """Extracts and persists a new long-term memory from this interaction."""
    memory = extract_memory(state["question"], state["answer"], llm)
    update_memory(memory)
    return {}

# --- GRAPH BUILDER ---

def build_graph():
    workflow = StateGraph(GraphState)

    # 1. Define nodes
    workflow.add_node("knowledge_retrieval", knowledge_retrieval)
    workflow.add_node("memory_retrieval", memory_retrieval)
    workflow.add_node("generate", generate)
    workflow.add_node("store_memory", store_memory)

    # 2. Wire the memory-augmented execution path
    workflow.set_entry_point("knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "memory_retrieval")
    workflow.add_edge("memory_retrieval", "generate")
    workflow.add_edge("generate", "store_memory")
    workflow.add_edge("store_memory", END)

    return workflow.compile()
