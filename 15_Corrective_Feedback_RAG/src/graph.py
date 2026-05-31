from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever, retrieve_docs, web_search
from src.evaluator import evaluate_retrieval, rewrite_query
from src.prompts import RAG_PROMPT

# Initialize model and database retriever hooks
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
retriever = get_retriever()

# --- WORKFLOW NODES ---

def retrieve(state: GraphState) -> Dict:
    """Retrieves documents from ChromaDB vector database."""
    query = state.get("rewritten_question") or state["question"]
    print(f"\n[CRAG] Retrieving documents for query: '{query}'...")
    docs = retrieve_docs(query, retriever)
    return {"context": docs}

def grade_retrieval_router(state: GraphState) -> str:
    """Conditional edge router evaluating whether retrieval is good or bad."""
    return evaluate_retrieval(state["question"], state["context"], llm)

def corrective_retrieval(state: GraphState) -> Dict:
    """Fallback step: rewrites query using LLM and fetches web results via DuckDuckGo."""
    rewritten_query = rewrite_query(state["question"], llm)
    web_docs = web_search(rewritten_query)
    return {
        "rewritten_question": rewritten_query,
        "context": web_docs
    }

def generate(state: GraphState) -> Dict:
    """Generates contextually grounded final answer."""
    docs = state["context"]
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = RAG_PROMPT.format(
        context=context_text,
        question=state["question"]
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- GRAPH BUILDER ---

def build_graph():
    workflow = StateGraph(GraphState)
    
    # 1. Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("corrective_retrieval", corrective_retrieval)
    workflow.add_node("generate", generate)
    
    # 2. Wire operational path
    workflow.set_entry_point("retrieve")
    
    # 3. Add conditional edge to evaluate matches
    workflow.add_conditional_edges(
        "retrieve",
        grade_retrieval_router,
        {
            "good": "generate",
            "bad": "corrective_retrieval"
        }
    )
    
    workflow.add_edge("corrective_retrieval", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
