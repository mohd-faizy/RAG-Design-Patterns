from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.state import GraphState
from src.retriever import get_retriever, retrieve_docs
from src.reranker import rerank_documents
from src.prompts import RAG_PROMPT

# Initialize LLM using ChatGroq
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# --------------------------------
# RETRIEVE NODE
# --------------------------------
def retrieval_node(state):
    print("\n[Node: Retrieve]")
    retriever = get_retriever()
    docs = retrieve_docs(state["question"], retriever)
    print(f"-> Retrieved {len(docs)} candidate documents.")
    return {
        "retrieved_docs": docs
    }

# --------------------------------
# RERANK NODE
# --------------------------------
def rerank_node(state):
    print("\n[Node: Rerank]")
    reranked = rerank_documents(
        query=state["question"],
        docs=state["retrieved_docs"],
        top_k=3
    )
    print(f"-> Selected top {len(reranked)} most relevant documents.")
    return {
        "reranked_docs": reranked
    }

# --------------------------------
# GENERATE NODE
# --------------------------------
def generate_node(state):
    print("\n[Node: Generate]")
    context = "\n\n".join([
        f"[Source {i+1}]: {d.page_content}"
        for i, d in enumerate(state["reranked_docs"])
    ])
    prompt = RAG_PROMPT.format(
        context=context,
        question=state["question"]
    )
    response = llm.invoke(prompt)
    return {
        "answer": response.content
    }

# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("rerank_node", rerank_node)
    workflow.add_node("generate_node", generate_node)

    workflow.set_entry_point("retrieval_node")
    workflow.add_edge("retrieval_node", "rerank_node")
    workflow.add_edge("rerank_node", "generate_node")
    workflow.add_edge("generate_node", END)

    return workflow.compile()
