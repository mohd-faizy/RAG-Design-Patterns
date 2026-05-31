from langgraph.graph import (
    StateGraph,
    END
)
from langchain_groq import ChatGroq
from src.state import GraphState
from src.colbert_retriever import ColBERTRetriever
from src.prompts import RAG_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

retriever = ColBERTRetriever()


# --------------------------------
# RETRIEVE
# --------------------------------
def retrieve(state):
    print(f"\n[Retriever] Querying ColBERT Index using Late Interaction for: '{state['question']}'...")
    docs = retriever.retrieve(state["question"])
    print(f"[Retriever] Retrieved top-{len(docs)} contextual passages via token MaxSim matching:")
    for i, d in enumerate(docs, 1):
        print(f"  {i}. {d.strip()}")
    return {
        "context": docs
    }


# --------------------------------
# GENERATE
# --------------------------------
def generate(state):
    print("\n[Generator] Synthesizing answer based on contextual late interactions...")
    context_text = "\n\n".join(state["context"])
    
    if not context_text:
        context_text = "No matching token interactions were found in the ColBERT index."

    prompt = RAG_PROMPT.format(
        context=context_text,
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

    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()
