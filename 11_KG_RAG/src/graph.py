from langgraph.graph import (
    StateGraph,
    END
)
from langchain_groq import ChatGroq
from src.state import GraphState
from src.kg_retriever import KGRetriever
from src.prompts import RAG_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

retriever = KGRetriever()


# --------------------------------
# RETRIEVE GRAPH CONTEXT
# --------------------------------
def retrieve(state):
    print(f"\n[Retriever] Querying Neo4j Graph for subgraphs matching: '{state['question']}'...")
    context = retriever.retrieve(state["question"])
    print(f"[Retriever] Found {len(context)} relevant entity relationships in local subgraph:")
    for relation in context:
        print(f"  - {relation.strip()}")
    return {
        "context": context
    }


# --------------------------------
# GENERATE ANSWER
# --------------------------------
def generate(state):
    print("\n[Generator] Generating answer grounded on retrieved subgraph...")
    context_text = "\n".join(state["context"])
    
    # If context is empty, supply a friendly fallback message
    if not context_text:
        context_text = "No direct relationships matching this entity were found in the knowledge graph database."

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
