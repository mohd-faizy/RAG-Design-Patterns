from typing import Dict

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_groq import ChatGroq

from src.vector_retriever import VectorRetriever
from src.bm25_retriever import BM25Retriever
from src.web_retriever import WebRetriever
from src.fusion import reciprocal_rank_fusion
from src.prompts import RAG_PROMPT
from src.state import GraphState

vector_retriever = VectorRetriever()
web_retriever = WebRetriever()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph(docs):
    # BM25 is built once from the pre-loaded docs — no redundant reload
    bm25_retriever = BM25Retriever(docs)

    # --------------------------------
    # MULTI SOURCE RETRIEVAL
    # --------------------------------
    def retrieve(state: GraphState) -> Dict:
        query = state["question"]

        vector_docs = vector_retriever.retrieve(query)
        bm25_docs = bm25_retriever.retrieve(query)
        web_docs = web_retriever.retrieve(query)

        fused_docs = reciprocal_rank_fusion([
            vector_docs,
            bm25_docs,
            web_docs
        ])

        return {
            "context": fused_docs[:5]
        }

    # --------------------------------
    # GENERATE
    # --------------------------------
    def generate(state: GraphState) -> Dict:
        context_text = "\n\n".join([
            doc.page_content
            for doc in state["context"]
        ])

        prompt = RAG_PROMPT.format(
            context=context_text,
            question=state["question"]
        )

        response = llm.invoke(prompt)

        return {
            "answer": response.content
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
