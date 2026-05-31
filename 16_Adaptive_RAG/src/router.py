from langchain_groq import ChatGroq

def classify_question(question: str, llm: ChatGroq) -> str:
    """
    Classifies the query complexity to choose the best retrieval strategy.
    Returns one of: no_retrieval | vector | hybrid | web
    """
    prompt = f"""
    You are an advanced query router.
    Your task is to classify the user's question into the most appropriate category to select the optimal retrieval strategy.

    Local Database Scope:
    - The local database contains highly detailed technical documentation about **Retrieval-Augmented Generation (RAG) Design Patterns**, AI agent frameworks, vector databases, LangChain, and LangGraph.
    - Questions about RAG architectures (Standard, Hybrid, Contextual, Hierarchical, Reranker, Multi-Hop, KG, Agentic, Deep Research), LangChain, LangGraph, or LLM evaluation/retrieval are in the local database scope!

    Categories (choose ONLY one):
    - no_retrieval   (General trivia, math, simple greetings, basic coding questions, or general reasoning NOT related to RAG or AI architectures)
    - vector         (Specific queries about standard RAG, chunking, embeddings, or simple local database topics)
    - hybrid         (Complex queries about RAG, hybrid search, or detailed AI architectures/LangGraph workflow patterns)
    - web            (Real-time, current events, or questions completely outside the local database scope like public companies, politics, weather, or other domains)

    User Question:
    {question}

    Return ONLY the exact category name (no punctuation, no preambles):
    no_retrieval OR vector OR hybrid OR web
    """
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    
    # Normalize to valid categories
    valid_routes = {"no_retrieval", "vector", "hybrid", "web"}
    for valid in valid_routes:
        if valid in route:
            return valid
    return "vector"  # Safe default fallback
