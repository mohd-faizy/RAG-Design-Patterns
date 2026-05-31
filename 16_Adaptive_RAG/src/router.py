from langchain_groq import ChatGroq

def classify_question(question: str, llm: ChatGroq) -> str:
    """
    Classifies the query complexity to choose the best retrieval strategy.
    Returns one of: no_retrieval | vector | hybrid | web
    """
    prompt = f"""
    Classify the complexity of the following question to choose the best retrieval strategy.

    Categories (return ONLY one):
    - no_retrieval   (simple factual, math, general knowledge requiring no documents)
    - vector         (specific topic that can be found in a local database)
    - hybrid         (requires both semantic and keyword search)
    - web            (requires real-time or current information from the internet)

    Question:
    {question}

    Return ONLY the category name. Nothing else.
    """
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    
    # Normalize to valid categories
    valid_routes = {"no_retrieval", "vector", "hybrid", "web"}
    for valid in valid_routes:
        if valid in route:
            return valid
    return "vector"  # Safe default fallback
