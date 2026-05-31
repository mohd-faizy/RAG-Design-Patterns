from langchain_groq import ChatGroq

def evaluate_retrieval(question: str, docs: list, llm: ChatGroq) -> str:
    """Evaluates whether the retrieved context is relevant for answering the question."""
    if not docs:
        print("\n[CRAG] No documents retrieved. Grading: 'bad'")
        return "bad"

    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Evaluate whether the retrieved context is relevant for answering the question.
    
    Question:
    {question}
    
    Context:
    {context}
    
    Answer ONLY:
    good or bad
    """
    
    response = llm.invoke(prompt)
    grade = response.content.strip().lower()
    print(f"\n[CRAG] Evaluator Grade: '{grade}'")
    
    return "good" if "good" in grade else "bad"

def rewrite_query(question: str, llm: ChatGroq) -> str:
    """Rewrites the query to improve web search retrieval quality."""
    prompt = f"""
    Rewrite the following query to improve search retrieval on the web.
    Output ONLY the improved raw query text. Do not include any explanation, intro, or bullet prefix.
    
    Question:
    {question}
    
    Improved Query:
    """
    
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    print(f"[CRAG] Query Rewritten: '{rewritten}'")
    
    return rewritten
