from langchain_groq import ChatGroq

def generate_followup_query(question: str, context: str, llm: ChatGroq) -> str:
    """
    Generates a follow-up search query based on what was retrieved in the first hop.
    This bridges the gap between initial evidence and the final answer.
    """
    prompt = f"""
    You are a multi-hop reasoning system.

    Based on the initial context retrieved, generate a focused follow-up search query to find the missing piece of information needed to fully answer the original question.

    Output ONLY the raw follow-up query. No explanation or preamble.

    Original Question:
    {question}

    Initial Context Retrieved:
    {context}

    Follow-up Query:
    """
    response = llm.invoke(prompt)
    followup = response.content.strip()
    print(f"\n[Multi-Hop RAG] Follow-up Query: '{followup}'")
    return followup

def generate_answer(question: str, contexts: list[str], llm: ChatGroq) -> str:
    """Synthesizes a final answer by reasoning across all gathered evidence."""
    combined_context = "\n\n---\n\n".join(contexts)
    
    prompt = f"""
    You are a deep reasoning AI assistant.

    Use the multi-hop evidence below to synthesize a comprehensive answer. Connect the dots across multiple pieces of evidence.

    Evidence:
    {combined_context}

    Question:
    {question}

    Final Answer:
    """
    response = llm.invoke(prompt)
    return response.content
