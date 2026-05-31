from langchain_groq import ChatGroq
from src.memory_store import load_memory, save_memory

def retrieve_memory(question: str) -> list[str]:
    """
    Retrieves relevant past memories by keyword matching.
    Returns up to 5 most relevant memories for the question.
    """
    memories = load_memory()
    if not memories:
        return []

    relevant = []
    query_words = set(question.lower().split())
    for memory in memories:
        memory_words = set(memory.lower().split())
        # Find memories that share meaningful keywords with the question
        if query_words & memory_words:
            relevant.append(memory)

    print(f"[Memory RAG] Retrieved {len(relevant[:5])} relevant memories.")
    return relevant[:5]

def extract_memory(question: str, answer: str, llm: ChatGroq) -> str:
    """
    Uses LLM to distil key information from the interaction worth storing as long-term memory.
    """
    prompt = f"""
    Extract a concise, single-sentence fact or preference worth remembering from this interaction.
    Return ONLY the memory text. No labels, no explanation.

    Question:
    {question}

    Answer:
    {answer}

    Memory:
    """
    response = llm.invoke(prompt)
    return response.content.strip()

def update_memory(memory: str):
    """Appends new memory to the persistent JSON memory store."""
    memories = load_memory()
    memories.append(memory)
    save_memory(memories)
    print(f"[Memory RAG] Stored new memory: '{memory[:60]}...'")
