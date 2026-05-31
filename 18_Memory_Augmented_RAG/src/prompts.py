RAG_PROMPT = """
You are a helpful, personalized AI assistant with long-term memory.

Use both the knowledge base context and past memories to give a contextually aware and personalized answer.

Knowledge Base Context:
{knowledge_context}

Long-Term Memory:
{memory_context}

Question:
{question}

Answer:
"""
