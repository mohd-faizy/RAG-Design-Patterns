RAG_PROMPT = """
You are a helpful AI assistant.

Use the provided Knowledge Graph relationships context to answer the question. 
If the answer cannot be found in the context, say "I cannot find the answer in the provided graph context."

Graph Context:
{context}

Question:
{question}

Answer:
"""
