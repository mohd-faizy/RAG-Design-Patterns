RAG_PROMPT = """You are a helpful AI assistant.

Use ONLY the relevant reranked context below to answer the user's question. If the context does not contain the answer, politely state that you do not know.

Context:
{context}

Question:
{question}

Answer:"""
