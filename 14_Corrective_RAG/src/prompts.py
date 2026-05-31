RAG_PROMPT = """
You are a helpful AI assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

RETRIEVAL_GRADER_PROMPT = """
Evaluate retrieval quality.

Question:
{question}

Documents:
{documents}

Are the documents sufficient to answer the question?

Answer only:
yes or no
"""

HALLUCINATION_PROMPT = """
Check whether the answer is supported by context.

Context:
{context}

Answer:
{answer}

Answer only:
yes or no
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query for better retrieval.

Original Query:
{question}

Improved Query:
"""
