RAG_PROMPT = """
You are a helpful AI assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

DOC_GRADER_PROMPT = """
You are a grader.

Evaluate whether the document is relevant to the user question.

Question:
{question}

Document:
{document}

Answer only:
yes or no
"""

HALLUCINATION_PROMPT = """
You are checking whether an answer is grounded in facts.

Context:
{context}

Answer:
{answer}

Is the answer supported by the context?

Answer only:
yes or no
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query to improve retrieval quality.

Original Query:
{question}

Improved Query:
"""
