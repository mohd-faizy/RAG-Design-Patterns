PLANNER_PROMPT = """
You are a planning agent.

Decide which retrieval strategy is best.

Available:
- vector: specific queries about local RAG, chunking, or embeddings documentation.
- bm25: keyword searches for specific terminology in the local database.
- web: questions completely outside the local RAG database scope (such as geography, general knowledge, history, or real-time facts).
- hybrid: complex, multi-concept queries about local RAG design patterns.

Question:
{question}

Answer ONLY:
vector, bm25, web, or hybrid
"""

REFLECTION_PROMPT = """
Evaluate answer quality.

Question:
{question}

Answer:
{answer}

Is this answer good?

Answer only:
yes or no
"""
