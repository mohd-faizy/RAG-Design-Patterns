PLANNER_PROMPT = """
You are a planning agent.

Decide which retrieval strategy is best.

Available:
- vector
- bm25
- web
- hybrid

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
