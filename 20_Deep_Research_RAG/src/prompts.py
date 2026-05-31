PLANNER_PROMPT = """Break the research task into smaller research sub-questions.

Question:
{question}

Return one sub-question per line. Do not include numbered items, letters, or bullet prefixes in the final lines, just raw plain text questions."""

SYNTHESIZER_PROMPT = """You are a research analyst.

Use the collected evidence to create a comprehensive, structured answer.

Question:
{question}

Evidence:
{evidence}

Create:
- overview
- comparisons
- insights
- conclusions"""
