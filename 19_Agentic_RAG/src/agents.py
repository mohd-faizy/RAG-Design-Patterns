from langchain_groq import ChatGroq

from src.prompts import (
    PLANNER_PROMPT,
    REFLECTION_PROMPT
)
from src.tools import (
    vector_search,
    bm25_search,
    web_search
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# PLANNER AGENT
# --------------------------------
def planner_agent(question):
    prompt = PLANNER_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)
    return response.content.strip().lower()


# --------------------------------
# VECTOR AGENT
# --------------------------------
def vector_agent(question):
    return vector_search.invoke(question)


# --------------------------------
# BM25 AGENT
# --------------------------------
def bm25_agent(question):
    return bm25_search.invoke(question)


# --------------------------------
# WEB AGENT
# --------------------------------
def web_agent(question):
    return web_search.invoke(question)


# --------------------------------
# HYBRID AGENT
# --------------------------------
def hybrid_agent(question):
    vector_results = vector_agent(question)
    bm25_results = bm25_agent(question)

    return f"""
    VECTOR RESULTS:
    {vector_results}

    BM25 RESULTS:
    {bm25_results}
    """


# --------------------------------
# REFLECTION AGENT
# --------------------------------
def reflection_agent(
    question,
    answer
):
    prompt = REFLECTION_PROMPT.format(
        question=question,
        answer=answer
    )

    response = llm.invoke(prompt)
    return response.content.strip().lower()
