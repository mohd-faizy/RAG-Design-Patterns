from langchain_groq import ChatGroq

from src.prompts import QUERY_REWRITE_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def rewrite_query(question):
    prompt = QUERY_REWRITE_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)
    return response.content.strip()
