from langchain_groq import ChatGroq

from src.prompts import (
    RETRIEVAL_GRADER_PROMPT,
    HALLUCINATION_PROMPT
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def evaluate_retrieval(question, docs):
    docs_text = "\n\n".join([
        doc.page_content
        if hasattr(doc, "page_content")
        else str(doc)
        for doc in docs
    ])

    prompt = RETRIEVAL_GRADER_PROMPT.format(
        question=question,
        documents=docs_text
    )

    response = llm.invoke(prompt)
    return response.content.strip().lower()


def check_hallucination(context, answer):
    context_text = "\n\n".join([
        doc.page_content
        if hasattr(doc, "page_content")
        else str(doc)
        for doc in context
    ])

    prompt = HALLUCINATION_PROMPT.format(
        context=context_text,
        answer=answer
    )

    response = llm.invoke(prompt)
    return response.content.strip().lower()
