from langchain_groq import ChatGroq

from src.prompts import (
    DOC_GRADER_PROMPT,
    HALLUCINATION_PROMPT
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def grade_documents(question, docs):
    filtered_docs = []

    for doc in docs:
        prompt = DOC_GRADER_PROMPT.format(
            question=question,
            document=doc.page_content
        )

        response = llm.invoke(prompt)
        result = response.content.strip().lower()

        if "yes" in result:
            filtered_docs.append(doc)

    return filtered_docs


def check_hallucination(context, answer):
    context_text = "\n\n".join([
        doc.page_content
        for doc in context
    ])

    prompt = HALLUCINATION_PROMPT.format(
        context=context_text,
        answer=answer
    )

    response = llm.invoke(prompt)
    return response.content.strip().lower()
