from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def generate_document_context(document_text):
    prompt = f"""
    Generate a concise contextual summary
    for the following document.

    Document:
    {document_text}

    Context Summary:
    """

    response = llm.invoke(prompt)
    return response.content.strip()
