from langchain_groq import ChatGroq
from src.prompts import SYNTHESIZER_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# SYNTHESIZE RESEARCH
# --------------------------------
def synthesize_research(question, evidence):
    combined = "\n\n".join(evidence)
    prompt = SYNTHESIZER_PROMPT.format(
        question=question,
        evidence=combined
    )

    response = llm.invoke(prompt)
    return response.content
