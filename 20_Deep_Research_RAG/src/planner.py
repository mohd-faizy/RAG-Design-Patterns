import re
from langchain_groq import ChatGroq
from src.prompts import PLANNER_PROMPT

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------
# CREATE RESEARCH PLAN
# --------------------------------
def create_research_plan(question):
    prompt = PLANNER_PROMPT.format(question=question)
    response = llm.invoke(prompt)

    tasks = response.content.split("\n")

    cleaned_tasks = []
    for t in tasks:
        item = t.strip()
        if not item:
            continue
        # Strip common bullets
        item = item.lstrip("-*• ")
        # Strip numbers like "1. " or "1) "
        item = re.sub(r'^\d+[\.\)]\s*', '', item).strip()
        if item:
            cleaned_tasks.append(item)

    return cleaned_tasks
