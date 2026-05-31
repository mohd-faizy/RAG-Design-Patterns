import re
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from src.state import GraphState
from src.tools import TOOLS

# Single shared LLM instance
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Tool registry for routing
TOOL_MAP = {tool.name: tool for tool in TOOLS}

SYSTEM_PROMPT = """You are a ReAct AI agent.
You can use tools to answer the user's question.

Available Tools:
1. vector_search: Semantic vector retrieval tool. Takes a query string.
2. bm25_search: Keyword BM25 retrieval tool. Takes a query string.
3. web_search: Web search retrieval tool. Takes a query string.

You MUST respond in one of the following formats:

Format 1 (To use a tool):
Thought: <your reasoning>
Action: <tool_name>[<query>]

Format 2 (To provide the final answer):
Thought: <your reasoning>
Final Answer: <your final response based on tool observations>

Do NOT use any other format. Always think step by step before calling a tool or providing the final answer."""

def call_model(state: GraphState) -> Dict:
    messages = state["messages"]
    
    # Reconstruct history into a string prompt
    prompt = SYSTEM_PROMPT + "\n\n"
    for msg in messages:
        if isinstance(msg, HumanMessage) or msg.type == "human":
            prompt += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage) or msg.type == "ai":
            prompt += f"Assistant: {msg.content}\n"
        else:
            prompt += f"System: {msg.content}\n"
            
    prompt += "\nAssistant thought and action:"
    
    response = llm.invoke(prompt)
    content = response.content
    
    action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", content)
    final_match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
    
    if final_match:
        # Loop is ending, print thought process and return clean final answer
        print(f"\n[Agent Thought]:\n{content}\n" + "-"*40)
        clean_answer = final_match.group(1).strip()
        return {
            "messages": [AIMessage(content=clean_answer)],
            "next_step": "end"
        }
    elif action_match:
        print(f"\n[Agent Thought & Action]:\n{content}\n" + "-"*40)
        return {
            "messages": [AIMessage(content=content)],
            "next_step": "tools"
        }
    else:
        # Fallback
        print(f"\n[Agent Raw Output]:\n{content}\n" + "-"*40)
        return {
            "messages": [AIMessage(content=content)],
            "next_step": "end"
        }

def call_tool(state: GraphState) -> Dict:
    messages = state["messages"]
    last_msg_content = messages[-1].content
    
    action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", last_msg_content)
    if not action_match:
        return {"messages": [HumanMessage(content="Observation: Error - No valid action format found.")]}
        
    tool_name = action_match.group(1).strip()
    tool_input = action_match.group(2).strip()
    
    print(f"\n[Executing Tool] {tool_name} with input: '{tool_input}'")
    
    if tool_name in TOOL_MAP:
        try:
            observation = TOOL_MAP[tool_name].invoke(tool_input)
        except Exception as e:
            observation = f"Error executing tool: {e}"
    else:
        observation = f"Tool '{tool_name}' not found."
        
    print(f"[Observation]:\n{observation[:200]}...")
    
    return {
        "messages": [HumanMessage(content=f"Observation: {observation}")]
    }

def should_continue(state: GraphState) -> str:
    if state.get("next_step") == "tools":
        return "tools"
    return "end"

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
