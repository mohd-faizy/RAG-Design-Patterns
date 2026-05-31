from langgraph.graph import (
    StateGraph,
    END
)

from src.state import GraphState
from src.planner import create_research_plan
from src.researcher import research_task
from src.synthesizer import synthesize_research


# --------------------------------
# PLAN RESEARCH
# --------------------------------
def planning_node(state):
    print("\n[Planner] Analyzing query and creating research plan...")
    tasks = create_research_plan(state["question"])
    print(f"[Planner] Generated {len(tasks)} research sub-questions:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    return {
        "tasks": tasks
    }


# --------------------------------
# RESEARCH NODE
# --------------------------------
def research_node(state):
    tasks = state["tasks"]
    print(f"\n[Researcher] Initiating multi-source research for {len(tasks)} sub-questions...")
    
    all_evidence = []
    for i, task in enumerate(tasks, 1):
        print(f"  -> Researching ({i}/{len(tasks)}): '{task}'...")
        results = research_task(task)
        print(f"     Found {len(results)} pieces of evidence.")
        all_evidence.extend(results)

    return {
        "evidence": all_evidence
    }


# --------------------------------
# SYNTHESIS NODE
# --------------------------------
def synthesis_node(state):
    print("\n[Synthesizer] Compiling and synthesizing gathered evidence...")
    answer = synthesize_research(
        state["question"],
        state["evidence"]
    )
    print("[Synthesizer] Analyst report successfully compiled.")
    return {
        "answer": answer
    }


# --------------------------------
# BUILD GRAPH
# --------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(
        "planning_node",
        planning_node
    )

    workflow.add_node(
        "research_node",
        research_node
    )

    workflow.add_node(
        "synthesis_node",
        synthesis_node
    )

    workflow.set_entry_point(
        "planning_node"
    )

    workflow.add_edge(
        "planning_node",
        "research_node"
    )

    workflow.add_edge(
        "research_node",
        "synthesis_node"
    )

    workflow.add_edge(
        "synthesis_node",
        END
    )

    return workflow.compile()
