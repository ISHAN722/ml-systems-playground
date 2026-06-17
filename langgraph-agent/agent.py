import json
import requests
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from tools import TOOLS, TOOL_DESCRIPTIONS


# ------------------------------------------------------------------ #
# State — explicitly typed, passed between every node                 #
# ------------------------------------------------------------------ #

class AgentState(TypedDict):
    question: str
    messages: list[dict]
    tool_calls: list[dict]
    final_answer: str


# ------------------------------------------------------------------ #
# Nodes                                                                #
# ------------------------------------------------------------------ #

def reason(state: AgentState) -> AgentState:
    """LLM decides what to do next - call a tool or answer."""
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in state["messages"]
    )


    prompt = f"""You are a helpful research assistant with access to these tools:
    {TOOL_DESCRIPTIONS}

    To use a tool, respond with:
    TOOL: tool_name
    INPUT: tool input

    To give a final answer, respond with:
    ANSWER: your answer

    Conversation so far:
    {history}

    What do you do next?"""

    response = requests.post("http://localhost:11434/api/generate", json = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200}
    })
    content = response.json()["response"].strip()
    print(f"\n[Agent reasoning]\n{content}")

    state["messages"].append({"role": "assistant", "content": content})
    return state

def act(state: AgentState) -> AgentState:
    """Execute the tool the agent chose."""
    last = state["messages"][-1]["content"]

    lines = last.split("\n")
    tool_name = None
    tool_input = None

    for i, line in enumerate(lines):
        if line.startswith("TOOL:"):
            tool_name = line.replace("TOOL:", "").strip()
            # remove parenthetical if model added it
            if "(" in tool_name:
                tool_name = tool_name.split("(")[0].strip()
        if line.startswith("INPUT:"):
            tool_input = line.replace("INPUT:", "").strip()

    if tool_name and tool_input and tool_name in TOOLS:
        print(f"\n[Tool call] {tool_name}({tool_input})")
        result = TOOLS[tool_name](tool_input)
        print(f"[Tool result] {result[:200]}")

        state["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "result": result
        })
        state["messages"].append({
            "role": "tool",
            "content": f"Tool {tool_name} returned: {result}"
        })
    else:
        state["messages"].append({
            "role": "tool",
            "content": f"Could not parse tool call from: {last[:100]}"
        })

    return state

def should_continue(state: AgentState) -> str:
    """Edge function — continue looping or end?"""
    last = state["messages"][-1]["content"]
    if "ANSWER:" in last:
        return "end"
    if len(state["tool_calls"]) >= 5:  # safety limit
        return "end"
    return "continue"

# ------------------------------------------------------------------ #
# Graph                                                                #
# ------------------------------------------------------------------ #

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("reason", reason)
    graph.add_node("act", act)

    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {
        "continue": "act",
        "end": END
    })

    graph.add_edge("act", "reason")

    return graph.compile()

