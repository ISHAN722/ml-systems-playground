import sys
import time
import json
import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from agent_tracer import AgentTrace

sys.path.insert(0, "../rag-demo")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../rag-demo/index/faiss.index")
with open("../rag-demo/index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def ollama(prompt: str, system: str = "") -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300}
    })
    ms = (time.perf_counter() - start) * 1000
    return response.json()["response"].strip(), ms


def search_docs(query: str) -> str:
    emb = embedder.encode([query]).astype("float32")
    _, indices = index.search(emb, 2)
    return "\n".join(chunks[i] for i in indices[0])


def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


TOOLS = {"search_docs": search_docs, "calculate": calculate}

TOOL_DESCRIPTIONS = """
- search_docs: Search knowledge base about Nexora Robotics. Input: a search query string.
- calculate: Evaluate a math expression. Input: a Python math expression like 150 * 2.205
"""


def parse_tool_call(thought: str) -> tuple[str, str]:
    lines = [l.strip() for l in thought.split("\n") if l.strip()]
    tool_name = ""
    tool_input = ""
    for line in lines:
        if line.startswith("TOOL:"):
            tool_name = line.replace("TOOL:", "").strip()
            if "(" in tool_name and ")" in tool_name:
                tool_input = tool_name[tool_name.find("(")+1:tool_name.rfind(")")]
                tool_input = tool_input.replace("query=", "").replace('"', '').replace("'", "").strip()
                tool_name = tool_name[:tool_name.find("(")].strip()
        elif line.startswith("INPUT:"):
            tool_input = line.replace("INPUT:", "").strip().strip('"').strip("'")
    return tool_name.strip(), tool_input.strip()


def run_agent(question: str) -> AgentTrace:
    trace = AgentTrace(question=question)
    agent_start = time.perf_counter()
    messages = [{"role": "user", "content": question}]
    max_steps = 6

    print(f"\n{'='*60}")
    print(f"[Session: {trace.session_id}] Question: {question}")
    print('='*60)

    for step in range(max_steps):
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = f"""You have access to these tools:
{TOOL_DESCRIPTIONS}

IMPORTANT: Always respond in exactly one of these two formats:

Format 1 - use a tool:
TOOL: tool_name
INPUT: the input value

Format 2 - give final answer:
ANSWER: your complete answer here

Never use parentheses. Always put TOOL and INPUT on separate lines.

Conversation so far:
{history}

What do you do next?"""

        thought, reasoning_ms = ollama(prompt)
        print(f"\n[Step {step+1}] ({reasoning_ms:.0f}ms): {thought[:120]}")

        if "ANSWER:" in thought:
            answer = thought.split("ANSWER:")[-1].strip()
            trace.add_reasoning_step(thought, "final_answer", reasoning_ms)
            trace.finish(answer, success=True)
            print(f"\n[Final Answer]: {answer}")
            break
        elif "TOOL:" in thought:
            tool_name, tool_input = parse_tool_call(thought)
            trace.add_reasoning_step(thought, "tool_call", reasoning_ms)
            if tool_name in TOOLS and tool_input:
                tool_start = time.perf_counter()
                result = TOOLS[tool_name](tool_input)
                tool_ms = (time.perf_counter() - tool_start) * 1000
                trace.add_tool_call(tool_name, tool_input, result[:200], tool_ms)
                print(f"  → {tool_name}({tool_input[:50]}) = {result[:80]} ({tool_ms:.0f}ms)")
                messages.append({"role": "assistant", "content": thought})
                messages.append({"role": "tool", "content": f"{tool_name} returned: {result}"})
            else:
                print(f"  → Parse failed: name='{tool_name}' input='{tool_input}'")
                messages.append({"role": "tool", "content": "Tool call failed. Use TOOL: name on one line, INPUT: value on next line."})
        else:
            trace.add_reasoning_step(thought, "unclear", reasoning_ms)
            messages.append({"role": "assistant", "content": thought})
            messages.append({"role": "tool", "content": "Please respond with TOOL:/INPUT: or ANSWER:"})

    trace.total_latency_ms = (time.perf_counter() - agent_start) * 1000
    return trace


def print_dashboard(traces):
    print(f"\n{'='*60}")
    print("AGENTOPS DASHBOARD")
    print('='*60)
    summaries = [t.summary() for t in traces]
    avg_latency = sum(s["total_latency_ms"] for s in summaries) / len(summaries)
    avg_steps = sum(s["total_reasoning_steps"] for s in summaries) / len(summaries)
    avg_tools = sum(s["total_tool_calls"] for s in summaries) / len(summaries)
    success_rate = sum(1 for s in summaries if s["success"]) / len(summaries)
    print(f"\nAggregate metrics ({len(traces)} sessions):")
    print(f"  Success rate:        {success_rate:.0%}")
    print(f"  Avg latency:         {avg_latency:.0f}ms")
    print(f"  Avg reasoning steps: {avg_steps:.1f}")
    print(f"  Avg tool calls:      {avg_tools:.1f}")
    print(f"\nPer-session breakdown:")
    for s in summaries:
        status = "✓" if s["success"] else "✗"
        print(f"  [{status}] {s['session_id']} | {s['total_latency_ms']:.0f}ms | {s['total_reasoning_steps']} steps | tools: {s['tools_used']}")
        print(f"       Q: {s['question'][:70]}")
        print(f"       A: {s['final_answer'][:70]}")


if __name__ == "__main__":
    questions = [
        "How much weight can the Hexapod X1 carry, and what is that in pounds?",
        "Who founded Nexora and when?",
        "What was the percentage improvement from the Solis Logistics partnership?",
    ]
    traces = []
    for q in questions:
        trace = run_agent(q)
        traces.append(trace)
    print_dashboard(traces)
    with open("agent_traces.json", "w") as f:
        from dataclasses import asdict
        json.dump([asdict(t) for t in traces], f, indent=2)
    print(f"\nTraces saved to agent_traces.json")
