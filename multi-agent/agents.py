import requests
import sys
sys.path.insert(0, "../rag-demo")

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from langgraph.graph import StateGraph, END

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

# load FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../rag-demo/index/faiss.index")
with open("../rag-demo/index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def ollama(prompt: str, system: str = " ") -> str:
    response = requests.post(OLLAMA_URL, json = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300}
    })
    return response.json()["response"].strip()

# ------------------------------------------------------------------ #
# State                                                                #
# ------------------------------------------------------------------ #

class MultiAgentState(TypedDict):
    question: str
    research: str
    insights: str
    final_answer: str
    steps: list[str]


# ------------------------------------------------------------------ #
# Agent nodes                                                          #
# ------------------------------------------------------------------ #

def researcher(state: MultiAgentState) -> MultiAgentState:
    """Retrieves relevant information from the knowledge base."""
    print("\n[Researcher] Searching knowledge base...")
    question = state["question"]

    emb = embedder.encode([question]).astype("float32")
    _, indices = index.search(emb, 3)
    retrieved = [chunks[i] for i in indices[0]]
    research = "\n\n".join(retrieved)

    summary = ollama(
        f"Summarize the following information relevant to this question: {question}\n\nInformation:\n{research}",
        system="You are a research assistant. Summarize only what is directly relevant. Be concise."
    )

    print(f"  Found {len(retrieved)} chunks, summarized to {len(summary)} chars")
    state["research"] = summary
    state["steps"].append("researcher: retrieved and summarized relevant documents")
    return state


def analyst(state: MultiAgentState) -> MultiAgentState:
    """Extracts key insights from research."""
    print("\n[Analyst] Extracting insights...")
    insights = ollama(
        f"Question: {state['question']}\n\nResearch: {state['research']}\n\nExtract 3 key insights that directly answer the question.",
        system="You are an analyst. Extract specific, factual insights. Number them 1, 2, 3."
    )
    print(f"  Extracted insights ({len(insights)} chars)")
    state["insights"] = insights
    state["steps"].append("analyst: extracted key insights from research")
    return state


def writer(state: MultiAgentState) -> MultiAgentState:
    """Produces final polished answer."""
    print("\n[Writer] Composing final answer...")
    answer = ollama(
        f"Question: {state['question']}\n\nKey insights:\n{state['insights']}\n\nWrite a clear, concise answer to the question using these insights.",
        system="You are a professional writer. Write a clear, direct answer. 2-3 sentences maximum."
    )
    print(f"  Answer composed ({len(answer)} chars)")
    state["final_answer"] = answer
    state["steps"].append("writer: composed final answer")
    return state


def supervisor(state: MultiAgentState) -> str:
    """Decides which agent to call next."""
    if not state["research"]:
        return "researcher"
    if not state["insights"]:
        return "analyst"
    if not state["final_answer"]:
        return "writer"
    return "end"


# ------------------------------------------------------------------ #
# Build graph                                                          #
# ------------------------------------------------------------------ #

def build_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)

    graph.set_entry_point("supervisor_node")

    # supervisor routes between agents
    graph.add_node("supervisor_node", lambda state: state)
    graph.add_conditional_edges("supervisor_node", supervisor, {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "end": END,
    })

    # after each agent return to supervisor
    graph.add_edge("researcher", "supervisor_node")
    graph.add_edge("analyst", "supervisor_node")
    graph.add_edge("writer", "supervisor_node")

    return graph.compile()

if __name__ == "__main__":
    graph = build_graph()

    question = "What makes Nexora's Hexapod robots unique and what was the outcome of their Solis Logistics partnership?"

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    result = graph.invoke({
        "question": question,
        "research": "",
        "insights": "",
        "final_answer": "",
        "steps": [],
    })

    print(f"\n{'='*60}")
    print(f"Final Answer:\n{result['final_answer']}")
    print(f"\nSteps taken:")
    for step in result["steps"]:
        print(f"  - {step}")
    print('='*60)