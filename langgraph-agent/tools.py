import sys
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "../rag-demo")

# load FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../rag-demo/index/faiss.index")
with open("../rag-demo/index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def search_docs(query: str) -> str:
    """Search the document index for relevant information."""
    emb = embedder.encode([query]).astype("float32")
    _, indices = index.search(emb, 2)
    results = [chunks[i] for i in indices[0]]
    return "\n---\n".join(results)

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def summarize(text: str) -> str:
    """Summarize a long piece of text."""
    response = requests.post("http://localhost:11434/api/generate", json = {
        "model": "llama3.2:latest",
        "prompt": f"Summarize this in 2 sentences:\n{text}",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 150}
    })
    return response.json()["response"].strip()

# tools registry - maps tool name to function
TOOLS = {
    "search_docs": search_docs,
    "calculate": calculate,
    "summarize": summarize
}

TOOL_DESCRIPTIONS = """
- search_docs(query): Search the document index for information about Nexora Robotics
- calculate(expression): Evaluate a math expression e.g. calculate(150 * 2)
- summarize(text): Summarize a long piece of text
"""