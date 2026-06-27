import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def ollama(prompt: str, system: str = "") -> str:
    response = requests.post(OLLAMA_URL, json = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200}
    })
    return response.json()["response"].strip()

# ------------------------------------------------------------------ #
# Strategy 1: Full history                                             #
# ------------------------------------------------------------------ #

class FullHistoryMemory:
    def __init__(self):
        self.history = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)

    def token_estimate(self) -> int:
        return len(self.get_context().split()) * 1.3 # rough tokens

# ------------------------------------------------------------------ #
# Strategy 2: Summarization memory                                     #
# ------------------------------------------------------------------ #

class SummarizationMemory:
    def __init__(self, max_turns: int = 4):
        self.summary = ""
        self.recent = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.max_turns * 2:
            # compress oldest turns into summary
            to_compress = self.recent[:-self.max_turns * 2]
            compress_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
            new_summary = ollama(
                f"Existing summary: {self.summary}\n\nNew conversation to add:\n{compress_text}\n\nUpdate the summary to include the new information. Be concise.",
                system="You are a summarizer. Keep summaries under 100 words."
            )
            self.summary = new_summary
            self.recent = self.recent[-self.max_turns * 2:]

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary]: {self.summary}")
        parts.append("\n".join(f"{m['role']}: {m['content']}" for m in self.recent))
        return "\n".join(parts)

# ------------------------------------------------------------------ #
# Strategy 3: Vector memory                                            #
# ------------------------------------------------------------------ #

class VectorMemory:
    def __init__(self, top_k: int = 2):
        self.memories = []
        self.embeddings = []
        self.index = None
        self.top_k = top_k
        self.dim = 384

    def add(self, role: str, content: str):
        memory = f"{role}: {content}"
        self.memories.append(memory)
        emb = embedder.encode([memory]).astype("float32")
        self.embeddings.append(emb[0])
        # rebuild index
        embeddings_array = np.array(self.embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings_array)

    def get_context(self, query: str) -> str:
        if not self.index or len(self.memories) == 0:
            return ""
        query_emb = embedder.encode([query]).astype("float32")
        k= min(self.top_k, len(self.memories))
        _, indices = self.index.search(query_emb, k)
        relevant = [self.memories[i] for i in indices[0]]
        return "[Relevant memories]:\n" + "\n".join(relevant)

# ------------------------------------------------------------------ #
# Strategy 4: Structured/external memory                              #
# ------------------------------------------------------------------ #

class StructuredMemory:
    def __init__(self):
        self.facts = {}   # key value store of extracted facts

    def extract_and_store(self, content: str):
        """Ask LLM to extract structured facts from conversation."""
        raw = ollama(
            f"Extract any personal facts or preferences from this message as JSON key-value pairs. Return {{}} if none.\nMessage: {content}",
            system="Extract facts only. Return valid JSON with no explanation. Example: {\"name\": \"John\", \"preference\": \"vegetarian\"}"
        )
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            facts = json.loads(raw[start:end])
            self.facts.update(facts)
        except:
            pass

    def get_context(self) -> str:
        if not self.facts:
            return ""
        return "[Known facts about user]: " + json.dumps(self.facts)


# ------------------------------------------------------------------ #
# Demo conversation                                                    #
# ------------------------------------------------------------------ #

conversation = [
    ("user", "Hi, my name is Ishan and I'm studying ML systems"),
    ("assistant", "Great to meet you Ishan! ML systems is a fascinating field."),
    ("user", "I've been building RAG pipelines and gRPC servers"),
    ("assistant", "That's impressive! RAG and gRPC are core production skills."),
    ("user", "I also prefer Python over Java for ML work"),
    ("assistant", "Python is definitely the dominant language in ML."),
    ("user", "What do you remember about me?"),
]

print("=" * 60)
print("MEMORY STRATEGY COMPARISON")
print("=" * 60)

full = FullHistoryMemory()
summ = SummarizationMemory(max_turns=2)
vec = VectorMemory(top_k=2)
struct = StructuredMemory()

for role, content in conversation[:-1]:
    full.add(role, content)
    summ.add(role, content)
    vec.add(role, content)
    if role == "user":
        struct.extract_and_store(content)

final_question = conversation[-1][1]

print(f"\nFinal question: \"{final_question}\"")
print(f"\nToken estimate (full history): ~{full.token_estimate():.0f} tokens")

system = "You are a helpful assistant. Answer based only on what you know about the user from the context provided."

print("\n[1] Full history response:")
ctx1 = full.get_context() + f"\nuser: {final_question}"
r1 = ollama(ctx1, system=system)
print(r1[:300])

print("\n[2] Summarization memory response:")
ctx2 = summ.get_context() + f"\nuser: {final_question}"
r2 = ollama(ctx2, system=system)
print(r2[:300])

print("\n[3] Vector memory response:")
ctx3 = vec.get_context(final_question) + f"\nuser: {final_question}"
r3 = ollama(ctx3, system=system)
print(r3[:300])

print("\n[4] Structured memory response:")
ctx4 = struct.get_context() + f"\nuser: {final_question}"
r4 = ollama(ctx4, system=system)
print(r4[:300])

print("\n[Extracted structured facts]:")
print(struct.facts)