import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

def ollama(prompt: str, system: str = "") -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 150}
    })

    elapsed = (time.perf_counter() - start) * 1000
    return response.json()["response"].strip(), elapsed

def token_estimate(text: str) -> str:
    return int(len(text.split()) * 1.3)

# ------------------------------------------------------------------ #
# Knowledge base — mix of relevant and irrelevant chunks              #
# ------------------------------------------------------------------ #

ALL_CHUNKS = [
    # relevant
    "The Hexapod X1 can carry up to 150 kg and operates for 14 hours on a single charge.",
    "PathWeave is Nexora's proprietary navigation system that builds real-time 3D maps using lidar and thermal cameras.",
    "In 2034, Nexora partnered with Solis Logistics, reducing average order fulfillment time by 37%.",
    # noise — related topic but doesn't answer the question
    "Nexora Robotics was founded in 2031 in Austin, Texas, by former aerospace engineer Maya Chen.",
    "The company's headquarters is a converted aircraft hangar nicknamed 'The Nest'.",
    "Nexora's long-term goal is to develop Hexapods capable of operating in extreme environments.",
    "Maya Chen has spoken at several robotics conferences about the future of warehouse automation.",
    "The Hexapod design was inspired by biological hexapods found in nature.",
]

QUESTION = "What are the technical specifications of the Hexapod X1?"

SYSTEM = "You are a helpful assistant. Answer based only on the provided context. If the answer is not in the context, say so."

# ------------------------------------------------------------------ #
# Configuration 1: No context                                          #
# ------------------------------------------------------------------ #

def config_1_no_context():
    answer, ms = ollama(f"Question: {QUESTION}", system=SYSTEM)
    return answer, ms, 0

# ------------------------------------------------------------------ #
# Configuration 2: Full context, unordered, unstructured              #
# ------------------------------------------------------------------ #

def config_2_full_context():
    context = "\n".join(ALL_CHUNKS)
    prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}"
    answer, ms = ollama(prompt, system=SYSTEM)
    return answer, ms, token_estimate(prompt)

# ------------------------------------------------------------------ #
# Configuration 3: Pruned context (only relevant chunks)              #
# ------------------------------------------------------------------ #

def config_3_pruned():
    # only the 3 chunks that actually answer the question
    relevant = ALL_CHUNKS[:3]
    context = "\n".join(relevant)
    prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}"
    answer, ms = ollama(prompt, system=SYSTEM)
    return answer, ms, token_estimate(prompt)

# ------------------------------------------------------------------ #
# Configuration 4: Structured + ordered (best practice)               #
# ------------------------------------------------------------------ #

def config_4_structured():
    # most relevant first, structured delimiters
    relevant = ALL_CHUNKS[:3]
    context = "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(relevant))
    prompt = f"""<documents>
{context}
</documents>

<question>
{QUESTION}
</question>

Answer using only the documents above. Cite which document number supports each fact."""
    answer, ms = ollama(prompt, system=SYSTEM)
    return answer, ms, token_estimate(prompt)

# ------------------------------------------------------------------ #
# Run all configs                                                      #
# ------------------------------------------------------------------ #

print("=" * 60)
print(f"Question: {QUESTION}")
print("=" * 60)

configs = [
    ("1. No context", config_1_no_context),
    ("2. Full context (8 chunks, unstructured)", config_2_full_context),
    ("3. Pruned context (3 relevant chunks)", config_3_pruned),
    ("4. Structured + ordered", config_4_structured),
]

for name, fn in configs:
    print(f"\n[{name}]")
    answer, ms, tokens = fn()
    print(f"Tokens: ~{tokens} | Latency: {ms:.0f}ms")
    print(f"Answer: {answer[:250]}")
    print()