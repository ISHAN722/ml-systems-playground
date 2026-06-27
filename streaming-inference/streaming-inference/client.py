import httpx
import json
import time

BASE_URL = "http://localhost:8001"
PROMPT = "Explain what a transformer neural network is in simple terms."


# ------------------------------------------------------------------ #
# 1. Non-streaming                                                     #
# ------------------------------------------------------------------ #

print("=" * 60)
print("1. NON-STREAMING")
print("=" * 60)
start = time.perf_counter()
response = httpx.post(f"{BASE_URL}/generate", json={
    "prompt": PROMPT, "max_tokens": 80
}, timeout=60)
elapsed = (time.perf_counter() - start) * 1000
print(f"Time to first token: {elapsed:.0f}ms (full response)")
print(f"Response: {response.json()['text'][:200]}")


# ------------------------------------------------------------------ #
# 2. Streaming — direct to Ollama (bypasses FastAPI SSE issues)       #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("2. STREAMING (direct to Ollama)")
print("=" * 60)

first_token_time = None
start = time.perf_counter()
full_text = ""
token_count = 0

with httpx.stream("POST", "http://localhost:11434/api/generate", json={
    "model": "llama3.2:latest",
    "prompt": PROMPT,
    "stream": True,
    "options": {"num_predict": 80, "temperature": 0.7}
}, timeout=60) as response:
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                done = chunk.get("done", False)

                if token and first_token_time is None:
                    first_token_time = (time.perf_counter() - start) * 1000
                    print(f"Time to first token: {first_token_time:.0f}ms")
                    print("Streaming: ", end="", flush=True)

                print(token, end="", flush=True)
                full_text += token
                token_count += 1

                if done:
                    break
            except json.JSONDecodeError:
                pass

total_ms = (time.perf_counter() - start) * 1000
print(f"\nTotal time: {total_ms:.0f}ms, tokens: {token_count}")


# ------------------------------------------------------------------ #
# 3. Concurrent batch                                                  #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("3. CONCURRENT BATCH (3 prompts simultaneously)")
print("=" * 60)

prompts = [
    {"prompt": "What is 2+2?", "max_tokens": 20},
    {"prompt": "Name the capital of France.", "max_tokens": 20},
    {"prompt": "What color is the sky?", "max_tokens": 20},
]

start = time.perf_counter()
response = httpx.post(f"{BASE_URL}/generate/batch", json=prompts, timeout=60)
elapsed = (time.perf_counter() - start) * 1000

results = response.json()["results"]
for i, (prompt, result) in enumerate(zip(prompts, results)):
    print(f"\nPrompt {i+1}: \"{prompt['prompt']}\"")
    print(f"Response: {result.strip()[:100]}")

print(f"\nAll 3 concurrent requests: {elapsed:.0f}ms total")