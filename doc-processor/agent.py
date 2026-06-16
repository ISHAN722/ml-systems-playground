import requests
import json
from ingest import chunk_text

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

def call_ollama(prompt: str, system:str = "") -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()

def extract_json(text: str) -> dict:
    # strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        candidate = json.loads(text[start:end])
        if isinstance(candidate, dict) and len(candidate) > 1:
            return candidate
    except json.JSONDecodeError:
        return {}
    return {}

def process_document(doc: dict) -> dict:
    print(f"\nProcessing: {doc['filename']}")

    # use first 3 chunks as context (enough for summary + extraction)
    context = "\n\n".join(doc["chunks"][:3])

    system_prompt = system_prompt = """You are a JSON extractor. Output only raw JSON. 
                No thinking. No explanation. No markdown. 
                Start your response with { and end with }."""
    prompt = prompt = f"""You are analyzing an academic paper. Here is an excerpt:

{context[:1500]}

Based on this excerpt, fill in the following details:
- What is the title of this paper?
- Write a 2-3 sentence summary of what this paper proposes.
- List the key technical methods or algorithms used.
- List the main research topics.
- Classify into one category: nlp, systems, optimization, computer_vision, or other.
- What is novel about this paper in one sentence?

Respond with a JSON object with keys: title, summary, methods, topics, category, novelty."""

    raw = call_ollama(prompt, system_prompt)
    result = extract_json(raw)

    if not result:
        print(f"  Warning: could not parse JSON for {doc['filename']}")
        result = {"error": "parse_failed", "raw": raw[:200]}

    result["filename"] = doc["filename"]
    result["num_chunks"] = doc["num_chunks"]
    result["num_chars"] = doc["num_chars"]

    print(f"  ✓ Done — category: {result.get('category', 'unknown')}")
    return result


if __name__ == "__main__":
    from ingest import load_documents
    docs = load_documents("docs/")
    results = [process_document(doc) for doc in docs]
    print("\n\nAll results:")
    print(json.dumps(results, indent=2))
