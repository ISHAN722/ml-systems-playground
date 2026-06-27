import json
import os
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"


def ollama(prompt: str, system: str = "") -> str:
    response = requests.post(OLLAMA_URL, json = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 500}
    })
    return response.json()["response"].strip()

def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("[") if "[" in text else text.find("{")
    end = text.rfind("]") + 1 if "[" in text else text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except:
        return None


# ------------------------------------------------------------------ #
# 1. Generate Q&A pairs from a topic                                   #
# ------------------------------------------------------------------ #

def generate_qa_pairs(topic: str, n: int = 5) -> list[dict]:
    prompt = f"""Generate {n} question-answer pairs about "{topic}" for an ML engineer interview.

Return a JSON array with exactly {n} objects, each with:
- "question": a specific technical question
- "answer": a concise correct answer (1-2 sentences)
- "difficulty": "easy", "medium", or "hard"
- "keywords": list of 3-4 key terms that should appear in a correct answer

Return only the JSON array, no explanation."""

    raw = ollama(prompt, system="You are an ML interview question generator. Return only valid JSON arrays.")
    result = extract_json(raw)
    if result and isinstance(result, list):
        return result
    return []


# ------------------------------------------------------------------ #
# 2. Generate adversarial/edge cases                                   #
# ------------------------------------------------------------------ #

def generate_edge_cases(topic: str, n: int = 3) -> list[dict]:
    prompt = f"""Generate {n} tricky or adversarial questions about "{topic}" that might trip up a candidate.

These should be questions where:
- The obvious answer is wrong
- Common misconceptions are tested
- Edge cases or nuances matter

Return a JSON array with {n} objects, each with:
- "question": the tricky question
- "common_wrong_answer": what most people incorrectly say
- "correct_answer": the actual correct answer
- "why_tricky": one sentence explaining why this trips people up

Return only the JSON array."""

    raw = ollama(prompt, system="You are an ML interview expert. Return only valid JSON arrays.")
    result = extract_json(raw)
    if result and isinstance(result, list):
        return result
    return []


# ------------------------------------------------------------------ #
# 3. Generate variations of existing questions                         #
# ------------------------------------------------------------------ #

def generate_variations(question: str, n: int = 3) -> list[str]:
    prompt = f"""Generate {n} different ways to ask this question:
"{question}"

Return a JSON array of {n} strings, each a variation of the question.
Keep the same meaning but vary the phrasing, perspective, or framing.
Return only the JSON array."""

    raw = ollama(prompt, system="Return only valid JSON arrays of strings.")
    result = extract_json(raw)
    if result and isinstance(result, list):
        return result
    return []


# ------------------------------------------------------------------ #
# Run generation pipeline                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    topic = "RAG (Retrieval Augmented Generation)"
    output_file = "synthetic_dataset.json"

    print(f"Generating synthetic dataset for: {topic}")
    print("=" * 60)

    # generate Q&A pairs
    print("\n[1] Generating Q&A pairs...")
    qa_pairs = generate_qa_pairs(topic, n=5)
    print(f"Generated {len(qa_pairs)} Q&A pairs")
    for i, qa in enumerate(qa_pairs[:2]):  # show first 2
        print(f"  Q{i+1}: {qa.get('question', '')[:80]}")
        print(f"  A{i+1}: {qa.get('answer', '')[:80]}")
        print(f"  Difficulty: {qa.get('difficulty', '')} | Keywords: {qa.get('keywords', [])}")

    # generate edge cases
    print("\n[2] Generating edge cases...")
    edge_cases = generate_edge_cases(topic, n=3)
    print(f"Generated {len(edge_cases)} edge cases")
    for ec in edge_cases[:2]:
        print(f"  Q: {ec.get('question', '')[:80]}")
        print(f"  Wrong: {ec.get('common_wrong_answer', '')[:60]}")
        print(f"  Correct: {ec.get('correct_answer', '')[:60]}")
        print(f"  Why tricky: {ec.get('why_tricky', '')[:60]}")

    # generate variations of first question
    if qa_pairs:
        print("\n[3] Generating question variations...")
        first_q = qa_pairs[0].get("question", "")
        variations = generate_variations(first_q, n=3)
        print(f"Original: {first_q}")
        for i, v in enumerate(variations):
            print(f"  Variation {i+1}: {v}")

    # save full dataset
    dataset = {
        "topic": topic,
        "generated_at": datetime.now().isoformat(),
        "model": MODEL,
        "qa_pairs": qa_pairs,
        "edge_cases": edge_cases,
        "stats": {
            "total_qa": len(qa_pairs),
            "total_edge_cases": len(edge_cases),
            "difficulty_breakdown": {
                "easy": sum(1 for q in qa_pairs if q.get("difficulty") == "easy"),
                "medium": sum(1 for q in qa_pairs if q.get("difficulty") == "medium"),
                "hard": sum(1 for q in qa_pairs if q.get("difficulty") == "hard"),
            }
        }
    }

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n[4] Dataset saved to {output_file}")
    print(f"Stats: {dataset['stats']}")