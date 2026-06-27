import json
import hashlib
import os
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
REGISTRY_FILE = "prompt_registry.json"
EVAL_FILE = "eval_results.json"


# ------------------------------------------------------------------ #
# Prompt registry — versioned prompt storage                          #
# ------------------------------------------------------------------ #

def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"prompts": {}, "history": []}

def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent = 2)

def register_prompt(name: str, system_prompt: str, description: str = "") -> str:
    """Register a prompt version, return verison hash."""
    registry = load_registry()
    version_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]

    if name not in registry["prompts"]:
        registry["prompts"][name] = []

    version = {
        "hash": version_hash,
        "system_prompt": system_prompt,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "version": len(registry["prompts"][name]) + 1
    }

    registry["prompts"][name].append(version)
    registry["history"].append({
        "action": "register",
        "name": name,
        "hash": version_hash,
        "timestamp": datetime.now().isoformat()
    })

    save_registry(registry)
    print(f"Registered '{name}' v{version['version']} [{version_hash}]")
    return version_hash

def get_latest_prompt(name: str) -> dict:
    registry = load_registry()
    versions = registry["prompts"].get(name, [])
    if not versions:
        raise ValueError(f"Prompt '{name}' not found")
    return versions[-1]

def get_prompt_version(name: str, version: int) -> dict:
    registry = load_registry()
    versions = registry["prompts"].get(name, [])
    return versions[version - 1]

# ------------------------------------------------------------------ #
# Eval runner                                                          #
# ------------------------------------------------------------------ #

GOLDEN_DATASET = [
    {
        "input": "What is machine learning?",
        "expected_keywords": ["data", "pattern", "predict", "learn"],
        "expected_length": (20, 150)
    },
    {
        "input": "Explain overfitting in one sentence.",
        "expected_keywords": ["train", "generalize", "data"],
        "expected_length": (10, 80)
    },
    {
        "input": "What is a transformer?",
        "expected_keywords": ["attention", "neural", "model"],
        "expected_length": (20, 150)
    },
    {
        "input": "What is gradient descent?",
        "expected_keywords": ["loss", "minimize", "gradient", "update"],
        "expected_length": (20, 150)
    },
    {
        "input": "What is the difference between precision and recall?",
        "expected_keywords": ["true", "positive", "false"],
        "expected_length": (30, 200)
    },
]

def run_eval(system_prompt: str, prompt_name: str, version_hash: str) -> dict:
    """Run golden dataset eval against a prompt version."""
    results= []
    total_score = 0

    for item in GOLDEN_DATASET:
        response = requests.post(OLLAMA_URL, json = {
            "model": MODEL,
            "prompt": item["input"],
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 150}
        })

        answer = response.json()["response"].strip()

        # score: keyword coverage + length check
        words = answer.lower().split()
        keyword_hits = sum(1 for kw in item["expected_keywords"] if kw in words)
        keyword_score = keyword_hits / len(item["expected_keywords"])

        min_len, max_len = item["expected_length"]
        length_ok = min_len <= len(words) <= max_len
        length_score = 1.0 if length_ok else 0.0

        score = (keyword_score * 0.7 +length_score * 0.3)
        total_score += score

        results.append({
            "input": item["input"],
            "answer": answer[:100],
            "keyword_score": round(keyword_score, 2),
            "length_score": length_score,
            "score": round(score, 2)
        })

    avg_score = total_score / len(GOLDEN_DATASET)

    eval_result = {
        "prompt_name": prompt_name,
        "version_hash": version_hash,
        "timestamp": datetime.now().isoformat(),
        "avg_score": round(avg_score, 3),
        "results": results
    }

    # save eval results
    all_evals = []
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE) as f:
            all_evals = json.load(f)
    all_evals.append(eval_result)
    with open(EVAL_FILE, "w") as f:
        json.dump(all_evals, f, indent = 2)

    return eval_result

def compare_versions(name: str, v1: int, v2: int):
    """Compare eval scores between two prompt versions."""
    all_evals = []
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE) as f:
            all_evals = json.load(f)

    v1_hash = get_prompt_version(name, v1)["hash"]
    v2_hash = get_prompt_version(name, v2)["hash"]

    v1_eval = next((e for e in all_evals if e["version_hash"] == v1_hash), None)
    v2_eval = next((e for e in all_evals if e["version_hash"] == v2_hash), None)

    if not v1_eval or not v2_eval:
        print("Evals not found — run evals first")
        return

    delta = v2_eval["avg_score"] - v1_eval["avg_score"]
    print(f"\n{'='*50}")
    print(f"Prompt comparison: '{name}'")
    print(f"v{v1} [{v1_hash}]: {v1_eval['avg_score']:.3f}")
    print(f"v{v2} [{v2_hash}]: {v2_eval['avg_score']:.3f}")
    print(f"Delta: {delta:+.3f} ({'IMPROVED' if delta > 0 else 'REGRESSED' if delta < 0 else 'NO CHANGE'})")
    print(f"{'='*50}")


# ------------------------------------------------------------------ #
# Demo: register two versions, eval both, compare                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # version 1 — basic prompt
    v1_prompt = "You are a helpful assistant. Answer questions clearly."
    hash1 = register_prompt("ml-explainer", v1_prompt, "Basic prompt")

    # version 2 — improved prompt
    v2_prompt = """You are an expert ML educator. When answering questions:
- Be concise and precise
- Use technical terms correctly
- Give concrete examples where helpful
- Keep answers under 3 sentences"""
    hash2 = register_prompt("ml-explainer", v2_prompt, "Improved with constraints")

    print("\nRunning evals on v1...")
    r1 = run_eval(v1_prompt, "ml-explainer", hash1)
    print(f"v1 score: {r1['avg_score']:.3f}")

    print("\nRunning evals on v2...")
    r2 = run_eval(v2_prompt, "ml-explainer", hash2)
    print(f"v2 score: {r2['avg_score']:.3f}")

    compare_versions("ml-explainer", 1, 2)

    print("\nPrompt registry:")
    registry = load_registry()
    for name, versions in registry["prompts"].items():
        print(f"  {name}: {len(versions)} version(s)")
        for v in versions:
            print(f"    v{v['version']} [{v['hash']}] — {v['description']}")