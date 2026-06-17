import json
import os
import sys
import requests

sys.path.insert(0, "../rag-demo")

import faiss
import pickle
from sentence_transformers import SentenceTransformer

# load RAG system
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../rag-demo/index/faiss.index")
with open("../rag-demo/index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def retrieve(query, k=2):
    emb = embedder.encode([query]).astype("float32")
    _, indices = index.search(emb, k)
    return [chunks[i] for i in indices[0]]

def ollama(prompt, system = ""):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:latest",
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300}
    })
    return response.json()["response"].strip()


def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely based only on the context:"
    return ollama(prompt)


# ------------------------------------------------------------------ #
# Eval metrics (LLM-as-judge)                                         #
# ------------------------------------------------------------------ #

def eval_faithfulness(question, answer, context):
    """Is the answer grounded in the context, or is the model hallucinating?"""
    prompt = f"""You are an evaluation judge. Given a context, a question, and an answer, 
determine if the answer is fully supported by the context.

Context: {context}
Question: {question}
Answer: {answer}

Reply with a JSON object: {{"score": <0.0 to 1.0>, "reason": "<one sentence>"}}
1.0 = fully supported, 0.0 = not supported at all."""
    raw = ollama(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"score": 0.0, "reason": "parse failed"}


def eval_relevancy(question, context):
    """Is the retrieved context actually relevant to the question?"""
    prompt = f"""You are an evaluation judge. Given a question and retrieved context,
determine how relevant the context is for answering the question.

Question: {question}
Context: {context}

Reply with a JSON object: {{"score": <0.0 to 1.0>, "reason": "<one sentence>"}}
1.0 = highly relevant, 0.0 = completely irrelevant."""
    raw = ollama(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"score": 0.0, "reason": "parse failed"}

def eval_correctness(question, answer, ground_truth):
    """Does the answer match the ground truth?"""
    prompt = f"""You are an evaluation judge. Compare an answer to a ground truth.

Question: {question}
Ground truth: {ground_truth}
Answer: {answer}

Reply with a JSON object: {{"score": <0.0 to 1.0>, "reason": "<one sentence>"}}
1.0 = answer matches ground truth perfectly, 0.0 = completely wrong."""
    raw = ollama(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"score": 0.0, "reason": "parse failed"}


# ------------------------------------------------------------------ #
# Run evals                                                            #
# ------------------------------------------------------------------ #

with open("golden_dataset.json") as f:
    golden = json.load(f)

results = []
print("Running evals...\n")

for item in golden:
    q = item["question"]
    gt = item["ground_truth"]

    retrieved = retrieve(q)
    context = "\n".join(retrieved)
    answer = generate_answer(q, context)

    faith = eval_faithfulness(q, answer, context)
    rel = eval_relevancy(q, context)
    corr = eval_correctness(q, context, gt)

    result = {
        "question": q,
        "answer": answer,
        "ground_truth": gt,
        "faithfulness": faith,
        "relevancy": rel,
        "correctness": corr,
    }
    results.append(result)

    print(f"Q: {q}")
    print(f"A: {answer[:100]}")
    print(f"Faithfulness: {faith['score']} — {faith['reason']}")
    print(f"Relevancy:    {rel['score']} — {rel['reason']}")
    print(f"Correctness:  {corr['score']} — {corr['reason']}")
    print()

# summary
avg_faith = sum(r["faithfulness"]["score"] for r in results) / len(results)
avg_rel = sum(r["relevancy"]["score"] for r in results) / len(results)
avg_corr = sum(r["correctness"]["score"] for r in results) / len(results)

print("=" * 40)
print(f"Avg Faithfulness: {avg_faith:.2f}")
print(f"Avg Relevancy:    {avg_rel:.2f}")
print(f"Avg Correctness:  {avg_corr:.2f}")
print(f"Overall:          {(avg_faith + avg_rel + avg_corr) / 3:.2f}")

# save
os.makedirs("results", exist_ok=True)
with open("results/eval_results.json", "w") as f:
    json.dump({"summary": {
        "avg_faithfulness": avg_faith,
        "avg_relevancy": avg_rel,
        "avg_correctness": avg_corr,
    }, "details": results}, f, indent=2)
print("\nSaved to results/eval_results.json")