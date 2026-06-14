import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# load index and chunks
index = faiss.read_index("index/faiss.index")
with open("index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading TinyLlama...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
model.eval()
print("Ready.\n")

def retrieve(query, k = 2):
    query_emb = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

def generate(prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors = "pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def rag_answer(question):
    retrieved = retrieve(question)
    context = "\n".join(retrieved)

    prompt = f"<|system|>\nYou are a helpful assistant. Use the following context to answer the question,\nContext:\n{context}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"

    print("--- Retrieved context ---")
    for r in retrieved:
        print(r[:100] + "...")
    print("\n--- Answer ---")

    response = generate(prompt)
    # print only the assistant's reply
    answer = response.split("<|assistant|>")[-1].strip()
    print(answer)

if __name__ == "__main__":
    question = "How much weight can the Hexapod X1 carry and how long does its battery last?"
    rag_answer(question)