import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# load embedding model (small, fast, runs on CPU)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# read and chunk the document
with open("docs/sample.txt", "r") as f:
    text = f.read()

# simple chunking: split on paragraphs
chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print(f"Number of chunks: {len(chunks)}")

# embed chunks
print("Embedding chunks...")
embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# save index and chunks
faiss.write_index(index, "index/faiss.index")
with open("index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Index built with {index.ntotal} vectors of dimension {dim}")
print("Saved to index/")