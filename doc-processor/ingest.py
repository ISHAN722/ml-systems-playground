import fitz
import os

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = "".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def load_documents(docs_dir: str) -> list[dict]:
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(docs_dir, filename)
            print(f"Reading {filename}...")
            text = extract_text(path)
            chunks = chunk_text(text)
            documents.append({
                "filename": filename,
                "path": path,
                "text": text,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "num_chars": len(text)
            })
            print(f"  -> {len(chunks)} chunks, {len(text):,} chars")
    return documents

if __name__ == "__main__":
    docs = load_documents("docs/")
    print(f"\nLoaded {len(docs)} documents")
    for d in docs:
        print(f"  {d['filename']}: {d['num_chunks']} chunks")