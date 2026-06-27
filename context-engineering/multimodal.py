import requests
import base64
import json
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def vision_query(image_path: str, prompt: str) -> tuple[str, float]:
    """Send image + text prompt to moondream"""
    image_b64 = encode_image(image_path)
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json = {
        "model": "moondream",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200}
    })
    elpased = (time.perf_counter() - start) * 1000
    return response.json()["response"].strip(), elpased

def extract_structured(image_path: str) -> dict:
    """Extract structured information from image"""
    raw, _ = vision_query(
        image_path,
        """Analyze this image and respond with a JSON object containing:
        {
        "description": "one sentence description of what this image shows",
        "text_content": "any text visible in the image, or null if none",
        "main_objects": ["list of main objects or elements"],
        "image_type": "one of: photo, diagram, screenshot, chart, document, other"
        }
        JSON only, no explanation."""
    )
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"raw": raw}
    
if __name__ == "__main__":
    image_path = "book.png"

    if not Path(image_path).exists():
        print(f"Image not found at {image_path}. Update image_path variable.")
        exit(1)

    print(f"Image: {image_path}")
    print("=" * 60)

    # 1. basic description
    print("\n[1] Basic description:")
    desc, ms = vision_query(image_path, "Describe this image in one sentence.")
    print(f"  {desc} ({ms:.0f}ms)")

    # 2. specific question
    print("\n[2] Specific question:")
    answer, ms = vision_query(image_path, "What colors are most prominent in this image?")
    print(f"  {answer} ({ms:.0f}ms)")

    # 3. structured extraction
    print("\n[3] Structured extraction:")
    structured = extract_structured(image_path)
    print(f"  {json.dumps(structured, indent=2)}")

    # 4. combine with RAG — use visual description to search knowledge base
    print("\n[4] Vision + RAG pipeline:")
    description, _ = vision_query(image_path, "Describe this image briefly.")
    print(f"  Visual description: {description[:100]}")
    print(f"  → This description could now be used as a RAG query to find related documents")
    print(f"  → Production use: process product images, find matching specs in knowledge base")