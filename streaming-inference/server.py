import asyncio
import json
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.post("/generate")
async def generate(request: GenerateRequest):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": request.prompt,
            "stream": False,
            "options": {"num_predict": request.max_tokens, "temperature": 0.7}
        })
    data = response.json()
    return {"text": data["response"], "done": True}


async def token_stream(prompt: str, max_tokens: int):
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        done = chunk.get("done", False)
                        yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                        if done:
                            break
                    except json.JSONDecodeError:
                        pass


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    return StreamingResponse(
        token_stream(request.prompt, request.max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/generate/batch")
async def generate_batch(requests: list[GenerateRequest]):
    async def single(req: GenerateRequest):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": req.prompt,
                "stream": False,
                "options": {"num_predict": req.max_tokens, "temperature": 0.7}
            })
        return response.json()["response"]

    results = await asyncio.gather(*[single(r) for r in requests])
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)