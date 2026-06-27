import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"


def ollama(prompt: str, system: str = "") -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200}
    })
    return response.json()["response"].strip()


async def run():
    server_params = StdioServerParameters(
        command="/Users/ishan/miniconda3/bin/python3",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # list available tools
            tools = await session.list_tools()
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}"
                for t in tools.tools
            )
            print(f"Connected to MCP server. Available tools:\n{tool_descriptions}\n")

            # -------------------------------------------------------- #
            # Agentic loop — LLM decides which tools to call           #
            # -------------------------------------------------------- #

            tasks = [
                "Create a note titled 'ML Systems' with content: RAG pipelines, gRPC serving, quantization, evals",
                "Create a note titled 'Interview Prep' with content: System design, prompt engineering, fine-tuning vs RAG",
                "List all my notes",
                "Search my notes for anything about RAG",
                "Read my Interview Prep note",
            ]

            for task in tasks:
                print(f"\n{'='*50}")
                print(f"Task: {task}")
                print('='*50)

                # ask LLM which tool to use
                decision = ollama(
                    f"""You have access to these tools:
{tool_descriptions}

Task: {task}

Respond with ONLY a JSON object:
{{"tool": "tool_name", "arguments": {{...}}}}""",
                    system="You are an agent. Pick the right tool and arguments. JSON only, no explanation."
                )

                print(f"LLM decision: {decision[:150]}")

                # parse and execute
                try:
                    start = decision.find("{")
                    end = decision.rfind("}") + 1
                    parsed = json.loads(decision[start:end])
                    tool_name = parsed["tool"]
                    arguments = parsed.get("arguments", {})

                    result = await session.call_tool(tool_name, arguments)
                    print(f"Result: {result.content[0].text}")
                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run())