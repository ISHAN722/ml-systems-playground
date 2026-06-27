import json
import os
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

NOTES_FILE = "notes.json"

def load_notes() -> dict:
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE) as f:
            return json.load(f)
    return {}

def save_notes(notes: dict):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

app = Server("notes-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_note",
            description="Create a new note with a title and content",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["title", "content"]
            }
        ),
        Tool(
            name="read_note",
            description="Read a note by title",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"}
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="list_notes",
            description="List all available note titles",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="search_notes",
            description="Search notes by keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"}
                },
                "required": ["keyword"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    notes = load_notes()

    if name == "create_note":
        notes[arguments["title"]] = arguments["content"]
        save_notes(notes)
        return [TextContent(type="text", text=f"Note '{arguments['title']}' created.")]

    elif name == "read_note":
        title = arguments["title"]
        if title in notes:
            return [TextContent(type="text", text=f"{title}:\n{notes[title]}")]
        return [TextContent(type="text", text=f"Note '{title}' not found.")]

    elif name == "list_notes":
        if not notes:
            return [TextContent(type="text", text="No notes found.")]
        return [TextContent(type="text", text="Notes:\n" + "\n".join(f"- {t}" for t in notes))]

    elif name == "search_notes":
        kw = arguments["keyword"].lower()
        matches = {t: c for t, c in notes.items() if kw in t.lower() or kw in c.lower()}
        if not matches:
            return [TextContent(type="text", text=f"No notes found for '{kw}'.")]
        result = "\n\n".join(f"{t}:\n{c}" for t, c in matches.items())
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())