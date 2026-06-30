# /// script
# dependencies = [
#   "mcp>=1.9.0",
#   "aiohttp>=3.9",
# ]
# ///
"""MCP bridge — runs inside the Harbor container.

Exposes mcp-grafana tools over HTTP so agent-server can call them from the host.

Endpoints:
  GET  /health  — liveness check
  GET  /tools   — list available MCP tools
  POST /call    — execute a tool by name

Auth: X-Bridge-Token header must match BRIDGE_TOKEN env var (if set).
Config:
  BRIDGE_PORT  — port to bind on (default 8099)
  BRIDGE_TOKEN — shared secret for request auth
  MCP_URL      — mcp-grafana URL (default http://o11y-stack:8080/mcp)
"""

import asyncio
import json
import logging
import os

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8099"))
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")
MCP_URL = os.environ.get("MCP_URL", "http://o11y-stack:8080/mcp")

_tools_cache: list[dict] | None = None
_tools_lock = asyncio.Lock()


def _check_auth(request: web.Request) -> bool:
    if not BRIDGE_TOKEN:
        return True
    return request.headers.get("X-Bridge-Token") == BRIDGE_TOKEN


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_list_tools(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    global _tools_cache
    async with _tools_lock:
        if _tools_cache is None:
            logging.info("Discovering tools from MCP server at %s", MCP_URL)
            _tools_cache = await _discover_tools()
            logging.info("Discovered %d tools: %s", len(_tools_cache), [t["name"] for t in _tools_cache])

    return web.json_response({"tools": _tools_cache})


async def handle_call_tool(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = body.get("name")
    arguments = body.get("arguments") or {}

    if not name:
        return web.json_response({"error": "missing name"}, status=400)

    try:
        logging.info("Calling tool %s with args: %s", name, json.dumps(arguments)[:200])
        output = await _call_tool(name, arguments)
        logging.info("Tool %s returned %d chars", name, len(output))
        return web.json_response({"output": output})
    except Exception as e:
        logging.error("Tool %s failed: %s", name, e)
        return web.json_response({"error": str(e)}, status=500)


async def _discover_tools() -> list[dict]:
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if t.inputSchema else {},
                }
                for t in result.tools
            ]


async def _call_tool(name: str, arguments: dict) -> str:
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts)


app = web.Application()
app.router.add_get("/health", handle_health)
app.router.add_get("/tools", handle_list_tools)
app.router.add_post("/call", handle_call_tool)

if __name__ == "__main__":
    print(f"MCP bridge starting on 0.0.0.0:{BRIDGE_PORT}, MCP_URL={MCP_URL}", flush=True)
    web.run_app(app, host="0.0.0.0", port=BRIDGE_PORT, print=None)
