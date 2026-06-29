# /// script
# dependencies = [
#   "mcp>=1.9.0",
#   "httpx>=0.27",
# ]
# ///
"""Corvo o11y-bench runner — executes inside the Harbor container.

Connects to mcp-grafana via streamable-http, runs an agent loop by calling
Corvo agent-server's /internal/o11y-bench/complete endpoint for each LLM step,
executes tool calls via MCP, and writes an ATIF-v1.7 trajectory.

Config via env vars:
  CORVO_URL     — Corvo agent-server base URL (default: http://host.docker.internal:4111)
  CORVO_SECRET  — shared secret for X-O11Y-Bench-Secret header
  MCP_URL       — mcp-grafana URL (default: http://o11y-stack:8080/mcp)
  O11Y_SCENARIO_TIME_ISO — scenario clock override
"""

import asyncio
import copy
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def scenario_clock_iso() -> str:
    env_ts = os.environ.get("O11Y_SCENARIO_TIME_ISO", "").strip()
    if env_ts:
        return env_ts
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_system_prompt() -> str:
    return Path("/app/system_prompt.txt").read_text().strip()


def _load_task_prompt_template() -> str:
    return Path("/app/task_prompt.txt").read_text().strip()


SYSTEM_PROMPT = _load_system_prompt()
TASK_PROMPT_TEMPLATE = _load_task_prompt_template()
MAX_AGENT_STEPS = 50

# mimo-v2.5-pro pricing (Xiaomi overseas list price = Standard token-plan rate)
# Source: inferfix-web PR #309
_MIMO_INPUT_PER_TOKEN = 4.35e-7   # $0.435 / 1M
_MIMO_OUTPUT_PER_TOKEN = 8.7e-7   # $0.870 / 1M


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * _MIMO_INPUT_PER_TOKEN + output_tokens * _MIMO_OUTPUT_PER_TOKEN


def relax_mcp_tool_input_schema_for_llm(schema: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties")
            if not props:
                node.setdefault("additionalProperties", True)
        for v in node.values():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, list):
                for item in v:
                    walk(item)

    walk(out)
    return out


async def discover_tools(session: Any) -> list[dict[str, Any]]:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        raw_schema = tool.inputSchema if tool.inputSchema else {}
        schema = relax_mcp_tool_input_schema_for_llm(raw_schema)
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return tools


async def call_mcp_tool(session: Any, name: str, arguments: dict[str, Any]) -> str:
    result = await session.call_tool(name, arguments)
    parts = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def make_atif_step(
    step_id: int,
    source: str,
    message: str,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "message": message,
    }
    if tool_calls is not None:
        step["tool_calls"] = tool_calls
    if observation is not None:
        step["observation"] = observation
    if metrics is not None:
        step["metrics"] = metrics
    return step


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {}
    return {}


async def call_corvo_complete(
    client: Any,
    corvo_url: str,
    corvo_secret: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """POST to Corvo /internal/o11y-bench/complete and return parsed response."""
    resp = await client.post(
        f"{corvo_url}/internal/o11y-bench/complete",
        json={"messages": messages, "tools": tools},
        headers={"X-O11Y-Bench-Secret": corvo_secret},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


async def run_agent() -> None:
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    corvo_url = os.environ.get("CORVO_URL", "http://host.docker.internal:4111")
    corvo_secret = os.environ.get("CORVO_SECRET", "")
    mcp_url = os.environ.get("MCP_URL", "http://o11y-stack:8080/mcp")

    statement = Path("/app/instruction.txt").read_text().strip()
    env_ts = scenario_clock_iso()
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        current_time=env_ts,
        statement=statement,
    )

    agent_dir = Path("/logs/agent")
    agent_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    trajectory_id = str(uuid.uuid4())
    atif_steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    stats = {"input": 0, "output": 0, "cost": 0.0}
    step_id = 0
    tool_call_count = 0
    start = time.time()

    def flush_trajectory() -> None:
        trajectory = {
            "schema_version": "ATIF-v1.7",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "agent": {
                "name": "corvo",
                "version": "0.1.0",
                "model_name": "corvo",
                "tool_definitions": tool_defs,
            },
            "steps": atif_steps,
            "final_metrics": {
                "total_prompt_tokens": stats["input"],
                "total_completion_tokens": stats["output"],
                "total_cached_tokens": 0,
                "total_cost_usd": stats["cost"],
                "total_steps": step_id,
                "extra": {
                    "total_tool_calls": tool_call_count,
                    "reasoning_effort": "off",
                    "elapsed_seconds": time.time() - start,
                },
            },
        }
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    try:
        print(f"Connecting to MCP at {mcp_url}...")
        async with streamable_http_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await discover_tools(session)
                tool_defs[:] = [t["function"] for t in tools]
                print(f"Discovered {len(tools)} tools, calling Corvo at {corvo_url}")

                step_id += 1
                atif_steps.append(make_atif_step(step_id, "system", SYSTEM_PROMPT))
                step_id += 1
                atif_steps.append(make_atif_step(step_id, "user", task_prompt))

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt},
                ]

                async with httpx.AsyncClient() as http:
                    step = 0
                    while True:
                        step += 1
                        if step > MAX_AGENT_STEPS:
                            print(f"\nStep limit ({MAX_AGENT_STEPS}) reached")
                            break
                        print(f"[{step}]", end=" ", flush=True)

                        resp_data = await call_corvo_complete(
                            http, corvo_url, corvo_secret, messages, tools
                        )

                        content: str = resp_data.get("content") or ""
                        tool_calls_raw: list[dict[str, Any]] = resp_data.get("tool_calls") or []
                        usage = resp_data.get("usage") or {}
                        step_in = usage.get("prompt_tokens", 0) or 0
                        step_out = usage.get("completion_tokens", 0) or 0
                        stats["input"] += step_in
                        stats["output"] += step_out
                        stats["cost"] += estimate_cost_usd(step_in, step_out)

                        atif_tool_calls = [
                            {
                                "tool_call_id": tc.get("id", f"tc_{i}"),
                                "function_name": tc.get("function", {}).get("name", ""),
                                "arguments": parse_tool_arguments(
                                    tc.get("function", {}).get("arguments", {})
                                ),
                            }
                            for i, tc in enumerate(tool_calls_raw)
                        ]

                        if not tool_calls_raw:
                            step_id += 1
                            atif_steps.append(make_atif_step(step_id, "agent", content))
                            flush_trajectory()
                            print("done")
                            break

                        # Add assistant message with tool_calls for next round
                        messages.append({
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls_raw,
                        })
                        tool_call_count += len(tool_calls_raw)

                        observation_results: list[dict[str, Any]] = []
                        for tc in tool_calls_raw:
                            fn = tc.get("function", {}).get("name", "")
                            fa = parse_tool_arguments(tc.get("function", {}).get("arguments", {}))
                            tc_id = tc.get("id", f"tc_{tool_call_count}")
                            try:
                                out = await call_mcp_tool(session, fn, fa)
                                print(f"{fn}({len(out)})", end=" ", flush=True)
                            except Exception as e:
                                out = f"Error: {e}"
                                print(f"{fn}(ERR)", end=" ", flush=True)

                            observation_results.append({"source_call_id": tc_id, "content": out})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": out,
                            })

                        step_id += 1
                        atif_steps.append(make_atif_step(
                            step_id,
                            "agent",
                            content,
                            tool_calls=atif_tool_calls,
                            observation={"results": observation_results},
                        ))
                        flush_trajectory()
                        print()
    finally:
        elapsed = time.time() - start
        print(f"\n{elapsed:.1f}s | {stats['input']}in {stats['output']}out")
        flush_trajectory()


if __name__ == "__main__":
    asyncio.run(run_agent())
