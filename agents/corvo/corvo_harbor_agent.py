"""Corvo Harbor agent for o11y-bench (orchestrator mode).

Starts an MCP bridge inside the Harbor container, then delegates the full
agent loop to the Corvo agent-server via POST /internal/o11y-bench/run.
The agent-server runs the task through the Mastra orchestrator with the
MCP tools bridged back into the container.

Usage:
  CORVO_URL=http://<host-ip>:8080 \\
  CORVO_SECRET=<secret> \\
  mise run bench:job -- \\
    --model corvo/corvo \\
    --task-name query-cpu-metrics \\
    --agent-import-path agents.corvo.corvo_harbor_agent:CorvoHarborAgent
"""

import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

import httpx
from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

BRIDGE_SCRIPT = Path(__file__).parent / "mcp_bridge.py"
BRIDGE_PORT = 8099
BRIDGE_STARTUP_TIMEOUT_S = 30

# Harbor viewer reads this path to show agent stdout in the report.
VIEWER_STDOUT_PATH = "/logs/agent/command-0/stdout.txt"


def _build_atif_trajectory(
    instruction: str,
    final_answer: str,
    task_id: str,
    run_id: str | None,
) -> dict[str, Any]:
    """Build a minimal ATIF-v1.7 trajectory so the verifier can parse the final answer."""
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": task_id,
        "trajectory_id": run_id or str(uuid.uuid4()),
        "agent": {
            "name": "corvo",
            "version": "0.2.0",
            "mode": "orchestrator",
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": instruction},
            {"step_id": 2, "source": "agent", "message": final_answer},
        ],
    }


class CorvoHarborAgent(BaseAgent):
    """Harbor agent that runs o11y tasks via the Corvo Mastra orchestrator."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "corvo"

    def version(self) -> str:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(command="mkdir -p /app /logs/agent/command-0")
        await environment.upload_file(
            source_path=BRIDGE_SCRIPT,
            target_path="/app/mcp_bridge.py",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        bridge_token = secrets.token_hex(16)
        mcp_url = os.environ.get("MCP_URL", "http://o11y-stack:8080/mcp")

        # Start the MCP bridge in the background inside the container.
        await environment.exec(
            command=(
                f"bash -c 'BRIDGE_TOKEN={bridge_token} "
                f"MCP_URL={mcp_url} "
                f"BRIDGE_PORT={BRIDGE_PORT} "
                f"uv run /app/mcp_bridge.py > /logs/agent/bridge.log 2>&1 &'"
            ),
        )

        # Wait for the bridge to become healthy.
        wait_result = await environment.exec(
            command=(
                f"bash -c 'for i in $(seq 1 {BRIDGE_STARTUP_TIMEOUT_S}); do "
                f"curl -sf http://localhost:{BRIDGE_PORT}/health && exit 0; "
                f"sleep 1; done; echo \"Bridge did not start\" >&2; exit 1'"
            ),
        )
        if wait_result.return_code != 0:
            raise NonZeroAgentExitCodeError("MCP bridge failed to start in time")

        # Get the container's Docker network IP so agent-server can reach the bridge from host.
        ip_result = await environment.exec(command="hostname -I | awk '{print $1}'")
        container_ip = ip_result.stdout.strip().split()[0]
        bridge_url = f"http://{container_ip}:{BRIDGE_PORT}"

        corvo_url = os.environ.get("CORVO_URL", "http://host.docker.internal:4111")
        corvo_secret = os.environ.get("CORVO_SECRET", "")
        scenario_time = os.environ.get("O11Y_SCENARIO_TIME_ISO", "")
        task_id = str(uuid.uuid4())

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{corvo_url}/internal/o11y-bench/run",
                json={
                    "taskId": task_id,
                    "instruction": instruction,
                    "scenarioTimeIso": scenario_time,
                    "mcpBridgeUrl": bridge_url,
                    "mcpBridgeToken": bridge_token,
                },
                headers={"X-O11Y-Bench-Secret": corvo_secret},
                timeout=600.0,
            )
            resp.raise_for_status()
            data = resp.json()

        final_answer = data.get("finalAnswer", "")
        usage = data.get("usage") or {}
        run_id = data.get("runId")

        # Populate Harbor context with token/cost metadata.
        if isinstance(usage, dict):
            context.n_input_tokens = (
                usage.get("inputTokens")
                or usage.get("promptTokens")
                or usage.get("prompt_tokens")
            )
            context.n_output_tokens = (
                usage.get("outputTokens")
                or usage.get("completionTokens")
                or usage.get("completion_tokens")
            )

        # Write ATIF trajectory to host logs_dir and upload to container.
        # Harbor will download /logs/agent/ → self.logs_dir after run completes,
        # so the verifier reads from there. We write here and upload to keep both
        # in sync regardless of which path the verifier uses.
        trajectory = _build_atif_trajectory(instruction, final_answer, task_id, run_id)
        trajectory_json = json.dumps(trajectory, indent=2)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(trajectory_json)

        await environment.upload_file(
            source_path=self.logs_dir / "trajectory.json",
            target_path="/logs/agent/trajectory.json",
        )

        # Write final answer as the viewer stdout so it appears in the Harbor report.
        (self.logs_dir / "stdout.txt").write_text(final_answer)
        await environment.upload_file(
            source_path=self.logs_dir / "stdout.txt",
            target_path=VIEWER_STDOUT_PATH,
        )
        (self.logs_dir / "stdout.txt").unlink(missing_ok=True)
