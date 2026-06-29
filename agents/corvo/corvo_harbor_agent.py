"""Corvo Harbor agent for o11y-bench.

Uploads corvo_runner.py into the container and runs it. The runner connects
to mcp-grafana directly (same network), then calls Corvo's LLM completion
endpoint on the host for each step of the agent loop.

Usage:
  CORVO_URL=http://host.docker.internal:4111 \\
  CORVO_SECRET=<secret> \\
  ANTHROPIC_API_KEY=... \\
  ANTHROPIC_BASE_URL=... \\
  mise run bench:job -- \\
    --model corvo \\
    --task-name query-cpu-metrics \\
    --agent-import-path agents.corvo.corvo_harbor_agent:CorvoHarborAgent
"""

import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNNER_SCRIPT = Path(__file__).parent / "corvo_runner.py"
SYSTEM_PROMPT = Path(__file__).parent.parent / "system_prompt.txt"
TASK_PROMPT = Path(__file__).parent.parent / "task_prompt.txt"

VIEWER_COMMAND_STDOUT_PATH = "/logs/agent/command-0/stdout.txt"


def _build_runner_command() -> str:
    command = (
        "set -o pipefail; "
        "mkdir -p /logs/agent/command-0; "
        f'uv run /app/corvo_runner.py 2>&1 | tee "{VIEWER_COMMAND_STDOUT_PATH}"'
    )
    return f"bash -lc {shlex.quote(command)}"


class CorvoHarborAgent(BaseAgent):
    """Harbor agent that delegates LLM completions to Corvo agent-server."""

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
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(command="mkdir -p /app")
        await environment.upload_file(
            source_path=RUNNER_SCRIPT,
            target_path="/app/corvo_runner.py",
        )
        await environment.upload_file(
            source_path=SYSTEM_PROMPT,
            target_path="/app/system_prompt.txt",
        )
        await environment.upload_file(
            source_path=TASK_PROMPT,
            target_path="/app/task_prompt.txt",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        instruction_path = self.logs_dir / "instruction.txt"
        instruction_path.write_text(instruction)
        await environment.upload_file(
            source_path=instruction_path,
            target_path="/app/instruction.txt",
        )

        corvo_url = os.environ.get("CORVO_URL", "http://host.docker.internal:4111")
        corvo_secret = os.environ.get("CORVO_SECRET", "")

        env: dict[str, str] = {
            "CORVO_URL": corvo_url,
            "CORVO_SECRET": corvo_secret,
            "MCP_URL": "http://o11y-stack:8080/mcp",
            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
            "PYTHONPATH": "/app",
            "O11Y_SCENARIO_TIME_ISO": os.environ.get("O11Y_SCENARIO_TIME_ISO", ""),
        }

        result = await environment.exec(command=_build_runner_command(), env=env)

        try:
            await environment.download_file(
                source_path="/logs/agent/trajectory.json",
                target_path=self.logs_dir / "trajectory.json",
            )
            trajectory = json.loads((self.logs_dir / "trajectory.json").read_text())
            fm = trajectory.get("final_metrics") or {}
            context.n_input_tokens = fm.get("total_prompt_tokens")
            context.n_output_tokens = fm.get("total_completion_tokens")
            context.cost_usd = fm.get("total_cost_usd")
        except Exception:
            pass

        if result.return_code != 0:
            raise NonZeroAgentExitCodeError(
                f"corvo_runner.py exited with code {result.return_code}"
            )
