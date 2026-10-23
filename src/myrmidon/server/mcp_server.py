"""MCP server: exposes the tool registry to any MCP client (Claude Code, ...).

One registration path: each ToolSpec's pydantic params model becomes the MCP
input schema; the handler funnels through ToolRegistry.call, so validation,
gates and provenance apply identically to MCP callers and
the native agent loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from myrmidon.config import MyrmidonConfig
from myrmidon.core.observation import model_view
from myrmidon.core.tool import REGISTRY, ToolContext, ToolSpec
from myrmidon.engines.discovery import discover
from myrmidon.execution.local import LocalBackend
from myrmidon.knowledge.inputsets import InputSetStore
from myrmidon.knowledge.protocols import ProtocolStore


def build_backend(config: MyrmidonConfig):
    if config.backend == "slurm":
        from myrmidon.execution.slurm import SlurmBackend

        compute = config.compute
        return SlurmBackend(
            config.slurm,
            cores_per_node=compute.cores_per_node if compute.node_exclusive else None,
            env_script=str(config.env_script) if config.env_script else None,
        )
    return LocalBackend(mpirun=config.mpirun_bin)


def build_context(config: MyrmidonConfig | None = None) -> ToolContext:
    config = config or MyrmidonConfig.load()
    import myrmidon.basic_tools as basic
    from myrmidon.core.envscript import apply_env_script, ensure_own_bin_on_path

    basic.register_all()
    # the engines may come from module loads: source the machine's env script before
    # discovery so the client need not have; console scripts of this
    # environment (mimicpy) resolve even when launched by absolute path
    env_status = apply_env_script(config.env_script)
    ensure_own_bin_on_path()
    return ToolContext(
        config=config,
        workspace=None,
        backend=build_backend(config),
        engines=discover(config),
        inputsets=InputSetStore.with_layers(config.knowledge_layers),
        protocols=ProtocolStore.with_layers(config.knowledge_layers),
        env_script=env_status.as_dict(),
    )


def _make_handler(spec: ToolSpec, ctx: ToolContext):
    model = spec.params_model

    def handler(params) -> dict:  # annotation injected below
        raw = params.model_dump() if hasattr(params, "model_dump") else dict(params)
        obs = REGISTRY.call(spec.name, ctx, raw)
        return model_view(obs)

    handler.__name__ = spec.name
    handler.__doc__ = spec.description
    handler.__annotations__ = {"params": model, "return": dict}
    return handler


def exposed_tools(ctx: ToolContext) -> list[ToolSpec]:
    """What this server shows the model: every registered tool whose engines are present
    on this machine. There is no narrower list: a tool the model cannot see is
    indistinguishable from one that does not exist, and it reimplements the engine."""
    return REGISTRY.available(ctx.engines)


def approx_tokens(spec: ToolSpec) -> int:
    """What one tool costs the model per turn, roughly: its schema and description."""
    blob = json.dumps(spec.params_model.model_json_schema()) + spec.description
    return len(blob) // 4


def build_server(config: MyrmidonConfig | None = None, ctx: ToolContext | None = None,
                 workspace: Path | None = None) -> MCPServer:
    ctx = ctx or build_context(config)
    if workspace is not None:
        from myrmidon.core.workspace import Workspace

        ctx.workspace = Workspace.open(Path(workspace))
    mcp = MCPServer("myrmidon")
    exposed = exposed_tools(ctx)
    for spec in exposed:
        mcp.add_tool(
            _make_handler(spec, ctx), name=spec.name, description=spec.description
        )
    return mcp
