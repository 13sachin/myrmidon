"""MCP server: exposes the tool registry to any MCP client (Claude Code, ...).

One registration path: each field of a ToolSpec's pydantic params model is one MCP
argument, so the model fills a flat argument list rather than one nested object (an
open model serving tool calls through an OpenAI-compatible endpoint tends to hand a
nested object over as a JSON string, which the client then refuses); the handler funnels
through ToolRegistry.call, so validation, gates and provenance apply identically to MCP
callers and the native agent loop.
"""

from __future__ import annotations

import inspect
import json
import typing
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from myrmidon.config import MyrmidonConfig
from myrmidon.core.observation import model_view
from myrmidon.core.tool import REGISTRY, ToolContext, ToolSpec
from myrmidon.engines.discovery import discover
from myrmidon.execution.local import LocalBackend
from myrmidon.knowledge.inputsets import InputSetStore
from myrmidon.knowledge.protocols import ProtocolStore
from myrmidon.knowledge.tool_profiles import shown


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
    import myrmidon.tools as basic
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


def _lenient_type(annotation, metadata=()):
    """A field's type, its constraints on it, widened by the string spellings ``lenient``
    reads: the constraints stay on the type they were written for."""
    from myrmidon.core.tool import takes_list, takes_none

    own = Annotated[(annotation, *metadata)] if metadata else annotation
    if takes_list(annotation) or (takes_none(annotation) and annotation is not str
                                  and str not in typing.get_args(annotation)):
        return own | str
    return own


def _make_handler(spec: ToolSpec, ctx: ToolContext):
    """A handler whose signature is the params model's fields, one keyword argument
    each, carrying the field's type, default and description; the SDK reads the
    signature to build the tool's input schema."""
    model = spec.params_model
    defaults = {name: field.get_default(call_default_factory=True)
                for name, field in model.model_fields.items() if not field.is_required()}

    def handler(**kwargs) -> dict:
        # the SDK passes every field, its default included: what the call left at its
        # default is passed as left out, so a tool can tell what the call stated
        stated = {k: v for k, v in kwargs.items() if k not in defaults or v != defaults[k]}
        obs = REGISTRY.call(spec.name, ctx, stated)
        return model_view(obs)

    parameters = []
    for name, field in model.model_fields.items():
        default = (inspect.Parameter.empty if field.is_required()
                   else field.get_default(call_default_factory=True))
        # the field's constraints travel in its metadata; its default is the parameter's.
        # The schema also admits the string spellings the registry reads as meant
        # (core/tool.py `lenient`), so a client's own check does not refuse them first
        annotation = Annotated[(_lenient_type(field.annotation, field.metadata),
                                Field(description=field.description))]
        parameters.append(inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation))
    handler.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    handler.__name__ = spec.name
    handler.__doc__ = spec.description
    return handler


def exposed_tools(ctx: ToolContext) -> list[ToolSpec]:
    """What this server shows the model: the tools whose engines are present on this
    machine, narrowed to the core set and the config's `tool_profile` unless it is "all".
    A tool outside the profile is not hidden from the model: `tool_lookup` names it, and
    `tool_run` calls it."""
    names = shown(ctx.config.tool_profile)
    available = REGISTRY.available(ctx.engines)
    return available if names is None else [s for s in available if s.name in names]


TOOL_RUN = ("Call a tool tool_lookup names that this list does not show, by name, with its "
            "parameters as a JSON object; without them it returns the tool's parameter "
            "schema. The call is the tool's own, gated and recorded under its name.")


def _tool_run(ctx: ToolContext):
    def tool_run(name: str, params: dict | None = None) -> dict:
        try:
            spec = REGISTRY.get(name)
        except Exception:
            return {"ok": False, "summary": f"no tool {name!r}: tool_lookup names them"}
        if spec not in REGISTRY.available(ctx.engines):
            return {"ok": False, "summary": f"{name} needs an engine this machine does not "
                    "have: " + ", ".join(sorted(spec.requires))}
        schema = spec.params_model.model_json_schema()
        unknown = sorted(set(params or {}) - set(spec.params_model.model_fields))
        if params is None or unknown:
            said = f"{name} takes no {', '.join(unknown)}; " if unknown else ""
            return {"ok": False, "summary": said + f"{name}'s parameters are in `schema`",
                    "schema": schema}
        result = model_view(REGISTRY.call(name, ctx, params))
        if not result.get("ok") and (result.get("error") or {}).get("code") == "INVALID_PARAMS":
            result["schema"] = schema
        return result

    tool_run.__doc__ = TOOL_RUN
    return tool_run


def approx_tokens(spec: ToolSpec) -> int:
    """What one tool costs the model per turn, roughly: its schema and description."""
    blob = json.dumps(spec.params_model.model_json_schema()) + spec.description
    return len(blob) // 4


def build_server(config: MyrmidonConfig | None = None, ctx: ToolContext | None = None,
                 workspace: Path | None = None) -> MCPServer:
    ctx = ctx or build_context(config)
    ctx.model_driven = True  # the server's calls are the client's model's
    if workspace is not None:
        from myrmidon.core.workspace import Workspace

        ctx.workspace = Workspace.open(Path(workspace))
    mcp = MCPServer("myrmidon")
    exposed = exposed_tools(ctx)
    for spec in exposed:
        mcp.add_tool(
            _make_handler(spec, ctx), name=spec.name, description=spec.description
        )
    if len(exposed) < len(REGISTRY.available(ctx.engines)):
        mcp.add_tool(_tool_run(ctx), name="tool_run", description=TOOL_RUN)
    return mcp
