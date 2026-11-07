"""File tools: read, write and check files in a study or its project."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myrmidon.core.artifacts import ArtifactRef
from myrmidon.core.contracts import Contract, Produces
from myrmidon.core.mandate import MANDATE_FILE
from myrmidon.core.observation import (
    Observation,
)
from myrmidon.core.tool import ToolContext, tool
from myrmidon.core.validation import decide
from myrmidon.engines.inputs import materialize_symlink, symlink_component
from myrmidon.engines.registry import InputCheck, adapters


class FileWriteParams(BaseModel):
    path: str = Field(description="workspace-relative path, e.g. '02_run/em.mdp'")
    content: str = Field(description="full file content")
    materialize: bool = Field(
        default=False,
        description="if the path goes through a symlink, replace that link with a private "
        "real copy before writing (never write through a link into shared files)",
    )


def _written_kind(ctx, params: dict) -> str:
    return "mdp" if str(params.get("path", "")).endswith(".mdp") else "file"


@tool(
    "file_write",
    description="Write a text file inside the active workspace (e.g. an .mdp you "
    "authored). The file is hashed into provenance. Paths outside the workspace are "
    "refused.",
    tags={"workspace", "files"},
    contract=Contract(produces=(Produces(kind_of=_written_kind),)),
)
def file_write(ctx: ToolContext, params: FileWriteParams) -> Observation:
    root = ctx.workspace.root
    target = (root / params.path).resolve()
    if not str(target).startswith(str(root) + "/"):
        return Observation(
            ok=False, summary=f"refused: '{params.path}' escapes the workspace"
        )
    if target == (root / MANDATE_FILE).resolve():
        return Observation(
            ok=False, summary=f"refused: {MANDATE_FILE} is written only by mandate_write, "
            "whose digest the researcher's approval is bound to")
    link = symlink_component(root, params.path)
    if link is not None:
        real = link.resolve()
        if not params.materialize:
            return Observation(
                ok=False,
                summary=f"refused: '{params.path}' goes through the symlink "
                f"{ctx.workspace.relpath(link)} -> {real}; writing would change the shared "
                "target for everything that links it (a force-field itp zeroed this way "
                "invalidates every fit that links it). Write to a real copy instead — "
                "materialize=true replaces the link with a private copy first.",
            )
        materialized = materialize_symlink(link)
        target = (root / params.path).resolve()
    else:
        materialized = None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(params.content)
    role = "mdp" if target.suffix == ".mdp" else "file"
    note = (
        f" (replaced symlink {ctx.workspace.relpath(link)} with a private copy of "
        f"{materialized})" if materialized else ""
    )
    return Observation(
        ok=True,
        summary=f"Wrote {ctx.workspace.relpath(target)} "
        f"({len(params.content)} chars){note}",
        artifacts=[ArtifactRef.from_path(target, role, root=root)],
    )


class FileReadParams(BaseModel):
    path: str = Field(
        description="the path as the researcher gave it: relative to the study or to "
        "the project directory it was made from, or absolute inside either; a directory "
        "lists what is in it, and an empty path lists the study")
    max_chars: int = Field(default=8000, ge=1, le=100_000)


@tool(
    "file_read",
    navigation=True,
    description="Read a text file (inputs, mdps, small logs), truncated at max_chars, or list a "
                "directory. A relative path is looked up in the study, then in the project "
                "directory it was made from. Read-only.",
    tags={"workspace", "files"},
)
def file_read(ctx: ToolContext, params: FileReadParams) -> Observation:
    root = ctx.workspace.root
    # The researcher's files are in the project directory, never in a new study, and the
    # study's own root ('', '.') is readable too. Nothing beyond the two: what is read
    # goes to the model's server, and the machine's secrets are files too.
    target = ctx.workspace.resolve(params.path).resolve()
    allowed = [root.resolve()] + ([Path(ctx.workspace.origin).resolve()]
                                  if ctx.workspace.origin else [])
    inside = next((a for a in allowed if target.is_relative_to(a)), None)
    # a study made from $HOME has the home directory as its project: hidden paths there
    # (~/.config/myrmidon/secrets.env, ~/.ssh) are not the researcher's inputs
    if inside is not None and inside != allowed[0] and any(
            part.startswith(".") for part in target.relative_to(inside).parts):
        inside = None
    if inside is None:
        return Observation(
            ok=False,
            summary=f"refused: '{params.path}' is outside {_searched(ctx)}, or hidden; "
                    "file_read looks only there. Ask the researcher to copy the file into "
                    "the project directory.",
        )
    where = _where(ctx, target)
    if target.is_dir():
        # A directory is the question "what is in here"; an agent with no listing guesses
        # file names, or writes a script instead of calling the tool it needs. Listing is
        # a look, like reading.
        entries = sorted(target.iterdir(), key=lambda q: (q.is_dir(), q.name))
        lines = [f"{q.name}/" if q.is_dir() else f"{q.name}  ({q.stat().st_size} bytes)"
                 for q in entries]
        rel = (params.path.rstrip("/") or ".") + where
        parsers = _parsers_for(entries, root.resolve(), ctx.workspace.ledger)
        return Observation(
            ok=True, kind="observe",
            summary=(f"{rel}/ holds {len(lines)} entr{'y' if len(lines) == 1 else 'ies'}:\n"
                     + "\n".join(lines) if lines else f"{rel}/ is empty")
            + ("".join(f"\n{line}" for line in parsers) if parsers else ""),
        )
    if not target.is_file():
        return Observation(
            ok=False,
            summary=f"file not found: {params.path} (looked in {_searched(ctx)}) — "
                    "file_read on the directory that should hold it lists what is there",
        )
    text = target.read_text(errors="replace")
    truncated = len(text) > params.max_chars
    return Observation(
        ok=True,
        summary=(f"{target}{where}:\n" if where else "")
        + ("[truncated]\n" if truncated else "") + text[: params.max_chars],
        # a look at a file, like file_check: it consumes nothing and produces nothing,
        # so it is an observation and never a step of the study
        kind="observe",
    )


def _where(ctx: ToolContext, target: Path) -> str:
    """Said beside a path that is not in the study, so the model passes on a path that
    resolves (the absolute one) rather than the one that found it."""
    if target.is_relative_to(ctx.workspace.root.resolve()):
        return ""
    return f" (in the project directory {ctx.workspace.origin}, not in the study)"


def _searched(ctx: ToolContext) -> str:
    origin = ctx.workspace.origin
    return (f"the study {ctx.workspace.root} and the project directory {origin}"
            if origin else f"the study {ctx.workspace.root}")


def _parsers_for(entries: list[Path], root: Path, ledger=None,
                 limit: int = 5) -> list[str]:
    """For each listed file some registered tool parses, the call that parses it, as a
    line — unless
    the ledger shows that tool already read these very bytes. A stale offer is a loop: a
    model that has diagnosed a log and lists its directory again is offered the same call
    again."""
    from myrmidon.core.artifacts import sha256_file
    from myrmidon.core.tool import REGISTRY

    read = set()  # (tool, sha256) of every file a tool has consumed in this study
    if ledger is not None:
        read = {(ev.tool, ref.sha256) for ev in ledger.events() for ref in ev.inputs}
    actions = []
    for path in entries:
        if not path.is_file():
            continue
        for tool_name, reader in REGISTRY.readers(path):
            if read and (tool_name, sha256_file(path)) in read:
                continue
            # a file outside the study is offered by its absolute path, which resolves
            # the same for every tool
            rel = (str(path.relative_to(root)) if path.is_relative_to(root)
                   else str(path))
            actions.append(f"{path.name} is {reader.what}: {tool_name} "
                           f"{reader.param}={rel} parses it")
            if len(actions) >= limit:
                return actions
    return actions



class FileCheckParams(BaseModel):
    path: str = Field(description="the engine input file (.mdp, a CPMD input, or any file "
                      "an input set rendered): relative to the study or its project "
                      "directory, or absolute")
    tpr: str = Field(default="", description="for a CPMD/MiMiC input: the .tpr it will run "
                     "with, so the QM/MM coupling can be checked against it")


@tool(
    "file_check",
    description="Check one engine input file without running it: what the sensors notice, where "
                "every value came from, and what changed by hand since it was rendered. Use before"
                " running an edited file, or to answer 'where did this number come from'. A look, "
                "not a step.",
    tags={"files", "validation", "knowledge"},
)
def file_check(ctx: ToolContext, params: FileCheckParams) -> Observation:
    from myrmidon.engines.file_origin import file_origin
    from myrmidon.engines.file_origin import render as render_origin
    from myrmidon.engines.inputs import resolve_in_ws

    if not ctx.workspace.resolve(params.path).is_file():
        return Observation(
            ok=False, summary=f"no file at {params.path} in {_searched(ctx)}",
            path=params.path,
        )
    real = resolve_in_ws(ctx, params.path)
    # the engine whose input it is says what it makes of it
    check = next((found for engine in adapters() if engine.input_findings
                  and (found := engine.input_findings(ctx, params.path, real, params.tpr))),
                 InputCheck("other", "file", [], []))
    kind, warnings, checked = check.kind, list(check.findings), list(check.checked)
    origin = file_origin(ctx, params.path)
    known = {w.id for w in warnings}
    warnings += [w for w in origin.findings if w.id not in known]
    checked = [*checked, "parameter_source"]
    decision = decide(warnings)
    label = check.label
    obs = Observation(
        ok=not decision.blocking,
        summary=f"{label} check of {params.path}: {len(decision.blocking)} blocking, "
        f"{len(decision.advisory)} advisory across: {', '.join(checked)}\n"
        + render_origin(origin),
        path=params.path,
        file_kind=kind,
        findings=[*decision.blocking, *decision.unapproved, *decision.advisory],
        provenance=origin.provenance,
        sourced=origin.sourced,
        inputset=origin.inputset,
        keys=[k.model_dump() for k in origin.keys],
        yours=[k.key for k in origin.yours],
        edited=origin.edited,
    )
    obs.kind = "observe"  # a look at a file: counted apart from the work, never a node
    return obs
