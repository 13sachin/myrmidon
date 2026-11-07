"""The published pipelines a study may start from: the stages a mandate is written from,
the deliverables they are measured for, the Hamiltonian they are computed with, and what each
stage costs on this machine."""

from __future__ import annotations

from pydantic import BaseModel, Field

from myrmidon.core.observation import Observation, ProtocolHit
from myrmidon.core.tool import ToolContext, tool


class ProtocolLookupParams(BaseModel):
    query: str = Field(
        default="",
        description="words to match against ids, titles, stages and applicability. Empty "
        "lists every published pipeline this machine can see.",
    )


def _protocol_hit(ctx: ToolContext, w, *, full: bool = False) -> ProtocolHit:
    from myrmidon.execution.pace import read_paces

    prices = w.priced(ctx.config.compute, read_paces(ctx.config.studies_root))
    return ProtocolHit(
        id=w.id,
        title=w.title,
        applicability=w.applicability,
        system_class=w.system_class,
        scope=w.scope,
        layer=w.layer,
        version=w.version,
        given=w.given,
        stages=[{**s.model_dump(mode="json", exclude_none=True),
                 **({"core_h": prices[s.name].core_h} if prices[s.name].core_h else {})}
                for s in w.stages],
        deliverables=[d.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
                      for d in w.deliverables] if full else [],
        hamiltonian=(w.hamiltonian.model_dump(mode="json", exclude_none=True, exclude={"mm"})
                     if w.hamiltonian is not None else None),
        total_core_h=round(sum(p.core_h or 0.0 for p in prices.values()), 3),
        unpriced=[f"{name} ({p.basis})" for name, p in prices.items() if p.core_h is None],
        citations=[s.text for s in w.sources] if full else [],
        pitfalls=w.pitfalls,
        notes=w.notes if full else "",
    )


@tool(
    "protocol_lookup",
    navigation=True,
    description="Published pipelines to start a study from: the stages a mandate is written "
    "from, their deliverables and Hamiltonian, what you bring, what each stage costs here. A "
    "STARTING POINT, never a rule: price one on the built system with `protocol_preview`, or "
    "write stages of your own and cite none.",
    tags={"knowledge"},
    needs_workspace=False,
)
def protocol_lookup(ctx: ToolContext, params: ProtocolLookupParams) -> Observation:
    from myrmidon.knowledge.protocols import ProtocolStore

    store = ctx.protocols or ProtocolStore.default()
    q = params.query.strip()
    if q:
        try:
            exact, _ = store.get_versioned(q)
            return Observation(
                ok=True,
                summary=f"1 published pipeline matched: {exact.id}@v{exact.version}. "
                f"To price its stages for the mandate: protocol_preview "
                f"protocol={exact.id}.",
                protocols=[_protocol_hit(ctx, exact, full=True)],
            )
        except KeyError:
            pass
    found = store.search(q) if q else sorted(
        store.all(), key=lambda w: w.id
    )
    full = len(found) == 1
    hits = [_protocol_hit(ctx, w, full=full) for w in found]
    what = f"matching {q!r}" if q else "the whole store"
    first = hits[0].id if hits else "<id>"
    return Observation(
        ok=True,
        # naming the next call, in the imperative, at the point of need: the tool that
        # finds the input says which tool makes the thing, or a session reads every input
        # set and then writes prose
        summary=f"{len(hits)} published pipeline(s), {what}. To price one for the "
        f"mandate: protocol_preview protocol={first}. Each is a starting point — adapt "
        "it, or carry out stages of your own and cite none.",
        protocols=hits,
    )
