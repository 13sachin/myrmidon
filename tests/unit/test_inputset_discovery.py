"""Finding an input set and seeing what it supplies, refuses, and why.

Before this, an empty query matched nothing, so "what inputsets do I have" had no answer,
and a value's reason was absent from the wire shape, so the only way to learn why a value
was what it is was to override it and read the response.

There is no `tunable` here any more. An input set states values and their reasons; which
value a study is *varying* is the study's own declaration, and it lives on the plan.
"""

from myrmidon.core.tool import REGISTRY


def _lookup(ctx, query=None):
    return REGISTRY.call("inputset_lookup", ctx, {} if query is None else {"query": query})


def test_required_values_are_never_also_supplied(ctx):
    """Refusing to supply a value and supplying it are exclusive."""
    for inputset in _lookup(ctx).inputsets:
        assert not (set(inputset.required_from_user) & set(inputset.parameters))

