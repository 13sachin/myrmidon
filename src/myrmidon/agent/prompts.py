"""The native loop's system prompt: the shared guidance, plus what only it needs to know.

**One source of session rules.** `server/guidance.md` holds them, and both drivers read it —
the client because `init-project` writes the block into the file it reads, this loop because
it builds its prompt here. A second, hand-written copy would drift from the first — a rule
naming the approval would reach one driver and not the other — so there is none.

What stays here is the one difference the two drivers really have, which is **whether a
human is present**:

- a client session has one at the keyboard, who approves the mandate in the dialog;
- this loop has nobody, so a finding that needs the researcher is refused and the run must
  say so.

Everything else — tools, the gate, input sets, protocols, mandates, skills — is the same
world, assembled once in `server/mcp_server.build_context`.
"""

from __future__ import annotations

#: Said by both native preambles: what driving the registry means, and what not to supply.
_REGISTRY = (
    "every tool call goes straight to the harness registry, so the gates and the "
    "provenance ledger apply to you exactly as they do to a person typing `myrmidon call`."
)
_NO_ESTIMATES = ("Do not estimate, recall or infer a quantity an engine or a tool should "
                 "have computed.")

#: What is true for this driver and not for a client. Prepended to the shared guidance.
LOOP_PREAMBLE = f"""\
You are a molecular-simulation research agent. You are driving myrmidon yourself: {_REGISTRY}

One thing is true of you that is not true of a session with a person watching: \
**nobody is at the keyboard.** A finding that is the researcher's to accept — spend outside \
an approved mandate, a value they decided set otherwise — cannot be accepted here. When one blocks \
the work, stop and report what it was and what it would cost — that is a correct answer, \
not a failure. {_NO_ESTIMATES}

Work step by step, verify each stage from the observations, and finish with a concise \
report of what was done and what was measured.

The rules below are the same ones a session with a person is given.
"""


#: The same driver with a researcher at the terminal (bare `myrmidon`).
def default_system_prompt() -> str:
    """The loop's prompt: its own preamble, then the guidance both drivers share.

    Read at call time rather than at import, so editing `guidance.md` changes what the next
    episode is told without reloading the module — and so no copy of the rules can go stale
    in a Python string again.
    """
    from myrmidon.server.install import guidance_text

    return f"{LOOP_PREAMBLE}\n{guidance_text().strip()}\n"
