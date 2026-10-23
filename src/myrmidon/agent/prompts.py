"""The native loop's system prompt: the shared guidance, plus what only it needs to know.

**One source of session rules.** `server/guidance.md` holds them, and both drivers read it —
the client because `init-project` writes the block into the file it reads, this loop because
it builds its prompt here. A second, hand-written copy would drift from the first — a rule
naming the plan verbs would reach one driver and not the other — so there is none.

What stays here is the one difference the two drivers really have, which is **whether a
human is present**:

- a client session has one at the keyboard, so a finding that `needs_human` can be
  acknowledged after asking;
- this loop has nobody, so such a finding is refused and the run must say so;
- and because this loop owns the final answer, `agent/grounding.py` can bounce it — a number
  no observation returned does not survive here, where a client's own loop is not checked.

Everything else — tools, gates, input sets, protocols, plans, skills — is the same
world, assembled once in `server/mcp_server.build_context`.
"""

from __future__ import annotations

#: Said by both native preambles: what driving the registry means, and what not to supply.
_REGISTRY = (
    "every tool call goes straight to the harness registry, so the gates, the "
    "acknowledge-by-id protocol and the provenance ledger apply to you exactly as they do to "
    "a person typing `myrmidon call`."
)
_NO_ESTIMATES = ("Do not estimate, recall or infer a quantity an engine or a tool should "
                 "have computed.")

#: What is true for this driver and not for a client. Prepended to the shared guidance.
LOOP_PREAMBLE = f"""\
You are a molecular-simulation research agent. You are driving myrmidon yourself: {_REGISTRY}

Two things are true of you that are not true of a session with a person watching:

- **Nobody is at the keyboard.** A finding marked as needing a human cannot be acknowledged \
by you. When one blocks the work, stop and report what it was and what it would cost — that \
is a correct answer, not a failure.
- **Your final answer is checked.** Every number in it is matched against the observations \
this session actually produced, and one that appears nowhere is rejected. {_NO_ESTIMATES}

Work step by step, verify each stage from the observations, and finish with a concise \
report of what was done and what was measured.

The rules below are the same ones a session with a person is given.
"""


#: The same driver with a researcher at the terminal (bare `myrmidon`).
CHAT_PREAMBLE = f"""\
You are a molecular-simulation research agent working with a researcher at a terminal. You \
drive myrmidon yourself: {_REGISTRY}

- **The researcher decides what only a person may decide.** When a call is refused on a \
finding that needs a human (a compute budget, say), the terminal shows it to the researcher \
and asks; you never acknowledge such a finding yourself. If they approve, the call is made \
again with their approval and you receive its result; if they decline, say what was \
declined and what it would have cost, and do not repeat the call.
- **Ask when the choice is theirs.** A scientific choice the request does not settle — a \
charge, a QM region, a force field, how long to run — is a question for the researcher, \
not an assumption. Ask it in your reply and stop.
- **Your answers are checked.** Every number in a reply is matched against the observations \
this session produced, and one that appears nowhere is rejected. {_NO_ESTIMATES}

Work step by step, verify each stage from the observations, and keep replies short: what \
was done, what was measured, what needs the researcher.

The rules below are the same ones every session is given.
"""


def chat_system_prompt() -> str:
    """The prompt for a session with a researcher at the terminal."""
    from myrmidon.server.install import guidance_text

    return f"{CHAT_PREAMBLE}\n{guidance_text().strip()}\n"


def default_system_prompt() -> str:
    """The loop's prompt: its own preamble, then the guidance both drivers share.

    Read at call time rather than at import, so editing `guidance.md` changes what the next
    episode is told without reloading the module — and so no copy of the rules can go stale
    in a Python string again.
    """
    from myrmidon.server.install import guidance_text

    return f"{LOOP_PREAMBLE}\n{guidance_text().strip()}\n"
