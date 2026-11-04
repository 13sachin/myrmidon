"""The endpoints myrmidon names, as inference backends Hermes can route a session to.

One `endpoints.toml` holds the machine's OpenAI-compatible servers; this registers each of
them as a Hermes provider, so `hermes --provider <name>` and the model picker offer the
same servers a myrmidon run uses. Hermes discovers model providers through its own root
(`plugins/model-providers/`), separate from the session plugin's.

Registration happens at import, which is a Hermes start: it reads the two files myrmidon
writes beside this plugin and asks nothing over the network, so a session does not wait and
a machine that is offline still gets its providers.

The key is never written anywhere: it is read from the file myrmidon keeps it in (mode 600)
and put into this process's environment, which is where a provider's ``env_vars`` tells
Hermes to look for it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: the endpoints and the path to their keys, written by `myrmidon mcp install` on every
#: launch. Read with the standard library: this runs in Hermes's interpreter, which has no
#: myrmidon to import and no YAML library of its own.
ENDPOINTS_FILE = "endpoints.json"
#: `export NAME=value`, as the secrets file writes it.
SECRET_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _secrets(path: str) -> dict[str, str]:
    """The variables the secrets file exports, read without a shell. Never printed."""
    out: dict[str, str] = {}
    if not path:
        return out
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        found = SECRET_LINE.match(line)
        if not found:
            continue
        value = found.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            out[found.group(1)] = value
    return out


def _endpoints() -> tuple[list[dict], dict[str, str]]:
    """The endpoints myrmidon named, and the secrets that reach them."""
    try:
        view = json.loads((Path(__file__).parent / ENDPOINTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], {}
    if not isinstance(view, dict):
        return [], {}
    listed = [ep for ep in (view.get("endpoints") or []) if isinstance(ep, dict)]
    return listed, _secrets(str(view.get("secrets", "")))


def register_endpoints() -> list[str]:
    """Register one provider per endpoint that has models. Returns the names registered."""
    from providers import register_provider
    from providers.base import ProviderProfile

    endpoints, secrets = _endpoints()
    registered: list[str] = []
    for endpoint in endpoints:
        name = str(endpoint.get("name") or "")
        base = str(endpoint.get("api_base") or "")
        key_env = str(endpoint.get("key_env") or "")
        models = tuple(str(m) for m in (endpoint.get("models") or []) if m)
        if not (name and base and key_env and models):
            continue
        if key_env in secrets and not os.environ.get(key_env):
            os.environ[key_env] = secrets[key_env]
        register_provider(ProviderProfile(
            name=f"myrmidon-{name}",
            aliases=(name,),
            display_name=f"{name} (myrmidon endpoints.toml)",
            description=f"{name} — an OpenAI-compatible endpoint myrmidon names",
            env_vars=(key_env,),
            base_url=base,
            auth_type="api_key",
            default_aux_model=models[0],
            fallback_models=models,
        ))
        registered.append(name)
    return registered


try:
    REGISTERED = register_endpoints()
except Exception:  # a machine with no endpoints registers none; Hermes's own providers stand
    REGISTERED = []
