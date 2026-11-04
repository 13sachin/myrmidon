"""Named model endpoints: several OpenAI-compatible servers side by side, each chosen per
run by name.

A provider's client reads one base URL and one key from the environment, so two
OpenAI-compatible endpoints could only be used one at a time, by swapping secrets files. A
named endpoint carries its base URL and the *name* of the environment variable that holds its
key, so every key can be loaded at once and a run says which endpoint it used:

    # ~/.config/myrmidon/endpoints.toml  (beside the machine config; holds no secret)
    [endpoints.site-a]
    api_base = "https://site-a.example/v1"
    key_env = "SITE_A_API_KEY"

    myrmidon --endpoint site-a --model openai/<model-id>

A key never enters this file, a manifest, a transcript or a report: an ``api_key`` field is
refused, and the resolved key is handed to the model call only. A transcript records the
endpoint's name. ``model_listing`` reads the endpoint's own entry for a model: two servers can
serve one model string with different weights, and an alias can be repointed; the listing is
what makes either visible.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

ENDPOINTS_NAME = "endpoints.toml"
#: the listing fields a manifest keeps: what identifies the served model, nothing else.
#: Not `created`: a served OpenAI-compatible endpoint may fill it with the time of the
#: request, so two runs of one model would record different manifests — a field that
#: changes with the clock identifies nothing.
LISTING_FIELDS = ("id", "owned_by", "root", "parent")


class EndpointError(ValueError):
    """An endpoint that is not defined, or whose key is not in the environment."""


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a stray `api_key = "..."` is refused

    name: str = ""
    api_base: str = Field(description="the server's OpenAI-compatible base URL")
    key_env: str = Field(description="the environment variable that holds its key")
    # the context its models serve, in tokens, when it differs from what a client assumes:
    # a client registering the endpoint's models gives them this
    context_window: int | None = Field(default=None, gt=0)

    @field_validator("api_base")
    @classmethod
    def _url(cls, v: str) -> str:
        if not re.match(r"^https?://\S+$", v):
            raise ValueError(f"api_base must be an http(s) URL, not {v!r}")
        return v.rstrip("/")

    @field_validator("key_env")
    @classmethod
    def _env_name(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            raise ValueError(f"key_env names an environment variable, not a key: {v!r}")
        return v

    def key(self) -> str:
        value = os.environ.get(self.key_env)
        if not value:
            raise EndpointError(
                f"endpoint {self.name!r}: ${self.key_env} is not set in this environment — "
                "source your secrets file (`export` on each line)")
        return value

    def call_kwargs(self) -> dict:
        """What each model call carries to reach this endpoint. Never recorded."""
        return {"api_base": self.api_base, "api_key": self.key()}

    def record(self) -> dict:
        """What a run may record about it: never the key, nor its variable's value."""
        return {"name": self.name, "api_base": self.api_base, "key_env": self.key_env}


def endpoints_path() -> Path:
    """$MYRMIDON_ENDPOINTS, else endpoints.toml beside the machine config."""
    env = os.environ.get("MYRMIDON_ENDPOINTS")
    if env:
        return Path(env).expanduser()
    from myrmidon.bootstrap import config_path

    return config_path().parent / ENDPOINTS_NAME


def load_endpoints(path: Path | None = None) -> dict[str, Endpoint]:
    path = path or endpoints_path()
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise EndpointError(f"{path}: not valid TOML: {exc}") from None
    out = {}
    for name, body in (data.get("endpoints") or {}).items():
        try:
            out[name] = Endpoint.model_validate({**body, "name": name})
        except ValueError as exc:
            hint = (" — a key belongs in the environment variable `key_env` names"
                    if "api_key" in str(exc) else "")
            raise EndpointError(f"{path}: endpoint {name!r}: {exc}{hint}") from None
    return out


def get_endpoint(name: str, path: Path | None = None) -> Endpoint:
    endpoints = load_endpoints(path)
    if name not in endpoints:
        known = ", ".join(sorted(endpoints)) or "none defined"
        raise EndpointError(
            f"no endpoint {name!r} in {path or endpoints_path()} ({known})")
    return endpoints[name]


def served_model_id(model: str) -> str:
    """The id the endpoint serves for a model string: a provider prefix (``openai/``,
    PydanticAI's ``openai:``) is ours, not the server's; any other slash is part
    of the id (``org/model``)."""
    for prefix in ("openai/", "openai:"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def secrets_path() -> Path:
    """The secrets file beside the machine config: `export NAME=value` per line, chmod 600."""
    from myrmidon.bootstrap import SECRETS_NAME, config_path

    return config_path().parent / SECRETS_NAME


def load_secrets(path: Path | None = None) -> dict[str, str]:
    """The variables the secrets file exports, read without a shell. Never printed."""
    path = path or secrets_path()
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, sep, value = line.partition("=")
        if not sep or not name.strip().isidentifier():
            continue
        out[name.strip()] = value.strip().strip("'\"")
    return out


def apply_secrets(path: Path | None = None) -> list[str]:
    """Set every variable the secrets file exports that this process lacks, for this
    process only, and name the ones set: a command a client's extension runs has the
    client's environment, not the shell the person sourced the file in."""
    added = []
    for name, value in load_secrets(path).items():
        if name not in os.environ:
            os.environ[name] = value
            added.append(name)
    return added


ENDPOINTS_HEADER = (
    "# Named OpenAI-compatible model endpoints (myrmidon endpoints). Holds no secret:\n"
    "# each key lives in secrets.env under the variable key_env names.\n")


def write_endpoint(name: str, api_base: str, key_env: str, path: Path | None = None) -> str:
    """Put one endpoint into endpoints.toml, keeping the others; the file is rewritten from
    what it holds. Returns "created" | "updated" | "unchanged"."""
    path = path or endpoints_path()
    current = load_endpoints(path)
    kept = current[name].context_window if name in current else None  # set by hand
    endpoint = Endpoint.model_validate({"name": name, "api_base": api_base, "key_env": key_env,
                                        "context_window": kept})
    if name in current and current[name] == endpoint:
        return "unchanged"
    existed = path.is_file()
    current[name] = endpoint
    lines = [ENDPOINTS_HEADER]
    for ep_name, ep in sorted(current.items()):
        lines.append(f'\n[endpoints.{ep_name}]\napi_base = "{ep.api_base}"\n'
                     f'key_env = "{ep.key_env}"\n'
                     + (f"context_window = {ep.context_window}\n" if ep.context_window else ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return "updated" if existed else "created"


def ensure_secret_line(key_env: str, path: Path | None = None) -> str:
    """Make sure the secrets file exists (chmod 600) and has a line for the variable, empty
    for the person to fill: `export KEY_ENV=`. Returns "created" | "added" | "present"."""
    import re
    import stat

    from myrmidon.bootstrap import SECRETS_TEMPLATE

    path = path or secrets_path()
    created = not path.is_file()
    text = SECRETS_TEMPLATE if created else path.read_text()
    if re.search(rf"^\s*(export\s+)?{re.escape(key_env)}=", text, re.M):
        state = "present"
    else:
        text = text.rstrip("\n") + f"\nexport {key_env}=\n"
        state = "created" if created else "added"
    if state != "present" or created:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return state


MODELS_CACHE_NAME = "endpoints.models.json"
MODELS_CACHE_TTL_S = 24 * 3600


def cached_models(endpoint: Endpoint, refresh: bool = False, timeout: float = 10.0,
                  fetch: bool = True) -> list[str]:
    """The model ids an endpoint serves, from a cache beside endpoints.toml refreshed once
    a day: a client registers the provider at every start, and a start must not wait on
    the network. Empty when the endpoint cannot be asked and nothing is cached.
    ``fetch=False`` reads the cache alone, whatever its age: the path a launch takes."""
    import json
    import time

    import httpx

    cache = endpoints_path().parent / MODELS_CACHE_NAME
    data: dict = {}
    if cache.is_file():
        try:
            data = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    entry = data.get(endpoint.name) or {}
    fresh = time.time() - float(entry.get("at", 0)) < MODELS_CACHE_TTL_S
    if not fetch or (entry.get("models") and fresh and not refresh):
        return list(entry.get("models") or [])
    try:
        models = sorted(str(m["id"]) for m in list_models(endpoint, timeout) if m.get("id"))
    except (EndpointError, httpx.HTTPError, ValueError):
        return list(entry.get("models") or [])
    data[endpoint.name] = {"at": time.time(), "models": models}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1) + "\n")
    except OSError:
        pass
    return models


def list_models(endpoint: Endpoint, timeout: float = 10.0) -> list[dict]:
    """The endpoint's ``GET {api_base}/models`` entries; raises when it cannot be read."""
    import httpx

    resp = httpx.get(f"{endpoint.api_base}/models", timeout=timeout,
                     headers={"Authorization": f"Bearer {endpoint.key()}"})
    resp.raise_for_status()
    return [e for e in resp.json().get("data", []) if isinstance(e, dict)]


def model_listing(endpoint: Endpoint, model: str, timeout: float = 10.0) -> dict:
    """The endpoint's own entry for ``model``, reduced to the fields that identify what is
    served; ``{"error": ...}`` when it cannot be read."""
    served = served_model_id(model)
    try:
        entries = list_models(endpoint, timeout)
    except EndpointError:
        raise
    except Exception as exc:  # the run says it could not check, rather than failing
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    entry = next((e for e in entries if e.get("id") == served), None)
    if entry is None:
        return {"error": f"{served!r} is not in the endpoint's model list "
                         f"({len(entries)} model(s) listed)"}
    return {k: entry[k] for k in LISTING_FIELDS if k in entry}
