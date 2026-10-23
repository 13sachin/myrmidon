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
