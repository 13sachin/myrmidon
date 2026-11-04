"""Myrmidon error hierarchy.

Errors raised here are harness-level failures (misconfiguration, misuse).
Scientific failures (a crashed run, a refused validation) are NOT exceptions:
they are returned as Observations with ok=False so the agent can react to them.
"""


class MyrmidonError(Exception):
    """Base class for all harness errors."""


class ConfigError(MyrmidonError):
    """Invalid or unreadable configuration."""


class EngineNotFoundError(MyrmidonError):
    """A required simulation engine is not available on this machine."""

    def __init__(self, engine: str, hint: str = ""):
        self.engine = engine
        msg = f"engine '{engine}' not found anywhere on this machine"
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)




class WorkspaceError(MyrmidonError):
    """Workspace directory is missing, corrupt, or not a myrmidon workspace."""


class UnknownToolError(MyrmidonError):
    """Tool name not present in the registry."""


class JobNotFoundError(MyrmidonError):
    """Job id not present in the job store."""
