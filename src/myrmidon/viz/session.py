"""Live ChimeraX session driven over its REST interface.

One ChimeraX process per workspace, launched on first use and reused across
tool calls: the structure stays loaded, so each iteration of a figure is a
sub-second command against existing state instead of a cold reload. Session
coordinates (pid, port) live in the workspace state dir.

Response shape (``remotecontrol rest start ... json true``) of ChimeraX 1.12:

    {"json values": [...], "python values": [...],
     "log messages": {"note": [...], "warning": [...], "error": [...], "bug": [...]},
     "error": null | {"type": "UserError", "message": "..."}}

On macOS the GUI must run (offscreen rendering is Linux-only), which doubles
as a live view the user can watch and grab; on Linux we pass
``--nogui --offscreen``.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from myrmidon.errors import MyrmidonError

STATE_FILENAME = "viz_session.json"
_STARTUP_TIMEOUT_S = 90  # GUI cold start on a laptop can take a while


class VizSessionError(MyrmidonError):
    pass


@dataclass
class CommandResult:
    command: str
    ok: bool
    error: str | None
    notes: list[str]
    warnings: list[str]
    json_values: list  # decoded from the JSON-encoded strings ChimeraX returns

    def text(self) -> str:
        return "".join(self.notes)


class VizSession:
    def __init__(self, port: int, pid: int):
        self.port = port
        self.pid = pid

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def _state_path(cls, state_dir: Path) -> Path:
        return state_dir / STATE_FILENAME

    @classmethod
    def attach(cls, state_dir: Path) -> VizSession | None:
        """Reattach to a previously launched session, or None if it is gone."""
        path = cls._state_path(state_dir)
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text())
            session = cls(port=int(state["port"]), pid=int(state["pid"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not session.alive():
            path.unlink(missing_ok=True)
            return None
        return session

    @classmethod
    def launch(cls, chimerax_bin: str, state_dir: Path) -> VizSession:
        port = _free_port()
        args = [chimerax_bin]
        if sys.platform != "darwin":
            args += ["--nogui", "--offscreen"]
        args += ["--cmd", f"remotecontrol rest start port {port} json true"]
        # --nogui ChimeraX reads commands from stdin and exits (code 0) on EOF,
        # so an inherited /dev/null or closed stdin kills the REST server the
        # instant it starts. Hand it a pipe whose
        # write end it also inherits: nobody ever closes it, so no EOF, and the
        # process outlives whichever tool call launched it.
        stdin_r, stdin_w = os.pipe()
        try:
            proc = subprocess.Popen(  # noqa: S603
                args,
                stdin=stdin_r,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(stdin_w,),
                cwd=state_dir,  # anything it drops (core files!) stays out of the repo
                start_new_session=True,  # outlive the tool call, die on close_session
            )
        finally:
            os.close(stdin_r)
            os.close(stdin_w)
        session = cls(port=port, pid=proc.pid)
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise VizSessionError(
                    f"ChimeraX exited with code {proc.returncode} during startup "
                    f"(command: {' '.join(args)})"
                )
            if session.alive(timeout=2):
                cls._state_path(state_dir).write_text(
                    json.dumps({"port": port, "pid": proc.pid})
                )
                return session
            time.sleep(0.5)
        proc.terminate()
        raise VizSessionError(
            f"ChimeraX REST interface did not answer on port {port} within "
            f"{_STARTUP_TIMEOUT_S}s"
        )

    @classmethod
    def ensure(cls, chimerax_bin: str, state_dir: Path) -> tuple[VizSession, bool]:
        """Attach to the workspace's session, launching if needed.

        Returns (session, started) where started is True on a fresh launch.
        """
        session = cls.attach(state_dir)
        if session is not None:
            return session, False
        return cls.launch(chimerax_bin, state_dir), True

    def close(self, state_dir: Path) -> None:
        if sys.platform == "darwin":
            # GUI session: a clean `exit` closes the window the user is watching.
            with contextlib.suppress(VizSessionError):  # REST already gone — signal below
                self.run("exit", timeout=10)
        else:
            # --nogui ChimeraX 1.12 aborts on `exit` sent over REST and dumps a
            # ~400 MB core; SIGTERM ends it cleanly and the
            # session holds no state worth a graceful shutdown.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(self.pid, 15)
        for _ in range(20):
            if not _pid_alive(self.pid):
                break
            time.sleep(0.25)
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(self.pid, 15)
        self._state_path(state_dir).unlink(missing_ok=True)

    def alive(self, timeout: float = 5) -> bool:
        try:
            self._request("version", timeout=timeout)
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    # -- command execution ---------------------------------------------------

    def _request(self, command: str, timeout: float) -> dict:
        # The command MUST travel in the query string: ChimeraX's REST server
        # ignores urlencoded POST bodies ("command" parameter missing).
        query = urllib.parse.urlencode({"command": command})
        url = f"http://127.0.0.1:{self.port}/run?{query}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())

    def run(self, command: str, timeout: float = 120) -> CommandResult:
        """Run one ChimeraX command; never raises on command errors, only on
        transport failure (session died)."""
        try:
            raw = self._request(command, timeout=timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise VizSessionError(
                f"lost the ChimeraX session while running '{command}': {exc}"
            ) from exc
        logs = raw.get("log messages") or {}
        error = raw.get("error")
        message = None
        if error:
            message = error.get("message") or str(error)
        elif logs.get("error"):
            # Some failures surface only as log-level errors while the
            # top-level "error" stays null (seen on ChimeraX 1.12).
            message = " ".join(logs["error"]).strip()
        json_values = []
        for value in raw.get("json values") or []:
            if isinstance(value, str):
                try:
                    json_values.append(json.loads(value))
                except json.JSONDecodeError:
                    json_values.append(value)
            else:
                json_values.append(value)
        return CommandResult(
            command=command,
            ok=message is None,
            error=message,
            notes=list(logs.get("note") or []),
            warnings=list(logs.get("warning") or []),
            json_values=json_values,
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pid_alive(pid: int) -> bool:
    # A session this process launched is its child: after it exits it stays a zombie, which
    # `kill(pid, 0)` still finds, until it is reaped — so it is reaped here, or every close
    # would wait its full five seconds for a process that has ended. Another process's
    # session is not ours to reap, and falls through to the signal check.
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
