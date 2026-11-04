"""``python -m myrmidon`` — the same CLI as the ``myrmidon`` console script, for a
shell that does not have the environment's bin directory on PATH."""

from myrmidon.cli import app

app(prog_name="myrmidon")
