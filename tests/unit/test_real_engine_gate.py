"""Running the tests never spends real compute unless the developer says so.

On a slurm machine the integration tests find real engines; finding them is not
permission to queue a job, so a bare `pytest` must not collect them at all.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bare_pytest_collects_only_unit_tests():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/unit"]


def test_integration_tests_skip_without_explicit_authorisation():
    env = {k: v for k, v in os.environ.items() if k != "MYRMIDON_REAL_ENGINES"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration", "-q", "-rs",
         "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert " passed" not in out and " failed" not in out, out
    assert "set MYRMIDON_REAL_ENGINES=1" in out, out
