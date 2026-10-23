"""env_script: the server owns its environment — sourcing, idempotence,
failure reporting, the sbatch line, and doctor's client report."""

import os
import stat

from myrmidon.config import MyrmidonConfig


def _script(tmp_path, body):
    p = tmp_path / "env_engines.sh"
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_engine_on_path_but_not_runnable_is_unavailable(tmp_path, monkeypatch, mock_engines):
    """The other half of the same failure: a binary that resolves on PATH but cannot load
    its shared libraries is NOT available. Reporting it as available moves the exit-127
    loader error from `doctor` into the middle of a study."""
    from myrmidon.engines.discovery import find_gromacs

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gmx_stub_d"
    stub.write_text("#!/bin/sh\necho 'gmx_stub_d: error while loading shared libraries: "
                    "libfftw3.so.3: cannot open shared object file' >&2\nexit 127\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    info = find_gromacs(MyrmidonConfig(gromacs_bin="gmx_stub_d",
                                       env_script=tmp_path / "env.sh"))
    assert info.available is False
    assert info.path == str(stub)
    assert "libfftw3.so.3" in info.hint and "not runnable" in info.hint
    assert "module-provided library is missing" in info.hint
    assert "env.sh" in info.hint  # names the script that was supposed to provide it
    # a runnable binary is still available (the mock gmx exits 0)
    assert find_gromacs(MyrmidonConfig(gromacs_bin="gmx")).available

