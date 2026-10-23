"""`myrmidon watch`: the feed a person reads while an unattended run works.

The feed is the ledger, not a model's output stream — so it is the same view whichever
agent framework is driving, and it works after the fact, from any terminal.
"""



from myrmidon.core.watch import running_progress


def test_a_running_job_shows_the_engines_own_counter(planted_run, tmp_path):
    """Without this a twenty-minute segment is a frozen screen, and a frozen screen
    cannot be told apart from a hang."""
    from myrmidon.execution.backend import JobSpec, JobState
    from myrmidon.execution.jobstore import JobRecord

    log = tmp_path / "md.log"
    log.write_text(
        "           Step           Time\n              0        0.00000\n"
        "           Step           Time\n           5000       10.00000\n"
        "Performance:      123.456        0.194\n"
    )
    record = JobRecord(
        job_id="job-x", backend="local", handle="1",
        spec=JobSpec(argv=["gmx", "mdrun", "-deffnm", "md"], cwd=tmp_path),
        state=JobState.RUNNING, submitted_at="2000-01-01T00:00:00",
    )
    assert "step 5000" in running_progress(record)

    cpmd = tmp_path / "cpmd.out"
    cpmd.write_text(
        "       NFI   TEMPP           EKS      ECLASSIC         DIS    TCPU\n"
        "         7     2.8     -45.62742     -45.61378   0.213E-03    1.74\n"
        "        SUBSYSTEM TEMPERATURES:\n"
        "        T(QM) =                                              14.05\n"
    )
    mimic = record.model_copy(update={"meta": {"kind": "mimic"}})
    assert "step 7" in running_progress(mimic)

