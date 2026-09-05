"""``run_status`` — how a run directory is doing, and nothing else.

*Replaces `test_job.py` (22 tests), retired 2026-09-04 with the decoder
it exercised.* Eighteen of those tests asserted on fields of the
eleven-field `JobResult` — job type, engine-body summary, plot buckets,
per-stage input envelope, geometry, progress, the source-file index —
none of which had a reader anywhere in the tree. One more
(`test_no_direct_out_grep_in_decoder`) was a lint whose whole body was
`assert src.count("read_text") < 8`.

The four kept here are the ones about STATUS, which is the only thing
anybody asked the decoder for. They are unchanged in what they assert;
they now ask `run_status` directly instead of taking one field out of a
summary that computed ten others to get it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse.dirs.job import run_status


def _multi_stage(tmp_path):
    """Several .out files — a real staged SIESTA run."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from support.junction import job_run_dir
    return job_run_dir(tmp_path, out_names=(
        "hemeC-stage1-scf_not_conv-5fr.out",
        "hemeC-stage2-run3-finished-42fr.out",
    ))


def _mw_log(dirpath, name, *, concluded):
    body = (
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "# job: w\n"
        "# units: energy=eV, force=eV/Ang, coords=Ang\n"
        "\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "kind: initial_preview\n"
        "n_atoms: 1\n"
        "coordinates (Ang):\n"
        "   H       0.0 0.0 0.0\n"
        "==== molwatch step 0 end ====\n"
    )
    if concluded:
        body += "\n# concluded: 2026-08-19T12:00:00\n"
    p = Path(dirpath) / name
    p.write_text(body)
    return p


def test_status_shape(tmp_path):
    s = run_status(_multi_stage(tmp_path))
    assert s["state"] in ("running", "stale", "finished", "failed")
    for fld in ("state", "detail", "last_change_at", "active_source"):
        assert fld in s


def test_a_crashed_run_reads_failed(tmp_path):
    """A run whose active `.out` carries a SIESTA fatal marker is
    `failed`.

    Regression (2026-07-27): the combiner branched on a `run_state` no
    engine parser emits, so `failed` was unreachable and a crashed run
    reported "stale" or "running" for ever.
    """
    (tmp_path / "crash.fdf").write_text(
        "SystemLabel crash\nMD.TypeOfRun CG\nMD.NumCGsteps 100\n")
    # Minimal but REAL SIESTA .out: the banner is what `can_parse`
    # sniffs, and the last line is a registered fatal marker.
    (tmp_path / "crash.out").write_text(
        "                           Welcome to SIESTA\n"
        "reinit: System Label: crash\n"
        "siesta: ERROR: out of memory in dense solver\n")
    assert run_status(tmp_path)["state"] == "failed"


def test_a_concluded_molwatch_log_finishes_a_run_with_no_out(tmp_path):
    """A PySCF attempt: no `.out` ever exists, the concluded molwatch
    log is the result file, and the answer is `finished`."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=True)
    s = run_status(tmp_path)
    assert s["state"] == "finished"
    assert s["detail"] == "job_completed"


def test_a_seed_molwatch_log_is_a_live_view_not_a_result(tmp_path):
    """The prep-time seed has no conclusion footer, so it contributes
    nothing: the run reads as running with no result yet — never as
    finished, and never as a state the seed's mtime could steer."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=False)
    s = run_status(tmp_path)
    assert s["state"] == "running"
    assert s["detail"] == "no result file yet"
