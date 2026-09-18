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


# `test_status_shape` stood here.  It asserted `s.state in ("running",
# "stale", "finished", "failed")` -- the set of EVERY answer the function can
# give, so it passed whatever came back -- and then that four keys were
# present.  This file's own docstring already said as much.
#
# `run_status` returns a frozen `RunStatus` since 2026-09-09: the four fields
# are the declaration, and `__post_init__` refuses a state outside
# `RUN_STATES` where the wrong value is WRITTEN rather than where it is read.
# Demonstrated: `RunStatus("Finished", "d")` raises.


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
    assert run_status(tmp_path).state == "failed"


def test_a_concluded_molwatch_log_finishes_a_run_with_no_out(tmp_path):
    """A PySCF attempt: no `.out` ever exists, the concluded molwatch
    log is the result file, and the answer is `finished`."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=True)
    s = run_status(tmp_path)
    assert s.state == "finished"
    assert s.detail == "job_completed"


def test_a_finished_pyscf_run_reads_its_own_stdout(tmp_path):
    """THE DEFECT § 5c.2 EXISTS FOR.  A PySCF spectrum deck writes no
    molwatch log at all, so the only evidence of how it ended is the stdout
    the wrapper captured -- and nothing looked for it.  Two directories in
    `projects/` reported `running` this way, the older for 97.6 days, with
    their end line sitting in the file.

    The end line is the SPECTRUM deck's, which is not the relaxation deck's:
    that is why a reader taught only "Job complete in" fixes neither.
    """
    from molbuilder.pyscf.vibration_emitters import END_MARKER
    (tmp_path / "spectra.spectra-run0.pyscf.log").write_text(
        "converged SCF energy = -1028.273\n"
        "Phase 4 done: 36 modes with ES data\n"
        f"{END_MARKER} 5090.8 s\n"
        "Results: /x/spectra.spectra.json\n", encoding="utf-8")
    s = run_status(tmp_path)
    assert s.state == "finished"
    assert s.detail == "job_completed"
    assert s.active_source == "spectra.spectra-run0.pyscf.log"


def test_a_seed_molwatch_log_is_a_live_view_not_a_result(tmp_path):
    """The prep-time seed has no conclusion footer, so it contributes
    nothing: the run reads as running with no result yet — never as
    finished, and never as a state the seed's mtime could steer."""
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=False)
    s = run_status(tmp_path)
    assert s.state == "running"
    assert s.detail == "no result file yet"


# ---------------------------------------------------------------------------
#  The state nothing asserted
# ---------------------------------------------------------------------------

def test_a_dead_run_goes_stale_rather_than_running_for_ever(tmp_path):
    """No ending marker and no growth is a dead job, not a live one.

    `run_status`'s docstring names staleness as one of the two reasons
    the module exists -- *"Only the filesystem can [tell], so the age
    check lives here and nowhere else"* -- and NOTHING in the tree
    asserted it.  The two places that mention the state both spell
    `assert state in ("running", "stale", "finished", "failed")`, which
    is membership in the set of every possible answer and is therefore
    free: it passes whatever the code returns.

    Measured 2026-09-05: deleting the whole `elif age_s > 60.0` branch
    left **368 tests passing**.  The user-visible consequence is a job
    the scheduler killed -- no marker written, no further writes --
    reporting `running` on the Results tab and in `jobset status`
    for ever, which is precisely the 2026-07-27 regression the
    `failed` test above was written for, in the neighbouring branch.
    """
    import os

    (tmp_path / "dead.fdf").write_text("SystemLabel dead\n")
    out = tmp_path / "dead.out"
    # Started, never finished: no ">> End of run", no fatal marker.
    out.write_text("Siesta Version: 5.4.2\nsiesta: iscf   Eharris\nscf:  1  -100.0\n")

    old = _wall_now_for_test() - 3600.0        # an hour with no write
    os.utime(out, (old, old))

    s = run_status(tmp_path)
    assert s.state == "stale", (
        f"an hour-dead run reports {s['state']!r}: {s}")
    assert "no file growth" in s.detail, s.detail

    # ...and a run touched JUST NOW is still running, or the check above
    # would pass on a clock bug that ages everything.
    now = _wall_now_for_test()
    os.utime(out, (now, now))
    fresh = run_status(tmp_path)
    assert fresh.state == "running", (
        f"a run written this second reports {fresh['state']!r}: {fresh}")


def _wall_now_for_test() -> float:
    """The same clock `run_status` measures age against."""
    from molbuilder.parse.dirs.job import _wall_now
    return _wall_now()


def test_a_live_run_is_not_stale_because_its_stdout_is_block_buffered(tmp_path):
    """LIVENESS IS NOT THE SPEAKER (`model/parse.md` § 5.5).

    The wrapper runs `python script > $_out_file 2>&1` with no `-u`, so an
    engine's stdout is BLOCK-BUFFERED: a real PySCF log grew 13 KB across
    146 s -- two flushes in the whole run.  The progress log flushes per step
    and is what proves the run alive, but it is deliberately NOT a SPEAKER
    until its footer concludes (a seed must not outrank a result), so asking
    the speaker's mtime reports a live run as dead.

    Staleness is therefore measured on the FRESHEST run-output file, while
    which file SPEAKS is unchanged.
    """
    import os

    log = tmp_path / "w_01_coarse-run0.pyscf.log"
    log.write_text("cycle= 1 E= -76.3\n", encoding="utf-8")   # no end line
    old = _wall_now_for_test() - 600.0                          # ten minutes
    os.utime(log, (old, old))

    # No progress log yet: the stdout is all there is, and it has not moved.
    assert run_status(tmp_path).state == "stale"

    # The run IS alive -- it is stepping, and the step log says so.
    mw = _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=False)
    now = _wall_now_for_test()
    os.utime(mw, (now, now))
    s = run_status(tmp_path)
    assert s.state == "running", (
        f"a run whose progress log moved this second reports {s.state!r} "
        f"-- liveness was taken from the block-buffered stdout: {s}")
    # ...and the SPEAKER is still the stdout, not the unconcluded seed.
    assert s.active_source == "w_01_coarse-run0.pyscf.log"

    # Once nothing grows at all, it is stale again.
    os.utime(mw, (old, old))
    assert run_status(tmp_path).state == "stale"


def test_the_active_file_is_the_highest_stage_not_the_newest_write(tmp_path):
    """Stage first, mtime second — and only a re-run separates them.

    A staged run's stages finish in order, so on almost every real
    directory "highest stage" and "newest mtime" name the same file and
    the sort key's first component is invisible.  Measured 2026-09-05
    across all 115 run directories under `projects/`: deleting
    `_detect_stage` from the key changes **nothing**, and 473 tests still
    pass.  The rule was chosen deliberately (user: "use stage-then-mtime")
    and nothing in the tree held it.

    The case that separates them is a RE-RUN of an earlier stage — the
    coarse stage re-run to check something after the final one finished.
    mtime alone then makes the coarse `.out` the active file, so the
    Results tab reports the run's state from a stage it has left behind.
    """
    import os

    (tmp_path / "job.fdf").write_text("SystemLabel job\n")
    done = "Siesta Version: 5.4.2\nsiesta: iscf\n>> End of run:  1-JAN-2026\n"
    running = "Siesta Version: 5.4.2\nsiesta: iscf\nscf:  1  -100.0\n"

    final  = tmp_path / "job_03_final-run0.out"
    coarse = tmp_path / "job_01_coarse-run0.out"
    final.write_text(done)          # stage 3 finished...
    coarse.write_text(running)      # ...then stage 1 was re-run and is live

    now = _wall_now_for_test()
    os.utime(final,  (now - 600, now - 600))   # older write, higher stage
    os.utime(coarse, (now,       now))         # newest write, lower stage

    s = run_status(tmp_path)
    assert s.active_source == final.name, (
        "the newest write won over the highest stage — that is mtime-only "
        f"ordering: {s}")

    # Anti-vacuity: WITHIN one stage, mtime still decides. Otherwise the
    # assertion above would also pass on a key that ignored mtime entirely.
    later = tmp_path / "job_03_final-run1.out"
    later.write_text(running)
    os.utime(later, (now - 300, now - 300))    # newer than run0, same stage
    s2 = run_status(tmp_path)
    assert s2.active_source == later.name, (
        f"within one stage the later attempt must win: {s2}")


def test_reading_a_directory_writes_nothing_into_it(tmp_path, monkeypatch):
    """A status probe is a READ.

    `run_status` full-parsed every molwatch log through the registry, which
    opens a `ParseLogger`, so every Watch poll created and grew a
    `.parse.log` inside the user's project -- 177 of them, 31.5 MB, deleted
    by hand on 2026-09-18.  The `.out` half moved to a cheap door the same
    day; the molwatch half did not.
    """
    monkeypatch.delenv("MOLBUILDER_PARSE_LOG", raising=False)   # default ON
    _mw_log(tmp_path, "w_01_coarse.molwatch.log", concluded=True)
    # THE LOG MUST BE ONE THE REGISTRY CAN PARSE, or the old code raises
    # before opening its logger and this passes against the very thing it
    # refuses.  `_mw_log` writes a real log's shape; the sibling test above
    # proves the registry reads it.
    before = {p.name for p in tmp_path.iterdir()}
    for _ in range(3):
        run_status(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == before, (
        "reading the directory created files in it: "
        f"{ {p.name for p in tmp_path.iterdir()} - before }")
