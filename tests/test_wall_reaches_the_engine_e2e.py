"""The wall reaches the engine, and a warm file written as a job ended is
said at the next launch (`running-a-job.md` § 3.6; `engines/siesta.md`
§ 8.9) -- on the real SIESTA, the road's H2.

Driven as a person drives it: ``molbuilder describe`` -> the run card's
wall (``execution.time``) -> ``prep task`` -> ``launch task --mode direct
--foreground``, a run here told the wall its own job states (`submit._go`,
§ 3.6's table, *one run, here*), then the stage launched again, asked
dry, over a restart file -- SIESTA's geometry, ``.XV`` -- whose write time
is the job's end second.

PREVENTS: a wall the person stated reaching no engine, so a scheduler's kill
lands mid-write (the Sol relaxation of 2026-10-18 that restarted from mixed
numbers); and a continuation taking such a file in silence.
"""
from __future__ import annotations

import os
import re

import pytest

from _road import conda_hook, env_available
from support.road import group_words, jobset

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


@pytest.fixture(scope="module")
def walled(tmp_path_factory):
    """An H2 optimization whose coarse stage ran here under the 20 s wall
    its run card states -- its run folder and what that launch said -- and
    what the launch of the stage again said over its restart file, the
    ``.XV``, re-dated to the job's end."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runs import run_of
    from molbuilder.task import of as task_of
    from support.real_runs import h2_optimization, on_the_road
    with on_the_road(tmp_path_factory, "walled") as tree:
        # THE WALL ITS RUN CARD STATES (`architecture.md` § 5.2's
        # ladder), 20 s, one rank, so the engine, told what is left and the
        # 600 s slack, stops at its first check.
        bundle = h2_optimization(tree, "P/optimization/H2wall",
                                 execution={"mpi_np": 1, "omp_threads": 1,
                                            "time": "00:00:20"})
        r = jobset("prep", "task", *group_words(bundle, "coarse"),
                   "--bundle", bundle, "--target", "this")
        assert r.exit_code == 0, r.output
        launched = jobset("launch", "task", "--stage", "coarse", "--bundle",
                          bundle, "--mode", "direct", "--foreground", "--yes")
        ran_in = latest_attempt(stage_home(bundle, task_of(bundle),
                                           "coarse").dir)
        run = run_of(ran_in)
        # THE GEOMETRY SIESTA SAVED, RE-DATED TO THE JOB'S END SECOND, as a kill mid-write
        # leaves it; the next launch of the stage reads the two times.  ITS
        # WRITE TIME IS THE ONE THING SET BY HAND: a kill mid-write is a
        # scheduler's, and no run here is killed by one -- the file, the run
        # and the launch that reads them are the road's own.
        from molbuilder.parse.registry import parse as _parse
        ended_at = _parse(run.file(".monitor.log")).metrics["ended_at"]
        from datetime import datetime
        end_s = datetime.fromisoformat(ended_at).timestamp()
        xv = run.on_label(".XV")
        os.utime(xv, (end_s + 1, end_s + 1))
        # THE SAME STAGE LAUNCHED AGAIN continues from its own latest run,
        # its restart files copied in (`job-system.md` § 5.4) -- asked, not
        # sent.
        plan = jobset("launch", "task", "--stage", "coarse", "--bundle",
                      bundle, "--mode", "direct", "--foreground", "--yes",
                      "--dry-run")
        yield {"ran_in": ran_in, "out": run.stdout, "wrap": run.session_log,
               "launched": launched, "ended_at": ended_at,
               "next": plan.output}


def test_a_run_here_is_told_its_cards_wall_and_its_engine_stops_before_it(
        walled):
    """`running-a-job.md` § 3.6 (*one run, here*: the wall its own job
    states, in its environment), `siesta.md` § 8.9: the launch tells the run
    the wall its run card states; the run script counts what is left and
    tells SIESTA on its command line in the catalogue's words; SIESTA stops
    at its first check and dies in its own word, `OUT_OF_TIME`, with its
    restart state saved -- the launch says FAILED first and exits 1
    (`job-system.md` § 6.0, *A run here, waited on*), the run door reads
    `failed` with the wall as the cause, not the SCF warning printed on the
    way out, and the run is its one start: nothing in the job starts the
    engine again -- a launch again continues it (§ 3.5).

    The banner line and the session log's wall line are § 3.6's *What you
    see of the wall*, read by their label and value -- the banner pads its
    labels to the longest it prints (§ 1, since 2.64.0), so no column is
    pinned.

    MUTATION THIS MUST FAIL AGAINST: the card's wall not exported to a run
    here (`submit._go`; SIESTA relaxes H2 to its end, converged); the wall
    read as an SCF failure (the grammar without `OUT_OF_TIME`)."""
    from molbuilder.wrapper_log import WRAPPER_LOG_START
    from molbuilder.parse.dirs.rundir import run_state_of
    launched = walled["launched"]
    assert launched.exit_code == 1, launched.output
    assert any(ln.lstrip().startswith("FAILED") and "coarse" in ln
               and "OUT_OF_TIME" in ln
               for ln in launched.output.splitlines()), launched.output
    wrap = walled["wrap"].read_text(errors="replace")
    assert re.search(r"^\s*Time limit\s*: 20s from the job's start; SIESTA stops "
                     r"600s before it\s*$",
                     wrap, re.M), wrap[:3000]
    assert re.search(r"\btime limit: \d+s left of 20s; SIESTA stops 600s before "
                     r"it\b", wrap), wrap[-2000:]
    assert "stopped at its time limit:" in wrap, wrap[-2000:]
    # ONE START (`project-layout.md` § 1.6.1): the session log holds the
    # run script's one start, the engine run once.
    assert wrap.count(WRAPPER_LOG_START) == 1, wrap[-3000:]
    from molbuilder.runs import run_of
    run = run_of(walled["ran_in"])
    text = walled["out"].read_text(errors="replace")
    assert "OUT_OF_TIME: Time is up." in text, text[-3000:]
    assert ">> End of run" not in text
    st = run_state_of(walled["ran_in"], run.names)
    assert st.state == "failed", (st.state, st.detail)
    assert "OUT_OF_TIME" in st.detail, st.detail


def test_a_warm_file_written_as_the_job_ended_is_said_at_the_next_launch(walled):
    """`running-a-job.md` § 3.6: the launch compares each file it takes with
    the second its run's monitor closed, and names a file written in that
    second or after -- a fact shown, the launch still going.

    MUTATION THIS MUST FAIL AGAINST: the comparison dropped (nothing said),
    or the monitor log's closing time not read (`ended_at` absent)."""
    said = walled["next"]
    assert "continues from" in said, said
    assert "H2.XV -- WARNING: written as the job ended" in said, said
    assert walled["ended_at"][11:19] in said, said
