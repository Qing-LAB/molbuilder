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
from pathlib import Path

import pytest

from _road import conda_hook, env_available, live_siesta
from support.road import H2, group_words, jobset, road_structure
from support.road import molbuilder, one_rank

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _write_h2(tree: Path) -> None:
    from molbuilder.workingcopy_structure import StructureCodec
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    StructureCodec().write(road_structure(H2),
                           tree / "P" / "structure" / "h2.xyz")


@pytest.fixture(scope="module")
def walled(isolated_projects_root_module, tmp_path_factory):
    """An H2 optimization whose coarse stage ran here under the 20 s wall
    its run card states -- its run folder and what that launch said -- and
    what the launch of the stage again said over its restart file, the
    ``.XV``, re-dated to the job's end."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runs import run_of
    from molbuilder.task import of as task_of
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory):
        _write_h2(tree)
        bundle = tree / "P" / "opt" / "H2wall"
        r = molbuilder("describe", "--structure", "P/structure/h2.xyz",
                   "--bundle", "P/opt/H2wall", "--engine", "siesta",
                   "--calculation", "optimization",
                   "--name", "H2", "--psml-lib", "pseudopotential",
                   "--stage-strategy", "publishable")
        assert r.exit_code == 0, r.output
        # A RETRY BUDGET, which the wall must not spend (§ 3.5) -- and THE
        # WALL ITS RUN CARD STATES (`running-a-job.md` § 5.3.1's chain), 20
        # s, so the engine, told what is left and the 600 s slack, stops at
        # its first check.
        one_rank(bundle, continue_retries=1, time="00:00:20")
        r = jobset("prep", "task", *group_words(bundle, "coarse"),
                   "--bundle", bundle, "--target", "this")
        assert r.exit_code == 0, r.output
        launched = jobset("launch", "task", "--stage", "coarse", "--bundle",
                          bundle, "--mode", "direct", "--foreground", "--yes")
        ran_in = latest_attempt(stage_home(bundle, task_of(bundle),
                                           "coarse").dir)
        run = run_of(ran_in)
        # THE GEOMETRY SIESTA SAVED, RE-DATED TO THE JOB'S END SECOND, as a kill mid-write
        # leaves it; the next launch of the stage reads the two times.
        from molbuilder.parse.registry import parse as _parse
        ended_at = _parse(run.file(".monitor.log", 0)).metrics["ended_at"]
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
    way out, and the wrapper does not retry it in the job.

    MUTATION THIS MUST FAIL AGAINST: the card's wall not exported to a run
    here (`submit._go`; SIESTA relaxes H2 to its end, converged); the wall
    read as an SCF failure (the
    grammar without `OUT_OF_TIME`); a warm retry after it, its try 1."""
    from molbuilder.parse.dirs.rundir import run_state_of
    launched = walled["launched"]
    assert launched.exit_code == 1, launched.output
    assert any(ln.lstrip().startswith("FAILED") and "coarse" in ln
               and "OUT_OF_TIME" in ln
               for ln in launched.output.splitlines()), launched.output
    wrap = walled["wrap"].read_text(errors="replace")
    assert "Wall          : 20s from the job's start" in wrap, wrap[:3000]
    assert "wall: " in wrap and "SIESTA stops 600s before it" in wrap, wrap[-2000:]
    assert "stopped at its wall (MaxWalltime)" in wrap, wrap[-2000:]
    # NO WARM RETRY IN THE JOB: a retry would be the run's next try, in its
    # folder, and the run door names the newest.
    from molbuilder.runs import run_of
    run = run_of(walled["ran_in"])
    assert run.run == 0, sorted(p.name for p in walled["ran_in"].iterdir())
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
