"""The wall reaches the engine, and a warm file written as a job ended is
said at the next launch (`running-a-job.md` § 3.6; `engines/siesta.md`
§ 8.9) -- on the real SIESTA, the road's H2.

Driven as a person drives it: ``molbuilder describe`` -> ``prep task`` -> ``launch
task --mode direct --foreground`` with the wall a sent job carries in its
environment (``MB_WALL_S``, what a launch to a queue exports on its line),
then the next stage launched over a density file whose write time is the
job's end second.

PREVENTS: a wall the launch stated reaching SLURM and never the engine, so
the scheduler's kill lands mid-write (the Sol relaxation of 2026-10-18 that
restarted from mixed numbers); and a continuation taking such a file in
silence.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from _road import conda_hook, env_available, live_siesta
from support.road import H2, group_words, jobset, road_structure
from support.road import molbuilder, read_request, write_request

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


def _one_rank(bundle: Path, **card) -> None:
    request = read_request(bundle)
    request["execution"] = {**request.get("execution", {}), "mpi_np": 1,
                            "omp_threads": 1, **card}
    write_request(bundle, request)


@pytest.fixture(scope="module")
def walled(isolated_projects_root_module, tmp_path_factory):
    """A flat H2 optimization whose coarse stage ran under a 20 s wall the
    environment stated, as a sent job's does -- and what the launch of the
    next stage said over its density file, re-dated to the job's end."""
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory) as mp:
        _write_h2(tree)
        flat = tree / "P" / "opt" / "H2wall"
        r = molbuilder("describe", "--structure", "P/structure/h2.xyz",
                   "--bundle", "P/opt/H2wall", "--engine", "siesta",
                   "--calculation", "optimization", "--shape", "flat",
                   "--name", "H2", "--psml-lib", "pseudopotential",
                   "--stage-strategy", "publishable")
        assert r.exit_code == 0, r.output
        # A RETRY BUDGET, which the wall must not spend (§ 3.5).
        _one_rank(flat, continue_retries=1)
        r = jobset("prep", "task", *group_words(flat, "coarse"),
                   "--bundle", flat, "--target", "this")
        assert r.exit_code == 0, r.output
        # THE WALL A SENT JOB CARRIES (`scheduler.md` § 6a): 20 s, so the
        # engine, told what is left and the 600 s slack, stops at its
        # first check.
        mp.setenv("MB_WALL_S", "20")
        r = jobset("launch", "task", "--stage", "coarse", "--bundle", flat,
                   "--mode", "direct", "--foreground", "--yes")
        # THE RUN FAILS BY DESIGN -- SIESTA dies at its wall -- and the
        # launch says so with SIESTA's own word (running-a-job.md § 3.6).
        assert r.exit_code != 0 and "OUT_OF_TIME" in r.output, r.output
        mp.delenv("MB_WALL_S")
        out = flat / "H2_01_coarse-run0.out"
        wrap = sorted(flat.glob("H2_01_coarse.runwrap-*.log"))[-1]
        # THE GEOMETRY SIESTA SAVED, RE-DATED TO THE JOB'S END SECOND, as a kill mid-write
        # leaves it; the next stage's launch reads the two times.
        from molbuilder.parse.registry import parse as _parse
        mon = sorted(flat.glob("H2_01_coarse-run0.monitor.log"))[-1]
        ended_at = _parse(mon).metrics["ended_at"]
        from datetime import datetime
        end_s = datetime.fromisoformat(ended_at).timestamp()
        dm = flat / "H2.XV"
        os.utime(dm, (end_s + 1, end_s + 1))
        # THE SAME STAGE LAUNCHED AGAIN continues from its own latest run,
        # the files in the folder (`job-system.md` § 5.4) -- asked, not sent.
        plan = jobset("launch", "task", "--stage", "coarse", "--bundle",
                      flat, "--mode", "direct", "--foreground", "--yes",
                      "--dry-run")
        yield {"flat": flat, "out": out, "wrap": wrap,
               "ended_at": ended_at, "next": plan.output}


def test_the_engine_stops_itself_before_the_wall_and_ends_normally(walled):
    """`running-a-job.md` § 3.6, `siesta.md` § 8.9: the run script counts
    what is left of the wall and tells SIESTA on its command line in the
    catalogue's words; SIESTA stops at its first check and dies in its own
    word, `OUT_OF_TIME`, with its restart state saved -- the run door reads
    `failed` with the wall as the cause, not the SCF warning printed on the
    way out, and the wrapper does not retry it in the job.

    MUTATION THIS MUST FAIL AGAINST: the wall written into the deck at prep
    (a launch's `--time` would not reach it), or not passed at all (SIESTA
    relaxes H2 to its end, converged); the wall read as an SCF failure (the
    grammar without `OUT_OF_TIME`); a warm retry after it."""
    from molbuilder.parse.dirs.rundir import run_state_of
    from molbuilder.runfiles import RunNames
    wrap = walled["wrap"].read_text(errors="replace")
    assert "wall: " in wrap and "SIESTA stops 600s before it" in wrap, wrap[-2000:]
    assert "stopped at its wall (MaxWalltime)" in wrap, wrap[-2000:]
    assert "_mb_warm_retry" not in wrap.split("stopped at its wall")[-1]
    text = walled["out"].read_text(errors="replace")
    assert "OUT_OF_TIME: Time is up." in text, text[-3000:]
    assert ">> End of run" not in text
    st = run_state_of(walled["flat"], RunNames.of("H2", "01_coarse", "flat"))
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
    assert "H2.XV in the folder -- WARNING: written as the job ended" in said, said
    assert walled["ended_at"][11:19] in said, said
