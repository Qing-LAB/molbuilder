"""THE RUN SCRIPT'S TWO ZONES OF YOUR OWN LINES -- `execution/job-contracts.md`
§ 3.5, `execution/running-a-job.md` § 1 *(the user, 2026-11-19: one zone
before the engine, one after it)*.

Real PySCF H2 calculations prepared on the road (`support.real_runs`), each
run script's zones written as a person writes them -- lines typed between a
zone's two markers, the one place of the file that is theirs -- prepared
again, the lines kept, and launched here in the foreground, one case a
calculation: what ran, in which order, what reached the engine, and what the
run concluded.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from support.real_runs import (RealRun, makes_a_real_run, on_the_road,
                               script_dry_runs, typed)

#: (the case, the lines before the engine, the lines after it): each a
#: calculation of its own.
CASES = [
    ("both zones, the engine succeeding",
     ['date +%s.%N > own-before.stamp', "export MB_OWN_SEEN=yes"],
     ['printf %s "$MB_OWN_SEEN" > own-after.seen',
      'date +%s.%N > own-after.stamp']),
    ("a limit before the engine reaching it",
     ["ulimit -v 60000"],
     ["echo ran > own-after.ran"]),
    ("a failing line before the engine",
     ["false"],
     ["echo ran > own-after.ran"]),
]


def _typed_into(script: Path, zone: str, lines) -> None:
    """``lines`` typed between ``zone``'s two markers of ``script``, what
    was there before them replaced -- as a person edits their zone."""
    from molbuilder.deck_record import begin_marker, end_marker
    text = script.read_text().splitlines()
    begin = text.index(begin_marker(zone))
    end = text.index(end_marker(zone))
    script.write_text("\n".join(text[:begin + 1] + list(lines) + text[end:]) + "\n")


@makes_a_real_run("module", "molbuilder-pySCF")
def own_lines_ran(tmp_path_factory):
    """Each of :data:`CASES` -- an H2 optimization on PySCF, its run script's
    zones typed, prepared again and launched: ``{case: what it left}`` --
    the run's folder, what the zones held after the second prep, the
    launch's exit code."""
    from molbuilder.deck_record import BLOCK_USER_BEFORE_ENGINE, BLOCK_USER_CUSTOM
    from support.road import described, structure_saved
    out = {}
    with on_the_road(tmp_path_factory, "own-lines", engine="pyscf") as tree:
        for i, (case, before, after) in enumerate(CASES):
            run = RealRun(tree=tree, bundle=described(
                tree, f"P/optimization/OWN{i}", structure=structure_saved(tree),
                name=f"OWN{i}", calculation="optimization", engine="pyscf",
                execution={"threads": 1}))
            prep = ("prep", "task", "--group", "1", "--target", "this")
            assert typed(run, "prep", *prep).exit_code == 0, run.said["prep"]
            stage = run.attempt("coarse").parent
            script = next(stage.glob("*.run.sh"))
            _typed_into(script, BLOCK_USER_BEFORE_ENGINE, before)
            _typed_into(script, BLOCK_USER_CUSTOM, after)
            assert typed(run, "prep again", *prep).exit_code == 0, run.said["prep again"]
            again = script.read_text()
            dry = tmp_path_factory.mktemp(f"own-dry-{i}")
            script_dry_runs(run, stage, dry)
            launched = typed(run, "launch", "launch", "task", "--stage", "coarse",
                             "--mode", "direct", "--foreground", "--yes")
            out[case] = {"folder": run.attempt("coarse"), "script": again,
                         "exit": launched.exit_code, "said": launched.output,
                         "dry": dry / stage.name,
                         "dry said": run.said["script dry run"]}
    return out


def _left(ran):
    folder = ran["folder"]
    one = lambda pattern: next(folder.glob(pattern), None)
    concluded = one("*.concluded")
    return {
        "rc": (concluded.read_text().split()[0] if concluded else None),
        "log": one("*.runwrap.log").read_text(),
        "output": one("*.pyscf.log"),
        "file": lambda name: (folder / name) if (folder / name).exists() else None,
    }


@pytest.mark.parametrize("case", [c[0] for c in CASES])
def test_a_prep_again_keeps_your_lines(case, own_lines_ran):
    """`job-contracts.md` § 3.5: each zone's lines are "copied line for line,
    unchanged" into the new file -- the run script's two zones each kept by
    its own markers.

    THE SILENT FAILURE: a prep again dropping a zone, or carrying one's lines
    into the other -- a cleanup line moved before the engine."""
    from molbuilder.deck_record import (BLOCK_USER_BEFORE_ENGINE, BLOCK_USER_CUSTOM,
                                        begin_marker, end_marker)
    _case, before, after = next(c for c in CASES if c[0] == case)
    text = own_lines_ran[case]["script"].splitlines()
    held = lambda zone: text[text.index(begin_marker(zone)) + 1:
                             text.index(end_marker(zone))]
    assert held(BLOCK_USER_BEFORE_ENGINE) == before
    assert held(BLOCK_USER_CUSTOM) == after


def test_the_lines_before_run_before_the_engine_and_the_lines_after_after_it(
        own_lines_ran):
    """`job-contracts.md` § 3.5: the zone before the engine runs "before the
    engine starts, in the shell that starts it", the zone after it "after
    the engine ... before the run concludes" -- an export made before
    reaching the lines after, the engine's output written between them.

    THE SILENT FAILURE: lines a person put in for the engine run after it,
    where they reach nothing."""
    ran = own_lines_ran["both zones, the engine succeeding"]
    left = _left(ran)
    assert ran["exit"] == 0 and left["rc"] == "rc=0", (ran["said"], left["rc"])
    before = float(left["file"]("own-before.stamp").read_text())
    after = float(left["file"]("own-after.stamp").read_text())
    written = left["output"].stat().st_mtime
    assert before <= written <= after, (before, written, after)
    stamped = left["file"]("own-after.stamp").stat().st_mtime
    concluded = next(ran["folder"].glob("*.concluded")).stat().st_mtime
    assert stamped <= concluded, ("the lines after the engine ran after the "
                                  "run concluded", stamped, concluded)
    assert left["file"]("own-after.seen").read_text() == "yes", (
        "the export made before the engine did not reach the lines after it")


def test_a_limit_set_before_the_engine_reaches_it_and_the_lines_after_still_run(
        own_lines_ran):
    """`job-contracts.md` § 3.5: "what they set reaches the engine"; the
    lines after it run "whatever its code" -- a
    memory limit too small for PySCF set before it: the engine fails under
    it, the lines after run, the run concludes with the engine's code.

    THE SILENT FAILURE: a limit that never reached the engine; a cleanup that
    ran only when the engine succeeded."""
    ran = own_lines_ran["a limit before the engine reaching it"]
    left = _left(ran)
    assert left["rc"] not in (None, "rc=0"), left["rc"]
    assert ran["exit"] != 0
    assert left["file"]("own-after.ran"), "the lines after the engine did not run"


def test_a_failing_line_before_the_engine_stops_the_run(own_lines_ran):
    """`job-contracts.md` § 3.5: a line before the engine that fails "stops
    the run, as any line of the script does under `set -e`: it concludes
    `stopped before its end`" -- the engine not started.

    THE SILENT FAILURE: the engine started anyway, without what the person's
    lines were to give it."""
    ran = own_lines_ran["a failing line before the engine"]
    left = _left(ran)
    assert left["output"] is None, "the engine started after a failing line before it"
    assert ran["exit"] != 0, ran["said"]
    concluded = next(ran["folder"].glob("*.concluded")).read_text()
    assert "stopped before its end" in concluded, concluded
    assert left["file"]("own-after.ran") is None


def test_a_dry_run_runs_none_of_your_lines(own_lines_ran):
    """`job-contracts.md` § 3.5: the lines before the engine run "after a dry
    run's exit, so a dry run runs none of them".

    THE SILENT FAILURE: a dry run -- a person's look at what would run --
    running the lines that stage files or load modules."""
    ran = own_lines_ran["both zones, the engine succeeding"]
    assert ran["dry said"].exit_code == 0, ran["dry said"].output
    assert not (ran["dry"] / "own-before.stamp").exists()
