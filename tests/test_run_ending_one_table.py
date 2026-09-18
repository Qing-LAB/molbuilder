"""The cheap ending scan and the full parser cannot disagree.

`model/parse.md` § 2b P-S4 -- one reader per question.  Two entry points
answer *"how did this run end"*: :func:`scan_ending` (stdlib-only, one
pass) and the full :class:`SiestaParser` (which also builds Frames).  They
share ``FATAL_MARKERS``, and this file is the guard that keeps them
sharing it.

The history is why the guard exists.  `jobset/summarize.py` carried a
private `_DONE_MARKERS` tuple whose own comment knew a capped benchmark
with ``SCF.MustConverge .false.`` exits cleanly -- while the parser beside
it did not.  The bench summary asked both and rendered the wrong one: six
healthy trials shown as failures.
"""
from __future__ import annotations

import pathlib

import pytest

from molbuilder.parse.engines._run_ending import scan_ending
from molbuilder.parse.engines.siesta import SiestaParser

_FIXTURES = sorted(
    (pathlib.Path(__file__).parent / "watch/fixtures/siesta_frozen").glob("*.out"))


def test_the_scanner_needs_no_numpy():
    """It is stdlib-only ON PURPOSE: answering how a run ended is a
    substring scan, and a caller should not pay for arrays it will not
    read."""
    import ast
    src = (pathlib.Path("molbuilder/parse/engines/_run_ending.py")
           .read_text(encoding="utf-8"))
    imported = {n.split(".")[0]
                for node in ast.walk(ast.parse(src))
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for n in ([a.name for a in node.names]
                          + ([node.module] if isinstance(node, ast.ImportFrom)
                             and node.module else []))}
    assert "numpy" not in imported, f"the cheap door grew a heavy import: {imported}"


@pytest.mark.parametrize("out", _FIXTURES, ids=lambda p: p.name)
def test_both_doors_give_the_same_ending(out):
    """Every frozen fixture, both ways."""
    cheap = scan_ending(out.read_text(errors="replace"))
    full  = SiestaParser.parse(str(out))
    assert cheap.run_state == full.run_state, (
        f"{out.name}: the scan says {cheap.run_state!r}, the parser says "
        f"{full.run_state!r} -- two answers to one question is the defect "
        f"this file exists to prevent")
    assert cheap.scf_converged == full.scf_converged, (
        f"{out.name}: convergence disagrees "
        f"({cheap.scf_converged!r} vs {full.scf_converged!r})")


def test_a_capped_benchmark_reads_ended_through_BOTH_doors(tmp_path):
    """The live case, asserted on both sides at once."""
    text = ("Siesta Version: 5.4.2\n"
            "   scf:    1 -1740000.0 -1740000.0 -1740000.0  0.9  0.5  30.0\n"
            "SCF_NOT_CONV: SCF did not converge  in maximum number of steps.\n"
            "Using DM_out to compute the final energy and forces\n"
            ">> End of run:  25-AUG-2026   4:08:52\nJob completed\n")
    f = tmp_path / "bench-run0.out"; f.write_text(text, encoding="utf-8")
    assert scan_ending(text).run_state == "ended"
    assert SiestaParser.parse(str(f)).run_state == "ended"
    assert scan_ending(text).scf_converged is False


# ---- § 5.5: one reader per ROLE, and the end line lives once ----------- #


def test_every_run_output_role_has_a_reader_and_nothing_else_does():
    """`model/parse.md` § 5.5's enforcement, and the whole of it.

    The catalogue declares WHICH files are a run's output (`runfiles.WRITTEN`'s
    `output` column); `READERS` declares HOW each one says it ended.  Equality
    is what keeps *"adding an engine is two edits"* true across the layer
    split: a row added without a reader, or a reader for a file the catalogue
    does not call output, fails here instead of in a directory that reports
    `running` for 97 days.
    """
    from molbuilder.parse.engines._run_ending import READERS
    from molbuilder.runfiles import run_output_roles
    assert set(READERS) == set(run_output_roles()), (
        f"READERS has {sorted(set(READERS) - set(run_output_roles()))} extra "
        f"and is missing {sorted(set(run_output_roles()) - set(READERS))}")


@pytest.mark.parametrize("calculation,module", [
    ("optimization", "molbuilder.pyscf.input"),
    ("vibration",    "molbuilder.pyscf.vibration_emitters"),
])
def test_the_deck_prints_the_end_line_its_reader_looks_for(calculation, module):
    """The end text exists ONCE -- rendered from the same constant the reader
    imports (`model/parse.md` § 5.5: a format molbuilder GENERATES does not
    get a sniffed reader).

    This is the drift guard, and it is what was missing: PySCF's two decks
    print two different end lines, the spectrum deck's was read by nothing,
    and the two spectrum directories in the tree reported `running` for
    months with their answer sitting in the file.

    MUTATION THIS MUST FAIL AGAINST: respell the emitted `print(...)` without
    respelling the constant -- or respell the constant without the print.
    Either breaks the first assertion, because the rendered deck is searched
    for the constant the reader will look for.
    """
    import importlib

    import numpy as np

    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.parse.engines._run_ending import scan_pyscf_ending
    from molbuilder.pyscf.input import spec_for
    from molbuilder.script_emit import render_deck
    from molbuilder.structure import Structure

    marker = importlib.import_module(module).END_MARKER
    struct = Structure(elements=["O", "H", "H"],
                       positions=np.array([[0.0, 0.0, 0.119],
                                           [0.0, 0.757, -0.477],
                                           [0.0, -0.757, -0.477]]))
    cfg = PySCFConfig(optimize=(calculation == "optimization"))
    deck = render_deck(spec_for(struct, cfg, calculation=calculation),
                       struct, cfg, verbose=False)
    printed = [ln for ln in deck.splitlines()
               if ln.lstrip().startswith("print(") and marker in ln]
    assert printed, (
        f"the {calculation} deck prints no line carrying {marker!r} -- the "
        f"emitter and its reader have drifted apart")

    # ...and a log carrying that line reads as ENDED.
    assert scan_pyscf_ending(f"some output\n{marker} 12.3 s\n").run_state \
        == "ended"


def test_a_caught_failure_above_the_end_line_is_still_a_finished_run():
    """The real shape, from `BDT/optimization/BDT-only-pySCF`: the frequency
    analysis raised, the script CAUGHT it, said so, and went on to write the
    optimized geometry and print its end line.  The run ended -- the energy
    and the geometry are on disk -- so the end line outranks the report.
    """
    from molbuilder.parse.engines._run_ending import (PYSCF_END_MARKER,
                                                      scan_pyscf_ending)
    text = ("=== Stage: harmonic frequencies + thermochemistry ===\n"
            "Frequency analysis FAILED: unsupported format string passed to "
            "numpy.ndarray.__format__\n"
            "Wrote /tmp/pyscf_relax_optimized.xyz\n"
            f"\n{PYSCF_END_MARKER} 145.8 s\n")
    assert scan_pyscf_ending(text).run_state == "ended"


def test_an_uncaught_exception_is_stopped_and_names_its_own_line():
    """Python's traceback IS sniffed, because it is Python's shape and not
    one we print -- and the sentence a person wants is the exception line at
    the bottom, not the header at the top."""
    from molbuilder.parse.engines._run_ending import scan_pyscf_ending
    text = ("converged SCF energy = -76.4\n"
            "Traceback (most recent call last):\n"
            '  File "job.py", line 88, in <module>\n'
            "    mf.kernel()\n"
            "ValueError: basis set 'nonesuch' not found\n")
    got = scan_pyscf_ending(text)
    assert got.run_state == "stopped"
    assert got.error_message == "ValueError: basis set 'nonesuch' not found"


def test_a_log_that_has_only_started_is_running_not_unknown():
    """§ 2b P-S1: nothing IN a file separates a slow SCF step from a job the
    scheduler killed, so content answers `running` and the FILESYSTEM decides
    -- which is `parse/dirs/job.py`'s half, not this one's."""
    from molbuilder.parse.engines._run_ending import scan_pyscf_ending
    assert scan_pyscf_ending("cycle= 1 E= -76.3\n").run_state == "running"


def test_ending_of_dispatches_on_the_role_and_refuses_anything_else(tmp_path):
    """One door over a run-output file, keyed on what the file IS.

    The role is read off the NAME with no label (`runfiles.role_of`), which
    is what lets a caller holding one path ask at all.  A file that is not run
    output is refused rather than answered `unknown`: `READERS`' keys are the
    catalogue's own `run_output_roles()`, so getting here means the caller
    asked about the wrong file.
    """
    from molbuilder.parse.engines._run_ending import (END_MARKER,
                                                      PYSCF_END_MARKER,
                                                      ending_of)
    siesta = tmp_path / "job_01_coarse-run0.out"
    siesta.write_text(f"{END_MARKER}:  25-AUG-2026\n", encoding="utf-8")
    pyscf = tmp_path / "job_01_coarse-run0.pyscf.log"
    pyscf.write_text(f"{PYSCF_END_MARKER} 3.2 s\n", encoding="utf-8")
    assert ending_of(siesta).run_state == "ended"
    assert ending_of(pyscf).run_state == "ended"

    deck = tmp_path / "job_01_coarse.fdf"
    deck.write_text("SystemLabel job\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a run-output file"):
        ending_of(deck)


def test_a_seeded_progress_log_says_nothing_about_a_run(tmp_path):
    """`output == "progress"`: a molwatch log is written at PREP, before the
    engine exists, so an empty one must not outrank a real result.  It speaks
    only once its footer concludes."""
    from molbuilder.parse.engines._run_ending import ending_of
    seed = tmp_path / "job_01_coarse.molwatch.log"
    seed.write_text("# engine: pyscf\n# label: job\n", encoding="utf-8")
    assert ending_of(seed).run_state == "running"
