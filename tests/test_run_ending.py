"""How a run ended, asked through the one door -- `ending_of(path)`.

`model/parse.md` § 2b and § 5.5.  A SIESTA output's ending is the reading
pass's (`siesta_reader`), which the registered parser reads with too, so its
cases live with the parser (`tests/watch/test_siesta_parser_exit_status.py`,
`tests/test_siesta_scf_not_conv_is_not_a_death.py`).  Here: the door's own
dispatch, and the PySCF stdout's reader, whose shapes were measured on real
logs -- each test names the log.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.engines._run_ending import (PYSCF_END_MARKER,
                                                  PYSCF_SPECTRUM_END_MARKER,
                                                  ending_of)


def _pyscf(tmp_path, text):
    """How a PySCF stdout holding ``text`` says its run ended."""
    log = tmp_path / "job_01_coarse-run0.pyscf.log"
    log.write_text(text, encoding="utf-8")
    return ending_of(log)


@pytest.mark.parametrize("calculation,constant", [
    ("optimization", "END_MARKER"),
    ("vibration",    "SPECTRUM_END_MARKER"),
])
def test_the_deck_prints_the_end_line_its_reader_looks_for(tmp_path,
                                                           calculation,
                                                           constant):
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
    from molbuilder.pyscf.input import spec_for
    from molbuilder.script_emit import render_deck
    from molbuilder.structure import Structure

    # THE EMITTERS' PACKAGE declares both lines in one stdlib module, which
    # both decks print from and which travels beside the job for the
    # monitor's reader (`pyscf/end_lines.py`, `run-reports.md` § 2.3).
    marker = getattr(importlib.import_module("molbuilder.pyscf.end_lines"),
                     constant)
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
    assert _pyscf(tmp_path, f"some output\n{marker} 12.3 s\n").run_state \
        == "ended"


def test_an_uncaught_exception_is_stopped_and_names_its_own_line(tmp_path):
    """Python's traceback IS sniffed, because it is Python's shape and not
    one we print -- and the sentence a person wants is the exception line at
    the bottom, not the header at the top."""
    text = ("converged SCF energy = -76.4\n"
            "Traceback (most recent call last):\n"
            '  File "job.py", line 88, in <module>\n'
            "    mf.kernel()\n"
            "ValueError: basis set 'nonesuch' not found\n")
    got = _pyscf(tmp_path, text)
    assert got.run_state == "stopped"
    assert got.error_message == "ValueError: basis set 'nonesuch' not found"


def test_a_log_that_has_only_started_is_running_not_unknown(tmp_path):
    """§ 2b P-S1: nothing IN a file separates a slow SCF step from a job the
    scheduler killed, so content answers `running` and the FILESYSTEM decides
    -- which is `parse/dirs/job.py`'s half, not this one's."""
    assert _pyscf(tmp_path, "cycle= 1 E= -76.3\n").run_state == "running"


def test_ending_of_dispatches_on_the_role_and_refuses_anything_else(tmp_path):
    """One door over a run-output file, keyed on what the file IS.

    The role is read off the NAME with no label (`runfiles.role_of`), which
    is what lets a caller holding one path ask at all.  A file that is not run
    output is refused rather than answered `unknown`: `READERS`' keys are the
    catalogue's own `run_output_roles()`, so getting here means the caller
    asked about the wrong file.
    """
    siesta = tmp_path / "job_01_coarse-run0.out"
    siesta.write_text(">> End of run:  25-AUG-2026   4:08:52\n",
                      encoding="utf-8")
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
    seed = tmp_path / "job_01_coarse.molwatch.log"
    seed.write_text("# engine: pyscf\n# label: job\n", encoding="utf-8")
    assert ending_of(seed).run_state == "running"


def test_the_decks_own_source_is_not_mistaken_for_its_end_line(tmp_path):
    """PySCF ECHOES THE DECK INTO THE LOG, and the echo contains the marker.

    `Mole.build()` calls `dump_input()`, which writes the deck's own source to
    `mol.stdout` -- and when the deck writes no separate `.log`, `mol.stdout`
    IS the stdout the wrapper captures.  So the line

        print(f'Total wall time: {t1 - t0:.1f} s')

    appears while the molecule is still being built, in the first seconds of
    the run.  Measured on `projects/BDT/spectrum/BDT-only`: the echo is line
    1139 of 11606; the real end line is 11605.  A substring test answered
    `ended` for a job 17% of the way through a 62-minute run -- and since
    `ended` outranks the traceback branch, a CRASHED run answered `ended` too.

    The anchor separates them exactly: every real end line is printed at
    column 0, every echoed one begins `print(`.

    MUTATION THIS MUST FAIL AGAINST: `any(m in line ...)` instead of
    `line.startswith(m)`.
    """
    for marker in (PYSCF_END_MARKER, PYSCF_SPECTRUM_END_MARKER):
        echoed = ("#INFO: **** input file is /x/job.py ****\n"
                  "#INFO: ******************** input file end ********************\n"
                  f"print(f'{marker} {{t1 - t0:.1f}} s')\n"
                  "converged SCF energy = -1028.27\n")
        assert _pyscf(tmp_path, echoed).run_state == "running", (
            f"the deck's own source line for {marker!r} was read as the run's "
            f"end -- a running job reports finished from its first seconds")
        # ...and the real line, printed at column 0, still ends it.
        assert _pyscf(tmp_path, echoed + f"{marker} 3746.9 s\n"
                      ).run_state == "ended"


def test_a_crash_after_the_echo_is_still_a_crash(tmp_path):
    """The other half of the same defect: `ended` outranks the traceback, so
    an echo that set it made a CRASHED run report finished."""
    text = (f"print(f\"\\n{PYSCF_END_MARKER} {{time.time() - t0:.1f}} s\")\n"
            "Traceback (most recent call last):\n"
            '  File "job.py", line 88, in <module>\n'
            "MemoryError\n")
    assert _pyscf(tmp_path, text).run_state == "stopped"
