"""What a SIESTA run says it ran with -- its build and its solver -- read
from REAL output.

``runtime_info["siesta_build"]`` is the header SIESTA and TBtrans both print
(`parse/engines/siesta_grammar.py` ``read_build_line``);
``runtime_info["siesta_diag"]`` is the solver SIESTA resolved, which it prints
once as ``diag:`` lines (``Src/diag_option.F90`` ``print_diag``,
``read_diag_line``).  Both are pinned here on a frozen real run.

*This file was built on an INVENTED header until 2026-09-26* -- ``Siesta
Version :``, ``ELPA support``, ``* Running on 4 MPI processes``, ``redata:
Diagonalization algorithm``, an ``ELPA: NVIDIA GPU detected`` banner -- none
of which SIESTA 5.4.2 prints; the parser matched the invented solver lines,
so no real run had a solver on record while every test here passed.  The two
refusals below keep a synthetic line each, because no real run can reach
them, and say so.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import dedent

from molbuilder.parse.engines.siesta import SiestaParser

#: A REAL SIESTA 5.4.2 run, frozen (``tests/watch/fixtures/siesta_frozen``).
_REAL = (Path(__file__).parent / "watch" / "fixtures" / "siesta_frozen"
         / "hemeC-stage2-run3-finished-42fr.out")


def _parse(tmp_path, text=None, name="job.out"):
    out = tmp_path / name
    if text is None:
        shutil.copy(_REAL, out)
    else:
        out.write_text(text)
    return SiestaParser.parse(str(out))


def _runtime_warnings(traj):
    return [w for w in traj.parse_warnings if w.category == "runtime_info"]


def test_a_real_run_states_its_build_and_its_solver(tmp_path):
    """The header as SIESTA printed it, and the solver as it resolved it --
    with no warning, because every token is one SIESTA writes."""
    traj = _parse(tmp_path)
    build, diag = (traj.runtime_info["siesta_build"],
                   traj.runtime_info["siesta_diag"])
    assert build["version"].startswith("5.4.2")
    assert build["parallelisations"] == ["MPI"]
    assert diag["algorithm"] == "D&C"
    assert (diag["diag_blocksize"], diag["distribution"]) == (8, "2 x 4")
    assert _runtime_warnings(traj) == []


def _with_line(tmp_path, pattern, replacement):
    """The real run with one of its own lines rewritten -- the shape SIESTA
    prints, a token no SIESTA prints yet."""
    lines = _REAL.read_text(errors="replace").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(pattern))
    lines[i] = replacement
    return _parse(tmp_path, "\n".join(lines) + "\n")


def test_an_unknown_parallelisation_token_is_recorded_and_flagged(tmp_path):
    """SYNTHETIC -- no real run reaches it: a future ``Parallelisations:``
    token is recorded verbatim AND flagged, so a shape the parser does not
    know surfaces rather than passing for ground truth."""
    traj = _with_line(tmp_path, "Parallelisations:",
                      "Parallelisations: HYBRID(MPI:8)")
    assert (traj.runtime_info["siesta_build"]["parallelisations"]
            == ["HYBRID(MPI:8)"])
    warns = _runtime_warnings(traj)
    assert warns and "HYBRID(MPI:8)" in warns[0].error


def test_an_unknown_solver_is_recorded_and_flagged(tmp_path):
    """SYNTHETIC -- no real run reaches it (SIESTA stops on an algorithm it
    does not know): one a newer SIESTA adds is recorded as printed AND
    flagged against ``Src/diag_option.F90``'s vocabulary."""
    traj = _with_line(tmp_path, "diag: Algorithm",
                      "diag: Algorithm                                     "
                      "= quantum-magic")
    assert traj.runtime_info["siesta_diag"]["algorithm"] == "quantum-magic"
    warns = _runtime_warnings(traj)
    assert warns and "QUANTUM-MAGIC" in warns[0].error


def test_no_header_yields_no_build_dict(tmp_path):
    """A truncated .out with no header lines neither crashes the parser nor
    synthesises a build or a solver -- the keys stay absent."""
    traj = _parse(tmp_path, dedent("""\
        siesta: System type = molecule
        """), name="headerless.out")
    assert "siesta_build" not in traj.runtime_info
    assert "siesta_diag" not in traj.runtime_info
