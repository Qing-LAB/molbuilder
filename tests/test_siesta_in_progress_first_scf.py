"""L1 tests for the SIESTA parser's synthetic in-progress Frame.

A long-running geometry-opt + SCF on a heavy system (~200 atoms,
e.g. an Au-thiol-Au junction) can spend 5-30 minutes inside its
FIRST SCF cycle before the first ``outcoor`` block lands.  During
that window the parser was buffering SCF cycle data internally but
not exposing any Frame to the Results-tab inspector -- so the
inspector appeared stalled with "no data yet" and the user had no
way to tell whether the run was progressing OR silently diverging
until SIESTA finally errored out.

The parser now emits a synthetic Frame at EOF whenever:
  * the run is still ``ongoing`` (no End-of-run / fatal marker), AND
  * ``current_scf`` is non-empty (at least one SCF cycle parsed)

This Frame is flagged ``in_progress=True`` so consumers (the
Results-tab inspector + the JS trajectory renderer) can hide the
trajectory animation controls -- the placeholder geometry is NOT
meaningful -- and render only the SCF convergence chart from the
attached ``scf_history``.
"""
from __future__ import annotations

import os
import tempfile
from textwrap import dedent

from molbuilder.parsers.siesta import SiestaParser


def _parse(out_body: str):
    with tempfile.NamedTemporaryFile(
            "w", suffix=".out", delete=False, encoding="utf-8") as tf:
        tf.write(out_body)
        path = tf.name
    try:
        return SiestaParser.parse(path)
    finally:
        os.unlink(path)


# Minimal preamble + an SCF header + 3 SCF rows, NO outcoor.
# Models a run that's still inside its first SCF cycle.
_FRESH_FIRST_SCF = dedent("""\
    Running on 4 procs

       Compiler version: GNU-14.3.0
       Compiler flags  : -O2
       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       scf:    2   -806618.752906  -805857.884182  -805858.389926  0.445086  0.781420 39.179556
       scf:    3   -806787.607589  -806373.289526  -806373.632436  0.395470  0.199362 25.859179
    """)


# Same SCF preamble + outcoor + End-of-run marker.  Models a CLEAN
# completed run: no in-progress frame should be emitted regardless
# of ``current_scf`` state at EOF.
_FRESH_COMPLETED = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       siesta: E_KS(eV) =        -804434.909422

    outcoor: Relaxed atomic coordinates (Ang):
        0.000000  0.000000  0.000000  1  1  Au

    >> End of run
    """)


# Preamble + zero SCF cycles + zero outcoor.  Pathological / very
# early run state; the parser must NOT emit a phantom in-progress
# frame just because the run is "ongoing".
_EMPTY_PREAMBLE_ONLY = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

    reinit: Reading from file foo.fdf
    """)


def test_fresh_first_scf_emits_in_progress_frame():
    """The motivating case: heavy run, deep into its first SCF cycle,
    no outcoor block yet.  The parser must emit a single synthetic
    Frame so the Results inspector has something to render."""
    traj = _parse(_FRESH_FIRST_SCF)
    assert len(traj.frames) == 1, (
        "expected exactly one synthetic in-progress frame; got "
        f"{len(traj.frames)}"
    )
    f = traj.frames[0]
    assert f.in_progress is True
    assert f.scf_history is not None
    assert len(f.scf_history) == 3
    # Most-recent finite energy should be picked up as the frame energy
    # so the energy plot has a data point.
    assert f.energy is not None
    assert f.energy < 0  # SCF energies are negative for stable systems
    # The run state stays "ongoing" -- the synthetic frame must NOT
    # mark the run as finished.
    assert traj.run_state == "ongoing"


def test_clean_completed_run_does_not_emit_in_progress_frame():
    """A run that reached its End-of-run marker MUST NOT have an
    in-progress frame tacked on -- that would mislead the user into
    thinking the run was still active."""
    traj = _parse(_FRESH_COMPLETED)
    assert len(traj.frames) == 1
    assert traj.frames[0].in_progress is False
    assert traj.run_state == "finished"


def test_empty_preamble_does_not_emit_in_progress_frame():
    """Defensive: when no SCF cycle has been parsed at all (very
    early in the run, before scf:1 lands), the parser MUST NOT
    invent a frame from nothing -- there's no useful chart to
    render and a phantom frame would mislead the inspector."""
    traj = _parse(_EMPTY_PREAMBLE_ONLY)
    assert len(traj.frames) == 0
