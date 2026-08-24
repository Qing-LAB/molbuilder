"""L1 tests for the SIESTA parser's IterSCF timer attribution.

SIESTA emits one line right after each completed SCF cycle::

    timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49

The parser attaches the CUMULATIVE ``Calls`` and ``Time`` (seconds)
to the SCF cycle dict that was just appended.  The Results-tab
inspector then computes per-iteration deltas (Time_N - Time_N-1)
client-side and shows a per-iter wall-time annotation on the SCF
chart -- the canonical "is this run progressing at a reasonable
pace?" signal.

Field-by-field defensive parsing: a Fortran column-width overflow
(``******``) in any one field MUST NOT drop the whole attribution.
Each field parses independently; only the ones that decode cleanly
land in the cycle dict.  The JS falls back to the cycle index when
``elapsed_s`` is missing.
"""
from __future__ import annotations

import os
import tempfile
from textwrap import dedent

import pytest

from molbuilder.parse.engines.siesta import SiestaParser


def _parse(out_body: str):
    with tempfile.NamedTemporaryFile(
            "w", suffix=".out", delete=False, encoding="utf-8") as tf:
        tf.write(out_body)
        path = tf.name
    try:
        return SiestaParser.parse(path)
    finally:
        os.unlink(path)


# Two SCF iterations, two clean timer lines after them.
_TWO_ITERS_CLEAN = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49
       scf:    2   -806618.752906  -805857.884182  -805858.389926  0.445086  0.781420 39.179556
       timer: Routine,Calls,Time,% = IterSCF        2      75.400  52.10
    """)


# Iter 1 clean, iter 2 with asterisks in the Time field (Fortran
# fixed-width overflow on a >27-hour cumulative run).
_OVERFLOWED_TIME = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49
       scf:    2   -806618.752906  -805857.884182  -805858.389926  0.445086  0.781420 39.179556
       timer: Routine,Calls,Time,% = IterSCF        2     ******  52.10
    """)


# Asterisks in % only -- still must attach Calls and Time.
_OVERFLOWED_PERCENT_ONLY = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       timer: Routine,Calls,Time,% = IterSCF        1      40.820  *****
    """)


# Timer line arrives WITHOUT a preceding scf-data line (parser was
# torn or the .out is malformed).  Must NOT attach to a phantom cycle.
_TIMER_WITHOUT_SCF = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

       timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49
    """)


def test_two_clean_iters_attribute_cumulative_calls_and_time():
    """Both cycles get both keys; per-iter delta = Time_N - Time_N-1
    is computable client-side."""
    traj = _parse(_TWO_ITERS_CLEAN)
    # The in-progress frame holds the SCF history (no outcoor in the
    # fixture; see test_siesta_in_progress_first_scf for that contract).
    assert len(traj.frames) == 1
    history = traj.frames[0].scf_history
    assert history is not None and len(history) == 2
    a, b = history
    assert a["cumulative_calls"]       == 1
    assert a["elapsed_s"]  == 40.820
    assert b["cumulative_calls"]       == 2
    assert b["elapsed_s"]  == 75.400
    # Spot-check the per-iter delta the JS will compute:
    assert (b["elapsed_s"]
            - a["elapsed_s"]) == pytest.approx(34.580)


def test_overflowed_time_drops_only_walltime_keeps_calls():
    """``******`` in Time MUST NOT drop the entire attribution -- the
    Calls field is still useful (the JS can show "iter N took ?s,
    cumulative ?s, instance count N")."""
    traj = _parse(_OVERFLOWED_TIME)
    history = traj.frames[0].scf_history
    assert history is not None and len(history) == 2
    iter2 = history[1]
    assert iter2["cumulative_calls"] == 2, (
        "Calls field was clean -- must still attach"
    )
    assert "elapsed_s" not in iter2, (
        "Time field was asterisks -- must NOT attach a NaN or "
        "garbage value; JS will fall back to cycle index"
    )


def test_overflowed_percent_only_keeps_time_and_calls():
    """Percent isn't load-bearing; an overflow there must not affect
    the Time/Calls attribution -- we don't even consume %."""
    traj = _parse(_OVERFLOWED_PERCENT_ONLY)
    history = traj.frames[0].scf_history
    assert history is not None and len(history) == 1
    iter1 = history[0]
    assert iter1["cumulative_calls"]      == 1
    assert iter1["elapsed_s"] == 40.820


def test_timer_without_scf_silently_ignored():
    """Defensive: a timer line that arrives BEFORE any SCF cycle (or
    AFTER an SCF cycle has been flushed and current_scf is empty)
    must NOT crash or attach to a phantom dict.  The .out under
    parse here has no SCF cycles at all -- the parser must return
    no frames (the run_state is still ``ongoing`` per the in-progress
    contract, but there's no SCF cycle history to render)."""
    traj = _parse(_TIMER_WITHOUT_SCF)
    # No scf cycle parsed -> no in-progress frame to attach the timer
    # to.  Empty frames; no exception.
    assert len(traj.frames) == 0


def test_frame_elapsed_s_carries_last_cycle_elapsed_s():
    """Symmetric per-iteration timing (2026-06-20).  ``Frame.elapsed_s``
    is populated from the LAST SCF cycle's ``elapsed_s`` so per-CG-step
    time = ``frames[i+1].elapsed_s - frames[i].elapsed_s`` without any
    client-side stitching.

    This pins the same code path for both the committed-frame branch
    (outcoor-bounded) and the in-progress-frame branch at EOF.  The
    fixture below has no outcoor block -> in-progress frame, which is
    the easier case to construct and runs through the SAME time
    surfacing logic.
    """
    traj = _parse(_TWO_ITERS_CLEAN)
    assert len(traj.frames) == 1
    f = traj.frames[0]
    # In-progress frame -- still gets its elapsed from the last cycle.
    assert f.in_progress is True
    # Last cycle's cumulative time was 75.400 s -- that is how far into
    # the run SIESTA was when it finished its 2nd SCF cycle.
    assert f.elapsed_s == pytest.approx(75.400)
    # And it is NOT offered as a time of day.  A SIESTA .out carries no
    # epoch anywhere, so the honest answer is None (parse.md 2a, P-T2):
    # this is the assertion that stops 75.4 s being rendered as a date.
    assert f.wall_clock_s is None


# Same two timed iterations, but bounded by an outcoor block so the
# frame is COMMITTED rather than in-progress.  SIESTA's outcoor rows are
# ``X Y Z species_index atom_index element`` (6 tokens); thinner rows
# are ignored and no frame commits.
_TWO_ITERS_COMMITTED = dedent("""\
    Running on 4 procs

       Parallelisations: MPI, OpenMP

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
       scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
       timer: Routine,Calls,Time,% = IterSCF        1      40.820  49.49
       scf:    2   -806618.752906  -805857.884182  -805858.389926  0.445086  0.781420 39.179556
       timer: Routine,Calls,Time,% = IterSCF        2      75.400  52.10

    outcoor: Atomic coordinates (Ang):
        0.00000000  0.00000000  0.00000000  1  1  H
        1.00000000  0.00000000  0.00000000  1  2  H

    """)


def test_committed_frame_reports_elapsed_and_no_wall_clock():
    """The COMMITTED-frame branch, pinned separately from the
    in-progress one — they are two code paths that each construct a
    Frame, and a mutation-test showed the committed one was reachable
    with no assertion on its clocks at all.  It is also the path that
    produces the run-state badge for a real bench run.

    ``elapsed_s`` carries the last cycle's cumulative timer, and
    ``wall_clock_s`` is None because a SIESTA .out states no time of
    day anywhere (docs/model/parse.md § 2a, P-T2).  When this frame
    handed its 75.4 elapsed seconds over as a wall clock, the browser
    formatted it as a date and the badge read "Dec 31, 5:06 PM".
    """
    traj = _parse(_TWO_ITERS_COMMITTED)
    assert len(traj.frames) == 1
    f = traj.frames[0]
    # The two branches are told apart by the STRUCTURE, not by
    # in_progress: the committed branch carries the real outcoor
    # geometry, the in-progress branch synthesises a placeholder.  (The
    # flag stays True here only because this fixture never reaches
    # "End of run" -- irrelevant to which branch built the Frame.)
    assert f.structure.elements == ["H", "H"], (
        "fixture must exercise the committed-frame branch; a placeholder "
        "structure means the outcoor block stopped committing a frame")
    assert f.elapsed_s == pytest.approx(75.400)
    assert f.wall_clock_s is None


def test_frame_elapsed_s_is_none_when_no_iter_scf_timer():
    """When SIESTA didn't emit any ``timer: ... IterSCF`` lines (older
    build, or stripped output), the per-cycle dicts carry no
    ``elapsed_s`` and ``Frame.elapsed_s`` falls back to ``None`` rather
    than crashing or picking up some other field."""
    out_body = dedent("""\
        Running on 4 procs

           Parallelisations: MPI, OpenMP

             iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
           scf:    1   -798748.382767  -804434.909422  -804435.406774  2.772203  1.493296 90.109736
           scf:    2   -806618.752906  -805857.884182  -805858.389926  0.445086  0.781420 39.179556
        """)
    traj = _parse(out_body)
    assert len(traj.frames) == 1
    assert traj.frames[0].elapsed_s is None
    assert traj.frames[0].wall_clock_s is None
