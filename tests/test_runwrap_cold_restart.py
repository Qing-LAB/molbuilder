"""Regression test for the 2026-06-14 ``--cold`` / ``--from-scratch``
flag on the SIESTA + PySCF run wrappers.

User-visible contract (job-contracts.md § 4.1 -- a NAME SWEEP, U17):

  * ``bash <name>.run.sh --cold`` NAMES everything matching the run's
    id -- minus what molbuilder itself wrote (deck, template, .psml,
    wrappers, molbuilder's logs) -- and **refuses**, changing nothing;
    ``--force`` says yes to that refusal and removes them, so
    SIESTA's ``DM.UseSaveDM`` / ``MD.UseSaveCG`` / ``MD.UseSaveXV``
    find nothing surviving, and the calc starts strictly from the .fdf
    coords + conditions.
  * **Nothing is kept aside** *(user, 2026-08-18)*.  Keeping a state
    is ``molbuilder checkpoint save`` and it is never automatic
    (`checkpointing.md` § 2).
  * Combinable with ``--continue`` (cold = no-op when there is nothing
    to name, which is the typical case mid-run).
  * Idempotent: re-running with ``--cold`` on a directory that is
    already clean says so and proceeds.

Motivation (2026-06-14 BDT incident): stage 2 ran without the
frozen-atom constraints the user intended (separate UI bug).  The
resulting .DM/.XV/.CG were physically inconsistent with what the
user wanted; any subsequent run that warm-started from them would
inherit the contamination.  ``--cold`` lets the user re-run from a
known clean state without having to manually ``rm`` the files.
"""
from __future__ import annotations

import pytest


# The engine/conda stubs this suite needs are `conftest.py`'s
# `product_toolchain_is_the_suites_own` -- ONE home.


def test_the_exception_is_anchored_on_the_label_not_widened_to_a_star():
    """§ 4.1's exception must name OUR files, not every file of that shape.

    ``--cold``'s "except what molbuilder wrote" list is derived from the one
    enumeration, ``identity.OUR_FILE_PATTERNS`` (E-1, 2026-08-13).  How it is
    READ is the thing this pins: each pattern's ``{label}`` becomes the
    run's label, never ``*``.

    **The widening was defended as harmless and was not.**  It read
    ``{label}`` -> ``*`` until 2026-08-17, on the argument that the sweep's own
    globs already anchor on the label -- which says the widening cannot make
    the sweep visit MORE files, and says nothing about the exception matching
    more of them.  It held only while every pattern ended in a suffix nobody
    but molbuilder writes.  ``{label}.xyz`` joined the list on 2026-08-16 and
    widened to ``*.xyz``, which claimed PySCF's ``<JOB>_optimized.xyz`` --
    warm state, and the whole reason ``--cold`` exists.

    So this guards the CLASS rather than that one file: the next shared suffix
    added to ``OUR_FILE_PATTERNS`` for another reader's sake must not quietly
    re-open it -- no engine file of today's shows it, so the rendered arm is
    read.  One glob is exempt by design and named here -- ``*.psml`` is
    element-named, not run-named.
    """
    from molbuilder.runwrap import _cold_restart_block

    block = _cold_restart_block("myjob")
    # THE EXCEPTION ARM: the one holding molbuilder's own names, `*.psml`
    # among them.
    line = [l for l in block.splitlines()
            if l.strip().endswith(") continue ;;") and "*.psml" in l]
    assert len(line) == 1, "the exception case arm moved or multiplied"
    pats = line[0].strip()[:-len(") continue ;;")].split("|")
    loose = sorted(p for p in pats
                   if p != "*.psml" and not p.startswith('"$_warm_label"'))
    assert loose == [], (
        f"an exception is not anchored on the label: {loose}.\n"
        f"A pattern that does not start with the label protects every file "
        f"of that shape from --cold, including the engine output the sweep "
        f"exists to remove.")


#: `job-contracts.md` § 4.1, `project-layout.md` § 1.6.7 -- the run script's
#: own ``--cold`` in a run's folder (``--force`` saying yes), each row the
#: label, what the folder holds that the sweep removes, and what it keeps.
#: The sweep visits the label's names (``<label>.*``, ``<label>_*``) and
#: removes all of them but the files molbuilder wrote -- a try's numbered
#: files among them -- whatever the engine.
_SWEEPS = [
    pytest.param(
        "H2",
        ["H2_optimized.xyz", "H2.chk", "H2.XV", "H2.never-listed"],
        ["H2_01_coarse.py", "H2_01_coarse-run0.out",
         "H2_01_coarse-run0.monitor.log", "H2_01_coarse-run17.monitor.log",
         "H2_01_coarse-run0.util.csv"],
        id="the label's warm state removed, a file nobody listed among it; "
           "molbuilder's deck and every try's files kept"),
    pytest.param(
        "Au-run2",
        ["Au-run2.DM", "Au-run2.XV"],
        ["Au-run2_01_coarse.fdf", "Au-run2_01_coarse-run0.out"],
        id="a label ending in -run<N> has its warm files removed, never "
           "taken for a try's"),
    pytest.param(
        "H",
        ["H.DM"],
        ["H.psml", "H_01_coarse.fdf"],
        id="a label that is an element's symbol keeps its pseudopotential"),
]


@pytest.mark.parametrize("label,removed,kept", _SWEEPS)
def test_a_cold_start_sweeps_the_warm_state_and_keeps_every_try(
        tmp_path, label, removed, kept):
    """The rendered sweep, run as bash on the files a run's folder holds,
    since which names a ``case`` arm takes is the shell's answer, not the
    text's.

    API-LEVEL: the generated sweep, run on files a run's folder holds.

    MUTATIONS THIS MUST FAIL AGAINST: the exception arm (molbuilder's own
    names, `identity.OUR_FILE_PATTERNS`) gone -- the deck removed; a
    `-run*` row dropped from that enumeration -- a try's monitor file
    removed; an arm keeping every name holding ``-run<N>`` -- ``Au-run2``'s
    warm state kept; ``*.psml`` dropped from the arm."""
    import subprocess
    from molbuilder.runwrap import _cold_restart_block
    block = _cold_restart_block(label)
    for name in removed + kept:
        (tmp_path / name).write_text("x")
    r = subprocess.run(["bash", "-c", "_cold=1\n_force=1\n" + block],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == sorted(kept), (left, r.stderr)
