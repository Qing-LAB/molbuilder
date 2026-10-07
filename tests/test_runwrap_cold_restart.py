"""Regression test for the 2026-06-14 ``--cold`` / ``--from-scratch``
flag on the SIESTA + PySCF run wrappers.

User-visible contract (job-contracts.md § 4.1 -- a NAME SWEEP, U17):

  * ``bash <name>.run.sh --cold`` NAMES everything matching the run's
    id -- minus what molbuilder itself wrote (deck, template, .psml,
    wrappers, molbuilder's logs) -- and **refuses**, changing nothing;
    ``--force`` says yes to that refusal, and the run overwrites them as
    it goes.
    SIESTA's ``DM.UseSaveDM`` / ``MD.UseSaveCG`` / ``MD.UseSaveXV``
    find nothing surviving, so the calc starts strictly from the .fdf
    coords + conditions.
  * **Nothing is moved or copied** *(user, 2026-08-18)*.  Keeping a state
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


# The engine/conda stubs this suite needs are `conftest.py`'s
# `product_toolchain_is_the_suites_own` -- ONE home.


def test_the_exception_is_anchored_on_the_id_not_widened_to_a_star():
    """§ 4.1's exception must name OUR files, not every file of that shape.

    ``--cold``'s "except what molbuilder wrote" list is derived from the one
    enumeration, ``identity.OUR_FILE_PATTERNS`` (E-1, 2026-08-13).  How it is
    READ is the thing this pins: each pattern's ``{label}`` becomes the run's
    actual id, never ``*``.

    **The widening was defended as harmless and was not.**  It read
    ``{label}`` -> ``*`` until 2026-08-17, on the argument that the sweep's own
    globs already anchor on the id — which says the widening cannot make the
    sweep visit MORE files, and says nothing about the exception matching more
    of them.  It held only while every pattern ended in a suffix nobody but
    molbuilder writes.  ``{label}.xyz`` joined the list on 2026-08-16 (so
    ``prep`` would stop calling a hand-over's input structure an engine
    leftover) and widened to ``*.xyz``, which claimed PySCF's
    ``<JOB>_optimized.xyz`` — warm state, and the whole reason ``--cold``
    exists.

    So this guards the CLASS rather than that one file: the next shared suffix
    added to ``OUR_FILE_PATTERNS`` for the other reader's sake must not quietly
    re-open it.  One glob is exempt by design and named here — ``*.psml`` is
    element-named, not run-named.
    """
    from molbuilder.runwrap import _cold_restart_block

    block = _cold_restart_block("myjob", engine="pyscf", label="myjob")
    line = [l for l in block.splitlines()
            if l.strip().endswith(") continue ;;")]
    assert len(line) == 1, "the exception case arm moved or multiplied"
    pats = line[0].strip()[:-len(") continue ;;")].split("|")

    bare = sorted(p for p in pats if p.startswith("*"))
    assert bare == ["*.psml"], (
        f"an exception is anchored on nothing but a suffix: {bare}.\n"
        f"A pattern that starts with `*` protects every file of that shape "
        f"from --cold, including the engine output the sweep exists to move. "
        f"Anchor it on the run's id -- `\"$_warm_label\"` and the basename.")

    # ...and both spellings are present, because the sweep visits both.
    assert any("_warm_label" in p for p in pats)
    assert any(p.startswith("myjob") for p in pats)
