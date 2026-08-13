"""Editing an id after something has run — the one case that is not a rename.

**Module:** L2, inside the ``validation`` package. Imports ``issues`` (L1) and
``runwrap`` (L2) for the warm-file inventory. To be called by the web Build
route (P10) and by the shared Task Setup tab (P11), which are the two surfaces
where a person can retype an id.

**Contract:** [`execution/run-identity.md`](?doc=execution/run-identity.md)
§ 1 (*"a label edited between runs — the warm files no longer match, and a run
that should have resumed starts cold instead"*) · § 3 rule 1 (normalisation
happens once and the result is stored) · § 5 (what is **reported** rather than
prevented, and its two authors) · [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
§ 4.2 (the per-engine warm-file inventory).

**Why a warning and not a refusal.** `job-system.md § 2` decision 5 is the
shipped doctrine — *molbuilder informs, and the user decides to continue* — and
§ 5 of this contract is an entire section about preferring a message to a wider
pin. Renaming a calculation after a stage has run is a legitimate thing to
want; what is never legitimate is doing it *without being told* that the
geometry the last stage produced will not be found. So this says exactly which
files stop matching, and gets out of the way.

**No inventory of engine outputs, and that is the design** (decided
2026-08-08 — `job-contracts.md § 4.2`). SIESTA's output set depends on its
version and on which options are on, so a list is a snapshot of one build
pretending to be a rule, and its failure is silent: a file nobody listed is a
file nobody sees. :func:`warm_files_present` therefore asks the question by
**subtraction** — anything named after the label that molbuilder did not
write came from the run. That inversion works because the two halves are not
symmetric: we cannot know what an engine produces, and we always know what we
produced (``identity.OUR_FILE_PATTERNS``).

**One function here still uses a list, on purpose.** :func:`_foreign_state`
asks a different question — *are there restart files belonging to some other
calculation* — and answering it needs some notion of what engine state looks
like, since a stranger's file is not named after our id. An incomplete list
there under-reports a **warning**, which is mild; an incomplete list in
`warm_files_present` would have wrongly said *nothing has run*, which is the
dangerous direction — the same failure a composite id passed to it would
produce, and the reason that parameter is named ``label``. Same shape of imperfection, opposite consequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..identity import is_ours
from ..issues import Issue
from ..warmfiles import inventory as _warm_inventory


#: engine -> the shipped warm-restart SUFFIXES an id keys.
#:
#: Still ``runwrap``'s own tuples, not a copy: a new warm hook lands in one
#: list there (with its ``--cold`` glob entry, per § 4.2's pinned lesson) and
#: this sees it without being edited.
#:
#: **It asked for filenames until 2026-08-10** — ``_SIESTA_WARM_SUFFIX_FILES``
#: and ``_PYSCF_WARM_FILES``, two helpers that interpolated a basename into the
#: tuples — and :func:`_foreign_state` then stripped the id back off to recover
#: the suffixes it actually wanted. Building a name in order to undo the
#: construction is a re-derivation with extra steps; those helpers existed for
#: the shell attempt-directory block, which P7 unit 1 retired, and this reads
#: the tuples directly now.
def _engine_inventory(engine: str) -> Optional[Sequence[str]]:
    """The engine's warm inventory, DERIVED from the one rules file (U3,
    2026-08-13) — validation stops reaching into runwrap for a vocabulary
    neither owns — and resolved AT USE, never at import (C-b, same day):
    the module-level dict that stood here loaded both engines' TOMLs the
    moment anything imported ``validation``, so one malformed file broke
    every entry point.  ``None`` for an engine with no rules file — the
    caller already treats unknown engines as *not refused twice*."""
    try:
        return _warm_inventory(engine)
    except Exception:
        return None


def warm_files_present(directory, label: str, engine: str = "") -> List[str]:
    """What the **engine** left in *directory* under *label*, sorted.

    *"Has anything run here?"* — answered **by name, not by enumeration**
    (`job-contracts.md § 4.2`, decided 2026-08-08). Anything named after the
    label that molbuilder did not itself write came from the run, whatever
    version of the engine produced it and whichever options were on.

    **The label, and never the run id** (`run-identity.md § 2.0a`, decision 26).
    Every name this globs is a filename, and a filename's stem is the label; the
    id carries the structure's formula as well and matches nothing on disk.
    Handing this an id would return ``[]`` from a directory full of warm files —
    and ``[]`` here means *nothing has run*, which is the one answer that makes
    the caller destroy a geometry without asking.

    > **This replaced a per-engine list, and the reasoning behind that list was
    > wrong in a way worth keeping visible.** It named thirteen SIESTA suffixes
    > and I chose it over a shorter one *deliberately*, on the grounds that a
    > directory holding only a `.TSHS` has state the id keys. That was true. The
    > premise underneath it was not: **completeness was never purchasable**,
    > because the file set depends on the engine's version and options. A list
    > can only ever be a snapshot of one build's behaviour, and the failure is
    > silent — a file nobody listed is a file nobody sees.

    ``engine`` is accepted and ignored, kept only so callers need not change;
    the question stopped being per-engine when it stopped being a list. It is
    scheduled for removal with the last caller that passes it.

    Symlinks are followed and a dangling one is not counted: a carried restart
    file that does not resolve is not state to continue from, and ``runstatus``
    reads them the same way.
    """
    if not label:
        return []
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.glob(f"{label}*")
        if p.is_file() and not is_ours(p.name, label))


def check_id_change(directory, old_label: str, new_label: str,
                    engine: str) -> List[Issue]:
    """§ 1. Warn when retyping a calculation's name stops being a rename.

    Takes **labels**, not run ids, for :func:`warm_files_present`'s reason: the
    files are keyed by the label. A user edits ``run.name``; the label and the
    id both follow, and it is the label that is on disk (§ 2.0a).

    Before anything has run there is no state to orphan, so a name may be
    edited freely and this returns nothing — that is the *"editable once"* half
    of the rule, and it is the common case: a person naming a calculation and
    then thinking better of it.

    Once files keyed by ``old_label`` exist, the same edit means something else
    entirely. The engine finds nothing under the new name and starts cold,
    which is the quiet failure § 1 opens with: **the run does not fail, it
    silently starts over**, and the only evidence is a wall-clock cost the user
    notices days later.

    The finding names the files by name rather than counting them. *"3 warm
    files"* tells a user nothing; ``BDT_Au.XV`` tells them the relaxed geometry
    is what they are about to walk away from.
    """
    if not old_label or not new_label or old_label == new_label:
        return []
    orphaned = warm_files_present(directory, old_label, engine)
    if not orphaned:
        return []
    return [Issue(
        "warn",
        f"changing the name from {old_label!r} to {new_label!r} is not a "
        f"rename. {len(orphaned)} file(s) here are named by the old one and "
        f"would stop matching: {', '.join(orphaned)}. The engine keys its "
        f"restart state on this literal, so the next run finds nothing under "
        f"{new_label!r} and "
        f"starts cold rather than failing -- the work is not lost on disk, but "
        f"it is no longer continued from. Rename the files too, or keep the "
        f"name and change the folder instead",
        where="run.id")]   # the anchor stays: a finding's `where` IS its
                           # stable id (validation-findings), so it is not
                           # retitled on a vocabulary change


def check_prior_state(directory, run_id: str, engine: str,
                      *, cell=None) -> List[Issue]:
    """§ 5 — the cases answered with a message rather than a wider pin.

    § 5's table has four rows and a *"who says it"* column; this is **the
    surface's half**, called at check time, when the choice is still open.
    Three rows name the surface:

    - *prior state that matches* — nothing is wrong, and the user should still
      know before they start, so ``info``;
    - *prior state from another calculation* — the engine will not load it, but
      a one-job directory holding a second job's restart files is
      `job-contracts.md § 2.1` Rule 1 being broken, so ``warn``;
    - *changed cell parameters* — the deck's cell was rebuilt and the run will
      ignore it, so ``warn``: an intent silently discarded is worse than one
      refused.

    The fourth row, *the structure moved under a saved description*, is
    **deliberately not here**. Its author in that column is *the reader, at
    preflight* — it is answered from the description's own witness
    (`stages.md § 6.3`), not from what is lying in a directory, and putting it
    here would give one rule two homes.

    **This never contradicts the wrapper banner, and cannot weaken it.** § 5's
    two authors say the same things at different times: the banner at run time,
    after the user committed, and this at check time, before. The banner is
    always present, so it is the one that must never be weakened — this only
    ever *adds* a message earlier, and the one thing the banner cannot say
    (that state came from a different calculation, because nothing beside it
    records which run made it) is exactly the row this can.
    """
    out: List[Issue] = []
    mine = warm_files_present(directory, run_id, engine)
    if mine:
        out.append(Issue(
            "info",
            f"prior state found for this key: {', '.join(mine)}. A stage set "
            f"to continue will resume from it; one set to clean will not look "
            f"at it, and will leave it where it is",
            where="identity.prior_state"))
        out.extend(_cell_row(directory, run_id, engine, cell))

    foreign = _foreign_state(directory, run_id, engine)
    if foreign:
        out.append(Issue(
            "warn",
            f"this directory also holds restart state from a different "
            f"calculation: {', '.join(foreign)}. {engine} keys its warm files "
            f"on the run id, so it will not read them -- but one directory is "
            f"meant to hold one job (job-contracts.md § 2.1 Rule 1), and a "
            f"second job's files here is usually a sign something was written "
            f"in the wrong place",
            where="identity.foreign_state"))
    return out


def _cell_row(directory, run_id: str, engine: str, cell) -> List[Issue]:
    """§ 5's first row. **A changed cell is a no-op, not a mismatch.**

    A ``.XV`` carries its own cell *and its own frame*, and on a continue the
    saved one **wins**. So widening the vacuum changes the deck and changes
    nothing about the run — which is precisely why the cell cannot be in the id
    (it is derived, § 2) and why this is reported instead.
    """
    if cell is None or engine != "siesta":
        return []
    xv = Path(directory) / f"{run_id}.XV"
    if not xv.is_file():
        return []
    try:
        import numpy as np

        from ..parse.coords.siesta_xv import _read_xv_cell
        saved = _read_xv_cell(xv)
        if saved is None:
            return []                       # unreadable is bundle.py's to say
        if np.allclose(np.asarray(cell, dtype=float), saved,
                       rtol=1e-6, atol=1e-6):
            return []
    except Exception:
        # A cell we cannot read is not a finding.  This row exists to explain
        # a silent no-op; inventing a second failure out of a parse problem
        # would be worse than staying quiet, and the parsers already report
        # unreadable files where that is their job.
        return []
    return [Issue(
        "warn",
        f"{xv.name} was written under different cell parameters. A .XV "
        f"carries its own cell, and on a continue the saved one wins -- so "
        f"the cell in the new deck will be ignored and this run will use the "
        f"saved one. To actually change the cell, start this stage clean",
        where="identity.saved_cell")]


def _foreign_state(directory, run_id: str, engine: str) -> List[str]:
    """Restart files in *directory* keyed by some id other than *run_id*.

    The suffixes come from the shipped inventory rather than being listed again
    — which keeps this correct for PySCF, where a "suffix" is
    ``_optimized.xyz`` rather than an extension.
    """
    suffixes = _engine_inventory(engine)
    if not suffixes or not run_id:
        return []
    d = Path(directory)
    if not d.is_dir():
        return []
    found = set()
    for suffix in suffixes:
        for p in d.glob(f"*{suffix}"):
            stem = p.name[:-len(suffix)] if suffix else p.name
            if stem and stem != run_id and p.is_file():
                found.add(p.name)
    return sorted(found)


# ``check_overwrite`` stood here until U14 (2026-08-12) -- RETIRED, not
# rehomed.  It implemented "§ 6 -- refuse unless the user says overwrite",
# a rule run-identity.md softened away on 2026-08-08 ("Warn, do not
# refuse"), which is why it had zero callers: the design had moved and the
# function pinned the old one.  What § 6 still requires -- before writing,
# SAY what is in the folder -- is served by :func:`warm_files_present`
# (the evidence) and asked at the surface (`jobset/_cli._ask_if_underway`),
# where a question belongs.




__all__ = ["check_id_change", "check_prior_state", "warm_files_present"]
