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

**Why the inventory is imported rather than listed.** Four inventories already
exist in this tree and they do not agree: ``runwrap``'s thirteen SIESTA
suffixes (the ``--cold`` move-aside glob, which `job-contracts.md § 4.2` calls
the authority), two inline copies inside the bash ``runwrap`` emits, and
``jobset/runstatus._WARM_FILES``, which lists three. A fifth copy here would be
the worst of all possible additions. ``runstatus``' shorter list is not
necessarily wrong — it answers *"can this stage resume"*, a narrower question
than *"what does this id key"* — but its citation points at
``script-execution.md``, which no longer exists, so which question it means is
no longer written down anywhere. Recorded, not fixed here: reconciling them is
its own change with its own review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

from ..issues import Issue
from ..runwrap import _PYSCF_WARM_FILES, _SIESTA_WARM_SUFFIX_FILES


#: engine -> the shipped function that names the files an id keys.
#:
#: The two callables are ``runwrap``'s own, not a copy of its suffix tuple:
#: a new warm-restart hook lands in one list there (with its ``--cold`` glob
#: entry, per § 4.2's pinned lesson) and this sees it without being edited.
_INVENTORY: Dict[str, Callable[[str], Sequence[str]]] = {
    "siesta": _SIESTA_WARM_SUFFIX_FILES,
    "pyscf":  _PYSCF_WARM_FILES,
}


def warm_files_present(directory, run_id: str, engine: str) -> List[str]:
    """The files in *directory* that are keyed by *run_id*, sorted.

    This is the *"has anything run"* question, asked the way the contract
    frames it: **not** whether an output exists, but whether state exists that
    the id names. § 1's failure mode is precisely that — an edited label leaves
    those files behind, matching nothing.

    Symlinks are followed and a dangling one is not counted: a carried restart
    file that no longer resolves is not state this calculation can continue
    from, and ``runstatus`` already reads them the same way.

    An unknown engine yields ``[]`` rather than raising. Whether the engine is
    one this backend supports is `stages.md § 6.6`'s first check and belongs to
    the preflight; answering it a second time here would put the same refusal
    in two places and let them disagree.
    """
    names = _INVENTORY.get(engine)
    if not names or not run_id:
        return []
    d = Path(directory)
    return sorted(n for n in names(run_id) if (d / n).is_file())


def check_id_change(directory, old_id: str, new_id: str,
                    engine: str) -> List[Issue]:
    """§ 1. Warn when retyping an id stops being a rename.

    Before anything has run there is no state to orphan, so an id may be
    edited freely and this returns nothing — that is the *"editable once"* half
    of the rule, and it is the common case: a person naming a calculation and
    then thinking better of it.

    Once files keyed by ``old_id`` exist, the same edit means something else
    entirely. The engine finds nothing under the new name and starts cold,
    which is the quiet failure § 1 opens with: **the run does not fail, it
    silently starts over**, and the only evidence is a wall-clock cost the user
    notices days later.

    The finding names the files by name rather than counting them. *"3 warm
    files"* tells a user nothing; ``BDT_Au.XV`` tells them the relaxed geometry
    is what they are about to walk away from.
    """
    if not old_id or not new_id or old_id == new_id:
        return []
    orphaned = warm_files_present(directory, old_id, engine)
    if not orphaned:
        return []
    return [Issue(
        "warn",
        f"changing the id from {old_id!r} to {new_id!r} is not a rename. "
        f"{len(orphaned)} file(s) here are named by the old id and would stop "
        f"matching: {', '.join(orphaned)}. The engine keys its restart state "
        f"on this literal, so the next run finds nothing under {new_id!r} and "
        f"starts cold rather than failing -- the work is not lost on disk, but "
        f"it is no longer continued from. Rename the files too, or keep the id "
        f"and change the folder instead",
        where="run.id")]


__all__ = ["check_id_change", "warm_files_present"]
