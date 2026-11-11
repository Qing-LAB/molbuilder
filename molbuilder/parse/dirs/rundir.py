"""A run folder's floor-1 questions that need the whole folder: what a viewer
should open in a folder read without its run (:func:`openable_in`), and how a
run is doing, asked with its launch record (:func:`run_state_of`).

Contract: `model/parse.md` § 5.  Floor 1 (`execution/architecture.md` § 2.1):
**this module knows no calculation.**  What a folder IS, the description it
belongs to, the label its files are named on and the run it speaks for are the
run door's (`molbuilder.runs`, floor 2), which asks these with what it read.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple


def claimed(path: str) -> bool:
    """Does a registered parser claim this file? — the REGISTRY's question.

    The door never offers a file `detect()` refuses.  *What is a run's output* and *what can a
    person open* are different questions with different owners
    (`model/parse.md` § 5.5); this is the second one, and the registry owns
    it.
    """
    from ..registry import detect
    from ..errors import ParseError
    try:
        detect(path)
        return True
    except (ParseError, OSError, ValueError, LookupError):
        return False


def _search_roles() -> "List[str]":
    """The dotted roles to try, in order, when nothing says what a folder's
    calculation produces: the catalogue's order of what a viewer opens
    (`runfiles.result_roles`) -- the engine's stdout first, which for SIESTA
    IS a trajectory source and for PySCF is refused by the registry and so
    drops out on its own, then the progress channel, then every
    calculation's product.  Built from the catalogue, not listed here."""
    from molbuilder.runfiles import ON_THE_LABEL, result_roles
    out = list(result_roles(None))
    out += [a.role for a in ON_THE_LABEL if a.calculation and a.role not in out]
    return [r for r in out if r.startswith(".")]


def openable_in(directory: str) -> Tuple[Optional[str], List[str]]:
    """*What should a viewer load here?* — and the trail of what was tried —
    for a folder read WITHOUT its run: by the catalogue's roles alone,
    newest first, each vetted by the registry (`model/parse.md` § 5.2).

    A run of OURS is opened by the run door instead (`runs.folder_answer`):
    its run's own file, read back with its label -- this floor knows no
    label, so it can only
    ask by a dotted role, and among one role's files it can only take the
    newest.

    ``attempts`` is not decoration: it is the BODY of the refusal a person
    reads when nothing matched, and it moves with the search so the message
    cannot drift from it.
    """
    from molbuilder.runfiles import find_by_role

    attempts: List[str] = []

    def offer(paths: List[str], how: str) -> Optional[str]:
        """The newest of *paths* the REGISTRY claims, with the trail written."""
        for cand in sorted(paths, key=lambda p: os.path.getmtime(p)
                           if os.path.isfile(p) else 0.0, reverse=True):
            if claimed(cand):
                attempts.append(f"{how} -> {os.path.basename(cand)}")
                return cand
            attempts.append(f"{how} -> {os.path.basename(cand)}: "
                            f"no parser claims it, not offered")
        return None

    for role in _search_roles():
        hits = [str(p) for p in find_by_role(directory, role)]
        if not hits:
            continue
        attempts.append(f"*{role} -> {len(hits)} match(es)")
        chosen = offer(hits, f"*{role}")
        if chosen:
            return chosen, attempts
    attempts.append("no file here is one a run of ours writes")
    return None, attempts


def run_state_of(directory, names, launched_in=None):
    """The run's state through the one door (`job.run_status`), asked with
    its launch record (`runrecord.launch_record`) as every reader asks it --
    its files named by its stage's names (`runfiles.RunNames`).  A launch
    record that does not read is the state ``unreadable``, the file named in
    its detail -- never read as launched or not launched, and never
    followed.  The run door (`runs`) asks it, with the run's names.
    ``launched_in`` is the folder
    the launch record lies in, ``directory`` itself by default: a run of
    points' one record, in the run's folder, answers for each point in it
    (`engines/transport.md` § 2a.11), and the scheduler's file it names is
    named from there (`running-a-job.md` § 4.2) -- settled once here, and
    handed to `run_status` as it is."""
    from molbuilder.runrecord import (LaunchRecordError, launch_record,
                                      launch_record_path)
    from .job import RunStatus, run_status
    if launched_in is None:
        launched_in = directory
    try:
        launch = launch_record(launched_in)
    except LaunchRecordError as e:
        return RunStatus(state="unreadable", detail=str(e))
    try:
        return run_status(directory, names.stem, launch=launch,
                          launched_in=launched_in)
    except LaunchRecordError as e:
        # A RECORD THAT READS, ONE OF WHOSE VALUES DOES NOT -- the wall it
        # was sent with (`runrecord.sent_wall_s`): the same state, the file
        # named (`project-layout.md` § 1.6.3).
        return RunStatus(state="unreadable",
                         detail=f"{launch_record_path(launched_in)}: {e}")


__all__ = ["claimed", "openable_in", "run_state_of"]
