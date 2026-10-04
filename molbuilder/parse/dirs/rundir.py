"""A run folder's floor-1 questions that need the whole folder: what a viewer
should open in a folder no calculation claims (:func:`openable_in`), and how a
run is doing, asked with its launch record (:func:`run_state_of`).

Contract: `model/parse.md` § 5.  Floor 1 (`execution/architecture.md` § 2.1):
**this module knows no calculation.**  What a folder IS, the description it
belongs to, the label its files are named on and the run it speaks for are the
run door's (`molbuilder.runs`, floor 2), which asks these with what it read.

*Until 2026-10-04 this module also held `JobDirParser` -- the directory door,
which composed these with the folder's place and the calculation's kind, read
from `task.json` raw, and with labels guessed from the folder's decks
(`labels_in`, `read_back`).  It moved up to `molbuilder.runs.folder_answer`
(plan B11, B14): the label is the description's, and only `read_task` opens
the description.*
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


def _claimed(path: str) -> bool:
    """Does a registered parser claim this file? — the REGISTRY's question.

    The door never offers a file `detect()` refuses.  It did until
    2026-09-18: pointed at a finished spectrum directory with no molwatch
    log, the chain returned `<job>_<stage>.log` — PySCF's own verbose
    logger, which no parser claims — and the caller's very next step was
    `detect()`, which refused it.  *What is a run's output* and *what can a
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
    calculation produces: the PROGRESS channel and every calculation's
    product first, then the engine's stdout -- which for SIESTA IS a
    trajectory source, and for PySCF is refused by the registry and so drops
    out on its own.  Built from the catalogue, not listed here."""
    from molbuilder.runfiles import ON_THE_LABEL, result_roles, stdout_roles
    out = list(result_roles(None))
    out += [a.role for a in ON_THE_LABEL if a.calculation and a.role not in out]
    out += [r for r in stdout_roles() if r not in out]
    return [r for r in out if r.startswith(".")]


def openable_in(directory: str, calculation: Optional[str] = None
                ) -> Tuple[Optional[str], List[str]]:
    """*What should a viewer load here?* — and the trail of what was tried —
    for a folder read WITHOUT its run: by the catalogue's roles alone,
    newest first, each vetted by the registry (`model/parse.md` § 5.2).

    ``calculation`` is the kind, when the caller knows it from the
    description (`runs.place_of`); the roles a calculation produces come
    first then (`runfiles.result_roles`).  A run of OURS is opened by the
    run door instead (`runs.folder_answer`): its speaking run's own file,
    read back with its label -- this floor knows no label, so it can only
    ask by a dotted role, and among one role's files it can only take the
    newest.  *(The search read a label off the folder's decks, and tried
    generic names for a folder nobody described, until 2026-10-04 -- a
    search for a run molbuilder's wrapper did not run: plan B11.)*

    ``attempts`` is not decoration: it is the BODY of the refusal a person
    reads when nothing matched, and it moves with the search so the message
    cannot drift from it.
    """
    from molbuilder.runfiles import find_by_role, result_roles

    attempts: List[str] = []

    def offer(paths: List[str], how: str) -> Optional[str]:
        """The newest of *paths* the REGISTRY claims, with the trail written."""
        for cand in sorted(paths, key=lambda p: os.path.getmtime(p)
                           if os.path.isfile(p) else 0.0, reverse=True):
            if _claimed(cand):
                attempts.append(f"{how} -> {os.path.basename(cand)}")
                return cand
            attempts.append(f"{how} -> {os.path.basename(cand)}: "
                            f"no parser claims it, not offered")
        return None

    tried = []
    for role in [r for r in result_roles(calculation) if r.startswith(".")] \
            + _search_roles():
        if role in tried:
            continue
        tried.append(role)
        hits = [str(p) for p in find_by_role(directory, role)]
        if not hits:
            continue
        attempts.append(f"*{role} -> {len(hits)} match(es)")
        chosen = offer(hits, f"*{role}")
        if chosen:
            return chosen, attempts
    attempts.append("no file here is one a run of ours writes")
    return None, attempts


def run_state_of(directory, basename: str):
    """``(RunStatus, problem)`` -- the run's state through the one door
    (`job.run_status`), asked with its launch record
    (`runrecord.launch_record`) as every reader asks it.  A launch record
    that does not read is the ``problem``, said beside a state the files
    answer without it -- never read as launched or not launched.  The run
    door (`runs`) asks it, with the run's own stem."""
    from molbuilder.runrecord import LaunchRecordError, launch_record
    from .job import run_status
    try:
        launch = launch_record(directory, basename)
    except LaunchRecordError as e:
        return run_status(directory, basename), str(e)
    return run_status(directory, basename, launch=launch), None


__all__ = ["openable_in", "run_state_of"]
