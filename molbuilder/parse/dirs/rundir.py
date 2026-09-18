"""`JobDirParser` — one run directory, the four questions asked about one.

Contract: `model/parse.md` § 5.  Plan: `plans/plan.md` § 5c, step 1.

**What this replaces, and what it does not.** A `JobDirParser` existed and was
DELETED 2026-09-04: it answered an eleven-field `JobResult`, ten of whose
fields had no reader anywhere, and reached the eleventh by parsing every
result file to build plots and then discarding them.  This is not that
returning.  The name was always right — it *is* the directory composer — but
every field here is written against a caller that exists today, and § 5.0's
table names each one.

**Why a door at all.** Six functions across three modules answer questions
about a run directory, and the seventh consumer is the Results file picker in
the BROWSER — which asks nothing, because there is no door to ask, and so
decides what to list from filenames in JavaScript.  That guess is why a
finished transport run was invisible on the Results tab until a presenter was
written for it, and why a ladder cannot be told from five unrelated runs.

**This module composes; it does not re-parse.** `run_status` and
`_enumerate_files` stay in `job.py`, `engine_of` in `contract.py`; they are
CALLED here.  The one thing absorbed bodily is the openable-discovery chain,
which lived in `web/blueprints/watch.py` — the web layer, which nothing below
it can import, which is the whole reason it had to move rather than be
called.

**Step 1 of the migration is this file plus a proof**, and no caller moves
until the proof passes: the new door must answer identically to the six
functions on the real tree, the way the `run_status` split was proved
(113/113) before its deletion was allowed.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ..base import DirParser
from ..types import ParseResult, RunDirResult


#: geomeTRIC's trajectory role, tried by the PySCF rung of the chain.
#
#: IMPORTED, not spelled.  `pyscf/input.py` declares it beside the writer that
#: emits it and warm-files.toml that declares it; a literal here would be a
#: second home for one string.  *(Step 1 shipped it as a literal on 2026-09-18
#: and this is the correction -- the `web/watch` copy it was absorbed from had
#: imported it all along.)*
from ...pyscf.input import ROLE_GEOM_TRAJ   # noqa: E402  -- re-exported below


def _newest(paths: List[str]) -> Optional[str]:
    """The most recently modified of *paths*, or None when empty.

    Newest wins because a STAGED run leaves one log per rung in the same
    directory under the flat shape, and the one a person opening the folder
    means is the one still being written.
    """
    real = [p for p in paths if os.path.isfile(p)]
    if not real:
        return None
    return max(real, key=lambda p: os.path.getmtime(p))


def openable_in(directory: str) -> Tuple[Optional[str], List[str]]:
    """*What should a viewer load here?* — and the trail of what was tried.

    **Absorbed verbatim from `web/blueprints/watch.py::_resolve_run_directory`
    2026-09-18** (`plans/plan.md` § 5c step 1).  Behaviour is unchanged on
    purpose: § 5.2 calls this "the discovery chain, unchanged in behaviour",
    and the migration proves equivalence before any caller moves.

    Four rungs, first hit wins (`job-contracts.md` § 2.4):

      1. any ``*.molwatch.log`` — newest, which is the staged run's latest;
      2. ``*.fdf`` → its ``SystemLabel`` → ``<label>.molwatch.log``, ``.out``;
      3. ``*.py`` → its ``JOB`` → ``<job>.molwatch.log``, ``.log``,
         ``<job>_geom_optim.xyz`` — **and the deck filename's stem tried the
         same way**, because a staged deck is ``<job>_<token>.py`` while
         ``JOB`` stays bare, so every staged spelling was the unstaged one
         until 2026-08-19 and a staged run without a molwatch seed resolved
         to nothing;
      4. generic: ``run.out``, ``siesta.log``, ``*.out``, ``*_geom_optim.xyz``.

    ``attempts`` is not decoration: it is the BODY of the refusal a person
    reads when nothing matched, and it moves with the chain so the message
    cannot drift from the search.
    """
    from molbuilder.runfiles import (RunFileError, compose as _rf,
                                     find_by_role)

    attempts: List[str] = []

    def by_role(role: str) -> List[str]:
        return [str(p) for p in find_by_role(directory, role)]

    # 1. a molwatch log directly in the directory
    log_hits = by_role(".molwatch.log")
    attempts.append(f"*.molwatch.log -> {len(log_hits)} match(es)")
    if log_hits:
        return _newest(log_hits), attempts

    # 2. SIESTA: the deck names the label, the label names the outputs
    from molbuilder.parse.fdf import system_label
    fdf_hits = by_role(".fdf")
    attempts.append(f"*.fdf -> {len(fdf_hits)} match(es)")
    for fdf in fdf_hits:
        label = system_label(_read_head(fdf))
        if not label:
            attempts.append(
                f"  {os.path.basename(fdf)}: SystemLabel not found")
            continue
        for role in (".molwatch.log", ".out"):
            base = _rf(label, role)
            cand = os.path.join(directory, base)
            attempts.append(f"  -> {base}: "
                            f"{'found' if os.path.isfile(cand) else 'missing'}")
            if os.path.isfile(cand):
                return cand, attempts

    # 3. PySCF: the deck names JOB, and the deck's own stem carries the rung
    from molbuilder.pyscf.input import job_name
    py_hits = by_role(".py")
    attempts.append(f"*.py -> {len(py_hits)} match(es)")
    for py in py_hits:
        name = job_name(_read_head(py))
        if not name:
            attempts.append(f"  {os.path.basename(py)}: JOB not found")
            continue
        py_stem = os.path.splitext(os.path.basename(py))[0]
        stems = [name] if py_stem == name else [name, py_stem]
        for stem in stems:
            for role in (".molwatch.log", ".log", ROLE_GEOM_TRAJ):
                try:
                    base = _rf(stem, role)
                except RunFileError:
                    # A STEM OFF DISK IS NOT NECESSARILY A LABEL.  `py_stem`
                    # is a FILENAME, so `my.job.py` yields `my.job`, which
                    # § 2.1 refuses -- rightly, a dotted label cannot be read
                    # back out of a filename.  A RESOLVER SAYS WHAT IT TRIED
                    # AND MOVES ON; raising here turned the Watch tab into a
                    # 500 for a person who put a dot in a filename.
                    attempts.append(
                        f"  {stem}: not a run-file label (§ 2.1), skipped")
                    break
                cand = os.path.join(directory, base)
                attempts.append(
                    f"  -> {base}: "
                    f"{'found' if os.path.isfile(cand) else 'missing'}")
                if os.path.isfile(cand):
                    return cand, attempts

    # 4. generic names, for a directory molbuilder did not write
    for fname in ("run.out", "siesta.log"):
        cand = os.path.join(directory, fname)
        attempts.append(f"{fname}: "
                        f"{'found' if os.path.isfile(cand) else 'missing'}")
        if os.path.isfile(cand):
            return cand, attempts
    out_hits = by_role(".out")
    if out_hits:
        # The trail names the file actually returned -- `attempts` is the
        # body of the refusal a person reads (§ 5.2).
        chosen = _newest(out_hits)
        attempts.append(f"*.out -> picked {os.path.basename(chosen or '')}")
        return chosen, attempts
    # ONE glob, because there is one spelling: the role is `_geom_optim.xyz`
    # and everything in front of it -- label, and the token when there is one
    # -- is what the star covers.
    #
    # NOT `find_by_role`: an UNDERSCORE role cannot be told from a stage name
    # without a label, and this rung has none -- the whole point of rung 4 is
    # a directory whose label nothing has stated.  `find_by_role` raises for
    # exactly that reason, which the first draft of this move learnt by
    # running it (2026-09-18, the equivalence proof).
    optim_glob = "*" + ROLE_GEOM_TRAJ
    optim_hits = glob.glob(os.path.join(directory, optim_glob))
    if optim_hits:
        chosen = _newest(optim_hits)
        attempts.append(f"{optim_glob} -> "
                        f"picked {os.path.basename(chosen or '')}")
        return chosen, attempts

    return None, attempts


def _read_head(path: str, limit: int = 65536) -> str:
    """The first chunk of a deck — enough to find `SystemLabel` / `JOB`.

    The whole file is not read: an fdf can be multi-MB of coordinates and the
    label is in its head.  Absorbed with the chain.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


class JobDirParser(DirParser):
    """A run directory → :class:`RunDirResult`.

    Composes the readers that already exist — `job.run_status`,
    `job._enumerate_files`, `contract.engine_of` — plus the discovery chain
    above.  It inlines no file-level parsing: every file it reads goes through
    a registered `FileParser`, via those callees (`model/parse.md` § 1's rule
    for a DirParser, and § 5.4's).
    """
    name = "jobdir"
    label = "A run directory"
    output = RunDirResult

    @classmethod
    def can_parse(cls, run_dir: Path) -> bool:
        """Is this a run directory? — it holds something a run produced.

        Cheap on purpose (§ 1's ABC): a listing, no parse.  A directory with a
        deck but no output is still one — that is `not_run`, not `not mine`.
        """
        d = Path(run_dir)
        if not d.is_dir():
            return False
        from molbuilder.runfiles import find_by_role
        for role in (".out", ".molwatch.log", ".fdf", ".py"):
            if find_by_role(str(d), role):
                return True
        return False

    @classmethod
    def parse(cls, run_dir: Path) -> RunDirResult:
        from .job import _enumerate_files, run_status
        from ..contract import engine_of

        d = str(Path(run_dir))
        st = run_status(d)
        files = _enumerate_files(Path(d))
        openable, attempts = openable_in(d)
        return RunDirResult(
            **ParseResult.envelope(cls.name, d),
            run_dir=d,
            engine=engine_of(d),
            files={k: list(v) for k, v in files.items()},
            # ACTIVE IS THE STATUS'S OWN PICK -- stage, then mtime (§ 5.1,
            # user ruling 2026-09-04).  It is not recomputed here: two rules
            # for "which file speaks for the directory" is the defect that
            # ruling settled, and `summarize`'s highest-`-runN` lost because
            # a run index says nothing about which STAGE a file belongs to.
            active=st.active_source,
            openable=openable,
            attempts=attempts,
            status={"state": st.state, "detail": st.detail,
                    "last_change_at": st.last_change_at,
                    "active_source": st.active_source},
        )
