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


def _calculation_of(directory: str) -> Optional[str]:
    """What calculation is this? — the DIRECTORY's own account of itself.

    `task.json` is the file `prep` reads, so this is the same fact the run
    was built from rather than a guess off the filenames.  The name and the
    reader are `molbuilder.task`'s; nothing here re-spells either.

    ``None`` when the directory does not say — one molbuilder did not write,
    or one prepped before the key existed.  That is a real answer, not a
    failure: the search below then asks what ANY run produces.
    """
    from molbuilder.task import FILENAME as _TASK, read_json as _read_json
    try:
        return (_read_json(os.path.join(directory, _TASK)) or {}
                ).get("calculation") or None
    except (OSError, ValueError, TypeError):
        return None


def _search_roles() -> "List[str]":
    """The roles to try, in order, for a directory that did not say what it is.

    Built from the catalogue, not listed here: the PROGRESS channel first
    (every run has one), then whatever any calculation names as its product,
    then the trajectory, then the engine's stdout — which for SIESTA IS a
    trajectory source, and for PySCF is refused by the registry and so drops
    out of this list on its own.
    """
    from molbuilder.runfiles import WRITTEN, result_roles, stdout_roles
    out = list(result_roles(None))
    out += [a.role for a in WRITTEN if a.calculation and a.role not in out]
    if ROLE_GEOM_TRAJ not in out:
        out.append(ROLE_GEOM_TRAJ)
    out += [r for r in stdout_roles() if r not in out]
    return out


def openable_in(directory: str) -> Tuple[Optional[str], List[str]]:
    """*What should a viewer load here?* — and the trail of what was tried.

    **THREE QUESTIONS, THREE OWNERS**, which is the same shape `run_status`
    took on 2026-09-18 and the reason this is no longer a ladder:

      | what calculation is this?  | the DIRECTORY | `task.json`              |
      | what does it produce?      | the CATALOGUE | `runfiles.result_roles`  |
      | can anything open it?      | the REGISTRY  | `detect()`               |

    **The calculation decides, so there is no preference order to tune.**  A
    vibration run is FOR its `.spectra.json` — the deck rewrites it
    atomically at every phase boundary and it carries its own `phase_*`
    flags, so it is the live view during the run and the result after it.  An
    optimization is for its trajectory.  Neither switches at conclusion; the
    "unconcluded progress log first" rule this replaces was an
    optimization-shaped rule generalised to every kind, and it sent every
    spectrum run's viewer to a molwatch log holding one `initial_preview`
    block.

    WHAT REMAINS A SEARCH, and why: a directory that does not say what it is.
    Then the deck is asked for its label and the label's files are looked up
    — through `runfiles.find`, which knows the attempt counter, where this
    hand-rolled `compose` + `isfile` did not and so missed every
    `-run<N>` spelling.  Every candidate goes through the registry either
    way.

    ``attempts`` is not decoration: it is the BODY of the refusal a person
    reads when nothing matched, and it moves with the search so the message
    cannot drift from it.
    """
    from molbuilder.runfiles import find, find_by_role, result_roles

    attempts: List[str] = []

    def by_role(role: str) -> List[str]:
        return [str(p) for p in find_by_role(directory, role)]

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

    # 1. WHAT THIS CALCULATION PRODUCES, from the catalogue.
    calc = _calculation_of(directory)
    attempts.append(f"calculation: {calc or '(not stated in task.json)'}")
    for role in result_roles(calc):
        hits = by_role(role)
        attempts.append(f"*{role} -> {len(hits)} match(es)")
        chosen = offer(hits, f"*{role}")
        if chosen:
            return chosen, attempts

    # 2. THE DIRECTORY DID NOT SAY WHAT IT IS, so the DECK is asked for the
    #    label and the label's own files are looked up.  Both engines, one
    #    loop: the deck role and the reader that pulls the label out of it
    #    are the only per-engine facts, and neither is a role vocabulary.
    from molbuilder.parse.fdf import system_label
    from molbuilder.pyscf.input import job_name
    stems: List[str] = []
    for deck_role, label_of in ((".fdf", system_label), (".py", job_name)):
        deck_hits = by_role(deck_role)
        attempts.append(f"*{deck_role} -> {len(deck_hits)} match(es)")
        for deck in deck_hits:
            label = label_of(_read_head(deck))
            if label and label not in stems:
                stems.append(label)
            # THE DECK'S OWN STEM TOO: a staged deck is `<job>_<token>` while
            # the label inside it stays bare, so a staged run whose seed is
            # missing resolved to nothing until 2026-08-19.
            stem = os.path.splitext(os.path.basename(deck))[0]
            if stem not in stems:
                stems.append(stem)

    # ROLE FIRST, THEN NEWEST -- and the order between those two is the whole
    # rule.  The role says what the file IS; mtime picks WHICH ONE, which in
    # the flat shape is the latest rung (`model/parse.md` § 5.1, the same rule
    # `run_status` picks `active` by).  Gathering across every stem before
    # offering is what keeps that true: returning on the first deck handed a
    # four-stage flat run its FIRST stage, because decks sort by name
    # (measured 2026-09-18 on four real directories).
    for role in _search_roles():
        pool: List[str] = []
        for s in stems:
            # A DOTTED STEM NEEDS NO GUARD HERE, and that is a property of
            # `find` rather than luck: it READS names and returns what
            # matches, where the `compose` this replaced BUILT one and
            # refused `my.job` (§ 2.1).  There is no raise left to catch --
            # `tests/test_path_framework_doors.py` asserts the skip message
            # never appears -- so the `except RunFileError` that stood here
            # went with the composer on 2026-09-18.
            pool += [str(path) for path, _rec in
                     find(directory, s, role=role, roles=(ROLE_GEOM_TRAJ,))]
        if not pool:
            continue
        attempts.append(f"  *{role} across {len(stems)} label(s)"
                        f" -> {len(pool)} match(es)")
        chosen = offer(pool, f"  *{role}")
        if chosen:
            return chosen, attempts

    # 3. NO DECK NAMED A LABEL, so the roles are searched WITHOUT one.
    #    `find_by_role` is the label-less half of the same door, and it takes
    #    a dotted role only -- which is its own rule, not a limitation
    #    invented here: an underscore role cannot be told from a stage name
    #    without a label, so the trajectory keeps the glob below.
    for role in _search_roles():
        if not role.startswith("."):
            continue
        hits = by_role(role)
        if not hits:
            continue
        attempts.append(f"*{role} (no label) -> {len(hits)} match(es)")
        chosen = offer(hits, f"*{role}")
        if chosen:
            return chosen, attempts

    # 4. GENERIC NAMES, for a directory molbuilder did not write at all.
    generic = [os.path.join(directory, n) for n in ("run.out", "siesta.log")]
    # ONE glob for the trajectory, because there is one spelling: the role is
    # `_geom_optim.xyz` and the star covers the label and the token.  NOT
    # `find_by_role` -- an UNDERSCORE role cannot be told from a stage name
    # without a label, and having no label is the whole point of this rung.
    generic += glob.glob(os.path.join(directory, "*" + ROLE_GEOM_TRAJ))
    present = [c for c in generic if os.path.isfile(c)]
    attempts.append(f"generic names -> {len(present)} match(es)")
    chosen = offer(present, "generic")
    if chosen:
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
            # ruling settled.  `summarize`'s highest-`-runN` rule was
            # WITHDRAWN, not beaten: it is handed a basename that already
            # carries the stage, so the stage is not a variable there
            # (`plan.md` § 5c, `model/parse.md` § 5.1).
            active=st.active_source,
            openable=openable,
            attempts=attempts,
            status={"state": st.state, "detail": st.detail,
                    "last_change_at": st.last_change_at,
                    "active_source": st.active_source},
        )
