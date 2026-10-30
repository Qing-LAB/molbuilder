"""Where a calculation's files live — the layout, and how to FIND it.

**Module:** floor 1, and **stdlib-only**, which is the property the whole
design rests on.  The monitor ships beside a job
(``runwrap.MONITOR_BUNDLE``) and runs under the JOB's python with no
molbuilder installed; ``config_dir.py`` already travels for that reason.  A
path module a running job cannot import is a path module the job works
around.

**Contract:** [`project-layout.md`](?doc=execution/project-layout.md) § 4.5 —
*for every name it composes, the framework owns the search.*  The run-file
GRAMMAR stays in `runfiles` (§ 2.2a); this owns the TREE and the finders.

Also [`job-contracts.md`](?doc=execution/job-contracts.md) § 6.3 (the
names).

WHAT IS HERE.  The one layout (`project-layout.md` § 1: a folder per stage,
`<NN>_<stage>/`, a folder per run in it, `run-<n>/`): the stage finder, the
benchmark's container, the trial, and the ATTEMPT, `run-<n>`, with its
composer, its reader and its finder.  The run-file GRAMMAR is next door in
`runfiles`; this is the tree those names sit in.

WHAT IT DELIBERATELY DOES NOT DO.  It does not read a description: the
**surfaces** hold a calculation's folder and hand down what they read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# ══ THE ATTEMPT ════════════════════════════════════════════════════════════
#
# ``run-<n>``, the directory one run of a stage or a trial is -- its tries
# among it (`project-layout.md` § 1.5, § 1.6.1 — immutable once it has run,
# which is why a launch again opens the next rather than landing on top of
# one).

#: What an attempt directory starts with.  One home, so the composer and the
#: reader below cannot drift, and neither can a caller.
ATTEMPT_PREFIX = "run-"
#: A stage's first run folder, ``run-0``, which its prep opens
#: (`project-layout.md` § 1.6.2).
FIRST_ATTEMPT = 0


def attempt_name(n: int) -> str:
    """What attempt *n*'s directory is called.

    Separate from :func:`attempt_dir` because a caller that is LISTING
    attempts wants the name and not a path -- `runstatus` prints them in a
    row, and composing a path to take ``.name`` off it again is the kind of
    detour that sends the next person back to an f-string.
    """
    return f"{ATTEMPT_PREFIX}{int(n)}"


def frame_token(frame: int) -> str:
    """A frame's folder token: its index, three digits -- ``f000`` the
    first, frame index 0 -- a transport point's frame in its cited set
    (`engines/transport.md` § 2a.9) and a vibration's optical call
    (`engines/vibration.md` § 5.11 § 2) alike; the code's index everywhere
    but the UI's display, a folder name included (`model/structure.md`
    § 2.2e)."""
    return f"f{int(frame):03d}"


def frame_index(token) -> Optional[int]:
    """The frame index a folder token names -- ``f003`` is 3 -- the reader
    of :func:`frame_token`; ``None`` for a name that is no frame's."""
    name = str(token)
    return int(name[1:]) if (len(name) > 1 and name[0] == "f"
                             and name[1:].isdigit()) else None


def attempt_dir(container, n: int) -> Path:
    """The directory attempt *n* of this stage or trial runs in."""
    return Path(container) / attempt_name(n)


def run_folder_of(path) -> Optional[Path]:
    """The run folder ``path`` lies in -- ``04_device/run-1`` of
    ``04_device/run-1/f001/v0.2``: its parts up to its ``run-<n>``
    (:func:`attempt_index`) -- or ``None`` when no part is one.  ``path`` is
    RELATIVE to its calculation, as a record names a run; a folder on disk
    asks where it is (`calcdirs.where`, its ``run_folder``), since an
    absolute path's first ``run-<n>`` may lie above the calculation."""
    parts = Path(path).parts
    at = next((i for i, part in enumerate(parts)
               if attempt_index(part) is not None), None)
    return None if at is None else Path(*parts[:at + 1])


def attempt_index(name: str) -> Optional[int]:
    """The ``n`` in ``run-<n>``, or ``None`` when this is not one.

    The reader for :func:`attempt_dir`, per § 4.5.  **Not padded** (§ 4.3),
    so ``run-10`` is ten and not a tenth: a caller sorting these must sort the
    INTEGERS, which is the whole reason this returns one.

    Takes a bare directory NAME.  A caller holding a path passes ``p.name``;
    accepting either would make ``run-1/run-2`` ambiguous.
    """
    if not name.startswith(ATTEMPT_PREFIX):
        return None
    tail = name[len(ATTEMPT_PREFIX):]
    return int(tail) if tail.isdigit() else None


def attempts_in(container) -> "list[int]":
    """Every attempt index present in *container*, ascending.

    The finder half -- every caller's.
    """
    c = Path(container)
    try:
        found = [attempt_index(d.name) for d in c.iterdir() if d.is_dir()]
    except OSError:
        return []
    return sorted(n for n in found if n is not None)



# ══ THE BENCH CONTAINER AND THE TRIAL ══════════════════════════════════════
#
# `project-layout.md` § 2.6: the benchmark rows carry NO circled number --
# they are "nested containers inside a stage (④), not levels of the tree."
# So a trial is a QUALIFIER on ④, and both names below are layout, which is
# why they live here beside `stages_in` rather than in the module that
# happened to need them first: the address layer is floor 1, and a rule the
# framework cannot reach is a rule its callers re-spell, which is § 4.5's
# whole subject.

#: What a TRIAL directory starts with: ``bench-`` and the point, one
#: qualifier (§ 6.3's separator rule) -- ``01_coarse/bench/bench-G1N4C6``, in
#: its stage's container (:func:`bench_container`).
TRIAL_PREFIX = "bench-"


def trial_name(point: str) -> str:
    """What the trial for *point* is called -- ``bench-<point>``.

    `job-contracts.md` § 6.3 is the authority: ``bench-`` plus the coordinate
    as ONE qualifier.  The composer half of the pair; :func:`trial_point`
    is its reader.
    """
    return f"{TRIAL_PREFIX}{point}"


def trial_point(name: str) -> Optional[str]:
    """The ``<point>`` in ``bench-<point>``, or None when this is not one.

    The reader for :func:`trial_name`, per § 4.5.  Takes a bare directory
    NAME for :func:`attempt_index`'s reason.
    """
    if not name.startswith(TRIAL_PREFIX):
        return None
    point = name[len(TRIAL_PREFIX):]
    return point or None


def trial_label(label: str, point: str) -> str:
    """``<label>-<point>`` -- THE LABEL A TRIAL'S FILES ARE NAMED ON: the
    calculation's, with the point's token appended (`project-layout.md`
    § 2.3.2).  A trial is relabelled so its warm files never meet the real
    run's: SIESTA finds them by ``SystemLabel``.

    THE ONE COMPOSER, and every reader asks it: prep's element label
    (`resolve._label_for`), the run door's reading of a trial's folder
    (`runs`), a trial's deck read back for its stage token
    (`jobset.naming`), and the job a status row reads (`runstatus`).
    """
    return f"{label}-{point}"


def stages_in(root) -> "list[tuple[int, str]]":
    """Every stage a calculation's folder holds, as ``(seq, name)`` -- the
    finder half of a stage's folder, ``<NN>_<stage>/`` (§ 4.5: for every
    name the framework composes, it owns the search).

    A stage's number is READ OFF WHAT IS THERE, never off the description
    (`project-layout.md` § 4.1, § 4.2): its directory.  A stage the
    description no longer names still holds its number here -- gaps are
    honest.
    """
    try:                                    # inside molbuilder
        from .identity import parse_token
    except ImportError:                     # beside a job (mb_vibration.pyz)
        from identity import parse_token
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return []
    found = set()
    for e in entries:
        if e.is_dir():
            got = parse_token(e.name)
            if got:
                found.add(got)
    return sorted(found)


#: A stage's benchmark container, inside the stage's folder
#: (`<NN>_<stage>/bench`).
BENCH_DIRNAME = "bench"


def bench_container(token: str) -> str:
    """Where a stage's bench state lives, **relative to the calculation
    root**: ``<NN>_<stage>/bench`` -- `job-contracts.md` § 6.3, *"benchmark |
    bench/ inside the stage it measures"*.  An empty token is refused: every
    description has a stage (`engines/stages.md` § 6.5).  This is the ONE
    spelling of that rule."""
    if not token:
        raise ValueError("a benchmark's container is its stage's -- no stage "
                         "token was given")
    return f"{token}/{BENCH_DIRNAME}"


def bench_containers_in(root) -> "list[tuple[str, str]]":
    """Every bench container under *root*, as ``(relative name, stage token)``
    -- the containers :func:`bench_container` names, each its stage's: the
    SEARCH half § 4.5 requires, the stage token the container's parent."""
    try:                                    # inside molbuilder
        from .identity import parse_token
    except ImportError:                     # beside a job
        from identity import parse_token
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return []
    return [(f"{d.name}/{BENCH_DIRNAME}", d.name) for d in entries
            if d.is_dir() and parse_token(d.name)
            and (d / BENCH_DIRNAME).is_dir()]


__all__ = [
    "stages_in",
    # the bench container and the trial -- ④'s qualifier, not a level
    "TRIAL_PREFIX", "trial_name", "trial_point", "bench_container",
    "BENCH_DIRNAME", "bench_containers_in",
    # the attempt -- composer, reader, finder (§ 4.5's pairing)
    "run_folder_of", "ATTEMPT_PREFIX", "FIRST_ATTEMPT", "attempt_name", "attempt_dir",
    "attempt_index", "attempts_in",
    # a frame's folder
    "frame_token",
]
