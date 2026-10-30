"""`calcdir.json` — a directory's own account of where it sits.

*Not to be confused with `molbuilder.placement`, which is a different
question with the same English word: where the CONFIGURED tree sits on a
machine.  This module is about a directory inside a CALCULATION.*

Contract: `execution/project-layout.md` § 1.4a, invariant 6b.

**Why this file exists.** § 1.4 has always said a directory here is a
*container* or a *run*, and then said the quiet part: *"there is no directory
where the two are mixed and something has to tell them apart."*  Nothing in the
TREE has to.  **A reader handed a path does**, and could not — the answer is not
decidable from the name: a stage directory is a container, a swept rung's
``run-<n>/`` holds its points' runs, and a hand-made folder with one ``.fdf``
in it is a run.

**Two fields, and each earns its place** (§ 1.4a).  Everything else a reader
might want — what the folder is (a stage's, a run's, a point's, a benchmark's),
its stage, its run, its trial and its point — is READ OFF ITS PLACE in the names
the layout gives its folders (`identity.parse_token`, `paths.attempt_index`,
`trial_point`, the benchmark's and the launch folder's names): :func:`where`,
the one answer.  A fact the tree's names already answer
does not get a second home.

**What the names cannot answer, and so is stored here.**  Container or run: a
swept rung's ``run-<n>/`` holds its points' runs where a rung that runs once is
a run.  Whether a rung sweeps is the description's, which this floor does not
read, and only the code that created the directory knew which it was making.
That is the whole content of this record.

**Absence is not a refusal** (§ 1.4a).  A directory with no record is read
ALONE: its files, which a parser claims, which one to open, how its run ended.
What it cannot answer is everything relational, and the reader says so rather
than guessing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

# TWO WAYS, because this module travels beside a job in ``mb_monitor.pyz``
# and ``mb_vibration.pyz`` (`runwrap.MONITOR_COMPANIONS`,
# `VIBRATION_COMPANIONS`), where the package is not installed.
try:                                       # inside molbuilder
    from .identity import parse_token
    from .paths import BENCH_DIRNAME, attempt_index, trial_point
    from .persist import json_text, read_json
    from .runfiles import LAUNCH_DIR
except ImportError:                        # beside a job
    from identity import parse_token
    from paths import BENCH_DIRNAME, attempt_index, trial_point
    from persist import json_text, read_json
    from runfiles import LAUNCH_DIR

#: The record's filename.  VISIBLE, not dotted: it sits beside `task.json`,
#: `job-set.json`, `run.json` and `environment.json`, and a person listing a
#: directory should see the thing that explains it (user ruling:
#: *"explicit information is better than implicit, so people can see the
#: JSON"*).
try:                                       # inside molbuilder
    from .runfiles import CALCDIR_FILE as FILENAME  # noqa: E402 -- the catalogue's
except ImportError:                        # beside a job
    from runfiles import CALCDIR_FILE as FILENAME  # noqa: E402

SCHEMA = "molbuilder/calcdir@1"

#: § 1.4's own two words.  What a folder IS -- a stage's, an attempt, a
#: point's, a benchmark's, a trial's, support -- is READ OFF ITS PLACE
#: (:func:`where`, § 1.4a) -- not a role.
CONTAINER = "container"
RUN = "run"
#: The one folder that is neither: the engine's scratch of the run above it -- a device run's ``tsgf/``
#: (`engines/transport.md` § 2a.11): read alone, no run asked of it
#: (project-layout.md § 1.4a, invariant 6b).
SCRATCH = "scratch"
#: A transmission point's ``shared/``: the energies it shares with the
#: finished calculation its calculation reuses, copied from the point that
#: computed them (`engines/transport.md` § 2a.16) -- read with the point, by
#: its one join (`transport.energies.joined`), never as a run.
SHARED = "shared"
ROLES = (CONTAINER, RUN, SCRATCH, SHARED)

#: WHAT A FOLDER IS in its calculation (§ 1.4a, *Where am I*), read off its
#: place in the layout's names: the calculation's root -- its description is
#: there; a stage's folder, ``<NN>_<stage>/``; an attempt, ``run-<n>/`` -- a
#: stage's or a trial's; a point's folder -- below a run of points, or the one
#: prep renders a point's deck into below its stage; the engine's scratch of
#: the run above (the role's own word); a stage's benchmark,
#: ``<NN>_<stage>/bench/``; a trial, ``bench-<point>/``; support --
#: ``pseudos/``, a ``launch/``.
CALCULATION = "calculation"
STAGE = "stage"
ATTEMPT = "attempt"
POINT = "point"
BENCHMARK = "benchmark"
TRIAL = "trial"
SUPPORT = "support"


@dataclass(frozen=True)
class Placement:
    """What a directory said about itself.  ``of`` is as written — relative."""
    role: str
    of: str


def record(directory, *, role: str, root) -> Tuple[Path, str]:
    """``(path, text)`` of the stamp that says what *directory* is, recording
    the calculation *root* it belongs to -- what `prep` puts in its plan
    before anything is written (`jobset.planned`), and `launch` with the
    attempt it opens.

    ``of`` is stored RELATIVE, which is what survives renaming or moving a
    whole calculation — the convention `.gathered-from` already uses.  The
    caller passes the root it already has; nothing is searched for here.
    """
    if role not in ROLES:
        raise ValueError(
            f"placement role must be one of {ROLES}, got {role!r} -- what a "
            f"folder is (a stage's, an attempt, a point's, a trial...) is read "
            f"off its place, not a role (project-layout.md § 1.4a)")
    directory = Path(directory)
    of = os.path.relpath(Path(root).resolve(), directory.resolve())
    return (directory / FILENAME,
            json_text({"schema": SCHEMA, "role": role, "of": of}))


def read(directory) -> Optional[Placement]:
    """*What is this directory?* — or ``None`` when it does not say.

    ``None`` is an ANSWER (§ 1.4a): the directory is read alone.  It is not an
    error, and every caller has a sane branch for it, so a malformed or
    unreadable record answers the same way rather than raising into a viewer.
    """
    try:
        said = read_json(Path(directory) / FILENAME)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(said, dict):
        return None
    role, of = said.get("role"), said.get("of")
    if role not in ROLES or not isinstance(of, str):
        return None
    return Placement(role=role, of=of)


def description_path(directory) -> Path:
    """``<directory>/task.json`` -- THE ONE PLACE the description's name, the
    catalogue's (`runfiles.TASK_FILE`), is joined to a folder: the ground
    floor's, so *is this a calculation's folder* is answered here
    (:func:`described`) without reading the description; its writer and
    `jobset migrate` reach it through its owner (`task.path_of`)."""
    try:                                    # inside molbuilder
        from .runfiles import TASK_FILE
    except ImportError:                     # beside a job
        from runfiles import TASK_FILE
    return Path(directory) / TASK_FILE


def described(directory) -> bool:
    """Does ``directory`` hold a calculation's description -- is it a
    calculation's root (`project-layout.md` § 1.4a, invariant 2)?  The
    description is not read: its file is there or it is not."""
    return description_path(directory).is_file()


def root_of(directory) -> Optional[Path]:
    """The calculation root this directory belongs to, or ``None``.

    Two ways a directory can answer, and they are asked in the order § 1.4a
    puts them:

    1. **it holds `task.json`** — then it IS the root (invariant 2: a
       calculation directory is the one the description is in), and no record
       is needed or written there;
    2. **it holds `calcdir.json`** — then ``of`` points at the root.

    ``None`` means neither, which is the read-alone case.  A dangling ``of``
    — an attempt copied out of its calculation — also answers ``None``: *this
    attempt's calculation is not here* is the honest reading, where a search
    would have adopted whatever it found.  The root of :func:`where`'s
    answer, which asks both.
    """
    return where(directory).root


@dataclass(frozen=True)
class Where:
    """WHERE A FOLDER IS (`project-layout.md` § 1.4a, *Where am I*): the
    calculation's ``root``; ``what`` the folder is -- :data:`CALCULATION`,
    :data:`STAGE`, :data:`ATTEMPT`, :data:`POINT`, :data:`SCRATCH`,
    :data:`BENCHMARK`, :data:`TRIAL` or :data:`SUPPORT`; ``role``, its
    record's (``None`` at the root, a container, which holds no record); and
    where it sits -- the ``stage`` token (``01_coarse``), the ``run`` index,
    the benchmark ``trial``'s point, and for a point's folder the ``point``,
    its path below the run or the stage (``f001/v0.2``); ``run_folder``, the
    run's own ``run-<n>/`` it lies in -- itself for an attempt, the one above
    a point's folder or a scratch folder -- where a launch record lies.  A
    folder no
    calculation claims -- with neither `task.json` nor a `calcdir.json`
    naming its root -- has none of them."""
    root: Optional[Path] = None
    what: Optional[str] = None
    role: Optional[str] = None
    stage: Optional[str] = None
    run: Optional[int] = None
    trial: Optional[str] = None
    point: Optional[str] = None
    run_folder: Optional[Path] = None


def where(directory) -> Where:
    """WHERE ``directory`` IS in its calculation -- the one answer (§ 1.4a,
    *Where am I*).  The root, as :func:`root_of` says it; what the folder
    is, its record's role read at its place in the layout's names: below the
    root a stage's ``<NN>_<stage>/`` or support; below a stage its attempts ``run-<n>/``, its benchmark
    ``bench/``, its ``launch/``, else a point prep renders into; below a
    benchmark its trials ``bench-<point>/`` and its ``launch/``; below a
    trial its attempts; below an attempt its points, or its scratch, as its
    role says.  A folder that states no place -- no `task.json`, no
    `calcdir.json` naming its root -- answers an empty :class:`Where`: it is
    read alone, nothing above it searched for (§ 1.4a: a directory states
    its place, nothing infers it)."""
    d = Path(directory)
    # THE ROOT, the two ways a folder says it (:func:`root_of`): its
    # description is here, or its record names the folder it is in.
    if described(d):
        return Where(root=d, what=CALCULATION)
    said = read(d)
    if said is None:
        return Where()
    root = (d / said.of).resolve()
    if not described(root):
        return Where()
    role = said.role
    try:
        parts = d.resolve().relative_to(root).parts
    except ValueError:
        return Where(root=root, role=role)
    stage = run = trial = None
    below = []
    deep = 0          # how many of the parts reach the run's own folder
    at: Optional[str] = CALCULATION
    for i, name in enumerate(parts, start=1):
        if at == CALCULATION:
            if parse_token(name):
                stage, at = name, STAGE
            else:
                at = SUPPORT
        elif at == STAGE:
            n = attempt_index(name)
            if n is not None:
                run, at, deep = n, ATTEMPT, i
            elif name == BENCH_DIRNAME:
                at = BENCHMARK
            elif name == LAUNCH_DIR:
                at = SUPPORT
            else:
                at = POINT
                below.append(name)
        elif at == BENCHMARK:
            point = trial_point(name)
            trial, at = (point, TRIAL) if point else (trial, SUPPORT)
        elif at == TRIAL:
            n = attempt_index(name)
            if n is not None:
                run, at, deep = n, ATTEMPT, i
            else:
                at = None
        elif at in (ATTEMPT, POINT):
            at = POINT
            below.append(name)
    what = role if role in (SCRATCH, SHARED) else at
    return Where(root=root, what=what, role=role, stage=stage, run=run,
                 trial=trial,
                 point="/".join(below) if what == POINT else None,
                 run_folder=root.joinpath(*parts[:deep]) if deep else None)


def _levels(rel: str) -> Tuple[str, ...]:
    """The folder levels of ``rel`` -- a point's path below its run or its
    stage (`execution/runs-of-points.md` § 2.4) -- refused by name when it
    is absolute or climbs out (``..``); ``()`` for ``""``, a point with no
    level."""
    if not rel:
        return ()
    p = PurePosixPath(str(rel))
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"{rel!r} is no folder below a run or a stage: a "
                         f"point's folder is a relative path that stays "
                         f"below it")
    return p.parts


def prepared_folder(stage_dir, rel: str) -> Path:
    """A POINT'S PREPARED FOLDER below its stage -- where prep writes its
    deck and run script (`execution/runs-of-points.md` § 2.4): the point's
    own ``prepared`` (`stages.Point.prepared`, `vibration_phases.Call.
    prepared`), at prep, before any run is open; :meth:`RunPlace.prepared`
    once one is."""
    return Path(stage_dir).joinpath(*_levels(rel))


@dataclass(frozen=True)
class RunPlace:
    """A RUN'S PLACES, COMPOSED -- the writing half of the grammar
    :func:`where` reads (`project-layout.md` § 1.4a;
    `execution/runs-of-points.md` § 2.4): the stage's folder and the run's
    ``run-<n>/`` in it, and every folder a run of points holds or reads --
    a point's below the run, a container's, the run's scratch, a point's
    prepared folder below the stage -- each composed from the point type's
    own words (``rel``: which folder below the run; ``prepared``: which
    below the stage), and each spelled as the walk script names it from the
    stage's folder.  Of every folder it composes, :func:`where` reads back
    the same run (``run_folder``) and the same point (``point``).  File
    names inside a folder are the catalogue's (`runfiles`), never
    composed here."""
    stage_dir: Path
    run_dir: Path

    @classmethod
    def of(cls, run_dir) -> "RunPlace":
        """The places of the run in ``run_dir`` -- a ``run-<n>/`` folder,
        refused by name otherwise."""
        run_dir = Path(run_dir)
        if attempt_index(run_dir.name) is None:
            raise ValueError(f"{run_dir} is no run's folder (run-<n>/)")
        return cls(stage_dir=run_dir.parent, run_dir=run_dir)

    @classmethod
    def at(cls, folder) -> "RunPlace":
        """The places of the run ``folder`` lies in -- a point's, a
        container's, the run's own -- by :func:`where`; refused by name
        for a folder below no run."""
        got = where(folder).run_folder
        if got is None:
            raise ValueError(f"{folder} lies below no run (its folders' "
                             f"records name none)")
        return cls.of(got)

    def point(self, rel: str) -> Path:
        """THE FOLDER OF THE POINT ``rel`` below the run -- a container's
        too; the run itself for a point with no level (``""``)."""
        return self.run_dir.joinpath(*_levels(rel))

    def prepared(self, rel: str) -> Path:
        """A point's PREPARED FOLDER ``rel`` below the stage
        (:func:`prepared_folder`)."""
        return prepared_folder(self.stage_dir, rel)

    def scratch(self, name: str) -> Path:
        """The run's SCRATCH folder ``name`` -- the rung's ``scratch_dir``
        (`Job.scratch_dir`)."""
        return self.run_dir.joinpath(*_levels(name))

    def walked(self, rel: str, name: Optional[str] = None) -> str:
        """A folder below the run as the walk script -- and every row that
        names a point's folder -- names it from the stage's folder:
        ``run-1/f001/v0.2``; with ``name``, a file in it,
        ``run-1/fc/CO2.DM``."""
        return "/".join((self.run_dir.name, *_levels(rel))
                        + ((name,) if name else ()))

    def walked_all(self, rel: str) -> str:
        """Every folder below the container ``rel``, as the walk's loop step
        names them from the stage's folder: ``run-1/optical/*/``."""
        return self.walked(rel) + "/*/"
