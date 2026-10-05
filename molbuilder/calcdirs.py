"""`calcdir.json` — a directory's own account of where it sits.

*Not to be confused with `molbuilder.placement`, which is a different
question with the same English word: where the CONFIGURED tree sits on a
machine.  This module is about a directory inside a CALCULATION.*

Contract: `execution/project-layout.md` § 1.4a, invariant 6b.

**Why this file exists.** § 1.4 has always said a directory here is a
*container* or a *run*, and then said the quiet part: *"there is no directory
where the two are mixed and something has to tell them apart."*  Nothing in the
TREE has to.  **A reader handed a path does**, and could not — the answer is not
decidable from the name, and § 1.4's own closing paragraph settles that three
times over: a flat calculation root IS a run, a hierarchical stage directory is
a container, and a hand-made folder with one ``.fdf`` in it is a run.

So every reader guessed, and they did not guess alike: one took every directory
for a run and reported a `pseudos/` folder as *running*; one took the
description to be in the directory it was handed, true only in the flat shape;
one walked up a fixed number of levels hoping to find it.

**Two fields, and each earns its place** (§ 1.4a).  Everything else a reader
might want — which stage, which attempt index, whether a container is a stage or
support — is DERIVED from ``of`` through `jobset.materialize.job_dir_names`, the
naming authority that wrote those directory names.  Measured over a real
three-calculation tree before this module was written: every stage container
resolved to its job, every attempt to its job and index, both shapes' roots
through ``shape``.  A fact the tree already answers does not get a second home.

**What the authority cannot answer, and so is stored here.**  It returns a bench
trial as ``<NN>_<name>/bench/bench-<point>`` — an exact job match, structurally
identical to a stage's own ``<NN>_<name>`` — and § 1.4 calls the first a
container and the second a run.  Same shape, opposite answers, and only the code
that created the directory knew which it was making.  That is the whole content
of this record.

**Absence is not a refusal** (§ 1.4a).  A directory with no record is read
ALONE: its files, which a parser claims, which one to open, how its run ended.
What it cannot answer is everything relational, and the reader says so rather
than guessing.  That is what a tree written before this rule gets, and why
there is no migration step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .persist import json_text, read_json, write_bytes

#: The record's filename.  VISIBLE, not dotted: it sits beside `task.json`,
#: `job-set.json`, `run.json` and `environment.json`, and a person listing a
#: directory should see the thing that explains it (user ruling:
#: *"explicit information is better than implicit, so people can see the
#: JSON"*).
from .runfiles import CALCDIR_FILE as FILENAME  # noqa: E402 -- the catalogue's

SCHEMA = "molbuilder/calcdir@1"

#: § 1.4's own two words, and there is no third.  `stage`, `axis`, `trial` and
#: *support* are DERIVED readings of these two (§ 1.4a) -- not roles.
CONTAINER = "container"
RUN = "run"
ROLES = (CONTAINER, RUN)


@dataclass(frozen=True)
class Placement:
    """What a directory said about itself.  ``of`` is as written — relative."""
    role: str
    of: str


def record(directory, *, role: str, root) -> Tuple[Path, str]:
    """``(path, text)`` of the stamp that says what *directory* is, recording
    the calculation *root* it belongs to -- what :func:`write` writes, and
    what `prep` puts in its plan before anything is written
    (`jobset.planned`).

    ``of`` is stored RELATIVE, which is what survives renaming or moving a
    whole calculation — the convention `.gathered-from` already uses.  The
    caller passes the root it already has; nothing is searched for here.
    """
    if role not in ROLES:
        raise ValueError(
            f"placement role must be one of {ROLES}, got {role!r} -- "
            f"`stage`/`axis`/`trial`/support are DERIVED readings, not roles "
            f"(project-layout.md § 1.4a)")
    directory = Path(directory)
    of = os.path.relpath(Path(root).resolve(), directory.resolve())
    return (directory / FILENAME,
            json_text({"schema": SCHEMA, "role": role, "of": of}))


def write(directory, *, role: str, root) -> Path:
    """Stamp *directory* (:func:`record`), now."""
    target, text = record(directory, role=role, root=root)
    write_bytes(target, text.encode("utf-8"))
    return target


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


# `container_or_run(directory)` stood here until 2026-10-04 -- § 1.4a's
# table, which read a root's `shape` out of `task.json` raw.  This floor
# reads no description (`execution/architecture.md` § 2.1, § 3.2): a folder's
# place is the run door's, `runs.place_of`, which reads the description
# through its one reader, `read_task` (plan B14).


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
    would have adopted whatever it found.
    """
    from .task import FILENAME as TASK_FILENAME

    directory = Path(directory)
    if (directory / TASK_FILENAME).is_file():
        return directory
    said = read(directory)
    if said is None:
        return None
    root = (directory / said.of).resolve()
    return root if (root / TASK_FILENAME).is_file() else None
