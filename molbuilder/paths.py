"""Where a calculation's files live — the layout, and how to FIND it.

**Module:** floor 1, and **stdlib-only**, which is the property the whole
design rests on.  The monitor ships beside a job
(``runwrap.MONITOR_COMPANIONS``) and runs under the JOB's python with no
molbuilder installed; ``config_dir.py`` already travels for that reason.  A
path module a running job cannot import is a path module the job works
around.

**Contract:** [`project-layout.md`](?doc=execution/project-layout.md) § 4.5 —
*for every name it composes, the framework owns the search.*  The run-file
GRAMMAR stays in `runfiles` (§ 2.2a); this owns the TREE and the finders.

*(Was ``jobset/shape.py``, floor 4, until 2026-09-08.  It sat there for one
import — ``from ..task import SHAPES``, a two-element tuple of literals, held
by floor 2 and read by this module alone.  Moving that constant here inverts
one line and drops the whole naming-and-layout surface to a floor a shipped
job can reach.)*

Also [`engines/stages.md`](?doc=engines/stages.md) § 6.7 (*required, never
inferred*; *"`prep` **reads** it; it does not decide it"*) and
[`job-contracts.md`](?doc=execution/job-contracts.md) § 6.3 (the names, in
both shapes).

WHAT IS HERE.  `Shape` — the two layouts and the questions a layout answers —
and the ATTEMPT, `run-<n>`, with its composer, its reader and its finder.  The
run-file GRAMMAR is next door in `runfiles`; this is the tree those names sit
in.

WHY SHAPE IS AN OBJECT AND NOT AN ``if shape ==`` IN FOUR MODULES.  The two
layouts do not differ by one branch — they differ in **what a stage is told
apart BY**:

    hierarchical   a PATH      01_coarse/run-0/…      every stage its own tree
    flat           a FILENAME  bdt_01_coarse-run0.out  one directory, all of them

Every layer built for the hierarchy assumes the distinction lives in the path,
because for the hierarchy it does. Handed a flat calculation those layers do not
fail — they answer about **the wrong stage**, or about all of them at once, and
say nothing. That asymmetry is the whole reason this is a type: a caller asks
*where does this stage's state live* and gets an answer that is correct in both,
instead of asking *which directory* and being right in one.

WHAT IT DELIBERATELY DOES NOT DO.  It does not read a description, and it does
not guess: `engines/stages.md` § 6.7 makes the shape a required field, and a
default here would be the inference that section refuses. The **surfaces**
resolve it — they are the ones holding a bundle directory — and hand it down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: The two layouts, and the vocabulary both the description and the tree
#: check against.  It lives HERE, not in ``task``, so this module imports
#: nothing above floor 1 -- ``task`` imports it back for its own validation.
#: A description DECLARES the shape and nothing infers it
#: (`engines/stages.md` § 6.7).
SHAPES = ("flat", "hierarchical")


@dataclass(frozen=True)
class Shape:
    """One of the two layouts, and the questions a layout can answer.

    Construct with :meth:`named`, which refuses anything that is not one of
    ``project-layout.md § 1``'s two.
    """

    name: str

    @classmethod
    def named(cls, name: str) -> "Shape":
        """The shape called ``name``, or ``ValueError`` naming both."""
        if name not in SHAPES:
            raise ValueError(
                f"shape {name!r} is not one of {' / '.join(SHAPES)}. It is a "
                f"required field of the description and is never inferred "
                f"(engines/stages.md § 6.7)")
        return cls(name)

    # -- the two questions a layout answers ---------------------------- #

    def stage_dir(self, token: str) -> str:
        """Where this stage's work happens, **relative to the bundle root**.

        Hierarchical gives it a directory of its own (``01_coarse``); flat is
        **depth 1** (`project-layout.md` § 1) and everything happens in the
        bundle root, which is ``"."`` — a real path a caller can join, so no
        caller needs an ``if`` to handle "no directory".
        """
        return token if self.name == "hierarchical" else "."

    def stage_glob(self, token: str, label: str) -> str:
        """A glob matching the files **this stage** produced, inside
        :meth:`stage_dir`.

        This is the half that does not exist in the hierarchy's world view.
        There, the directory has already selected the stage, so anything in it
        belongs to it. In flat, one directory holds every stage, and they are
        told apart by the deck's token carried in each filename
        (`job-contracts.md § 6.3`) — so the *name* does the selecting.

        Returning a glob rather than a boolean is what lets one caller write
        one line: ``(base / sh.stage_dir(t)).glob(sh.stage_glob(t, label))``
        is correct in both, and there is no second code path to keep in step.
        """
        return "*" if self.name == "hierarchical" else f"{label}_{token}*"

    @property
    def keeps_attempts_as_directories(self) -> bool:
        """Whether a re-run makes a **directory** (``run-1``) or an index.

        `project-layout.md` § 1: hierarchical separates attempts by directory,
        flat by an **output index** (``-run0.out``) that the wrapper writes.
        So the whole attempt layer — ``prepare_attempt``, ``latest_attempt``,
        ``run.json`` — is hierarchical's, and in flat there is nothing to open:
        *"continuing: free — the next stage finds them lying there."*
        """
        return self.name == "hierarchical"

    def __str__(self) -> str:
        return self.name



# ══ THE ATTEMPT ════════════════════════════════════════════════════════════
#
# ``run-<n>``, the directory one try of a stage or a trial runs in
# (`project-layout.md` § 1.5 — immutable once it has run, which is why a
# re-run opens the next rather than landing on top of one).
#
# IT HAD NO COMPOSER.  `f"run-{n}"` was written NINE times across four
# modules, and the regex that reads it back sat in `materialize` and was
# imported across into `prep` — a name spelled ten ways for a rule stated
# once in the document.  One of those nine was added on 2026-09-08 by the
# commit that gave the TRIAL directory a home, which is the habit exactly:
# a caller reaches for an f-string because there is nothing to ask.

#: What an attempt directory starts with.  One home, so the composer and the
#: reader below cannot drift, and neither can a caller.
ATTEMPT_PREFIX = "run-"


def attempt_name(n: int) -> str:
    """What attempt *n*'s directory is called.

    Separate from :func:`attempt_dir` because a caller that is LISTING
    attempts wants the name and not a path -- `runstatus` prints them in a
    row, and composing a path to take ``.name`` off it again is the kind of
    detour that sends the next person back to an f-string.
    """
    return f"{ATTEMPT_PREFIX}{int(n)}"


def attempt_dir(container, n: int) -> Path:
    """The directory attempt *n* of this stage or trial runs in."""
    return Path(container) / attempt_name(n)


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

    The finder half.  `materialize.attempts` is the caller that had it, and
    kept its own regex to do it.
    """
    c = Path(container)
    try:
        found = [attempt_index(d.name) for d in c.iterdir() if d.is_dir()]
    except OSError:
        return []
    return sorted(n for n in found if n is not None)


__all__ = [
    # the layout
    "SHAPES", "Shape",
    # the attempt -- composer, reader, finder (§ 4.5's pairing)
    "ATTEMPT_PREFIX", "attempt_name", "attempt_dir",
    "attempt_index", "attempts_in",
]
