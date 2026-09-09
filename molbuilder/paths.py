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



# ══ THE BENCH CONTAINER AND THE TRIAL ══════════════════════════════════════
#
# `project-layout.md` § 2.6: the benchmark rows carry NO circled number --
# they are "nested containers inside a stage (④), not levels of the tree."
# So a trial is a QUALIFIER on ④, and both names below are layout, which is
# why they live here beside `stage_dir` rather than in the module that
# happened to need them first.
#
# THEY MOVED DOWN 2026-09-09.  Both rules were `jobset/materialize.py`'s,
# floor 4, and the address layer is floor 1 -- a rule the framework cannot
# reach is a rule its callers re-spell, which is § 4.5's whole subject.
# `materialize.bench_container` and `materialize.job_dir_name` still exist
# and still answer; they now ask these.

#: What a TRIAL directory starts with.  Dash-joined, where the container is
#: underscore-joined -- § 6.3's separator rule, and what keeps
#: ``bench-G1K4C6`` (a trial) apart from ``bench_01_coarse`` (a container).
TRIAL_PREFIX = "bench-"


def trial_name(point: str) -> str:
    """What the trial for *point* is called -- ``bench-<point>``.

    `job-contracts.md` § 6.3 is the authority: ``bench-`` plus the coordinate
    as ONE qualifier.  The composer half of the pair; :func:`trials_in` is
    the finder.
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


def bench_container(shape: "Shape", token: str = "") -> str:
    """Where a stage's bench state lives, **relative to the calculation root**.

    ``<NN>_<stage>/bench`` in the hierarchy; ``bench_<NN>_<stage>`` at the root
    of a FLAT calculation; bare ``bench`` for a stageless one.
    `job-contracts.md` § 6.3: *"benchmark | bench/ inside the stage it
    measures"* -- in flat there IS no stage directory to sit inside, so the
    token qualifies the container's own name instead.

    **The token qualifies it in flat because it once did not**, and two flat
    stages' benchmarks then shared one root ``bench/``: each prep overwrote the
    other's job-set, plan and verdict (2026-08-12 plan A5).

    This is the ONE spelling of that rule.  It sat in two places until
    2026-08-13 and the two disagreed in BOTH non-hierarchical layouts, so
    `launch` launched trials in directories the underway-ask never looked at.
    """
    sd = shape.stage_dir(token) if token else "."
    if sd == ".":
        return f"bench_{token}" if token else "bench"
    return f"{sd}/bench"


def bench_containers_in(root, shape: "Optional[Shape]" = None
                        ) -> "list[tuple[str, Optional[str]]]":
    """Every bench container under *root*, as ``(relative name, stage token)``.

    ``shape=None`` means **either layout** -- for a caller that has no
    description to read one from.  That is not the inference
    `engines/stages.md` § 6.7 forbids: that rule is about a description
    DECLARING its shape, and this is a search saying it does not know which
    tree it is walking.  The declared containers of the two layouts cannot
    collide (``<NN>_<stage>/bench`` is one level down; ``bench_<NN>_<stage>``
    is at the root), so the union is exact rather than a guess.

    The SEARCH half of :func:`bench_container`, which § 4.5 requires and which
    did not exist -- a caller looking for *the benchmarks in this calculation*
    had to know that the hierarchy hides them one level down inside each rung
    while flat qualifies their own name.  That asymmetry is what made
    ``bench_container`` need three shapes in one docstring.

    The stage token is returned rather than re-derived: in flat it is IN the
    container's name (``bench_01_coarse``) and in the hierarchy it is the
    PARENT directory, so a caller reading either spelling itself would be
    writing this function again with one of the two arms missing -- which is
    the fault (A5, 2026-08-12) that let two flat stages share one ``bench/``.
    """
    r = Path(root)
    out: "list[tuple[str, Optional[str]]]" = []
    if shape is None:
        seen: set = set()
        for name in SHAPES:
            for row in bench_containers_in(r, Shape.named(name)):
                if row[0] not in seen:
                    seen.add(row[0])
                    out.append(row)
        return sorted(out)
    try:
        entries = sorted(r.iterdir())
    except OSError:
        return out
    if shape.keeps_attempts_as_directories:
        if (r / "bench").is_dir():
            out.append(("bench", None))
        for d in entries:
            if d.is_dir() and (d / "bench").is_dir():
                out.append((f"{d.name}/bench", d.name))
        return out
    for d in entries:
        if not d.is_dir():
            continue
        if d.name == "bench":
            out.append(("bench", None))
        elif d.name.startswith("bench_"):
            out.append((d.name, d.name[len("bench_"):] or None))
    return out


def trials_in(container) -> "list[str]":
    """Every trial POINT present in *container*, sorted -- :func:`trial_name`'s
    search.

    Returns the points, not the paths: a caller that wants a path composes one,
    and a caller listing a sweep wants the coordinates.  `materialize.trials_in`
    returns paths for its own callers and is the other half of the same pair.
    """
    c = Path(container)
    try:
        found = [trial_point(d.name) for d in c.iterdir() if d.is_dir()]
    except OSError:
        return []
    return sorted(p for p in found if p is not None)


__all__ = [
    # the layout
    "SHAPES", "Shape",
    # the bench container and the trial -- ④'s qualifier, not a level
    "TRIAL_PREFIX", "trial_name", "trial_point", "bench_container",
    "bench_containers_in", "trials_in",
    # the attempt -- composer, reader, finder (§ 4.5's pairing)
    "ATTEMPT_PREFIX", "attempt_name", "attempt_dir",
    "attempt_index", "attempts_in",
]
