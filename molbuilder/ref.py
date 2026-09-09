"""One address for a calculation's files, and the three verbs over it.

**Module:** floor 1, and **stdlib-only** but for its two floor-1 siblings
(`paths`, `runfiles`).  The monitor ships beside a job
(``runwrap.MONITOR_COMPANIONS``) and runs under the JOB's python; a path
module a running job cannot import is a path module the job works around.

**Contract:** [`plans/plan.md`](?doc=plans/plan.md) § 5l — *the paths STANDARD:
one address, three verbs.*  § 5l.1 gives the address, § 5l.1a settles how ⑤
renders in the two shapes, § 5l.2 gives the verbs and says why there is no
fourth.  [`project-layout.md`](?doc=execution/project-layout.md) § 2.6 remains
the authority on the tree; **this module serves that hierarchy and never
invents a level of its own.**

WHY THIS EXISTS.  § 5k's rule -- *for every name it composes, the framework
owns the search* -- was right, and it was applied by adding a door for whatever
question a call site happened to ask.  The bill came due the same day: four
APIs answering *which files are this rung's*, three answering *what is this
file called*, sixteen across three modules answering *where does this live*,
and two answering *what stage is this file's* which were **measured to
disagree**.  ~40 public functions, five of them added in a single day to fit
one call site each.

    THE RULING (user, 2026-09-08).  "The API should be rigid standard, but
    accommodating for a certain flexibility -- but it cannot allow for any
    random cases.  All real needs, if they need to differentiate and build in
    a hierarchical system, have to follow the hierarchy of that system to
    start with.  They may have their own labels or design systems, but that's
    it."

So the direction of fit reverses.  A use that does not fit the standard is a
use that changes -- and the bounded flexibility is a catalogue ROW and its
FIELDS, never a new function.

WHAT IS HERE, AND WHAT IS NEXT DOOR.  This is layer 2 (the address) and layer 5
(the verbs) of § 5l.5's six.  The catalogue and the name grammar are
`runfiles`; the layout is `paths`.  This module composes those two and adds
nothing of its own to either -- if a rule about a NAME or a DIRECTORY appears
below, it is in the wrong file.

    catalogue   runfiles.WRITTEN / QUALIFIERS / FIELDS / roles()
    address     Ref                                          <- here
    grammar     runfiles.compose / parse
    layout      paths.Shape / attempt_name / bench_container
    verbs       compose / find / parse                       <- here

WHAT IT DELIBERATELY DOES NOT DO.  It does not read a description and it does
not guess a shape: `engines/stages.md` § 6.7 makes the shape a required field,
so every verb takes one.  It owns ③-⑤ only -- `projects.py` owns ① project and
② topic, and a second owner for those is the duplication this standard exists
to prevent, which is why an address is relative to a CALCULATION root that the
caller supplies.

THE BROWSER IS NOT A FOURTH VERB.  48k lines of JS cannot import this, so *one
address* would be false the moment a page needed a path.  The rule is the one
M5 applied: a surface ASKS, never composes.  Four browser re-implementations
were deleted on that basis; a fifth is a § 5l violation, not a JS problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import runfiles as _rf
from .paths import (Shape, attempt_name, attempt_index, attempts_in,
                    bench_container, bench_containers_in, trial_name,
                    trial_point, trials_in)

__all__ = ["Ref", "AddressError", "compose", "find", "parse"]


class AddressError(ValueError):
    """An address that names no place in § 2.6's tree.

    Distinct from :class:`runfiles.RunFileError`, which is about a NAME.  This
    is about a coordinate: a role outside the catalogue, an attempt in a shape
    that has none, a bench point that is not a qualifier on a stage.
    """


# ══ THE ADDRESS ════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Ref:
    """Where one file (or one directory) sits, in § 2.6's coordinates.

    § 2.6 numbers the tree ① project → ② topic → ③ calculation → ④ stage →
    ⑤ attempt, and says the benchmark rows carry **no number** -- they are
    *"nested containers inside a stage (④), not levels of the tree."*  This
    record holds ③'s label through ⑤, and the caller holds ①② as the root.

    **`label` is the FILENAME's label and is NOT the folder name.**  § 2.6
    row ③: the calculation folder is *"whatever the user types"* and *"the
    folder is not derived"* -- what makes it a calculation is the `task.json`
    inside it.  The two are free to differ, and a door that assumes otherwise
    is wrong on a folder someone renamed.

    **An absent coordinate is a STATEMENT, never a default.**  ``stage=None``
    means *this file crosses rungs*; ``attempt=None`` means *not
    attempt-scoped*; ``role=None`` addresses the DIRECTORY rather than a file.
    That is already `runfiles`' rule for ``stage`` -- it refuses ``""`` rather
    than reading it as None -- and here it is the rule on every axis.

    **`bench` is a qualifier on ④, not a sixth level.**  A trial is a stage's
    sub-container; inside it an attempt is ⑤ exactly as anywhere else.

    **`counters` and `fields` stay separate, and that is § 6.3's rule.**  A
    hyphen announces a COUNTER and an underscore a NAME, which is what lets a
    name be read back at all.  Collapsing them into one dictionary would lose
    the separator rule and the declared ordering that gives one name one
    spelling.

    **⑤ and the wrapper's `run` counter are two coordinates** (§ 5l.1a,
    measured): the hierarchy tells attempts apart by DIRECTORY and the wrapper
    indexes inside it starting again at 0, so ``run-2/x-run0.out`` is attempt
    2, wrapper index 0.  Flat has no attempt directory, and there the wrapper's
    index is the only thing that tells them apart -- so in flat the two ARE one
    slot, and :meth:`checked_for` refuses an address that fills it twice.
    """

    label: str
    #: A role from :func:`runfiles.roles`, or None to address the DIRECTORY.
    role: Optional[str] = None
    #: ④, the stage token ``<NN>_<name>``.  None means the file crosses rungs.
    stage: Optional[str] = None
    #: The trial POINT (``G1K4C6``), a qualifier on ④.  None means not a trial.
    bench: Optional[str] = None
    #: ⑤.  Only the hierarchy has one; flat spells it ``run`` (§ 5l.1a).
    attempt: Optional[int] = None
    #: The declared numeric qualifiers, ``(keyword, n)`` in declared order.
    counters: "tuple[tuple[str, int], ...]" = ()
    #: The role template's own keys -- the bounded flexibility (§ 5l.1).
    fields: "tuple[tuple[str, str], ...]" = ()

    def __post_init__(self) -> None:
        if not self.label:
            raise AddressError(
                "an address needs a label -- it is the FILENAME's label, "
                "which § 2.6 row ③ says is not the folder name")
        if self.stage == "" or self.bench == "":
            raise AddressError(
                "an empty stage or bench is not 'no stage' -- pass None, "
                "which STATES that the file crosses rungs (§ 5l.1)")
        if self.role is not None and self.role not in _rf.roles():
            raise AddressError(
                f"role {self.role!r} is not in the catalogue. Extensibility "
                f"is a catalogue ROW and its fields, never a spelling at a "
                f"call site (§ 5l.2). Declared: "
                f"{', '.join(sorted(_rf.roles()))}")
        if self.attempt is not None and self.attempt < 0:
            raise AddressError(
                f"attempt {self.attempt} is negative; attempts count from "
                f"{_rf.FIRST_ATTEMPT} and are not padded (§ 4.3)")
        for kw, _n in self.counters:
            if kw not in _rf.QUALIFIERS:
                raise AddressError(
                    f"{kw!r} is not a declared counter. § 6.3 gives the "
                    f"hyphen one meaning and runfiles.QUALIFIERS is the list: "
                    f"{', '.join(_rf.QUALIFIERS)}")
        for name, _v in self.fields:
            if name not in _rf.FIELDS:
                raise AddressError(
                    f"{name!r} is not a declared field. Its SHAPE lives once "
                    f"in runfiles.FIELDS: {', '.join(sorted(_rf.FIELDS))}")

    # -- reading one coordinate back ----------------------------------- #

    @property
    def run(self) -> Optional[int]:
        """The wrapper's index in the FILENAME, or None when it carries none.

        Not ⑤.  See the class docstring: in the hierarchy these are different
        numbers, and the one place they agree is flat.
        """
        return dict(self.counters).get("run")

    def checked_for(self, shape: "Shape") -> "Ref":
        """This address, checked against *shape*, or ``AddressError``.

        **The address itself is shape-independent** -- ⑤ is always
        :attr:`attempt`, in both layouts, so a caller builds one `Ref` and
        either shape renders it.  What differs is only the SPELLING, and that
        belongs to the verbs: the hierarchy renders ⑤ as a ``run-<n>``
        directory, flat renders it as the wrapper's filename index
        (`project-layout.md` § 1.5a).

        So the only thing to check is the one collision that spelling creates.
        In flat, ⑤ and the ``run`` counter are the same slot in the same name,
        and an address carrying both is two numbers claiming one coordinate --
        the ``_stage_state(label, stage, out_glob)`` fault § 5l.3 removes.
        Refused rather than resolved: silently preferring one would make the
        round trip lie about the file it came from.
        """
        if (not shape.keeps_attempts_as_directories
                and self.attempt is not None and self.run is not None):
            raise AddressError(
                f"flat spells ⑤ with the wrapper's own index "
                f"(`project-layout.md` § 1.5a), so attempt={self.attempt} and "
                f"run={self.run} are two numbers for one coordinate. Drop one")
        return self


# ══ VERB 1 — COMPOSE ═══════════════════════════════════════════════════════

def directory(root, shape: "Shape", ref: "Ref") -> Path:
    """Where *ref* lives -- the layout half of :func:`compose`.

    Public because a caller that wants the DIRECTORY wants a path and not a
    file in it: ``compose`` with ``role=None`` answers the same thing and this
    is the name that says so.  Every rule it applies belongs to `paths`; this
    is the order they compose in, which is the only thing § 2.6 adds.
    """
    ref = ref.checked_for(shape)
    d = Path(root)
    if ref.bench is not None:
        # A trial lives INSIDE the stage's bench container, and the container
        # is where the sweep's own record sits -- one rule, so "record here,
        # trials there" cannot come back on one side only.
        d = d / bench_container(shape, ref.stage or "") / trial_name(ref.bench)
    else:
        sd = shape.stage_dir(ref.stage) if ref.stage else "."
        if sd != ".":
            d = d / sd
    if ref.attempt is not None and shape.keeps_attempts_as_directories:
        d = d / attempt_name(ref.attempt)
    return d


def compose(root, shape: "Shape", ref: "Ref") -> Path:
    """Full coordinates → the one path, directory and filename together.

    **Returning a whole path is deliberate** (§ 5l.2): a caller that wants a
    file wants a path, and the three-call detour -- stage directory, then
    attempt directory, then a filename -- is exactly where a caller starts
    joining strings.

    With ``role=None`` this is :func:`directory`.  The filename half is
    `runfiles.compose`, which refuses a missing field and refuses a value that
    does not match the field's declared shape, because both would produce a
    name :func:`parse` cannot read back.
    """
    ref = ref.checked_for(shape)
    d = directory(root, shape, ref)
    if ref.role is None:
        return d
    counters = dict(ref.counters)
    if not shape.keeps_attempts_as_directories and ref.attempt is not None:
        # The one place ⑤ crosses into the name grammar. `checked_for` has
        # already refused an address that says it twice.
        counters["run"] = ref.attempt
    return d / _rf.compose(ref.label, ref.role, ref.stage,
                           **counters, **dict(ref.fields))


# ══ VERB 2 — FIND ══════════════════════════════════════════════════════════

#: What the search walks when a coordinate is left open.  ``None`` in a
#: :class:`Ref` is a STATEMENT ("not attempt-scoped"); an omitted keyword HERE
#: is a question ("any attempt").  They are different, which is why `find`
#: takes keywords and not a Ref.
_ANY = object()


def _stage_dirs(root: Path, shape: "Shape") -> "list[Optional[str]]":
    """Every stage token with a directory under *root*, plus None.

    Hierarchical only -- in flat the stage is in the FILENAME, so there is
    nothing on disk to enumerate and `runfiles.find(stage=)` answers it.  Reads
    what is there rather than a description: this module may not import one
    (`engines/stages.md` § 6.7 makes the shape a field, and the ladder lives
    two floors up).
    """
    if not shape.keeps_attempts_as_directories:
        return [None]
    out: "list[Optional[str]]" = [None]
    try:
        for d in sorted(root.iterdir()):
            if d.is_dir() and _rf.is_stage_token(d.name):
                out.append(d.name)
    except OSError:
        pass
    return out


def find(root, shape: "Shape", label: str, *,
         role=_ANY, stage=_ANY, bench=_ANY, attempt=_ANY,
         run=_ANY) -> "list[tuple[Path, Ref]]":
    """Partial coordinates → every match, ordered.

    An omitted keyword means **any**; passing ``None`` means the coordinate is
    ABSENT -- ``stage=None`` finds the files that cross rungs and nothing else,
    where omitting ``stage`` finds those and every rung's too.  That
    distinction is why this takes keywords rather than a :class:`Ref`, in which
    ``None`` is a statement.

    Ordered by path, so a caller that wants the latest attempt takes the last
    and one that wants them all walks in order.

    **The reading is :func:`parse`'s**, which is `runfiles.parse`'s, not a
    second reader: a name this module cannot read is not ours and is left out.
    That is what keeps a foreign file sitting in the same directory from being
    reported as an attempt -- the defect that made ``attempt_concluded`` crash
    on a person's own ``my.relaxation.fdf``.
    """
    root = Path(root)
    hits: "list[tuple[Path, Ref]]" = []
    for d in _search_dirs(root, shape, stage, bench, attempt):
        try:
            entries = sorted(p for p in d.iterdir() if p.is_file())
        except OSError:
            continue
        for p in entries:
            got = parse(root, shape, p, label)
            if got is None:
                continue
            if role is not _ANY and got.role != role:
                continue
            if stage is not _ANY and got.stage != stage:
                continue
            if bench is not _ANY and got.bench != bench:
                continue
            if attempt is not _ANY and got.attempt != attempt:
                continue
            if run is not _ANY and got.run != run:
                continue
            hits.append((p, got))
    return sorted(hits, key=lambda t: str(t[0]))


def _search_dirs(root: Path, shape: "Shape",
                 stage, bench, attempt) -> "list[Path]":
    """Every directory a match could sit in, derived from the LAYOUT.

    Derived and not globbed: the candidates are composed by the same rules
    :func:`compose` uses, so a directory this walk visits is a directory an
    address can name.  A glob would find `bench-` and `run-` shaped names
    anywhere and call them coordinates, which is how a level gets invented at a
    call site -- the thing N1's guard exists to catch.
    """
    out: "list[Path]" = []
    stages = ([stage] if stage is not _ANY and stage is not None
              else _stage_dirs(root, shape) if stage is not _ANY
              else _stage_dirs(root, shape))
    if stage is None:
        stages = [None]
    for tok in stages:
        containers: "list[Path]" = []
        if bench is not None:
            # In flat the token is IN the container's own name, so a search
            # with no stage in hand cannot compose it -- it asks `paths` for
            # the containers that are there. `bench_containers_in` is
            # `bench_container`'s search half, and exists because this was the
            # first caller to need it.
            if tok is None and not shape.keeps_attempts_as_directories:
                bcs = [root / rel for rel, _t in bench_containers_in(root, shape)]
            else:
                bcs = [root / bench_container(shape, tok or "")]
            for bc in bcs:
                points = ([bench] if bench is not _ANY else trials_in(bc))
                containers += [bc / trial_name(pt) for pt in points]
        if bench is _ANY or bench is None:
            sd = shape.stage_dir(tok) if tok else "."
            containers.append(root if sd == "." else root / sd)
        for c in containers:
            if not shape.keeps_attempts_as_directories:
                out.append(c)
                continue
            if attempt is not _ANY and attempt is not None:
                out.append(c / attempt_name(attempt))
            elif attempt is None:
                out.append(c)
            else:
                out.append(c)
                out += [c / attempt_name(n) for n in attempts_in(c)]
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


# ══ VERB 3 — PARSE ═════════════════════════════════════════════════════════

def parse(root, shape: "Shape", path, label: str,
          roles: "tuple[str, ...]" = ()) -> "Optional[Ref]":
    """A path → its coordinates, or None when it is not ours.

    **None is the answer for a foreign file, never an exception.**  A caller
    holding a directory holds other people's files too, and § 5l.3 is explicit
    that a foreign stem is not in the address space: the caller should be told
    that, not served by loosening the grammar to accept it.

    The FILENAME is read by `runfiles.parse`, which needs the label because a
    role may contain underscores and so may a label, so nothing can find the
    boundary by looking at the string alone.  The DIRECTORY is read by
    `paths`' readers -- :func:`paths.attempt_index` and
    :func:`paths.trial_point` -- so one rule composes each and one reads it.

    Where the two disagree, the FILENAME wins on ④ and the DIRECTORY wins on
    ⑤: the name is what `runfiles` composed, and the attempt directory is the
    hierarchy's own answer, which the wrapper (indexing from 0 inside it)
    cannot see.
    """
    root, p = Path(root), Path(path)
    got = _rf.parse(p.name, label, roles=roles)
    if got is None:
        return None
    try:
        rel = p.parent.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None

    parts = list(rel.parts)
    attempt = None
    if parts and shape.keeps_attempts_as_directories:
        n = attempt_index(parts[-1])
        if n is not None:
            attempt = n
            parts.pop()
    bench = None
    if parts:
        pt = trial_point(parts[-1])
        if pt is not None:
            bench = pt
            parts.pop()
            # Its container is the stage's, and the stage is the filename's --
            # so the remaining parts are checked, not read.
            if parts and parts[-1] == "bench":
                parts.pop()

    counters = got.counters
    if not shape.keeps_attempts_as_directories:
        # Flat spells ⑤ as the wrapper's index (§ 5l.1a), so the address says
        # so rather than reporting a counter and a missing attempt.
        attempt = dict(counters).get("run")
        counters = tuple(c for c in counters if c[0] != "run")
    ref = Ref(label=got.label, role=got.role, stage=got.stage, bench=bench,
              attempt=attempt, counters=counters, fields=got.fields)
    # The address must name the path it came from, or it is not this path's
    # address.  This is the round trip § 5l.2 rests on, checked rather than
    # assumed -- a layout rule that drifts fails HERE and not at a call site.
    return ref if compose(root, shape, ref) == root / rel / p.name else None
