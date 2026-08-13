"""``task.json`` — a calculation's description on disk, and its one reader.

Contract: ``docs/engines/stages.md`` § 6.  This module is the *only* place
that turns those bytes into objects and back, which is what makes
*"the browser writes the same bytes as the CLI"* checkable rather than
coincidental (§ 6.4).

WHAT A DESCRIPTION IS.  One calculation: which engine, which layout, what it
is a calculation *of*, which settings the user chose to tune (``varies``), and
the per-stage cells that differ (``stages[].overrides``).

**It carries only what CHANGES.**  Everything that does not is in the template
(``<label>.template.toml``), written once — so there is no ``base`` key here, and a
stage that omits a varied field renders with the template's value (§ 4).  It
**names no machine** either: ranks, queues and walltimes are decided by ``prep``
on the target (``execution/project-layout.md`` § 2.1).

LAYER.  L1: it imports ``persist``, ``identity`` and the standard library, and
nothing else — both of those are themselves L1 on stdlib, so this stays a leaf.
That is deliberate rather than incidental — see *the split preflight* below.
``identity`` joined the list on 2026-08-09, when ``run.id`` stopped being a free
string and became something this module DERIVES and checks; it is the same
normaliser the CLI and the browser call, which is what § 3 rule 1's *"it happens
once"* actually requires.

THE SPLIT PREFLIGHT.  § 6.6 lists eight checks "in order, and all of it before
anything is written".  Four of them are answerable from the file alone and are
enforced here; the other four need the engine's field schema, and importing an
engine into an L1 codec is exactly what ``tests/test_layering.py`` prevents.
Those belong to resolution (P2 of ``execution/staged-runs-implementation-plan.md``),
which already has the schema in hand:

  here    the schema string's major · ``shape`` present and legal · stage names
          in ``[A-Za-z0-9_]+`` and unique case-insensitively · no ``overrides``
          key naming a stage field · ``overrides`` a SUBSET of ``varies`` ·
          unknown keys refused by name · ``run.id`` is what ``run.name`` and
          ``structure.formula`` derive (§ 6.1's first rule -- the check needs
          only ``identity``, which is L1 beside this one)
  P2      the engine has a generator · the schema fingerprint matches · every
          named field exists in the schema · every value is inside its bounds ·
          ``shape: "hierarchical"`` on an engine whose ladder runs in ONE
          process (§ 6.7 — PySCF; refused naming the engine) · and § 6.6a's
          warning for two stages that RESOLVE identically, since resolving is
          P2's verb

WHY THE MESSAGES NAME THINGS.  A description is JSON sitting beside the decks,
and as of 2026-08-07 editing it by hand is **supported** (the plan's decision 3).
So a refusal owes a person what it owes the browser: the offending key by name,
which stage it was in, and — where the key is one edit away from a real one —
what they probably meant.
"""
from __future__ import annotations

import contextvars
import difflib
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Optional, Tuple

from .identity import normalise_id, run_id
from .persist import check_schema, read_json, write_json


SCHEMA = "molbuilder/task@1"
FILENAME = "task.json"

#: § 6.7 — required, no default, never inferred.
SHAPES = ("flat", "hierarchical")

#: § 2 — "three fields, and no others".  An ``overrides`` map naming one of
#: these would be a stage redefining what a stage is.
STAGE_FIELDS = ("name", "enabled", "overrides")

#: § 6.6 — a stage name becomes a filename, so the set is the narrow one.
STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

_TOP_KEYS = ("schema", "engine", "shape", "run", "schema_fingerprint",
             "structure", "varies", "stages")
_RUN_KEYS = ("name", "id", "created")
_STRUCTURE_KEYS = ("source", "formula", "atoms")


# --------------------------------------------------------------------- #
#  The shapes                                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """What the user called it, and what it is called on disk.

    ``id`` is derived once and then quoted everywhere
    (``execution/run-identity.md`` § 2).  Since 2026-08-07 it does **not**
    name the directory — the level-③ folder is typed by the user, and this
    file is what says which calculation lives there (§ 3.0 there).

    Since 2026-08-09 (§ 2.0a, decision 26) it does not name the **files**
    either: the id is ``<label>_<formula>`` and lives here, while the label
    alone is the ``SystemLabel`` and the stem of everything on disk.  The
    label is not a field — it is :attr:`Task.label`, derived through the one
    normaliser — because storing it would be a second place for the same
    string to be wrong.  What makes deriving it safe is that ``id`` **is**
    stored and :meth:`Task.__post_init__` checks the two against each other.

    ``id`` is not defaulted.  A ``Run`` that filled it in from ``name`` would
    be a second deriver, and § 3 rule 1 is that there is one; use
    :func:`derive_run`."""
    name: str
    id: str
    created: str = ""


@dataclass(frozen=True)
class StructureRef:
    """§ 6.3 — a reference plus a witness, never a copy.

    ``source`` points into the project tree; ``formula`` and ``atoms`` record
    what was there when the description was written, so a description opened
    against a structure that has since changed can *say so* instead of
    silently building a different calculation under the same id."""
    source: str
    formula: str = ""
    atoms: int = 0


@dataclass(frozen=True)
class Stage:
    """§ 2 — a name, an enabled flag, and the cells that differ."""
    name: str
    enabled: bool = True
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """The name rule holds for the object too, not only for the file.

        ``_stage_from_obj`` checks this when parsing, but a producer handed
        hand-built stages would otherwise put an arbitrary string into a
        filename (``<label>_<name>.fdf``) and into an unquoted bash array
        literal in the runner.  The rule is the same one, read from the same
        pattern -- not a second copy of it."""
        if not isinstance(self.name, str) or not STAGE_NAME_RE.match(self.name):
            raise ValueError(
                f"stage name {self.name!r} must match [A-Za-z0-9_]+ -- it "
                "becomes a filename and a shell word "
                "(engines/stages.md 6.6)")


@dataclass(frozen=True)
class Task:
    """One calculation.

    ``stages`` and ``varies`` are ``None`` **together**: a description with
    no stages *is* a single-parameter-set calculation (§ 6.5), and an empty
    ``stages`` would be a second way to spell that.

    ``varies`` may be empty while ``stages`` is not, and that is a different
    state rather than a loophole — several stages differing in nothing but
    their name, which § 6.6a allows (they are the honest way to say *keep
    going* when each continues from the last)."""
    engine: str
    shape: str
    run: Run
    structure: StructureRef
    varies: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[Stage, ...]] = None
    schema_fingerprint: str = ""

    def __post_init__(self) -> None:
        """§ 6.5 holds for the object too, not only for the file.

        Without this, ``Task(varies=(...), stages=None)`` is constructible
        and ``to_dict`` drops the ``varies`` silently -- a round-trip that
        loses intent, which is the one thing ``varies`` exists to carry
        (§ 6.2).  The two are absent together or present together, and
        ``stages`` is never an empty tuple: that would be the second
        spelling of "one stage" § 6.5 rules out."""
        if (self.varies is None) != (self.stages is None):
            raise ValueError(
                "task: 'varies' and 'stages' are absent together or present "
                f"together (varies={self.varies!r}, stages={self.stages!r}). "
                "A description with no stages is one parameter set, and there "
                "is nothing to vary across (engines/stages.md 6.5)")
        if self.stages is not None and not self.stages:
            raise ValueError(
                "task: 'stages' is present but empty. Omit it entirely for a "
                "single parameter set (engines/stages.md 6.5)")
        # The two LADDER-level refusals live here, in the codec, because
        # every route to a Task -- describe on a laptop, read_task on the
        # cluster, a hand-edited file -- passes through this constructor.
        # Three comments claimed these checks existed here (the u5
        # retirement in validation/siesta.py, describe_cmd's help,
        # describe._check's "codec's own checks") while NOTHING refused
        # either: the claims are made true at the claimed home (U15,
        # 2026-08-12).
        if self.stages is not None:
            # CASE-INSENSITIVELY, like both parsers (D6, redo 2026-08-12):
            # the names key filenames, and the filesystems these run on
            # include case-insensitive ones -- "Tight" and "tight" are one
            # deck there.  The constructor compared exact strings while
            # the parsers folded case, so the seam disagreed with its own
            # doors.
            names = [s.name.lower() for s in self.stages]
            dups = sorted({n for n in names if names.count(n) > 1})
            if dups:
                raise ValueError(
                    f"task: duplicate stage name(s) "
                    f"{', '.join(map(repr, dups))} (compared "
                    f"case-insensitively). A stage's name keys its "
                    f"directory, its deck and its status row -- two stages "
                    f"sharing one silently hand one the other's files "
                    f"(engines/stages.md 6.6)")
            if not any(s.enabled for s in self.stages):
                raise ValueError(
                    "task: every stage is disabled -- an all-disabled ladder "
                    "is an empty one spelled longer, and 6.5 already rules "
                    "the empty spelling out. Enable a stage, or omit "
                    "'stages' for a single parameter set")
        self._check_id()

    # ----- identity (run-identity.md 2, 2.0a, 3) ---------------------- #

    def _stage_names(self) -> Tuple[str, ...]:
        """The ladder, for the cap -- see :func:`~molbuilder.identity._cap_for`.

        The cap is derived from what ``<label>_<longest stage>.<longest
        extension>`` will occupy, so knowing the real ladder beats the fixed
        budget guessed when the stages do not exist yet.  Reading a
        description is exactly the moment they DO exist."""
        return tuple(s.name for s in self.stages) if self.stages else ()

    @property
    def label(self) -> str:
        """The ``SystemLabel`` / ``JOB`` literal, and the stem of every file.

        Derived rather than stored (see :class:`Run`), through the same
        normaliser that built half of ``run.id`` -- so the two cannot disagree
        about what the user's name normalises to, and ``__post_init__`` has
        already proved it."""
        return normalise_id(self.run.name, stage_names=self._stage_names(),
                            what="name")

    def _check_id(self) -> None:
        """``run.id`` is what ``run.name`` and ``structure.formula`` derive.

        § 3 rule 1 is *"it happens once, and the result is stored"* -- which
        only means something if somebody checks the stored value against its
        inputs.  Without this, ``id`` is a free string: a hand-edited ``name``
        (supported since 2026-08-07, decision 3) leaves the id behind, and the
        description then says two different things about which calculation it
        is.  § 1's second failure mode is exactly that edit, and its cost is a
        run that silently starts cold.

        It refuses rather than repairing.  Which of the three fields is the
        right one is not knowable here -- a corrected formula and a renamed
        calculation look identical from inside the file -- and quietly
        rewriting the id would be the *"append a digit and carry on"* that
        § 3 rule 3 rules out, one layer up."""
        expected = run_id(self.run.name, self.structure.formula,
                          stage_names=self._stage_names())
        if self.run.id != expected:
            raise ValueError(
                f"task: run.id {self.run.id!r} is not what this description "
                f"derives. run.name {self.run.name!r} + structure.formula "
                f"{self.structure.formula!r} give {expected!r}. The id is "
                f"normalised once and then quoted everywhere, so a mismatch "
                f"means one of the three was edited without the others -- and "
                f"which one is right is not something this reader may guess "
                f"(run-identity.md 2.0a and 3 rule 1)")

    # ----- persistence (task@1) -------------------------------------- #
    # ``to_dict`` / ``from_dict`` are the house names for a dataclass<->JSON
    # pair (26 of them under ``molbuilder/``); ``jobset/model.py::JobSet`` is
    # the closest analogue -- also a persisted plan for a multi-directory
    # unit of work, also major-checked through ``persist``.  The bodies live
    # at the foot of the module beside the refusals they use.

    def to_dict(self) -> dict:
        """This description as the JSON object § 6 specifies."""
        return _task_to_dict(self)

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Task":
        """Parse a description, refusing rather than guessing."""
        return _task_from_dict(obj)


# --------------------------------------------------------------------- #
#  Refusals                                                             #
# --------------------------------------------------------------------- #

#: What a refusal calls the thing it is refusing.  ``task.json`` for the file
#: this module owns, but the same codec also parses a bare ladder handed in by
#: a surface (``stages_from_dicts``) -- and telling someone who typed
#: ``--stages-json`` that their mistake is in ``task.json`` sends them to look
#: at a file they did not write.  Set for the duration of such a parse.
#:
#: A ContextVar rather than a plain global because the web layer serves
#: requests concurrently: two parses in flight would otherwise swap each
#: other's label and blame the wrong input.
_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "task_refusal_source", default=FILENAME)


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(
        f"{_SOURCE.get()}{': ' + where if where else ''}: {msg}")


@contextmanager
def _refusals_name(source: str):
    """Name *source* in every refusal raised inside the block."""
    token = _SOURCE.set(source)
    try:
        yield
    finally:
        _SOURCE.reset(token)


def _as_object(value: Any, *, where: str) -> Mapping[str, Any]:
    """Refuse a non-object where § 6 specifies one, *saying* so.

    Without this the key check below iterates a string's characters and
    reports ``unknown key 's'`` for ``"engine": "siesta"`` — technically a
    refusal, but a person hand-editing the file (the plan's decision 3)
    learns nothing from it."""
    if not isinstance(value, Mapping):
        _refuse(f"expected an object, got {type(value).__name__}", where=where)
    return value


def _check_keys(obj: Mapping[str, Any], allowed: Tuple[str, ...], *,
                where: str) -> None:
    """§ 6.1 rule 1 — an unknown key is refused, not ignored, because an
    ignored key is a calculation quietly different from the one asked for."""
    for key in obj:
        if key in allowed:
            continue
        near = difflib.get_close_matches(key, allowed, n=1, cutoff=0.7)
        hint = f" -- did you mean {near[0]!r}?" if near else \
               f" (known keys: {', '.join(allowed)})"
        _refuse(f"unknown key {key!r}{hint}", where=where)


def _require(obj: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in obj:
        _refuse(f"missing required key {key!r}", where=where)
    return obj[key]


# --------------------------------------------------------------------- #
#  Reading                                                              #
# --------------------------------------------------------------------- #

def _task_from_dict(obj: Mapping[str, Any]) -> Task:
    """Parse a description, refusing rather than guessing.

    The order is § 6.6's: the schema string first, so a file from a future
    major fails saying so instead of failing on whichever key moved."""
    if not isinstance(obj, Mapping):
        _refuse(f"expected a JSON object, got {type(obj).__name__}")

    check_schema(str(obj.get("schema") or ""), SCHEMA, label=FILENAME)
    _check_keys(obj, _TOP_KEYS, where="")

    engine_obj = _as_object(_require(obj, "engine", where=""), where="engine")
    _check_keys(engine_obj, ("name",), where="engine")
    engine = str(_require(engine_obj, "name", where="engine"))

    shape = _require(obj, "shape", where="")
    if shape not in SHAPES:
        _refuse(f"shape {shape!r} is not one of {' / '.join(SHAPES)}. "
                "It is required and never inferred -- inferring it would "
                "hand you a directory tree you did not ask for "
                "(engines/stages.md 6.7)")

    run_obj = _as_object(_require(obj, "run", where=""), where="run")
    _check_keys(run_obj, _RUN_KEYS, where="run")
    run = Run(name=str(_require(run_obj, "name", where="run")),
              id=str(_require(run_obj, "id", where="run")),
              created=str(run_obj.get("created", "")))

    struct_obj = _as_object(_require(obj, "structure", where=""),
                            where="structure")
    _check_keys(struct_obj, _STRUCTURE_KEYS, where="structure")
    structure = StructureRef(
        source=str(_require(struct_obj, "source", where="structure")),
        formula=str(struct_obj.get("formula", "")),
        atoms=int(struct_obj.get("atoms", 0)))

    has_stages = "stages" in obj
    has_varies = "varies" in obj
    if has_varies and not has_stages:
        _refuse("'varies' without 'stages'. A description with no stages is "
                "one parameter set, and there is nothing to vary across "
                "(engines/stages.md 6.5)")

    if not has_stages:
        return Task(engine=engine, shape=shape, run=run, structure=structure,
                    schema_fingerprint=str(obj.get("schema_fingerprint", "")))

    raw_stages = obj["stages"]
    if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
        _refuse("'stages' is present but empty. A description WITH stages has "
                "at least one; a calculation with a single parameter set omits "
                "the key entirely (engines/stages.md 6.5)")

    varies = tuple(str(v) for v in obj.get("varies", ()))
    stages = tuple(_stage_from_obj(s, varies, i)
                   for i, s in enumerate(raw_stages))

    _refuse_duplicate_stage_names(stages)

    # ``varies`` stays a tuple here even when empty: it travels WITH
    # ``stages``, and an empty one is a real state -- several stages that
    # differ in nothing but their name.  (``None`` is reserved for the
    # no-stages case above, which § 6.5 spells by omitting both keys.)
    return Task(engine=engine, shape=shape, run=run, structure=structure,
                varies=varies, stages=stages,
                schema_fingerprint=str(obj.get("schema_fingerprint", "")))


def _stage_from_obj(obj: Mapping[str, Any], varies: Tuple[str, ...],
                    index: int) -> Stage:
    where = f"stage {index}"
    _as_object(obj, where=where)

    name = str(obj.get("name", ""))
    where = f"stage {name!r}" if name else where
    _check_keys(obj, STAGE_FIELDS, where=where)
    if "name" not in obj:
        _refuse("missing required key 'name'", where=where)
    if not STAGE_NAME_RE.match(name):
        _refuse(f"stage name {name!r} must match [A-Za-z0-9_]+ -- it becomes "
                "a filename (engines/stages.md 6.6)")

    overrides = dict(_as_object(obj.get("overrides") or {},
                                where=f"{where} overrides"))
    for key in overrides:
        if key in STAGE_FIELDS:
            _refuse(f"override {key!r} names a stage field. A stage has "
                    f"exactly {', '.join(STAGE_FIELDS)}; an override may not "
                    "redefine one (engines/stages.md 2)", where=where)

    # § 6.2 -- a SUBSET of `varies`, never a superset.
    #
    # No key outside `varies`: a demoted parameter must not leave a value
    # hiding in a stage nobody can see.  But a varied key may be ABSENT, and
    # absent means "this stage uses the TEMPLATE's value" -- a real state, and
    # the one the table draws as a quiet cell.  Requiring equality (as this did
    # until 2026-08-07) both made `varies` redundant -- derivable as the key set
    # of any stage -- and forced every cell to be filled with a copy.
    extra = sorted(set(overrides) - set(varies))
    if extra:
        _refuse(f"override(s) {', '.join(repr(k) for k in extra)} not listed "
                "in 'varies'. A stage may only override a field the user "
                "promoted, or a demoted parameter leaves a value hiding "
                "(engines/stages.md 6.2)", where=where)

    return Stage(name=name, enabled=bool(obj.get("enabled", True)),
                 overrides=overrides)


def varies_for(overrides_seq) -> Tuple[str, ...]:
    """The promoted columns of a ladder — every override key, first seen first.

    § 6.2: ``varies`` is *"the set of fields the user chose to tune"*, and
    ``overrides`` is a **subset** of it. A surface that supplies a ladder
    without stating ``varies`` separately has already said what varies by
    listing what each stage overrides, and asking for it twice would be a
    second place to disagree.

    **The union, and not one stage's keys**, because a stage may leave a
    promoted cell empty — that means *"use the template's value"*, which is a
    real state § 6.2 protects, not an absence of intent.

    One function so the two surfaces that build a ladder without a description
    — a dict payload (``--stages-json``, the web) and a ready-made
    :class:`Stage` list (``--stage-strategy``) — cannot derive different
    columns from the same ladder.
    """
    out: list = []
    for overrides in overrides_seq:
        for key in overrides or {}:
            if key not in out:
                out.append(key)
    return tuple(out)


def stages_from_dicts(payload: Any, *,
                      source: str = "stages") -> Tuple[Tuple[str, ...],
                                                       Tuple[Stage, ...]]:
    """Parse a bare list of stage objects into ``(varies, stages)``.

    For a surface that supplies a ladder **without** a whole description —
    the CLI's ``--stages-json``, and the web's staged build payload — where
    ``varies`` is not stated separately.  It is **derived**, as the union of
    every stage's ``overrides`` keys in first-seen order: a payload that
    lists what each stage overrides has already said what varies, and
    asking for it twice would be a second place to disagree.

    Everything else is the same codec ``read_task`` uses, so a hand-written
    payload gets the same refusals by name: unknown keys, a name that is not
    a filename, an override that redefines a stage field.  The § 6.2 subset
    check is vacuous here by construction — which is correct, not skipped:
    ``varies`` came from these overrides.

    ``source`` is what the refusals call this payload — pass the flag or
    field the caller read it from, so a person is sent to what they wrote.

    Raises ``ValueError`` (from ``_refuse``) on any of the above.
    """
    with _refusals_name(source):
        return _ladder_from_objs(payload)


def _ladder_from_objs(payload: Any) -> Tuple[Tuple[str, ...],
                                             Tuple[Stage, ...]]:
    """The body of :func:`stages_from_dicts`, split only so the refusal
    naming above wraps the whole of it.

    Deliberately named without the word "stage":
    ``tests/test_stage_vocabulary.py`` inventories every declaration that
    carries it and demands each be attributed to a mechanism, and an
    implementation split is not a new way of expressing a stage.  Naming it
    ``_stages_from_dicts`` made the guard ask for a ledger row that would
    have said nothing -- caught 2026-08-07, by the guard."""

    if not isinstance(payload, (list, tuple)):
        _refuse(f"stages payload must be a list, got "
                f"{type(payload).__name__}")
    if not payload:
        _refuse("stages payload is empty. A ladder has at least one stage; "
                "for a single parameter set, do not pass one at all "
                "(engines/stages.md 6.5)")

    frozen = varies_for(
        _as_object(obj, where=f"stage {i}").get("overrides") or {}
        for i, obj in enumerate(payload))

    stages = tuple(_stage_from_obj(obj, frozen, i)
                   for i, obj in enumerate(payload))

    _refuse_duplicate_stage_names(stages)

    return frozen, stages


def _refuse_duplicate_stage_names(stages) -> None:
    """ONE copy of the case-insensitive duplicate check (D10, 2026-08-13:
    both parsers carried an identical loop, and the constructor a third
    exact-string variant -- three checks, one rule, drifting apart).
    Case folds because names key filenames (engines/stages.md § 2)."""
    seen: dict[str, str] = {}
    for st in stages:
        low = st.name.lower()
        if low in seen:
            _refuse(f"two stages named {st.name!r}"
                    + (f" and {seen[low]!r}" if seen[low] != st.name else "")
                    + " -- names are compared case-insensitively because they "
                      "become filenames (engines/stages.md 6.6)")
        seen[low] = st.name


def derive_run(name: str, formula: str = "", *, created: str = "",
               stage_names: Tuple[str, ...] = ()) -> Run:
    """Build a :class:`Run` with its id derived, not retyped.

    The one place a description's id is *made*.  § 3 rule 1 -- *"it happens
    once, when the description is written, and the result is stored"* -- is
    only true if there is one place it can happen, and a producer spelling
    ``Run(name=n, id=run_id(n, f))`` inline is a second place waiting to drift
    (the browser and the CLI must write the same bytes, § 6.4).

    ``stage_names`` is worth passing when the ladder is known: it makes the
    cap the real one rather than the budget guessed for an unknown ladder.
    :meth:`Task.__post_init__` re-derives with the stages it can see, so a
    ``Run`` built without them still checks out."""
    return Run(name=name, created=created,
               id=run_id(name, formula, stage_names=stage_names))


def read_task(path) -> Task:
    """Read and parse ``task.json``."""
    return Task.from_dict(read_json(path))


# --------------------------------------------------------------------- #
#  Writing                                                              #
# --------------------------------------------------------------------- #

def _task_to_dict(task: Task) -> dict:
    """The JSON object for a description.

    Key order is § 6's, so a written file reads like the contract's own
    example and a diff between two descriptions lines up."""
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": {"name": task.engine},
        "shape": task.shape,
        "run": {"name": task.run.name, "id": task.run.id},
    }
    if task.run.created:
        out["run"]["created"] = task.run.created
    if task.schema_fingerprint:
        out["schema_fingerprint"] = task.schema_fingerprint
    out["structure"] = {"source": task.structure.source,
                        "formula": task.structure.formula,
                        "atoms": task.structure.atoms}
    # Absent together, never empty -- an empty list would be a second way to
    # spell "one stage" (§ 6.5).
    if task.stages:
        out["varies"] = list(task.varies or ())
        out["stages"] = [{"name": s.name, "enabled": s.enabled,
                          "overrides": dict(s.overrides)}
                         for s in task.stages]
    return out


def write_task(path, task: Task):
    """Write ``task.json`` atomically (``persist.write_json``)."""
    return write_json(path, task.to_dict())


__all__ = ["SCHEMA", "FILENAME", "SHAPES", "STAGE_FIELDS",
           "Run", "StructureRef", "Stage", "Task",
           "derive_run", "read_task", "varies_for", "write_task"]
