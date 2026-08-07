"""``task.json`` — a calculation's description on disk, and its one reader.

Contract: ``docs/engines/stages.md`` § 6.  This module is the *only* place
that turns those bytes into objects and back, which is what makes
*"the browser writes the same bytes as the CLI"* checkable rather than
coincidental (§ 6.4).

WHAT A DESCRIPTION IS.  One calculation: which engine, which layout, what it
is a calculation *of*, one value for every schema field (``base``), which of
those fields the user chose to tune (``varies``), and the per-stage cells that
vary (``stages[].overrides``).  It **names no machine** — ranks, queues and
walltimes are decided by ``prep`` on the target, not written here
(``execution/project-layout.md`` § 2.1).

LAYER.  L1: it imports ``persist`` and the standard library, and nothing else.
That is deliberate rather than incidental — see *the split preflight* below.

THE SPLIT PREFLIGHT.  § 6.6 lists eight checks "in order, and all of it before
anything is written".  Four of them are answerable from the file alone and are
enforced here; the other four need the engine's field schema, and importing an
engine into an L1 codec is exactly what ``tests/test_layering.py`` prevents.
Those belong to resolution (P2 of ``execution/staged-runs-implementation-plan.md``),
which already has the schema in hand:

  here    the schema string's major · ``shape`` present and legal · stage names
          in ``[A-Za-z0-9_]+`` and unique case-insensitively · no ``overrides``
          key naming a stage field · ``overrides`` keys equal to ``varies`` ·
          unknown keys refused by name
  P2      the engine has a generator · the schema fingerprint matches · every
          named field exists in the schema · every value is inside its bounds ·
          and § 6.6a's warning for two stages that RESOLVE identically, since
          resolving is P2's verb

WHY THE MESSAGES NAME THINGS.  A description is JSON sitting beside the decks,
and as of 2026-08-07 editing it by hand is **supported** (the plan's decision 3).
So a refusal owes a person what it owes the browser: the offending key by name,
which stage it was in, and — where the key is one edit away from a real one —
what they probably meant.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Optional, Tuple

from .persist import check_schema_major, read_json, write_json


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
             "structure", "base", "varies", "stages")
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
    file is what says which calculation lives there (§ 3.0 there)."""
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


@dataclass(frozen=True)
class Task:
    """One calculation.

    ``stages`` and ``varies`` are ``None`` together, never empty: a
    description with no stages **is** a single-parameter-set calculation
    (§ 6.5), and an empty list would be a second way to spell that."""
    engine: str
    shape: str
    run: Run
    structure: StructureRef
    base: Mapping[str, Any] = field(default_factory=dict)
    varies: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[Stage, ...]] = None
    schema_fingerprint: str = ""

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

def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(f"{FILENAME}{': ' + where if where else ''}: {msg}")


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

    check_schema_major(str(obj.get("schema") or ""), SCHEMA, label=FILENAME)
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

    base = dict(_as_object(_require(obj, "base", where=""), where="base"))

    has_stages = "stages" in obj
    has_varies = "varies" in obj
    if has_varies and not has_stages:
        _refuse("'varies' without 'stages'. A description with no stages is "
                "one parameter set, and there is nothing to vary across "
                "(engines/stages.md 6.5)")

    if not has_stages:
        return Task(engine=engine, shape=shape, run=run, structure=structure,
                    base=base,
                    schema_fingerprint=str(obj.get("schema_fingerprint", "")))

    raw_stages = obj["stages"]
    if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
        _refuse("'stages' is present but empty. A description WITH stages has "
                "at least one; a calculation with a single parameter set omits "
                "the key entirely (engines/stages.md 6.5)")

    varies = tuple(str(v) for v in obj.get("varies", ()))
    stages = tuple(_stage_from_obj(s, varies, i)
                   for i, s in enumerate(raw_stages))

    seen: dict[str, str] = {}
    for st in stages:
        low = st.name.lower()
        if low in seen:
            _refuse(f"two stages named {st.name!r}"
                    + (f" and {seen[low]!r}" if seen[low] != st.name else "")
                    + " -- names are compared case-insensitively because they "
                      "become filenames (engines/stages.md 6.6)")
        seen[low] = st.name

    return Task(engine=engine, shape=shape, run=run, structure=structure,
                base=base, varies=varies or None, stages=stages,
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

    # § 6.2 -- exactly the keys in `varies`, no more and no fewer.  "No more"
    # is the load-bearing half: a demoted parameter must not leave a value
    # hiding in a stage nobody can see.
    extra = sorted(set(overrides) - set(varies))
    if extra:
        _refuse(f"override(s) {', '.join(repr(k) for k in extra)} not listed "
                "in 'varies'. Every stage's overrides holds exactly the varied "
                "keys, so a demoted parameter cannot leave a value hiding "
                "(engines/stages.md 6.2)", where=where)
    missing = sorted(set(varies) - set(overrides))
    if missing:
        _refuse(f"missing override(s) {', '.join(repr(k) for k in missing)} "
                "listed in 'varies'", where=where)

    return Stage(name=name, enabled=bool(obj.get("enabled", True)),
                 overrides=overrides)


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
    out["base"] = dict(task.base)
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
           "read_task", "write_task"]
