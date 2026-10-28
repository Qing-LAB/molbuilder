"""The calculation kinds -- ``data/kind_table.toml`` read, once, into one
:class:`CalculationKind` per table, each with its part on every engine it
runs on (`execution/architecture.md` § 3.3).

**What differs between kinds is a field here, never a branch** (A18): a
kind's name as a person reads it, its rungs' roles and what each computes --
a kind with roles has a linked ladder, one without an independent one
(`job-system.md` § 5.4) -- the results it gives, what its description
carries beyond every kind's fields, the charge it fixes, why it takes no
benchmark, and the tab that hands it over or where it describes itself;
and on each engine it runs on, the
engine's own name for it, the shape it fixes and why, how its stages play
its roles (its rungs, every one in order, where a stage's role is its
name), and the code parts the pair names -- its run of points, its rung's
builder, the inputs its rungs take and their gather.  The engines a kind
runs on are its parts -- no second list says which engine runs which kind
-- and each is the engine table's (`engine_table`).

Floor 1 (`execution/architecture.md` § 2.1): data, read once; it imports no
kind and no engine.  A part that is code is named ``module:NAME`` and
resolved by the floor it lives on, through :func:`part`.
"""
from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .paths import SHAPES

#: The file, beside the parameter catalogue.
TABLE_FILE = Path(__file__).resolve().parent / "data" / "kind_table.toml"

#: What a rung computes: an SCF, or a product that runs none.  Each is also
#: a named set of roles a catalogue item's ``stages`` may cite
#: (`engines/template.md` § 6.4; :func:`role_set`) -- ``"scf"``, every rung
#: that runs an SCF.
COMPUTES = ("scf", "product")

#: The fields a description may carry beyond every kind's -- each a
#: kind's to carry (``description.carries``): the electrode rename, the
#: voltages -- stated whenever carried -- the finished calculation it
#: extends.
DESCRIPTION_FIELDS = ("swap_electrodes", "bias", "reuses")

_KIND_KEYS = ("noun", "roles", "results", "description", "charge",
              "no_benchmark", "handover", "on")
_PART_KEYS = ("words", "shape", "why", "stage_roles", "points", "rung",
              "inputs", "gather")

#: The parts a pair may name, each ``module:NAME`` (:func:`part_of`).
PART_REFS = ("points", "rung", "inputs", "gather")


class KindTableError(ValueError):
    """The kind table does not say what a kind must, a kind or an engine
    named that it does not hold, or a part that does not resolve -- said
    with the record and the key."""


@dataclass(frozen=True)
class Role:
    """One rung's role: what the rung does, named by its kind, and what it
    computes (:data:`COMPUTES`)."""
    name: str
    computes: str


@dataclass(frozen=True)
class Result:
    """One result a kind gives (`engines/template.md` § 6.2a): its key --
    what an item's ``result`` names -- and the card's title and the one
    line under it, saying what it gives and what it costs."""
    key: str
    title: str
    line: str


@dataclass(frozen=True)
class Description:
    """What a kind's description carries beyond every kind's fields."""
    #: The slot it cites instead of holding a structure -- its structure IS
    #: that citation, and its identity is the citation -- or ``None``: it
    #: holds a structure.
    cites: Optional[str] = None
    #: The fields only it may state (:data:`DESCRIPTION_FIELDS`).
    carries: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedCharge:
    """The charge a kind fixes, and why."""
    value: int
    why: str


@dataclass(frozen=True)
class KindOnEngine:
    """A kind's part on one engine: what the pair decides."""
    kind: str
    engine: str
    #: The engine's own name for the kind, where it has one -- *TranSIESTA*.
    words: Optional[str] = None
    #: The shape it fixes, or ``None`` where it fixes none.
    shape: Optional[str] = None
    #: The one sentence every surface says the fixed shape in.
    why: Optional[str] = None
    #: How its stages play the kind's roles: ``"own"``, its stages its rungs
    #: -- every role, in order, a stage's role its name; ``"every"``, each
    #: stage every role; or the roles each stage plays -- ``None`` for a
    #: kind without roles.
    stage_roles: Any = None
    #: The stages it fixes, in order -- its rungs, where ``stage_roles`` is
    #: ``"own"`` -- or ``None``.
    stages: Optional[Tuple[str, ...]] = None
    #: ``module:NAME`` of its run of points' declaration (floor 4), or
    #: ``None`` for a pair whose every rung runs once.
    points: Optional[str] = None
    #: ``module:NAME`` of its rung's builder (the conductor's), or ``None``.
    rung: Optional[str] = None
    #: For a kind whose rungs take inputs: ``module:NAME`` of
    #: ``(task, stage) -> [(stage upstream, file)]``, which rung hands which
    #: file (floor 3, `jobset.ladder`) -- ``None`` where a stage builds on
    #: no other.
    inputs: Optional[str] = None
    #: With ``inputs``: ``module:NAME`` of the gather that takes them at a
    #: rung's first launch (the ready door's, `jobset.ready`).
    gather: Optional[str] = None

    @property
    def fixed_shape(self) -> Optional[Dict[str, str]]:
        """``{"shape", "why"}`` -- the shape the pair fixes and why -- or
        ``None`` (`web/task-setup.md` § 4)."""
        if self.shape is None:
            return None
        return {"shape": self.shape, "why": self.why}


@dataclass(frozen=True)
class CalculationKind:
    """One kind of calculation, as the kind table states it."""
    name: str
    #: Its name as a person reads it -- *transport calculation*.
    noun: str
    #: Its rungs' roles -- a kind with roles has a LINKED ladder, each
    #: stage building on what its kind says; one without, an INDEPENDENT
    #: one, each stage continuing from the one before (`job-system.md`
    #: § 5.4).
    roles: Tuple[Role, ...] = ()
    results: Tuple[Result, ...] = ()
    #: What its description carries beyond every kind's fields.
    description: Description = Description()
    #: The charge it fixes, with why, or ``None``: its charge is resolved
    #: (`electronic_state`).
    charge: Optional[FixedCharge] = None
    #: Why it takes no benchmark, or ``None``: it takes one where its
    #: engine has a benchmark lane.
    no_benchmark: Optional[str] = None
    #: The tab that hands it over to Task setup, by its key in
    #: `web/tabs.py` -- or ``None``: it describes on its own tab, nothing
    #: awaiting.
    handover: Optional[str] = None
    #: Its part on each engine it runs on, by the engine's name.
    parts: Mapping[str, KindOnEngine] = None  # type: ignore[assignment]

    @property
    def role_names(self) -> Tuple[str, ...]:
        """Its rungs' roles, in order; ``()`` for a kind without roles,
        every rung of which reads every item it carries."""
        return tuple(r.name for r in self.roles)

    @property
    def engines(self) -> Tuple[str, ...]:
        """The engines it runs on -- its parts."""
        return tuple(self.parts)

    @property
    def with_article(self) -> str:
        """Its name with its article, as a sentence says it -- *an
        optimization*, *a transport calculation*."""
        return f"{'an' if self.noun[:1] in 'aeiou' else 'a'} {self.noun}"

    def on(self, engine: str) -> KindOnEngine:
        """Its part on ``engine``, or :class:`KindTableError` naming the
        engines it runs on."""
        got = self.parts.get(str(engine))
        if got is None:
            raise KindTableError(
                f"{self.with_article} does not run on {engine!r} -- it runs "
                f"on {', '.join(self.engines)} ({TABLE_FILE.name})")
        return got


def _refuse(where: str, why: str) -> None:
    raise KindTableError(f"{TABLE_FILE.name} {where}: {why}")


def _part(kind: str, engine: str, body: object,
          roles: Tuple[str, ...]) -> KindOnEngine:
    where = f"[{kind}.on.{engine}]"
    if not isinstance(body, dict):
        _refuse(where, "is not a table")
    unknown = sorted(set(body) - set(_PART_KEYS))
    if unknown:
        _refuse(where, f"unknown key(s) {', '.join(unknown)} -- a part "
                       f"states {', '.join(_PART_KEYS)}")
    shape, why = body.get("shape"), body.get("why")
    if (shape is None) != (why is None):
        _refuse(where, "a fixed shape and its sentence, `why`, come together")
    if shape is not None and shape not in SHAPES:
        _refuse(where, f"shape {shape!r} is not one of {', '.join(SHAPES)}")
    sr = body.get("stage_roles")
    if roles and sr is None:
        _refuse(where, "no stage_roles -- how its stages play the kind's "
                       "roles: \"own\", \"every\" or a list of roles")
    if sr is not None and not roles:
        _refuse(where, "stage_roles on a kind without roles")
    if isinstance(sr, list):
        bad = [r for r in sr if r not in roles]
        if bad or not sr:
            _refuse(where, f"stage_roles names {bad or 'no role'} -- the "
                           f"kind's roles are {', '.join(roles)}")
        sr = tuple(sr)
    elif sr is not None and sr not in ("own", "every"):
        _refuse(where, f"stage_roles {sr!r} is \"own\", \"every\" or a "
                       f"list of roles")
    if ("inputs" in body) != ("gather" in body):
        _refuse(where, "a rung's inputs and the gather that takes them come "
                       "together")
    if "inputs" in body and not roles:
        _refuse(where, "inputs on a kind without roles -- its stages "
                       "continue from the one before")
    return KindOnEngine(kind=kind, engine=engine, words=body.get("words"),
                        shape=shape, why=why, stage_roles=sr,
                        stages=tuple(roles) if sr == "own" else None,
                        points=body.get("points"), rung=body.get("rung"),
                        inputs=body.get("inputs"), gather=body.get("gather"))


def _kind(name: str, body: object, engines: Tuple[str, ...]
          ) -> CalculationKind:
    where = f"[{name}]"
    if not isinstance(body, dict):
        _refuse(where, "is not a table")
    unknown = sorted(set(body) - set(_KIND_KEYS))
    if unknown:
        _refuse(where, f"unknown key(s) {', '.join(unknown)} -- a kind "
                       f"states {', '.join(_KIND_KEYS)}")
    if not body.get("noun"):
        _refuse(where, "no noun -- its name as a person reads it")
    for key in ("roles", "results"):
        if not isinstance(body.get(key, []), list) or not all(
                isinstance(r, dict) for r in body.get(key, [])):
            _refuse(where, f"{key} is a list of tables")
    if not isinstance(body.get("on", {}), dict):
        _refuse(where, "on is a table, one part per engine")
    roles = []
    for r in body.get("roles", ()):
        if set(r) != {"name", "computes"} or r["computes"] not in COMPUTES:
            _refuse(where, f"a role is {{name, computes}}, computes one of "
                           f"{', '.join(COMPUTES)} -- got {r!r}")
        roles.append(Role(name=str(r["name"]), computes=str(r["computes"])))
    results = []
    for r in body.get("results", ()):
        if set(r) != {"key", "title", "line"}:
            _refuse(where, f"a result is {{key, title, line}} -- got {r!r}")
        results.append(Result(key=str(r["key"]), title=str(r["title"]),
                              line=str(r["line"])))
    for key, named in (("roles", [r.name for r in roles]),
                       ("results", [r.key for r in results])):
        twice = sorted({n for n in named if named.count(n) > 1})
        if twice:
            _refuse(where, f"{key} names {', '.join(twice)} twice")
    desc = body.get("description", {})
    if not isinstance(desc, dict) or set(desc) - {"cites", "carries"}:
        _refuse(where, "description is a table of cites and carries")
    carries = desc.get("carries", [])
    if (not isinstance(carries, list)
            or set(carries) - set(DESCRIPTION_FIELDS)):
        _refuse(where, f"description.carries is a list of "
                       f"{', '.join(DESCRIPTION_FIELDS)}")
    if "cites" in desc and not (isinstance(desc["cites"], str)
                                and desc["cites"]):
        _refuse(where, "description.cites is the slot's name")
    charge = body.get("charge")
    if charge is not None and not (
            isinstance(charge, dict) and set(charge) == {"value", "why"}
            and isinstance(charge["value"], int)
            and not isinstance(charge["value"], bool)):
        _refuse(where, "charge is {value, why} -- the charge it fixes, a "
                       "whole number, and why")
    for key in ("no_benchmark", "handover"):
        if key in body and not (isinstance(body[key], str) and body[key]):
            _refuse(where, f"{key} is a sentence")
    on = body.get("on") or {}
    if not on:
        _refuse(where, "no part on any engine -- the engines it runs on are "
                       "its parts")
    stray = sorted(set(on) - set(engines))
    if stray:
        _refuse(where, f"a part on {', '.join(stray)}, which the engine "
                       f"table does not hold ({', '.join(engines)})")
    names = tuple(r.name for r in roles)
    parts = {e: _part(name, e, b, names) for e, b in on.items()}
    return CalculationKind(name=name, noun=str(body["noun"]),
                           roles=tuple(roles), results=tuple(results),
                           description=Description(
                               cites=desc.get("cites"),
                               carries=tuple(carries)),
                           charge=(FixedCharge(value=charge["value"],
                                               why=str(charge["why"]))
                                   if charge is not None else None),
                           no_benchmark=body.get("no_benchmark"),
                           handover=body.get("handover"),
                           parts=parts)


@lru_cache(maxsize=1)
def _table() -> Dict[str, CalculationKind]:
    from .engine_table import engines
    raw = tomllib.loads(TABLE_FILE.read_text(encoding="utf-8"))
    known = engines()
    return {name: _kind(name, body, known) for name, body in raw.items()}


def kinds() -> Tuple[str, ...]:
    """Every kind, in the table's order."""
    return tuple(_table())


def kind(name: str) -> CalculationKind:
    """The kind called ``name``, or :class:`KindTableError` naming the kinds
    there are."""
    table = _table()
    if str(name) not in table:
        raise KindTableError(
            f"no calculation kind {name!r} -- the kinds are "
            f"{', '.join(table)} ({TABLE_FILE.name})")
    return table[str(name)]


def pair(calculation: str, engine: str) -> Optional[KindOnEngine]:
    """``calculation``'s part on ``engine`` -- ``None`` where the table holds
    no such kind, or the kind does not run on that engine: for a reader that
    shows what a pair fixes and leaves the refusal to the description's
    reader (`task`), which asks :meth:`CalculationKind.on`."""
    got = _table().get(str(calculation))
    return got.parts.get(str(engine)) if got is not None else None


def carrying(field: str) -> Tuple[str, ...]:
    """The kinds whose description may state ``field``
    (:attr:`Description.carries`), in the table's order -- what a refusal
    of it on another kind names."""
    return tuple(k for k, rec in _table().items()
                 if field in rec.description.carries)


def role_set(name: str) -> Tuple[str, ...]:
    """The roles that compute ``name`` (:data:`COMPUTES`) -- the named set a
    catalogue item's ``stages`` cites -- across every kind, in the table's
    order; :class:`KindTableError` naming the sets for a name none is."""
    if name not in COMPUTES:
        raise KindTableError(
            f"no set of roles {name!r} -- the sets are what a rung "
            f"computes: {', '.join(COMPUTES)}")
    return tuple(r.name for k in _table().values() for r in k.roles
                 if r.computes == name)


def part_of(calculation: str, engine: str, key: str) -> Any:
    """THE CODE PART ``calculation``'s pair on ``engine`` names under
    ``key`` (:data:`PART_REFS`), imported now -- by the floor that asks,
    the one the part lives on -- or ``None`` where the pair names none, or
    there is no such pair (the description's reader refuses that)."""
    got = pair(calculation, engine)
    ref = getattr(got, key) if got is not None else None
    if ref is None:
        return None
    return part(ref, record=f"{calculation} on {engine}", key=key)


def part(ref: str, *, record: str, key: str) -> Any:
    """THE OBJECT A RECORD NAMES -- ``module:NAME`` -- imported by the floor
    that asks, the one it lives on (`execution/architecture.md` § 3.3); a
    name that does not resolve refused naming the record and the key."""
    module, sep, attr = str(ref).partition(":")
    try:
        if not sep:
            raise ValueError("not \"module:NAME\"")
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError, ValueError) as exc:
        raise KindTableError(
            f"{record}'s {key}, {ref!r}, does not resolve: {exc}") from exc


__all__ = ["TABLE_FILE", "COMPUTES", "DESCRIPTION_FIELDS", "PART_REFS",
           "Description", "FixedCharge", "carrying", "KindTableError", "Role", "Result",
           "KindOnEngine", "CalculationKind", "kinds", "kind", "pair",
           "role_set", "part_of", "part"]
