"""The calculation kinds -- ``data/kind_table.toml`` read, once, into one
:class:`CalculationKind` per table, each with its part on every engine it
runs on (`execution/architecture.md` § 3.3).

**What differs between kinds is a field here, never a branch** (A18): a
kind's name as a person reads it, its rungs' roles and what each computes --
a kind with roles has a linked ladder, one without an independent one
(`job-system.md` § 5.4), and a role that is a lead names the region it
is cut from -- the results it gives, what its description
carries beyond every kind's fields, the charge it fixes, why it takes no
benchmark, the tab that hands it over or where it describes itself, its
topic -- the folder a new project lays down for it -- whether its deck
relaxes the structure, what its root opens and what a rung's folder opens
where that is not a run's own result, and its science checks (a part);
and on each engine it runs on, the
engine's own name for it, the shape it fixes and why, how its stages play
its roles (its rungs, every one in order, where a stage's role is its
name), the ladder a new description starts from, what relaxes the
structure first, a stage older versions wrote that it no longer holds, and
the code parts the pair names -- its run of points, its rung's builder, the
inputs its rungs take and their gather, and what `summarize task` adds up.  The engines a kind
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

from .identity import electrode_hs_stem
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

#: What a calculation's root opens where that is not one of its runs'
#: results (``root_opens``): its description, through which its report is
#: opened -- composed on read from the folder (`web/results.md` § 0.1).
ROOT_OPENS = ("description",)

#: What a rung's run folder opens where that is not the run's own result
#: (a role's ``opens``): its calculation's report, at the point the folder
#: is (`web/results.md` § 0.1) -- a rung whose engine writes no file a
#: viewer reads on its own.
ROLE_OPENS = ("report",)

_KIND_KEYS = ("noun", "topic", "roles", "results", "description",
              "charge", "no_benchmark", "handover", "relaxes",
              "root_opens", "validator", "on")
_PART_KEYS = ("words", "shape", "why", "stage_roles", "points", "rung",
              "inputs", "gather", "sum", "retired", "ladder", "relaxation")

#: The parts a pair may name, each ``module:NAME`` (:func:`part_of`).
PART_REFS = ("points", "rung", "inputs", "gather", "sum")


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
    #: The region of the junction a LEAD is cut from -- its files are on
    #: that region's name (:func:`rung_label`) and it samples the transport
    #: axis with its own count -- or ``None``: the rung is no lead
    #: (`execution/architecture.md` § 3.3).
    lead: Optional[str] = None
    #: What the rung's run folder opens where that is not the run's own
    #: result (:data:`ROLE_OPENS`) -- or ``None``: its own result.
    opens: Optional[str] = None


@dataclass(frozen=True)
class Retired:
    """A stage older versions wrote that a pair no longer holds beside its
    others: its name, the role it played, the version that retired it, and
    why -- the reader refuses a description still holding it, naming
    ``jobset migrate``, which folds it into the others."""
    stage: str
    role: str
    since: str
    why: str


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
    #: The ladder a new description starts from, each stage with no
    #: overrides -- its fixed stages, else as it lists them (a vibration's
    #: one stage, ``freq``, which a person may add to) -- or ``None``: its
    #: engine's ladder (the engine table's ``presets``).
    ladder: Optional[Tuple[str, ...]] = None
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
    #: ``module:NAME`` of what ``summarize task`` adds up for it -- a
    #: `jobset.summarize.TaskSum` (floor 6) -- or ``None``: its stages each
    #: write their own result.
    sum: Optional[str] = None
    #: A stage older versions wrote that it no longer holds, or ``None``.
    retired: Optional[Retired] = None
    #: What relaxes the structure first, in a person's words, where the
    #: pair relaxes before it measures -- a vibration's -- or ``None``.
    relaxation: Optional[str] = None

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
    #: The folder every new project lays down for its calculations --
    #: a project's run topics are the kinds' (`job-contracts.md` § 2.5).
    topic: str
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
    #: Whether its deck relaxes the structure -- an optimization's -- which
    #: the progress log seeded for its run states as its targets.
    relaxes: bool = False
    #: What its root opens where that is not one of its runs' results
    #: (:data:`ROOT_OPENS`), or ``None``.
    root_opens: Optional[str] = None
    #: ``module:NAME`` of its science checks, where it has its own -- run by
    #: the settings gate on every deck of it (`validation.validate`) -- or
    #: ``None``: its engine's checks are its science.
    validator: Optional[str] = None
    #: Its part on each engine it runs on, by the engine's name.
    parts: Mapping[str, KindOnEngine] = None  # type: ignore[assignment]

    def role(self, name: str) -> Optional[Role]:
        """Its role called ``name``, or ``None``: it has none of that
        name."""
        return next((r for r in self.roles if r.name == name), None)

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
    def leads(self) -> Dict[str, str]:
        """Its leads, each role's name to the region it is cut from, in
        order; ``{}`` for a kind without leads -- one that is open along
        the transport axis on no rung (`engines/siesta.md` § 6.1)."""
        return {r.name: r.lead for r in self.roles if r.lead}

    @property
    def label_tail(self) -> int:
        """How much longer than the calculation's label its longest
        lead's stem is (:func:`rung_label`) -- what the label's cap budgets
        (`identity.normalise_id`); 0 for a kind without leads."""
        return max((len(electrode_hs_stem("", region).encode("utf-8"))
                    for region in self.leads.values()), default=0)

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
    # THE STAGES IT FIXES: its rungs where its stages play their own roles,
    # read off the roles -- and so the ladder a new description starts from;
    # else the ladder as the pair lists it, where it lists one (a
    # vibration's one stage), each a new description's, with no overrides.
    stages = tuple(roles) if sr == "own" else None
    ladder = body.get("ladder")
    if ladder is not None:
        if sr == "own":
            _refuse(where, "ladder on a pair whose stages are its rungs -- "
                           "they are read off the kind's roles")
        if (not isinstance(ladder, list) or not ladder
                or not all(isinstance(x, str) and x for x in ladder)):
            _refuse(where, "ladder is a list of stage names")
        ladder = tuple(ladder)
    else:
        ladder = stages
    if ("inputs" in body) != ("gather" in body):
        _refuse(where, "a rung's inputs and the gather that takes them come "
                       "together")
    if "inputs" in body and not roles:
        _refuse(where, "inputs on a kind without roles -- its stages "
                       "continue from the one before")
    retired = body.get("retired")
    if retired is not None:
        if (not isinstance(retired, dict)
                or set(retired) != {"stage", "role", "since", "why"}
                or not all(isinstance(v, str) and v
                           for v in retired.values())):
            _refuse(where, "retired is {stage, role, since, why} -- the "
                           "stage, the role it played, the version that "
                           "retired it and why")
        if retired["role"] not in roles:
            _refuse(where, f"retired names role {retired['role']!r} -- the "
                           f"kind's roles are {', '.join(roles) or 'none'}")
        retired = Retired(**retired)
    relaxation = body.get("relaxation")
    if relaxation is not None and not (isinstance(relaxation, str)
                                       and relaxation):
        _refuse(where, "relaxation is a sentence -- what relaxes the "
                       "structure first")
    return KindOnEngine(kind=kind, engine=engine, words=body.get("words"),
                        shape=shape, why=why, stage_roles=sr,
                        stages=stages, ladder=ladder,
                        points=body.get("points"), rung=body.get("rung"),
                        inputs=body.get("inputs"), gather=body.get("gather"),
                        sum=body.get("sum"), retired=retired,
                        relaxation=relaxation)


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
    if not (isinstance(body.get("topic"), str) and body["topic"]):
        _refuse(where, "no topic -- the folder a new project lays down "
                       "for it")
    for key in ("roles", "results"):
        if not isinstance(body.get(key, []), list) or not all(
                isinstance(r, dict) for r in body.get(key, [])):
            _refuse(where, f"{key} is a list of tables")
    if not isinstance(body.get("on", {}), dict):
        _refuse(where, "on is a table, one part per engine")
    roles = []
    for r in body.get("roles", ()):
        if (not {"name", "computes"} <= set(r)
                <= {"name", "computes", "lead", "opens"}
                or r["computes"] not in COMPUTES
                or ("lead" in r and not (isinstance(r["lead"], str)
                                         and r["lead"]))
                or ("opens" in r and r["opens"] not in ROLE_OPENS)):
            _refuse(where, f"a role is {{name, computes}}, a lead's region "
                           f"and what its folder opens -- computes one of "
                           f"{', '.join(COMPUTES)}, opens one of "
                           f"{', '.join(ROLE_OPENS)} -- got {r!r}")
        roles.append(Role(name=str(r["name"]), computes=str(r["computes"]),
                          lead=r.get("lead"), opens=r.get("opens")))
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
    for key in ("no_benchmark", "handover", "validator"):
        if key in body and not (isinstance(body[key], str) and body[key]):
            _refuse(where, f"{key} is a sentence")
    if not isinstance(body.get("relaxes", False), bool):
        _refuse(where, "relaxes is true or false")
    if body.get("root_opens", ROOT_OPENS[0]) not in ROOT_OPENS:
        _refuse(where, f"root_opens is one of {', '.join(ROOT_OPENS)}")
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
                           topic=body["topic"],
                           roles=tuple(roles), results=tuple(results),
                           description=Description(
                               cites=desc.get("cites"),
                               carries=tuple(carries)),
                           charge=(FixedCharge(value=charge["value"],
                                               why=str(charge["why"]))
                                   if charge is not None else None),
                           no_benchmark=body.get("no_benchmark"),
                           handover=body.get("handover"),
                           relaxes=body.get("relaxes", False),
                           root_opens=body.get("root_opens"),
                           validator=body.get("validator"),
                           parts=parts)


@lru_cache(maxsize=1)
def _table() -> Dict[str, CalculationKind]:
    from .engine_table import engines
    raw = tomllib.loads(TABLE_FILE.read_text(encoding="utf-8"))
    known = engines()
    table = {name: _kind(name, body, known) for name, body in raw.items()}
    # A LEAD IS FOUND BY ITS STAGE'S NAME (:func:`lead_of`), so its name is
    # one no other kind's role has.
    for name, rec in table.items():
        for lead in rec.leads:
            also = [k for k, other in table.items() if k != name
                    and lead in other.role_names]
            if also:
                _refuse(f"[{name}]", f"lead {lead!r} is a role of "
                                     f"{', '.join(also)} too -- a lead's "
                                     f"name is its kind's alone")
    return table


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


def lead_of(stage: Optional[str]) -> Optional[str]:
    """The region the lead named ``stage`` is cut from, or ``None``: the
    stage is no kind's lead."""
    for rec in _table().values():
        if stage in rec.leads:
            return rec.leads[stage]
    return None


def rung_label(task_label: str, stage: Optional[str]) -> str:
    """THE LABEL A RUNG'S FILES ARE NAMED ON (`job-contracts.md` § 2.1,
    rule 2): the calculation's own for every rung but a lead's, whose files
    -- the engine's ``.TSHS``, ``.DM``, ``.XV`` and ours alike -- are on the
    region it is cut from (``<label>_L-electrode``,
    `identity.electrode_hs_stem`), which the device deck's
    ``TS.Elec.<name> HS`` line names.  Asked by the element's label
    (`resolve._label_for`), the job's names (`jobset.naming.run_names`) and
    the run door's reading of a folder (`runs._position`), so no reader
    composes a lead's name a second way.  ``stage`` is the rung's name;
    ``None`` -- no ladder -- is the label."""
    region = lead_of(stage)
    return electrode_hs_stem(task_label, region) if region else task_label


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


__all__ = ["TABLE_FILE", "COMPUTES", "DESCRIPTION_FIELDS", "ROOT_OPENS",
           "ROLE_OPENS", "PART_REFS",
           "Description", "FixedCharge", "carrying", "KindTableError", "Role", "Result",
           "Retired", "KindOnEngine", "CalculationKind", "kinds", "kind",
           "pair", "lead_of", "rung_label", "role_set", "part_of", "part"]
