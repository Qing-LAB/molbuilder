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
where that is not a run's own result, its science checks (a part) and the
module that exports its results, the labels its regions are with what each
is for, the checks its description takes (a part), the kind its citation
is a run of, and the code the shared doors ask of a kind that cites or
extends -- its description's parts;
and on each engine it runs on, the
engine's own name for it, how its stages play
its roles (its rungs, every one in order, where a stage's role is its
name), the ladder a new description starts from, what relaxes the
structure first, a stage older versions wrote that it no longer holds, and
its run of points (its record, `execution/runs-of-points.md` § 2.1), the
order its decks list atoms in, and the code parts the pair names -- its run
of points' declaration, its rung's builder, the inputs its rungs take and
their gather, what `summarize task` adds up, and what a scratch folder's
status line reads.  The engines a kind
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
from typing import Any, Dict, Mapping, Optional, Tuple

from . import data_dir
from .identity import electrode_hs_stem
from .persist import unknown_keys

#: The file, beside the parameter catalogue.
TABLE_FILE = data_dir() / "kind_table.toml"

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
              "root_opens", "validator", "export", "regions",
              "description_checks", "on")
_PART_KEYS = ("words", "stage_roles", "points", "rung",
              "inputs", "gather", "continues", "sum", "retired", "ladder",
              "relaxation", "atom_order", "scratch_status")

#: The parts a pair may name, each ``module:NAME`` (:func:`part_of`) --
#: ``points`` its run of points' declaration (:attr:`RunOfPoints.declaration`).
PART_REFS = ("points", "rung", "inputs", "gather", "continues", "sum",
             "scratch_status")

#: The parts a kind's description may name for the shared doors that ask of
#: a kind that cites or extends (:func:`description_part`): the limits of
#: extending a finished calculation, the frames its citation holds, the
#: citation's atom order recorded at a prep that names no group, how it is
#: described from its citation or the calculation it extends, and the reuse
#: of the calculation it extends -- each ``module:NAME``, or a module named
#: whole where :data:`WHOLE_MODULE_PARTS` says so.
DESCRIPTION_PARTS = ("extension", "cited_frames", "order_record", "describe",
                     "reuse")
#: THE DOORS A ``reuse`` PART PROVIDES, asked of the module it names: by
#: prep (``decide``, ``copy_in``) and launch (``takes_over``,
#: ``take_into``, ``shared_into``, ``carry_shared``).
REUSE_DOORS = ("decide", "copy_in", "takes_over", "take_into",
               "shared_into", "carry_shared")
#: THE DOORS A ``describe`` PART PROVIDES, asked by the command line's
#: `molbuilder describe`: the describe itself, what it does not take, and
#: its refusal.
DESCRIBE_DOORS = ("describe", "not_taken", "NotDescribed")
#: THE DOORS AN ``export`` PART PROVIDES, asked by `jobset export`: the
#: export itself, the name it writes, and the options it asks beyond the
#: form (``ASKS``).
EXPORT_DOORS = ("export", "name_of", "ASKS")
#: THE PARTS NAMED AS A MODULE WHOLE -- a description's ``describe`` and
#: ``reuse``, a kind's ``export`` -- each with the doors it provides: read
#: as the table loads (named whole, never ``module:NAME``) and as the part
#: resolves (every door there, :func:`description_part`,
#: :func:`export_of`); every other part is ``module:NAME``.
WHOLE_MODULE_PARTS = {"describe": DESCRIBE_DOORS, "reuse": REUSE_DOORS,
                      "export": EXPORT_DOORS}

#: A RUN OF POINTS' RECORD (`execution/runs-of-points.md` § 2.1): when its
#: run is finished -- every point DONE, or every point RAN, the words the
#: job reads in the index launch copies the record into (`runpoints.DONE`,
#: `runpoints.RAN`: the job holds no kind table, and this table loads before
#: the package can import that module) -- and whether its points share ONE
#: wall or are told EACH their own.
FINISHED = ("done", "ran")
WALL_ONE, WALL_EACH = "one", "each"
WALLS = (WALL_ONE, WALL_EACH)
#: The moments a kind's own steps run at, around a point's engine
#: (`runs-of-points.md` § 1.2): before it, after it, and the run's finish.
STEP_MOMENTS = ("before", "after", "finish")
_POINTS_KEYS = ("declaration", "unit", "finished", "wall", "count",
                "outcome", "steps")


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
    #: A LEAD's label, what it is for -- the sentence the label palette shows
    #: beside it -- or ``None`` for a role that is no lead.
    about: Optional[str] = None


@dataclass(frozen=True)
class Region:
    """A label besides the leads that a kind reads on its structure's atoms
    (``regions``) -- a transport junction's ``bridge`` -- and what it is
    for, the sentence the label palette shows beside it."""
    name: str
    about: str


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
    #: The kind of run its citation is -- an optimization, whose order rule
    #: judges the cited deck -- or ``None``: it cites nothing.
    cited_kind: Optional[str] = None
    #: Its parts for the shared doors (:data:`DESCRIPTION_PARTS`), by key --
    #: each ``module:NAME``, or a module named whole
    #: (:data:`WHOLE_MODULE_PARTS`).
    parts: Mapping[str, str] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FixedCharge:
    """The charge a kind fixes, and why."""
    value: int
    why: str


@dataclass(frozen=True)
class RunOfPoints:
    """A PAIR'S RUN OF POINTS, as the kind table states it
    (`execution/runs-of-points.md` § 2.1): what never changes for the pair
    -- the floor-4 part that declares its points, what one point is called,
    when its run is finished, its wall, its words (parts of floor 1 that
    travel beside the job) and its own steps around a point's engine."""
    #: ``module:NAME`` of its declaration (`jobset.points.PointsOf`).
    declaration: str
    #: What one point is called in a person's words -- *point*, *call*.
    unit: str
    #: When the run is finished (:data:`FINISHED`): every point ``done``,
    #: or every point ``ran``.
    finished: str
    #: Whether its points share ``one`` wall or are told ``each`` their own
    #: (:data:`WALLS`).
    wall: str
    #: ``module:NAME`` of the count's words, ``(rows, facts) -> str``.
    count: str
    #: ``module:NAME`` of a round's outcome, ``(index, rows, round, run_dir,
    #: names) -> Optional[str]``.
    outcome: str
    #: Its own steps, each a verb of the job's bundle, by moment
    #: (:data:`STEP_MOMENTS`) -- ``{}`` for a kind that works not between
    #: its points.
    steps: Mapping[str, str] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class KindOnEngine:
    """A kind's part on one engine: what the pair decides."""
    kind: str
    engine: str
    #: The engine's own name for the kind, where it has one -- *TranSIESTA*.
    words: Optional[str] = None
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
    #: Its run of points (:class:`RunOfPoints`), or ``None`` for a pair
    #: whose every rung runs once.
    run_of_points: Optional[RunOfPoints] = None
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
    #: ``module:NAME`` of ``(stages, stage) -> Optional[str]``, the stage a
    #: stage of a linked kind continues from -- ``stages`` ``[(name,
    #: resolved config)]`` in ladder order (floor 3, `jobset.ladder`) -- or
    #: ``None`` where none continues from another (a displacement sweep's
    #: later stage, the first, `engines/vibration.md` § 5.9).
    continues: Optional[str] = None
    #: ``module:NAME`` of what ``summarize task`` adds up for it -- a
    #: `jobset.summarize.TaskSum` (floor 6) -- or ``None``: its stages each
    #: write their own result.
    sum: Optional[str] = None
    #: A stage older versions wrote that it no longer holds, or ``None``.
    retired: Optional[Retired] = None
    #: What relaxes the structure first, in a person's words, where the
    #: pair relaxes before it measures -- a vibration's -- or ``None``.
    relaxation: Optional[str] = None
    #: The order its decks list atoms in, where it is not the structure's
    #: own -- a word of `atom_permutation.SORT_KEY_NAMES` -- or ``None``: the
    #: user order (:func:`atom_order`).
    atom_order: Optional[str] = None
    #: ``module:NAME`` of the line Status shows about a rung's scratch
    #: folder, for a pair whose rung keeps one, or ``None``.
    scratch_status: Optional[str] = None

    @property
    def points(self) -> Optional[str]:
        """``module:NAME`` of its run of points' declaration (floor 4), or
        ``None`` for a pair whose every rung runs once."""
        return (self.run_of_points.declaration
                if self.run_of_points is not None else None)


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
    #: The module that exports its results -- ``export(calc, fmt,
    #: **asks)``, ``name_of(calc, fmt)``, ``ASKS`` -- or ``None``: it
    #: exports nothing (:func:`export_of`).
    export: Optional[str] = None
    #: The labels besides its leads that it reads on its structure's atoms,
    #: each with its sentence (:class:`Region`), in order.
    regions: Tuple[Region, ...] = ()
    #: ``module:NAME`` of its description's own checks before any deck --
    #: ``(task, ladder) -> [Issue]``, run by `validation.task.preflight` --
    #: or ``None``.
    description_checks: Optional[str] = None
    #: Its part on each engine it runs on, by the engine's name.
    parts: Mapping[str, KindOnEngine] = None  # type: ignore[assignment]

    def role(self, name: str) -> Optional[Role]:
        """Its role called ``name``, or ``None``: it has none of that
        name."""
        return next((r for r in self.roles if r.name == name), None)

    def opens_report(self, name: str) -> bool:
        """Whether a run folder of its rung ``name`` opens its calculation's
        report -- the role's ``opens`` (:data:`ROLE_OPENS`): a rung whose
        engine writes no file a viewer reads on its own, its result the
        report at the point the folder is (`web/results.md` § 0.1)."""
        role = self.role(name)
        return role is not None and role.opens == ROLE_OPENS[0]

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
    def labels(self) -> Tuple[Tuple[str, str], ...]:
        """THE LABELS IT READS on its structure's atoms, each with what it is
        for: its leads' regions, then its other ``regions``, in order -- the
        label palette's names and sentences (`web/molview.md`)."""
        return (tuple((r.lead, r.about) for r in self.roles if r.lead)
                + tuple((g.name, g.about) for g in self.regions))

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

    def engine_label(self, engine: str) -> str:
        """``engine``'s name for this kind: its part's own (*TranSIESTA*),
        else the engine's (`engine_table`), for one it does not run on --
        what an engine tab is named (`engines/transport.md` § 3.8.8)."""
        from .engine_table import engine as _engine
        got = self.parts.get(str(engine))
        return got.words if got is not None and got.words else \
            _engine(engine).words

    def why_not_on(self, engine: str) -> Optional[str]:
        """Why it cannot be described on ``engine``, or ``None`` when it
        can -- the one sentence a disabled engine tab and every door that
        refuses the engine say (`engines/transport.md` § 3.8.8)."""
        from .engine_table import engines
        if str(engine) in self.parts:
            return None
        ships = " and ".join(self.engine_label(e) for e in self.engines)
        if engine not in engines():
            return (f"{engine!r} is no engine molbuilder knows; {self.topic} "
                    f"ships on {ships}")
        return (f"no {self.engine_label(engine)} {self.topic} backend is "
                f"built yet; {self.topic} ships on {ships}")

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


def _run_of_points(where: str, body: object) -> RunOfPoints:
    """A pair's run of points read and checked (`runs-of-points.md`
    § 2.1): every key it must state, each value one of its words; a part's
    resolving is the floor's that asks it."""
    if not isinstance(body, dict):
        _refuse(where, "is not a table")
    said = unknown_keys(body, _POINTS_KEYS)
    if said:
        _refuse(where, said)
    missing = [k for k in _POINTS_KEYS if k != "steps" and k not in body]
    if missing:
        _refuse(where, f"no {', '.join(missing)} -- a run of points states "
                       f"each")
    for key, words in (("finished", FINISHED), ("wall", WALLS)):
        if body[key] not in words:
            _refuse(where, f"{key} is {body[key]!r} -- one of "
                           f"{', '.join(words)}")
    for key in ("declaration", "count", "outcome"):
        if ":" not in str(body[key]):
            _refuse(where, f"{key} is {body[key]!r} -- a part is "
                           f"\"module:NAME\"")
    steps = body.get("steps") or {}
    if not isinstance(steps, dict):
        _refuse(where, f"steps is {steps!r} -- a table of its moments, "
                       f"{', '.join(STEP_MOMENTS)}")
    odd = sorted(set(steps) - set(STEP_MOMENTS))
    if odd:
        _refuse(where, f"steps names {', '.join(odd)} -- its moments are "
                       f"{', '.join(STEP_MOMENTS)}")
    return RunOfPoints(declaration=str(body["declaration"]),
                       unit=str(body["unit"]), finished=body["finished"],
                       wall=body["wall"], count=str(body["count"]),
                       outcome=str(body["outcome"]),
                       steps={k: str(v) for k, v in steps.items()})


def _part(kind: str, engine: str, body: object,
          roles: Tuple[str, ...]) -> KindOnEngine:
    where = f"[{kind}.on.{engine}]"
    if not isinstance(body, dict):
        _refuse(where, "is not a table")
    said = unknown_keys(body, _PART_KEYS)
    if said:
        _refuse(where, said)
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
            _refuse(where, "ladder on a pair whose stage_roles is \"own\" "
                           "-- its stages are the kind's roles, read off them")
        if (not isinstance(ladder, list) or not ladder
                or not all(isinstance(x, str) and x for x in ladder)):
            _refuse(where, "ladder is a list of stage names")
        ladder = tuple(ladder)
    else:
        ladder = stages
    if ("inputs" in body) != ("gather" in body):
        _refuse(where, "a stage's inputs and the gather that takes them come "
                       "together")
    if "inputs" in body and not roles:
        _refuse(where, "inputs on a kind without roles -- its stages "
                       "continue from the one before")
    if "continues" in body and not roles:
        _refuse(where, "continues on a kind without roles -- its stages "
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
    from .atom_permutation import SORT_KEY_NAMES
    atom_order = body.get("atom_order")
    if atom_order is not None and atom_order not in SORT_KEY_NAMES:
        _refuse(where, f"atom_order is {atom_order!r} -- one of "
                       f"{', '.join(SORT_KEY_NAMES)} (atom_permutation)")
    for key in PART_REFS:
        if key == "points" or key not in body:
            continue                    # `points` is a table, checked below
        if not (isinstance(body[key], str) and ":" in body[key]):
            _refuse(where, f"{key} is {body[key]!r} -- a part is "
                           f"\"module:NAME\"")
    points = body.get("points")
    return KindOnEngine(kind=kind, engine=engine, words=body.get("words"),
                        stage_roles=sr,
                        stages=stages, ladder=ladder,
                        run_of_points=(_run_of_points(f"[{kind}.on.{engine}"
                                                      f".points]", points)
                                       if points is not None else None),
                        rung=body.get("rung"),
                        inputs=body.get("inputs"), gather=body.get("gather"),
                        continues=body.get("continues"),
                        sum=body.get("sum"), retired=retired,
                        relaxation=relaxation, atom_order=atom_order,
                        scratch_status=body.get("scratch_status"))


def _kind(name: str, body: object, engines: Tuple[str, ...]
          ) -> CalculationKind:
    where = f"[{name}]"
    if not isinstance(body, dict):
        _refuse(where, "is not a table")
    said = unknown_keys(body, _KIND_KEYS)
    if said:
        _refuse(where, said)
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
                <= {"name", "computes", "lead", "about", "opens"}
                or r["computes"] not in COMPUTES
                or ("lead" in r and not (isinstance(r["lead"], str)
                                         and r["lead"]))
                or ("lead" in r) != ("about" in r)
                or ("about" in r and not (isinstance(r["about"], str)
                                          and r["about"]))
                or ("opens" in r and r["opens"] not in ROLE_OPENS)):
            _refuse(where, f"a role is {{name, computes}}, a lead's region "
                           f"with what its label is for (lead, about) and "
                           f"what its folder opens -- computes one of "
                           f"{', '.join(COMPUTES)}, opens one of "
                           f"{', '.join(ROLE_OPENS)} -- got {r!r}")
        roles.append(Role(name=str(r["name"]), computes=str(r["computes"]),
                          lead=r.get("lead"), opens=r.get("opens"),
                          about=r.get("about")))
    regions = []
    raw_regions = body.get("regions", [])
    if not isinstance(raw_regions, list):
        _refuse(where, "regions is a list of {name, about}")
    for g in raw_regions:
        if (not isinstance(g, dict) or set(g) != {"name", "about"}
                or not all(isinstance(v, str) and v for v in g.values())):
            _refuse(where, f"a region is {{name, about}} -- the label and "
                           f"what it is for -- got {g!r}")
        regions.append(Region(name=g["name"], about=g["about"]))
    results = []
    for r in body.get("results", ()):
        if set(r) != {"key", "title", "line"}:
            _refuse(where, f"a result is {{key, title, line}} -- got {r!r}")
        results.append(Result(key=str(r["key"]), title=str(r["title"]),
                              line=str(r["line"])))
    for key, named in (("roles", [r.name for r in roles]),
                       ("results", [r.key for r in results]),
                       ("labels", [r.lead for r in roles if r.lead]
                        + [g.name for g in regions])):
        twice = sorted({n for n in named if named.count(n) > 1})
        if twice:
            _refuse(where, f"{key} names {', '.join(twice)} twice")
    desc = body.get("description", {})
    if not isinstance(desc, dict) or set(desc) - {
            "cites", "carries", "cited_kind", *DESCRIPTION_PARTS}:
        _refuse(where, f"description is a table of cites, carries, "
                       f"cited_kind and its parts "
                       f"({', '.join(DESCRIPTION_PARTS)})")
    carries = desc.get("carries", [])
    if (not isinstance(carries, list)
            or set(carries) - set(DESCRIPTION_FIELDS)):
        _refuse(where, f"description.carries is a list of "
                       f"{', '.join(DESCRIPTION_FIELDS)}")
    if "cites" in desc and not (isinstance(desc["cites"], str)
                                and desc["cites"]):
        _refuse(where, "description.cites is the slot's name")
    # WHAT A CITATION IS, AND THE DOORS OF A KIND THAT CITES OR EXTENDS, come
    # with what makes them asked: a citation with a cited slot, the
    # extension and the reuse with a carried `reuses`.
    for key, needs, has in (
            ("cited_kind", "cites", "cites" in desc),
            ("cited_frames", "cites", "cites" in desc),
            ("order_record", "cites", "cites" in desc),
            ("describe", "cites or carries reuses",
             "cites" in desc or "reuses" in carries),
            ("extension", "carries reuses", "reuses" in carries),
            ("reuse", "carries reuses", "reuses" in carries)):
        if key in desc and not has:
            _refuse(where, f"description.{key} on a kind whose description "
                           f"does not state {needs}")
        if key in desc and not (isinstance(desc[key], str) and desc[key]):
            _refuse(where, f"description.{key} is a name")
    # A KIND THAT CITES OR CARRIES `reuses` NAMES WHAT ANSWERS FOR IT --
    # the doors that read a citation's frames, or extend a finished
    # calculation, ask the part, never a module by name.
    for needs, has, keys in (("cites", "cites" in desc,
                              ("cited_frames", "describe")),
                             ("carries reuses", "reuses" in carries,
                              ("extension", "reuse", "describe"))):
        for key in (keys if has else ()):
            if key not in desc:
                _refuse(where, f"description {needs} and names no {key} "
                               f"part")
    for key in DESCRIPTION_PARTS:
        if key not in desc:
            continue
        whole = key in WHOLE_MODULE_PARTS
        if (":" in str(desc[key])) == whole:
            _refuse(where, f"description.{key} is {desc[key]!r} -- "
                           + (f"a {key} part is a module, named whole"
                              if whole else "a part is \"module:NAME\""))
    charge = body.get("charge")
    if charge is not None and not (
            isinstance(charge, dict) and set(charge) == {"value", "why"}
            and isinstance(charge["value"], int)
            and not isinstance(charge["value"], bool)):
        _refuse(where, "charge is {value, why} -- the charge it fixes, a "
                       "whole number, and why")
    for key, what, ok in (
            ("no_benchmark", "a sentence", bool),
            ("handover", "the tab's name", bool),
            ("validator", "a part, \"module:NAME\"", lambda v: ":" in v),
            ("export", "a module, named whole", lambda v: ":" not in v),
            ("description_checks", "a part, \"module:NAME\"",
             lambda v: ":" in v)):
        if key in body and not (isinstance(body[key], str) and body[key]
                                and ok(body[key])):
            _refuse(where, f"{key} is {what}")
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
                               carries=tuple(carries),
                               cited_kind=desc.get("cited_kind"),
                               parts={k: desc[k] for k in DESCRIPTION_PARTS
                                      if k in desc}),
                           charge=(FixedCharge(value=charge["value"],
                                               why=str(charge["why"]))
                                   if charge is not None else None),
                           no_benchmark=body.get("no_benchmark"),
                           handover=body.get("handover"),
                           relaxes=body.get("relaxes", False),
                           root_opens=body.get("root_opens"),
                           validator=body.get("validator"),
                           export=body.get("export"),
                           regions=tuple(regions),
                           description_checks=body.get("description_checks"),
                           parts=parts)


@lru_cache(maxsize=1)
def _table() -> Dict[str, CalculationKind]:
    from .engine_table import engines
    raw = tomllib.loads(TABLE_FILE.read_text(encoding="utf-8"))
    known = engines()
    table = {name: _kind(name, body, known) for name, body in raw.items()}
    # A CITATION'S KIND IS A KIND OF THE TABLE.
    for name, rec in table.items():
        cited = rec.description.cited_kind
        if cited is not None and cited not in table:
            _refuse(f"[{name}.description]", f"cited_kind {cited!r} is no "
                                             f"kind of the table "
                                             f"({', '.join(table)})")
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


def run_of_points(calculation: str, engine: str) -> Optional[RunOfPoints]:
    """``calculation``'s run of points on ``engine`` (`execution/
    runs-of-points.md` § 2.1) -- ``None`` for a pair whose every rung runs
    once, or no such pair."""
    got = pair(calculation, engine)
    return got.run_of_points if got is not None else None


def own_structure(calculation: str, stage: Optional[str]) -> Optional[str]:
    """What a deck of ``stage`` (its name or its token) of a ``calculation``
    is when its atoms are A STRUCTURE OF ITS OWN -- a lead rung's cell, cut
    from the region its role names (`model/overview.md` § 2.2a, *a structure
    derived from the user's*) -- in words, *a structure of its own (the
    L-electrode cell)*; ``None`` for every other deck."""
    from .identity import stage_name
    got = _table().get(str(calculation))
    region = got.leads.get(stage_name(stage)) if got is not None else None
    return (f"a structure of its own (the {region} cell)" if region
            else None)


def atom_order(calculation: str, engine: str,
               stage: Optional[str] = None) -> Optional[str]:
    """THE ORDER a deck of ``stage`` (its name or its token) of a
    ``calculation`` on ``engine`` lists atoms in -- the pair's
    ``atom_order``, a word of `atom_permutation.SORT_KEY_NAMES` -- or
    ``None``, the user order: a pair that states none, and a deck of a
    structure of its own (:func:`own_structure`).  Prep sorts by it, the
    composer too, and the run door and the working copy read from it which
    runs' decks may be in an engine's order (`model/overview.md` § 2.2a,
    I9)."""
    got = pair(calculation, engine)
    if got is None or got.atom_order is None:
        return None
    return None if own_structure(calculation, stage) else got.atom_order


def labels() -> Tuple[Tuple[str, str], ...]:
    """Every label a kind reads on its structure's atoms, each with what it
    is for (:attr:`CalculationKind.labels`), in the table's order -- what
    the server hands MolView beside every structure, the label palette's
    (`web/molview.md`)."""
    return tuple(lab for rec in _table().values() for lab in rec.labels)


def description_part(calculation: str, key: str) -> Any:
    """THE CODE a ``calculation``'s description names under ``key``
    (:data:`DESCRIPTION_PARTS`), imported now by the floor that asks -- a
    ``module:NAME``'s object, or a module named whole
    (:data:`WHOLE_MODULE_PARTS`), refused naming each door it does not
    provide -- or ``None``: its description names none (a kind that cites
    or extends nothing)."""
    if key not in DESCRIPTION_PARTS:
        raise KindTableError(f"no description part {key!r} -- they are "
                             f"{', '.join(DESCRIPTION_PARTS)}")
    ref = kind(calculation).description.parts.get(key)
    if ref is None:
        return None
    if key not in WHOLE_MODULE_PARTS:
        return part(ref, record=f"{calculation}'s description", key=key)
    return _whole_module(ref, record=f"{calculation}'s description", key=key)


def _whole_module(ref: str, *, record: str, key: str) -> Any:
    """A PART NAMED AS A MODULE WHOLE (:data:`WHOLE_MODULE_PARTS`),
    imported now, and refused -- naming ``record`` and ``key`` -- when it
    does not import, or provides not every door its key lists."""
    doors = WHOLE_MODULE_PARTS[key]
    try:
        module = importlib.import_module(ref)
    except ImportError as exc:
        raise KindTableError(
            f"{record}'s {key}, {ref!r}, does not import: {exc}") from exc
    # A DOOR IS CODE -- a function or a class -- or, as an export's
    # ``ASKS``, the tuple of names it asks.
    lacks = [d for d in doors
             if not (callable(v := getattr(module, d, None))
                     or isinstance(v, tuple))]
    if lacks:
        raise KindTableError(
            f"{record}'s {key}, {ref!r}, provides no {', '.join(lacks)} -- "
            f"a {key} part provides {', '.join(doors)}")
    return module


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
            f"no set of roles {name!r} -- the sets are what a stage "
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


def export_of(calculation: str) -> Any:
    """THE MODULE THAT EXPORTS ``calculation``'s results -- its kind's
    ``export``, imported now by the floor that asks -- or ``None``: the
    kind exports nothing.  A module that does not import, or provides not
    every door :data:`EXPORT_DOORS` lists, is refused naming the kind."""
    ref = kind(calculation).export
    if ref is None:
        return None
    return _whole_module(ref, record=calculation, key="export")


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
           "ROLE_OPENS", "PART_REFS", "FINISHED", "WALL_ONE", "WALL_EACH", "WALLS", "STEP_MOMENTS",
           "RunOfPoints", "run_of_points", "Description", "FixedCharge", "carrying", "KindTableError", "Role", "Result",
           "Retired", "KindOnEngine", "CalculationKind", "kinds", "kind",
           "pair", "lead_of", "rung_label", "role_set", "part_of", "part",
           "export_of", "Region", "DESCRIPTION_PARTS", "DESCRIBE_DOORS", "REUSE_DOORS", "EXPORT_DOORS", "WHOLE_MODULE_PARTS", "own_structure", "atom_order", "labels", "description_part"]
