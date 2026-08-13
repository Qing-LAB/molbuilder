"""``resolve`` — the description and the machine become a `ParameterSet`.

**Module:** L2, **floor 3**. Reads the template (floor 2), the description
(floor 2), the ``Environment`` (floor 1), and the allocation; returns objects
and **writes nothing**. Imported by ``jobset/prep``.

**Contract:** [`execution/generator.md`](?doc=execution/generator.md) § 5 (the
object), § 4 (capability ⊇ allocation ⊇ sweep, and the pin channel), § 6.1 (what
floor 3 may read) · [`execution/project-layout.md`](?doc=execution/project-layout.md)
§ 2.3.1 step 2 · [`engines/stages.md`](?doc=engines/stages.md) § 4 (effective
config = the template ⊕ a stage's ``overrides``).

THE ONE IDEA THIS MODULE EXISTS FOR.  **`prep` step 2 always produces a LIST.**
A production run is that list with one element; a benchmark is the same list
with N.  Steps 3, 4 and 5 loop over it without ever asking which they are in, so
there is no ``if benchmark:`` below floor 7 — there is nothing to ask, because
the length is data.

WHAT IT CLOSES.  Until this existed, floor 3 had no input: ``stages_to_jobset``
took an in-memory config assembled from CLI flags, and the description was
emitted *beside* the layout rather than consumed to produce it.  That is the one
defect the 2026-08-11 source read found — *every floor writes its artifact and
reads none of them* — and this module is the edge that was missing.

PRECEDENCE, AND IT IS TOTAL (§ 5)::

    the template's values
      ⊕ this stage's overrides
      ⊕ this sweep point's values
      ⊕ any pin

Every element is renderable on its own, and no downstream reader re-derives a
value.  ``provenance`` records which source set each one, which is what makes
`project-layout.md` M3's *"the numbers were wrong"* answerable rather than a
shrug.

THE ALLOCATION RIDES ON THE ELEMENT, and that is the point.  The deck writer
needs the rank count — ``BlockSize``'s ceiling is orbitals ÷ ranks — and the
wrapper writer needs the whole of it.  Both read **one object**, resolved once.
Three call sites used to build the same wrapper from loose keyword arguments and
one of them forgot ``max_memory_mb``; when the allocation is one object carried
on the element, a call site cannot forget half of it.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, List, Mapping, Optional, Sequence, Tuple,
                    Union)

from .jobset.model import Resources


class ResolveError(Exception):
    """A parameter set could not be resolved. Nothing was rendered."""


#: The allocation's own field names — everything on :class:`Resources`.
#:
#: A sweep axis naming one of these lands on the **resources**; an axis naming
#: anything else lands on the **values** (`generator.md` § 4's two families).
#: One list, so the split is a lookup rather than a judgement.
#:
#: Named for the allocation and **not** "machine axes", because the two are not
#: the same set: ``continue_retries`` rides on ``Resources`` as the exchange
#: struct to the wrapper, but it is a retry *policy* and names no machine, so it
#: is legitimately also a template item. The split this list drives is *"where
#: does this value go"*, which is exactly the allocation's shape.
ALLOCATION_FIELDS: Tuple[str, ...] = tuple(f.name for f in
                                      dataclasses.fields(Resources))


@dataclass(frozen=True)
class ResolvedConfig:
    """One configuration, complete and renderable on its own (§ 5)."""

    #: The resolved engine config — what the deck writer renders.
    values: Any
    #: **This element's own allocation.** Per element, because a sweep over
    #: ``mpi_np`` gives every trial a different rank count.
    resources: Resources
    #: Which sweep coordinate this is; ``{}`` for a production run. It is what
    #: names the trial directory, so a trial needs no naming rule of its own.
    point: Mapping[str, Any] = field(default_factory=dict)
    #: The ``SystemLabel`` in force. A trial's is **relabelled**, which is what
    #: structurally stops a benchmark reading the real run's warm files.
    label: str = ""
    #: ``{field: source}`` — ``template`` · ``stage`` · ``sweep`` · ``pin``.
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_trial(self) -> bool:
        """Whether this element is a sweep point rather than the run itself."""
        return bool(self.point)

    def render_config(self):
        """The config **the deck writer is handed** — values ⊕ the allocation.

        § 5: *"the deck writer needs the rank count — `BlockSize`'s ceiling is
        orbitals ÷ ranks — and the wrapper writer needs the whole of it. Both
        read one object."* This is that one object, projected for the emitter.

        **Why the allocation has to reach the deck at all**, when floor 2 may
        never name a machine: a deck is rendered at `prep`, on the machine that
        will run it, and it records what it assumed in its BENCH-MARKS block. A
        deck rendered without a rank count says ``mpi_np auto`` and then gets
        launched at 32 — which is the disagreement `check_launch_matches_deck`
        was written to catch, and the ``-np 14`` / ``propor: IMAX = 0`` crash
        that prompted it.

        So the machine facts are **not** on the description and **are** on the
        emitter's argument. Those are different objects, and conflating them is
        what put ``mpi_np`` in the template in the first place.
        """
        machine = {k: v for k, v in dataclasses.asdict(self.resources).items()
                   if v is not None and k in _EMITTER_FIELDS
                   and hasattr(self.values, k)}
        return (dataclasses.replace(self.values, **machine) if machine
                else self.values)


#: Allocation fields the DECK WRITER also needs, under the config's own names.
#: Deliberately not every field of ``Resources``: a partition or a wall time
#: reaches the wrapper and never the deck, and handing them to the emitter would
#: invite it to render one.
_EMITTER_FIELDS = ("mpi_np", "max_memory_mb")


@dataclass(frozen=True)
class MachineTranslation:
    """The specialisation's coupling: its own axes → allocation fields.

    `project-layout.md` § 2.3.1a splits the framework from the specialisation,
    and this is the specialisation's half stated as data: **which axes are
    ours** (``axes``) and **what machine ask each point implies**
    (``to_resources``).  The benchmark's is ``("G", "K", "C")`` with
    ``mpi_np = G·K, cpus_per_task = C`` and a ``gres`` read off the
    environment's GPU type — which is why ``to_resources`` receives the
    ``Environment`` as well as the point (`generator.md` § 6.1: the
    environment is one of floor 3's inputs, and the translation is where it
    is consumed).

    **The axes are declared, not inferred**, so the resolver can refuse an
    axis nobody owns by name: a bare callable cannot be asked what it
    consumes, and an axis silently consumed by nothing is *present but not
    honoured* — the shape this design exists to delete.
    """
    #: The axes this translation owns, in the order the grid declares them.
    axes: Tuple[str, ...]
    #: ``(point, environment) -> {allocation field: value}``.
    to_resources: Callable[[Mapping[str, Any], Any], Mapping[str, Any]]


@dataclass(frozen=True)
class ParameterSet:
    """What `prep` step 2 hands to steps 3, 4 and 5 — **always a list**.

    ``len == 1`` is a production run and ``len > 1`` a sweep, and **no reader
    below floor 7 may branch on which**: the length is the whole of the
    difference, which is what makes *kind* a value rather than a fork
    (`architecture.md` § 0's fifth axis).
    """
    elements: Tuple[ResolvedConfig, ...]
    #: Which axes produced this set, in the order they were given. Empty for a
    #: run. Kept because *"what did we vary"* is a question the summary asks and
    #: re-deriving it from the points would be guessing at intent.
    axes: Tuple[str, ...] = ()
    #: The stage this set resolves, or ``None`` for a description with no ladder.
    stage: Optional[str] = None

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

    @property
    def is_sweep(self) -> bool:
        return len(self.elements) > 1


# --------------------------------------------------------------------- #
#  The sweep                                                            #
# --------------------------------------------------------------------- #

def _points(sweep) -> Tuple[Mapping, ...]:
    """The sweep's points, in the order the caller declared them.

    Two spellings, one meaning (`generator.md` § 4): a **mapping of axes**
    (``{axis: [values]}``) is a cross product in declaration order, and a
    **sequence of points** (``[{axis: value, ...}, ...]``) is taken verbatim —
    the form a DEPENDENT enumeration needs, where a later axis's values hang
    on an earlier axis's choice (the benchmark's cores-per-rank bracket
    depends on the ranks-per-GPU pick, so its grid is not a product).

    ``None``, ``{}`` or ``[]`` give **one empty point** rather than none —
    which is what makes a production run "the sweep of length one" instead of
    a case the caller has to remember to special-case.
    """
    if not sweep:
        return ({},)
    if isinstance(sweep, Mapping):
        import itertools
        names = list(sweep)
        combos = itertools.product(*(list(sweep[n]) for n in names))
        return tuple(dict(zip(names, c)) for c in combos)
    return tuple(dict(p) for p in sweep)


def _check_fits(asks: Mapping[str, Any], allocation: Resources) -> None:
    """§ 4.1: *a sweep's ask that exceeds the allocation is refused.*

    ``asks`` is the point's machine half AFTER any translation, so a derived
    rank count is bounded exactly like one the sweep named directly — its keys
    are allocation fields by construction.

    **Not clamped, and not checked against capability instead.** Clamping would
    silently measure something other than what was asked for, and capability is
    the wrong bound: what you may use in this job is what you *asked* for, and
    asking for less is often the better choice (the priority trade).
    """
    for axis, value in asks.items():
        ceiling = getattr(allocation, axis, None)
        if ceiling is None or not isinstance(value, int) \
                or not isinstance(ceiling, int):
            continue
        if value > ceiling:
            raise ResolveError(
                f"this sweep asks for {axis}={value}, which exceeds this "
                f"prep's allocation of {axis}={ceiling}.\n"
                f"  A sweep is bounded by what you ASKED FOR, not by what the "
                f"machine has -- asking for less is often the better choice, "
                f"because how a job is scheduled depends on how much you ask "
                f"for (generator.md 4.1).\n"
                f"  Either raise the allocation for this prep, or drop the "
                f"point. It is refused rather than clamped, because a clamped "
                f"trial measures something you did not ask for.")


# --------------------------------------------------------------------- #
#  The resolver                                                         #
# --------------------------------------------------------------------- #

def resolve(template_text: str, task, config_cls, *,
            allocation: Resources,
            stage: Optional[str] = None,
            sweep: Union[Mapping[str, Sequence[Any]],
                         Sequence[Mapping[str, Any]], None] = None,
            pins: Optional[Mapping[str, Any]] = None,
            translation: Optional[MachineTranslation] = None,
            environment: Any = None) -> ParameterSet:
    """Turn the description and this machine's allocation into a `ParameterSet`.

    ``template_text`` is the calculation's ``<label>.template.toml``; ``task``
    its parsed description; ``config_cls`` the engine's config dataclass.
    ``stage`` names which rung of the ladder to resolve — required when the
    description has one, refused when it does not.

    ``sweep`` is ``{axis: [values]}`` (a cross product) or an explicit
    ``[{axis: value, ...}, ...]`` (a dependent enumeration — see
    :func:`_points`), and turns the result into a list of that many points;
    ``pins`` are the parameters given a value for this prep alone (§ 4.2 —
    ``BlockSize`` is the only member today, and it is a member by rule: its
    right value depends on the allocation).

    ``translation`` is the specialisation's coupling
    (:class:`MachineTranslation`): its declared axes, and what machine ask
    each point implies.  It is an input to the ONE builder rather than a
    second builder: an axis that names neither a schema field, an allocation
    field, nor a declared translation axis is refused by name, and what the
    translation returns is bounded by the allocation exactly like a
    directly-named machine axis (§ 4.1).  ``environment`` is floor 1's answer
    (`generator.md` § 6.1) and is handed to the translation, which is where a
    machine fact like the GPU type is legitimately read.

    Raises :class:`ResolveError` when the stage does not exist, when a sweep
    point exceeds the allocation, when an axis names nothing, or when a pin
    names something the schema does not have.
    """
    from .template import config_from_template

    base = config_from_template(template_text, config_cls)
    provenance = {f.name: "template" for f in dataclasses.fields(config_cls)}

    stage_obj = _stage_of(task, stage)
    if stage_obj is not None and stage_obj.overrides:
        base = _apply(base, stage_obj.overrides)
        provenance.update({k: "stage" for k in stage_obj.overrides})

    known = {f.name for f in dataclasses.fields(config_cls)}
    pins = dict(pins or {})
    unknown = sorted(set(pins) - known)
    if unknown:
        raise ResolveError(
            f"pin(s) {', '.join(repr(k) for k in unknown)} name nothing in the "
            f"{config_cls.__name__} schema. A pin overrides a template item for "
            f"this prep only; it cannot invent one.")

    elements: List[ResolvedConfig] = []
    points = _points(sweep)
    for point in points:
        prov = dict(provenance)

        # A point's axes split by WHERE they land, and the split is a lookup
        # rather than a judgement: a name on Resources is a machine axis, a
        # name in the schema is a parameter axis (generator.md § 4), and a
        # name on the translation's declared axes is the specialisation's own
        # coordinate.  An axis on none of the three lists names nothing, and
        # is refused BY NAME — with or without a translation, because an axis
        # silently consumed by nothing would be present but not honoured.
        machine = {k: v for k, v in point.items() if k in ALLOCATION_FIELDS}
        params = {k: v for k, v in point.items()
                  if k not in ALLOCATION_FIELDS and k in known}
        owned = set(translation.axes) if translation is not None else set()
        orphans = sorted(set(point) - set(machine) - set(params) - owned)
        if orphans:
            raise ResolveError(
                f"sweep axis(es) {', '.join(repr(k) for k in orphans)} name "
                f"neither a {config_cls.__name__} field, an allocation field, "
                f"nor a declared translation axis"
                f"{' (' + ', '.join(sorted(owned)) + ')' if owned else ''}. "
                f"An axis is a parameter, a machine ask, or the "
                f"specialisation's own coordinate (generator.md 4).")
        ours = [a for a in (translation.axes if translation else ()) if a in point]
        if ours:
            missing = [a for a in translation.axes if a not in point]
            if missing:
                raise ResolveError(
                    f"point {dict(point)!r} carries translation axis(es) "
                    f"{', '.join(repr(a) for a in ours)} but not "
                    f"{', '.join(repr(a) for a in missing)}. A translation's "
                    f"axes travel together; a partial coordinate would make "
                    f"the machine ask a guess.")
            delta = dict(translation.to_resources(point, environment))
            bad = sorted(set(delta) - set(ALLOCATION_FIELDS))
            if bad:
                raise ResolveError(
                    f"the translation returned "
                    f"{', '.join(repr(k) for k in bad)}, which name nothing "
                    f"on Resources. A translation may only answer with "
                    f"allocation fields "
                    f"({', '.join(sorted(ALLOCATION_FIELDS))}).")
            machine.update(delta)
        # The bound applies to what the point ASKS FOR, translated or not --
        # a trial whose derived rank count exceeds the allocation is the same
        # refusal as one that named mpi_np directly (§ 4.1).
        _check_fits(machine, allocation)

        values = base
        if params:
            values = _apply(values, params)
            prov.update({k: "sweep" for k in params})
        if pins:
            values = _apply(values, pins)
            prov.update({k: "pin" for k in pins})

        resources = dataclasses.replace(allocation, **machine)
        # § 6.2's translation boundary (job-contracts.md): floor 3 maps
        # config → exchange HERE, and ``continue_retries`` is the one
        # rider that is legitimately also a template item (a retry
        # POLICY, not a machine fact — ALLOCATION_FIELDS' own note).  An
        # explicitly stated allocation wins; otherwise the resolved
        # config's answer rides the element, exactly as the § 6.2 row
        # ("translated at resolve.py") describes.  Until 2026-08-13 the
        # note above CLAIMED the ride and nothing performed it: the web
        # route handed the value straight to the wrapper writer while the
        # CLI route resolved an allocation that never carried it, so the
        # wrapper rendered NO retry loop and `job-system.md § 4.1`'s
        # "travels the whole way" was true of one road out of two
        # (final review A-5).
        if resources.continue_retries is None:
            budget = getattr(values, "continue_retries", None)
            if budget is not None:
                resources = dataclasses.replace(resources,
                                                continue_retries=budget)

        elements.append(ResolvedConfig(
            values=values,
            resources=resources,
            point=dict(point),
            label=_label_for(task.label, point),
            provenance=prov,
        ))

    return ParameterSet(elements=tuple(elements),
                        axes=_axes_of(sweep, points),
                        stage=(stage_obj.name if stage_obj else None))


def resolved_ladder(template_text: str, task, config_cls) -> List[Tuple[str, Any]]:
    """``[(stage name, resolved config)]`` for every ENABLED stage, in
    ladder order — the sequence checks' input (`engines/stages.md` § 6.4 /
    § 6.6a), resolved by the SAME primitives as `prep` step 2 (template ⊕
    stage overrides), so what the checks compare is what the decks would
    say.  No machine fact is involved: a sequence question never needs the
    allocation.

    Exists so the surface that surfaces those findings does not re-derive
    the resolution with primitives of its own — the caller-re-derivation
    habit `job-system.md` § 9 diagnoses (added with A-8, 2026-08-13, which
    found the § 6.6a warning had no production caller at all).
    """
    from .template import config_from_template
    base = config_from_template(template_text, config_cls)
    out: List[Tuple[str, Any]] = []
    for s in (task.stages or ()):
        if not getattr(s, "enabled", True):
            continue
        out.append((s.name,
                    _apply(base, s.overrides) if s.overrides else base))
    return out


def _axes_of(sweep, points: Tuple[Mapping, ...]) -> Tuple[str, ...]:
    """Which axes produced this set, in the order they were given.

    The mapping form declares them directly; the explicit-points form carries
    them in each point, so the first point's keys speak for the set (every
    point of one enumeration answers the same question).
    """
    if not sweep:
        return ()
    if isinstance(sweep, Mapping):
        return tuple(sweep)
    return tuple(points[0]) if points and points[0] else ()


def _stage_of(task, stage: Optional[str]):
    """The rung this prep is for, refusing rather than picking one."""
    if not task.stages:
        if stage:
            raise ResolveError(
                f"this description has no ladder, so there is no stage "
                f"{stage!r} to resolve. A calculation with a single parameter "
                f"set is prepped without naming one (engines/stages.md 6.5).")
        return None
    if not stage:
        raise ResolveError(
            f"this description has a ladder, so a stage has to be named. "
            f"Available: {', '.join(s.name for s in task.stages)}.")
    for s in task.stages:
        if s.name == stage:
            return s
    raise ResolveError(
        f"no stage named {stage!r} in this description. Available: "
        f"{', '.join(s.name for s in task.stages)}.")


def _apply(cfg, overrides: Mapping[str, Any]):
    """The template ⊕ overrides seam — ``engines/stages.md`` § 4's one place.

    ``effective_config`` lives under ``siesta/`` for historical reasons and its
    body is **entirely engine-agnostic**: it checks each key against the
    dataclass's own fields, widens ``int`` to ``float`` where the field declares
    one, and returns a new object. Calling it here keeps *"the one place this
    happens"* true rather than growing a second implementation beside it.
    """
    from .siesta.input import effective_config
    from .task import Stage
    return effective_config(cfg, Stage(name="resolve", overrides=dict(overrides)))


def _label_for(base: str, point: Mapping[str, Any]) -> str:
    """The ``SystemLabel`` in force for this element.

    A production run keeps the calculation's own label. **A trial is
    relabelled**, and that is structural rather than tidy: SIESTA finds its warm
    files by ``SystemLabel``, so a trial carrying the real run's label could
    read — or overwrite — the real run's ``.DM`` and ``.XV``
    (`project-layout.md` § 2.3.2).
    """
    return base if not point else f"{base}-{point_token(point)}"


def point_token(point: Mapping[str, Any]) -> str:
    """A sweep coordinate as one filename-safe token — ``G1K4C6``.

    **One function, fed by data** — which is why a trial directory needs no
    naming rule of its own (`generator.md` § 5). Keys in the order the sweep
    declared them, because that is the order a person wrote and reading a
    directory listing should match the command they typed.

    **Concatenated, no inner separator** — `job-contracts.md` § 6.3 (the
    cross-layer authority) renders the benchmark's coordinate ``bench-G1K4C6``:
    the ``-`` announces ONE qualifier, so a separator inside the token would
    read as more of them.  This joined with ``-`` until the bench fold (C6,
    2026-08-11), which is when the G/K/C abbreviation stopped being a second
    rendering in the benchmark module and became this function's ordinary
    output.  The token is an identifier, never a parser target: what varied
    lives in ``ParameterSet.axes`` and each element's ``point``, as data.
    """
    parts = []
    for k, v in point.items():
        part = f"{k}{_flat(v)}"
        if not _TOKEN_RE.fullmatch(part):
            raise ResolveError(
                f"axis {k}={v!r} renders as {part!r}, which leaves the label "
                f"charset [A-Za-z0-9_] (job-contracts.md 6.3; the token "
                f"becomes a SystemLabel and a directory name, and `-` inside "
                f"it would announce a qualifier that is not there). Spell the "
                f"value inside the charset, or drop the point.")
        parts.append(part)
    return "".join(parts)


#: One rendered part of the token. ``-`` is excluded on purpose: it is the
#: qualifier separator (§ 6.3), so a value carrying one (``-2``, ``a100:1``)
#: is refused rather than smuggled into a SystemLabel.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+\Z")


def _flat(v: Any) -> str:
    return str(v).replace(".", "p").replace(" ", "")


__all__ = ["ParameterSet", "ResolvedConfig", "ResolveError", "ALLOCATION_FIELDS",
           "MachineTranslation", "resolve", "point_token"]
