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
import functools
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
#: The riders: fields that travel ON ``Resources`` without being machine
#: axes.  The note above already drew this line -- *"the two are not the same
#: set"* -- and the implementation did not, taking every field of the struct.
#:
#: ``continue_retries`` was the standing example and never bit, because
#: nobody sweeps a retry budget.  ``use_gpu`` bit immediately: it rides here
#: since 2026-08-23 so the wrapper is told rather than grepping the deck
#: (`execution/gpu.md` G7) -- and it is ALSO the bench's grid-family axis, so
#: classifying it as a machine axis sent the family flag to the allocation and
#: it never reached the config that renders the deck.  Every GPU trial emitted
#: ``Diag.ELPA.GPU .false.``: the sweep asked for a GPU family and measured a
#: CPU one under GPU labels.
#:
#: A rider lands on the VALUES, and `resolve` copies it onto the allocation
#: afterwards -- which is how one value legitimately reaches both the deck and
#: the wrapper without being two facts.
#: ``notify_*`` join them for the same reason ``continue_retries`` did: they
#: ride ``Resources`` to reach the wrapper, and they name no machine.  They
#: differ from both standing examples in where they come from -- not the
#: template's values but the DESCRIPTION's own `notify` block, applied at the
#: same seam as `task.allocation` -- so nothing resolves them and nobody
#: sweeps them.  Listed here so `ALLOCATION_FIELDS` keeps meaning *the
#: machine axes*, which is the question that list actually answers.
_RIDERS: Tuple[str, ...] = ("continue_retries", "use_gpu",
                            "notify_on_scf", "notify_every_hours")

ALLOCATION_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(Resources) if f.name not in _RIDERS)

#: EVERY field ``Resources`` can hold, riders included.  A different question
#: from :data:`ALLOCATION_FIELDS` and it must not be answered with that list:
#: *"is this a machine axis?"* decides where a SWEEP AXIS lands, while *"can
#: Resources hold this?"* decides whether a TRANSLATION's answer is
#: representable.  They were one list until the riders were split out, and the
#: translation check below would then have refused a rider with the reason
#: *"names nothing on Resources"* -- which would have been false.
_RESOURCE_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(Resources))


def _emitter_fields(config_cls) -> tuple:
    """Allocation fields the DECK WRITER also needs, under the config's own
    names -- **derived from the schema, never listed here.**

    Deliberately not every field of ``Resources``: a partition or a wall time
    reaches the wrapper and never the deck, and handing them to the emitter
    would invite it to render one.  The line between them is already drawn on
    the field itself, by the machine-answered flag, so this asks it.

    It was ``_EMITTER_FIELDS = ("mpi_np", "max_memory_mb")`` until 2026-08-17
    -- a hand-kept list of exactly the fields the catalogue already marks,
    and the one that actually drove behaviour while the catalogue's own
    marking was read by nothing.  Adding a fourth machine-answered setting
    changed nothing anywhere; now it arrives here on its own.
    """
    import dataclasses
    return tuple(f.name for f in dataclasses.fields(config_cls)
                 if f.metadata.get("allocation"))


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
        fields = _emitter_fields(type(self.values))
        machine = {k: v for k, v in dataclasses.asdict(self.resources).items()
                   if v is not None and k in fields
                   and hasattr(self.values, k)}
        return (dataclasses.replace(self.values, **machine) if machine
                else self.values)




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

    # § 7's membership rule, spelled ONCE (template.template_fields): a
    # field tagged as the allocation's is a machine fact floor 2 must
    # never carry, so overrides, pins and parameter axes are all gated on
    # the TEMPLATE-ELIGIBLE set.  Until 2026-08-13 every gate here used
    # ``dataclasses.fields`` names, so a hand-edited override or pin
    # naming ``mpi_np`` passed and the deck rendered for a rank count the
    # allocation never granted (final review A-9).
    from .template import template_fields
    known = template_fields(config_cls)
    machine_facts = ({f.name for f in dataclasses.fields(config_cls)}
                     - known)

    stage_obj = _stage_of(task, stage)
    if stage_obj is not None and stage_obj.overrides:
        bad = sorted(set(stage_obj.overrides) & machine_facts)
        if bad:
            raise ResolveError(
                f"stage {stage_obj.name!r} overrides "
                f"{', '.join(map(repr, bad))} -- machine fact(s) the "
                f"description must never carry (engines/template.md § 7): "
                f"they arrive as the ALLOCATION at prep, on the machine "
                f"that runs the job.")
        base = effective_config(base, stage_obj.overrides)
        provenance.update({k: "stage" for k in stage_obj.overrides})

    pins = dict(pins or {})
    bad = sorted(set(pins) & machine_facts)
    if bad:
        raise ResolveError(
            f"pin(s) {', '.join(map(repr, bad))} name machine fact(s) "
            f"(engines/template.md § 7). A pin overrides a template item "
            f"for this prep only; the machine's asks travel as the "
            f"allocation, never as items.")
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
            hint = ""
            mistagged = sorted(set(orphans) & machine_facts)
            if mistagged:
                # e.g. ``omp_threads``: a config field whose EXCHANGE name
                # is the allocation's (cpus_per_task) -- § 7 again, with
                # the road named instead of the bare refusal.
                hint = (f"  ({', '.join(map(repr, mistagged))} is a "
                        f"machine fact -- sweep an allocation axis "
                        f"({', '.join(sorted(ALLOCATION_FIELDS))}) or a "
                        f"declared translation axis instead, "
                        f"engines/template.md § 7.)")
            raise ResolveError(
                f"sweep axis(es) {', '.join(repr(k) for k in orphans)} name "
                f"neither a template item of {config_cls.__name__}, an "
                f"allocation field, "
                f"nor a declared translation axis"
                f"{' (' + ', '.join(sorted(owned)) + ')' if owned else ''}. "
                f"An axis is a parameter, a machine ask, or the "
                f"specialisation's own coordinate (generator.md 4)." + hint)
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
            bad = sorted(set(delta) - set(_RESOURCE_FIELDS))
            if bad:
                raise ResolveError(
                    f"the translation returned "
                    f"{', '.join(repr(k) for k in bad)}, which name nothing "
                    f"on Resources. A translation may only answer with "
                    f"fields the allocation can hold "
                    f"({', '.join(sorted(_RESOURCE_FIELDS))}).")
            machine.update(delta)
        # The bound applies to what the point ASKS FOR, translated or not --
        # a trial whose derived rank count exceeds the allocation is the same
        # refusal as one that named mpi_np directly (§ 4.1).
        _check_fits(machine, allocation)

        values = base
        if params:
            values = effective_config(values, params)
            prov.update({k: "sweep" for k in params})
        if pins:
            values = effective_config(values, pins)
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

        # ``use_gpu`` RIDES THE SAME WAY, 2026-08-23 (`execution/gpu.md` G7).
        # The catalogue item declares `read_by = ["wrapper"]` and the wrapper
        # satisfied that by GREPPING the rendered deck for `Diag.ELPA.GPU` --
        # a layer re-deriving what this one already holds, and doing it by
        # matching a SIESTA keyword, so a PySCF GPU run could not route at
        # all.  An explicitly stated allocation still wins; otherwise the
        # resolved config's answer rides the element, exactly as
        # `continue_retries` does above and for the same reason.
        if resources.use_gpu is None:
            wants_gpu = getattr(values, "use_gpu", None)
            if wants_gpu is not None:
                resources = dataclasses.replace(resources,
                                                use_gpu=bool(wants_gpu))

        elements.append(ResolvedConfig(
            values=values,
            resources=resources,
            point=dict(point),
            label=_label_for(task.label, point),
            provenance=prov,
        ))

    # Two points may render ONE label -- the token drops out-of-set
    # characters (`job-contracts.md` § 6.3), so "ELPA-1" and "ELPA1" would
    # collide -- and a shared label is a shared trial directory and a shared
    # ``SystemLabel``: the second point overwrites the first's warm files
    # and results.  Refused by point, not deduped: both were declared.
    _seen: dict = {}
    for el in elements:
        if el.label in _seen:
            raise ResolveError(
                f"two sweep points render the same label {el.label!r}: "
                f"{dict(_seen[el.label])!r} and {dict(el.point)!r}.  The "
                f"coordinate token stays inside [A-Za-z0-9_] by dropping "
                f"other characters (job-contracts.md 6.3), so distinct "
                f"values can collide -- drop one point or re-spell a "
                f"declared value.")
        _seen[el.label] = el.point
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
                    effective_config(base, s.overrides) if s.overrides else base))
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
    """The rung this prep is for, refusing rather than picking one.

    Never picks the lone stage of a one-rung ladder either:
    `engines/stages.md` § 6.5 makes one
    stage an ordinary stage, so it is named like any other and guessing it
    would be the implicit rule that rule exists to delete.
    """
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


@functools.lru_cache(maxsize=1)
def _catalogue_types() -> Mapping[str, str]:
    """``{item name: declared type}`` from the catalogue, parsed ONCE.

    Cached because the ⊕ operator asks per override and ``resolve`` applies it
    four times per sweep element: parsing an 1100-line file each time measured
    **24 ms per call**, which a sweep multiplies. The catalogue is a shipped
    data file and does not change inside a process.
    """
    from . import template as _t
    try:
        return {i.name: i.type
                for i in _t.catalogue().items}
    except Exception:                       # pragma: no cover - defensive
        return {}


def _annotated(config, name: str):
    """The field's RESOLVED annotation, ``Optional[...]`` unwrapped.

    The fallback both predicates below share when the catalogue has no item
    for a name -- a config class used in a test fixture, or a field that is
    not a template item.  It resolves the annotation properly rather than
    string-matching it, which is the bug they replaced.
    """
    import typing

    from . import template as _t
    try:
        ann = typing.get_type_hints(type(config)).get(name)
    except Exception:
        return None
    inner, _ = _t._unwrap_optional(ann)
    return inner


def _declares_float(config, name: str) -> bool:
    """Whether the CATALOGUE declares *name* a float (`template.md` § 5)."""
    declared = _catalogue_types().get(name)
    if declared is not None:
        return declared == "float"
    return _annotated(config, name) is float


def _declares_a_triple(config, name: str) -> bool:
    """Whether the CATALOGUE declares *name* a fixed-length sequence --
    ``int3`` (``kgrid``) or ``float3`` (``kgrid_displacement``)."""
    import typing
    declared = _catalogue_types().get(name)
    if declared is not None:
        return declared in ("int3", "float3")
    return typing.get_origin(_annotated(config, name)) is tuple


def effective_config(template, overrides: Mapping[str, Any], *,
                     where: str = ""):
    """Resolve one stage against the backbone: **the one place this happens.**

    ``engines/stages.md`` § 4::

        effective config = the template's values ⊕ that stage's ``overrides``

    ``template`` is the science backbone — every field, with the value the
    user set or the default they did not touch.  ``overrides`` is the mapping
    of cells that differ.

    ``where`` names whoever supplied the overrides — a stage's name, usually.
    It is used **only** in the refusal, because *"stage 'tight' overrides a
    field that does not exist"* is findable and *"an override does not exist"*
    is not.  It is deliberately not a parameter of the operation: the operator
    needs the cells, and the label is for the person reading the error.

    **It takes a MAPPING, not a Stage** *(2026-08-14)*.  A stage is where
    overrides usually come from, but the operator needs only the cells: a sweep
    point and a pin are the same shape and neither is a stage.  ``resolve``
    used to fabricate a ``Stage(name="resolve")`` purely to satisfy the old
    signature — packaging invented to fit the parameter rather than the other
    way round.

    **It lives HERE, in floor 3, and not in an engine package** *(moved
    2026-08-14, audit § 6.1 / § 25.1)*.  The body is entirely engine-agnostic —
    it reads the dataclass's own fields — and while it sat under ``siesta/``,
    floor 3 and the validation layer both imported from one engine to do
    something neither engine owns: PySCF had to import SIESTA's module to
    resolve its own stages.  ``generator.md`` § 7's test is that adding an
    engine adds files and edits none.

    Two rules from § 4 shape what this returns, and both are about keeping a
    stage from becoming a special case:

    **R1 — one object is validated and rendered.**  What comes back is an
    ordinary ``SiestaConfig``, so the shipped validator (``validation.validate``)
    and the shipped emitter (``render_fdf``) both take it unchanged.  Nothing
    downstream learns the word "stage".

    **R2 — a stage is validated as a resolved whole, never as a diff.**  Two
    overrides can each be reasonable and jointly wrong, so the caller hands
    the validator *this object*, with the stage's name only as a label.

    **A stage may name ANY field of the shared schema** (§ 1.2).  It is not a
    privileged four: ``mesh_cutoff``, ``basis_size`` and ``kgrid`` were
    unreachable before this function existed, and nothing about them is
    special now.  An override naming a field the schema does not have is
    refused **by name**, which is the half of § 6.6's preflight that
    ``molbuilder/task.py`` could not reach — it has no schema.

    A varied field the stage does *not* name keeps the template's value
    (§ 6.2's subset rule): omitting a key means "this stage is at the
    backbone value", which is what the table draws as a quiet cell.
    """
    known = {f.name for f in dataclasses.fields(type(template))}
    overrides = dict(overrides or {})

    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        raise ValueError(
            f"{where + ': ' if where else ''}override(s) "
            f"{', '.join(repr(k) for k in unknown)} name no field of "
            f"{type(template).__name__}. A stage may override any field of "
            f"the shared schema, but only a field of it "
            f"(engines/stages.md 1.2, 6.6)."
        )

    # An override that arrived from JSON carries JSON's types, and JSON has
    # one number.  ``{"mesh_cutoff": 150}`` is an int where the field declares
    # float, which renders ``MeshCutoff 150 Ry`` where the same value written
    # ``150.0`` renders ``MeshCutoff 150.0 Ry`` -- the same number, a different
    # deck.  Widening int -> float is lossless, so it is done here and the deck
    # reads the same however the description spelled it.
    #
    # NOTHING ELSE is coerced.  ``float -> int`` would silently truncate
    # ``relax_steps: 100.7`` to 100, and a string would quietly parse; both are
    # the caller's mistake and are refused BY NAME in the preflight
    # (``validation/task.py``), which is where a wrong value belongs.  Found by
    # the M2 seam walk, 2026-08-07.
    # **The DECLARED TYPE decides, not the annotation** (audit § 25.3, fixed
    # 2026-08-14).  This read ``f.type`` from the dataclass and compared it
    # against the string ``"float"`` -- and under ``from __future__ import
    # annotations`` a field's ``type`` is the SOURCE TEXT, so ``Optional[float]``
    # is not ``"float"`` and two fields were silently never widened:
    # ``spin_total`` and ``md_target_temperature``.  A stage overriding
    # ``spin_total: 2`` got an int where the deck wanted 2.0, while
    # ``mesh_cutoff: 300`` next to it widened correctly.
    #
    # The catalogue says ``float`` for all three, because that is what a
    # declared type is FOR (`template.md` § 5: what a parser cannot know).  So
    # the authority is the item, and the annotation is not consulted at all.
    #
    # AND JSON HAS ONE SEQUENCE, which is the same argument one shape up
    # (2026-08-25).  ``kgrid`` declares ``Tuple[int, int, int]`` and a
    # description can only spell it ``[4, 4, 1]``, so a stage that overrode
    # it left the config holding a LIST while every stage that did not held
    # the template's tuple -- one object, two shapes for one field,
    # depending on a cell nobody thinks of as a type decision.  Nothing
    # broke, because ``siesta/input.py`` writes ``tuple(cfg.kgrid)`` before
    # comparing; that defensive call IS the symptom.  ``list -> tuple`` is
    # lossless, so it is done here, exactly as ``template._shape`` already
    # does it for the template's own side of the ⊕.
    def _as_declared(k, v):
        if (_declares_float(template, k)
                and isinstance(v, int) and not isinstance(v, bool)):
            return float(v)
        if _declares_a_triple(template, k) and isinstance(v, list):
            return tuple(v)
        return v

    widened = {k: _as_declared(k, v) for k, v in overrides.items()}

    # ``replace`` builds a NEW object, so the template is untouched and every
    # stage resolves against the same backbone regardless of order.
    return dataclasses.replace(template, **widened)


def _label_for(base: str, point: Mapping[str, Any]) -> str:
    """The ``SystemLabel`` in force for this element.

    A production run keeps the calculation's own label. **A trial is
    relabelled**, and that is structural rather than tidy: SIESTA finds its warm
    files by ``SystemLabel``, so a trial carrying the real run's label could
    read — or overwrite — the real run's ``.DM`` and ``.XV``
    (`project-layout.md` § 2.3.2).
    """
    label = base if not point else f"{base}-{point_token(point)}"
    if len(label) > _SIESTA_LABEL_MAX:
        # REFUSED, never truncated (roadmap 7.10 M2, user 2026-08-24).
        # SIESTA silently cuts label-derived filenames at ~50 characters
        # and writes on: the Au-BDT-Au sweep's 58-char trial labels all
        # truncated to one identical 50-char stem -- ELPA1STAGE and
        # ELPA2STAGE trials differ only past the cut, so the two
        # identities COLLIDED, and only their separate directories kept
        # the runs apart.  A label that keys warm files must survive the
        # engine whole, or be refused where changing it is still free.
        raise ResolveError(
            f"the trial label {label!r} is {len(label)} characters; SIESTA "
            f"silently truncates label-derived filenames at about "
            f"{_SIESTA_LABEL_MAX}, which merges trial identities (two "
            f"Au-BDT-Au trials differing only in ELPA1STAGE/ELPA2STAGE "
            f"truncated to one stem, 2026-08-24).  Shorten the "
            f"calculation's label, or the sweep's axis values.")
    return label


#: SIESTA's practical bound on SystemLabel-derived filenames, observed on
#: 5.4.2: a 58-char label produced files cut at 50 characters.  Held a few
#: below the observed cut so suffixes like ``.DM`` stay unambiguous.
_SIESTA_LABEL_MAX = 48


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
    _machine = ("G", "K", "C")
    for k, v in point.items():
        slug = _flat(v)
        if k == "use_gpu" and "G" in point:
            # A RIDER the machine coordinate already states (roadmap 7.10
            # M2): G0 IS use_gpu=False and G>=1 IS use_gpu=True -- spelling
            # both wrote 11 redundant characters into every trial name of
            # the Au-BDT-Au sweep and helped push its labels past SIESTA's
            # limit.  The fact stays in the point's DATA untouched.
            continue
        if k in _machine:
            part = f"{k}{slug}"
        else:
            # A value that names itself (ELPA1STAGE) needs no axis prefix.
            # Only a STRING value can: a number's slug names nothing bare
            # (and a float's spelled decimal point -- 300.5 -> "300p5" --
            # would sneak past a has-a-letter test), a bool's "True" names
            # the wrong thing, and a slug two axes share keeps both keys.
            _self_naming = (isinstance(v, str)
                            and any(c.isalpha() for c in slug)
                            and sum(1 for k2, v2 in point.items()
                                    if k2 not in _machine
                                    and _flat(v2) == slug) == 1)
            part = slug if _self_naming else f"{k}{slug}"
        if not _TOKEN_RE.fullmatch(part):
            # Values were slugged by ``_flat``, so what still fails here is
            # the AXIS name itself, or a value that slugged to nothing while
            # its axis is also empty -- caller errors, not user spellings.
            raise ResolveError(
                f"axis {k}={v!r} renders as {part!r}, which leaves the label "
                f"charset [A-Za-z0-9_] (job-contracts.md 6.3; the token "
                f"becomes a SystemLabel and a directory name, and `-` inside "
                f"it would announce a qualifier that is not there). Spell the "
                f"axis inside the charset, or drop the point.")
        parts.append(part)
    return "".join(parts)


#: One rendered part of the token. ``-`` is excluded on purpose: it is the
#: qualifier separator (§ 6.3), so a value carrying one (``-2``, ``a100:1``)
#: is refused rather than smuggled into a SystemLabel.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+\Z")


def _flat(v: Any) -> str:
    # ``.`` is SPELLED (``p``) because a number's dot carries meaning; every
    # other out-of-set character is DROPPED (`job-contracts.md` § 6.3,
    # 2026-08-21): a value axis carries an engine's own spelling
    # ("ELPA-1Stage"), which the user cannot re-spell, so refusing it would
    # be unactionable.  Dropping is safe because `resolve` refuses two
    # points whose rendered labels collide (the guard below).
    return _DROP_RE.sub("", str(v).replace(".", "p").replace(" ", ""))


_DROP_RE = re.compile(r"[^A-Za-z0-9_]")


__all__ = ["ParameterSet", "ResolvedConfig", "ResolveError", "ALLOCATION_FIELDS",
           "MachineTranslation", "resolve", "point_token"]
