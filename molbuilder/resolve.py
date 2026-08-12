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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .jobset.model import Resources


class ResolveError(Exception):
    """A parameter set could not be resolved. Nothing was rendered."""


#: The allocation's own field names — everything on :class:`Resources`. A sweep
#: axis naming one of these is a **machine** axis and lands on the resources; an
#: axis naming anything else is a **parameter** axis and lands on the values
#: (`generator.md` § 4). One list, so the split is a lookup and not a guess.
MACHINE_AXES: Tuple[str, ...] = tuple(f.name for f in
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

def _points(sweep: Optional[Mapping[str, Sequence[Any]]]) -> Tuple[Tuple, ...]:
    """The cross product of the declared axes, in declaration order.

    ``None`` or ``{}`` gives **one empty point** rather than none — which is
    what makes a production run "the sweep of length one" instead of a case the
    caller has to remember to special-case.
    """
    if not sweep:
        return ({},)
    import itertools
    names = list(sweep)
    combos = itertools.product(*(list(sweep[n]) for n in names))
    return tuple(dict(zip(names, c)) for c in combos)


def _check_fits(point: Mapping[str, Any], allocation: Resources) -> None:
    """§ 4.1: *a sweep point that exceeds the allocation is refused.*

    **Not clamped, and not checked against capability instead.** Clamping would
    silently measure something other than what was asked for, and capability is
    the wrong bound: what you may use in this job is what you *asked* for, and
    asking for less is often the better choice (the priority trade).
    """
    for axis, value in point.items():
        if axis not in MACHINE_AXES:
            continue
        ceiling = getattr(allocation, axis, None)
        if ceiling is None or not isinstance(value, int) \
                or not isinstance(ceiling, int):
            continue
        if value > ceiling:
            raise ResolveError(
                f"sweep point {axis}={value} exceeds this prep's allocation "
                f"of {axis}={ceiling}.\n"
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
            sweep: Optional[Mapping[str, Sequence[Any]]] = None,
            pins: Optional[Mapping[str, Any]] = None) -> ParameterSet:
    """Turn the description and this machine's allocation into a `ParameterSet`.

    ``template_text`` is the calculation's ``<label>.template.toml``; ``task``
    its parsed description; ``config_cls`` the engine's config dataclass.
    ``stage`` names which rung of the ladder to resolve — required when the
    description has one, refused when it does not.

    ``sweep`` is ``{axis: [values]}`` and turns the result into a list of that
    many points; ``pins`` are the parameters given a value for this prep alone
    (§ 4.2 — ``BlockSize`` is the only member today, and it is a member by rule:
    its right value depends on the allocation).

    Raises :class:`ResolveError` when the stage does not exist, when a sweep
    point exceeds the allocation, or when a pin names something the schema does
    not have.
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
    for point in _points(sweep):
        _check_fits(point, allocation)
        prov = dict(provenance)

        # A point's axes split by WHERE they land, and the split is a lookup
        # rather than a judgement: a name on Resources is a machine axis, and
        # anything else is a parameter axis (generator.md § 4).
        machine = {k: v for k, v in point.items() if k in MACHINE_AXES}
        params = {k: v for k, v in point.items() if k not in MACHINE_AXES}

        values = base
        if params:
            values = _apply(values, params)
            prov.update({k: "sweep" for k in params})
        if pins:
            values = _apply(values, pins)
            prov.update({k: "pin" for k in pins})

        elements.append(ResolvedConfig(
            values=values,
            resources=dataclasses.replace(allocation, **machine),
            point=dict(point),
            label=_label_for(task.label, point),
            provenance=prov,
        ))

    return ParameterSet(elements=tuple(elements),
                        axes=tuple(sweep or ()),
                        stage=(stage_obj.name if stage_obj else None))


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
    """A sweep coordinate as one filename-safe token.

    **One function, fed by data** — which is why a trial directory needs no
    naming rule of its own (`generator.md` § 5). Keys in the order the sweep
    declared them, because that is the order a person wrote and reading a
    directory listing should match the command they typed.

    *(The benchmark's own `G<g>K<k>C<c>` abbreviation is a second rendering of
    this same idea and folds into it when `bench` does — `job-contracts.md`
    § 6.3, scheduled with the merge rather than guessed at here.)*
    """
    return "-".join(f"{k}{_flat(v)}" for k, v in point.items())


def _flat(v: Any) -> str:
    return str(v).replace(".", "p").replace(" ", "")


__all__ = ["ParameterSet", "ResolvedConfig", "ResolveError",
           "MACHINE_AXES", "resolve", "point_token"]
