"""The preflight's other half — the checks that need a field schema.

**Module:** L2, inside the ``validation`` package. Imports ``issues`` and
``task`` (both L1), ``template`` (L2) and an engine's config (L1) — the config
lazily, so importing ``validation`` does not drag the engine stack in. To be
called by ``describe._check`` on the host, ``prep``'s § 6.6 preflight
on the target (R5, 2026-08-12), and the Task-setup save door (gate ③,
2026-08-21).

**Contract:** [`engines/stages.md`](?doc=engines/stages.md) § 6.6 (the eight
checks, *"in order, and all of it before anything is written"*) · § 6.7 (`shape`
is required and never inferred) · [`science/validation.md`](?doc=science/validation.md)
§ 4.1 (one result type; ``where`` is the stable id; severity means the same
everywhere).

**Why this is a second file and not more of `task.py`.** § 6.6's eight checks
split cleanly: four are answerable from the description alone and live in the
codec, and four need the engine's field schema -- one of those four being the
DECLARED-TYPE row (added 2026-08-25), which needs the schema for the same
reason the bounds row does: only the field knows what it can hold. ``task.py`` is L1 — it imports
``persist`` and the standard library — and importing an engine into it is
exactly what ``tests/test_layering.py`` prevents. Its docstring carries the same
split, so the two halves cannot quietly diverge; **if you add a row here, add it
there.**

**Why this returns findings rather than raising.** § 6.6's refusal rows
become ``error`` Issues, and the sequence checks (§ 6.4 / § 6.6a) are
``warn`` — a function that raises could not express *"proceed, and say
so"* at all.  (The schema-fingerprint row — the original non-refusal —
retired 2026-08-14 with the fingerprint itself; `stages.md` § 6.6 records
the deletion.)  :func:`refuse_on_error` is the caller's one line when it
wants the exception. That also puts these on the one channel into the UI
(§ 4.1 R2), which is the point: a check nobody sees is worse than no check.

``task.py``'s half still raises, and that is not an inconsistency — it fails
while *parsing*, where there is no object yet to attach a finding to.
"""
from __future__ import annotations

import dataclasses
import re
import typing
from typing import Any, Dict, List, Optional

from ..issues import Issue, ValidationError


def preflight(task, config_cls=None, *,
              generators: Optional[Dict[str, Any]] = None,
              template_text: Optional[str] = None) -> List[Issue]:
    """§ 6.6's schema-dependent half, in the order the contract lists them.

    ``task`` is a parsed :class:`molbuilder.task.Task` — the codec's own four
    checks have already passed, or there would be no object. ``config_cls`` is
    the engine's config dataclass; when omitted it is resolved from
    ``task.engine``, and *that resolution failing is itself the first check*.

    ``generators`` maps engine name → config class, and defaults to what this
    backend actually ships. It is an argument so a test can ask "what happens
    for an engine we do not have" without inventing one in the registry.

    ``template_text``, when the caller has it, adds the SEQUENCE findings —
    § 6.4's loosening ladder and § 6.6a's identical-and-clean recompute.
    Both compare RESOLVED stages, so they need the template, and both are
    *"a warning, not a preflight row"* (§ 6.6a): they proceed.  Optional
    because half of § 6.6 is answerable from the description alone; but
    until 2026-08-13 NO production surface ran the sequence checks at all —
    `stages.md` :884 said "implemented" over a function only tests called
    (final review A-8).

    Every finding names what it refused (§ 6.6's right-hand column) — this is a
    file people edit by hand, so a refusal owes them the offending key, the
    stage it was in, and the bound it broke.
    """
    known = dict(generators if generators is not None else _shipped())
    out: List[Issue] = []

    # -- 1. the engine is one this backend has a generator for -------------
    if task.engine not in known:
        return [Issue(
            "error",
            f"engine {task.engine!r} has no generator in this backend. "
            f"Available: {', '.join(sorted(known)) or '(none)'}",
            where="task.engine")]
    cls = config_cls if config_cls is not None else known[task.engine]

    # -- 1b. § 6.8 -- only a SPEED knob may be swept -----------------------
    #
    # The codec checked the shape of ``bench`` and stopped there: deciding
    # whether a name is sweepable means reading the catalogue, and `task.py`
    # is L1.  Here we can.
    #
    # The rule is `template.md` § 6.2's own definition of ``execution``:
    # *"knobs that change speed and not the answer"*.  Sweep something
    # outside it and each point measures a DIFFERENT calculation -- so the
    # comparison the benchmark exists to make says nothing, and it says
    # nothing silently, which is why this is an error rather than a warning.
    out.extend(_bench_names_a_speed_knob(task, cls))
    out.extend(_execution_names_a_speed_knob(task))
    out.extend(_bench_points_fit_their_items(task))

    # (step 2 -- the schema fingerprint -- retired 2026-08-14 with the
    # fingerprint itself; stages.md § 6.6 records the deletion)

    # -- 3. every named field exists in the schema ------------------------
    fields = {f.name: f for f in dataclasses.fields(cls)}
    out.extend(_names_exist(task, cls, fields))

    # -- 4. every value is inside the schema's bounds ---------------------
    #  ...which begins with being a value that field can HOLD.  Found by the
    #  M2 seam walk (2026-08-07): `relax_steps: 100.7` is inside its range and
    #  is not an integer, and reached the deck as `MD.NumCGsteps 100.7`.
    out.extend(_values_are_the_declared_type(task, cls, fields))
    out.extend(_values_in_bounds(task, fields))

    # -- 5. the sequence's own findings (§ 6.4 / § 6.6a) -- warnings -------
    if template_text is not None and task.stages:
        from ..resolve import resolved_ladder
        from .stages import (check_identical_stages,
                             check_ladder_does_not_loosen)
        try:
            ladder = resolved_ladder(template_text, task, cls)
        except Exception:
            # An unresolvable template earns prep's OWN refusal moments
            # later, with its own wording (a template it cannot rebuild
            # from) -- a sequence WARNING must not preempt it as a crash.
            ladder = []
        out.extend(check_ladder_does_not_loosen(ladder))
        out.extend(check_identical_stages(ladder))
    return out



def _bench_names_a_speed_knob(task, cls) -> List[Issue]:
    """§ 6.8: every ``bench`` key is an ``execution``-category field.

    Read off the CONFIG CLASS's own metadata rather than a list kept here --
    a second list of which parameters are sweepable is the copy that goes
    stale, and this codebase has four documented cases of exactly that.
    """
    plan = getattr(task, "bench", None)
    if not plan:
        return []
    sweepable = {f.name for f in dataclasses.fields(cls)
                 if "execution" in (f.metadata.get("category") or ())}
    out: List[Issue] = []
    for name in sorted(plan):
        if name in sweepable:
            continue
        known = ", ".join(sorted(sweepable)) or "(none)"
        out.append(Issue(
            "error",
            f"bench names {name!r}, which {task.engine} does not declare as "
            f"an `execution` parameter -- so sweeping it would measure a "
            f"different calculation at each point and the comparison would "
            f"mean nothing. Sweepable here: {known}",
            where=f"task.bench.{name}"))
    return out


def _execution_names_a_speed_knob(task) -> List[Issue]:
    """`stages.md` § 6.8d: every name in an ``execution`` block -- the
    calculation's or a stage's -- is an `execution` catalogue item.

    THE SAME MEMBERSHIP `bench` USES, asked here because the two blocks are
    one vocabulary at two arities.  A name outside it would be carried to
    prep and refused there, or worse, carried into a deck as a value nobody
    can read back.
    """
    from ..template import catalogue, select
    blocks = [("execution", getattr(task, "execution", None) or {})]
    for st in (getattr(task, "stages", None) or ()):
        if getattr(st, "execution", None):
            blocks.append((f"stage {st.name!r} execution", dict(st.execution)))
    if not any(b for _w, b in blocks):
        return []
    # THE CATALOGUE'S `execution` ITEMS, PLUS THE TWO LANE ASKS.  `time` and
    # `domain` are not catalogue items -- an engine has no opinion about a
    # wall clock or a queue -- but a RUN owns them separately from the bench
    # (`stages.md` § 6.8e), so they are admitted here by name.
    from ..task import LANE_ASKS
    known = {i.name for i in select(catalogue(), engine=task.engine)
             if "execution" in (i.category or ())} | set(LANE_ASKS)
    out: List[Issue] = []
    for where, block in blocks:
        for name in sorted(block):
            if name in known:
                continue
            out.append(Issue(
                "error",
                f"{where} names {name!r}, which is not an execution setting "
                f"of {task.engine}.  This block says what the RUN uses -- "
                f"ranks, threads, the device, the solver.  A physics value "
                f"belongs in the template, and a per-stage one in that "
                f"stage's `overrides` (engines/stages.md 6.2).  Execution "
                f"settings here: {', '.join(sorted(known)) or '(none)'}",
                where=f"task.{where}.{name}"))
    return out


def _bench_points_fit_their_items(task) -> List[Issue]:
    """`generator.md` § 4.3a's shape half at DESCRIBE time: every declared
    point must fit its item -- a bool item takes true/false, an enum point
    must be one of the item's choices, and a repeated point would measure
    one configuration twice.  THE ONE HOME of those rules (R2-5 dedup,
    2026-08-21): `jobset/_cli.py::_declared_execution_pins` calls this
    same function as its backstop instead of carrying a copy -- the copies
    had diverged (allocation-item duplicates were caught only here).
    Surfaced at save so a typo'd declaration fails there, not after a
    queue on the cluster.  (Found
    live 2026-08-21: a matrix saved through the pre-U1 UI spelled
    'ELPA-1Stage' where the catalogue's choice is 'ELPA-1STAGE', and the
    first surface to say so was `prep bench` on Sol.)

    Membership is the sibling's question; an unknown name is skipped here
    because `_bench_names_a_speed_knob` already reported it.
    """
    plan = getattr(task, "bench", None)
    if not plan:
        return []
    from ..template import catalogue, select
    items = {i.name: i for i in select(catalogue(), engine=task.engine)
             if "execution" in (i.category or ())}
    out: List[Issue] = []
    for name in sorted(plan):
        it = items.get(name)
        if it is None:
            continue
        pts = list(plan[name])
        for v in pts:
            if it.type == "bool" and not isinstance(v, bool):
                out.append(Issue(
                    "error",
                    f"bench declares {name!r} = {v!r}; the item is a bool "
                    f"-- write true or false.",
                    where=f"task.bench.{name}"))
            elif it.type == "enum" and it.choices and v not in it.choices:
                out.append(Issue(
                    "error",
                    f"bench declares {name!r} = {v!r}; the choices are "
                    f"{', '.join(it.choices)}.",
                    where=f"task.bench.{name}"))
        if len(pts) != len(set(pts)):
            out.append(Issue(
                "error",
                f"bench declares {name!r} = {pts!r}; a repeated point "
                f"would measure one configuration twice.",
                where=f"task.bench.{name}"))
    return out


def _shipped() -> Dict[str, Any]:
    """The engines this backend has a generator for, resolved lazily.

    Lazily because importing an engine config at module scope would drag the
    whole engine stack into anything that imports ``validation``.
    """
    from ..config.pyscf import PySCFConfig
    from ..config.siesta import SiestaConfig
    return {"siesta": SiestaConfig, "pyscf": PySCFConfig}


def _names_exist(task, cls, fields) -> List[Issue]:
    """Every name in ``varies`` and every ``overrides`` key is a real field
    — a real TEMPLATE field: a machine fact refuses with § 7's story.

    ``varies`` is checked too, not only the overrides. ``task.py`` enforces
    ``overrides ⊆ varies`` without a schema, so a *misspelt* name that appears
    in both passes the codec cleanly — the subset holds, and both are wrong the
    same way. This is the only place that can catch it.

    A name that IS a field but is tagged as the allocation's (``mpi_np``,
    ``omp_threads``, ``max_memory_mb``) refuses too, with the machine-fact
    story rather than the typo story: floor 2 must never carry it
    (engines/template.md § 7), and until 2026-08-13 the existence check
    admitted it — a description varying ``mpi_np`` rendered a deck for a
    rank count the allocation never granted (final review A-9).
    """
    def _machine(name) -> bool:
        return bool(fields[name].metadata.get("allocation"))

    def _machine_msg(lead) -> str:
        return (f"{lead} a machine fact the description must never carry "
                f"(engines/template.md § 7): it arrives as the ALLOCATION "
                f"at prep, on the machine that runs the job")

    out: List[Issue] = []
    for name in (task.varies or ()):
        if name not in fields:
            out.append(Issue("error", _no_such_field(
                name, fields, cls, "'varies' names"), where="task.varies"))
        elif _machine(name):
            out.append(Issue("error", _machine_msg(
                f"'varies' names {name!r},"), where="task.varies"))
    for st in (task.stages or ()):
        for key in st.overrides:
            if key not in fields:
                out.append(Issue("error", _no_such_field(
                    key, fields, cls, f"stage {st.name!r} overrides"),
                    where="task.stages.overrides", stage=st.name))
            elif _machine(key):
                out.append(Issue("error", _machine_msg(
                    f"stage {st.name!r} overrides {key!r},"),
                    where="task.stages.overrides", stage=st.name))
    return out


def _no_such_field(name, fields, cls, lead) -> str:
    import difflib
    near = difflib.get_close_matches(name, list(fields), n=1, cutoff=0.75)
    hint = f" -- did you mean {near[0]!r}?" if near else ""
    return (f"{lead} {name!r}, which is not a field of "
            f"{cls.__name__}{hint}")


def _values_are_the_declared_type(task, cls, fields) -> List[Issue]:
    """An override's value must be one the field can actually hold.

    **Refused rather than coerced**, and the two halves of that decision are
    deliberate. ``effective_config`` widens ``int -> float`` because JSON has
    one number and ``150`` for a float field is the same value written
    differently — lossless, so silence is honest. Everything else is the
    caller's mistake: coercing ``relax_steps: 100.7`` to 100 would silently
    run a different calculation from the one described, and parsing ``"150"``
    would make a quoting slip invisible.

    **The annotation is RESOLVED, never sniffed as source text**
    *(2026-08-25)*. This read ``f.type`` and compared it against the literal
    strings ``"int"`` / ``"float"`` / ``"bool"``, so under ``from __future__
    import annotations`` — where a field's ``type`` IS its source text — it
    recognised those three spellings and nothing else. Every ``Optional[…]``
    field and every sequence field went unchecked: ``kgrid``,
    ``kgrid_displacement``, ``species_order``, ``ecp_atoms``. A description
    carrying ``"kgrid": "4,4,1"`` therefore passed this gate, saved cleanly,
    resolved into a config holding a string where a triple belongs, and died
    at ``prep`` inside the metadata range check as *"this is a programmer
    bug"* — a message that names neither the stage nor the key, because at
    that depth neither is still in hand (found live 2026-08-25). It is the
    same trick one file over that ``resolve._declares_float`` records itself
    getting wrong, for the same reason.

    Found by the M2 seam walk. Neither side could see it alone — ``task.py``
    has no schema to check a type against, and ``effective_config`` had no
    reason to think a value might not be one.
    """
    hints = typing.get_type_hints(cls)
    out: List[Issue] = []
    for st in (task.stages or ()):
        for key, value in st.overrides.items():
            f = fields.get(key)
            if f is None:
                continue                    # already reported by _names_exist
            bad = _type_complaint(hints.get(key, f.type), value)
            if bad:
                out.append(Issue(
                    "error",
                    f"stage {st.name!r} sets {key} = {value!r}, {bad}",
                    where=f"config.{key}", stage=st.name))
    return out


#: A declared type in the words someone editing the file would use. § 6.6's
#: row asks a refusal to name *what the field declares*, and "not a whole
#: number" does not tell a person who typed ``4,4,1`` that ``[4, 4, 1]`` is
#: what the description wants.
_PLURAL = {int: "whole numbers", float: "numbers", str: "text values",
           bool: "true/false values"}
_COUNT = {1: "one", 2: "two", 3: "three", 4: "four"}


def _spell(declared) -> str:
    origin = typing.get_origin(declared)
    args = [a for a in typing.get_args(declared) if a is not Ellipsis]
    elem = args[0] if args else None
    if origin is tuple and args:
        return (f"{_COUNT.get(len(args), str(len(args)))} "
                f"{_PLURAL.get(elem, 'values')}, written as a list")
    if origin is list:
        return f"a list of {_PLURAL.get(elem, 'values')}"
    return {int: "a whole number", float: "a number", bool: "true or false",
            str: "text"}.get(declared,
                             getattr(declared, "__name__", str(declared)))


def _type_complaint(declared, value) -> Optional[str]:
    """Why *value* cannot be this field's, or ``None`` if it can."""
    origin = typing.get_origin(declared)
    args = typing.get_args(declared)

    # ``Optional[X]`` — ``None`` is always legal and the rest is X's question.
    # A wider union declares no single shape to name, so it is left alone
    # rather than guessed at.
    if origin is typing.Union:
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _type_complaint(inner[0], value) if len(inner) == 1 else None

    if origin in (tuple, list):
        want = f"which is not {_spell(declared)}"
        elem = next((a for a in args if a is not Ellipsis), None)
        # A ``str`` is itself a sequence, and treating it as one is exactly
        # how ``"4,4,1"`` reached a ``Tuple[int, int, int]``: it has three
        # of something, and every check that asked only *"is it a sequence"*
        # said yes.  So it is refused FIRST, by name, with the list it was
        # trying to be.
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            return want + _write_it_as_a_list(value, args, origin)
        if origin is tuple and Ellipsis not in args and len(value) != len(args):
            return f"{want} -- it has {len(value)}, not {len(args)}"
        for v in value:
            if _scalar_complaint(elem, v):
                return f"{want} -- {v!r} is not one"
        return None

    return _scalar_complaint(declared, value)


def _write_it_as_a_list(value, args, origin) -> str:
    """The comma-text a person typed, shown back as the list to write.

    Only when it would actually BE the value — the same count, and every
    piece the element type. A guess that does not parse is worse than no
    guess, so there is no partial version of this.
    """
    if not isinstance(value, str):
        return ""
    pieces = [p for p in re.split(r"[,\sx]+", value.strip()) if p]
    if origin is tuple and Ellipsis not in args and len(pieces) != len(args):
        return ""
    elem = next((a for a in args if a is not Ellipsis), None)
    try:
        shaped = [elem(p) for p in pieces] if elem in (int, float) else pieces
    except (TypeError, ValueError):
        return ""
    return f" -- write {list(shaped)!r}"


def _scalar_complaint(declared, value) -> Optional[str]:
    """The same question for a single value. A type with no rule here is not
    a refusal: this gate says what it knows and leaves the rest to bounds."""
    is_bool = isinstance(value, bool)
    if declared is bool:
        return None if is_bool else "which is not true or false"
    if declared is int:
        if is_bool or not isinstance(value, (int, float)):
            return "which is not a whole number"
        if isinstance(value, float) and not value.is_integer():
            return (f"which is not a whole number -- this field counts "
                    f"things, so {value!r} would have to be rounded and the "
                    f"run would not be the one described")
        return None
    if declared is float:
        if is_bool or not isinstance(value, (int, float)):
            return "which is not a number"
        return None
    if declared is str:
        return None if isinstance(value, str) else "which is not text"
    return None

def _values_in_bounds(task, fields) -> List[Issue]:
    """Every override's value is inside the bound the schema declares.

    Numeric ``range`` and enum ``choices`` are both "bounds" in § 6.6's sense:
    each is the schema saying *these are the values this field may take*, and
    a description carrying anything else renders a deck the engine will reject
    or, worse, quietly reinterpret.

    A field with neither is unbounded on purpose and is not checked — the
    schema is the authority on what a bound is, and inventing one here would
    refuse a description for breaking a rule nobody wrote.
    """
    out: List[Issue] = []
    for st in (task.stages or ()):
        for key, value in st.overrides.items():
            f = fields.get(key)
            if f is None:
                continue                    # already reported by _names_exist
            rng = f.metadata.get("range")
            choices = f.metadata.get("choices")
            if choices and value not in choices:
                out.append(Issue(
                    "error",
                    f"stage {st.name!r} sets {key} = {value!r}, which is not "
                    f"one of {', '.join(repr(c) for c in choices)}",
                    where=f"config.{key}", stage=st.name))
            elif rng and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                lo, hi = rng
                if not (lo <= value <= hi):
                    out.append(Issue(
                        "error",
                        f"stage {st.name!r} sets {key} = {value!r}, outside "
                        f"the allowed range [{lo}, {hi}]",
                        where=f"config.{key}", stage=st.name))
    return out


def refuse_on_error(issues: List[Issue]) -> List[Issue]:
    """Raise if any finding is an ``error``; return the rest.

    § 6.6's *"all of it before anything is written"* is the caller's to honour
    — this is the one line that makes honouring it hard to forget.
    """
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValidationError(errors)
    return issues


__all__ = ["preflight", "refuse_on_error"]
