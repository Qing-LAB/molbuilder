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
codec, and four need the engine's field schema. ``task.py`` is L1 — it imports
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
    out.extend(_values_are_the_declared_type(task, fields))
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


def _values_are_the_declared_type(task, fields) -> List[Issue]:
    """An override's value must be one the field can actually hold.

    **Refused rather than coerced**, and the two halves of that decision are
    deliberate. ``effective_config`` widens ``int -> float`` because JSON has
    one number and ``150`` for a float field is the same value written
    differently — lossless, so silence is honest. Everything else is the
    caller's mistake: coercing ``relax_steps: 100.7`` to 100 would silently
    run a different calculation from the one described, and parsing ``"150"``
    would make a quoting slip invisible.

    Found by the M2 seam walk. Neither side could see it alone — ``task.py``
    has no schema to check a type against, and ``effective_config`` had no
    reason to think a value might not be one.
    """
    out: List[Issue] = []
    for st in (task.stages or ()):
        for key, value in st.overrides.items():
            f = fields.get(key)
            if f is None:
                continue                    # already reported by _names_exist
            bad = _type_complaint(f.type, value)
            if bad:
                out.append(Issue(
                    "error",
                    f"stage {st.name!r} sets {key} = {value!r}, {bad}",
                    where=f"config.{key}", stage=st.name))
    return out


def _type_complaint(declared, value) -> Optional[str]:
    """Why *value* cannot be this field's, or ``None`` if it can."""
    is_bool = isinstance(value, bool)
    if declared in ("int", int) and "Optional" not in str(declared):
        if is_bool or not isinstance(value, (int, float)):
            return "which is not a whole number"
        if isinstance(value, float) and not value.is_integer():
            return (f"which is not a whole number -- this field counts "
                    f"things, so {value!r} would have to be rounded and the "
                    f"run would not be the one described")
    elif declared in ("float", float) and "Optional" not in str(declared):
        if is_bool or not isinstance(value, (int, float)):
            return "which is not a number"
    elif declared in ("bool", bool) and not is_bool:
        return "which is not true or false"
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
