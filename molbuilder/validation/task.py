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
codec, and four need the engine's field schema and its catalogue -- one of
those four being the DECLARED-TYPE row, which asks the setting door
(`engines/template.md` § 5.4) for the catalogue's type, as the bounds row asks
it for the limits. ``task.py`` is L1 — it imports
``persist`` and the standard library — and importing an engine into it is
the upward import the layer rule forbids (`architecture.md` § 3, kept by
review). Its docstring carries the same
split, so the two halves cannot quietly diverge; **if you add a row here, add it
there.**

**Why this returns findings rather than raising.** § 6.6's refusal rows
become ``error`` Issues, and the sequence checks (§ 6.4 / § 6.6a) are
``warn`` — a function that raises could not express *"proceed, and say
so"* at all.  :func:`refuse_on_error` is the caller's one line when it
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
              template_text: Optional[str] = None,
              request=None) -> List[Issue]:
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
    because half of § 6.6 is answerable from the description alone.

    ``request``, when the caller holds the job request beside it
    (`execution/job-request.md` § 7), adds its checks: the run cards' and
    the benchmark's names and values (:func:`request_checks`), and -- with
    the template -- that no stage shares a group with one it builds on
    (`group.check_groups`).

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
    if request is not None:
        out.extend(request_checks(request))

    # -- 3. every named field exists in the schema ------------------------
    fields = {f.name: f for f in dataclasses.fields(cls)}
    out.extend(_names_exist(task, cls, fields))
    # -- 4. ...and every override's value may stand on its rung: the
    #  setting door (`engines/template.md` § 5.4), asked in the override's
    #  home -- a run setting, a shared item, one the rung fixes or does not
    #  read; `null`; the catalogue's type (`relax_steps: 100.7` would reach
    #  the deck as `MD.Steps 100.7`, the M2 seam walk, 2026-08-07); a
    #  component the kind fixes, a choice it does not offer, a limit.  One
    #  value, one refusal; the door `resolve` asks too.
    refused = _overrides_may_stand(task, fields)
    out.extend(refused)

    # The RESOLVED ladder (template ⊕ each stage's overrides) when the caller
    # has the template: the checks below that judge a rung as it will run
    # read it, and one of them refuses -- so it is asked before the range
    # check, whose warning stands aside for a refused value (§ 5.3).
    from .stages import (check_a_relaxation_takes_a_step,
                         check_identical_stages,
                         check_ladder_does_not_loosen)
    _engine, _kind = str(task.engine), _kind_of(task)
    ladder: list = []
    if template_text is not None and task.stages:
        from ..resolve import resolved_ladder
        try:
            ladder = resolved_ladder(template_text, task, cls)
        except Exception:
            # An unresolvable template earns prep's OWN refusal moments
            # later, with its own wording (a template it cannot rebuild
            # from) -- a sequence WARNING must not preempt it as a crash.
            ladder = []
    # A rung that relaxes takes a step (plan § 5w K4): judged on the rung as
    # the description resolves it.
    stepless = check_a_relaxation_takes_a_step(ladder, engine=_engine,
                                               kind=_kind)
    # NO STAGE SHARES A GROUP WITH ONE IT BUILDS ON (`job-request.md`
    # § 3.3): the request's numbers against the ladder's links, asked with
    # the template -- an independent ladder's are its resolved stages'
    # `restart`, so they are asked of a ladder that resolves; one that does
    # not is refused above, or by prep.
    if ladder and request is not None:
        from ..jobset.group import check_groups
        why = check_groups(task, request, template_text)
        if why:
            out.append(Issue("error", why, where="job_request.stages.group"))

    # -- 4b. a value that may stand outside its recommended range is warned
    out.extend(_values_in_range(
        task, cls, fields,
        refused={(i.stage, i.where.split(".", 1)[1])
                 for i in (*refused, *stepless)}))
    out.extend(stepless)
    out.extend(_bias_points_in_range(task))
    out.extend(_window_reaches_the_bias(task, ladder))

    # -- 5. the sequence's own findings (§ 6.4 / § 6.6a) -- warnings -------
    out.extend(check_ladder_does_not_loosen(ladder, engine=_engine,
                                            kind=_kind))
    out.extend(check_identical_stages(ladder))
    return out



def request_checks(request) -> List[Issue]:
    """THE JOB REQUEST'S OWN CHECKS that need the vocabulary
    (`execution/job-request.md` § 3.2, § 3.4, § 7): every value on a run
    card -- the calculation's or a stage's -- and every benchmark point one
    its item takes in that home, asked of the setting door
    (`engines/template.md` § 5.4) -- ``request`` read against the
    description, whose engine and kind it holds."""
    return [*_run_cards_may_stand(request),
            *bench_points_fit_their_items(request)]


def _items_of(request) -> Dict[str, Any]:
    """The catalogue's items for ``request``'s engine, by name."""
    from ..template import catalogue, select
    return {it.name: it
            for it in select(catalogue(), engine=str(request.engine))}


def _run_settings_said(request, home: str) -> str:
    """The settings ``home`` may hold on ``request``'s calculation, by the
    door's homes (`template.home_of`), as a refusal lists what may be
    stated instead."""
    from ..template import home_of
    kind = str(request.calculation)
    return ", ".join(sorted(n for n, it in _items_of(request).items()
                            if home in home_of(it, kind))) or "(none)"


def _run_cards_may_stand(request) -> List[Issue]:
    """Every name on a run card -- the calculation's, a stage's -- is a run
    setting, and its value one its item takes there (`job-request.md`
    § 3.2): asked of the setting door in the run card's home, a stage's
    card on that stage's rung -- a GPU count on a transport ladder's
    transmission, which runs on no GPU, is refused by name where the
    description is written, never met as G5's refusal naming a switch that
    rung does not have (`web/task-setup.md` § 6.2b).  The run's own
    scheduler asks, `time` and `domain`, are the job request's reader's
    (§ 3.2a)."""
    from ..job_request import LANE_ASKS
    from ..template import accepts, shown
    blocks = [("execution", dict(request.execution or {}), None)]
    for name, entry in request.stages.items():
        if entry.execution:
            blocks.append((f"stage {name!r} execution",
                           dict(entry.execution), name))
    if not any(b for _w, b, _s in blocks):
        return []
    items = _items_of(request)
    engine, kind = str(request.engine), _kind_of(request)
    out: List[Issue] = []
    for where, block, stage in blocks:
        for name in sorted(block):
            if name in LANE_ASKS:
                continue
            it = items.get(name)
            if it is None:
                out.append(Issue(
                    "error",
                    f"{where} names {name!r}, which is not an execution "
                    f"setting of {engine}.  This block says what the RUN "
                    f"uses -- "
                    f"ranks, threads, the device, the solver.  Execution "
                    f"settings here: {_run_settings_said(request, 'run_card')}, "
                    f"{', '.join(LANE_ASKS)}",
                    where=f"job_request.{where}.{name}"))
                continue
            clause = accepts(it, block[name], "run_card", engine=engine,
                             kind=kind, stage=stage)
            if clause:
                out.append(Issue(
                    "error",
                    f"{where} sets {name} = {shown(block[name])}{clause}.",
                    where=f"job_request.{where}.{name}"))
    return out


def bench_points_fit_their_items(request) -> List[Issue]:
    """Every benchmark row names a run setting -- a speed knob, so each
    point measures one calculation (`job-request.md` § 3.4: sweep anything
    else and the comparison says nothing, silently) -- and every point is
    one its item takes in a benchmark row, asked of the setting door
    (`engines/template.md` § 5.4); a repeated point would measure one
    configuration twice.  THE ONE HOME of those rules: prep's bench
    assembly (`jobset/prep_inputs.py::_declared_execution_pins`) asks this
    same function as its backstop."""
    plan = request.bench
    if not plan:
        return []
    from ..template import accepts, shown, why_not_in
    items = _items_of(request)
    engine, kind = str(request.engine), _kind_of(request)
    out: List[Issue] = []
    for name in sorted(plan):
        where = f"job_request.bench.{name}"
        it = items.get(name)
        if it is None:
            out.append(Issue(
                "error",
                f"bench names {name!r}, which is not a setting of {engine} "
                f"-- a benchmark row sweeps a run setting.  Sweepable here: "
                f"{_run_settings_said(request, 'bench')}", where=where))
            continue
        home = why_not_in(it, "bench", engine=engine, kind=kind)
        if home:
            out.append(Issue("error", f"bench names {name!r}{home}.",
                             where=where))
            continue
        pts = list(plan[name])
        refused = [Issue("error", f"bench declares {name} = {shown(v)}"
                                  f"{clause}.", where=where)
                   for v in pts
                   for clause in [accepts(it, v, "bench", engine=engine,
                                          kind=kind)] if clause]
        out.extend(refused)
        # ONE VALUE, ONE REFUSAL: a repeat is judged among points that stand.
        if not refused and len(pts) != len(set(pts)):
            out.append(Issue(
                "error",
                f"bench declares {name} = {shown(pts)}; a repeated point "
                f"would measure one configuration twice.", where=where))
    return out


def _shipped() -> Dict[str, Any]:
    """The engines this backend has a generator for, resolved lazily.

    Lazily because importing an engine config at module scope would drag the
    whole engine stack into anything that imports ``validation``.
    """
    from ..config.pyscf import PySCFConfig
    from ..config.siesta import SiestaConfig
    return {"siesta": SiestaConfig, "pyscf": PySCFConfig}


def config_class_for(task, generators=None):
    """The engine-config dataclass *task*'s engine is generated from, or None.

    ONE OWNER for the engine -> config resolution.  `web/blueprints/build.py`
    renders this task's preflight findings for the browser, and
    `issues_to_json` needs the dataclass to resolve each finding's
    ``workflow_group`` -- without it the key is OMITTED and the page cannot
    attach the finding to its workflow-group card (`web/ui-contract.md`
    Rule 2).

    Returns the CLASS, not an instance: `resolve_workflow_group` reads
    ``metadata["workflow_group"]`` off the field declarations, so no values are
    needed and constructing one would invent defaults that are not this task's.

    ``None`` for an engine this backend has no generator for -- which
    :func:`preflight` reports as its first error, and which leaves the caller
    exactly where it was before, rather than raising on the refusal path.
    """
    known = dict(generators if generators is not None else _shipped())
    return known.get(task.engine)


def _names_exist(task, cls, fields) -> List[Issue]:
    """Every name in ``varies`` and every ``overrides`` key is a real field
    — a real TEMPLATE field: a machine fact refuses with § 7's story.

    ``varies`` is checked too, not only the overrides. ``task.py`` enforces
    ``overrides ⊆ varies`` without a schema, so a *misspelt* name that appears
    in both passes the codec cleanly — the subset holds, and both are wrong the
    same way. This is the only place that can catch it.

    A ``varies`` name that IS a field is a column a stage may override, so
    the setting door is asked whether a stage's overrides may hold it at all
    (`engines/template.md` § 5.4, ``template.why_not_in``) -- a run setting
    is the run card's, a shared item the template's, one the rung fixes
    nobody's.  An override's own value is the door's in full
    (:func:`_overrides_may_stand`).
    """
    from ..template import items_of, why_not_in
    items = items_of(cls)
    engine, kind = str(task.engine), _kind_of(task)

    out: List[Issue] = []
    for name in (task.varies or ()):
        if name not in fields:
            out.append(Issue("error", _no_such_field(
                name, fields, cls, "'varies' names"), where="task.varies"))
            continue
        home = why_not_in(items[name], "override", engine=engine, kind=kind)
        if home:
            out.append(Issue("error", f"'varies' names {name!r}{home}.",
                             where="task.varies"))
    for st in (task.stages or ()):
        for key in st.overrides:
            if key not in fields:
                out.append(Issue("error", _no_such_field(
                    key, fields, cls, f"stage {st.name!r} overrides",
                    value=st.overrides[key], is_set=True),
                    where="task.stages.overrides", stage=st.name))
    return out


def _no_such_field(name, fields, cls, lead, *, value=None,
                   is_set: bool = False) -> str:
    import difflib
    from ..template import why_not_an_item
    gone = why_not_an_item(name, value=value, is_set=is_set)
    if gone is not None:
        # A RETIRED OR RENAMED ITEM is named as such, with the way on -- not
        # sent looking for a misspelling (`engines/template.md` § 3).
        return f"{lead} {name!r}: {gone}"
    near = difflib.get_close_matches(name, list(fields), n=1, cutoff=0.75)
    hint = f" -- did you mean {near[0]!r}?" if near else ""
    return (f"{lead} {name!r}, which is not a field of "
            f"{cls.__name__}{hint}")


def _bias_points_in_range(task) -> List[Issue]:
    """The bias list's points are values of `bias_voltage_v`, and a value
    outside its item's recommended range is WARNED, as any value is
    (`engines/template.md` § 5.3).  A repeated point is the codec's refusal
    (`task.py`), with the list's other shape rules."""
    bias = tuple(getattr(task, "bias", ()) or ())
    if not bias:
        return []
    from ..template import catalogue, one
    item = one(catalogue(), "bias_voltage_v", engine=str(task.engine))
    if item is None or not item.range:
        return []
    lo, hi = item.range
    out = [float(v) for v in bias if not lo <= float(v) <= hi]
    return [Issue(
        "warn",
        f"the bias list holds {', '.join(f'{v:g}' for v in out)} V, outside "
        f"the recommended range [{lo:g}, {hi:g}] V -- a recommendation, not "
        f"a limit (engines/template.md 5.3)",
        where="task.bias")] if out else []


def _window_reaches_the_bias(task, ladder) -> List[Issue]:
    """The transmission window reaches the largest listed voltage's bias
    window, ±(|V|/2 + 5 kT) -- the one rule `record.window_short_of` states
    (`engines/transport.md` § 2a.10, P2), judged here on the description as
    the transmission rung resolves it, so a list that the window cannot
    serve is refused where it is written: under the low-bias approximation
    the record integrates every listed voltage over the one 0 V slice, and
    a sweep's point writes its own voltage into that window.  Nothing to
    say without the resolved ladder (no template in hand)."""
    bias = tuple(getattr(task, "bias", ()) or ())
    if not bias or not ladder:
        return []
    cfg = next((c for name, c in ladder if name == "transmission"), None)
    if cfg is None:
        return []
    emin = getattr(cfg, "transmission_emin_ev", None)
    emax = getattr(cfg, "transmission_emax_ev", None)
    n = getattr(cfg, "transmission_n_points", None)
    temp = getattr(cfg, "electronic_temperature", None)
    if emin is None or emax is None or not n or temp is None:
        return []
    from ..constants import BOLTZMANN_EV_K
    from ..transport.record import sampled_window, window_short_of
    largest = max(bias, key=lambda v: abs(float(v)))
    short = window_short_of(*sampled_window(emin, emax, n), largest,
                            BOLTZMANN_EV_K * float(temp))
    return [Issue("error", short + " (engines/transport.md 2a.10)",
                  where="config.transmission_emax_ev",
                  stage="transmission")] if short else []


def _kind_of(task) -> str:
    return str(task.calculation)


def _overrides_may_stand(task, fields) -> List[Issue]:
    """Every override's value may stand on its rung -- the setting door,
    asked in a stage's overrides (`engines/template.md` § 5.4,
    ``template.accepts``): whether that home may hold the item at all (a run
    setting, a shared item, one the rung fixes, one the rung does not read
    -- plan § 5w K4), ``null``, the catalogue's declared type, a component
    the kind fixes, a choice it does not offer, a limit.  A description
    carrying anything else renders a deck the engine rejects or, worse,
    quietly reinterprets -- refused where the description is written, never
    first at prep on the machine that runs it.  One value, one refusal; a
    name no field carries is :func:`_names_exist`'s."""
    from ..template import accepts, catalogue, select, shown
    engine, kind = str(task.engine), _kind_of(task)
    items = {it.name: it for it in select(catalogue(), engine=engine)}
    out: List[Issue] = []
    for st in (task.stages or ()):
        for key, value in st.overrides.items():
            if key not in fields:
                continue                    # already reported by _names_exist
            clause = accepts(items[key], value, "override", engine=engine,
                             kind=kind, stage=st.name)
            if clause:
                out.append(Issue("error",
                                 f"stage {st.name!r} sets {key} = "
                                 f"{shown(value)}{clause}.",
                                 where=f"config.{key}", stage=st.name))
    return out


def _values_in_range(task, cls, fields, *,
                     refused=frozenset()) -> List[Issue]:
    """An override outside its item's recommended ``range`` is WARNED, as
    on every surface: a range is a recommendation, never a limit
    (`engines/template.md` § 5.3).  A field with none is unbounded on
    purpose.  ``refused`` holds the ``(stage, item)`` overrides the door
    refused (:func:`_overrides_may_stand`) or that a stepless relaxation
    drew: each drew that refusal, and draws nothing more here -- one value,
    one refusal."""
    from .metadata import outside_range
    from ..template import items_of
    items = items_of(cls)
    out: List[Issue] = []
    for st in (task.stages or ()):
        for key, value in st.overrides.items():
            if key not in fields or (st.name, key) in refused:
                continue
            item = items.get(key)
            rng = item.range if item is not None else None
            if not rng:
                continue
            lo, hi = rng
            # PER COMPONENT FOR A TRIPLE, through the helper the settings
            # gate asks too (`metadata.outside_range`).
            for i, v in outside_range(value, rng):
                shown = key if i is None else f"{key}[{i}]"
                out.append(Issue(
                    "warn",
                    f"stage {st.name!r} sets {shown} = {v!r}, outside the "
                    f"recommended range [{lo}, {hi}] -- a recommendation, "
                    f"not a limit (engines/template.md 5.3)",
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


__all__ = ["bench_points_fit_their_items", "preflight", "refuse_on_error"]
