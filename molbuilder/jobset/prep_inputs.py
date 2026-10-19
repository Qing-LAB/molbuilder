"""What a prep receives, assembled ONCE -- `execution/architecture.md` A12.

Two tuples, one per kind:

    prep_run_inputs  ->  (allocation, pins, chosen)       a run
    bench_inputs     ->  (points, pins, translation)      a benchmark sweep

and the pieces they are made of: the run's condition (`Task.run_condition`,
read by `declared_run_shape`, `run_inputs`), the bench's
declared pins and axes (`_declared_execution_pins`), and the bench grid --
enumerated, fit cell by cell on the benchmark's queue, and reported
(`_cells_on_the_queue`, `_local_over`, `_past_the_ask`, `_cell_*`).

**The conductor's own assembly, beside it** (`architecture.md` § 2.1's note:
only a surface imports the conductor, and its own modules).  A refusal is a
`PrepError`, which each surface shows as it is, and what a person is TOLD
comes back through the ``notes`` list a caller hands in -- the command line
prints it, the tab shows it -- rather than being printed here.
"""
from __future__ import annotations

from pathlib import Path

from .errors import PrepError


def _declared_execution_pins(task, bench_override=None):
    """The description's ``bench``, read as the OVERRIDE LANE it is (user rule,
    2026-08-20; `generator.md` § 4.3a): every non-machine entry overrides
    the template -- several points = an axis to try, ONE point = the value
    the bench's trials run with, applied at prep as a pin for them alone
    (the run's values are its own card's, `stages.md` § 6.8d).  Nothing migrates between files: the description
    stays exactly as edited, and prep is where a declaration is resolved.
    The run's condition is handed through here too (:func:`run_inputs`), so
    the two lanes split a name by one rule.

    Returns ``(pins, axes, value_axes)``: the one-point non-machine values
    as a pins dict, the machine-answered entries untouched (the grid's
    axes), and the MULTI-point non-machine entries as value axes --
    ``{name: [points...]}`` -- which `bench_inputs` crosses with the
    machine grid so every trial's deck carries its coordinates
    (§ 4.3a's built rule).

    Refused BY NAME, never repaired -- through the ONE shape checker
    (`validation/task.py::bench_points_fit_their_items`): an enum value
    outside the item's choices, a non-bool on a bool item, a repeated
    point.  The dispatch runs the
    full preflight BEFORE this helper, so on either door of the one prep
    entry the refusal normally lands there with every finding listed; the
    call here is the backstop for direct callers, first-refusal shaped as
    the entry's errors are.

    MEMBERSHIP is not this function's question (one door): every bench key
    must be an ``execution`` item, and `validation/task.py`'s
    ``_bench_names_a_speed_knob`` owns that rule -- a non-execution name
    was refused upstream and an unknown name here is simply skipped.
    Point SHAPES are `read_task`'s (task.py `bench_from_obj`): every
    value arrives as a non-empty tuple of scalars, so no scalar/empty
    arms exist here.

    ``task`` is the description its caller holds -- read once by the prep
    that asks (W55 B1).
    """
    from ..validation.task import bench_points_fit_their_items
    from .. import template as _T

    if bench_override is not None:
        # THE AXES AS THEY ARE BEING EDITED, not as they were last saved.
        # The task-setup card resolves its grid live, and its edits live in
        # the browser's model until the person saves -- so a live answer
        # read from disk would describe the previous state.  Through
        # `bench_from_obj`, the ONE normaliser, so the override arrives
        # in the same shape (and earns the same refusals) as a declaration
        # that came off disk.
        import dataclasses as _dc

        from ..task import bench_from_obj
        task = _dc.replace(task,
                           bench=bench_from_obj({"bench": bench_override}))
    declared = {k: list(v) for k, v in (task.bench or {}).items()}
    if not declared:
        return {}, {}, {}

    shape_errors = [i for i in bench_points_fit_their_items(task)
                    if i.severity == "error"]
    if shape_errors:
        raise PrepError(shape_errors[0].message)

    # THE ENGINE IS THE DESCRIPTION'S OWN (task.engine), so this door and
    # the shape checker it calls are keyed on one source.
    items = {i.name: i
             for i in _T.select(_T.catalogue(), engine=str(task.engine))
             if "execution" in (i.category or ())}
    pins, axes, value_axes = {}, {}, {}
    for name, pts in declared.items():
        it = items.get(name)
        if it is None:
            continue     # membership is the preflight's refusal, upstream
        if it.allocation:
            axes[name] = pts                 # machine-answered: the grid's
            continue                         # business, never a value
        if len(pts) > 1:
            value_axes[name] = pts           # a VALUE AXIS (§ 4.3a): the
            continue                         # grid multiplies per point
        pins[name] = pts[0]
    return pins, axes, value_axes


#: What makes a trial a MEASUREMENT rather than a run -- schema values
#: resolved like any other (`template.md` § 8.1: rebuild and render, never
#: splice): capped SCF, single point, forced cold, run-once, and
#: ``scf_must_converge: False`` so the cap ends CLEAN.  Applied OVER every
#: declared value (one-point pins and value-axis coordinates alike), and a
#: value AXIS naming one of these is refused -- its trials would be one
#: measurement under many labels.  One spelling for both uses (the pins
#: and that refusal); the per-key whys sit at the application site.
#: ``max_scf_iter: 3`` (user, 2026-08-21, from measured experience:
#: iterations 3-5 agree within seconds on a 444-atom junction, and the
#: bench reads SCALING and DEPENDENCY -- the knee -- not tight rankings):
#: iteration 1 never forms a timing delta, the iter-2 delta is dropped as
#: warm-up-adjacent, iteration 3 is the one clean sample (tuning.md § 2.12).
_MEASUREMENT_PINS = {"max_scf_iter": 3, "relax_steps": 0, "restart": "clean",
                     "continue_retries": 0, "scf_must_converge": False}


def _gpus_per_node(routing) -> "int | None":
    """The most GPUs one node of the target's GPU queues holds, or ``None``
    when no queue lists any -- § 4.3a's fallback when the probed node itself
    has none (a login node).  ``routing`` is the target's own menu, from the
    record the caller holds.

    A COUNT, and no card: which card a node carries is the machine's
    business (`scheduler.md` R2a; user, 2026-10-01).
    """
    from ..scheduler import domain_serves_gpu
    counts = [d.per_node for r in routing if domain_serves_gpu(r)
              for d in r.devices if d.per_node is not None]
    return max(counts) if counts else None


def _cells_on_the_queue(plan, *, routing, local_cores=None,
                        local_gpus=None):
    """Every enumerated bench cell, fit on the benchmark's queue --
    ``[(fam, (g, k, c), fits, over)]`` in enumeration order: ``fits`` the
    cell's fit on each queue of ``routing``, as ``{queue, holds, line}``
    (`scheduler.fit`, the one `place` itself asks, so a cell and a run get
    one reading), ``over`` the limits it is over -- empty when a node of
    some queue there holds it (`generator.md` § 4.3a).

    ``routing`` is the benchmark's ONE queue (:func:`bench_inputs`): a
    benchmark measures on the queue its prep names (the user, 2026-10-17);
    the Task setup card, before one is named, asks every queue of the
    record, so the person sees each one's fit while choosing.  A machine
    with no queue menu at all is fit on ``local_cores`` / ``local_gpus`` --
    the probed topology of the machine it is for, which IS the machine when
    nothing schedules for it; its one entry is then what the cell is over
    there, ``queue`` ``None``.
    """
    from ..scheduler.fit import Request, fit
    if not routing:
        out = []
        for fam, cell in plan:
            over = _local_over(cell, fam, local_cores, local_gpus)
            said = ({"queue": None, "holds": not over,
                     "line": "; ".join(o.message for o in over)},)
            out.append((fam, cell, said if over else (), over))
        return out
    out = []
    for fam, (g, k, c) in plan:
        want_gpu = bool(fam and g)
        req = Request(ranks=_cell_ranks(want_gpu, g, k), cpus_per_task=c,
                      gpus=g if want_gpu else None)
        fits = [fit(row, req) for row in routing]
        held = any(f.holds for f in fits)
        out.append((fam, (g, k, c),
                    tuple({"queue": f.domain, "holds": f.holds,
                           "line": f.line} for f in fits),
                    () if held else tuple(o for f in fits for o in f.over)))
    return out


def _local_over(cell, fam, cores_total, gpus_per_node):
    """What a cell is over on the target's probed node -- the fit where
    there is no queue (`generator.md` § 4.3a).

    Empty means it fits, or that the probe measured nothing to compare
    against: an unmeasured topology is silence (R3).  Each names its
    numbers (R4).
    """
    from ..scheduler.fit import Over
    g, k, c = cell
    ranks = _cell_ranks(fam, g, k)
    over = []
    if cores_total and ranks * c > cores_total:
        over.append(Over("box_cores", "the target's probed node",
                         asked=ranks * c, allowed=cores_total, unit="cores"))
    if fam and g and gpus_per_node is not None and g > gpus_per_node:
        over.append(Over("box_gpus", "the target's probed node", asked=g,
                         allowed=gpus_per_node, unit="GPUs"))
    return tuple(over)


def _past_the_ask(fam, cell, ask) -> list:
    """Why a cell is past what this prep allows -- its ranks, its cores per
    rank, its GPUs against the ask's (`generator.md` § 4.1): two statements
    of the person's disagreeing, so the cell is left out whether proposed
    or declared; empty when it is within or nothing was asked."""
    if ask is None:
        return []
    from ..scheduler.fit import Over
    from ..scheduler.quantities import parse_gres_flag
    g, k, c = cell
    gpus = g if (fam and g) else 0
    out = []
    for limit, asked, allowed, unit in (
            ("ask_ranks", _cell_ranks(fam, g, k), ask.mpi_np, "ranks"),
            ("ask_cpus", c, ask.cpus_per_task, "cores per rank"),
            ("ask_gpus", gpus,
             parse_gres_flag(ask.gres) if ask.gres not in (None, "")
             else None, "GPUs")):
        if allowed is not None and asked > int(allowed):
            out.append(Over(limit, "this prep", asked=asked,
                            allowed=int(allowed), unit=unit))
    return out


def _cell_ranks(fam, g, k) -> int:
    """How many ranks a cell runs -- ``G x K`` on the GPU family, ``K`` on
    the CPU one.

    ONE SPELLING, for the fit check, the shape line and the report.
    """
    return g * k if (fam and g) else k


def _cell_label(g, k, c) -> str:
    """A cell's name: the machine part of its trials' names, built by the
    one builder (`resolve.point_token` -- `job-contracts.md` § 6.3: "and by
    nothing else"), ``G0`` for a CPU cell on every machine.  A sweep with
    value axes crosses each cell with them, so its trials' names go on past
    this one (``G0K48C1ELPA1stage``): the fit list names the cell, the
    directories its trials."""
    from ..resolve import point_token
    return point_token({"G": g, "K": k, "C": c})


def _cell_shape(g, k, c) -> str:
    """What a cell ASKS FOR, in words -- ranks, cores each, and GPUs."""
    bit = f"{_cell_ranks(bool(g), g, k)} rank(s) x {c} core(s)"
    return bit + (f" + {g} GPU(s)" if g else "")


# THE CATALOGUE'S WORDS FOR A LAUNCH FIELD are mapped to `Resources`' own by
# the one name map, `model.AS_RESOURCE`; an item whose name maps to no field
# is DROPPED by `declared_run_shape`.  The run's own wall and queue
# (`stages.md` § 6.8e) are `Resources` fields under their own names.
from ..task import LANE_ASKS as _LANE_ASKS

from .model import AS_RESOURCE


def declared_run_shape(base, task, stage=None):
    """The run's LAUNCH SHAPE — the condition's machine items, as `Resources`
    fields.  ``{}`` when the condition states none.

    **A DIRECT MAP, because there is nothing to work out.** The template
    carries the physics and the deck knobs (`template.md` § 5); a launch
    value this block does not name is stated as a flag, or prep refuses the
    run (`architecture.md` § 5.2) -- nothing fills one in.  Not the grid
    enumerator: `bench_inputs` is a sweep's own machinery and must invent the
    axes a condition does not mention.

    The device ask is a COUNT -- ``gpu:N`` -- and names no card: which card
    a node carries is the machine's business (`scheduler.md` R2a).  The count is the run card's
    own ``gpu_count``, on every engine, carried as stated: whether the run
    uses the GPU is its resolved ``use_gpu``, and the two are asked together
    by the prep entry, of the job it resolves (`prep.plan_stage`,
    `execution/gpu.md` G5).
    """
    cond = task.run_condition(stage)
    from ..template import catalogue, select
    from .model import Resources
    items = {i.name: i for i in select(catalogue(),
                                       engine=getattr(task, "engine", ""))}
    known = {f.name for f in __import__("dataclasses").fields(Resources)}
    out = {}
    for name, val in sorted(cond.items()):
        if name in _LANE_ASKS:
            # THE RUN'S OWN SCHEDULER ASK (`stages.md` § 6.8e).  Not a
            # catalogue item, so `items` does not have it -- and falling
            # through the skip below would DROP a value a person typed,
            # which is the defect class this lane keeps producing.
            out[name] = val
            continue
        it = items.get(name)
        if it is None:
            continue                    # membership is validation's refusal
        if not it.allocation:
            continue                    # a parameter -- pins, never the launch
        field = AS_RESOURCE.get(name, name)
        if field == "gres":
            out["gres"] = int(val)      # spelled below, as the count it is
        elif field in known:
            out[field] = val
    if "gres" in out:
        out["gres"] = f"gpu:{out['gres']}"
    return out


def run_inputs(base, task, stage=None):
    """The run's two halves from its condition — ``(chosen, pins)``.

    One block, two destinations, split by the catalogue's own answer: a
    machine-answered item is the launch SHAPE (`declared_run_shape`), anything
    else is a PIN over the template — through `_declared_execution_pins`, the
    same door the bench's one-point declarations go through, so the two lanes
    cannot disagree about what a name means.
    """
    cond = task.run_condition(stage)
    pins = {}
    if cond:
        pins, _axes, _value_axes = _declared_execution_pins(
            task, {k: [v] for k, v in cond.items()})
    # THE SHAPE EVEN WHEN THE CARD IS EMPTY: a template whose `use_gpu` is on
    # is a device run with nothing on its card, and its count is asked for
    # like any other's (`gpu.md` G5 -- refused, unstated, by the prep entry
    # of the job it resolves).
    return declared_run_shape(base, task, stage), dict(pins or {})


def prep_run_inputs(base, task, stage, allocation=None):
    """Everything ``prep task`` needs, assembled ONCE -- ``(allocation, pins,
    chosen)``.

    **`architecture.md` A12**, and the contract stated there: the ask handed
    in is what the person is saying RIGHT NOW -- flags, or an EMPTY
    `Resources()` from a surface that has none, never ``None``.

    **THE UI IS NOT A SECOND FRAMEWORK** *(user, 2026-09-02: "I go through
    the same back end to generate the execution script or the CLI command.
    This is important because if you handcraft two branches to collect the
    parameter and generate the thing, then you have to maintain two branches
    of the logic.  The UI is not a different thing.  It goes through the same
    framework.  It just helps the user to visualize and to decide.")*

    So the browser's prep button and ``molbuilder jobset prep task`` call
    THIS, and neither assembles anything of its own.  ONE source states the
    run (`stages.md` § 6.8d, plan § 5w K5): ``execution``, the calculation's
    block with the rung's over it -- what the person ASKED for.  Its machine
    items are the launch shape, the rest are pins over the template.  The
    bench's one-point non-machine declarations pin the trials alone.

    A flag beats it and is already in ``allocation`` when it arrives.
    """
    from .model import Resources

    # NEVER NONE (§ 6.0a's contract): the run card and the description are
    # folded UNDER this field by field (step 2), and there is no
    # field-by-field merge onto nothing.  A surface with no flags passes an
    # empty ask, not an absent one.
    allocation = allocation if allocation is not None else Resources()

    # 1 · THE CONDITION -- the run card, the launch-shape ladder's first rung,
    #     under a flag (`architecture.md` § 5.2) -- folded where the rest of
    #     the allocation is, once, at step 2 (:func:`fold_allocation`: a
    #     flag, then the run card, then the description).
    chosen, cond_pins = run_inputs(base, task, stage)
    # WHETHER A GPU RUN STATES HOW MANY, AND A CPU RUN NONE (`execution/
    # gpu.md` G5) is the prep entry's to ask, of the job it resolves from
    # this assembly, beside the launch values' refusal -- and the Task
    # setup card shows the entry's preview of it.

    # 1b · THE CALCULATION'S SCHEDULER ASK is folded where the stage is
    #      resolved, once (`prep._resolve_stage`, :func:`fold_allocation`:
    #      a flag, then the run card, then the description).

    # 2 · THERE IS NO SECOND RUNG.  A benchmark's verdict is a REPORT a
    #     person reads (printed by `summarize`), and what the
    #     run uses is what that person then wrote in `execution`
    #     (`architecture.md` § 5.2, user ruling: "the run parameter needs to
    #     be explicitly decided/written").  A measurement that reaches the
    #     launch on its own is a second arrival route, and every silent-value
    #     defect this lane has had was a second arrival route.

    # 3 · THE PINS -- the condition's alone (5.2's deck/speed ladder).  The
    #     bench's one-point declarations pin its trials, never the run
    #     (`stages.md` § 6.8d's "and nowhere else").
    pins = dict(cond_pins) or None

    # 4 · NOTHING IS WORKED OUT FOR WHAT NOBODY STATED.  An unstated rank or
    #     thread count is refused by `launch_refusal`, which the entry asks
    #     with the whole assembly in hand (`architecture.md` § 5.2; user,
    #     2026-10-02: "explicit job config is the only way allowed").

    # `chosen` is returned for the PREVIEW to name; it is already folded in.
    return allocation, pins, chosen


def bench_refusal(task):
    """Why this description has no benchmark sweep -- ``None`` when it has
    one.  One answer for every door that asks: the prep entry's gate, the
    bench's own assembly (:func:`bench_inputs`, asked first, before any
    machine is read), and the Task setup tab, which offers the Measure step
    only where `prep bench` would take it (`job-system.md` § 5.3)."""
    if task.calculation == "transport":
        return ("a transport calculation has no benchmark sweep: its "
                "parameters come from its own template and each rung's run "
                "card, and its one axis is the bias list in task.json "
                "(engines/transport.md 2a.10).")
    # THE SEAM REFUSAL, BY NAME (E-J1).  The bench lane speaks SIESTA's
    # vocabulary today: the measurement pins name SiestaConfig fields
    # (`max_scf_iter`, `restart`, ...), and the GPU question is read under
    # SIESTA's `use_gpu`, so another engine's sweep could silently
    # enumerate a CPU grid (`engines/stages.md` § 6.8's recorded hazard).
    if str(task.engine) != "siesta":
        return (f"this description's engine is {task.engine!r}, and the "
                f"benchmark lane only speaks SIESTA today: its measurement "
                f"pins (a capped-SCF probe run) name SIESTA settings, so a "
                f"{task.engine} bench would measure nothing meaningful.  "
                f"Benchmark support for other engines is a recorded design "
                f"(engines/stages.md § 6.8); for now, size the run from "
                f"the engine's own scaling guidance in docs/engines/tuning.md.")
    return None


def fold_allocation(flags, declared, chosen=None):
    """``(Resources, where each value came from)`` -- the caller's
    allocation over the description's, with each field it holds named by
    its source:
    ``flag``, ``run card`` or ``description`` -- what a run's placement
    records beside the queue it was placed on (`job-system.md` § 6.0).
    ONE fold: the sources are read off the same precedence that fills the
    values, never worked out a second way.

    FIELD BY FIELD: a flag is what the person asks for now, so a stated
    one wins and an unstated one leaves the file's answer standing --
    whole-object precedence would make ``--np 8`` erase a memory ask
    nobody mentioned.  ``declared`` is the description's ``allocation``
    (`stages.md` § 6.8a); ``chosen``, the run card's launch shape
    (§ 6.8d, :func:`declared_run_shape`)."""
    from .model import Resources
    out = flags or Resources()
    import dataclasses as _dc
    known = {f.name for f in _dc.fields(Resources)}
    sources = {f: "flag" for f in known
               if getattr(out, f, None) not in (None, "")}
    patch = {}
    said = {}
    if declared:
        for name, val in (("domain", declared.domain),
                          ("time", declared.time),
                          ("mem", declared.mem)):
            if val and getattr(out, name, None) in (None, ""):
                patch[name] = val
                said[name] = "description"
        # THE BINDING SWITCH, when said: `False` is the value that matters,
        # so it is not tested for truth (`execution/gpu.md` G9).
        if declared.gpu_binding is not None and out.gpu_binding is None:
            patch["gpu_binding"] = declared.gpu_binding
            said["gpu_binding"] = "description"
    # ALREADY IN `Resources`' OWN WORDS -- `to_resources` speaks them, so
    # there is no name map here and no second place for one to drift.
    for name, val in sorted((chosen or {}).items()):
        if name in known and getattr(out, name, None) in (None, ""):
            patch[name] = val
            said[name] = "run card"
    sources.update(said)
    return (_dc.replace(out, **patch) if patch else out), sources


#: The queue a caller names no queue of its own for: the fold's
#: (:func:`fold_allocation`) -- prep's, its flags over the description.
_FOLDED = object()


def bench_inputs(base, target, *, bench_override=None, report=None,
                 notes=None, task=None, environment=None,
                 template_text=None, flags=None, on_a_queue: bool = True,
                 queue=_FOLDED):
    """The benchmark specialisation's three inputs — `project-layout.md`

    ``target`` is REQUIRED: this function exists to read a SPECIFIC
    machine's hardware.  Pass ``None`` deliberately to mean "this machine".

    § 2.3.1a's split, stated as data: WHERE the values come from (the grid,
    enumerated from the target's probed record, as explicit points), the
    point → Resources translation, and the trial pins.  The framework —
    `prep`'s five steps — receives a longer list and never asks why
    (`generator.md` § 2).

    **The grid is RESOLVED here, at prep — and the description may DECLARE
    it** (`generator.md` § 4.3a, user-settled 2026-08-17).
    ``task.json``'s ``bench`` names the points to try — *"try 4, 8 and 16
    ranks"* is true on every cluster, so it is portable and belongs with the
    calculation; what those points MEAN on this machine is resolved here.
    With no declaration the machine proposes: the grid is enumerated from
    the probed topology.

    **Whether this is a GPU grid is the DESCRIPTION's answer, not this
    function's assumption** (2026-08-17).  `web/task-setup.md` § 6.2 —
    *"use GPU or not is set up only at the Job Prep UI"* — makes ``use_gpu``
    a value the person chose, carried in the template like any other; and
    § 6.2 is equally explicit that the eigensolver is NOT the same question
    (``diag_algorithm`` is a `budget` item on the parameter tab).  The
    description answers, and the grid follows its answer.

    **A multi-point non-machine entry is a VALUE AXIS** (§ 4.3a): its points multiply the machine grid, each point carries
    its coordinates (the resolver's parameter lane applies them per
    trial), and ``use_gpu`` with two points is the grid-FAMILY axis --
    the grid enumerates once per flag, G=0 holding the CPU family's
    device coordinate.  See the section for the cap, naming, and
    split-submission halves of the rule.

    ``task``, ``environment`` and ``template_text`` are the description,
    the target's record and the template's text, when the caller has read
    them: the prep entry reads each once and hands them over (W55 B1).  The
    bench card's route has not, and they are read here.

    ``flags`` are the prep's own, folded over the description's
    ``allocation`` by the one fold (:func:`fold_allocation`) into the ASK:
    the benchmark's one queue, and what the person allows this prep -- its
    ranks, cores per rank and GPUs (`generator.md` § 4.1, the allocation
    the sweep must fit inside).  A cell past it is left out by name; a
    PROPOSED cell no node of the queue holds is left out with its fit, and
    one DECLARED whole -- its ranks, cores per rank and, on the GPU, its
    device count -- is kept with its fit said under it (`scheduler.md` R6); the
    prep is refused only when no cell is left (§ 4.3a: a cell that fits is
    never denied by a sibling that does not).  The bench card names no
    flags; it passes the queue picked on the page as ``queue`` (``None`` for
    none picked), in place of the description's, as it passes the axes
    being edited (``bench_override``).

    ``on_a_queue`` is whether the benchmark goes to a queue -- false where
    prep writes it no header (`--no-sbatch`, or a machine with no
    scheduler): it then runs where it is launched, and its cells are the
    target's own topology's.
    """
    # WHAT A PERSON IS TOLD comes back to the caller, never printed here:
    # the command line prints it, the Task setup tab shows it
    # (`prep.prep_task`, `job-system.md` § 5.3).
    note = notes.append if notes is not None else (lambda _text: None)
    from ..bench.grid import sweep_K, sweep_grid
    from ..resolve import MachineTranslation
    from ..task import FILENAME as TASK_FILENAME, read_task
    from ..template import (find_template, read_template, template_filename,
                            select as template_select)
    from .machine import machine_record
    if task is None:
        task = read_task(Path(base) / TASK_FILENAME)
    # A DESCRIPTION WITH NO BENCH is refused before any machine is read: a
    # refusal about the target would otherwise answer a question the
    # description never asks.
    why = bench_refusal(task)
    if why:
        raise PrepError(why)
    # THE ASK: the prep's flags over the description's allocation, by the
    # one fold every caller's ask is (`fold_allocation`).
    ask = fold_allocation(flags, task.allocation)[0]
    # The grid is enumerated from the TARGET's topology, not from whatever
    # box you happen to be typing on (P2, 2026-08-17).
    # READ, NOT SNAPSHOTTED: the bench card asks this on every edit, and
    # `prep bench` writes its snapshot at step 1 of the five, after the
    # under-way question (W52).
    if environment is None:
        environment = machine_record(base, target)
    # THE TARGET'S MENU, from the record in hand -- its probed queues -- for
    # every check below (`runtime_config.routing_of`): a calculation not
    # yet prepared cannot name the folder's menu when several machines are on
    # file.
    from ..runtime_config import routing_of
    every = menu = routing_of(environment)
    # A SCHEDULER WHOSE RECORD LISTS NO QUEUES has none to measure on --
    # prep's own refusal, in its words (`placement.no_queues_said`), so the
    # card says what prep will.
    if on_a_queue and environment.kind.queue and not menu:
        from .placement import no_queues_said
        raise PrepError(no_queues_said(base, target, who="the benchmark "))
    # A BENCHMARK MEASURES ON ONE QUEUE -- the one its prep names (`prep
    # --domain`, else `allocation.domain`; the user, 2026-10-17: "You need
    # one domain to bench"): every cell is checked against it alone.  A name
    # the record does not hold is the prep's refusal, in its words
    # (`placement.launch_refusal`).
    named = (getattr(ask, "domain", None) if queue is _FOLDED
             else queue or None)
    if named and menu and any(d.name == named for d in menu):
        menu = [d for d in menu if d.name == named]
    else:
        # NONE NAMED THAT THE RECORD LISTS -- the card before a queue is
        # picked: each cell fit on every queue, and none said to be its.
        named = None
    # A BENCHMARK THAT GOES TO NO QUEUE runs where it is launched: its cells
    # are the target's own topology's (`_cells_on_the_queue`).
    if not on_a_queue:
        every = menu = []
        named = None
    topo = getattr(environment, "topology", None)
    # GPUs per node as the record states them -- None when it does not say,
    # never read as 0 (`configuration.md` M-2).  WHERE THE RECORD LISTS
    # QUEUES, THE QUEUES' NODE TYPES ANSWER -- the one named first, else the
    # most any of the record's GPU queues holds, so a GPU cell proposed on
    # a CPU queue is shown left out by that queue's numbers (R6); the probed
    # node is one sampled node, never read as the cluster's (`gpu.md`
    # § 1.3; until 2026-10-19 the sample was asked first, so a 4-GPU login
    # sample proposed no 8-GPU cell).  A record with no queue is the
    # machine itself: its own count.
    node_gpus = getattr(topo, "gpus_per_node", None)
    gpn = ((_gpus_per_node(menu) or _gpus_per_node(every))
           if every else node_gpus) or 0
    cps = getattr(topo, "cores_per_socket", None)

    if template_text is None:
        try:
            _tpl = find_template(Path(base), task.label)
        except ValueError as exc:
            raise PrepError(str(exc)) from exc
        if _tpl is None:
            raise PrepError(f"no {template_filename(task.label)} beside "
                            f"{TASK_FILENAME}: a benchmark reads its grid's "
                            f"values from the template.")
        template_text = _tpl.read_text(encoding="utf-8")
    tmpl = read_template(template_text)
    # Through `select` -- `template.md` § 8.0 owns the rule.
    #
    # ``select`` rather than ``one`` because the question is *is it on?*: a
    # template that never carried the item answers "no", while ``one`` RAISES
    # on a name the file never had -- right for a caller that NEEDS the item,
    # wrong for one asking whether it exists.  One name for both engines
    # (`execution/gpu.md` G1).
    # THE DECLARED OVERRIDE LANE, split before anything is decided (user
    # rule, 2026-08-20): one-point non-machine entries are pins -- values
    # in force for every trial -- and the machine-answered entries are the
    # grid's axes.  A declared use_gpu pin OVERRIDES the template's
    # answer below, which is what makes the machine card's choice reach
    # the sweep without touching the template file.
    declared_pins, declared_axes, value_axes = _declared_execution_pins(
        task, bench_override)

    on_gpu = any(i.name == "use_gpu" and bool(i.value)
                 for i in template_select(tmpl, engine=task.engine))
    if "use_gpu" in declared_pins:
        on_gpu = bool(declared_pins["use_gpu"])

    # THE GRID-FAMILY AXIS (§ 4.3a, user 2026-08-21): use_gpu with two
    # points enumerates the machine grid once per flag -- the CPU family
    # holds the device count at G=0, the GPU family ranges it -- and the
    # flag rides each point as an ordinary value coordinate, so the deck's
    # answer and the point's family agree by construction (launch splits
    # the groups by each trial's GPU request, `model.gpu_request`).
    gpu_flags = None
    if "use_gpu" in value_axes:
        gpu_flags = [bool(v) for v in value_axes.pop("use_gpu")]
    families = gpu_flags if gpu_flags is not None else [bool(on_gpu)]
    mixed = gpu_flags is not None

    # What makes a trial a MEASUREMENT (the pins below) must win over what
    # it measures -- so an axis NAMING a measurement pin would render its
    # trials identical under different labels: one measurement, twice.
    _measured = sorted(set(value_axes) & set(_MEASUREMENT_PINS))
    if _measured:
        raise PrepError(
            f"task.json declares {', '.join(_measured)} as a value axis, "
            f"and the benchmark pins "
            f"{'it' if len(_measured) == 1 else 'them'} on every trial (a "
            f"trial is a measurement -- generator.md § 4.3a).  Its points "
            f"would render identical decks under different labels.  Drop "
            f"the entry.")

    # GPUs PER NODE, a count: the probed node's own, else the most a node of
    # the target's GPU queues holds -- a login node probes none, and the
    # cluster behind it has them (§ 4.3a: the probe records each partition's
    # gres on its domain row).  No card (`scheduler.md` R2a).  Needed only
    # to PROPOSE the device counts: declared counts are kept whatever the
    # record says, each cell's fit saying so (R6).
    if any(families) and not gpn:
        if not declared_axes.get("gpu_count"):
            # NAMED BY THE RECORD THAT ANSWERED, renewed in its own two
            # steps -- `record_and_renewal`, the one spelling.
            from ..scheduler.record import record_and_renewal
            _which, _redo = record_and_renewal(base, target)
            raise PrepError(
                f"this description asks for the GPU (use_gpu = "
                f"{'a cpu-vs-gpu axis' if mixed else 'true'}), so the "
                f"benchmark enumerates a GPU grid (G × ranks-per-GPU × "
                f"cores) -- and the record it reads ({_which}) "
                + ("does not say how many GPUs a node has"
                   if node_gpus is None
                   else "states no GPU on the node (gpus_per_node=0)")
                + (" and lists no queue with recorded GPUs" if on_a_queue
                   else ", where this benchmark runs (--no-sbatch)"
                   if environment.kind.queue
                   else ", where this benchmark runs")
                + f".  If that machine has "
                f"one, its record is out of date: {_redo}.  Or prep for the "
                f"machine that has the GPU (`--target`): the comparison is "
                f"by node type (asu-sol.md § 5.2).")

    # The axes come from the split above -- the value entries already left
    # as pins or value axes, so what remains is machine-answered by
    # construction.
    declared = {k: list(v) for k, v in declared_axes.items()}
    _KNOWN_AXES = ("mpi_np", "omp_threads", "gpu_count")
    _unresolvable = sorted(k for k in declared if k not in _KNOWN_AXES)
    if _unresolvable:
        # `max_memory_mb` (and PySCF's `threads`) are machine-answered
        # execution items TOO -- the helper hands every allocation item
        # through as an axis, and the grid resolves exactly these; without
        # this refusal a declared memory axis would be silently ignored.
        raise PrepError(
            f"task.json declares bench axes this machine translation does "
            f"not know: {', '.join(_unresolvable)}.  The axes a sweep can "
            f"resolve today are {', '.join(_KNOWN_AXES)} "
            f"(generator.md § 4.3a).")
    sockets = getattr(topo, "sockets", None)
    #: What the target's probed node holds -- what a cell is fit on where
    #: no queue answers for it (`_cells_on_the_queue`: no queues, or
    #: `--no-sbatch`).  Unknown when the record does not say both -- never
    #: a smaller figure (R3).
    cores_total = (sockets * cps) if (sockets and cps) else None
    # gpu_count alone does not declare a RANK grid: without mpi_np /
    # omp_threads the K x C half stays the machine's proposal, filtered
    # to the declared device counts below.
    grid_declared = bool(declared.get("mpi_np") or declared.get("omp_threads"))
    if grid_declared and not (declared.get("mpi_np")
                              and declared.get("omp_threads")):
        # BOTH OR NEITHER: a declared grid states its ranks and its cores
        # per rank.
        said, left = (("mpi_np", "omp_threads")
                      if declared.get("mpi_np") else ("omp_threads", "mpi_np"))
        raise PrepError(
            f"task.json's bench declares {said} and not {left} -- a "
            f"declared grid states both, the ranks and the cores per rank "
            f"(generator.md § 4.3a).  Declare {left}'s points too, or "
            f"neither, for the machine's proposal.")
    if grid_declared:
        # THE DECLARED GRID (§ 4.3a).  ``mpi_np`` is the TOTAL rank count a
        # point runs -- the same meaning it has everywhere else -- and
        # ``omp_threads`` the cores per rank.  A point is measured as
        # declared, never clamped: a clamped point would measure a
        # configuration nobody declared.
        ranks = [int(v) for v in declared["mpi_np"]]
        cores = [int(v) for v in declared["omp_threads"]]
        # A DECLARED POINT IS NOT JUDGED HERE against the probed topology:
        # a partition is a QUEUE holding many machine kinds (R0).  Each cell
        # is fit on the queue instead, in `_cells_on_the_queue` -- a cell
        # declared whole kept, its fit said -- or on this same topology
        # where no queue answers for it, because a workstation IS its own
        # machine.

    # THE DECLARED DEVICE COUNTS (user, 2026-08-21: "explicit is what we
    # need").  Declared, gpu_count is exact: those G values and no others,
    # each its own shelf -- a cell declared whole kept whatever the record
    # says, its fit on the queue saying what holds it (`scheduler.md` R6); a
    # (mpi_np, G) pair that cannot split EVENLY is dropped by name below
    # -- a trial is named and sized by its ranks per GPU (mpi_np = G·K),
    # which an uneven split does not have, and a benchmark never rounds a
    # number -- and refusing the whole prep would deny the divisible cells
    # and the CPU family a rank count they hold fine.
    gpu_counts = ([int(v) for v in declared.get("gpu_count")]
                  if declared.get("gpu_count") else None)
    if gpu_counts and not any(families):
        raise PrepError(
            "task.json declares gpu_count, but this bench resolves to the "
            "CPU family (use_gpu is false and not an axis) -- the "
            "device counts would be silently ignored.  Declare use_gpu "
            "= [true] (or the [true, false] axis), or drop gpu_count.")

    #: The (mpi_np, gpu_count) pairs dropped by the even-split rule, for
    #: the refusal that points at them.
    dropped: list = []

    def _family_cells(fam):
        """The machine cells of ONE family, as (G, K, C) -- G=0 is the CPU
        family's held coordinate (plain ranks), G>=1 the device count with
        G*K == the total rank count."""
        if grid_declared:
            if fam:
                counts = gpu_counts or range(1, gpn + 1)
                cells = sorted({(g, r // g, c)
                                for r in ranks for c in cores
                                for g in counts if r % g == 0})
                if gpu_counts:
                    bad = sorted({(r, g) for r in ranks for g in gpu_counts
                                  if r % g})
                    if bad:
                        dropped.extend(bad)
                        note(
                            "  dropped (a trial counts its ranks per GPU, "
                            "and these do not split EVENLY over the "
                            "GPUs): "
                            + ", ".join(f"mpi_np={r} x gpu_count={g}"
                                        for r, g in bad))
                return cells
            return [(0, r, c) for r in ranks for c in cores]
        # THE MACHINE PROPOSES FROM WHAT ITS RECORD SAYS: no cores per
        # socket, no proposal.
        ks = sweep_K(topo)
        if not ks:
            raise PrepError(
                "this machine's record does not say its cores per socket, "
                "so it proposes no grid -- declare the bench's mpi_np and "
                "omp_threads in task.json, or re-probe the machine "
                "(`molbuilder jobset probe --write`).")
        # ONE enumeration, both grids (`bench/grid.py`: the single source
        # of truth for the sweep grid, so no two consumers can define it
        # differently).  On CPU there is no device to range over, so G is
        # held at 0 and dropped from a single-family coordinate below.
        # A declared gpu_count FILTERS the probed grid to exactly those
        # device counts.
        # DECLARED DEVICE COUNTS ARE THE COUNTS, whatever the probed node
        # carries: the grid ranges up to the largest declared.
        reach = max(gpu_counts) if (fam and gpu_counts) else (gpn if fam
                                                              else 1)
        return [(g if fam else 0, k, c)
                for g, k, c in sweep_grid(reach, cps, ks, None)
                if not (fam and gpu_counts) or g in gpu_counts]

    # ENUMERATE, FIT EACH CELL ON THE QUEUE, SHOW THE OUTCOME (user,
    # 2026-08-30: "we don't have to fight with what language we use to
    # indicate error, but present the correct outcome"; 2026-10-17: the
    # record is shown, never a veto).  A PROPOSED cell no node holds is not
    # proposed; a cell DECLARED whole is kept whatever the record says, its
    # fit beside it -- the scheduler decides (`generator.md` § 4.3a).
    plan = [(fam, cell) for fam in families for cell in _family_cells(fam)]
    checked = _cells_on_the_queue(plan, routing=menu,
                                  local_cores=cores_total,
                                  local_gpus=node_gpus)
    # ...AND WHAT THIS PREP DOES NOT ALLOW: a cell past the person's own ask
    # is left out by name, proposed or declared -- two of their statements
    # disagreeing.  A cell is DECLARED when every coordinate of it is -- its
    # ranks and cores per rank, and on the GPU its device count; one with a
    # coordinate the machine proposed is the machine's proposal.
    marked = []
    for fam, cell, lines, over in checked:
        past = _past_the_ask(fam, cell, ask)
        declared_whole = grid_declared and (not fam or bool(gpu_counts))
        marked.append((fam, cell, lines,
                       past + ([] if declared_whole else list(over))))
    checked = marked

    kept    = [(f, cell, lines) for f, cell, lines, why in checked if not why]
    crossed = [(f, cell, why) for f, cell, lines, why in checked if why]

    # THE SAME REPORT THE TERMINAL PRINTS, AS DATA.  The task-setup card
    # shows this list live beside the axes being edited.  One enumerator,
    # two renderings.
    if report is not None:
        report.extend(
            {"label": _cell_label(g, k, c),
             "shape": _cell_shape(g, k, c),
             "family": "gpu" if fam else "cpu",
             "ranks": _cell_ranks(fam, g, k), "cores_each": c,
             "gpus": (g if (fam and g) else 0),
             # The wire carries the SENTENCES: the browser renders prose.
             "fit": list(lines), "why": [i.message for i in why]}
            for fam, (g, k, c), lines, why in checked)

    # The menu these cells are checked against is the TARGET's -- `prep --target sol` on a workstation reads
    # the record for Sol, a cluster you are not standing on at all
    # (`preparing-for-another-machine.md`).  So the count is about the
    # QUEUES, and says so.
    note(f"  bench grid: {len(checked)} combination(s) enumerated, "
         f"{len(kept)} kept" + (f" on {named}" if named else ""))
    for fam, (g, k, c), lines in kept:
        note(f"    {_cell_label(g, k, c):<11} "
             f"{_cell_shape(g, k, c):<37}")
        # EACH CELL'S FIT, said under it (`generator.md` § 4.3a) -- a
        # declared cell is kept whatever it says (R4, R6).
        for said in lines:
            note(f"      {said['line']}")
    # THE QUEUE'S JOB CAP (`scheduler.md` R14), said here where the grid can
    # still change: each kept cell is one shelf, one submitted job.  A cap
    # the record does not state is no limit (R3).
    cap = (getattr(menu[0], "max_submit_jobs", None)
           if named and len(menu) == 1 else None)
    if type(cap) is int and len(kept) > cap:
        note(f"  {named} takes {cap} submitted job(s) per user, and this "
             f"benchmark is {len(kept)} shelf job(s): with nothing of yours "
             f"already queued there, {cap} go and {len(kept) - cap} come back "
             f"QOSMaxSubmitJobPerUserLimit -- fewer if you already hold some. "
             f"A refused shelf's trials stay pending, and launching the "
             f"benchmark again sends exactly them.")
    if crossed:
        note(f"  left out ({len(crossed)}) -- "
             + (f"no node of {named} holds them" if named else
                "no node of the target's queues holds them" if menu else
                "the target's probed node does not hold them")
             + ", or this prep's ask does not allow them:")
        for fam, (g, k, c), why in crossed:
            # `.message`, NOT the finding itself, whose repr is not a
            # sentence.  The numbers are fields so callers can read them; the
            # SENTENCE is what a person is shown.
            note(f"    {_cell_label(g, k, c):<11} "
                 f"{_cell_shape(g, k, c):<37}  "
                 + "; ".join(w.message for w in why))

    cells = []
    for fam in families:
        fcells = [cell for f, cell, _dom in kept if f == fam]
        if fam and fcells:
            # The GPU-sharing note (user, 2026-08-23): ALWAYS state
            # ranks/GPU -- a cell's K -- warn past MPS's client ceiling, note
            # past this stack's tuned point.  Whether the cores fit a node is
            # each cell's fit above.  The two thresholds are the ones
            # `ask.gpu_share_notes` states at a run's prep, so the two
            # displays cannot disagree.
            from .ask import (GPU_TUNED_RANKS_PER_DEVICE,
                              MPS_MAX_CLIENTS_PER_DEVICE)
            shares = sorted({(g, k) for g, k, c in fcells})
            bits = []
            for g, k in shares:
                flag = ("  <- WARNING, past MPS's ceiling"
                        if k > MPS_MAX_CLIENTS_PER_DEVICE else
                        "  <- past the tuned point"
                        if k > GPU_TUNED_RANKS_PER_DEVICE else "")
                bits.append(f"G{g}K{k}: {k} rank(s)/GPU{flag}")
            note("  GPU sharing in this family: " + ", ".join(bits))
        if fam and not fcells:
            # every GPU cell was left out above, or fell to the
            # even-split rule -- both are listed by name, so this names
            # the consequence.
            if mixed:
                note(
                    "  NOTE: no GPU cell is left -- this sweep measures "
                    "only the CPU family.")
            else:
                # POINTING AT WHAT IS THERE: the cells left out, the pairs
                # dropped, either or both.
                seen = [w for w, there in (
                    ("the cells left out", any(f for f, _c, _w in crossed)),
                    ("the pairs dropped by the even-split rule",
                     bool(dropped))) if there]
                raise PrepError(
                    "no GPU cell is left"
                    + (f" -- see {' and '.join(seen)} above" if seen else "")
                    + ".  Adjust mpi_np / gpu_count in task.json, name a "
                      "queue whose GPU nodes hold them, or declare the cells "
                      "whole -- mpi_np, omp_threads and gpu_count: a "
                      "declared cell is kept, its fit said.")
        cells.extend((fam, cell) for cell in fcells)
    # THE VALUE-AXIS CARTESIAN (§ 4.3a): every machine cell is crossed
    # with the remaining declared value axes, in declaration order, and
    # each point CARRIES its coordinates -- the resolver's ordinary
    # parameter lane applies them to that trial's config (provenance
    # "sweep"), and the coordinate rides the trial's name and, as data,
    # `job-set.json`'s per-trial ``point``.
    if not cells:
        raise PrepError(
            "no bench cell is left -- see the cells left out above.")
    combos = [{}]
    for name, vals in value_axes.items():
        combos = [{**c, name: v} for c in combos for v in vals]

    points = []
    for fam, (g, k, c) in cells:
        # EVERY TRIAL CARRIES G -- ``G0`` for a CPU trial, on every
        # machine.
        if mixed:
            coord = {"G": g, "K": k, "C": c, "use_gpu": fam}
        else:
            coord = {"G": g, "K": k, "C": c}
        points.extend({**coord, **vc} for vc in combos)

    if mixed:
        # ONE translation serves both families: G=0 maps to plain ranks
        # and NO gres, G>=1 to G*K ranks plus the device ask -- which is
        # what lets `launch` put the CPU group on an allocation that
        # holds no device (§ 4.3a's split submission).
        translation = MachineTranslation(
            axes=("G", "K", "C"),
            to_resources=lambda p, _env: (
                {"mpi_np": p["G"] * p["K"], "cpus_per_task": p["C"],
                 "gres": f"gpu:{p['G']}"} if p["G"] else
                {"mpi_np": p["K"], "cpus_per_task": p["C"]}))
    elif on_gpu:
        translation = MachineTranslation(
            axes=("G", "K", "C"),
            to_resources=lambda p, _env: {
                "mpi_np": p["G"] * p["K"], "cpus_per_task": p["C"],
                "gres": f"gpu:{p['G']}"})
    else:
        translation = MachineTranslation(
            axes=("G", "K", "C"),
            to_resources=lambda p, _env: {
                "mpi_np": p["K"], "cpus_per_task": p["C"]})
    # The trial pins -- schema values resolved like any other (`template.md`
    # § 8.1: rebuild and render, never splice): capped SCF, single point,
    # forced cold -- and ``scf_must_converge: False``, the switch that makes
    # the cap CLEAN (`choose_winner` ranks only completed points).
    #
    # What is pinned here is what makes a trial a MEASUREMENT rather than a
    # run.  What the calculation IS -- the GPU, the eigensolver, the block
    # size -- is the description's, and pinning it here would measure a
    # configuration nobody asked to run.
    #
    # ``continue_retries: 0`` is what makes the trial run ONCE.  Without it
    # the capped SCF above guarantees non-convergence, the wrapper's retry
    # budget reads that as a failure and re-runs, and `summarize` reads the
    # HIGHEST run index -- so every trial that retried was timed on its second
    # run and every trial that did not was timed on its first.  Those are not
    # comparable, which is the one thing a sweep exists to be.
    # The declared values ride UNDER the measurement pins: what makes a
    # trial a measurement (capped SCF, forced cold, run-once) must win
    # over any declaration -- one-point declarations and value-axis
    # coordinates alike (the resolver applies pins over a point's
    # parameters, and the overlap refusal above bars an AXIS on a pin).
    pins = {**declared_pins, **_MEASUREMENT_PINS}
    return points, pins, translation
