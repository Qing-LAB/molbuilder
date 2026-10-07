"""What a prep receives, assembled ONCE -- `execution/architecture.md` A12.

Two tuples, one per kind:

    prep_run_inputs  ->  (allocation, pins, chosen)       a run
    bench_inputs     ->  (points, pins, translation)      a benchmark sweep

and the pieces they are made of: the run's condition (`Task.run_condition`,
read by `declared_run_shape`, `run_inputs`), the bench's
declared pins and axes (`_declared_execution_pins`), and the bench grid --
enumerated, checked cell by cell against the target's queues, and reported
(`_cells_this_machine_holds`, `_rank_reasons`, `_local_refusals`, `_cell_*`).

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
    (`validation/task.py::_bench_points_fit_their_items`): an enum value
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
    Point SHAPES are `read_task`'s (task.py `_bench_from_obj`): every
    value arrives as a non-empty tuple of scalars, so no scalar/empty
    arms exist here.

    ``task`` is the description its caller holds -- read once by the prep
    that asks (W55 B1).
    """
    from ..validation.task import _bench_points_fit_their_items
    from .. import template as _T

    if bench_override is not None:
        # THE AXES AS THEY ARE BEING EDITED, not as they were last saved.
        # The task-setup card resolves its grid live, and its edits live in
        # the browser's model until the person saves -- so a live answer
        # read from disk would describe the previous state.  Through
        # `_bench_from_obj`, the ONE normaliser, so the override arrives
        # in the same shape (and earns the same refusals) as a declaration
        # that came off disk.
        import dataclasses as _dc

        from ..task import _bench_from_obj
        task = _dc.replace(task,
                           bench=_bench_from_obj({"bench": bench_override}))
    declared = {k: list(v) for k, v in (task.bench or {}).items()}
    if not declared:
        return {}, {}, {}

    shape_errors = [i for i in _bench_points_fit_their_items(task)
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


def _gpus_per_node(base, routing=None) -> "int | None":
    """The most GPUs one node of the target's GPU queues holds, or ``None``
    when no queue lists any -- § 4.3a's fallback when the probed node itself
    has none (a login node).  ``routing`` is the target's own menu when the
    caller holds its record; else the folder's.

    A COUNT, and no card: which card a node carries is the machine's
    business (`scheduler.md` R2a; user, 2026-10-01).
    """
    from ..runtime_config import get_routing
    from ..scheduler.place import candidates
    rows = candidates(routing if routing is not None
                      else get_routing(project_dir=Path(base)),
                      prefer_gpu=True)
    counts = [d.per_node for r in rows for d in r.devices
              if d.per_node is not None]
    return max(counts) if counts else None


def _cells_this_machine_holds(base, plan, *,
                              local_cores=None, local_gpus=None,
                              routing=None):
    """Every enumerated bench cell, checked one by one against THIS
    machine's queues -- ``[(fam, (g, k, c), domains, why)]`` in enumeration
    order, ``domains`` naming every queue that would take the cell and
    ``why`` the reasons it fits nowhere.  Exactly one of the two is
    non-empty, so ``why`` alone decides kept-vs-crossed.

    **Enumerate everything, then cross out what the machine cannot hold,
    then show what is left** (user, 2026-08-30).

    The check is `scheduler.admits` over `scheduler.place.candidates` --
    the same pair `place` itself walks, so "some queue admits it" here and
    "placeable" at launch are one verdict, not two.  What is NOT done here
    is CHOOSING one: the queue each of a benchmark's sides goes to is named
    at its launch (`--domain`, `--gpu-domain`; `scheduler.md` R7).  So the
    row lists WHERE IT COULD GO.

    A machine with no queue menu at all is answered by ``local_cores`` /
    ``local_gpus`` -- the probed topology of the box this runs on, which
    IS the machine when nothing schedules for it.  That is the ONLY place
    topology may bound a cell: on a cluster it measures whichever node the
    probe happened to land on, and R0 says a queue holds many kinds.
    """
    from ..runtime_config import get_routing
    from ..scheduler.admit import Request, admits
    from ..scheduler.place import candidates
    if routing is None:
        try:
            routing = get_routing(project_dir=Path(base))
        except Exception:                                   # noqa: BLE001
            # No readable menu is not a small menu (R3): a record we cannot
            # read must not cross out work the machine may well run.
            routing = []

    if not routing:
        return [(fam, cell, (), _local_refusals(cell, fam,
                                                local_cores, local_gpus))
                for fam, cell in plan]

    out = []
    for fam, (g, k, c) in plan:
        want_gpu = bool(fam and g)
        req = Request(ranks=_cell_ranks(want_gpu, g, k), cpus_per_task=c,
                      gpus=g if want_gpu else None)
        # THE POOL IS THE FIT QUESTION'S, NOT THE PREFERENCE'S.  For a
        # device cell it is `candidates` -- gpu-capability is a filter,
        # since a cpu-only row states no inventory and R3 would read that
        # silence as permission.  For a CPU cell it is the WHOLE menu: a
        # CPU job runs anywhere, and `candidates` narrows to cpu-only rows
        # to express a PREFERENCE (idle devices cost).  Letting a
        # preference decide the fit question would cross out a 128-rank
        # CPU cell that Sol's `public` runs on 107 nodes, because the two
        # cpu-only queues happen to be smaller.
        pool = candidates(routing, prefer_gpu=True) if want_gpu \
            else list(routing)
        fits, why = [], []
        for row in pool:
            no = admits(row, req)
            if no:
                why.extend(no)
            else:
                fits.append(row.name)
        if not fits and not why:
            # An EMPTY POOL says nothing, and silence here would be read
            # as "it fits": the kept/crossed split is on ``why``, so a
            # cell nothing could even be offered to must carry a reason.
            # `place` refuses this case in the same words.
            from ..scheduler.admit import Refusal
            why = [Refusal("no_queue", "this machine",
                           note="no gpu-capable queue" if want_gpu
                                else "no queue for cpu work")]
        out.append((fam, (g, k, c), tuple(fits),
                    () if fits else _rank_reasons(why)))
    return out


def _rank_reasons(reasons):
    """The refusals worth showing, most actionable first.

    A cell no queue takes collects one reason per queue; duplicates -- Sol
    repeats the same node groups across debug/htc/general -- collapse, and
    the shortest come first.
    """
    seen, ranked = set(), []
    for r in reasons:
        key = (r.limit, r.message)
        if key not in seen:
            seen.add(key)
            ranked.append(r)
    ranked.sort(key=lambda r: len(r.message))
    return tuple(ranked)


def _local_refusals(cell, fam, cores_total, gpus_per_node):
    """Why THIS BOX cannot hold a cell -- the no-scheduler answer.

    Empty means it fits, or that the probe measured nothing to compare
    against: an unmeasured topology is silence, and silence never bars
    (R3).  Reasons name their numbers (R4).
    """
    # FINDINGS, NOT SENTENCES, as `admits` returns.  Everything downstream
    # reads the FIELDS: `_rank_reasons` dedups on `(limit, message)`, and
    # the task-setup card renders `.message` per cell.  The numbers stay
    # numbers; the sentence is rendered at the edge, once.
    from ..scheduler.admit import Refusal
    g, k, c = cell
    ranks = _cell_ranks(fam, g, k)
    why = []
    if cores_total and ranks * c > cores_total:
        why.append(Refusal("cores", "this machine",
                           asked=ranks * c, allowed=cores_total,
                           unit="cores"))
    if fam and g and gpus_per_node and g > gpus_per_node:
        why.append(Refusal("gpus", "this machine",
                           asked=g, allowed=gpus_per_node, unit="GPUs"))
    return tuple(why)


def _past_the_ask(fam, cell, ask) -> list:
    """Why a cell is past what this prep allows -- its ranks, its cores per
    rank, its GPUs against the ask's (`generator.md` § 4.1) -- as the
    findings a queue's refusal is (`admit.Refusal`); empty when it fits or
    nothing was asked."""
    if ask is None:
        return []
    from ..scheduler.admit import Refusal
    from ..scheduler.quantities import parse_gres_flag
    g, k, c = cell
    gpus = g if (fam and g) else 0
    out = []
    for limit, asked, allowed, unit in (
            ("ranks", _cell_ranks(fam, g, k), ask.mpi_np, "ranks"),
            ("cpus_per_task", c, ask.cpus_per_task, "cores per rank"),
            ("gpus", gpus,
             parse_gres_flag(ask.gres) if ask.gres not in (None, "")
             else None, "GPUs")):
        if allowed is not None and asked > int(allowed):
            out.append(Refusal(limit, "this prep's ask", asked=asked,
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


#: The catalogue's words for a launch field, and `Resources`' own.  Most
#: agree; these never did.  It is a NAME MAP and nothing else -- no default,
#: no enumeration, no arithmetic.  PySCF's ``threads`` is SIESTA's
#: ``omp_threads`` -- the cores one process runs on (`job-contracts.md`
#: § 6.2) -- and an item missing here is DROPPED by `declared_run_shape`.
#: `stages.md` § 6.8e -- the run's own wall and queue, carried through the
#: direct map under their own names (they are already `Resources` fields).
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
    by the prep entry, of the job it resolves (`prep._resolve_stage`,
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
    """Everything ``prep run`` needs, assembled ONCE -- ``(allocation, pins,
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

    So the browser's prep button and ``molbuilder jobset prep run`` call
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
    #     the allocation is, once, at step 2 (`prep._fold_allocation`: a
    #     flag, then the run card, then the description).
    chosen, cond_pins = run_inputs(base, task, stage)
    # WHETHER A GPU RUN STATES HOW MANY, AND A CPU RUN NONE (`execution/
    # gpu.md` G5) is the prep entry's to ask, of the job it resolves from
    # this assembly, beside the launch values' refusal -- and the Task
    # setup card shows the entry's preview of it.

    # 1b · THE CALCULATION'S SCHEDULER ASK is folded where the stage is
    #      resolved, once (`prep._resolve_stage`, `prep._fold_allocation`:
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


def bench_inputs(base, target, *, bench_override=None, report=None,
                 notes=None, task=None, environment=None,
                 template_text=None, ask=None):
    """The benchmark specialisation's three inputs — `project-layout.md`

    ``target`` is REQUIRED: this function exists to read a SPECIFIC
    machine's hardware.  Pass ``None`` deliberately to mean "this machine".

    § 2.3.1a's split, stated as data: WHERE the values come from (the grid,
    enumerated from THIS machine's probed topology, as explicit points), the
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

    ``ask`` is what the person allows this prep -- its flags' ranks, cores
    per rank and GPUs (`generator.md` § 4.1, the allocation the sweep must
    fit inside): a cell past it is crossed out by name with the cells no
    queue holds, and the prep refused only when none is left (§ 4.3a: a
    cell that fits is never denied by a sibling that does not).
    """
    # WHAT A PERSON IS TOLD comes back to the caller, never printed here:
    # the command line prints it, the Task setup tab shows it
    # (`prep.prep_stage`, `job-system.md` § 5.3).
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
    # The grid is enumerated from the TARGET's topology, not from whatever
    # box you happen to be typing on (P2, 2026-08-17).
    # READ, NOT SNAPSHOTTED: the bench card asks this on every edit, and
    # `prep bench` writes its snapshot at step 1 of the five, after the
    # under-way question (W52).
    if environment is None:
        environment = machine_record(base, target)
    # THE TARGET'S MENU, from the record in hand -- its probed queues -- for
    # every check below (`runtime_config.routing_of`): a calculation not
    # yet prepped cannot name the folder's menu when several machines are on
    # file.
    from ..runtime_config import routing_of
    menu = routing_of(environment)
    topo = getattr(environment, "topology", None)
    # GPUs per node as the record states them -- None when it does not say,
    # never read as 0 (`configuration.md` M-2).
    node_gpus = getattr(topo, "gpus_per_node", None)
    gpn = node_gpus or 0
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
    # (`execution/gpu.md` § 4).
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
    # gres on its domain row).  No card (`scheduler.md` R2a).
    if any(families) and not gpn:
        gpn = _gpus_per_node(base, menu) or 0
        if not gpn:
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
                + f" and lists no queue with recorded GPUs.  If that machine has "
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
    #: What the LOCAL box holds -- the ceiling only when this machine has
    #: no queues to answer for it (`_cells_this_machine_holds`).  Unknown
    #: when the record does not say both -- never a smaller limit (R3).
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
        # ``omp_threads`` the cores per rank.  A point the machine cannot
        # hold is refused BY NAME, not clamped: a clamped point would
        # measure a configuration nobody declared.
        ranks = [int(v) for v in declared["mpi_np"]]
        cores = [int(v) for v in declared["omp_threads"]]
        # A DECLARED POINT IS NOT REFUSED HERE against the probed topology:
        # a partition is a QUEUE holding many machine kinds (R0).  The
        # queues answer instead, cell by cell, in
        # `_cells_this_machine_holds` -- which falls back to this same
        # topology when there is no queue menu at all, because a
        # workstation IS its own machine.

    # THE DECLARED DEVICE COUNTS (user, 2026-08-21: "explicit is what we
    # need").  Declared, gpu_count is exact: those G values and no others,
    # each its own shelf.  A count the machine does not have is refused by
    # name (the same rule as a rank count the machine cannot hold); a
    # (mpi_np, G) pair that cannot split EVENLY is dropped by name below
    # -- ELPA's own rule is the same rank count on every device
    # (tuning.md § 2.12), and refusing the whole prep would deny the
    # divisible cells and the CPU family a rank count they hold fine.
    gpu_counts = ([int(v) for v in declared.get("gpu_count")]
                  if declared.get("gpu_count") else None)
    if gpu_counts and not any(families):
        raise PrepError(
            "task.json declares gpu_count, but this bench resolves to the "
            "CPU family (use_gpu is false and not an axis) -- the "
            "device counts would be silently ignored.  Declare use_gpu "
            "= [true] (or the [true, false] axis), or drop gpu_count.")
    if gpu_counts and any(families):
        _over = sorted(g for g in gpu_counts if g > (gpn or 0))
        if _over:
            raise PrepError(
                f"task.json declares gpu_count = {_over!r} and this "
                f"machine's record holds {gpn or 0} device(s) per node.  "
                f"Trim the declaration, or benchmark on the machine it "
                f"is meant to measure.")

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
                        note(
                            "  dropped (ranks must split EVENLY over the "
                            "devices -- ELPA's equal-share rule, "
                            "tuning.md § 2.12): "
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
        return [(g if fam else 0, k, c)
                for g, k, c in sweep_grid(gpn if fam else 1, cps, ks, None)
                if not (fam and gpu_counts) or g in gpu_counts]

    # ENUMERATE EVERYTHING, THEN CROSS OUT WHAT THIS MACHINE CANNOT HOLD,
    # THEN SHOW WHAT IS LEFT (user, 2026-08-30: "we don't have to fight
    # with what language we use to indicate error, but present the correct
    # outcome").  The grid is a proposal; the machine record is the thing
    # that decides; and what the person needs on screen is the surviving
    # list, not a sentence about a number.  Both families go through it.
    plan = [(fam, cell) for fam in families for cell in _family_cells(fam)]
    checked = _cells_this_machine_holds(base, plan,
                                        local_cores=cores_total,
                                        local_gpus=gpn, routing=menu)
    # ...AND WHAT THIS PREP DOES NOT ALLOW: a cell past the person's own ask
    # is crossed out by name, beside the ones no queue holds.
    marked = []
    for fam, cell, doms, why in checked:
        past = _past_the_ask(fam, cell, ask)
        marked.append((fam, cell, () if past else doms, past + list(why)))
    checked = marked

    kept    = [(f, cell, doms) for f, cell, doms, why in checked if not why]
    crossed = [(f, cell, why) for f, cell, doms, why in checked if why]

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
             # The wire carries the SENTENCES; the finding's `where` is for
             # callers inside the process (`_rank_reasons` ranks on it).  The
             # browser renders prose, so it gets prose.
             "fits": list(doms), "why": [i.message for i in why]}
            for fam, (g, k, c), doms, why in checked)

    # The menu these cells are checked against is the TARGET's -- `prep --target sol` on a workstation reads
    # the record for Sol, a cluster you are not standing on at all
    # (`preparing-for-another-machine.md`).  So the count is about the
    # QUEUES, and says so.
    note(f"  bench grid: {len(checked)} combination(s) enumerated, "
         f"{len(kept)} fit a queue")
    for fam, (g, k, c), doms in kept:
        # WHERE IT COULD GO, not where it will: the wall decides that and
        # is stated at `launch`.  Four names then an ellipsis -- enough to
        # see whether a cell has real room or is riding one queue.
        where = ", ".join(doms[:4]) + (" ..." if len(doms) > 4 else "")
        note(f"    {_cell_label(g, k, c):<11} "
             f"{_cell_shape(g, k, c):<37}"
             + (f"  fits: {where}" if where else ""))
    if crossed:
        note(f"  crossed out ({len(crossed)}) -- no queue takes them, or "
             f"this prep's ask does not allow them:")
        for fam, (g, k, c), why in crossed:
            # `.message`, NOT the Refusal itself, whose repr is not a
            # sentence.  The numbers are fields so callers can read them; the
            # SENTENCE is what a person is shown.
            note(f"    {_cell_label(g, k, c):<11} "
                 f"{_cell_shape(g, k, c):<37}  "
                 f"{why[0].message if hasattr(why[0], 'message') else why[0]}")

    cells = []
    for fam in families:
        fcells = [cell for f, cell, _dom in kept if f == fam]
        if fam and fcells:
            # Checks 1-3 of the GPU-sharing note (user, 2026-08-23): ALWAYS
            # state ranks/GPU, warn past MPS's 48-client ceiling, note past
            # this stack's ~4-rank tuned point.  Check 4 (node-fit) is the
            # crossing-out above -- this does not re-derive it, only
            # states the sharing fact for whatever survived it.  ONE
            # function (`ask.gpu_share_notes`) so this and the submission
            # display can never disagree about the arithmetic.
            from .ask import gpu_share_notes
            shares = sorted({(g, k) for g, k, c in fcells})
            bits = []
            for g, k in shares:
                share = gpu_share_notes(g, k)
                flag = ""
                if len(share) > 1:
                    flag = ("  <- WARNING, past MPS's ceiling"
                           if "WARNING" in share[1] else
                           "  <- past the tuned point")
                bits.append(f"G{g}K{k}: {k} rank(s)/GPU{flag}")
            note("  GPU sharing in this family: " + ", ".join(bits))
        if fam and not fcells:
            # every GPU cell was crossed out above, or fell to the
            # even-split rule -- both are listed by name, so this names
            # the consequence.
            if mixed:
                note(
                    "  NOTE: no GPU cell survived -- this sweep measures "
                    "only the CPU family.")
            else:
                raise PrepError(
                    "no GPU cell survived (see the crossed-out list "
                    "above, and the even-split rule).  Adjust mpi_np / "
                    "gpu_count in task.json, name a domain whose GPU "
                    "nodes are larger, or benchmark the card this "
                    "machine's queues actually offer.")
        cells.extend((fam, cell) for cell in fcells)
    # THE VALUE-AXIS CARTESIAN (§ 4.3a): every machine cell is crossed
    # with the remaining declared value axes, in declaration order, and
    # each point CARRIES its coordinates -- the resolver's ordinary
    # parameter lane applies them to that trial's config (provenance
    # "sweep"), and the coordinate rides the trial's name and, as data,
    # `job-set.json`'s per-trial ``point``.
    if not cells:
        raise PrepError(
            "no bench cell survived the declaration on this machine -- "
            "see the crossed-out list above.")
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
