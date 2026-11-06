"""What a prep receives, assembled ONCE -- `execution/architecture.md` A12.

Two tuples, one per kind:

    prep_run_inputs  ->  (allocation, pins, chosen)       a run
    bench_inputs     ->  (points, pins, translation)      a benchmark sweep

and the pieces they are made of: the run's condition (`JobRequest.
run_condition`, read by `resolve.declared_run_shape`, `resolve.card_settings`),
the bench's declared pins and axes (`_declared_execution_pins`), and the bench grid --
enumerated, fit cell by cell on the benchmark's queue, and reported
(`_cells_on_the_queue`, `_local_over`, `_past_the_ask`, `_cell_*`).

**The conductor's own assembly, beside it** (`architecture.md` § 2.2:
only a surface imports the conductor, and its own modules).  A refusal is a
`PrepError`, which each surface shows as it is, and what a person is TOLD
comes back through the ``notes`` list a caller hands in -- the command line
prints it, the tab shows it -- rather than being printed here.
"""
from __future__ import annotations

from pathlib import Path

from ..resolve import (MACHINE_AXES, MachineTranslation, card_settings,
                       declared_run_shape, point_token)
from .errors import PrepError
from .model import Resources


def _declared_execution_pins(request, bench_override=None):
    """The job request's ``bench``, read as the OVERRIDE LANE it is (user
    rule, 2026-08-20; `generator.md` § 4.3a): several points = an axis to
    try, ONE point = the value the bench's trials run with, applied at prep
    as a pin for them alone (the run's values are its own card's,
    `job-request.md` § 3.2).  Nothing migrates between files: the request
    stays exactly as edited, and prep is where a declaration is resolved.
    The run's condition is split by the same rule
    (`resolve.declared_run_shape`, `resolve.card_settings`).

    Returns ``(pins, axes, value_axes)``: the one-point non-machine values
    as a pins dict, the machine-answered entries untouched (the grid's
    axes), and the MULTI-point non-machine entries as value axes --
    ``{name: [points...]}`` -- which `bench_inputs` crosses with the
    machine grid so every trial's deck carries its coordinates
    (§ 4.3a's built rule).

    Refused BY NAME, never repaired -- through the ONE checker
    (`validation/task.py::bench_points_fit_their_items`, which asks the
    setting door, `engines/template.md` § 5.4): a row naming no run
    setting, a point its item does not take, a repeated point.  The
    dispatch runs the full preflight BEFORE this helper, so on either door
    of the one prep entry the refusal normally lands there with every
    finding listed; the call here is the backstop for direct callers,
    first-refusal shaped as the entry's errors are.  Point SHAPES are the
    request's reader's (`job_request.bench_from_obj`): every value arrives
    as a non-empty tuple of scalars, so no scalar/empty arms exist here.

    ``request`` is the job request its caller holds -- read once by the
    prep that asks (W55 B1).
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

        from ..job_request import bench_from_obj
        request = _dc.replace(request,
                              bench=bench_from_obj({"bench": bench_override}))
    declared = {k: list(v) for k, v in (request.bench or {}).items()}
    if not declared:
        return {}, {}, {}

    shape_errors = [i for i in bench_points_fit_their_items(request)
                    if i.severity == "error"]
    if shape_errors:
        raise PrepError(shape_errors[0].message)

    # THE ENGINE IS THE DESCRIPTION'S, held by the request read against it,
    # so this door and the shape checker it calls are keyed on one source.
    items = {i.name: i
             for i in _T.select(_T.catalogue(), engine=str(request.engine))
             if _T.is_run_setting(i)}
    pins, axes, value_axes = {}, {}, {}
    for name, pts in declared.items():
        it = items[name]                     # every name: the checker's
        pts = [_T.spell(it, v) for v in pts]
        if it.allocation:
            axes[name] = pts                 # machine-answered: the grid's
            continue                         # business, never a value
        if len(pts) > 1:
            value_axes[name] = pts           # a VALUE AXIS (§ 4.3a): the
            continue                         # grid multiplies per point
        pins[name] = pts[0]
    return pins, axes, value_axes


def _cells_on_the_queue(plan, *, routing, local_cores=None,
                        local_gpus=None):
    """Every bench cell with its fit on the benchmark's queue --
    ``[(fam, (g, n, c), fits)]`` in order: ``fits`` the cell's fit on each
    queue of ``routing``, as ``{queue, holds, line}`` (`scheduler.fit`, the
    one `place` itself asks, so a cell and a run get one reading).  Said,
    never a veto (`generator.md` § 4.3a).

    ``routing`` is the benchmark's ONE queue (:func:`bench_inputs`): a
    benchmark measures on the queue its prep names (the user, 2026-10-17);
    the Task setup card, before one is named, asks every queue of the
    record, so the person sees each one's fit while choosing.  A machine
    with no queue menu at all is fit on ``local_cores`` / ``local_gpus`` --
    the probed topology of the machine it is for, which IS the machine when
    nothing schedules for it; its one entry is then what the cell is over
    there, ``queue`` ``None``, and none where it holds.
    """
    from ..scheduler.fit import Request, fit
    if not routing:
        out = []
        for fam, cell in plan:
            over = _local_over(cell, fam, local_cores, local_gpus)
            said = ({"queue": None, "holds": False,
                     "line": "; ".join(o.message for o in over)},)
            out.append((fam, cell, said if over else ()))
        return out
    out = []
    for fam, (g, n, c) in plan:
        want_gpu = bool(fam and g)
        req = Request(ranks=n, cpus_per_task=c,
                      gpus=g if want_gpu else None)
        fits = [fit(row, req) for row in routing]
        out.append((fam, (g, n, c),
                    tuple({"queue": f.domain, "holds": f.holds,
                           "line": f.line} for f in fits)))
    return out


def _local_over(cell, fam, cores_total, gpus_per_node):
    """What a cell is over on the target's probed node -- the fit where
    there is no queue (`generator.md` § 4.3a).

    Empty means it fits, or that the probe measured nothing to compare
    against: an unmeasured topology is silence (R3).  Each names its
    numbers (R4).
    """
    from ..scheduler.fit import Over
    g, ranks, c = cell
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
    of the person's disagreeing, so the cell is left out; empty when it is
    within or nothing was asked."""
    if ask is None:
        return []
    from ..scheduler.fit import Over
    from ..scheduler.quantities import parse_gres_flag
    g, n, c = cell
    gpus = g if (fam and g) else 0
    out = []
    for limit, asked, allowed, unit in (
            ("ask_ranks", n, ask.mpi_np, "ranks"),
            ("ask_cpus", c, ask.cpus_per_task, "cores per rank"),
            ("ask_gpus", gpus,
             parse_gres_flag(ask.gres) if ask.gres not in (None, "")
             else None, "GPUs")):
        if allowed is not None and asked > int(allowed):
            out.append(Over(limit, "this prep", asked=asked,
                            allowed=int(allowed), unit=unit))
    return out


def _cell_label(g, n, c) -> str:
    """A cell's name: the machine part of its trials' names, built by the
    one builder (`resolve.point_token` -- `job-contracts.md` § 6.3: "and by
    nothing else"), ``G0`` for a CPU cell on every machine.  A sweep with
    value axes crosses each cell with them, so its trials' names go on past
    this one (``G0N48C1ELPA1stage``): the fit list names the cell, the
    directories its trials."""
    return point_token(dict(zip(MACHINE_AXES, (g, n, c))))


def _cell_resources(point, _environment) -> dict:
    """What a trial asks the machine for: N ranks of C cores, and its device
    count where it has one -- G=0 is a CPU trial, with NO gres, which is what
    lets `launch` put a CPU group on an allocation that holds no device
    (`generator.md` § 4.3a's split submission)."""
    asked = {"mpi_np": point["N"], "cpus_per_task": point["C"]}
    return {**asked, "gres": f"gpu:{point['G']}"} if point["G"] else asked


def _cell_shape(g, n, c) -> str:
    """What a cell ASKS FOR, in words -- ranks, cores each, and GPUs."""
    bit = f"{n} rank(s) x {c} core(s)"
    return bit + (f" + {g} GPU(s)" if g else "")


def prep_run_inputs(request, stage, allocation=None):
    """Everything ``prep task`` needs, assembled ONCE -- ``(allocation,
    run_settings, chosen)``.

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
    run (`job-request.md` § 3.2, plan § 5w K5): ``execution``, the
    calculation's card with the rung's over it -- what the person ASKED for.
    Its machine items are the launch shape, the rest are run settings over
    the catalogue's defaults.  The bench's one-point non-machine
    declarations pin the trials alone.

    A flag beats it and is already in ``allocation`` when it arrives.
    """

    # NEVER NONE (§ 6.0a's contract): the run card and the allocation are
    # folded UNDER this field by field (step 2), and there is no
    # field-by-field merge onto nothing.  A surface with no flags passes an
    # empty ask, not an absent one.
    allocation = allocation if allocation is not None else Resources()

    # 1 · THE CONDITION -- the run card, the launch-shape ladder's first rung,
    #     under a flag (`architecture.md` § 5.2) -- folded where the rest of
    #     the allocation is, once, at step 2 (:func:`fold_allocation`: a
    #     flag, then the run card, then the allocation).
    # THE SHAPE EVEN WHEN THE CARD IS EMPTY: a card whose `use_gpu` is on with
    # no count is a device run asking for none, refused, unstated, by the
    # prep entry of the job it resolves (`gpu.md` G5).
    chosen = declared_run_shape(request, stage)
    settings = card_settings(request, stage)
    # WHETHER A GPU RUN STATES HOW MANY, AND A CPU RUN NONE (`execution/
    # gpu.md` G5) is the prep entry's to ask, of the job it resolves from
    # this assembly, beside the launch values' refusal -- and the Task
    # setup card shows the entry's preview of it.

    # 1b · THE CALCULATION'S SCHEDULER ASK is folded where the stage is
    #      resolved, once (`prep._resolve_stage`, :func:`fold_allocation`:
    #      a flag, then the run card, then the allocation).

    # 2 · THERE IS NO SECOND RUNG.  A benchmark's verdict is a REPORT a
    #     person reads (printed by `summarize`), and what the
    #     run uses is what that person then wrote in `execution`
    #     (`architecture.md` § 5.2, user ruling: "the run parameter needs to
    #     be explicitly decided/written").  A measurement that reaches the
    #     launch on its own is a second arrival route, and every silent-value
    #     defect this lane has had was a second arrival route.

    # 3 · THE RUN SETTINGS -- the condition's alone (5.2's deck/speed
    #     ladder), each with its card.  The bench's one-point declarations
    #     pin its trials, never the run (`job-request.md` § 3.2's "and
    #     nowhere else").

    # 4 · NOTHING IS WORKED OUT FOR WHAT NOBODY STATED.  An unstated rank or
    #     thread count is refused by `launch_refusal`, which the entry asks
    #     with the whole assembly in hand (`architecture.md` § 5.2; user,
    #     2026-10-02: "explicit job config is the only way allowed").

    # `chosen` is returned for the PREVIEW to name; it is already folded in.
    return allocation, settings, chosen


def bench_refusal(engine: str, calculation: str):
    """Why a ``calculation`` on ``engine`` has no benchmark sweep -- ``None``
    when it has one.  One answer for every door that asks: the prep entry's
    gate, the bench's own assembly (:func:`bench_inputs`, asked first,
    before any machine is read), and the Task setup tab -- a description's
    and a hand-over's -- which offers the Measure step only where `prep
    bench` would take it (`job-system.md` § 5.0) -- from the two tables:
    the kind's ``no_benchmark``, and the engine's ``benchmark`` lane."""
    from ..engine_table import engine as _engine
    from ..kind_table import kind as _kind
    rec = _kind(calculation)
    if rec.no_benchmark:
        return (f"{rec.with_article} has no benchmark sweep: "
                f"{rec.no_benchmark}.")
    # THE SEAM REFUSAL, BY NAME (E-J1).  The bench lane speaks SIESTA's
    # vocabulary today: the measurement pins name SiestaConfig fields
    # (`max_scf_iter`, `restart`, ...), and the GPU question is read under
    # SIESTA's `use_gpu`, so another engine's sweep could silently
    # enumerate a CPU grid (`engines/stages.md` § 6.8's recorded hazard).
    if not _engine(str(engine)).benchmark:
        from ..engine_table import engines as _engines
        lanes = " and ".join(_engine(e).words for e in _engines()
                             if _engine(e).benchmark)
        return (f"Benchmarks measure {lanes} runs only; a {_engine(str(engine)).words} run is "
                f"sized from that engine's own scaling guidance.")
    return None


def fold_allocation(flags, declared, chosen=None):
    """``(Resources, where each value came from)`` -- the caller's
    allocation over the job request's, with each field it holds named by
    its source:
    ``flag``, ``run card`` or ``allocation`` -- what a run's placement
    records beside the queue it was placed on (`job-system.md` § 6.0).
    ONE fold: the sources are read off the same precedence that fills the
    values, never worked out a second way.

    FIELD BY FIELD: a flag is what the person asks for now, so a stated
    one wins and an unstated one leaves the file's answer standing --
    whole-object precedence would make ``--np 8`` erase a memory ask
    nobody mentioned.  ``declared`` is the job request's ``allocation``
    (`job-request.md` § 3.1); ``chosen``, the run card's launch shape
    (§ 3.2, `resolve.declared_run_shape`)."""
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
                said[name] = "allocation"
        # THE BINDING SWITCH, when said: `False` is the value that matters,
        # so it is not tested for truth (`execution/gpu.md` G9).
        if declared.gpu_binding is not None and out.gpu_binding is None:
            patch["gpu_binding"] = declared.gpu_binding
            said["gpu_binding"] = "allocation"
    # ALREADY IN `Resources`' OWN WORDS -- `to_resources` speaks them, so
    # there is no name map here and no second place for one to drift.
    for name, val in sorted((chosen or {}).items()):
        if name in known and getattr(out, name, None) in (None, ""):
            patch[name] = val
            said[name] = "run card"
    sources.update(said)
    return (_dc.replace(out, **patch) if patch else out), sources


#: The queue a caller names no queue of its own for: the fold's
#: (:func:`fold_allocation`) -- prep's, its flags over the job request.
_FOLDED = object()


def bench_inputs(calc, target, *, bench_override=None, report=None,
                 notes=None, environment=None,
                 flags=None, on_a_queue: bool = True,
                 queue=_FOLDED):
    """The benchmark specialisation's three inputs — `project-layout.md`

    ``target`` is REQUIRED: this function exists to read a SPECIFIC
    machine's hardware.  Pass ``None`` deliberately to mean "this machine".

    § 2.3.1a's split, stated as data: WHERE the values come from (the grid
    the job request states, as explicit points), the point → Resources
    translation, and the trial pins.  The framework —
    `prep`'s five steps — receives a longer list and never asks why
    (`generator.md` § 2).

    **The grid is RESOLVED here, at prep — and the job request STATES it**
    (`generator.md` § 4.3a; the user, 2026-11-04: *"Refuse, you state it"*).
    ``job-request.json``'s ``bench`` names the points to try — *"try 4, 8 and 16
    ranks"* is true on every cluster, so it is portable and belongs with the
    calculation; what those points MEAN on this machine is resolved here.
    Left unstated, the prep is refused, naming what to state.

    **Whether this is a GPU grid is the JOB REQUEST's answer, not this
    function's assumption** (2026-08-17).  `web/task-setup.md` § 6.2 —
    *"use GPU or not is set up only at the Job Prep UI"* — makes ``use_gpu``
    a value the person chose: the bench's points for it, else the
    calculation's run card, else the catalogue's default
    (`job-request.md` § 3.4); and the eigensolver is NOT the same question.
    The request answers, and the grid follows its answer.

    **A multi-point non-machine entry is a VALUE AXIS** (§ 4.3a): its points multiply the machine grid, each point carries
    its coordinates (the resolver's parameter lane applies them per
    trial), and ``use_gpu`` with two points is the grid-FAMILY axis --
    the grid enumerates once per flag, G=0 holding the CPU family's
    device coordinate.  See the section for the cap, naming, and
    split-submission halves of the rule.

    ``calc`` is the calculation as the verb opened it (A16) -- its
    description and template; ``environment`` the target's record when the
    caller has read it (the prep entry reads it once and hands it over, W55
    B1), else read here.

    ``flags`` are the prep's own, folded over the job request's
    ``allocation`` by the one fold (:func:`fold_allocation`) into the ASK:
    the benchmark's one queue, and what the person allows this prep -- its
    ranks, cores per rank and GPUs (`generator.md` § 4.1, the allocation
    the sweep must fit inside).  A cell past it is left out by name; every
    other cell is kept with its fit said under it (`scheduler.md` R6); the
    prep is refused when no cell is left (§ 4.3a).  The bench card names no
    flags; it passes the queue picked on the page as ``queue`` (``None`` for
    none picked), in place of the request's, as it passes the axes
    being edited (``bench_override``).

    ``on_a_queue`` is whether the benchmark goes to a queue -- false where
    prep writes it no header (`--no-sbatch`, or a machine with no
    scheduler): it then runs where it is launched, and its cells are the
    target's own topology's.
    """
    # WHAT A PERSON IS TOLD comes back to the caller, never printed here:
    # the command line prints it, the Task setup tab shows it
    # (`prep.prep_task`, `job-system.md` § 5.0).
    note = notes.append if notes is not None else (lambda _text: None)
    from ..runfiles import JOB_REQUEST_FILE
    from .machine import machine_record
    base, task, request = Path(calc.base), calc.task, calc.request
    # A DESCRIPTION WITH NO BENCH is refused before any machine is read: a
    # refusal about the target would otherwise answer a question the
    # description never asks.
    why = bench_refusal(str(task.engine), str(task.calculation))
    if why:
        raise PrepError(why)
    # THE ASK: the prep's flags over the job request's allocation, by the
    # one fold every caller's ask is (`fold_allocation`).
    ask = fold_allocation(flags, request.allocation)[0]
    # The cells are fit on the TARGET's queues, not on whatever box you happen
    # to be typing on (P2, 2026-08-17).
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
    menu = routing_of(environment)
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
        menu = []
        named = None
    topo = getattr(environment, "topology", None)
    # What a cell is fit on where no queue answers for it (no queues, or
    # `--no-sbatch`): the target's probed node.  Unknown when the record does
    # not say -- never a smaller figure (R3).
    node_gpus = getattr(topo, "gpus_per_node", None)
    cps = getattr(topo, "cores_per_socket", None)

    # THE DECLARED OVERRIDE LANE, split before anything is decided (user
    # rule, 2026-08-20): one-point non-machine entries are pins -- values
    # in force for every trial -- and the machine-answered entries are the
    # grid's axes.
    declared_pins, declared_axes, value_axes = _declared_execution_pins(
        request, bench_override)

    # WHETHER THE TRIALS ARE A GPU FAMILY (`job-request.md` § 3.4): the
    # bench's points for `use_gpu`, else the calculation's card, else the
    # catalogue's default -- what the calculation runs at, never a rung's
    # own card.  One name for both engines (`execution/gpu.md` G1).
    from .. import template as _T
    card = request.run_condition(None)
    on_gpu = (bool(card["use_gpu"]) if "use_gpu" in card
              else bool(_T.default_of(
                  _T.one(_T.catalogue(), "use_gpu", engine=str(task.engine)),
                  str(task.calculation))))
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

    # The axes come from the split above -- the value entries already left
    # as pins or value axes, so what remains is machine-answered by
    # construction, and one of the grid's own (`template.GRID_AXES`: the
    # checker the split asks refuses a row naming any other).
    declared = {k: list(v) for k, v in declared_axes.items()}
    sockets = getattr(topo, "sockets", None)
    #: What the target's probed node holds -- what a cell is fit on where
    #: no queue answers for it (`_cells_on_the_queue`: no queues, or
    #: `--no-sbatch`).  Unknown when the record does not say both -- never
    #: a smaller figure (R3).
    cores_total = (sockets * cps) if (sockets and cps) else None
    # A BENCHMARK STATES ITS GRID (the user, 2026-11-04: "Refuse, you state
    # it"): its ranks and cores per rank, on the GPU its device counts too.
    cat = _T.catalogue()
    unstated = [name for name in ("mpi_np", "omp_threads",
                                  *(("gpu_count",) if any(families) else ()))
                if not declared.get(name)]
    if unstated:
        said = [f'"{_T.one(cat, name, engine=str(task.engine)).label}" ({name})'
                for name in unstated]
        named_rows = " and ".join(filter(None, (", ".join(said[:-1]),
                                                said[-1])))
        raise PrepError(
            f"this benchmark does not state {named_rows}: a benchmark's grid "
            f"is yours to state, and molbuilder proposes none.  State the "
            f"points to try on Task setup's \"What to measure\" card, or in "
            f"{JOB_REQUEST_FILE}'s bench ("
            + ", ".join(f'"{name}": [N, ...]' for name in unstated)
            + ").")
    # ``mpi_np`` is the TOTAL rank count a point runs -- the same meaning it
    # has everywhere else -- and ``omp_threads`` the cores per rank.  A point
    # is measured as stated, never clamped: a clamped point would measure a
    # configuration nobody stated.
    ranks = [int(v) for v in declared["mpi_np"]]
    cores = [int(v) for v in declared["omp_threads"]]
    # THE DEVICE COUNTS (user, 2026-08-21: "explicit is what we need"): those
    # G values and no others, each its own shelf; every (mpi_np, G) pair is a
    # cell, its ranks spread over its devices as a run spreads them, an uneven
    # split said (`ask.gpu_share_notes`): a request is prepared, never vetoed
    # (`generator.md` § 4.3).
    gpu_counts = [int(v) for v in declared.get("gpu_count") or ()]
    if gpu_counts and not any(families):
        raise PrepError(
            f"{JOB_REQUEST_FILE} declares gpu_count, but this bench resolves "
            "to the CPU family (use_gpu is false and not an axis) -- the "
            "device counts would be silently ignored.  Declare use_gpu "
            "= [true] (or the [true, false] axis), or drop gpu_count.")

    def _family_cells(fam):
        """The machine cells of ONE family, as (G, N, C) -- G=0 is the CPU
        family's held coordinate, G>=1 the device count, N the ranks in
        all (`generator.md` § 4.3a, *The grid*)."""
        if fam:
            return sorted({(g, r, c)
                           for r in ranks for c in cores for g in gpu_counts})
        return [(0, r, c) for r in ranks for c in cores]

    # FIT EACH CELL ON THE QUEUE, SHOW THE OUTCOME (user, 2026-08-30: "we
    # don't have to fight with what language we use to indicate error, but
    # present the correct outcome"; 2026-10-17: the record is shown, never a
    # veto).  What this prep does not allow -- a cell past the person's own
    # ask -- is left out by name: two of their statements disagreeing.
    plan = [(fam, cell) for fam in families for cell in _family_cells(fam)]
    checked = [(fam, cell, lines, _past_the_ask(fam, cell, ask))
               for fam, cell, lines in _cells_on_the_queue(
                   plan, routing=menu, local_cores=cores_total,
                   local_gpus=node_gpus)]

    kept    = [(f, cell, lines) for f, cell, lines, why in checked if not why]
    crossed = [(f, cell, why) for f, cell, lines, why in checked if why]

    # THE SAME REPORT THE TERMINAL PRINTS, AS DATA.  The task-setup card
    # shows this list live beside the axes being edited.  One enumerator,
    # two renderings.
    if report is not None:
        report.extend(
            {"label": _cell_label(g, n, c),
             "shape": _cell_shape(g, n, c),
             "family": "gpu" if fam else "cpu",
             "ranks": n, "cores_each": c,
             "gpus": (g if (fam and g) else 0),
             # The wire carries the SENTENCES: the browser renders prose.
             "fit": list(lines)}
            for fam, (g, n, c), lines, _past in checked)

    # The menu these cells are checked against is the TARGET's -- `prep --target sol` on a workstation reads
    # the record for Sol, a cluster you are not standing on at all
    # (`preparing-for-another-machine.md`).  So the count is about the
    # QUEUES, and says so.
    note(f"  bench grid: {len(checked)} combination(s) enumerated, "
         f"{len(kept)} kept" + (f" on {named}" if named else ""))
    for fam, (g, n, c), lines in kept:
        note(f"    {_cell_label(g, n, c):<11} "
             f"{_cell_shape(g, n, c):<37}")
        # EACH CELL'S FIT, said under it (`generator.md` § 4.3a) -- the cell
        # is kept whatever it says (R4, R6).
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
        note(f"  left out ({len(crossed)}) -- this prep's ask does not "
             f"allow them:")
        for fam, (g, n, c), why in crossed:
            # `.message`, NOT the finding itself, whose repr is not a
            # sentence.  The numbers are fields so callers can read them; the
            # SENTENCE is what a person is shown.
            note(f"    {_cell_label(g, n, c):<11} "
                 f"{_cell_shape(g, n, c):<37}  "
                 + "; ".join(w.message for w in why))

    cells = []
    for fam in families:
        fcells = [cell for f, cell, _dom in kept if f == fam]
        if fam and fcells:
            # The GPU-sharing note (user, 2026-08-23): ALWAYS state the
            # ranks per GPU, an uneven split said, warn past MPS's client
            # ceiling, note past this stack's tuned point -- the one
            # sentence a run's prep says (`ask.gpu_share_notes`), so a cell
            # and a run cannot be said two ways.  Whether the cores fit a
            # node is each cell's fit above.
            from ..engine_table import engine as _engine
            from .ask import gpu_share_notes
            note("  GPU sharing in this family:")
            _one = _engine(str(task.engine)).one_process
            for g, n in sorted({(g, n) for g, n, c in fcells}):
                for line in gpu_share_notes(g, n, one_process=_one):
                    note("  " + line)
        if fam and not fcells:
            # every GPU cell was left out above, listed by name, so this
            # names the consequence.
            if mixed:
                note(
                    "  NOTE: no GPU cell is left -- this sweep measures "
                    "only the CPU family.")
            else:
                raise PrepError(
                    "no GPU cell is left -- see the cells left out.  "
                    f"Raise this prep's ask, or change mpi_np / gpu_count in "
                    f"{JOB_REQUEST_FILE}.")
        cells.extend((fam, cell) for cell in fcells)
    # THE VALUE-AXIS CARTESIAN (§ 4.3a): every machine cell is crossed
    # with the remaining declared value axes, in declaration order, and
    # each point CARRIES its coordinates -- the resolver's ordinary
    # parameter lane applies them to that trial's config (provenance
    # "sweep"), and the coordinate rides the trial's name and, as data,
    # `job-set.json`'s per-trial ``point``.
    if not cells:
        raise PrepError(
            "no bench cell is left -- see the cells left out.  Raise "
            f"this prep's ask, or change the bench in {JOB_REQUEST_FILE}.")
    combos = [{}]
    for name, vals in value_axes.items():
        combos = [{**c, name: v} for c in combos for v in vals]

    points = []
    for fam, cell in cells:
        # EVERY TRIAL CARRIES G -- ``G0`` for a CPU trial, on every
        # machine.
        coord = dict(zip(MACHINE_AXES, cell))
        if mixed:
            coord["use_gpu"] = fam
        points.extend({**coord, **vc} for vc in combos)

    translation = MachineTranslation(axes=MACHINE_AXES,
                                     to_resources=_cell_resources)
    # The trial pins -- schema values resolved like any other (`template.md`
    # § 8.1: rebuild and render, never splice): capped SCF, single point,
    # forced cold -- and ``scf_must_converge: False``, the switch that makes
    # the cap CLEAN (`choose_winner` ranks only completed points).
    #
    # What is pinned here is what makes a trial a MEASUREMENT rather than a
    # run.  What the run IS -- the GPU, the eigensolver, the block size --
    # is the calculation's run card's, and pinning it here would measure a
    # configuration nobody asked to run.
    # The declared values ride beside the measurement pins, and the two
    # never name one item: a pin says what is computed, no run setting, so
    # no benchmark row names one (`template.why_not_in`).  A trial runs
    # once because every run is one start (`project-layout.md` § 1.6.1).
    pins = {**declared_pins, **_T.TRIAL_PINS}
    return points, pins, translation
