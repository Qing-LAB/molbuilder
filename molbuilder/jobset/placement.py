"""A job's placement -- what it asks a queue for, whether every value of it
is stated, and how it fits the queue it names (`execution/architecture.md`
§ 5.2, `job-system.md` § 6.0, `scheduler.md` R6).
Floor 3 (`execution/architecture.md` § 2.1).
"""
from __future__ import annotations

from typing import Optional

from .model import AS_RESOURCE


#: The run card's launch-shape items: the catalogue's machine items that size
#: the processes -- not `gpu_count`, which G5 refuses on its own and only for a
#: device run, and not `max_memory_mb`, a cap whose absence asks for nothing.
_SHAPE_ITEMS = ("mpi_np", "omp_threads", "threads")

#: What a person calls each launch value, and the flag that states it -- the
#: words of the one refusal below and of the line that says where each value
#: came from.
_LAUNCH_WORDS = {
    "domain":        ("queue", "--domain QUEUE"),
    "time":          ("wall", "--time 2-00:00:00"),
    "mem":           ("memory", "--mem 64G"),
    "mpi_np":        ("ranks", "--np N"),
    "cpus_per_task": ("cores per rank", "--cpus-per-task N"),
    "gres":          ("GPUs", "--gpus N"),
    "max_memory_mb": ("memory cap", "--max-memory-mb MB"),
    "gpu_binding":   ("GPU binding", "allocation.gpu_binding"),
}


def launch_refusal(allocation, *, engine: str, header: bool, shape: bool,
                   stage=None, queues=(), asked=None, base=None,
                   target=None):
    """**Why this launch cannot be written** -- ``None`` when every value it
    needs is stated (`execution/architecture.md` § 5.2; user, 2026-10-02:
    *"explicit job config is the only way allowed"*).

    ONE ANSWER, asked by `prep.plan_stage` with the whole assembly in hand,
    before anything is written -- for a run and for a benchmark alike; launch
    sends what prep wrote and asks nothing again (`scheduler.md` R9).  The
    renderers take what is stated and ask nothing again:

    * ``shape`` -- a RUN's processes: the engine's own launch-shape items
      (the catalogue's, so PySCF is asked its threads and never a rank
      count).  A benchmark's shape is its grid point, so it passes ``False``.
    * ``header`` -- a ``.sbatch`` is written for a queue (the target has a
      scheduler, and ``--no-sbatch`` was not given): the queue, the wall and
      the memory.

    Nothing here fills a value in -- not the target's width, not a rank per
    GPU, not a thread count of one, not a queue's ceiling, not the
    scheduler's default memory.  Each was a value nobody stated for that run,
    and a run is hours before anyone learns which one it got.  ``queues`` --
    the target's own, as its record holds them -- is the record's fact: a
    run naming none is shown them, their machines and the fit of ``asked``
    on each -- the work's request (`ask.queue_table`, `submission.md` S5),
    so one can be named; ``base``
    and ``target`` say
    which record that was, for the refusal that finds it lists none.
    """
    from ..template import catalogue, select
    missing = []
    if shape:
        items = [item for item in select(catalogue(), engine=engine)
                 if item.name in _SHAPE_ITEMS]
        for item in items:
            field = AS_RESOURCE.get(item.name, item.name)
            if getattr(allocation, field, None) in (None, "", 0):
                missing.append((field, item.name))
        # A RANK COUNT FOR AN ENGINE THAT RUNS ONE PROCESS names nothing it
        # runs: its header asks one task, and launch would have sent the
        # count (`submit._sbatch_request`).
        if (getattr(allocation, "mpi_np", None) not in (None, "", 0)
                and "mpi_np" not in {i.name for i in items}):
            return (f"{'stage ' + repr(stage) + ' ' if stage else ''}states "
                    f"a rank count (--np) for {engine}, which runs one "
                    f"process -- a rank count names nothing it runs; the "
                    f"cores it runs on are its threads (the run card's "
                    f"`threads`, or --cpus-per-task).")
    if header:
        for field in ("domain", "time", "mem"):
            if getattr(allocation, field, None) in (None, ""):
                missing.append((field, field))
        named = getattr(allocation, "domain", None)
        listed = [q.name for q in queues]
        who = f"{'stage ' + repr(stage) + ' ' if stage else ''}"
        if not listed:
            # A SCHEDULER WHOSE RECORD LISTS NO QUEUES has none a job could
            # name: probed off its scheduler, or not probed there -- renewed
            # by the record that answered's own steps (`scheduler.md` R6).
            return no_queues_said(base, target, who=who, named=named)
        if named and named not in listed:
            # A QUEUE THE RECORD DOES NOT LIST is not a queue this job can be
            # sent to.
            return (f"{who}names the queue {named!r}, which the target's "
                    f"record does not list -- it lists: "
                    f"{', '.join(listed)}.")
    if not missing:
        return None
    where = []
    for field, key in missing:
        words, flag = _LAUNCH_WORDS[field]
        # WHERE A PERSON STATES IT, on either road: the Task setup card that
        # writes it, the task.json key that card writes, and the flag --
        # the refusal reaches the page and the terminal alike.
        if field in ("domain", "time"):
            card = (f'Task setup\'s "Where it runs" card ("allocation": '
                    f'{{"{key}": ...}} in task.json)'
                    + (f' or this run\'s run card ("execution": '
                       f'{{"{key}": ...}})' if shape else ""))
        elif field == "mem":
            card = ('Task setup\'s "Where it runs" card ("allocation": '
                    '{"mem": ...} in task.json)')
        else:
            card = (f'the run card on Task setup ("execution": '
                    f'{{"{key}": N}} in task.json)')
        line = f"  {words:<15} {card}, or {flag}"
        if field == "domain":
            line += "\n" + _the_queues(asked, queues, engine)
        where.append(line)
    what = ", ".join(f"{_LAUNCH_WORDS[f][0]} ({k})" for f, k in missing)
    return (f"{'stage ' + repr(stage) + ' ' if stage else ''}states no "
            f"{what} -- and nothing fills one in "
            f"(docs/execution/architecture.md § 5.2).  State each:\n"
            + "\n".join(where))


def no_queues_said(base, target, *, who: str = "",
                   named: Optional[str] = None) -> str:
    """The refusal of work sent to a scheduler whose record lists no
    queues -- none a job could name -- with the record that answered and
    its renewal (`scheduler.md` R6): prep's, and the benchmark card's."""
    from ..scheduler.record import record_and_renewal
    which, renew = record_and_renewal(base, target)
    return (f"{who}{'names the queue ' + repr(named) + ', and ' if named else 'goes to a scheduler, and '}"
            f"the record it reads ({which}) lists no queues at all -- none a "
            f"job could name.  If that machine has queues, its record is out "
            f"of date: {renew}.")


def _the_queues(asked, rows, engine: str) -> str:
    """The target's queues, their machines, and the work's fit on each --
    what of its wall, memory, cores and GPUs it states (`ask.queue_table`)."""
    from .ask import queue_table
    return queue_table(rows, request_of(asked,
                                        one_process=one_process(engine)))


def request_of(resources, *, one_process: bool):
    """THE REQUEST a run makes of its queue (`scheduler.fit.Request`), as
    its header asks it (`runwrap._render_sbatch_for`): its processes -- a
    SIESTA run's ranks; ``one_process`` for an engine that runs one (PySCF)
    -- the cores each runs on, its GPUs (the one door,
    `model.gpu_request`), its memory and its wall.  An unstated value is
    ``None``, which never bars (`scheduler.md` R7); prep has refused an
    unstated one already (:func:`launch_refusal`)."""
    from ..scheduler import Request, parse_mem_gb
    from ..scheduler.quantities import parse_walltime
    from .model import gpu_request
    r = resources
    gpus = gpu_request(r)
    return Request(ranks=1 if one_process else r.mpi_np,
                   cpus_per_task=r.cpus_per_task,
                   gpus=gpus.count if gpus.uses else None,
                   mem_gb=parse_mem_gb(r.mem) if r.mem else None,
                   walltime_s=(parse_walltime(str(r.time)) if r.time
                               else None))


def one_process(engine) -> bool:
    """Whether ``engine`` runs ONE process -- PySCF, a Python script with
    OpenMP threads -- so its request asks one task.  The one rule, asked by
    an engine's name or by its deck (:func:`deck_runs_one_process`)."""
    return str(engine) == "pyscf"


def deck_runs_one_process(suffix: str) -> bool:
    """:func:`one_process` for the engine whose deck ``suffix`` is -- the
    catalogue's answer (`runfiles.engine_of_role`); a suffix no engine's
    deck has runs more than one (`runwrap`)."""
    from ..runfiles import engine_of_role
    return one_process(engine_of_role(str(suffix).lower()))


#: The values a run's placement records the source of -- its queue, wall
#: and memory, its ranks, cores per rank and GPUs (`job-system.md` § 6.0).
_PLACED = ("domain", "time", "mem", "mpi_np", "cpus_per_task", "gres",
           "max_memory_mb", "gpu_binding")

def placed(resources, environment, *, one_process: bool, what: str,
           sources=None):
    """``(placement, fit, refusal)`` -- the queue a job names, bound on the
    target's record with its whole request (:func:`request_of`) by the one
    binding (`scheduler.place`): what its job records of it -- the queue's
    name, partition and qos, and where each value came from (``sources``,
    `prep_inputs.fold_allocation`; a walk's, each field's every source
    among its members, as a tuple) -- and its FIT, the line prep says of it:
    the nodes that hold it and each limit it is over (`scheduler.md` R4,
    R6).  The fit is said, never recorded: it decides nothing.

    Never refused for the fit: the scheduler decides.  ``(None, "", why)``
    only when a statement is missing -- ``what`` names no queue the record
    lists (`job-system.md` § 5.0, checkpoint 4; § 6.0).  Launch sends what
    this placed and asks nothing again (`scheduler.md` R9)."""
    from ..runtime_config import routing_of
    from ..scheduler.place import Unplaceable, place
    named = getattr(resources, "domain", None)
    try:
        bound = place(routing_of(environment),
                      request_of(resources, one_process=one_process),
                      named=named)
    except Unplaceable as exc:
        return None, "", f"{what}: {exc}"
    if bound is None:
        return None, "", None
    said = sources or {}
    return ({"domain": named, "partition": bound.partition, "qos": bound.qos,
             "from": {f: said[f] for f in _PLACED
                      if getattr(resources, f, None) not in (None, "")
                      and f in said}}, bound.fit.line, None)


def _sources_said(sources) -> str:
    """Where each launch value came from, by source -- ``wall, memory from
    the description; ranks, cores per rank from the run card``."""
    by_source: dict = {}
    sources = sources or {}
    for field in _LAUNCH_WORDS:            # in the one order of the words
        if field in sources:
            by_source.setdefault(sources[field], []).append(
                _LAUNCH_WORDS[field][0])
    return "; ".join(f"{', '.join(fields)} from the {source}"
                     if source != "flag" else f"{', '.join(fields)} from a flag"
                     for source, fields in by_source.items())


def placement_line(placement) -> str:
    """A run's placement, as both doors say it -- the queue it was placed
    on and where each value came from (`job-system.md` § 6.0): ``placed on
    short (cpu/public) -- queue from the run card; wall, memory from the
    description; ranks, cores per rank from the run card``."""
    if not placement:
        return ""
    said = _sources_said(placement.get("from"))
    return (f"placed on {placement.get('domain')} "
            f"({placement.get('partition')}/{placement.get('qos') or '-'})"
            + (f" -- {said}" if said else ""))


def stated_line(sources) -> str:
    """A run prepared with no header -- a machine with no scheduler, or
    ``--no-sbatch`` -- said as a placement is: where each value it states
    came from (`prep_inputs.fold_allocation`)."""
    said = _sources_said(sources)
    return f"no header -- {said}" if said else ""


__all__ = ["placed", "launch_refusal", "no_queues_said", "one_process",
           "placement_line", "request_of", "stated_line"]
