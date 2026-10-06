"""A job's placement -- what it asks a queue for, whether every value of it
is stated, and whether the queue it names can take it
(`execution/architecture.md` § 5.2, `job-system.md` § 6.0).
Floor 3 (`execution/architecture.md` § 2.1).

A module of its own since 2026-10-03 (W55 B7): the check that every launch
value is stated lived in the prep assembly, and launch's one request
imported it from there -- so launch, floor 5, reached the conductor.
"""
from __future__ import annotations


from .model import AS_RESOURCE


#: The run card's launch-shape items: the catalogue's machine items that size
#: the processes -- not `gpu_count`, which G5 refuses on its own and only for a
#: device run, and not `max_memory_mb`, a cap whose absence asks for nothing.
_SHAPE_ITEMS = ("mpi_np", "omp_threads", "threads")

#: What a person calls each launch value, and the flag that states it -- the
#: words of the one refusal below.
_LAUNCH_WORDS = {
    "domain":        ("queue", "--domain QUEUE"),
    "time":          ("wall", "--time 2-00:00:00"),
    "mem":           ("memory", "--mem 64G"),
    "mpi_np":        ("ranks", "--np N"),
    "cpus_per_task": ("cores per rank", "--cpus-per-task N"),
}


def launch_refusal(allocation, *, engine: str, header: bool, shape: bool,
                   stage=None, queues=(), base=None, target=None,
                   at_launch: bool = False):
    """**Why this launch cannot be written** -- ``None`` when every value it
    needs is stated (`execution/architecture.md` § 5.2; user, 2026-10-02:
    *"explicit job config is the only way allowed"*).

    ONE ANSWER, asked at each moment a launch is written: by `prep_stage`
    with the whole assembly in hand, before anything is written -- for a run
    and for a benchmark alike -- and by launch's one request
    (`submit._sbatch_request`) of what it sends, where a launch flag may
    have stated a value.  The renderers below them take what is stated and
    ask nothing again:

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
    the target's own, by name -- is the record's fact, shown so the person
    can choose one; ``base`` and ``target`` say which record that was, for
    the refusal that finds it lists none.
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
        # count (`submit._sbatch_request`).  `--np` on a PySCF run passed
        # silently until 2026-10-05.
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
        if named and named not in queues:
            # A QUEUE THE RECORD DOES NOT LIST is not a queue this job can be
            # sent to -- the header fell back to the menu's first row,
            # silently, until 2026-10-02.  A record that lists none at all
            # was probed off its scheduler, or not probed there.
            said = f"{'stage ' + repr(stage) + ' ' if stage else ''}names "
            if queues:
                return (f"{said}the queue {named!r}, which the target's "
                        f"record does not list -- it lists: "
                        f"{', '.join(queues)}.")
            # NONE AT ALL: probed off its scheduler, or not probed there --
            # renewed by the record that answered's own steps (W54 R5: this
            # printed the bare probe command whatever the target, and
            # nothing about a snapshot).
            from ..scheduler.record import record_and_renewal
            which, renew = record_and_renewal(base, target)
            return (f"{said}the queue {named!r}, and the record it reads "
                    f"({which}) lists no queues at all.  If that machine has "
                    f"queues, its record is out of date: {renew}.")
    if not missing:
        return None
    where = []
    for field, key in missing:
        words, flag = _LAUNCH_WORDS[field]
        if at_launch:
            # LAUNCH READS NO DESCRIPTION: what it can still be told is a
            # flag on this launch (it named task.json until 2026-10-06,
            # which launch never reads -- the unit 11 review).
            where.append(f"  {words:<15} {flag} on this launch")
            continue
        if field in ("domain", "time"):
            card = (f'"allocation": {{"{key}": ...}} in task.json'
                    + (f' or "execution": {{"{key}": ...}} (this run)'
                       if shape else ""))
        elif field == "mem":
            card = '"allocation": {"mem": ...} in task.json'
        else:
            card = f'"execution": {{"{key}": N}} in task.json'
        line = f"  {words:<15} {card}, or {flag}"
        if field == "domain" and queues:
            line += (f"\n  {'':<15} (the target's record lists: "
                     f"{', '.join(queues)})")
        where.append(line)
    what = ", ".join(f"{_LAUNCH_WORDS[f][0]} ({k})" for f, k in missing)
    return (f"{'stage ' + repr(stage) + ' ' if stage else ''}states no "
            f"{what} -- and nothing fills one in "
            f"(docs/execution/architecture.md § 5.2).  State each:\n"
            + "\n".join(where)
            + ("\n  (the description's values are read at prep: to state "
               "one there, go back to the state saved before the prep and "
               "prep anew)" if at_launch else ""))


def request_of(resources, *, one_process: bool):
    """THE REQUEST a run makes of its queue (`scheduler.admit.Request`), as
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
    """Whether ``engine`` runs ONE process -- PySCF, whose deck is a Python
    script -- so its request asks one task (`runwrap._render_sbatch_for`'s
    rule, by the same fact: the deck's suffix)."""
    from .engines import engine_seam
    return engine_seam(str(engine)).suffix == ".py"


#: The values a run's placement records the source of -- its queue, wall
#: and memory, its ranks, cores per rank and GPUs (`job-system.md` § 6.0).
_PLACED = ("domain", "time", "mem", "mpi_np", "cpus_per_task", "gres")

#: The values each limit a queue refuses is asked against
#: (`scheduler.admit.Refusal.limit`).
_REFUSED = {"walltime": ("time",), "mem": ("mem",),
            "cores": ("mpi_np", "cpus_per_task"),
            "cpus_per_job": ("mpi_np", "cpus_per_task"),
            "gpus": ("gres",)}


def _stated_at(field: str, source: str) -> str:
    """Where a launch value was stated, as the person changes it: the
    flag on this prep, the run card, or the description."""
    words, flag = {**_LAUNCH_WORDS, "gres": ("GPUs", "--gpus N")}[field]
    if source == "flag":
        return f"{words}: {flag.split()[0]} on this prep"
    if source == "run card":
        return f"{words}: the run card (`execution` in task.json)"
    return f"{words}: the description (`allocation` in task.json)"


def admitted(resources, environment, *, one_process: bool, stage=None,
             sources=None):
    """``(placement, refusal)`` -- the queue a run names, admitted on the
    target's record with its whole request (:func:`request_of`) by the
    binding launch asks too (`scheduler.place`), and what its job records
    of it: the queue's name, partition and qos, and where each value came
    from (``sources``, `prep._fold_allocation`) -- or ``(None, why)``: a GPU
    run naming a queue with no GPUs, more cores or memory than its nodes
    hold, a wall longer than it allows is refused at prep, naming what was
    asked and what the queue offers (`job-system.md` § 5.0, checkpoint 4;
    § 6.0).  Launch admits what it sends again, against the machine as it
    stands then (`scheduler.md` R9)."""
    from ..runtime_config import routing_of
    from ..scheduler.place import Unplaceable, place
    from .model import gpu_request
    named = getattr(resources, "domain", None)
    try:
        bound = place(routing_of(environment),
                      request_of(resources, one_process=one_process),
                      prefer_gpu=gpu_request(resources).uses, named=named)
    except Unplaceable as exc:
        # WHERE EACH REFUSED VALUE WAS STATED, read off the fold that
        # filled it (`job-system.md` § 5.0, checkpoint 4: "the file and key
        # where each value is stated") -- this named fixed places until
        # 2026-10-05, the unit 11 review.
        said, at = sources or {}, []
        for r in exc.reasons:
            for f in _REFUSED.get(r.limit, ()):
                if (getattr(resources, f, None) not in (None, "", 0)
                        and f in said):
                    line = _stated_at(f, said[f])
                    if line not in at:
                        at.append(line)
        other = "name another of the record's queues"
        if any(r.limit == "gpus" for r in exc.reasons):
            from ..scheduler import domain_serves_gpu
            able = [d.name for d in (environment.domains or ())
                    if domain_serves_gpu(d)]
            other = (f"name a queue with GPUs: {', '.join(able)}" if able
                     else "the target's record lists no queue with GPUs")
        return None, (f"{'stage ' + repr(stage) + ' ' if stage else ''}does "
                      f"not fit the queue {named!r} on the target's record:\n    "
                      + "\n    ".join(r.message for r in exc.reasons)
                      + ("\n  Ask for less where it is stated:\n    "
                         + "\n    ".join(at) if at else "")
                      + f"\n  -- or {other}.")
    if bound is None:
        return None, None
    said = sources or {}
    return ({"domain": named, "partition": bound.partition, "qos": bound.qos,
             "from": {f: said[f] for f in _PLACED
                      if getattr(resources, f, None) not in (None, "")
                      and f in said}}, None)


def placement_line(placement) -> str:
    """A run's placement, as both doors say it -- the queue it was admitted
    on and where each value came from (`job-system.md` § 6.0): ``placed on
    short (cpu/public) -- queue from the run card; wall, memory from the
    description; ranks, cores per rank from the run card``."""
    if not placement:
        return ""
    words = dict(_LAUNCH_WORDS, gres=("GPUs", "--gpus N"))
    by_source: dict = {}
    for field, source in (placement.get("from") or {}).items():
        by_source.setdefault(source, []).append(words.get(field, (field,))[0])
    said = "; ".join(f"{', '.join(fields)} from the {source}"
                     if source != "flag" else f"{', '.join(fields)} from a flag"
                     for source, fields in by_source.items())
    return (f"placed on {placement.get('domain')} "
            f"({placement.get('partition')}/{placement.get('qos')})"
            + (f" -- {said}" if said else ""))


__all__ = ["admitted", "launch_refusal", "one_process",
           "placement_line", "request_of"]
