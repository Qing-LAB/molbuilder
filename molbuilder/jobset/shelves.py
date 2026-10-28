"""A BENCHMARK'S SHELVES -- its trials as the jobs a queue is sent
(`generator.md` § 4.3a, `job-system.md` § 6.0): by the side each trial runs
on, then by its exact resource ask, each shelf one allocation walking its
trials one after another.

Answered once, here, for the two steps that need it: prep writes each
shelf's one header from its request, placed on the benchmark's one queue
(`prep.walk_header_of`); launch walks each shelf's trials still to run and
sends that header (`submit._plan_shelves`).

* the side a trial runs on (:func:`sides_of`) -- its GPU request;
* the shelves, each side's widest first (:func:`shelves_of`), named by
  their side and machine cell (`runfiles.GroupNames.of_trials`);
* a shelf's one request (:func:`shelf_request`) -- its trials' shared ask.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from ..resolve import MACHINE_AXES
from ..runfiles import GroupNames

if TYPE_CHECKING:
    from .model import Resources


class ShelfError(ValueError):
    """A sweep's trials cannot be shelved as prep wrote them -- the message
    says which trial and why."""


def _uses_gpu(job) -> bool:
    """Whether ``job`` runs on the GPU -- its request, by the one door every
    reader asks (`model.gpu_request`)."""
    from .model import GpuRequestError, gpu_request
    try:
        return gpu_request(job.resources).uses
    except GpuRequestError as exc:
        raise ShelfError(f"trial {job.name!r}: {exc}") from None


def sides_of(jobs) -> Dict[str, list]:
    """A sweep's trials by the side they run on -- ``{"cpu": [...], "gpu":
    [...]}`` -- each trial's GPU request, read off the trial itself; the
    sweep's order kept within each."""
    sides: Dict[str, list] = {"cpu": [], "gpu": []}
    for j in jobs:
        sides["gpu" if _uses_gpu(j) else "cpu"].append(j)
    return sides


def _ask(job) -> tuple:
    """The exact resource ask that makes a shelf: trials on one shelf fit
    ONE allocation with nothing idle, so the key is everything its request
    would widen over -- ranks, cores per rank, GPUs."""
    from .model import gpu_request
    r = job.resources
    return (r.mpi_np, r.cpus_per_task, gpu_request(r).count or 0)


def _width(key) -> tuple:
    """Widest first across shelves: the cores, then the devices."""
    n, c, g = key
    return (n * max(c, 1), g)


def _cell(jobs) -> str:
    """A shelf's machine cell -- ``G2N48C1`` -- THE SPELLING ITS TRIALS
    CARRY, read off a trial's own point: every trial on a shelf shares one
    resource ask, so any member answers, and the value axes that differ
    between them are dropped -- the shelf is the machine cell, not the
    point."""
    from ..resolve import point_token
    pt = jobs[0].point or {}
    if "K" in pt and "N" not in pt:
        # A TRIAL NAMED BY ITS RANKS PER GPU, as a benchmark was prepared
        # before 2.46.6 (`job-contracts.md` § 6.3): its trials are named by
        # their ranks in all now, so it is prepared again.
        raise ShelfError(
            f"trial {jobs[0].name!r} is named by its ranks per GPU (K), as "
            f"molbuilder before 2.46.6 prepared a benchmark; a trial is "
            f"named by its ranks in all (N) now -- go back to the state "
            f"saved before the benchmark's prep, and prepare it again "
            f"(`molbuilder jobset prep bench <stage>`)")
    if not all(a in pt for a in MACHINE_AXES):
        raise ShelfError(
            f"trial {jobs[0].name!r} carries no machine point "
            f"({', '.join(MACHINE_AXES)}) -- not a trial prep wrote, so its "
            f"shelf cannot be named")
    return point_token({a: pt[a] for a in MACHINE_AXES})


def shelves_of(jobs) -> List[Tuple[str, GroupNames, list]]:
    """The shelves of a sweep's trials -- ``(side, names, trials)``: the CPU
    side's, then the GPU side's, each side's widest first, so the expensive
    measurements land first and an early stop still summarizes to a
    verdict.  A shelf's name is its side and its cell, never what else the
    sweep holds, so it keeps its name across launches."""
    out: List[Tuple[str, GroupNames, list]] = []
    for side, trials in sides_of(jobs).items():
        by: Dict[tuple, list] = {}
        for j in trials:
            by.setdefault(_ask(j), []).append(j)
        for key in sorted(by, key=_width, reverse=True):
            out.append((side, GroupNames.of_trials(side, _cell(by[key])),
                        by[key]))
    return out


def shelf_request(trials) -> "Resources":
    """The one allocation a shelf asks for -- its trials' shared request:
    their ranks, cores per rank and GPUs (one ask by construction,
    :func:`shelves_of`), and the wall, the memory, the GPU binding and the
    GPU use prep stated on every trial -- the benchmark's own, one value
    over a sweep.  Trials disagreeing on any of them were not written by
    one prep, and are refused rather than sent an allocation that fits only
    some."""
    from .model import Resources
    asks = {_ask(j) for j in trials}
    shared = {(j.resources.mem, j.resources.time, j.resources.gpu_binding,
               _uses_gpu(j), j.resources.gres or "") for j in trials}
    if len(asks) > 1 or len(shared) > 1:
        raise ShelfError(
            f"the shelf's trials ({', '.join(j.name for j in trials)}) do "
            f"not share one request -- prep states one allocation over a "
            f"sweep (generator.md § 4.3a); prepare the benchmark again.")
    r = trials[0].resources
    return Resources(mpi_np=r.mpi_np, cpus_per_task=r.cpus_per_task,
                     gres=r.gres, use_gpu=_uses_gpu(trials[0]),
                     exclusive=any(j.resources.exclusive for j in trials),
                     gpu_binding=r.gpu_binding, mem=r.mem, time=r.time,
                     domain=r.domain)


__all__ = ["MACHINE_AXES", "ShelfError", "shelf_request", "shelves_of",
           "sides_of"]
