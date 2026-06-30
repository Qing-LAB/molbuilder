"""SIESTA stage-ladder → :class:`JobSet` producer
(docs/protocols/staged-execution.md § 2.1).

This is the ONLY place SIESTA stage knowledge meets the engine-agnostic
``jobset`` framework.  It turns ``cfg.stages`` (the validated
``SiestaStageSpec`` ladder) into a ``ladder`` JobSet:

  * one ``Job`` per enabled stage, its input ``<label>_<name>.fdf``;
  * a dependency chain whose kind comes from each stage's
    ``on_nonconvergence`` (§ 5): ``proceed`` -> ``afterany`` (run the next
    stage regardless), else (``halt`` / ``continue``) -> ``afterok`` (next
    runs only if this stage ultimately converged; ``continue``'s terminal
    failure mode is halt);
  * carry-forward restart files (§ 8 D1): ``.XV`` always; ``.DM`` when
    ``cfg.use_save_dm``; ``.CG`` only when consecutive stages share
    ``relax_type`` (optimizer history is algorithm-specific).

Per-stage SCHEDULER resources are kept OUT of ``SiestaStageSpec`` (that's
the science-knob widget); they ride on ``Job.resources`` and are supplied
via the ``resources_for`` seam — default: every stage inherits the
job-level config (mpi_np / omp), everything else resolved at submit time.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..jobset.model import Carry, Job, JobSet, Resources


def _dep_kind(prev_policy: str) -> str:
    """SLURM dependency kind for the edge OUT of a stage with the given
    ``on_nonconvergence`` policy (§ 5).  ``proceed`` lets the next stage run
    regardless (afterany); ``halt`` and ``continue`` both end in
    success-or-failure, so the next stage runs only on success (afterok)."""
    return "afterany" if prev_policy == "proceed" else "afterok"


def stages_to_jobset(
    cfg,
    *,
    shared: Optional[list] = None,
    resources_for: Optional[Callable[[str], Resources]] = None,
) -> JobSet:
    """Build the ladder :class:`JobSet` from ``cfg.stages``.

    ``shared`` is the static package symlinked into every stage dir
    (pseudopotentials, ``mb_monitor.py``, …) — the caller supplies it since
    only the bundle layer knows the concrete filenames.  ``resources_for``
    optionally returns a per-stage :class:`Resources` override (keyed by
    stage name); absent → each stage inherits the job-level config.

    Raises ``ValueError`` if the stage ladder is structurally invalid
    (delegates to ``validate_siesta_stages`` — the same gate the Build tab
    uses), so a producer can't emit a JobSet the engines would choke on.
    """
    from ..config.siesta import validate_siesta_stages

    label = cfg.system_label
    stages = list(getattr(cfg, "stages", []) or [])
    errs = validate_siesta_stages(stages)
    if errs:
        raise ValueError(
            "cannot build a stage JobSet from an invalid ladder:\n  - "
            + "\n  - ".join(errs))

    enabled = [s for s in stages if s.enabled]
    use_dm = bool(getattr(cfg, "use_save_dm", True))

    def _res(name: str) -> Resources:
        if resources_for is not None:
            r = resources_for(name)
            if r is not None:
                return r
        # Default: inherit the job-level parallel knobs; scheduler
        # resources (domain/time/exclusive/mem/gres) resolve at submit.
        # This is the config->exchange translation boundary: SiestaConfig's
        # ``omp_threads`` becomes the exchange field ``cpus_per_task``.
        return Resources(mpi_np=getattr(cfg, "mpi_np", None),
                         cpus_per_task=getattr(cfg, "omp_threads", None))

    jobs = []
    prev = None
    for s in enabled:
        carry = []
        if prev is not None:
            carry.append(Carry(f"{label}.XV", prev.name))
            if use_dm:
                carry.append(Carry(f"{label}.DM", prev.name))
            if prev.relax_type == s.relax_type:   # same optimizer only
                carry.append(Carry(f"{label}.CG", prev.name))
        jobs.append(Job(
            name=s.name,
            script=f"{label}_{s.name}.fdf",
            resources=_res(s.name),
            depends_on=(prev.name if prev is not None else None),
            dep_kind=(_dep_kind(prev.on_nonconvergence)
                      if prev is not None else "afterok"),
            carry=carry,
        ))
        prev = s

    return JobSet(
        name=label,
        engine="siesta",
        kind="ladder",
        shared=list(shared or []),
        jobs=jobs,
    )


__all__ = ["stages_to_jobset"]
