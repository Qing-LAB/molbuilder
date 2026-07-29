"""SIESTA stage-ladder → :class:`JobSet` producer
(docs/execution/job-system.md).

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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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


# --------------------------------------------------------------------- #
#  Pure bundle producer (staged-execution.md § 15.3 Promotion A)         #
# --------------------------------------------------------------------- #


@dataclass
class StageBundle:
    """The contents of a multi-stage SIESTA bundle, as DATA — no filesystem.

    Produced by :func:`build_siesta_stage_bundle`; the caller writes it
    through its own file layer (the CLI via raw paths, the web via the
    concealed file-access framework), so the ONE producer serves both
    front-ends (staged-execution.md § 15.3).

    * ``fdf_files`` — ``{filename: fdf_text}``, one entry per ENABLED stage
      (``<label>_<name>.fdf``); all share ``cfg.system_label`` so SIESTA's
      ``.XV`` auto-read warm-restarts each stage.
    * ``runner_name`` / ``runner_text`` — the ``<label>.run.sh`` bash runner
      (write executable, 0o755).
    * ``pseudo_species`` — the species the caller must place pseudos for
      (the bundle-root shared package).
    * ``jobset`` — the ladder :class:`JobSet` (``None`` when
      ``emit_jobset=False``); serialise with ``jobset.write(dir/'job-set.json')``.
    """
    fdf_files: Dict[str, str]
    runner_name: str
    runner_text: str
    pseudo_species: List[str] = field(default_factory=list)
    jobset: Optional[JobSet] = None


def build_siesta_stage_bundle(
    struct,
    cfg,
    *,
    cell=None,
    shared: Optional[List[str]] = None,
    resources_for: Optional[Callable[[str], Resources]] = None,
    emit_jobset: bool = True,
) -> StageBundle:
    """Produce a multi-stage SIESTA bundle's contents as :class:`StageBundle`.

    PURE — no filesystem, no scheduler.  Reuses the existing tested
    renderers (``render_siesta_stage_fdfs`` / ``render_siesta_stages_runner``)
    + ``stages_to_jobset``; it only bundles them behind one seam so the CLI
    (``cli._emit_siesta_multi_stage``) and the web Build endpoint don't each
    re-glue the sequence (§ 15.3 Promotion A).

    ``cfg.system_label`` MUST already be the on-disk stem — the caller aligns
    it (the CLI to the ``.fdf`` filename, the web to the bundle basename), the
    single point where the ``<stem>_<stage>.fdf`` convention meets the
    SystemLabel that drives ``.XV``/``.DM`` warm-restart.

    ``cell`` rides on ``struct`` by default (``struct.resolve_cell()`` — the
    § 15.6 contract: the cell is carried by the structure, explicit in every
    stage ``.fdf``); pass it only when read separately from a file (the CLI).

    ``shared`` is the bundle-root static package symlinked into each stage
    dir; when ``emit_jobset`` and ``shared is None`` it defaults to the
    expected ``<species>.psml`` names (PSML-first, matching
    ``/api/siesta/install-pseudos``).  ``resources_for`` is the per-stage
    scheduler override seam (§ 6).

    Raises ``ValueError`` (from ``render_siesta_stage_fdfs`` /
    ``stages_to_jobset``) if no stage is enabled or the ladder is invalid.
    """
    from .input import (
        render_siesta_stage_fdfs,
        render_siesta_stages_runner,
        _detect_species,
    )

    fdf_files = render_siesta_stage_fdfs(struct, cfg, cell=cell)
    runner_name = f"{cfg.system_label}.run.sh"
    runner_text = render_siesta_stages_runner(cfg, siesta_cmd="siesta")

    species = (list(cfg.species_order) if getattr(cfg, "species_order", None)
               else _detect_species(struct.elements))

    jobset = None
    if emit_jobset:
        _shared = shared if shared is not None else [f"{s}.psml" for s in species]
        jobset = stages_to_jobset(cfg, shared=_shared,
                                  resources_for=resources_for)

    return StageBundle(
        fdf_files=fdf_files,
        runner_name=runner_name,
        runner_text=runner_text,
        pseudo_species=species,
        jobset=jobset,
    )


__all__ = ["stages_to_jobset", "StageBundle", "build_siesta_stage_bundle"]
