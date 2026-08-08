"""SIESTA stage-ladder → :class:`JobSet` producer
(docs/execution/job-system.md).

This is the ONLY place SIESTA stage knowledge meets the engine-agnostic
``jobset`` framework.  It turns a **template** (one ``SiestaConfig``) plus a
**ladder** (a list of :class:`molbuilder.task.Stage`, from ``task.json``)
into a ``ladder`` JobSet:

  * one ``Job`` per enabled stage, its input ``<label>_<name>.fdf``;
  * a dependency chain whose kind comes from the non-convergence policy
    (§ 5): ``proceed`` -> ``afterany`` (run the next stage regardless), else
    (``halt`` / ``continue``) -> ``afterok`` (next runs only if this stage
    ultimately converged; ``continue``'s terminal failure mode is halt);
  * carry-forward restart files (§ 8 D1): ``.XV`` always; ``.DM`` when
    ``cfg.use_save_dm``; ``.CG`` only when consecutive stages resolve to the
    same ``relax_type`` (optimizer history is algorithm-specific).

**Where the ladder comes from, and where it does not.**  An engine config
carries no stage list (``engines/stages.md`` § 1.1) — the ladder is the
user's decision about what varies, and it lives in the description.  So
every producer here takes ``stages`` as an argument, and resolves each one
against the template through the single ``effective_config`` seam.

Per-stage SCHEDULER resources are not stage fields either; they ride on
``Job.resources`` and are supplied via the ``resources_for`` seam —
default: every stage inherits the job-level config (mpi_np / omp /
continue_retries), everything else resolved at submit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional

from ..jobset.model import Carry, Job, JobSet, Resources
from ..task import Stage


#: The shipped ladder's per-stage non-convergence policy, keyed by stage
#: name.  It is **not** a stage field and **not** a shared-schema field
#: (``engines/stages.md`` § 3): its entire effect is the edge between one
#: attempt and the next, so it belongs to the JobSet producer's own input.
#: This constant is the shipped default FOR that input, and lives beside
#: :func:`default_siesta_stages` because the two describe one ladder.
#:
#: stage1 ``proceed``: a loose CG warm-up that runs out of steps has still
#: improved the geometry, and stage2 continues from it.  stage2 / stage3
#: ``halt``: a production tier that did not converge must fail loud.
DEFAULT_NONCONVERGENCE: Dict[str, str] = {
    "stage1": "proceed",
    "stage2": "halt",
    "stage3": "halt",
}


def default_siesta_stages(strategy: str = "publishable") -> List[Stage]:
    """The shipped SIESTA ladder as ``Stage`` objects.

    One stage per tier of :data:`~molbuilder.config.siesta.SIESTA_STAGE_PRESETS`,
    each carrying that tier's four values as its ``overrides``, enabled
    according to the named strategy preset.  So the science is stated once,
    in the presets table, and this only decides *which* tiers run and *how*
    they are shaped for the description.

    Replaces the deleted ``_default_siesta_stages`` + ``apply_siesta_stage_strategy``
    pair: building the ladder and applying the enable-mask were never two
    steps, they were one function split across a mutable default.

    Raises ``ValueError`` on an unknown strategy.
    """
    from ..config.siesta import (SIESTA_STAGE_PRESETS,
                                 SIESTA_STAGE_STRATEGY_PRESETS)

    if strategy not in SIESTA_STAGE_STRATEGY_PRESETS:
        valid = ", ".join(sorted(SIESTA_STAGE_STRATEGY_PRESETS))
        raise ValueError(
            f"unknown SIESTA stage strategy {strategy!r}; "
            f"choose from: {valid}")
    enables = SIESTA_STAGE_STRATEGY_PRESETS[strategy]
    return [
        Stage(name=f"stage{tier}",
              enabled=bool(enables[i]) if i < len(enables) else False,
              overrides=dict(SIESTA_STAGE_PRESETS[tier]))
        for i, tier in enumerate(sorted(SIESTA_STAGE_PRESETS))
    ]


def default_siesta_varies() -> List[str]:
    """The fields the shipped ladder varies, in first-seen order.

    This is what ``task.json``'s ``varies`` lists for the default ladder.
    **Derived from the overrides, never a second list** — § 6.2's subset
    rule is then checked against exactly the thing it was computed from,
    so the two cannot disagree.
    """
    from ..config.siesta import SIESTA_STAGE_PRESETS
    seen: List[str] = []
    for tier in sorted(SIESTA_STAGE_PRESETS):
        for k in SIESTA_STAGE_PRESETS[tier]:
            if k not in seen:
                seen.append(k)
    return seen


def _dep_kind(prev_policy: str) -> str:
    """SLURM dependency kind for the edge OUT of a stage with the given
    ``on_nonconvergence`` policy (§ 5).  ``proceed`` lets the next stage run
    regardless (afterany); ``halt`` and ``continue`` both end in
    success-or-failure, so the next stage runs only on success (afterok)."""
    return "afterany" if prev_policy == "proceed" else "afterok"


def stages_to_jobset(
    cfg,
    stages,
    *,
    shared: Optional[list] = None,
    resources_for: Optional[Callable[[str], Resources]] = None,
    on_nonconvergence: Optional[Mapping[str, str]] = None,
) -> JobSet:
    """Build the ladder :class:`JobSet` from a template and a stage list.

    ``cfg`` is the template — one ordinary ``SiestaConfig``.  ``stages`` is
    the ladder from the description.  ``shared`` is the static package
    symlinked into every stage dir (pseudopotentials, ``mb_monitor.py``, …)
    — the caller supplies it since only the bundle layer knows the concrete
    filenames.  ``resources_for`` optionally returns a per-stage
    :class:`Resources` override (keyed by stage name); absent → each stage
    inherits the job-level config.

    ``on_nonconvergence`` is this producer's own input (§ 3), keyed by stage
    name; a stage it does not name gets ``"halt"``.

    The ``.CG`` carry compares the stages' **resolved** ``relax_type``, not
    a stage field — a stage that does not override the optimizer has the
    template's, and carrying CG history into a Broyden stage would corrupt
    the restart either way.

    Raises ``ValueError`` if no stage is enabled, or if a stage overrides a
    field the shared schema does not have (from ``effective_config``), so a
    producer can't emit a JobSet the engines would choke on.
    """
    from .input import _enabled_stages, effective_config

    label = cfg.system_label
    enabled = _enabled_stages(stages)
    use_dm = bool(getattr(cfg, "use_save_dm", True))
    policy_of = dict(on_nonconvergence or {})
    # Resolve once, up front: the refusal for an unknown override should
    # arrive before any Job is built, and the ``.CG`` carry needs the
    # resolved optimizer anyway.
    resolved = {s.name: effective_config(cfg, s) for s in enabled}

    def _res(name: str) -> Resources:
        if resources_for is not None:
            r = resources_for(name)
            if r is not None:
                return r
        # Default: inherit the job-level parallel knobs; scheduler
        # resources (domain/time/exclusive/mem/gres) resolve at submit.
        # This is the config->exchange translation boundary: SiestaConfig's
        # ``omp_threads`` becomes the exchange field ``cpus_per_task``, and
        # its ``continue_retries`` rides across under its own name --
        # the one Resources field that becomes no sbatch flag at all
        # (job-contracts.md § 6.2).
        eff = resolved[name]
        return Resources(mpi_np=getattr(eff, "mpi_np", None),
                         cpus_per_task=getattr(eff, "omp_threads", None),
                         continue_retries=getattr(eff, "continue_retries", None))

    jobs = []
    prev = None
    for s in enabled:
        carry = []
        if prev is not None:
            carry.append(Carry(f"{label}.XV", prev.name))
            if use_dm:
                carry.append(Carry(f"{label}.DM", prev.name))
            if resolved[prev.name].relax_type == resolved[s.name].relax_type:
                carry.append(Carry(f"{label}.CG", prev.name))
        jobs.append(Job(
            name=s.name,
            script=f"{label}_{s.name}.fdf",
            resources=_res(s.name),
            depends_on=(prev.name if prev is not None else None),
            dep_kind=(_dep_kind(policy_of.get(prev.name, "halt"))
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
    stages,
    *,
    cell=None,
    shared: Optional[List[str]] = None,
    resources_for: Optional[Callable[[str], Resources]] = None,
    on_nonconvergence: Optional[Mapping[str, str]] = None,
    emit_jobset: bool = True,
) -> StageBundle:
    """Produce a multi-stage SIESTA bundle's contents as :class:`StageBundle`.

    PURE — no filesystem, no scheduler.  Reuses the existing tested
    renderers (``render_siesta_stage_fdfs`` / ``render_siesta_stages_runner``)
    + ``stages_to_jobset``; it only bundles them behind one seam so the CLI
    (``cli._emit_siesta_multi_stage``) and the web Build endpoint don't each
    re-glue the sequence (§ 15.3 Promotion A).

    ``cfg`` is the template and ``stages`` the ladder — see this module's
    docstring for why the second is an argument rather than a field of the
    first.  ``on_nonconvergence`` is the producers' shared policy input
    (§ 3), defaulting to :data:`DEFAULT_NONCONVERGENCE`.

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

    policy = (DEFAULT_NONCONVERGENCE if on_nonconvergence is None
              else on_nonconvergence)

    fdf_files = render_siesta_stage_fdfs(struct, cfg, stages, cell=cell)
    runner_name = f"{cfg.system_label}.run.sh"
    runner_text = render_siesta_stages_runner(cfg, stages, siesta_cmd="siesta",
                                              on_nonconvergence=policy)

    species = (list(cfg.species_order) if getattr(cfg, "species_order", None)
               else _detect_species(struct.elements))

    jobset = None
    if emit_jobset:
        _shared = shared if shared is not None else [f"{s}.psml" for s in species]
        jobset = stages_to_jobset(cfg, stages, shared=_shared,
                                  resources_for=resources_for,
                                  on_nonconvergence=policy)

    return StageBundle(
        fdf_files=fdf_files,
        runner_name=runner_name,
        runner_text=runner_text,
        pseudo_species=species,
        jobset=jobset,
    )


__all__ = ["DEFAULT_NONCONVERGENCE", "default_siesta_stages",
           "default_siesta_varies", "stages_to_jobset", "StageBundle",
           "build_siesta_stage_bundle"]
