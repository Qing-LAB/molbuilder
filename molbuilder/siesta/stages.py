"""SIESTA stage-ladder → :class:`JobSet` producer
(docs/execution/job-system.md).

This is the ONLY place SIESTA stage knowledge meets the engine-agnostic
``jobset`` framework.  It turns a **template** (one ``SiestaConfig``) plus a
**ladder** (a list of :class:`molbuilder.task.Stage`, from ``task.json``)
into a ``ladder`` JobSet:

  * one ``Job`` per enabled stage, its input ``<label>_<NN>_<name>.fdf``;
  * a dependency chain whose kind comes from the non-convergence policy
    (§ 5): ``proceed`` -> ``afterany`` (run the next stage regardless), else
    (``halt`` / ``continue``) -> ``afterok`` (next runs only if this stage
    ultimately converged; ``continue``'s terminal failure mode is halt);
  * **the warm-restart declaration** — SIESTA's group, stated where
    ``run-identity.md`` § 4 rule 1 says it belongs: ``restart: continue``
    means ``.XV``, ``.DM`` and ``.CG``, and the last is conditioned on the
    optimizer rather than resolved here, because the run it will be compared
    against is named at ``prep`` and not at produce (:func:`_warm_declaration`).

``Carry`` is still emitted, and is **derived** from that declaration rather
than spelled a second time: it is the same rule projected onto the one source
a chain can know in advance, the stage immediately before.  P7 unit 2 retires
those edges, at which point the projection is a deletion.

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

from ..jobset.model import (Carry, Job, JobSet, Resources, WarmFile,
                            warm_carry)
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
#  Keyed on the shipped ladder's stage NAMES (``SIESTA_STAGE_NAMES``), which
#  became coarse/medium/tight on 2026-08-10 -- the policy is per stage, so its
#  keys are stage names and they move when the ladder's do.
DEFAULT_NONCONVERGENCE: Dict[str, str] = {
    "coarse": "proceed",
    "medium": "halt",
    "tight":  "halt",
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
    from ..config.siesta import (SIESTA_STAGE_NAMES, SIESTA_STAGE_PRESETS,
                                 SIESTA_STAGE_STRATEGY_PRESETS)

    if strategy not in SIESTA_STAGE_STRATEGY_PRESETS:
        valid = ", ".join(sorted(SIESTA_STAGE_STRATEGY_PRESETS))
        raise ValueError(
            f"unknown SIESTA stage strategy {strategy!r}; "
            f"choose from: {valid}")
    enables = SIESTA_STAGE_STRATEGY_PRESETS[strategy]
    out: List[Stage] = []
    for i, tier in enumerate(sorted(SIESTA_STAGE_PRESETS)):
        overrides = dict(SIESTA_STAGE_PRESETS[tier])
        # § 4 rule 3: "A first stage is normally 'clean' and everything after
        # it 'continue'.  Nothing special is needed to say so."  Positional,
        # not tiered -- which is why it is set here and not in the presets
        # table, where every value is a property of that TIER's science.
        #
        # Keyed on POSITION IN THE TABLE, not on which stage is first
        # *enabled*.  Both readings are defensible and this one keeps an
        # invariant the ladder already had and the tests already assert: a
        # strategy preset says which tiers run and never retunes one.  Under
        # the other reading, `loose-only` (which disables stage2) and
        # `vib-quality` (which enables it) would hand stage2 different
        # overrides, so picking a strategy would silently change what a stage
        # computes.
        #
        # The case that reading protected -- stage1 off, so stage2 is first
        # and says 'continue' with nothing before it -- no shipped mask
        # produces, and it is not silent when it happens: the carry is empty
        # (`stages_to_jobset` has no predecessor to take from) and § 5's
        # surface + wrapper banner both report the absence.
        #
        # Before P3 unit 4 the ladder carried no `restart` at all and
        # continued anyway, because the three use_save_* booleans defaulted
        # True -- continuation by accident rather than by description, which
        # is exactly what § 4 rule 2 exists to end.
        overrides["restart"] = "clean" if i == 0 else "continue"
        # The tier's NAME, from the one table -- not ``f"stage{tier}"``, which
        # this built until 2026-08-10.  Decision 27 puts the ordinal in the
        # artifact token, so a name that is itself a position says the number
        # twice (``<label>_01_stage1.fdf``) and the science none.
        out.append(Stage(name=SIESTA_STAGE_NAMES[tier],
                         enabled=bool(enables[i]) if i < len(enables) else False,
                         overrides=overrides))
    return out


def _dep_kind(prev_policy: str) -> str:
    """SLURM dependency kind for the edge OUT of a stage with the given
    ``on_nonconvergence`` policy (§ 5).  ``proceed`` lets the next stage run
    regardless (afterany); ``halt`` and ``continue`` both end in
    success-or-failure, so the next stage runs only on success (afterok)."""
    return "afterany" if prev_policy == "proceed" else "afterok"


#: The key a `.CG` carry is conditioned on.  Named once because two modules
#: must spell it identically for the comparison to mean anything, and a typo
#: on either side reads as *"the optimizers disagree"* -- which withholds the
#: file silently rather than failing.  ``JobSet.validate`` catches the
#: declaration half; this constant is why there is only one spelling to catch.
OPTIMIZER_TRAIT = "optimizer"


def _traits(eff) -> Dict[str, str]:
    """The opaque per-job facts a warm file can be conditioned on (§ 4 rule 1).

    One entry today: which relaxation algorithm this stage runs. The `jobset`
    layer compares it as a string and never learns what it means, which is what
    keeps SIESTA's restart group out of the engine-agnostic core.
    """
    return {OPTIMIZER_TRAIT: str(getattr(eff, "relax_type", "") or "")}


def _warm_declaration(label: str, eff) -> List[WarmFile]:
    """What a stage with this resolved config takes from a run it continues.

    `run-identity.md` § 4 rule 4 fixes the set: *"`restart: continue` means the
    geometry, the density and the optimizer's history — `.XV`, `.DM`, `.CG` —
    because that is what continuing a relaxation *is*"*, and rule 2 makes
    ``restart`` the ONE field that says so. A ``clean`` stage declares nothing,
    and that is the same answer its deck gives: ``_continues`` gates
    ``MD.UseSaveXV`` / ``DM.UseSaveDM`` / ``MD.UseSaveCG`` too, so files placed
    for a ``clean`` stage would sit unread — § 4's *"present but not
    honoured"*, the silent half of the pair.

    Only `.CG` is conditional, and the condition is the pair's, not this
    stage's: *"a CG state is meaningless to a Broyden stage, so blindly
    carrying it would corrupt the restart"* (`job-system.md` § 4.1). Which
    source it will be compared against is unknown here — `--from` names it at
    `prep` — so what is declared is the **condition**, and
    :func:`~molbuilder.jobset.materialize.warm_carry` evaluates it once both
    stages are in hand.
    """
    from .input import _continues
    if not _continues(eff):
        return []
    return [WarmFile(f"{label}.XV"),
            WarmFile(f"{label}.DM"),
            WarmFile(f"{label}.CG", requires_same=OPTIMIZER_TRAIT)]


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

    The ``.CG`` condition is keyed on the stages' **resolved** ``relax_type``,
    not on a stage field — a stage that does not override the optimizer has
    the template's, and carrying CG history into a Broyden stage would corrupt
    the restart either way.  It is a *condition*, evaluated by
    :func:`~molbuilder.jobset.model.warm_carry` once ``--from`` has named the
    source; the ``Carry`` list here is that same rule against the predecessor.

    Raises ``ValueError`` if no stage is enabled, or if a stage overrides a
    field the shared schema does not have (from ``effective_config``), so a
    producer can't emit a JobSet the engines would choke on.
    """
    from .input import _enabled_stages, effective_config

    label = cfg.system_label
    enabled = _enabled_stages(stages)
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

    # The token each stage's files carry -- from the SAME helper the renderer
    # uses, so the JobSet cannot name a script the renderer did not write.
    from .input import _stage_tokens
    _token_of = {st.name: tok for st, tok in _stage_tokens(stages)}

    jobs = []
    prev = None
    for s in enabled:
        jobs.append(Job(
            # The JOB is named for the stage -- that is what a dependency edge
            # and a --stage-resources key point at.  The SCRIPT is named with
            # the stage's artifact token, because that is what
            # ``render_siesta_stage_fdfs`` actually wrote to disk.  The two
            # were one string until 2026-08-10, and after decision 27 a JobSet
            # built from the old rule would point every job at a file that
            # does not exist.
            name=s.name,
            script=f"{label}_{_token_of[s.name]}.fdf",
            resources=_res(s.name),
            depends_on=(prev.name if prev is not None else None),
            dep_kind=(_dep_kind(policy_of.get(prev.name, "halt"))
                      if prev is not None else "afterok"),
            warm=_warm_declaration(label, resolved[s.name]),
            traits=_traits(resolved[s.name]),
        ))
        prev = s

    # The chained form, DERIVED from the declaration rather than written a
    # second time.  ONE rule, two renderings: ``warm`` is what a stage takes
    # from whatever run it is pointed at -- which `prep` resolves, because
    # ``--from`` is where the source is named -- and ``carry`` is that same
    # rule projected onto the one source a CHAIN can know in advance, the
    # stage immediately before it.  They were two independent spellings of the
    # `.XV`/`.DM`/`.CG` rule until 2026-08-10, which is
    # `staged-runs-architecture.md` 12c's failure mode exactly: two lists that
    # agree today with nothing keeping them agreeing.  When P7 unit 2 retires
    # the inter-stage edges this loop is a deletion, not an untangling.
    for i, job in enumerate(jobs[1:], start=1):
        job.carry = [Carry(name, jobs[i - 1].name)
                     for name in warm_carry(job, jobs[i - 1])]

    return JobSet(
        name=label,
        engine="siesta",
        kind="ladder",
        shared=list(shared or []),
        jobs=jobs,
    )


# --------------------------------------------------------------------- #
#  Pure bundle producer (job-system.md § 4.1, Promotion A)               #
# --------------------------------------------------------------------- #


@dataclass
class StageBundle:
    """The contents of a multi-stage SIESTA bundle, as DATA — no filesystem.

    Produced by :func:`build_siesta_stage_bundle`; the caller writes it
    through its own file layer (the CLI via raw paths, the web via the
    concealed file-access framework), so the ONE producer serves both
    front-ends (job-system.md § 4.1).

    **ONE PACKAGE, for either layout** (decision 29). The decks and the JobSet
    are the same whichever shape the description asks for; what differs is how
    the stages are kept apart on disk, and `prep` applies that —
    `project-layout.md` § 1: *"The browser **always writes the same thing** …
    `prep` translates that into a runnable directory in whichever shape you ask
    for"*, with **Chosen: at `prep`** in both of its columns.

    Two earlier shapes of this object were wrong in opposite directions, and
    both are worth remembering. It first carried the flat runner **always** and
    a JobSet whenever a caller passed ``emit_jobset`` — so a produce emitted
    both layouts and the shape was settled by whatever command came next. The
    fix for that branched the **producer** on ``shape``, which put the decision
    one layer too early: flat then got a bash runner and no JobSet, so it could
    not use the framework at all. Neither is a choice the description makes.

    * ``fdf_files`` — ``{filename: fdf_text}``, one entry per ENABLED stage
      (``<label>_<NN>_<name>.fdf``); all share ``cfg.system_label`` so SIESTA's
      ``.XV`` auto-read warm-restarts each stage.
    * ``pseudo_species`` — the species the caller must place pseudos for
      (the bundle-root shared package).
    * ``jobset`` — the ladder :class:`JobSet`, in **either** shape: it is what
      makes ``jobset prep`` / ``submit run --chain`` the launcher for both,
      which is the user's decision of 2026-08-10 (*"the prep, deployment and
      execution chain of command is the same framework"*).
    """
    fdf_files: Dict[str, str]
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
) -> StageBundle:
    """Produce a multi-stage SIESTA bundle's contents as :class:`StageBundle`.

    PURE — no filesystem, no scheduler.  Reuses the existing tested
    renderer (``render_siesta_stage_fdfs``)
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
    dir; for a hierarchical bundle with ``shared is None`` it defaults to the
    expected ``<species>.psml`` names (PSML-first, matching
    ``/api/siesta/install-pseudos``).  ``resources_for`` is the per-stage
    scheduler override seam (§ 6).

    **It takes no ``shape``.** § 6.7's *"`prep` **reads** it; it does not decide
    it"* is a statement about which layer applies the layout, and the layer is
    `prep` — not this one. The description still carries the shape; this
    producer simply has no use for it.

    Raises ``ValueError`` (from ``render_siesta_stage_fdfs`` /
    ``stages_to_jobset``) if no stage is enabled or the ladder is invalid.
    """
    from .input import render_siesta_stage_fdfs, _detect_species

    policy = (DEFAULT_NONCONVERGENCE if on_nonconvergence is None
              else on_nonconvergence)

    # ONE PACKAGE, for either layout (decision 29).  The decks and the JobSet
    # are the same whichever shape the description asks for -- what differs is
    # how the stages are KEPT APART on disk, and `prep` is where that is
    # applied (`project-layout.md` § 1: "the browser always writes the same
    # thing ... prep translates that into a runnable directory in whichever
    # shape you ask for", and its table reads "Chosen: at prep" in BOTH
    # columns).
    fdf_files = render_siesta_stage_fdfs(struct, cfg, stages, cell=cell)
    species = (list(cfg.species_order) if getattr(cfg, "species_order", None)
               else _detect_species(struct.elements))
    _shared = shared if shared is not None else [f"{s}.psml" for s in species]

    return StageBundle(
        fdf_files=fdf_files,
        pseudo_species=species,
        jobset=stages_to_jobset(cfg, stages, shared=_shared,
                                resources_for=resources_for,
                                on_nonconvergence=policy),
    )


__all__ = ["DEFAULT_NONCONVERGENCE", "OPTIMIZER_TRAIT",
           "default_siesta_stages",
           "stages_to_jobset", "StageBundle",
           "build_siesta_stage_bundle"]
