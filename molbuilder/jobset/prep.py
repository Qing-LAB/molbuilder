"""Prep — the five steps of `project-layout.md` § 2.3.1.

:func:`prep_calculation` is the five entire, on the described route: a
description plus its template in, one rendered deck and wrapper **per
element** of the resolved :class:`~molbuilder.resolve.ParameterSet` out.
:func:`prep_jobset` is steps 4–5 alone, over an existing ``job-set.json`` —
the shared tail of the described route, and the library surface a
hand-built set uses directly.  It stopped being a CLI route on 2026-08-12
(U2/U4): `prep` is described-only, because the pre-made-bundle arm had no
producer left.

The framework here was first built inside the benchmark and the general part
lifted out (§ 2.3.1a: *benchmarking is `prep` whose parameters are a set
rather than a point* — the five steps are general, the grid is the
specialisation).

Wrappers render **once per distinct ``job.script``, in the bundle root, from
the real file**: ``write_run_wrapper`` resolves symlinks, so rendering from a
materialized link would land the wrapper beside the resolved target instead
of where the job runs.  On the described route every element renders its own
deck, so per-script is per-element and each wrapper carries its own
element's resources.  *(A "legacy sweep whose jobs share one script"
paragraph stood here promising its own fold "with bench (plan step 6)" —
the fold landed 2026-08-12 and no producer emits shared-script sets; for a
HAND-BUILT set that shares one script, the first job's resources still
become the wrapper defaults and ``submit`` passes each job's own as flags,
which is now a property of the fallback rather than a design of its own;
R8.)*
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .materialize import job_dir_names, shape_of, materialize, relink
from .model import Job, JobSet, Resources


class PrepError(Exception):
    """A JobSet could not be prepped (invalid set, or a script missing from
    the bundle root)."""


def resolve_target(base_dir) -> Path:
    """**Step 1 of the five: resolve the machine** (`project-layout.md`
    § 2.3.1) — probe cores, GPUs, scheduler and conda, and persist the answer
    as ``environment.json`` beside the bundle.

    **This step existed only inside the benchmark until 2026-08-10.**
    `bench/prep.py` did it; `prep_jobset` did not do it at all, so a staged
    calculation went straight to rendering wrappers on a machine nobody had
    asked about. § 2.3.1a is explicit about how to read that: *"`bench prep`
    is the one place this framework is already built, and it was built inside
    the benchmark because that is where the need appeared first … the general
    part needs lifting out of it"* — and *"stating it the other way round
    would make the general case look like a special case of the special
    case."*

    So the module moved out of `bench/` and became ``molbuilder/environment``.
    Its persisted artifact was **already** registered as
    ``molbuilder/environment@1`` (`job-contracts.md` § 6.1), which is the
    schema saying it was never the benchmark's to own.

    Written once per bundle and **not** overwritten on a later prep: the file
    records what this machine is, and re-probing on every stage would make two
    stages of one calculation disagree about their own target for no reason a
    user asked for. Delete it to force a re-probe.

    Returns the path to ``environment.json``.  Failure to probe is **not**
    fatal here — `prep` still has wrappers to render and a tree to lay out,
    and the deck/launch agreement (`agreement.launch_agreement`) is what
    actually refuses a wrong launch.
    """
    from ..environment import resolve_environment
    out = Path(base_dir) / "environment.json"
    if out.is_file():
        return out
    try:
        env = resolve_environment()
    except Exception:                     # pragma: no cover - probe is best-effort
        return out
    out.write_text(env.to_json() + "\n", encoding="utf-8")
    return out


def prep_jobset(jobset: JobSet, base_dir, *, env: str = None,
                emit_sbatch: bool = True, record_dir=None) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir``.

    Steps, in order:
      1. render each **distinct** ``job.script``'s ``.run.sh`` (and
         ``.sbatch`` when ``emit_sbatch`` and a scheduler is configured) in
         the bundle root, from the real file — reusing
         ``runwrap.write_run_wrapper`` (no reinvention).  The header carries
         the first-seen job's resources as defaults; ``submit`` overrides
         per job via CLI flags, so the defaults never decide the answer.
      2. ``materialize`` — data symlinks (shared package, script, carry).
      3. symlink each job's wrappers (+ shipped ``mb_monitor.py``) into its
         ``point-<name>/`` dir, so ``submit`` can ``sbatch``/``bash`` them
         there.

    Returns the per-job directories.  Raises :class:`PrepError` on an
    invalid JobSet or a script that isn't in the bundle root.
    """
    from ..runwrap import write_run_wrapper

    # The allocation is NOT a parameter here (U2, 2026-08-12; it was, and
    # re-applying it over per-element resources was the review's "stomp").
    # `project-layout.md` M4 still holds -- an allocation is an input to
    # *prep* -- but it enters ONCE, at resolve, where each element folds it
    # into its own resources (generator.md § 5); by this floor every job
    # already carries the answer.
    errs = jobset.validate()
    if errs:
        raise PrepError(
            "cannot prep an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"bundle root not found: {base}")

    # ---- 0. resolve the machine (§ 2.3.1 step ONE) ---------------------- #
    # Idempotent by contract (resolve_target early-returns on an existing
    # environment.json).  On the described route `prep_calculation` already
    # ran step 1; this call is the LEGACY route's step 1 -- prep_jobset is a
    # public entry for bundles that predate `describe` -- not a re-decision.
    resolve_target(base)

    # ---- 1. render wrappers once per distinct script (in the root) ------ #
    rendered: set = set()
    for job in jobset.jobs:
        if job.script in rendered:
            continue
        script_path = base / job.script
        if not script_path.is_file():
            raise PrepError(
                f"job {job.name!r}: script {job.script!r} not in bundle root "
                f"{base} (render the inputs before prep).")
        r = job.resources
        write_run_wrapper(
            script_path,
            env=env,
            mpi_np=r.mpi_np,
            cpus_per_task=r.cpus_per_task,
            time=r.time,
            gres=r.gres,
            mem=r.mem,
            exclusive=r.exclusive,
            # The warm-retry budget, which becomes no sbatch flag: the
            # wrapper bakes it into its own retry loop at install time
            # (running-a-job.md § 3.5).  This line is the second half of the
            # road job-contracts.md § 6.2 describes -- without it the field
            # was carried the whole way here and then dropped, which is why
            # `job-system.md § 4.1` recorded the SIESTA ladder as never
            # having implemented `continue` (2026-08-07, P2 unit 3).
            continue_retries=r.continue_retries,
            # Until 2026-08-11 this line did not exist, so a staged run
            # silently dropped a cap the user had set: `cli.py` and the web
            # blueprint both passed it and this call site did not.  Carried on
            # the allocation, it cannot be forgotten by one of three.
            max_memory_mb=r.max_memory_mb,
            emit_sbatch=emit_sbatch,
        )
        rendered.add(job.script)

    # ---- 2. data symlinks ---------------------------------------------- #
    dirs = materialize(jobset, base)

    # ---- 3. link the rendered wrappers (+ monitor) into each job dir ---- #
    has_monitor = (base / "mb_monitor.py").exists()
    # NOT named ``dirs``: step 2 above binds that to materialize's list of
    # created Paths, which is this function's return value.
    dir_of = job_dir_names(jobset, shape_of(jobset, base_dir))
    for job in jobset.jobs:
        d = base / dir_of[job.name]
        if d.resolve() == base.resolve():
            # FLAT: the wrappers step 1 just rendered, and the monitor, are
            # already in the directory this job runs in.  Linking here would
            # unlink the real files and point at the bundle's PARENT -- the
            # same destruction `materialize` guards against, one step later.
            continue
        stem = Path(job.script).stem
        # Computed prefix, not "..": a nested trial dir is depth 2 and a
        # hardcoded one-level hop would dangle (see materialize's own note).
        up = os.path.relpath(str(base), str(d))
        for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
            if (base / wrapper).exists():
                relink(d, os.path.join(up, wrapper), wrapper)
        if has_monitor:
            relink(d, os.path.join(up, "mb_monitor.py"), "mb_monitor.py")

    # ---- 4. emit STAGE-PLAN.md (§ 5 D3; mirrors bench's BENCH-PLAN.md) --- #
    # The reviewable plan lands in the bundle at prep, not just on the
    # `jobset plan` command's stdout.  It carries the CONFIG PROVENANCE --
    # which files supplied the effective execution settings -- so a
    # behaviour difference between two machines is explained by the bundle
    # itself (user request 2026-08-12; secrets excluded by construction).
    from ..runtime_config import config_provenance, format_provenance
    from .plan import render_plan
    # The plan lands BESIDE the job-set it describes: the run's at the
    # root, a bench's inside its stage's bench/ container -- so a bench
    # prep can never overwrite the run's reviewable plan (U1, 2026-08-12).
    plan_dir = Path(record_dir) if record_dir is not None else base
    (plan_dir / "STAGE-PLAN.md").write_text(
        render_plan(jobset) + "\n\n"
        + format_provenance(config_provenance(project_dir=base)) + "\n",
        encoding="utf-8")
    return dirs


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineSeam:
    """What an engine supplies for `prep` to run the five steps over it —
    `generator.md` § 7's seam ("a plugin, not a branch"), stated as data.

    Everything engine-specific that the loop below needs lives HERE, so the
    loop itself never asks which engine it is in.  ``_job_for`` branched on
    ``task.engine == "siesta"`` until 2026-08-12, which was § 7's forbidden
    ``if`` one floor down from where it was deleted.
    """
    #: The config class the template rebuilds into.
    config_cls: type
    #: ``(structure, config, stage_token=) -> deck text`` -- the token is
    #: a RENDER ARGUMENT (step 7, C7): the emitter never learns the word,
    #: the deck's filename carries it.
    render_deck: Callable
    #: The deck's type suffix (``.fdf``).
    suffix: str
    #: ``config -> the engine's identity literal`` (``SystemLabel`` / ``JOB``).
    label_of: Callable
    #: ``(config, label) -> config`` — the identity WRITTEN, for a trial's
    #: relabelling.  Filename relabelling alone is not the § 2.3.2
    #: protection: the deck's own ``SystemLabel`` line is what keys the warm
    #: files, and until 2026-08-12 it kept the run's label (found by the
    #: first sweep that ever rendered a deck).
    relabel: Callable
    #: ``(label, config) -> warm-file declaration`` for the Job.
    warm_for: Callable
    #: ``config -> traits`` the launcher routes on (GPU solver, …).
    traits_for: Callable


def _engine_seam(engine: str) -> EngineSeam:
    if engine == "siesta":
        from ..config.siesta import SiestaConfig
        from ..siesta.input import render_fdf
        from ..siesta.stages import _traits, _warm_declaration
        return EngineSeam(config_cls=SiestaConfig, render_deck=render_fdf,
                          suffix=".fdf",
                          label_of=lambda cfg: cfg.system_label,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, system_label=label),
                          warm_for=_warm_declaration, traits_for=_traits)
    raise PrepError(
        f"no deck writer for engine {engine!r}. An engine supplies its schema "
        f"and a deck writer (generator.md § 7); this backend has neither for "
        f"that name.")


def _environment_for(base: Path):
    """**Step 1**, and its answer is *returned* rather than only written.

    ``resolve_target`` persisted ``environment.json`` and its return value was
    discarded by the only caller — so floor 1 resolved a machine and nothing
    downstream ever heard the answer. That is the same defect as floor 2's, one
    storey down, and it is why this returns the object.
    """
    import json as _json

    from ..environment import Environment, resolve_environment
    path = resolve_target(base)
    if path.is_file():
        # NARROW except, deliberately: this called a method that did not
        # exist (`from_json`) from 2026-08-11 to 2026-08-12 and the broad
        # `except Exception` swallowed the AttributeError -- so the persisted
        # answer was never read back and every prep silently re-probed.  A
        # hand-edited file earns tolerance (fall through to a fresh probe);
        # a programming error does not.
        try:
            return Environment.from_dict(
                _json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, KeyError):
            pass                       # malformed file -> re-probe below
    try:
        return resolve_environment()
    except Exception:                  # pragma: no cover - probing is optional
        return None


def _structure_for(task, base: Path):
    """The structure this calculation is *of*, from the reference in the
    description (`stages.md` § 6.3 — a reference plus a witness, never a copy).

    Looked for beside the calculation first and at the recorded path second, so
    a folder carried to a cluster with its structure alongside resolves without
    the original tree existing there.

    **The witness is checked and the mismatch is loud**: a description opened
    against a structure that has since changed would otherwise build a
    different calculation under the same id — § 1's second failure mode.
    """
    from .. import load as _load_structure
    src = Path(task.structure.source)
    for candidate in (base / src.name, src):
        if candidate.is_file():
            struct = _load_structure(candidate)
            break
    else:
        raise PrepError(
            f"the structure this calculation describes is not here: "
            f"{task.structure.source!r}. `task.json` records a REFERENCE to it "
            f"(engines/stages.md § 6.3), so `prep` needs the file to be either "
            f"beside the calculation or still at that path.")
    if struct.formula != task.structure.formula or \
            struct.n_atoms != task.structure.atoms:
        raise PrepError(
            f"the structure has changed since this calculation was described: "
            f"the description witnesses {task.structure.formula} with "
            f"{task.structure.atoms} atoms, and {candidate} now holds "
            f"{struct.formula} with {struct.n_atoms}.\n"
            f"  Describing again is the honest fix -- rendering this deck would "
            f"build a different calculation under the same id.")
    return struct


def prep_calculation(base_dir, stage: Optional[str] = None, *,
                     allocation=None, env: str = None,
                     emit_sbatch: bool = True,
                     sweep=None, pins=None, translation=None) -> List[Path]:
    """**`prep`, entire** — the five steps of `project-layout.md` § 2.3.1, in
    the order it calls *forced rather than chosen*.

    1. **resolve the machine** — probe it, persist ``environment.json``;
    2. **resolve the parameters** — the description ⊕ this stage ⊕ the sweep ⊕
       the pins, into a :class:`~molbuilder.resolve.ParameterSet`;
    3. **render the deck(s)** — one per element of that set;
    4. **render the wrapper**;
    5. **build the run directory**.

    **Steps 2 and 3 did not exist here until 2026-08-11**, and their absence was
    stated by the code as a refusal: `prep` demanded that the decks already be
    in the bundle root, because they were finished at ``molbuilder fdf`` time on
    a machine that could not know the rank count. That is the *one real
    migration* — the producer ran at *produce* and belonged at *prep* — and
    steps 1 and 3 are now on the same side of the split, which is what
    § 2.3.1's *"step 3 cannot precede step 1"* was always about.

    Returns the per-job directories. Raises :class:`PrepError`.
    """
    from ..resolve import ResolveError, resolve
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..template import SUFFIX as TEMPLATE_SUFFIX

    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"calculation folder not found: {base}")
    desc = base / TASK_FILENAME
    if not desc.is_file():
        raise PrepError(
            f"no {TASK_FILENAME} in {base}. `prep` turns a DESCRIPTION into a "
            f"runnable directory; write one first with `jobset describe`.")

    # ---- 1. resolve the machine ---------------------------------------- #
    environment = _environment_for(base)

    # ---- 2. resolve the parameters ------------------------------------- #
    task = read_task(desc)
    template_path = base / f"{task.label}{TEMPLATE_SUFFIX}"
    if not template_path.is_file():
        raise PrepError(
            f"no {template_path.name} beside {TASK_FILENAME}. The portable "
            f"folder is a template PLUS a description (project-layout.md § 2.1) "
            f"and `prep` rebuilds the config from the template.")
    seam = _engine_seam(task.engine)
    try:
        pset = resolve(template_path.read_text(encoding="utf-8"), task,
                       seam.config_cls, allocation=(allocation or Resources()),
                       stage=stage, sweep=sweep, pins=pins,
                       translation=translation, environment=environment)
    except ResolveError as exc:
        raise PrepError(str(exc)) from exc

    # ---- 3. render the deck(s) ----------------------------------------- #
    struct = _structure_for(task, base)
    token = _token_for(task, pset.stage)
    jobs: List[Job] = []
    for element in pset:
        stem = f"{element.label}_{token}" if token else element.label
        script = f"{stem}{seam.suffix}"
        # The deck is rendered from values ⊕ THIS element's allocation, so it
        # records the rank count it actually assumed.  Rendering from the
        # values alone emits `mpi_np auto` and the launch check then refuses a
        # deck that `prep` itself just made -- which is how this was found.
        # The stage's artifact TOKEN reaches the emitter here.  It feeds three
        # names -- the deck, the engine's stdout, and the molwatch log -- and
        # leaving it unset made two stages of one calculation write to a single
        # `<label>.molwatch.log`.  Caught by the trajectory-log tests when
        # `molbuilder fdf` was deleted, which is the channel that used to carry
        # it.  C7 replaces `cfg.stage` with a render ARGUMENT; until then the
        # field is how the emitter is told, and `prep` -- which holds the
        # StageRef -- is what tells it.
        cfg = element.render_config()
        if element.is_trial:
            # The deck's OWN identity line carries the trial label -- this,
            # not the filename, is what keys SIESTA's warm files away from
            # the real run's (project-layout.md § 2.3.2).
            cfg = seam.relabel(cfg, element.label)
        # The stage's artifact token is a RENDER ARGUMENT (C7, 2026-08-12):
        # `prep` holds the StageRef, so `prep` says it, per call -- the
        # config field that used to carry it is gone, and the emitter never
        # learns the word (engines/stages.md § 1.1).
        (base / script).write_text(
            seam.render_deck(struct, cfg, stage_token=(token or None)),
            encoding="utf-8")
        _seed_trajectory_log(struct, cfg, base, engine=task.engine,
                             label=seam.label_of(cfg),
                             token=(token or None))
        jobs.append(_job_for(element, script, task, pset.stage, seam))

    # ---- 4 + 5, and the record floor 3 leaves behind -------------------- #
    # ``kind`` is INTENT, not length (review 2026-08-12: a one-point grid is
    # still a benchmark).  A sweep's whole record — its job-set, its plan,
    # later its verdict — lives in the stage's ``bench/`` container
    # (job-contracts.md § 6.3's Directories row, the cross-layer authority),
    # so two stages' benchmarks can never collide.  The ROOT job-set.json
    # is the RUN's plan and MERGES per stage, so prepping `tight` no longer
    # erases `coarse` — status, and the cross-stage ``--from`` carry whose
    # pair rule needs the source job on file, read the whole ladder.
    kind = "sweep" if sweep is not None else "ladder"
    js = JobSet(name=task.label, engine=task.engine, kind=kind,
                shared=_shared_for(base), jobs=jobs)
    if kind == "sweep":
        record_dir = base / _bench_container(task, token)
        record_dir.mkdir(parents=True, exist_ok=True)
    else:
        record_dir = base
        js = _merge_run_jobset(base / "job-set.json", js)
    js.write(record_dir / "job-set.json")
    # The allocation is NOT passed on: every job already carries its own
    # resolved resources, per element (generator.md § 5).  Passing it made
    # prep_jobset re-apply the BASE allocation over every job — the review's
    # "stomp": each trial's wrapper rendered with the base rank count
    # instead of its own translated G·K.
    return prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=record_dir)


def _bench_container(task, token: str) -> str:
    """Where a stage's bench state lives — ``<NN>_<stage>/bench`` in the
    hierarchy, ``bench`` at the root of a flat (or stageless) calculation.
    `job-contracts.md` § 6.3: *"benchmark | bench/ inside the stage it
    measures"*."""
    from .shape import Shape
    sd = Shape.named(task.shape).stage_dir(token) if token else "."
    return "bench" if sd == "." else f"{sd}/bench"


def _merge_run_jobset(path: Path, new: JobSet) -> JobSet:
    """The root ``job-set.json`` is the RUN's whole plan: each stage's prep
    updates its OWN row and leaves the others standing.

    Until 2026-08-12 every prep wrote only its own elements, so `prep run
    tight` erased `coarse` from floor 3 — breaking the status rollup and,
    worse, the ``--from`` pair rule: with the source job gone, `warm_carry`
    read the pair as unverified and silently withheld ``.CG``
    (`project-layout.md` § 2.3.4 row 3).
    """
    if not path.is_file():
        return new
    try:
        old = JobSet.load(path)
    except ValueError:
        return new              # unreadable or legacy: replaced outright
    if old.kind != "ladder":
        return new              # a pre-container sweep leftover: replaced
    fresh = {j.name for j in new.jobs}
    kept = [j for j in old.jobs if j.name not in fresh]
    merged = dataclasses.replace(
        new, jobs=kept + list(new.jobs),
        shared=sorted(set(old.shared) | set(new.shared)))
    # The plan's order is the LADDER's, not the order stages were prepped
    # in: re-prepping `coarse` must not move it below `medium`.  The seq
    # token is zero-padded (§ 6.3) so it sorts as it reads.
    from .materialize import stage_refs
    refs = stage_refs(merged)
    return dataclasses.replace(
        merged, jobs=sorted(merged.jobs,
                            key=lambda j: (refs[j.name].token or "", j.name)))


def _seed_trajectory_log(struct, cfg, base: Path, *, engine: str,
                         label: str, token=None) -> None:
    """Write the one-block preview the Watch tab discovers before a run starts.

    The deck NAMES its trajectory log; something has to CREATE it, or the tab
    has nothing to find until the engine writes its first step. That seeding
    lived inside ``convert`` — which writes a deck to disk — and `prep` renders
    the text and writes it itself, so the preview was silently skipped.

    **Found by the trajectory-log tests when `molbuilder fdf` was deleted.**
    They named a real property of the product, not of the verb, which is why
    they were repointed rather than retired.

    ``engine`` and ``label`` come through the caller from the
    :class:`EngineSeam` — this function hardcoded ``"siesta"`` and read
    ``cfg.system_label`` until 2026-08-12, which was the seam leaking.
    """
    if not getattr(cfg, "write_molwatch_log", False):
        return
    from ..trajectory_log import molwatch_log_basename, write_initial_preview
    # ``token`` is the caller's, same as the render argument (C7): the
    # config no longer carries a stage, and nothing here re-derives one.
    # The stage's own convergence targets travel with its log, so the Watch
    # tab's threshold line is THIS stage's and not the ladder's first.  They
    # come from the RESOLVED config, which is the whole point of resolving
    # before rendering: `coarse` and `tight` disagree about both of these.
    targets = {}
    for key, attr in (("max_force_ev_per_ang", "relax_force_tol"),
                      ("max_steps", "relax_steps")):
        value = getattr(cfg, attr, None)
        if value is not None:
            targets[key] = value
    write_initial_preview(
        struct,
        base / molwatch_log_basename(label, token),
        job=label, engine=engine,
        stage_name=token, convergence_targets=(targets or None))


def _token_for(task, stage_name: Optional[str]) -> str:
    """This stage's ``<NN>_<name>`` — the ONE namer (decision 27).

    ``NN`` is the stage's place in the **full** ladder, so disabling one leaves
    a gap rather than renumbering what follows: renumbering would hand an
    existing output to a stage that did not produce it.
    """
    if not task.stages or not stage_name:
        return ""
    from ..identity import stage_token
    for i, s in enumerate(task.stages, start=1):
        if s.name == stage_name:
            return stage_token(i, s.name)
    # Unreachable through prep_calculation -- resolve._stage_of already
    # refused an unknown stage -- and LOUD rather than "" if a future caller
    # reaches it another way: an empty token would silently drop the stage
    # from every artifact name (job-contracts.md § 6.3).
    raise PrepError(f"stage {stage_name!r} is not in this description's "
                    f"ladder: {', '.join(s.name for s in task.stages)}.")


def _job_for(element, script: str, task, stage_name: Optional[str],
             seam: EngineSeam) -> Job:
    """One element of the parameter set as one :class:`Job`.

    ``resources`` is **copied from the element**, never re-derived: the element
    resolved it once, from the allocation, and a second derivation here is the
    habit `generator.md` § 5 exists to end.

    The **name** answers *which job is this*, and there are exactly three
    answers because there are three things an element can be: a trial (named by
    its sweep coordinate), a rung of a ladder (named by the stage), or the whole
    calculation (named by its label).
    """
    from ..resolve import point_token

    if element.point:
        name = point_token(element.point)
    elif stage_name:
        name = stage_name
    else:
        name = task.label

    return Job(name=name, script=script, resources=element.resources,
               warm=seam.warm_for(element.label, element.values),
               traits=seam.traits_for(element.values))


def _shared_for(base: Path) -> List[str]:
    """The static package every job links — the pseudopotentials
    (``*.psml``; the shared-package data-file set, `project-layout.md` § 2.1).
    """
    return sorted(p.name for p in base.glob("*.psml"))

__all__ = ["prep_calculation", "prep_jobset", "PrepError", "resolve_target"]
