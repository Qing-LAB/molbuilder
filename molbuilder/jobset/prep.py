"""Prep engine — render the per-job launchers and lay out the tree
(docs/execution/job-system.md, step 5).

This is the step BETWEEN the pure ``materialize`` (data symlinks) and the
``submit`` launch.  It mirrors what the benchmark already does in
``bench/generate.py`` — render the wrappers **once per distinct script, in
the bundle root, from the REAL file** (so ``write_run_wrapper``'s
``Path.resolve()`` is a no-op and the ``.run.sh`` / ``.sbatch`` land where
intended), then symlink those wrappers into each job's ``point-<name>/`` dir
alongside the data symlinks.  Per-job resource *variation* is NOT baked here
— it is applied by ``submit`` as scheduler CLI flags over the shared
wrapper, exactly as the bench launch line does.  That is what lets one
rendered ``.sbatch`` serve every point of a sweep.

Why render-in-root-then-symlink (not render-in-each-dir): the materialized
script is a SYMLINK back to the bundle root; ``write_run_wrapper`` resolves
symlinks and would write the wrapper next to the *resolved* target.
Rendering from the real bundle-root file is the only placement that is both
correct and consistent with the benchmark (shared wrapper, CLI-flag
variation), so the two job-set kinds stay one mechanism.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .materialize import job_dir_names, shape_of, materialize, relink
from .model import Job, JobSet, Resources


class PrepError(Exception):
    """A JobSet could not be prepped (invalid set, or a script missing from
    the bundle root)."""


# --------------------------------------------------------------------- #
#  Does the deck agree with the launch?  (P6 unit 2 · project-layout      #
#  § 2.3.1 — "step 3 cannot precede step 1")                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class LaunchAgreement:
    """Whether a deck was rendered for the launch it is about to get.

    ``verdict`` is one of:

    | ``"silent"`` | the deck makes no claim about its launch, so there is nothing to disagree with — and nothing to refuse |
    | ``"agrees"`` | the rank count it was rendered for is the one it will get |
    | ``"differs"`` | the two are different numbers, or one is `auto` and the other is not |

    ``rendered_for`` is what the deck's BENCH-MARKS block records (an int, or
    the string ``"auto"``); ``launching_at`` is the job's ``mpi_np`` (an int,
    or ``None`` meaning *let the wrapper decide*).
    """
    verdict:      str
    rendered_for: Optional[Union[int, str]] = None
    launching_at: Optional[int] = None

    @staticmethod
    def _fmt(v) -> str:
        return "auto" if v in ("auto", None) else str(v)

    @property
    def rendered_text(self) -> str:
        return self._fmt(self.rendered_for)

    @property
    def launch_text(self) -> str:
        return self._fmt(self.launching_at)


def launch_agreement(job_dir, job) -> LaunchAgreement:
    """Read the deck's launch claim and compare it with this job's.

    `project-layout.md § 2.3.1` states the five steps and says the order is
    **forced**: *"Step 3 cannot precede step 1, because a deck carries values
    that depend on how it will be launched — a block size derived from the rank
    count … A parameter that depends on the launch cannot be decided before the
    launch is known."*

    Today step 3 happens at `molbuilder fdf`, on whatever machine typed it, and
    the rank count is resolved hours later by the wrapper. **The two halves of
    one ordered sequence run in different places, and nothing carried the
    first's answer to the third.** On 2026-08-10 that produced a deck rendered
    with no rank count — so ``BlockSize`` from the size-only branch — launched
    at ``-np 14``, and SIESTA refused at startup with *"You have too many
    processors for the system size"*.

    P4 unit 5 put the launch quantity **into** the deck, which is why that
    failure was diagnosable at all. **Recording is not agreeing**: this is the
    comparison, and it lives here rather than in `submit` because `prep` is the
    step that owns the deck and the wrapper (§ 2.3), and because a person is
    owed the answer at the moment they are still deciding — not at the moment
    they are committing cluster time.
    """
    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        return LaunchAgreement("silent")
    from ..parse.scripts.bench_marks import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
    if not marks or "mpi_np" not in marks:
        # A deck with no BENCH-MARKS block says nothing about its launch, so
        # there is nothing to disagree with.  The check is an agreement between
        # two statements, never a demand that every deck make one.
        return LaunchAgreement("silent")
    rendered_for = marks["mpi_np"]                 # an int, or the str "auto"
    launching_at = job.resources.mpi_np            # an int, or None == auto
    agree = ((rendered_for == "auto" and launching_at is None)
             or rendered_for == launching_at)
    return LaunchAgreement("agrees" if agree else "differs",
                           rendered_for, launching_at)


def check_launch_matches_deck(job_dir, job) -> None:
    """Refuse a launch the deck was not rendered for (P6 unit 2).

    Three outcomes, and the middle one is the live defect:

    * deck ``auto`` + launch ``auto`` — both defer to the wrapper. Fine.
    * deck ``auto`` + launch ``N`` — the deck's launch-derived values were
      computed with **no** rank count, and now one is being imposed. Refused.
    * deck ``N`` + launch ``M`` — refused, with both numbers named.

    This is the last honest moment, not the first: :func:`launch_agreement`
    answers the same question at `prep`, where it is still cheap to change
    your mind. Both call one comparison, so the warning and the refusal cannot
    come to different conclusions.
    """
    from .submit import SubmitError
    a = launch_agreement(job_dir, job)
    if a.verdict != "differs":
        return
    deck = os.path.basename(job.script)
    raise SubmitError(
        f"job {job.name!r}: this deck was rendered for mpi_np "
        f"{a.rendered_text}, and you are launching it at "
        f"{a.launch_text}.\n"
        f"  {deck} derives values from the rank count -- BlockSize above "
        f"all -- so a deck rendered for one launch is wrong for another "
        f"(project-layout.md § 2.3.1: a parameter that depends on the launch "
        f"cannot be decided before the launch is known).\n"
        f"  Re-render the deck for this launch, or launch it at "
        f"{a.rendered_text}.  The deck records what it assumed in its "
        f"BENCH-MARKS block, which is what made this checkable.")


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
    and the deck/launch agreement (:func:`launch_agreement`) is what actually
    refuses a wrong launch.
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
                emit_sbatch: bool = True, allocation=None) -> List[Path]:
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

    # ---- the ALLOCATION -- what you asked for, on this prep --------------- #
    # `project-layout.md` M4: an allocation is an input to `prep`, not a field
    # of the description and not a decision at submit.  Where it states a value
    # it WINS over whatever the JobSet carried, because the JobSet was built on
    # a machine that, by construction, did not know this one.
    if allocation is not None:
        stated = {k: v for k, v in dataclasses.asdict(allocation).items()
                  if v is not None}
        jobset = dataclasses.replace(jobset, jobs=[
            dataclasses.replace(j, resources=dataclasses.replace(j.resources,
                                                                 **stated))
            for j in jobset.jobs])

    errs = jobset.validate()
    if errs:
        raise PrepError(
            "cannot prep an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"bundle root not found: {base}")

    # ---- 0. resolve the machine (§ 2.3.1 step ONE) ---------------------- #
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
        for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
            if (base / wrapper).exists():
                relink(d, f"../{wrapper}", wrapper)
        if has_monitor:
            relink(d, "../mb_monitor.py", "mb_monitor.py")

    # ---- 4. emit STAGE-PLAN.md (§ 5 D3; mirrors bench's BENCH-PLAN.md) --- #
    # The reviewable plan lands in the bundle at prep, not just on the
    # `jobset plan` command's stdout.
    from .plan import render_plan
    (base / "STAGE-PLAN.md").write_text(render_plan(jobset) + "\n",
                                        encoding="utf-8")
    return dirs


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

#: What an engine must supply for `prep` to render its decks: the config class
#: its template rebuilds into, and a function from (structure, config) to deck
#: text. **Two things, and adding an engine edits no shared logic** —
#: `generator.md` § 7's seam, stated as data.
def _engine_seam(engine: str):
    if engine == "siesta":
        from ..config.siesta import SiestaConfig
        from ..siesta.input import render_fdf
        return SiestaConfig, render_fdf, ".fdf"
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
    from ..environment import Environment, resolve_environment
    path = resolve_target(base)
    if path.is_file():
        try:
            return Environment.from_json(path.read_text(encoding="utf-8"))
        except Exception:              # pragma: no cover - a hand-edited file
            pass
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
                     sweep=None, pins=None) -> List[Path]:
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
    config_cls, render_deck, suffix = _engine_seam(task.engine)
    try:
        pset = resolve(template_path.read_text(encoding="utf-8"), task,
                       config_cls, allocation=(allocation or Resources()),
                       stage=stage, sweep=sweep, pins=pins)
    except ResolveError as exc:
        raise PrepError(str(exc)) from exc

    # ---- 3. render the deck(s) ----------------------------------------- #
    struct = _structure_for(task, base)
    token = _token_for(task, pset.stage)
    jobs: List[Job] = []
    for element in pset:
        stem = f"{element.label}_{token}" if token else element.label
        script = f"{stem}{suffix}"
        # The deck is rendered from values ⊕ THIS element's allocation, so it
        # records the rank count it actually assumed.  Rendering from the
        # values alone emits `mpi_np auto` and the launch check then refuses a
        # deck that `prep` itself just made -- which is how this was found.
        (base / script).write_text(render_deck(struct, element.render_config()),
                                   encoding="utf-8")
        jobs.append(_job_for(element, script, task, pset.stage))

    # ---- 4 + 5, and the record floor 3 leaves behind -------------------- #
    js = JobSet(name=task.label, engine=task.engine,
                kind=("sweep" if pset.is_sweep else "ladder"),
                shared=_shared_for(base), jobs=jobs)
    js.write(base / "job-set.json")
    return prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       allocation=allocation)


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
    return ""


def _job_for(element, script: str, task, stage_name: Optional[str]) -> Job:
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
    from ..siesta.stages import _traits, _warm_declaration

    if element.point:
        name = point_token(element.point)
    elif stage_name:
        name = stage_name
    else:
        name = task.label

    siesta = task.engine == "siesta"
    return Job(name=name, script=script, resources=element.resources,
               warm=(_warm_declaration(element.label, element.values)
                     if siesta else []),
               traits=(_traits(element.values) if siesta else {}))


def _shared_for(base: Path) -> List[str]:
    """The static package every job links: whatever data files travel."""
    return sorted(p.name for p in base.glob("*.psml"))

__all__ = ["prep_calculation", "prep_jobset", "PrepError", "resolve_target",
           "launch_agreement", "check_launch_matches_deck",
           "LaunchAgreement"]
