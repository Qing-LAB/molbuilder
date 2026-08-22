"""Prep — THE CONDUCTOR of `docs/execution/script-preparation.md`.

It walks floors 1 -> 4 in order and owns no decision of its own: the five
steps, and the eleven sub-steps inside step 3, are that document's (§ 4);
what each engine supplies at each step is its § 5.  **It may call, but it
may never decide** -- a value settled here is a value no floor owns, which
is the shape of the "stomp" bugs (§ 3.3).

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
become the wrapper defaults and ``launch`` passes each job's own as flags,
which is now a property of the fallback rather than a design of its own;
R8.)*
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .. import script_emit as _sc
from .materialize import job_dir_names, shape_of, materialize, relink
from ..issues import calling as _calling
from .model import FILENAME as JOBSET_FILENAME, Job, JobSet, Resources


class PrepError(Exception):
    """A JobSet could not be prepped (invalid set, or a script missing from
    the bundle root)."""


from contextlib import contextmanager


@contextmanager
def _user_error_as_prep():
    """Translate the USER-FIXABLE refusal classes the steps below raise into
    :class:`PrepError`, so every caller of the two prep entries has ONE class
    to catch.  Without this the described route's two most likely
    first-contact failures -- missing pseudos (``ValidationError`` out of the
    deck render) and an unset ``script_generation.activation``
    (``RuntimeConfigError`` out of the wrapper render) -- escaped the CLI as
    raw tracebacks (2026-08-12 plan A8).  Only the NAMED classes translate: a
    ``TypeError`` here is a bug and should look like one.

    **The hook boundary's attribution is carried across** (§ 4.6). ``str(exc)``
    does not include an exception's notes, so a refusal that came out of a
    named engine hook would arrive at the CLI saying what went wrong and not
    whose it was -- and the note would reach a traceback nobody sees."""
    from ..issues import ValidationError, notes_of
    from ..runtime_config import RuntimeConfigError
    from ..runwrap import WrapperError
    try:
        yield
    except (ValidationError, RuntimeConfigError, WrapperError) as exc:
        note = notes_of(exc)
        raise PrepError(str(exc) + (f"\n  ({note})" if note else "")) from exc


def resolve_target(base_dir, target: Optional[str] = None) -> Path:
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
    from ..environment import FILENAME, machine_for, write_environment
    out = Path(base_dir) / FILENAME
    if out.is_file():
        return out
    # `machine_for()` WITHOUT a bundle: the calculation has no record yet (we
    # just early-returned if it did), so this is the MACHINE scope -- what
    # `jobset probe` wrote -- and a fresh probe only when that is absent too.
    # Snapshotting the machine's answer rather than re-probing is what makes
    # one probe serve every calculation here (configuration.md § 5, M-3).
    # ``target`` names WHICH machine this is for (P2).  Without it: this
    # machine's record, else a fresh probe.  With it: the named record, and an
    # error naming the ones that exist if it is not among them -- a benchmark
    # prepped for a cluster was silently measured against the workstation it
    # was prepped ON, because there was one record and nobody said otherwise.
    env = machine_for(target=target, probe=(target is None))
    if env is None:                       # pragma: no cover - probe is best-effort
        return out
    return write_environment(env, out)


def prep_jobset(jobset: JobSet, base_dir, *, env: str = None,
                emit_sbatch: bool = True, record_dir=None,
                log=None) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir``.

    Steps, in order:
      1. render each **distinct** ``job.script``'s ``.run.sh`` (and
         ``.sbatch`` when ``emit_sbatch`` and a scheduler is configured) in
         the bundle root, from the real file — reusing
         ``runwrap.write_run_wrapper`` (no reinvention).  The header carries
         the first-seen job's resources as defaults; ``launch`` overrides
         per job via CLI flags, so the defaults never decide the answer.
      2. ``materialize`` — data symlinks (shared package, script, carry).
      3. symlink each job's wrappers (+ shipped ``mb_monitor.py``) into its
         ``point-<name>/`` dir, so ``launch`` can ``sbatch``/``bash`` them
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
    if log is not None:
        log.phase("STEP 4 · WRAPPERS — how each deck is launched")
    rendered: set = set()
    for job in jobset.jobs:
        if job.script in rendered:
            if log is not None:
                log.note(f"{job.name}: shares {job.script}'s wrapper, "
                         f"already rendered")
            continue
        script_path = base / job.script
        if not script_path.is_file():
            raise PrepError(
                f"job {job.name!r}: script {job.script!r} not in bundle root "
                f"{base} (render the inputs before prep).")
        with _user_error_as_prep():
            # The ALLOCATION, whole (architecture.md § 3.1, rule A8).  This
            # call listed nine of the wrapper's eleven keyword arguments until
            # 2026-08-17 and omitted `omp_threads`, so every deck here shipped
            # a `.sbatch` asking for `-c N` beside a `.run.sh` that baked an
            # OMP default of 1 -- invisible under sbatch, where
            # SLURM_CPUS_PER_TASK outranks the default, and silently flat on a
            # workstation, which is where a benchmark's cores-per-rank axis
            # stopped measuring anything.
            #
            # It is the second time this door lost a field to a hand-copied
            # argument list: `max_memory_mb` went the same way on 2026-08-11,
            # and that fix moved the field onto `Resources` without changing
            # how it is passed.  Passing the object is what buys the sentence
            # that fix wrote -- *carried on the allocation, it cannot be
            # forgotten by one of them.*
            write_run_wrapper(
                script_path,
                resources=job.resources,
                env=env,
                emit_sbatch=emit_sbatch,
            )
        rendered.add(job.script)
        if log is not None:
            _stem = Path(job.script).stem
            log.received(job.script, _flat_resources(job.resources)
                         + (f", env={env}" if env else ""))
            for _w in (f"{_stem}.run.sh", f"{_stem}.sbatch"):
                if (base / _w).is_file():
                    log.produced(_w, f"{len((base / _w).read_text().splitlines())}"
                                     f" lines")
                elif _w.endswith(".sbatch"):
                    log.note(f"{_w}: not written "
                             + ("(emit_sbatch off)" if not emit_sbatch
                                else "(no scheduler configured)"))

    # ---- 2. data symlinks ---------------------------------------------- #
    dirs = materialize(jobset, base)
    if log is not None:
        log.phase("STEP 5 · RUN DIRECTORY — where each job will be launched")
        log.received("shape", str(shape_of(jobset, base_dir)))
        log.received("shared package",
                     ", ".join(jobset.shared) or "nothing (W5)")
        for _d in dirs:
            log.produced(_d.name if _d.resolve() != base.resolve() else ".",
                         "the bundle root — flat runs here, nothing is linked"
                         if _d.resolve() == base.resolve() else str(_d))

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
    # WHICH file supplied the warm-file vocabulary (U6a provenance): the
    # engine's own, or this calculation's fine-tuned copy -- a surprising
    # carry must be debuggable from the plan alone (§ 4.2a).
    try:
        from ..warmfiles import load_warm_files
        _vocab = f"warm-files: {load_warm_files(jobset.engine, base).path}\n"
    except Exception:
        _vocab = ""    # an engine without a rules file has no line to print
    (plan_dir / "STAGE-PLAN.md").write_text(
        render_plan(jobset) + "\n\n" + _vocab
        + format_provenance(config_provenance(project_dir=base)) + "\n",
        encoding="utf-8")
    if log is not None:
        log.produced("STAGE-PLAN.md", str(plan_dir / "STAGE-PLAN.md"))

    return dirs


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineSeam:
    """What an engine supplies for `prep` to run the steps over it —
    `script-preparation.md` § 4's seam, stated as data.

    **That document indexes the questions by the STEP that asks them**, which is
    the ordering to read them in: a bag of callables cannot answer *"what does
    this engine still owe?"*, and against the steps a gap is a blank row.

    **Fifteen questions, ten members.**  One is answered by shared code
    (`validation.validate`), and four arrive together through ``spec_for`` --
    the layout, the syntax, the record's values and the check rules are all the
    engine describing its deck, so they ride on one ``DeckSpec`` rather than on
    four seam members.  A member that answers ``None`` is answering *nothing*,
    which is a real answer and a recorded one (§ 4, W5): PySCF gives it to
    ``provide_data``, ``shared_package`` and ``sibling_artifacts``, and to
    ``bench_marks`` on the spec.

    Everything engine-specific that the loop below needs lives HERE, so the
    loop itself never asks which engine it is in.  ``_job_for`` branched on
    ``task.engine == "siesta"`` until 2026-08-12, which was § 7's forbidden
    ``if`` one floor down from where it was deleted.
    """
    #: The config class the template rebuilds into.
    config_cls: type
    #: ``(structure, config, stage_token=) -> DeckSpec`` — the engine
    #: DESCRIBES its deck; the framework renders, writes and checks it
    #: (`script-preparation.md` § 4.3).  The token is a RENDER ARGUMENT (step
    #: 7, C7): the emitter never learns the word, the deck's filename carries
    #: it.
    #:
    #: **It handed back finished TEXT until 2026-08-18**, and that one fact was
    #: what kept the framework's step-3 runner unreachable: given text, the
    #: conductor had no form to pass on, so it performed the write and the
    #: check itself and the ORDER of step 3 was stated in two places.  Given a
    #: form, the framework can also re-derive what the deck was supposed to
    #: contain, so nothing has to be carried alongside the text to make the
    #: check possible.
    spec_for: Callable
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
    #: ``(label, config, calculation, base_dir) -> warm-file declaration``
    #: for the Job -- the engine reads its § 4.2a rules file for the TYPE
    #: (U2), and ``base_dir`` lets a calculation's own fine-tuned copy win
    #: (U6a; the comment said 3 args while the call passed 4 -- C-d).
    warm_for: Callable
    #: ``config -> traits`` the launcher routes on (GPU solver, …).
    traits_for: Callable
    #: ``(struct, config, deck_path) -> None`` — the sibling files this
    #: engine's deck TEXT promises (E6: a charged SIESTA deck instructs
    #: running a script; the promise must be kept on every route that
    #: renders the deck).  ``None`` for an engine whose decks promise
    #: nothing.
    sibling_artifacts: Optional[Callable] = None
    #: ``(base_dir) -> [filename]`` — which files in the calculation are the
    #: SHARED PACKAGE every job links to.  The engine that put them there is
    #: the one that can name them: this was a ``*.psml`` glob in shared code,
    #: a SIESTA fact stated a floor below where SIESTA may speak, so a second
    #: engine with data files of its own would have shipped none of them.
    #: ``None`` for an engine that puts nothing in — the package is then empty,
    #: which is the honest answer rather than an accident of a glob.
    shared_package: Optional[Callable] = None
    #: ``(struct, config, base_dir) -> None`` — the DATA FILES this engine's
    #: deck cannot run without, put into the calculation.
    #:
    #: *Named ``stage_data`` for about a minute, until `test_stage_vocabulary`
    #: refused it: "stage" is this project's core noun and here it was being
    #: borrowed as a verb, which is the collision that ledger exists to catch
    #: — the same refusal `submit._staged_for_launch` earned.*  Distinct from
    #: ``sibling_artifacts``, which is about what a deck's own text promises;
    #: this is about what the ENGINE will open.  ``None`` for an engine that
    #: needs none (PySCF's basis sets ship inside PySCF).
    #:
    #: It belongs to `prep` because `project-layout.md` § 2.6 puts the copy on
    #: the machine that runs the job — where the library lives is a fact about
    #: that machine — and because `prep` is already what decides the shared
    #: package.  Added 2026-08-18: the rule *"a calculation copies the
    #: pseudopotentials it needs into its own shared package"* was written and
    #: unowned, so `jobset init` performed it and the browser's hand-over
    #: did not, and a calculation described in the browser prepped, laid out
    #: its directories and reported success with no pseudopotentials in it.
    provide_data: Optional[Callable] = None


def _siesta_sibling_artifacts(struct, cfg, deck_path: Path) -> None:
    """The sibling files a SIESTA deck's own text PROMISES.

    A charged deck instructs ``python3 makov_payne_correction.py`` in its
    header -- a promise only ``convert`` kept until E6 (redo 2026-08-12):
    the described route rendered the same header and never wrote the
    script, so `prep` shipped an instruction to run a file that did not
    exist.  Same writer both routes, so they cannot drift."""
    from ..chemistry import resolve_net_charge
    from ..siesta.makov_payne import emit_correction_script
    try:
        q = resolve_net_charge(struct, getattr(cfg, "net_charge", None))
    except Exception:
        q = 0
    if q != 0:
        emit_correction_script(fdf_path=deck_path,
                               system_label=cfg.system_label, q=q)


def _siesta_provide_pseudos(struct, cfg, base: Path) -> None:
    """Put the pseudopotentials this deck needs into the calculation.

    SIESTA opens ``<element>.psml`` in the directory it runs from and has no
    search path, so a missing file is not a preference — it is a run that
    cannot start, after a queue wait and however long MPI takes to come up.

    **Idempotent, and the folder wins.**  Anything already here was put here by
    an earlier prep, by `jobset init`, or by travelling with the folder;
    `copy_pseudopotentials` leaves it alone.  Only what is missing is fetched,
    from the library named by ``psml_lib``.

    **The species come from the STRUCTURE**, which `prep` has just loaded and
    checked against the description's witness — not from a list in the
    description.  A recorded list would be a second answer to *which elements
    is this calculation of*, and the structure is the first.

    A species in neither place stops `prep` **by name**, before a deck is
    written.

    **And then the science protocol runs on what is actually there.**
    `science/pseudopotentials.md` exists because a defective `S.psml` with a
    dead p-channel shipped into a real run on 2026-06-26: wrong sulfur bonding,
    and `propor: ERROR: IMAX=0` — but only at high rank counts, so a small run
    would have reported plausible, wrong numbers instead of crashing. The check
    that catches that class reads the pseudopotentials themselves.

    It has always run against ``psml_lib`` — the LIBRARY — and it is gated on
    that field being set, so a calculation whose files are already beside it and
    whose ``psml_lib`` is empty had **nothing checked at all**: the only thing
    said was *"psml_lib is not set … once set, this preflight will check
    coverage"*, while three real pseudopotentials sat in the folder the run
    would open them from. This step makes that state the normal one, so it runs
    the protocol here, against the **calculation** — which is where the files
    the run reads actually are — and refuses on the same ERROR statuses the
    preflight and `molbuilder pseudo check` refuse on, from the same shared
    constant.
    """
    from ..pseudos import resolve_psml_lib
    from ..siesta.input import _detect_species, copy_pseudopotentials

    species = _detect_species(struct.elements)
    if not species:
        return
    have = {p.stem for p in base.glob("*.psml")}
    want = [s for s in species if s not in have]
    if not want:
        _screen_pseudos(species, cfg, base)
        return

    lib_raw = getattr(cfg, "psml_lib", None)
    if not lib_raw:
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)} and none are in {base.name}/, but no "
            f"pseudopotential directory is set.  Set `psml_lib` in the "
            f"template to the library they live in -- the convention is the "
            f"bare name `pseudopotential`, which means the projects tree "
            f"this calculation lives in (project-layout.md § 2.6, "
            f"job-contracts.md § 2.5a).")
    lib = resolve_psml_lib(str(lib_raw), dest_dir=base)
    if not lib.is_dir():
        # Name the anchor the SPELLING asked for, in the rule's own words.
        # This used to print only the resolved path, which under the old
        # cascade was whichever candidate was tried last -- on Sol that was
        # `<calc>/projects/pseudopotential`, a folder assembled from the
        # user's working directory that nobody had chosen (2026-08-21).
        from ..pseudos import describe_psml_anchor
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)}, and the library they should come from is "
            f"not a directory.  "
            + describe_psml_anchor(str(lib_raw), dest_dir=base)
            + f"  Put the .psml files there, or set `psml_lib` to a "
              f"directory that has them.")
    missing = copy_pseudopotentials(want, lib, base)
    if missing:
        raise PrepError(
            f"this calculation needs {', '.join(f'{m}.psml' for m in missing)}"
            f" and there is none in {base.name}/ or in {lib}.  SIESTA opens "
            f"<element>.psml in the directory it runs from and has no search "
            f"path, so it would refuse at startup.  Put the file in the "
            f"library, or point `psml_lib` at one that has it.")
    _screen_pseudos(species, cfg, base)


def _screen_pseudos(species, cfg, base: Path) -> None:
    """`science/pseudopotentials.md` § 1, run on the calculation's own files.

    Same engine (`pseudos.check_coverage`), same severity set
    (`pseudos.ERROR_STATUSES`) and same XC-family table
    (`pseudos.expected_xc_family`) as the render-time preflight and the
    `molbuilder pseudo check` CLI, so the three surfaces cannot disagree about
    what blocks.  What differs is only WHICH DIRECTORY is read: this one asks
    about the files the run will open.

    The three blocking statuses are `missing`, `dead_projector` and
    `xc_family_mismatch` — a file absent, a valence channel physically absent,
    or the wrong XC family.  The rest are advisory and are printed.
    """
    from ..pseudos import ERROR_STATUSES, check_coverage, expected_xc_family
    import sys as _sys
    entries = check_coverage(
        species, base,
        expected_xc_family=expected_xc_family(
            getattr(cfg, "xc_authors", "") or ""),
        expected_xc_authors=(getattr(cfg, "xc_authors", "") or "") or None,
    )
    blocking = [e for e in entries if e.status in ERROR_STATUSES]
    for e in entries:
        if e.status != "ok" and e not in blocking:
            print(f"  note: {e.message}", file=_sys.stderr)
    if blocking:
        raise PrepError(
            "the pseudopotentials in this calculation do not pass the "
            "screening (science/pseudopotentials.md § 1):\n  - "
            + "\n  - ".join(f"{e.element}: {e.message}" for e in blocking)
            + "\n  These are the checks that exist because a dead-channel "
              "S.psml once shipped into a real run -- wrong bonding, and a "
              "propor IMAX=0 crash that only appeared at high rank counts.")


def _engine_seam(engine: str) -> EngineSeam:
    if engine == "siesta":
        from ..config.siesta import SiestaConfig
        from ..siesta.input import spec_for as _siesta_spec
        from ..siesta.stages import _traits, _warm_declaration
        return EngineSeam(config_cls=SiestaConfig, spec_for=_siesta_spec,
                          suffix=".fdf",
                          label_of=lambda cfg: cfg.system_label,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, system_label=label),
                          warm_for=_warm_declaration, traits_for=_traits,
                          sibling_artifacts=_siesta_sibling_artifacts,
                          provide_data=_siesta_provide_pseudos,
                          shared_package=_siesta_shared_package)
    if engine == "pyscf":
        from ..config.pyscf import PySCFConfig
        from ..pyscf.input import spec_for as _pyscf_spec
        from ..pyscf.stages import _traits as _pyscf_traits
        from ..pyscf.stages import _warm_declaration as _pyscf_warm
        # NO ``provide_data`` and NO ``sibling_artifacts``, and both absences
        # are ANSWERS rather than omissions (`script-preparation.md` § 4, W5):
        # PySCF's basis sets ship inside PySCF, so there is no file to put in
        # the calculation; and its script's own text instructs nothing to be
        # run beside it, so there is no promise to keep.
        return EngineSeam(config_cls=PySCFConfig, spec_for=_pyscf_spec,
                          suffix=".py",
                          label_of=lambda cfg: cfg.job_name,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, job_name=label),
                          warm_for=_pyscf_warm, traits_for=_pyscf_traits)
        # NO ``shared_package``: PySCF's basis sets ship inside PySCF, so
        # there is nothing in the calculation for every job to link to.
    raise PrepError(
        f"no deck writer for engine {engine!r}. An engine supplies its "
        f"catalogue rows and an answer at each preparation step "
        f"(script-preparation.md § 4); this backend has neither for that "
        f"name.")


def _environment_for(base: Path, target: Optional[str] = None):
    """**Step 1**, and its answer is *returned* rather than only written.

    ``resolve_target`` persisted ``environment.json`` and its return value was
    discarded by the only caller — so floor 1 resolved a machine and nothing
    downstream ever heard the answer. That is the same defect as floor 2's, one
    storey down, and it is why this returns the object.

    **The reading and the precedence are not this module's** (N1). They were
    inline here -- a ``json.loads`` plus a hand-written fallback -- which is how
    the tree came to hold two readers of one file returning two different
    types.  ``environment.machine_for`` is the one door, and the narrow-except
    reasoning that used to live in this body moved with it.
    """
    from ..environment import machine_for
    resolve_target(base, target)       # step 1 proper: snapshot, idempotent
    # probe=True: this IS step 1, and if the snapshot could not be written
    # (read-only tree, a racing prep) `prep` still needs an answer to resolve
    # against rather than a None that silently drops the machine.
    return machine_for(base, target=target, probe=(target is None))


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
    from ..workingcopy_structure import StructureCodec
    codec = StructureCodec()
    src = Path(task.structure.source)
    # The codec, not the bare loader: describe writes the structure as the
    # pair when it carries metadata (--vacuum lives NOWHERE a bare .xyz can
    # put it), and the codec applies the .molstruct.json when present and
    # changes nothing when absent.  The suffix-corrected candidate is the
    # codec's own naming rule (a .pdb source travels as <stem>.xyz).
    for candidate in (base / src.name,
                      (base / src.name).with_suffix(codec.GEOMETRY_SUFFIX),
                      src):
        if candidate.is_file():
            struct = codec.load(candidate)
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
                     sweep=None, pins=None, translation=None,
                     target: Optional[str] = None,
                     pipeline_log: bool = False) -> List[Path]:
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

    ``pipeline_log`` writes a step-by-step record of what each step received,
    decided and produced, beside this prep's ``STAGE-PLAN.md``. Off by
    default: it is an observer of the pipeline, never a step in it, and no
    generated artifact differs either way (`script-preparation.md` § 4.5).

    Returns the per-job directories. Raises :class:`PrepError`.
    """
    from ..pipeline_log import PipelineLog, config_rows
    from ..resolve import ResolveError, resolve
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..template import template_path as _template_path
    # The container spelling is materialize's (the naming authority): ONE
    # function places the sweep's record and the trials' directories, so the
    # two can never disagree (A-1/A-2).  Imported once for the whole function
    # -- the log's own home is the same container, and a second import site
    # would be a second chance to spell it differently.
    from .materialize import bench_container
    from .shape import Shape

    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"calculation folder not found: {base}")
    desc = base / TASK_FILENAME
    if not desc.is_file():
        raise PrepError(
            f"no {TASK_FILENAME} in {base}. `prep` turns a DESCRIPTION into a "
            f"runnable directory; write one first with `jobset init`.")

    # ---- 1. resolve the machine ---------------------------------------- #
    environment = _environment_for(base, target)

    # ---- 2. resolve the parameters ------------------------------------- #
    task = read_task(desc)
    # THE LOG OPENS HERE, and not before: its NAME carries the stage token, and
    # the token needs the description.  Everything step 1 did is still in hand,
    # so nothing is lost by writing that phase a moment later than it ran.
    #
    # It is not closed on a refusal and does not need to be: every line is
    # flushed as it is written, so a prep that dies leaves a file ending at the
    # step that refused -- which is the step a reader is looking for.
    log = None
    if pipeline_log:
        _token = token_for(task, stage)
        _record = (base / bench_container(Shape.named(task.shape), _token)
                   if sweep is not None else base)
        _record.mkdir(parents=True, exist_ok=True)
        log = PipelineLog.open(_record, label=task.label, token=_token,
                               engine=task.engine, shape=task.shape)
        log.phase("STEP 1 · MACHINE — where this job will run")
        log.received("calculation", str(base))
        log.received(TASK_FILENAME, f"{task.label} · {task.engine} · "
                                    f"{task.shape} · "
                                    f"{len(task.stages or ())} stage(s)")
        for _group, _line in _environment_rows(environment):
            log.produced(_group, _line)
        # WHICH FILE supplied each execution setting -- through the ONE
        # formatter that already answers this, not a second table of the same
        # facts.  It is also the security boundary: `config_provenance`
        # publishes only the sections marked provenance-safe
        # (`configuration.md` § 4), so nothing else may be printed here.
        from ..runtime_config import config_provenance, format_provenance
        log.produced("config", "which file supplied each setting")
        log.text(format_provenance(config_provenance(project_dir=base)))
    # THE one place this name is formed (`template.template_path`).  Six
    # call sites spelled it in two incompatible ways until 2026-08-17.
    template_path = _template_path(base, task.label)
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
    if log is not None:
        log.phase("STEP 2 · RESOLVE — the values for this rung")
        log.received(template_path.name, f"{len(pset[0].provenance)} fields")
        log.received("stage", pset.stage or "(no ladder)")
        log.received("allocation", _flat_resources(allocation or Resources()))
        for _el in pset:
            _rows = config_rows(_el.values, _el.provenance,
                                _el.render_config())
            _decided = [r for r in _rows if r[2] != "template"]
            log.step(f"{_el.label} — what this rung decided")
            for _n, _v, _s in _decided:
                log.chose(_n, _v, _s)
            log.step(f"{_el.label} — the rest, as the template declares")
            for _n, _v, _s in _rows[len(_decided):]:
                log.chose(_n, _v, _s)
        log.produced("ParameterSet", f"{len(pset)} element(s) -> spec_for")

    # ---- 3. render the deck(s) ----------------------------------------- #
    struct = _structure_for(task, base)
    # The DATA FILES the engine will open, before any deck is written: a
    # missing pseudopotential is a run that cannot start, and finding that out
    # here costs a second (project-layout.md § 2.6).  Idempotent -- what is
    # already in the folder is left alone.  The elements come from the
    # structure, which has just been checked against the description's witness.
    if seam.provide_data is not None:
        with _user_error_as_prep(), _calling(
                "provide_data", engine=task.engine, log=log):
            seam.provide_data(struct, pset[0].render_config(), base)
    token = token_for(task, pset.stage)
    jobs: List[Job] = []
    for element in pset:
        stem = f"{element.label}_{token}" if token else element.label
        script = f"{stem}{seam.suffix}"
        # The deck is rendered from values ⊕ THIS element's allocation, so it
        # records the rank count it actually assumed.  Rendering from the
        # values alone emits `mpi_np auto` and the launch check then refuses a
        # deck that `prep` itself just made -- which is how this was found.
        # The stage's artifact TOKEN reaches the emitter here, as a RENDER
        # ARGUMENT (C7).  It feeds three names -- the deck, the engine's own
        # log, and the molwatch log -- and leaving it unset made two rungs of
        # one calculation write to a single `<label>.molwatch.log`.  `prep`
        # holds the StageRef, so `prep` says the word; no config field carries
        # it, for either engine (`stages.md` § 1.1).
        cfg = element.render_config()
        if element.is_trial:
            # The deck's OWN identity line carries the trial label -- this,
            # not the filename, is what keys SIESTA's warm files away from
            # the real run's (project-layout.md § 2.3.2).
            with _calling("relabel", engine=task.engine,
                          where=element.label, log=log):
                cfg = seam.relabel(cfg, element.label)
            # The forced cold has ONE setter -- the measurement pin
            # (`_MEASUREMENT_PINS`, resolved with provenance "pin") --
            # and ONE verifier, the submission door
            # (`agreement.check_trial_starts_cold`; user-settled
            # 2026-08-21: prep bakes the intent, submission determines
            # the actual state).  A hard replace stood here as a second,
            # provenance-invisible setter until then (Q6a).
        # The stage's artifact token is a RENDER ARGUMENT (C7, 2026-08-12):
        # `prep` holds the StageRef, so `prep` says it, per call -- the
        # config field that used to carry it is gone, and the emitter never
        # learns the word (engines/stages.md § 1.1).
        # The render's refusals (missing pseudos above all) are user-fixable
        # and translate to PrepError -- see _user_error_as_prep (A8).
        # STEP 3, WHOLE, IN ONE CALL: validate the settings, render the deck,
        # write it through the one writer (which keeps the reader's USER-CUSTOM
        # block), then read the file back and refuse one that does not say what
        # it was meant to say.  The conductor says WHEN; the framework owns the
        # order (`script-preparation.md` § 4.3).
        if log is not None:
            log.phase(f"STEP 3 · DECK — {script}")
            log.received("config", f"{type(cfg).__name__}"
                                   + (f", stage_token={token}" if token else "")
                                   + (f", trial {element.label}"
                                      if element.is_trial else ""))
        with _user_error_as_prep():
            with _calling("spec_for", engine=task.engine,
                          where=script, log=log):
                spec = seam.spec_for(struct, cfg,
                                     stage_token=(token or None),
                                     calculation=task.calculation)
            _sc.prepare_deck(spec, struct, cfg, base / script, log=log)
        if seam.sibling_artifacts is not None:
            with _calling("sibling_artifacts", engine=task.engine,
                          where=script, log=log):
                seam.sibling_artifacts(struct, cfg, base / script)
        with _calling("label_of", engine=task.engine, log=log):
            _label = seam.label_of(cfg)
        _seed_trajectory_log(struct, cfg, base, engine=task.engine,
                             label=_label, token=(token or None))
        if log is not None:
            log.step("what this deck's text PROMISES, kept")
            log.produced("sibling_artifacts",
                         "written" if seam.sibling_artifacts is not None
                         else "nothing (W5)")
            log.produced("trajectory log",
                         "seeded" if getattr(cfg, "write_molwatch_log", False)
                         else "not asked for")
        jobs.append(_job_for(element, script, task, pset.stage, seam,
                             base, log=log))

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
                shared=_shared_for(base, seam,
                                   engine=task.engine, log=log),
                jobs=jobs)
    if kind == "sweep":
        record_dir = base / bench_container(Shape.named(task.shape), token)
        record_dir.mkdir(parents=True, exist_ok=True)
    else:
        record_dir = base
        js = _merge_run_jobset(
            base / JOBSET_FILENAME, js,
            # The CURRENT ladder bounds what the merge keeps.
            # (`read_task` refuses a description without stages and
            # `Task` refuses an empty tuple, so the stage-less fallback
            # arms that stood here were unreachable -- U6 close.)
            ladder=frozenset(s.name for s in task.stages))
    js.write(record_dir / JOBSET_FILENAME)
    if log is not None:
        log.phase("FLOOR 3 · THE JOB-SET — what was declared to the runner")
        log.received("kind", kind)
        for _j in js.jobs:
            log.produced(_j.name, f"{_j.script}  "
                                  f"{_flat_resources(_j.resources)}")
        log.produced(JOBSET_FILENAME, str(record_dir / JOBSET_FILENAME))
    # The allocation is NOT passed on: every job already carries its own
    # resolved resources, per element (generator.md § 5).  Passing it made
    # prep_jobset re-apply the BASE allocation over every job — the review's
    # "stomp": each trial's wrapper rendered with the base rank count
    # instead of its own translated G·K.
    dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=record_dir, log=log)
    if log is not None:
        log.close()
    return dirs


def _merge_run_jobset(path: Path, new: JobSet,
                      ladder: Optional[frozenset] = None) -> JobSet:
    """The root ``job-set.json`` is the RUN's whole plan: each stage's prep
    updates its OWN row and leaves the others standing.

    Until 2026-08-12 every prep wrote only its own elements, so `prep run
    tight` erased `coarse` from floor 3 — breaking the status rollup and,
    worse, the ``--from`` pair rule: with the source job gone, `warm_carry`
    read the pair as unverified and silently withheld ``.CG``
    (`project-layout.md` § 2.3.4 row 3).

    ``ladder`` is the CURRENT task's stage-name set, and it bounds what is
    kept (2026-08-12): a row is standing only while its stage is still
    on the ladder — a stage removed from ``task.json`` used to stay in the
    plan forever, its deck gone.  A set whose NAME differs is a different
    calculation's plan and is replaced outright, same as the legacy cases.
    """
    if not path.is_file():
        return new
    try:
        old = JobSet.load(path)
    except ValueError:
        return new              # unreadable or legacy: replaced outright
    if old.kind != "ladder":
        return new              # a pre-container sweep leftover: replaced
    if old.name != new.name:
        return new              # a renamed calculation: the old plan is
                                # another name's plan, not rows to keep
    fresh = {j.name for j in new.jobs}
    kept = [j for j in old.jobs if j.name not in fresh
            and (ladder is None or j.name in ladder)]
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


def _flat_resources(resources) -> str:
    """One allocation as one line — the fields it actually carries.

    ``None`` means *not asked for* and is left out rather than printed as a
    null: a line of ``domain=None, time=None, gres=None`` is noise in a file
    whose whole value is that a person can read it.
    """
    return ", ".join(f"{k}={v}" for k, v in
                     dataclasses.asdict(resources).items() if v is not None) \
        or "(nothing asked for)"


def _environment_rows(environment) -> "List[tuple]":
    """Floor 1's answer as ``[(group, one line)]``.

    ONE ROW PER GROUP, not one line for the whole record: the probe's answer
    nests (topology, site, source), and flattening it produced a single line
    of nested dict reprs -- unreadable, in the one file whose whole claim is
    that a person can read it.
    """
    if environment is None:
        return [("environment", "none — this machine was not probed")]
    raw = (dataclasses.asdict(environment)
           if dataclasses.is_dataclass(environment)
           else environment if isinstance(environment, dict)
           else {"environment": environment})
    scalars, rows = [], []
    for k, v in raw.items():
        if v is None or v == {} or v == []:
            continue
        if isinstance(v, dict):
            # A group whose every member is None was PROBED and came back
            # empty (a workstation has no partition, no QOS, no account).
            # A bare heading with nothing after it says less than no row.
            line = ", ".join(f"{a}={b}" for a, b in v.items() if b is not None)
            rows.append((k, line or "nothing — probed, and this machine "
                                    "has none"))
        elif isinstance(v, (list, tuple)):
            rows.append((k, ", ".join(str(x) for x in v)))
        else:
            scalars.append(f"{k}={v}")
    return ([("environment", ", ".join(scalars) or "(no scalar facts)")]
            + sorted(rows))


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


def token_for(task, stage_name: Optional[str]) -> str:
    """This stage's ``<NN>_<name>`` — the ONE namer (decision 27).

    ``NN`` is the stage's place in the **full** ladder, so disabling one leaves
    a gap rather than renumbering what follows: renumbering would hand an
    existing output to a stage that did not produce it.

    Public since 2026-08-13: the CLI's container/underway surfaces need the
    same answer, and reaching in for a private name was the re-derivation
    habit the final review's C-c row names.
    """
    if not stage_name:
        return ""       # asked without naming a rung; every ladder has one
    from ..identity import StageRef
    # The ordinal rule is stated ONCE (StageRef.ladder -- decision 28's
    # pre-produce arm); this function reads the ref and spells the token.
    for ref in StageRef.ladder([s.name for s in task.stages]):
        if ref.name == stage_name:
            return ref.token
    # Unreachable through prep_calculation -- resolve._stage_of already
    # refused an unknown stage -- and LOUD rather than "" if a future caller
    # reaches it another way: an empty token would silently drop the stage
    # from every artifact name (job-contracts.md § 6.3).
    raise PrepError(f"stage {stage_name!r} is not in this description's "
                    f"ladder: {', '.join(s.name for s in task.stages)}.")


def _job_for(element, script: str, task, stage_name: Optional[str],
             seam: EngineSeam, base_dir=None, log=None) -> Job:
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

    with _calling("warm_for", engine=task.engine, where=name, log=log):
        warm = seam.warm_for(element.label, element.values,
                             task.calculation, base_dir)
    with _calling("traits_for", engine=task.engine, where=name, log=log):
        traits = seam.traits_for(element.values)
    return Job(name=name, script=script, resources=element.resources,
               warm=warm, traits=traits, point=dict(element.point))


def _siesta_shared_package(base: Path) -> List[str]:
    """SIESTA's shared package: the pseudopotentials it put in the folder.

    The same suffix ``_siesta_provide_pseudos`` copies, named by the engine
    that copied them (`script-preparation.md` § 4, the data-files step).
    """
    return sorted(p.name for p in base.glob("*.psml"))


def _shared_for(base: Path, seam: "EngineSeam" = None, *, engine: str = "",
                log=None) -> List[str]:
    """The static package every job links (`project-layout.md` § 2.1).

    **Asked of the engine, not guessed from the folder.**  An engine that puts
    no data files in has an empty package, and that is an answer rather than an
    accident of which suffix the glob happened to name.
    """
    if seam is None or seam.shared_package is None:
        return []
    with _calling("shared_package", engine=engine, log=log):
        return list(seam.shared_package(base))

__all__ = ["prep_calculation", "prep_jobset", "PrepError", "resolve_target"]
