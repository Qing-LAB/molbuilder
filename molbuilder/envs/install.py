"""Idempotent installer for the conda envs described by recipes.

``molbuilder envs install <name>`` is a thin wrapper around five
phases per recipe:

  1. ``conda create -n <env> -c <ch1> [-c <ch2>] ... <pkg1> <pkg2> ...``
  2. pip, from the recipe's :class:`PipPackage` records.  The PLAIN
     ones (default index, required, no forcing) go in one
     ``conda run -n <env> python -m pip install <pkgs>``; each package
     that needs its own terms gets its own step -- a forced reinstall,
     a non-fatal step for an optional one, or an install from a
     recorded source with the indexed build as a declared fallback.
  3. Each tuple in ``recipe.extra_steps`` dispatched via
     ``conda run -n <env> <argv>``.
  4. **(source-build recipes only)** ``builds.run_build_spec`` runs
     the recipe's :class:`BuildSpec`: clone + cmake + install for each
     component, with sentinel-resume.  Activate.d / deactivate.d
     hooks are rendered into the env's ``etc/conda/`` tree.
  5. Verify -- an ordinary step like the other four, built by
     :func:`verify_step_for` and run through the same door.  That
     function lives HERE and :mod:`molbuilder.envs.doctor` imports it;
     this line claimed the reverse ("re-uses molbuilder.envs.doctor")
     until 2026-09-13, which is the direction the dependency ran
     before the migration.

The installer never deletes an existing env; if the env already
exists, phases 2-4 still run (so installing twice doesn't break --
``pip install`` and the extra steps are idempotent in practice, with
one DELIBERATE exception: a ``PipPackage`` marked ``force`` reinstalls
every time, because its version cannot prove it is the declared build
and a no-op would leave the wrong one in place; the build_spec
executor has its own sentinel-based resume).  That makes
``install`` safe to re-run when, e.g., the recipe gains a new pip
dependency or the user wants to rebuild from a new SIESTA tag.

Phases are reported as a list of :class:`InstallStep` instances so
the CLI can surface failures with the exact command that broke.
``--dry-run`` returns the step plan without executing.

This module performs side effects -- the only one in the package
besides :mod:`molbuilder.envs.builds`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..diagnostics import Capabilities, get_capabilities, reset_capabilities
from . import builds as _builds
from .builds import conda_run_argv


from .recipes import effective_name, PipPackage, Recipe


class StepRole(str, Enum):
    """What a step is FOR.

    The runner's branches key on THIS, never on ``label``.  They used to key
    on the label -- ``== "conda create"`` decided whether the env-state
    machine ran at all, and ``== "verify"`` decided which phase a step
    belonged to -- while ``label``'s own docstring called it the CLI's
    per-step header text.  Renaming a label for clarity silently disabled the
    create-skip logic, and the install would then attempt `conda create` on
    an existing env every run.

    It is also residue of the shape the migration replaced: when each phase
    was a hand-written block, the phase was implicit in code POSITION; one
    loop has to ask what kind of step it is holding, and the only thing
    available to ask was a display string.
    """

    CREATE = "create"      #: the env itself -- needs no prefix, there is none yet
    REMOVE = "remove"      #: the env itself, taken away -- needs no prefix either
    PACKAGES = "packages"  #: a conda or pip install
    EXTRA = "extra"        #: a recipe-declared dispatch into the env
    VERIFY = "verify"      #: runs after any source build
    BUILD = "build"        #: adapted from `builds.py`'s own executor


#: How much of a step's combined output is kept.  The streamed copy
#: already reached the user's terminal; this is the excerpt the CLI
#: recaps and the web report stores, so it is an excerpt on purpose.
# One home, in the layer that produces the output (`builds.OUTPUT_LIMIT`).
# Re-exported here because `InstallStep.output` is documented against it.
OUTPUT_LIMIT = _builds.OUTPUT_LIMIT


@dataclass(frozen=True)
class InstallStep:
    """One command in the install plan.

    Attributes
    ----------
    label
        Short human tag (``"conda create"``, ``"pip install numpy"``,
        ``"extra"``, ``"verify"``), used for the per-step header line and
        the recap.  DISPLAY ONLY -- nothing branches on it; see
        :class:`StepRole`, which is what the runner asks.
    role
        What the step is for.  The runner and the CLI branch on this.
    argv
        Full command argv as it would be invoked.  For dry runs the
        CLI prints this verbatim.
    returncode
        Process exit code, or ``None`` for a step that hasn't run
        (either because it was a dry run, or because an earlier step
        failed and we short-circuited).
    output
        Combined stdout+stderr, trimmed to :data:`OUTPUT_LIMIT`; empty
        for not-run steps.
    fallbacks
        Alternative argvs, tried in order when ``argv`` fails.  A pip
        package that declares ``fallback_to_index`` renders as one step
        whose fallback is the indexed build: prefer the recorded source,
        accept the index rather than lose the package.  The first
        success wins and the step is RECOVERED -- reported under the argv
        that actually ran, never under the one that failed.
    fatal
        When ``False``, a non-zero exit is reported and the install
        CONTINUES.  Optional packages install this way -- one failing
        wheel must not take the env down, which is what
        ``PipPackage.optional`` promises.
    ignore_exit_code
        The exit code is not the verdict.  Some tools report through
        their output and exit non-zero anyway (``tleap`` does), so for
        them the substring check IS the verification.
    expect_contains
        Success also requires this substring in the output.  A tool can
        exit 0 while the thing we asked about is missing, and for a
        verify step that is the failure worth catching.
    """
    label: str
    argv: Tuple[str, ...]
    role: StepRole = StepRole.PACKAGES
    fallbacks: Tuple[Tuple[str, ...], ...] = ()
    fatal: bool = True
    ignore_exit_code: bool = False
    expect_contains: Optional[str] = None
    returncode: Optional[int] = None
    output: str = ""
    #: What became of this step.  Every step in an :class:`InstallResult`
    #: carries one -- dispatched through :func:`run_step`, or decided
    #: without dispatch by :func:`_undispatched`.  ``None`` only on a
    #: step that is still a PLAN (what ``--dry-run`` prints).
    outcome: Optional["Outcome"] = None

    def accepts(self, returncode: Optional[int], output: str) -> bool:
        """Did this attempt satisfy the step?

        The exit code is the usual verdict; two recipes need more, and
        both rules ride on the STEP rather than being read off the
        recipe at execution time.  That is what let verify stop being a
        phase with its own loop: the runner needs no knowledge of which
        phase it is serving.

        A process that never launched (``returncode is None``) is never
        accepted -- ignoring an exit code is not ignoring a missing
        process.
        """
        if returncode is None:
            return False
        if not self.ignore_exit_code and returncode != 0:
            return False
        if self.expect_contains and self.expect_contains not in output:
            return False
        return True


@dataclass(frozen=True)
class InstallResult:
    """Outcome of one ``install`` invocation."""
    recipe: Recipe
    effective_name: str
    steps: Tuple[InstallStep, ...]
    succeeded: bool
    build_result: Optional["_builds.BuildResult"] = None


class Outcome(str, Enum):
    """What became of one step.  Five states, and they are exhaustive.

    The runner used to decide this with nested conditions -- is the
    return code zero, is it None because the process never launched, are
    there alternatives, is the step optional -- and every time that
    tangle was edited a branch went missing.  Twice: a launch failure
    skipped the optional check and aborted an install it should have
    survived, and a recovered step was recorded under the command that
    had failed.

    Naming the states makes those omissions impossible to write.  There
    is one transition rule and it fits in a sentence: try each argv in
    turn; the first success is OK (or RECOVERED if it was not the first
    attempt); if none succeed the step is DEGRADED when optional and
    FAILED when not.
    """

    OK = "ok"                 #: ran, exit 0, first attempt
    RECOVERED = "recovered"   #: a declared alternative succeeded
    DEGRADED = "degraded"     #: every attempt failed; the step is optional
    FAILED = "failed"         #: every attempt failed; the step is required
    SKIPPED = "skipped"       #: deliberately not attempted

    @classmethod
    def decide(cls, *, ok: bool, first_attempt: bool,
               fatal: bool) -> "Outcome":
        """The transition rule, and the only place it is written.

        :func:`run_step` decides a dispatched step with it, and the
        build adapter decides a build phase's result with it -- which is
        how build steps stopped arriving with no outcome at all.  A step
        decided WITHOUT being dispatched does not come through here; see
        :func:`_undispatched`.
        """
        if ok:
            return cls.OK if first_attempt else cls.RECOVERED
        return cls.FAILED if fatal else cls.DEGRADED

    @property
    def is_success(self) -> bool:
        """Did the env end up with what this STEP was for?

        False for ``DEGRADED``: an optional package that could not be
        installed is honestly absent.  Do not use this to decide the
        INSTALL's verdict -- that is :attr:`stops_the_install`, and
        ``DEGRADED`` is exactly where the two answers differ.  The
        difference is the whole point of ``optional``.
        """
        return self in (Outcome.OK, Outcome.RECOVERED, Outcome.SKIPPED)

    @property
    def stops_the_install(self) -> bool:
        """Must everything after this step be abandoned?

        Only ``FAILED``.  ``DEGRADED`` is a step that failed and was
        allowed to; ``SKIPPED`` never ran.
        """
        return self is Outcome.FAILED

    @property
    def word(self) -> str:
        """The ONE name this outcome is printed under.

        There were two.  The live line said ``UNAVAILABLE -- optional,
        continuing`` while the recap of the same step said ``degraded``, and
        ``OK via the declared alternative`` was ``recovered`` a few lines
        later -- one run, two vocabularies, and a reader comparing them has to
        work out that they are the same fact (H5).  The word lives on the
        state, so a surface cannot invent a second one.
        """
        return self.name

    @property
    def note(self) -> str:
        """Why this outcome is not simply OK -- for the live line, which has
        room for it.  Empty where the word says everything."""
        return {
            Outcome.RECOVERED: "via the declared alternative",
            Outcome.DEGRADED: "optional, continuing",
        }.get(self, "")


def run_step(
    step: InstallStep,
    *,
    prefix: Optional[str] = None,
    sink=None,
    timeout: int = 3600,
) -> InstallStep:
    """Run one step to a settled :class:`Outcome`.

    THE ONE DOOR every dispatched command goes through -- the installer
    and ``repair`` both, so neither can learn about alternatives or
    optionality without the other.  They were separate procedures until
    2026-09-11, and had already drifted: repair issued a bare pip
    command that knew nothing of a package's fallbacks or its flags.

    Returns a copy of ``step`` carrying the outcome, the argv that
    ACTUALLY ran (not the one that was tried first), its exit code and
    its trimmed output.
    """
    attempts = [tuple(step.argv), *(tuple(a) for a in step.fallbacks)]
    rc: Optional[int] = None
    combined = ""
    ran = attempts[0]
    # The environment every attempt runs in, primary or alternative: host
    # leakage stripped, temp and pip cache inside the prefix.  It used to be
    # `run_step`'s `env` parameter, which no caller ever passed -- so no pip
    # step was ever sanitised -- and the temp dirs were set inside a shell
    # string that only existed on the workaround path.
    step_env = _builds.env_for_step(prefix)

    for n, argv in enumerate(attempts):
        if n and sink is not None:
            sink.write(f"    {step.label}: previous source was not "
                       f"accepted ({_why_rejected(step, rc, combined)}); "
                       f"trying the declared alternative\n")
            sink.flush()
        rc, combined = _builds.dispatch_into_env(
            argv, prefix, env=step_env, sink=sink, timeout=timeout)
        ran = argv
        if step.accepts(rc, combined or ""):
            return replace(
                step, argv=ran, returncode=rc,
                output=(combined or "")[:OUTPUT_LIMIT],
                outcome=Outcome.decide(ok=True, first_attempt=(n == 0),
                                       fatal=step.fatal))

    # `rc is None` -- the process never launched -- is not a special
    # case here.  It is simply "not accepted", which is the whole reason
    # the branch that used to treat it specially could forget `fatal`.
    return replace(
        step, argv=ran, returncode=rc,
        output=_rejection_output(step, rc, combined),
        outcome=Outcome.decide(ok=False, first_attempt=True,
                               fatal=step.fatal))


def _why_rejected(step: InstallStep, rc: Optional[int],
                  combined: str) -> str:
    """One phrase naming why an attempt was not accepted."""
    if rc is None:
        return "could not launch"
    if step.expect_contains and step.expect_contains not in (combined or ""):
        return f"missing `{step.expect_contains}`"
    return f"rc={rc}"


def _rejection_output(step: InstallStep, rc: Optional[int],
                      combined: str) -> str:
    """What a rejected step records.

    A step rejected for its OUTPUT needs the reason spelled out: the
    command may well have exited 0, and an operator reading "rc=0" under
    a failure would have nothing to go on.
    """
    text = combined or ""
    if rc is None:
        return (text or "step failed to launch")[:OUTPUT_LIMIT]
    out = text[:OUTPUT_LIMIT]
    if step.expect_contains and step.expect_contains not in text:
        out += (f"\n(missing expected substring "
                f"`{step.expect_contains}`)")
    return out


def pip_argv(conda: str, env_name: str, *specs: str,
             flags: Sequence[str] = ()) -> Tuple[str, ...]:
    """The one shape of a pip command dispatched into an env.

    ``python -m pip`` rather than ``pip`` sidesteps the common Ubuntu
    pitfall where ``~/.local/bin/pip`` precedes the env's own on PATH.

    It lives here because it had been WRITTEN OUT three times -- once in
    the planner and twice in `repair` -- and the copies had already
    drifted: repair knew nothing about a package's alternatives, so a
    fix that taught the installer about fallbacks left repair behind.
    """
    return conda_run_argv(conda, env_name,
                          "python", "-m", "pip", "install", *flags, *specs)


def pip_step_for(pkg: PipPackage, conda: str, env_name: str) -> InstallStep:
    """The step that installs ONE package, on its own terms.

    Everything that makes a package special is read off the record here
    and nowhere else, so `install` and `repair` cannot disagree about
    what installing it means:

    * ``force``             -> ``--force-reinstall --no-deps``;
    * ``optional``          -> a non-fatal step;
    * ``fallback_to_index`` -> the indexed build as a declared
      alternative, FLAGLESS.  Its job is to make sure the package
      exists, never to replace what is already there -- with the force
      flag an unreachable source during an unrelated install would
      overwrite a good tree with a worse one.
    """
    fallbacks: Tuple[Tuple[str, ...], ...] = ()
    if pkg.fallback_to_index:
        fallbacks = (pip_argv(conda, env_name, pkg.index_spec()),)
    return InstallStep(
        label=f"pip install {pkg.name}",
        argv=pip_argv(conda, env_name, pkg.spec(),
                      flags=pkg.install_flags()),
        fallbacks=fallbacks,
        fatal=not pkg.optional,
    )


def conda_argv(conda: str, subcommand: str, env_name: str,
               channels: Sequence[str], specs: Sequence[str]
               ) -> Tuple[str, ...]:
    """The one shape of a conda command line.

    ``create`` and ``install`` differ only in the subcommand, so they are one
    shape.  Writing it out twice is how the channel flags came to sit BEFORE
    the specs in the planner and AFTER them in `repair`.
    """
    argv: List[str] = [conda, subcommand, "-n", env_name, "-y"]
    for ch in channels:
        argv.extend(["-c", ch])
    argv.extend(specs)
    return tuple(argv)


def create_step_for(recipe: Recipe, conda: str,
                    env_name: str) -> InstallStep:
    """The step that creates the env, and how it degrades.

    OPTIONALITY MEANS SOMETHING DIFFERENT TO A SOLVER.  pip installs one
    package at a time, so an optional pip package is its own non-fatal STEP
    (`pip_step_for`).  conda solves everything at once: an optional conda
    package cannot be its own step without paying a SECOND solve, and a
    second solve may legitimately change versions for packages the first one
    already placed.  So optionality here is expressed as an ATTEMPT -- the
    full solve first, and if it fails, the same solve without the optional
    specs.  The env lands `RECOVERED`: created, minus something the recipe
    said it could live without.

    ONE DEGRADATION STEP, NOT A SEARCH.  With n optional specs, "find the
    largest subset that still solves" is 2**n solves at minutes each, so the
    rule is all of them or none of them.  Nothing is lost silently: the
    audit then names exactly which optional packages are absent, with the
    recipe's `reason` beside each, and `repair --include-optional` installs
    them one at a time -- paying the second solve there because the operator
    asked for it.

    A recipe with no optional conda package gets no fallback, so this is
    byte-identical to what the planner emitted before.
    """
    required = tuple(p.spec for p in recipe.conda_packages if not p.optional)
    fallbacks: Tuple[Tuple[str, ...], ...] = ()
    if len(required) != len(recipe.conda_packages):
        fallbacks = (conda_argv(conda, "create", env_name,
                                recipe.channels, required),)
    return InstallStep(
        label="conda create",
        role=StepRole.CREATE,
        argv=conda_argv(conda, "create", env_name,
                        recipe.channels, recipe.conda_specs),
        fallbacks=fallbacks,
    )


def conda_step_for(specs: Sequence[str], recipe: Recipe, conda: str,
                   env_name: str) -> InstallStep:
    """One batched ``conda install`` for packages already missing.

    `repair`'s half of the conda story.  Batched rather than per-package
    because each conda command is a whole solve; the operator has already
    accepted that second solve by running `repair`.
    """
    return InstallStep(
        label="conda install",
        role=StepRole.PACKAGES,
        argv=conda_argv(conda, "install", env_name, recipe.channels, specs),
    )


def verify_step_for(recipe: Recipe, conda: str,
                    env_name: str) -> Optional[InstallStep]:
    """The step that checks the env, or ``None`` if the recipe declares none.

    The one place a recipe's verify fields become a step, so the three
    callers that need one -- the planner, `run_install` and `doctor` --
    cannot disagree about what verifying this recipe means.  They used to
    each build the command and then each re-implement the accept rule;
    `doctor` and the installer had already drifted to different output
    limits, and a fourth copy would have been next.

    `verify_ignore_exit_code` / `verify_expect_contains` move onto the
    STEP here.  That is the whole trick: after this, verify is an
    ordinary step and the runner needs no special case for it.
    """
    if not recipe.verify_argv:
        return None
    return InstallStep(
        label="verify",
        role=StepRole.VERIFY,
        argv=conda_run_argv(conda, env_name, *recipe.verify_argv),
        ignore_exit_code=recipe.verify_ignore_exit_code,
        expect_contains=recipe.verify_expect_contains,
    )


def pip_steps_for(recipe: Recipe, conda: str,
                  env_name: str) -> List[InstallStep]:
    """Every pip step for a recipe -- ONE translator, as § 4.2 already says.

    The batch and the per-package steps were built inline in the planner, so
    *what a plain pip install means* lived in one place and *what a special one
    means* in another, and § 4.2's pseudocode named a `pip_steps_for` that did
    not exist.  A future per-step policy would have had to be written twice.
    """
    plain = [p for p in recipe.pip_packages if p.is_plain()]
    steps: List[InstallStep] = []
    if plain:
        steps.append(InstallStep(
            label="pip install",
            argv=pip_argv(conda, env_name, *(p.spec() for p in plain)),
        ))
    steps.extend(pip_step_for(pkg, conda, env_name)
                 for pkg in recipe.pip_packages if not pkg.is_plain())
    return steps


def extra_steps_for(recipe: Recipe, conda: str,
                    env_name: str) -> List[InstallStep]:
    """What an ``extra_steps`` entry MEANS -- one place, per § 4.2.

    It had no translator at all: the planner built the step, so there was
    nowhere to say what an extra step is.
    """
    return [InstallStep(label="extra", role=StepRole.EXTRA,
                        argv=conda_run_argv(conda, env_name, *extra))
            for extra in recipe.extra_steps]


def remove_step_for(env_name: str, conda: str, *,
                    prefix: Optional[str] = None) -> InstallStep:
    """Removing an env is a STEP, like everything else the installer does.

    `env-framework.md` § 5.4 records it as a defect rather than an exception:
    `--clean`'s wipe was a bare `subprocess.run`, so it carried no `Outcome`
    and no line in the result the verdict is derived from (§ 5.3) -- and it was
    an edge path for one GPU recipe until it was opened to every recipe, which
    promoted a private dispatch to the door `installation.md` calls *"the one
    door"* for wiping any env.

    Addressed by name, with the directory as the FALLBACK when the caller
    knows one: an ORPHAN -- a directory the registry does not list -- is
    exactly the env ``--clean`` is recommended for, and ``env remove -n``
    finds nothing to remove there (K-L9, 2026-09-13).  ``prefix`` is the
    probe's reading (`EnvState.prefix`); the pure plan has none and stays by
    name.  The caller has already refused the case where the name is the env
    we are running from (`installation.md` M5).

    **Its own role, and that is the fix** (2026-09-13).  It carried
    ``StepRole.CREATE`` -- the one role the runner exempts from needing a
    prefix -- and every CREATE-role step goes through `_create_decision`
    first, which answers *"already exists; skipping create"* for any env that
    is PRESENT.  So the removal was skipped for exactly the envs it exists to
    remove, the create after it was skipped for the same reason, and
    ``--clean`` was a plain re-install that printed a wipe banner.  The one
    test on the path faked the env absent, the single state the skip cannot
    fire in.
    """
    return InstallStep(
        label=f"remove env {env_name}",
        role=StepRole.REMOVE,
        argv=(conda, "env", "remove", "-n", env_name, "-y"),
        fallbacks=(((conda, "env", "remove", "--prefix", prefix, "-y"),)
                   if prefix else ()),
    )


def _plan(recipe: Recipe, env_name: str, conda: str) -> List[InstallStep]:
    """Build the step list without running anything."""
    steps: List[InstallStep] = []

    # Phase 1: conda create.
    steps.append(create_step_for(recipe, conda, env_name))

    # Phase 2: pip install.
    #
    # PLAIN packages (default index, no force, required) batch into ONE
    # call -- the shape and the speed the registry had before sources
    # existed.  Anything that needs different treatment gets its own
    # step, because the treatment IS per-package:
    #
    #   * ``force``    -> --force-reinstall --no-deps, for a package
    #                     whose version cannot prove it is current;
    #   * ``optional`` -> a NON-FATAL step, so one unavailable wheel
    #                     degrades the env instead of aborting it;
    #   * ``source``   -> installed from the recorded URL, with the
    #                     indexed build as a fallback when the record
    #                     says the base is still required.
    steps.extend(pip_steps_for(recipe, conda, env_name))

    # Phase 3: extra dispatch-into-env steps.
    steps.extend(extra_steps_for(recipe, conda, env_name))

    # Phase 4: verify (only if the recipe declares one).
    verify = verify_step_for(recipe, conda, env_name)
    if verify is not None:
        steps.append(verify)

    return steps


def plan_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
    clean: bool = False,
) -> Tuple[str, List[InstallStep]]:
    """Return ``(effective_name, steps)`` for the recipe.

    Pure planner -- does not run anything.  So ``--dry-run`` prints exactly
    what would happen, and a test can read the sequence without a subprocess,
    a fake binary or a temporary environment.

    ``clean`` puts the env removal at the FRONT of the plan, where it belongs:
    the wipe is a step (§ 5.4), and a step the surface dispatched on the side
    was a step ``--dry-run`` could not show and a reader could not check the
    order of.  Now the order IS the plan.

    Raises
    ------
    RuntimeError
        When no ``conda`` binary is reachable; without it the plan's
        commands would be unrunnable so building one is pointless.
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = effective_name(recipe, caps)
    steps = _plan(recipe, effective, caps.conda_binary)
    if clean:
        steps.insert(0, remove_step_for(effective, caps.conda_binary))
    return effective, steps


def _prefix_under_envs_dirs(info: Mapping[str, Any],
                            env_name: str) -> Optional[Path]:
    """The directory ``<envs_dir>/<name>`` that exists, from the manager's
    own search list -- or ``None``.  The one walk; the prefix resolver and
    the state probe each spelled it until 2026-09-13 (K-D6)."""
    for envs_dir in info.get("envs_dirs", []) or []:
        candidate = Path(envs_dir) / env_name
        if candidate.is_dir():
            return candidate
    return None


def _env_prefix(env_name: str, conda_binary: str) -> Optional[str]:
    """Return ``$CONDA_PREFIX`` for the named env, or ``None`` if not found.

    The prefix is what every step is addressed by (`installation.md` M2) and
    what the activation fallback needs, so a failure here means a step that
    cannot be dispatched at all -- which is reported as such rather than
    guessed around:

    0. **The startup snapshot**, when it knows this env.  `<mgr> env list
       --json` costs 1.2 s on a warm workstation and `doctor` asks this question
       once per recipe, so five redundant registry reads per report was the
       measured cost of not looking at the reading already taken.  A snapshot
       that does not know the env is not evidence of absence -- an env created
       mid-process is absent from it -- so the live tiers still run.
    1. ``<mgr> env list --json`` -- the canonical registry, read fresh.
       One reader, `diagnostics.conda_env_prefixes`.
    2. ``<mgr> info --json``'s ``envs`` array (FULL prefix paths).
       Distinct from envs_dirs (the search list) -- ``envs`` holds
       the actual prefixes the env manager has recorded, which
       catches envs at non-standard locations (mamba's
       ``~/.conda/envs/<name>`` default vs conda's
       ``<conda_root>/envs/<name>``).
    3. ``<mgr> info --json``'s ``envs_dirs`` + filesystem check --
       catches the case where the registry forgot the env (e.g. a
       prior ``conda env remove`` was interrupted, ``conda create``
       failed mid-flight leaving an orphan dir) but the directory IS
       on disk under a known envs_dir.
    """
    # Strategy 0: the reading already taken.
    known = get_capabilities().env_prefix(env_name)
    if known:
        return known
    # Strategy 1: the registry, fresh, through the one reader.
    from ..diagnostics import conda_env_prefixes, manager_info
    fresh = conda_env_prefixes(conda_binary).get(env_name)
    if fresh:
        return fresh
    # Strategies 2 + 3: info --json, read once.
    info = manager_info(conda_binary)
    # Strategy 2: envs (full paths, not the search list).
    for prefix in info.get("envs", []) or []:
        if Path(prefix).name == env_name and Path(prefix).is_dir():
            return prefix
    # Strategy 3: envs_dirs (search list).
    found = _prefix_under_envs_dirs(info, env_name)
    if found is not None:
        return str(found)
    # NO FIFTH STRATEGY, and that is the change (H10).  What stood here
    # derived `<manager root>/envs/<name>` from the binary's own path and
    # guessed at `~/.conda/envs` besides -- `installation.md` M2: the binary's
    # location says where the MANAGER is installed, not where it keeps envs,
    # and on the machines `envs.manager` exists for the two differ.  Every
    # strategy above asks the manager (its registry, its `envs`, its own
    # `envs_dirs`), so reaching here means the manager does not know this env.
    # `None` says so; a guess would answer with a path nothing created.
    return None


# --------------------------------------------------------------------- #
#  Env-state probe -- diagnose conda env state up front                  #
# --------------------------------------------------------------------- #


class EnvPresence(str, Enum):
    """Is this env installable-into?  Five states, exhaustive over the three
    observations `probe_env_state` makes (`env-framework.md` § 2.1)."""

    FRESH = "FRESH"        #: not registered, no directory -- create it
    PRESENT = "PRESENT"    #: registered, and the directory it names is an env
    ORPHAN = "ORPHAN"      #: a directory the registry does not know about
    GHOST = "GHOST"        #: a registry entry whose directory is gone
    BROKEN = "BROKEN"      #: a directory that is not an env


@dataclass(frozen=True)
class EnvState:
    """Result of probing the conda env's current state.

    The conda env can be in one of five states from the install's
    point of view.  We resolve it ONCE at the start of the install
    (before any subprocess work runs) so a partly-broken env doesn't
    cause failures 10 minutes into ``conda create``.

    Attributes
    ----------
    name : str
        The env name we probed.
    listed_in_registry : bool
        ``conda env list --json`` includes the env.
    dir_exists : bool
        The env directory exists on disk under one of conda's
        ``envs_dirs``.
    has_conda_meta : bool
        The env directory has a ``conda-meta/`` subdirectory (the
        marker that conda itself uses to recognise a directory as a
        real env).
    prefix : Optional[str]
        Absolute path to the env if it was resolved by either the
        registry or the filesystem; ``None`` for a fresh install.
    manager : Optional[str]
        The DETECTED env-manager binary this state was probed with, carried
        so that a remedy this state prints can name it
        (`installation.md` M3).  A state that knows the env exists but
        cannot name the manager that manages it can only print a literal
        ``conda``, which is no remedy on a machine that has micromamba.
    """
    name: str
    listed_in_registry: bool
    dir_exists: bool
    has_conda_meta: bool
    prefix: Optional[str]
    manager: Optional[str] = None

    @property
    def state(self) -> "EnvPresence":
        """Which of the five states this env is in -- the classification, made
        ONCE and answered as an enum.

        It used to be a display STRING, and `can_resume` compared that string
        to ``"PRESENT"``: renaming a label for clarity would have turned it
        False, which the code itself calls *"the worst answer for an env that
        is already wreckage"*, and four sites branched on it (H7).  Same shape
        `StepRole` was introduced to remove, on the other state machine.
        """
        reg = self.listed_in_registry
        dir_ok = self.dir_exists and self.has_conda_meta
        if not reg and not self.dir_exists:
            return EnvPresence.FRESH
        if reg and dir_ok:
            return EnvPresence.PRESENT
        if not reg and dir_ok:
            return EnvPresence.ORPHAN
        if reg and not self.dir_exists:
            return EnvPresence.GHOST
        if self.dir_exists and not self.has_conda_meta:
            return EnvPresence.BROKEN
        # UNREACHABLE over all eight combinations of the three observations,
        # and that exhaustivity is load-bearing rather than tidy: a state
        # nothing recognises answers False to BOTH `can_resume` and
        # `needs_cleanup`, and the installer reads that pair as "go ahead and
        # create" -- the worst answer for an env that is already wreckage.
        # `describe()` has no branch for one either.  So if a FOURTH
        # observation is ever added, fail here instead of inventing a state
        # that silently routes to the dangerous default.
        raise AssertionError(  # pragma: no cover
            f"unclassifiable env state for {self.name!r}: "
            f"listed={self.listed_in_registry} dir={self.dir_exists} "
            f"conda_meta={self.has_conda_meta} -- every combination of these "
            f"three should map to one of FRESH / PRESENT / ORPHAN / GHOST / "
            f"BROKEN; a new observation needs a new branch here AND in "
            f"can_resume, needs_cleanup and describe()")

    @property
    def state_label(self) -> str:
        """The state's name, for display.  Derived, so it cannot disagree."""
        return self.state.value

    @property
    def can_resume(self) -> bool:
        """``conda create`` can be skipped and downstream phases run."""
        return self.state is EnvPresence.PRESENT

    @property
    def needs_cleanup(self) -> bool:
        """User should run ``--clean`` or manually fix before installing."""
        return self.state in (EnvPresence.ORPHAN, EnvPresence.GHOST,
                              EnvPresence.BROKEN)

    def describe(self) -> str:
        """Multi-line human description of the state + recommendation."""
        s = self.state
        lines = [
            f"  Env name:           {self.name}",
            f"  Registry lists it:  {'yes' if self.listed_in_registry else 'no'}",
            f"  Directory exists:   {'yes' if self.dir_exists else 'no'}",
            f"  conda-meta/ present:{' yes' if self.has_conda_meta else ' no'}",
        ]
        if self.prefix:
            lines.append(f"  Prefix path:        {self.prefix}")
        lines.append(f"  State:              {s.value}")
        if s is EnvPresence.FRESH:
            lines.append("  → conda create will run (fresh install).")
        elif s is EnvPresence.PRESENT:
            lines.append("  → conda create will be SKIPPED; install resumes from this env.")
        elif s is EnvPresence.ORPHAN:
            lines.append("  → ORPHAN: directory exists but conda's registry doesn't")
            lines.append("    track it.  conda create will refuse with `prefix already")
            lines.append("    exists`.  RECOMMENDED: re-run with --clean to wipe the")
            lines.append("    directory and start fresh.")
        elif s is EnvPresence.GHOST:
            lines.append("  → GHOST: the registry lists this env but the directory")
            lines.append("    it names is gone.  Fix manually with:")
            lines.append(f"      {self.remove_cmd()}")
            lines.append("    or re-run with --clean which will do the same thing.")
        elif s is EnvPresence.BROKEN:
            lines.append("  → BROKEN: directory exists but is missing conda-meta/, so")
            lines.append("    it's not a real conda env -- residue from a failed")
            lines.append("    install, which the manager will not remove for you.")
            lines.append("    Remove the directory yourself, then re-run:")
            lines.append(f"      rm -rf {self.prefix}")
        return "\n".join(lines)

    def remove_cmd(self) -> str:
        """The manual removal line for this env.  See :func:`remove_env_cmd`."""
        return remove_env_cmd(self.manager, self.name)

    def is_the_running_env(self) -> bool:
        """True when this env is the one the current interpreter runs from.

        One home for the question -- :func:`runs_from_prefix` -- because
        ``doctor`` asks it about a prefix it resolved itself and has no
        ``EnvState`` to hand.
        """
        return runs_from_prefix(self.prefix)


def remove_env_cmd(manager: Optional[str], env_name: str) -> str:
    """The ONE spelling of "remove this env by hand".

    `installation.md` M3: a remedy naming a literal ``conda`` is a command the
    person may not have -- the detected binary is what goes in the line.  When
    no manager was detected the line says so instead of guessing one, because a
    wrong command is worse than an honest gap.

    It lives beside :func:`runs_from_prefix` because the two are always used
    together: the places that must not offer ``--clean`` are exactly the places
    that have to print this instead.
    """
    if not manager:
        return (f"<your env manager> env remove -n {env_name} -y"
                f"   (no manager detected)")
    return f"{manager} env remove -n {env_name} -y"


def runs_from_prefix(prefix: Optional[str]) -> bool:
    """Is `prefix` the env THIS interpreter is running from?

    The one answer to that question, asked by `--clean` before it removes an
    env and by ``doctor`` before it recommends that someone do so
    (`installation.md` M5: a remedy the program prints may not destroy
    working state).

    Both facts are GIVEN rather than derived (M2): the prefix came from the
    manager's own registry, and ``sys.prefix`` is the interpreter reporting
    where it lives.  Nothing here assumes a manager layout or an
    ``envs_dirs``, and ``CONDA_PREFIX`` is deliberately NOT consulted -- the
    shim dispatches the host env's python without activating, so that
    variable can name a different env entirely, or nothing at all.
    """
    if not prefix:
        return False
    try:
        return os.path.realpath(prefix) == os.path.realpath(sys.prefix)
    except OSError:  # pragma: no cover - realpath on a hostile path
        return False


def probe_env_state(env_name: str, conda_binary: str) -> EnvState:
    """Probe the conda env's current state.  Pure read; no side effects.

    Runs THREE independent checks (registry, the env's directory, its
    ``conda-meta/``) and combines the results into an :class:`EnvState`.
    Cheap -- one or two manager subprocesses, ~100 ms each on a warm
    system, much less than the cost of a single failed
    ``conda create``.

    **The directory checks are measured on the prefix the REGISTRY named**
    (`installation.md` M2: a path inside an env is asked for, never derived).
    The ``envs_dirs`` search is the fallback for the opposite case -- a
    directory that no registry entry mentions, which is what ORPHAN and BROKEN
    are -- so it is only consulted when the registry does not list the env.
    Measuring the search path *instead* of the named prefix is what made an env
    created with ``--prefix`` outside ``envs_dirs`` report GHOST while carrying
    a healthy prefix, and GHOST prints a removal command.
    """

    # Check 1: the registry, through the ONE reader (H2).  Read FRESH, not off
    # the startup snapshot: this probe runs during an install, where an env was
    # created or removed seconds ago and the snapshot is exactly wrong.
    from ..diagnostics import conda_env_prefixes
    prefix_from_registry: Optional[str] = conda_env_prefixes(
        conda_binary).get(env_name)
    listed = prefix_from_registry is not None

    # Checks 2 + 3: the directory, and its conda-meta/.
    #
    # Measured on ONE path, and which path comes from check 1: the prefix the
    # registry named when it named one, the envs_dirs search only when it did
    # not.  Those are the two different questions the five states are made of
    # -- "the env the manager knows about, is it still on disk" (PRESENT vs
    # GHOST) and "is there a directory the manager does NOT know about"
    # (ORPHAN, BROKEN) -- and answering the first by searching the second's
    # haystack is what reported GHOST for a healthy out-of-envs_dirs env.
    dir_exists = False
    has_conda_meta = False
    prefix_from_fs: Optional[str] = None
    if prefix_from_registry is not None:
        target = Path(prefix_from_registry)
        dir_exists = target.is_dir()
        has_conda_meta = (target / "conda-meta").is_dir()
    else:
        from ..diagnostics import manager_info
        candidate = _prefix_under_envs_dirs(manager_info(conda_binary), env_name)
        if candidate is not None:
            dir_exists = True
            prefix_from_fs = str(candidate)
            has_conda_meta = (candidate / "conda-meta").is_dir()

    prefix = prefix_from_registry or prefix_from_fs
    return EnvState(
        name=env_name,
        listed_in_registry=listed,
        dir_exists=dir_exists,
        has_conda_meta=has_conda_meta,
        prefix=prefix,
        manager=conda_binary,
    )


# NOTE: the helpers ``_env_listed_now`` and ``_env_prefix_dir_exists``
# used to live here.  Both were thin wrappers that duplicated logic
# already inside ``probe_env_state`` (the ONE source of truth for env
# presence).  Worse, they were combined via an ``OR`` in run_install
# that fired True for orphan directories (which conda create would
# then refuse with "prefix already exists"), shipping a false
# positive that masked --clean failures.  Replaced by direct
# ``probe_env_state(...).can_resume`` use in run_install; the helpers
# are gone, not deprecated, because nothing else called them.


# --------------------------------------------------------------------- #
#  Phase execution                                                       #
# --------------------------------------------------------------------- #


@dataclass
class _Dispatcher:
    """Where a recipe's steps get dispatched.

    Holds the env prefix -- read off the startup snapshot when it knows the
    env, a registry read when it does not -- so every phase of a recipe
    shares one resolution instead of asking again.
    ``conda create`` is the only step that needs no prefix; there is no
    env yet.  Everything after it is addressed by the prefix
    (`installation.md` M2), which is why resolving it once matters.
    """

    env_name: str
    conda_binary: str
    prefix: Optional[str] = None

    def ensure_prefix(self) -> Optional[str]:
        """Resolve the prefix if it is not known yet.

        Called before every dispatched step rather than once up front: a
        brand-new env has no prefix until ``conda create`` has run.
        """
        if self.prefix is None:
            sys.stderr.write(
                f"[install] resolving env prefix for `{self.env_name}`...\n")
            sys.stderr.flush()
            self.prefix = _env_prefix(self.env_name, self.conda_binary)
            if self.prefix is not None:
                sys.stderr.write(f"[install] env prefix: {self.prefix}\n")
                sys.stderr.flush()
        return self.prefix


#: `builds.py` keeps its own four-state verdict because it keeps its own
#: executor (sentinel resume).  This is the ONE place the two vocabularies
#: meet.  Deriving the outcome from the return code instead -- which the
#: adapter used to do -- got two of the four wrong: a sentinel-skipped phase
#: carries `returncode=0` and so reported OK, indistinguishable from having
#: actually run it; and an abandoned phase carries `None` and so reported
#: FAILED, inflating one real failure into one per remaining phase.
_BUILD_STATUS_OUTCOME = {
    "ok": Outcome.OK,
    "fail": Outcome.FAILED,
    "skip": Outcome.SKIPPED,
}


def _adapt_build_step(sresult) -> Optional[InstallStep]:
    """One build phase's result as an :class:`InstallStep`.

    ``None`` for a phase that was never reached (`status == "not-run"`),
    matching what the install loop itself does: it returns on the first step
    that stops the install and never records the rest.  The full per-phase
    detail, abandoned phases included, stays on ``InstallResult.build_result``
    for anyone who wants it.
    """
    outcome = _BUILD_STATUS_OUTCOME.get(sresult.status)
    if outcome is None:
        return None
    return InstallStep(
        label=f"build:{sresult.step.component}.{sresult.step.phase}",
        role=StepRole.BUILD,
        argv=sresult.step.argv,
        returncode=sresult.returncode,
        output=sresult.output,
        outcome=outcome,
    )


def _undispatched(step: InstallStep, outcome: Outcome,
                  why: str) -> InstallStep:
    """A step decided WITHOUT running it.

    ``returncode`` stays ``None``, because a step that did not run has no
    exit code.  The conda-create skip used to report 0, which made "I did
    not do this" indistinguishable from "I did this and it worked".

    ``why`` lands in ``output``, which is where :func:`_report` looks for
    the reason -- nothing streamed to the terminal for a step that never
    dispatched, so the record is the only account of it.
    """
    return replace(step, outcome=outcome, returncode=None, output=why)


#: The word the user sees for each outcome.  ONE mapping, so a step can
#: never be announced in a word that contradicts its verdict: two
#: failures used to print "SKIPPED" while aborting the install.
def _report(tag: str, done: InstallStep) -> None:
    """Announce one finished step on stderr."""
    rc = "" if done.returncode is None else f" (rc={done.returncode})"
    note = f" -- {done.outcome.note}" if done.outcome.note else ""
    sys.stderr.write(f"[{tag}] {done.label}: "
                     f"{done.outcome.word}{note}{rc}\n")
    # A step that never dispatched streamed nothing, so its reason has
    # only been recorded -- say it here or the user sees a bare verdict.
    if done.returncode is None and done.outcome is not Outcome.OK:
        first = (done.output or "").strip().splitlines()
        if first:
            sys.stderr.write(f"[{tag}]   {first[0]}\n")
    sys.stderr.flush()


def _create_decision(step: InstallStep, dispatcher: _Dispatcher, *,
                     force_resume: bool,
                     state: Optional[EnvState] = None
                     ) -> Optional[InstallStep]:
    """Whether ``conda create`` needs to run, as an outcome.

    Asked unconditionally: an existing env is always resumed into, which is
    what makes `install` idempotent.  A `skip_if_present` parameter stood here
    until 2026-09-13, defaulting to False, whose docstring said "set False
    only in tests" -- no test ever set it, and the one production caller that
    could reach a CREATE step passed True (H8).  Three answers, and all three
    are now states rather than a fabricated exit code plus a separate bool:

      * the env is usable   -> ``SKIPPED``, claiming no exit code;
      * the env is wreckage -> ``FAILED``, carrying ``--clean`` as the
        remedy (``conda create`` would refuse with "prefix already
        exists" and the state probe has already classified why);
      * otherwise           -> ``None``, meaning dispatch it.

    ``state`` is the reading the CALLER already took, when it took one: the CLI
    probes this env before it prints anything, and `run_install` then probed it
    again one call later -- the same two JSON documents read twice back to back,
    measured at three reads per install from the CLI (H6).  A caller that has
    CHANGED the machine since its reading passes nothing, which is what
    ``--clean`` does after removing the env: the stale reading there says PRESENT
    about an env that is gone, and skipping create on it would install into
    nothing.  Never read off the capabilities snapshot, which goes stale for the
    same reason and is not even this recipe's question.
    """
    if state is None:
        state = probe_env_state(dispatcher.env_name, dispatcher.conda_binary)
    # ``--force-resume``: the operator knows the env is usable even
    # though the probe says GHOST / ORPHAN / BROKEN -- typically mid
    # source-build, where the directory exists but conda-meta has not
    # been finalised yet.
    if state.can_resume or force_resume:
        why = ("already exists" if state.can_resume
               else f"--force-resume; state was {state.state_label}")
        return _undispatched(
            step, Outcome.SKIPPED,
            f"env `{dispatcher.env_name}` {why}; skipping create")
    if state.needs_cleanup:
        return _undispatched(
            step, Outcome.FAILED,
            f"env `{dispatcher.env_name}` is in state {state.state_label} "
            f"-- re-run with --clean to wipe before installing.")
    return None


def _run_steps(
    steps: Sequence[InstallStep],
    dispatcher: _Dispatcher,
    *,
    tag: str,
    executed: List[InstallStep],
    force_resume: bool = False,
    env_state: Optional[EnvState] = None,
) -> bool:
    """Run one phase's steps through the one door.

    Returns ``False`` when a step stopped the install.  Appends every
    step it decided -- dispatched or not -- to ``executed``, so the
    caller's verdict can be derived from the steps alone.

    EVERY phase runs here: conda-create + pip + extra, and verify after
    the build.  Verify used to have its own loop, which is how it came to
    re-implement the prefix resolution, the bypass, the launch-failure
    branch and the output trim, and to carry no outcome at all.
    """
    total = len(steps)
    for i, step in enumerate(steps, start=1):
        where = f"{tag} {i}/{total}"
        if step.role is StepRole.CREATE:
            decided = _create_decision(
                step, dispatcher,
                force_resume=force_resume, state=env_state)
            if decided is not None:
                executed.append(decided)
                _report(where, decided)
                if decided.outcome.stops_the_install:
                    return False
                continue
        elif step.role is StepRole.REMOVE:
            # Unconditional, and addressed by name: the point is that the env
            # exists, so there is nothing to decide and no prefix to require.
            pass
        elif dispatcher.ensure_prefix() is None:
            # FAIL LOUD.  Without a prefix the step cannot be addressed at
            # the env at all (M2), and a `-n` dispatch would then be resolved
            # against envs_dirs -- a different env, or an error about the wrong
            # thing entirely.
            executed.append(_undispatched(
                step, Outcome.FAILED,
                f"could not resolve env prefix for `{dispatcher.env_name}` "
                f"-- not dispatching {step.label}.  Run "
                f"`{dispatcher.conda_binary} env list` to confirm the env is "
                f"there; if it is and we cannot find it, file an issue "
                f"with that output."))
            _report(where, executed[-1])
            return False
        sys.stderr.write(f"[{where}] {step.label}: starting\n")
        sys.stderr.flush()
        # ONE DOOR, and the outcome decides what happens next -- no
        # nested conditions over return codes, alternatives and
        # optionality, which is where branches kept going missing.
        done = run_step(
            step,
            # A removal gets no prefix: it would only be used to make temp
            # directories inside the env about to be deleted.
            prefix=None if step.role is StepRole.REMOVE else dispatcher.prefix,
            sink=sys.stderr)
        executed.append(done)
        _report(where, done)
        if done.outcome.stops_the_install:
            return False
        if step.role is StepRole.REMOVE:
            # THE MACHINE CHANGED, and every reading taken before this line is
            # void: the directory the dispatcher held no longer exists, and the
            # process snapshot still lists it -- `_env_prefix` reads that
            # snapshot FIRST, so left alone it would hand the next step the
            # old directory.  Cleared here, by the installer, because the
            # installer is what changed the machine; the surface used to reset
            # the snapshot itself, BEFORE the removal ran, which accounted for
            # nothing.  `conda create` next puts the env wherever the manager
            # decides, and `ensure_prefix` then asks afresh.
            dispatcher.prefix = None
            reset_capabilities()
    return True


def run_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
    rebuild: Optional[str] = None,
    build_on_warnings: Optional["_builds.ConfirmWarningsCallback"] = None,
    build_on_progress: Optional["_builds.ProgressCallback"] = None,
    build_skip_network_check: bool = False,
    force_resume: bool = False,
    env_state: Optional[EnvState] = None,
    clean: bool = False,
) -> InstallResult:
    """Execute the install plan, stopping at the first failed step.

    Parameters
    ----------
    rebuild
        For recipes carrying a ``build_spec``, forwarded to
        :func:`builds.run_build_spec`.  ``None`` or ``"none"`` resumes
        from sentinels; ``"all"`` rebuilds everything; a component
        name (``"elpa"``, ``"siesta"``) rebuilds that component plus
        everything downstream of it.  Ignored for non-build recipes.
    build_on_warnings, build_on_progress
        Optional callbacks forwarded to
        :func:`builds.run_build_spec` for source-build recipes.
        ``build_on_warnings(report) -> bool`` lets the CLI surface
        non-fatal preflight warnings + ask the user to confirm;
        ``build_on_progress(event, step, result)`` lets the CLI
        render per-phase progress.
    build_skip_network_check
        Skip the per-component ``git ls-remote`` reachability check.
    env_state
        A reading of this env's state the caller already took, passed in so the
        same two registry documents are not read twice in a row (H6).  Ignored
        when ``clean`` is set, because this run is about to delete that env.
    clean
        Remove the env first: `remove_step_for` goes at the front of the plan
        and runs through the one door like every other step.
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = effective_name(recipe, caps)
    planned = _plan(recipe, effective, caps.conda_binary)
    if clean:
        # The wipe is the FIRST step, in the plan, not a dispatch the surface
        # does on the side (§ 5.4).  And a state read BEFORE it is then a
        # reading of an env this run is about to delete: `PRESENT` would skip
        # the `conda create` that has to follow, which is the 2026-06-15
        # regression.  So the create decision is made fresh, after the removal.
        planned.insert(0, remove_step_for(
            effective, caps.conda_binary,
            # the probe's directory, for an env the registry does not list
            prefix=(env_state.prefix
                    if env_state is not None and env_state.dir_exists
                    and not env_state.listed_in_registry else None)))
        env_state = None

    sys.stderr.write(
        f"[install] recipe `{recipe.name}` -> env `{effective}`\n"
    )
    sys.stderr.write(
        f"[install] env manager: {caps.conda_binary}\n"
    )
    sys.stderr.flush()

    # The prefix, resolved once for the whole run -- from the startup snapshot
    # when it knows this env, a registry read when it does not -- and handed
    # to every step through the dispatcher.  NOT for a `--clean` run: that
    # prefix names the env the first step removes, and `ensure_prefix` asks
    # afresh after `conda create` has made the new one.
    cached_prefix: Optional[str] = (
        None if clean else _env_prefix(effective, caps.conda_binary))
    if cached_prefix is not None:
        sys.stderr.write(
            f"[install] env prefix: {cached_prefix}\n"
        )
        sys.stderr.flush()
    # NOTE: ``env_exists`` is still never pre-computed from caps.  The
    # capabilities snapshot can be stale -- notably right after --clean, when
    # ``get_capabilities()`` returns the bound snapshot rather than
    # re-detecting -- and trusting it caused the 2026-06-15 "env already exists;
    # conda may have failed silently" regression.  ``env_state`` is a different
    # thing: a reading of THIS env that the caller took itself and has not
    # invalidated since, which is why the CLI stops paying for a second
    # identical probe one call later (H6).  A caller that changed the machine
    # passes nothing and the probe runs here.
    executed: List[InstallStep] = []
    dispatcher = _Dispatcher(env_name=effective, conda_binary=caps.conda_binary,
                   prefix=cached_prefix)

    # Reorder: conda-create + pip + extra_steps + (build_spec) + verify.
    # Verify is pulled out of `planned` and run AFTER the build phase so
    # it checks the built binary rather than the env that will hold it.
    # Both groups go through the SAME runner -- verify's own loop is what
    # used to re-implement the prefix resolution, the `conda run` bypass,
    # the launch-failure branch and the output trim, and to leave every
    # verify step carrying no outcome at all.
    verify_steps = [s for s in planned if s.role is StepRole.VERIFY]
    pre_verify = [s for s in planned if s.role is not StepRole.VERIFY]

    ok = _run_steps(pre_verify, dispatcher, tag="install", executed=executed,
                    force_resume=force_resume, env_state=env_state)

    # Build-spec phase: only if the recipe declares one AND nothing
    # failed before it.  `builds.run_build_spec` keeps its own executor
    # (it has sentinel-based resume, which no install step has), so what
    # happens here is an ADAPTER: its results become InstallSteps whose
    # outcome is decided by the same rule as everything else.
    build_result: Optional[_builds.BuildResult] = None
    if ok and recipe.build_spec is not None:
        if dispatcher.ensure_prefix() is None:
            executed.append(_undispatched(
                InstallStep(label="build", role=StepRole.BUILD,
                            argv=("internal", "resolve-env-prefix")),
                Outcome.FAILED,
                f"could not resolve $CONDA_PREFIX for env {effective!r}; "
                f"conda may have failed silently."))
            _report("build", executed[-1])
            ok = False
        else:
            build_result = _builds.run_build_spec(
                recipe.build_spec, dispatcher.prefix,
                conda_binary=caps.conda_binary,
                rebuild=rebuild,
                skip_network_check=build_skip_network_check,
                on_warnings=build_on_warnings,
                on_progress=build_on_progress,
            )
            if build_result.preflight_errors:
                executed.append(_undispatched(
                    InstallStep(label="build:preflight",
                                role=StepRole.BUILD,
                                argv=("preflight",)),
                    Outcome.FAILED,
                    "\n".join(build_result.preflight_errors)))
                ok = False
            else:
                for sresult in build_result.steps:
                    adapted = _adapt_build_step(sresult)
                    if adapted is not None:
                        executed.append(adapted)
                if not build_result.succeeded:
                    ok = False

    if ok and verify_steps:
        _run_steps(verify_steps, dispatcher, tag="verify", executed=executed)

    # DERIVED, not tracked alongside.  A separate `succeeded` bool was
    # what let the word printed to the user disagree with the verdict --
    # two failures announced themselves as "SKIPPED".  `stops_the_install`
    # and not `is_success` is the right predicate: a DEGRADED optional
    # package is honestly absent without the install having failed.
    # The build's verdict is IN the steps: a failed phase is an adapted FAILED
    # step, and preflight errors or a declined warning are an undispatched
    # one -- so the override that stood here ("builds.py owns its own verdict")
    # could never change the answer.
    succeeded = not any(s.outcome.stops_the_install for s in executed)

    return InstallResult(
        recipe=recipe,
        effective_name=effective,
        steps=tuple(executed),
        succeeded=succeeded,
        build_result=build_result,
    )


__all__ = [
    # The framework's surface, in the order the contract introduces it
    # (docs/ops/env-framework.md): a record becomes a step, one runner
    # runs it, one rule decides what became of it.
    "conda_argv",
    "conda_run_argv",
    "pip_steps_for",
    "extra_steps_for",
    "remove_step_for",
    "create_step_for",
    "conda_step_for",
    "pip_argv",
    "pip_step_for",
    "verify_step_for",
    "InstallStep",
    "StepRole",
    "Outcome",
    "run_step",
    "OUTPUT_LIMIT",
    # ...and the two entry points built on it.
    "InstallResult",
    "plan_install",
    "run_install",
]
