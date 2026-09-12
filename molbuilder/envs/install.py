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
  5. Verify (re-uses :mod:`molbuilder.envs.doctor`).

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

import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..diagnostics import Capabilities, get_capabilities
from . import builds as _builds
from .doctor import _effective_name


def _bypass_conda_run(argv: Sequence[str], env_prefix: str
                      ) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Rewrite ``conda run -n NAME --no-capture-output CMD...`` into a
    direct call that bypasses ``conda run``'s shell stub.

    Why: mamba 1.x's ``run`` implementation generates a stub that
    uses ``exec -- "$@"``; bash's ``exec`` builtin rejects ``--``
    with ``line 5: exec: --: invalid option`` and the whole step
    dies before the inner command starts.  Fixed in mamba 2.x, but
    a lot of HPC sites still ship 1.x.  The same bug bit the bash
    install-env.sh shim and was fixed there by calling the env's
    binary directly; we mirror the cure here.

    Implementation: build a bash wrapper that sets the same
    environment ``conda activate <env>`` would (PATH,
    LD_LIBRARY_PATH, CONDA_PREFIX, CONDA_DEFAULT_ENV) AND sources
    every ``<env>/etc/conda/activate.d/*.sh`` script, then execs
    the inner command.  Sourcing activate.d is load-bearing for
    source-built recipes (siesta-gpu installs binaries to
    ``<env>/opt/siesta-gpu-stack/siesta/bin``, NOT ``<env>/bin``;
    only the activate.d hook adds that path) plus any conda
    package that registers post-activate env mutations
    (cuda-version, openmpi, etc.).

    Returns ``(new_argv, env_overrides)``.  The wrapper handles
    env setup internally; ``env_overrides`` is therefore an empty
    dict and exists only to preserve the existing caller contract.
    """
    # Expected shape (from _plan):
    #     argv[0] = conda binary
    #     argv[1] = "run"
    #     argv[2] = "-n"             OR    "--prefix"
    #     argv[3] = <env name>       OR    <env prefix path>
    #     argv[4] = "--no-capture-output"
    #     argv[5:] = the actual command
    # builds.py uses --prefix + "--" separator, install.py uses -n.
    if len(argv) < 6 or argv[1] != "run":
        raise ValueError(f"_bypass_conda_run: unexpected shape {argv!r}")
    start = 5
    if start < len(argv) and argv[start] == "--":
        start += 1
    cmd: Tuple[str, ...] = tuple(str(a) for a in argv[start:])
    if not cmd:
        raise ValueError(f"_bypass_conda_run: empty inner cmd in {argv!r}")
    env_q = shlex.quote(env_prefix)
    env_name = shlex.quote(Path(env_prefix).name)
    activate_d = shlex.quote(f"{env_prefix}/etc/conda/activate.d")
    cmd_q = " ".join(shlex.quote(a) for a in cmd)
    # HPC strictness: constrain temp + cache dirs to the env prefix.
    # Many strict systems have small /tmp, restricted /var, or
    # per-user write quotas on $HOME.  Keeping pip's wheel cache,
    # tar/cmake/make's temp files, and ccache's compilation cache
    # under the env prefix means a single ``conda env remove`` truly
    # cleans up after this install -- nothing dangles in $HOME.
    # Wrapper diagnostics: echo every load-bearing detail to stderr
    # BEFORE the exec.  We cannot debug what we cannot see -- a
    # silent ``exec: --: invalid option`` failure with no other
    # context (which is what the user just hit) wastes everyone's
    # time.  Each line tagged ``[bypass]`` so it's filterable.
    # The debug echoes report the command EXACTLY as it will be exec'd,
    # and each echo argument is quoted as ONE shell word.
    #
    # This used to build ``echo "[bypass] cmd={shlex.quote(...)}"`` --
    # shlex.quote emits SINGLE quotes, dropped inside a DOUBLE-quoted
    # echo, so a command containing a double quote closed the echo early
    # and the remainder was parsed as shell.  The first step whose text
    # had one (the toolchain-shim step, whose body contains
    # ``B="$CONDA_PREFIX/bin"`` and a ``link() {`` function) died with
    # ``syntax error near unexpected token '('`` -- in the DIAGNOSTIC
    # line, while the command it was reporting was perfectly valid.
    # Quoting for the wrong context turned a debugging aid into the
    # failure it was meant to explain.
    #
    # ``cmd_q`` rather than a bare join: the logged line is then
    # copy-pasteable, and argv boundaries survive.
    debug_cmd_repr = cmd_q
    wrapper = (
        f'echo {shlex.quote(f"[bypass] env_prefix={env_prefix}")} >&2; '
        f'echo {shlex.quote(f"[bypass] cmd={debug_cmd_repr}")} >&2; '
        f"export CONDA_PREFIX={env_q}; "
        f"export CONDA_DEFAULT_ENV={env_name}; "
        f'export PATH={env_q}/bin"${{PATH:+:$PATH}}"; '
        f'export LD_LIBRARY_PATH={env_q}/lib"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        f"mkdir -p {env_q}/var/tmp {env_q}/var/cache/pip; "
        f"export TMPDIR={env_q}/var/tmp; "
        f"export PIP_CACHE_DIR={env_q}/var/cache/pip; "
        f"if [ -d {activate_d} ]; then "
        f'echo {shlex.quote(f"[bypass] sourcing activate.d/*.sh in {activate_d}")} >&2; '
        f'for _f in {activate_d}/*.sh; do '
        f'[ -f "$_f" ] || continue; '
        f'echo "[bypass]   . $_f" >&2; '
        f'. "$_f"; '
        f"done; "
        f"else "
        f'echo "[bypass] (no activate.d directory)" >&2; '
        f"fi; "
        f'echo "[bypass] final PATH=$PATH" >&2; '
        f'echo {shlex.quote(f"[bypass] exec: {debug_cmd_repr}")} >&2; '
        f"exec {cmd_q}"
    )
    # ``bash -c`` (no -l): we don't want the user's login files
    # sourced -- the activate.d sourcing above is the only env
    # setup we want.  System ``/bin/bash`` is universally present;
    # we don't depend on the env's bash being installed yet.
    return (("bash", "-c", wrapper), {})
from .recipes import PipPackage, Recipe


#: How much of a step's combined output is kept.  The streamed copy
#: already reached the user's terminal; this is the excerpt the CLI
#: recaps and the web report stores, so it is an excerpt on purpose.
OUTPUT_LIMIT = 4096


@dataclass(frozen=True)
class InstallStep:
    """One command in the install plan.

    Attributes
    ----------
    label
        Short tag (``"conda create"``, ``"pip install"``,
        ``"extra"``, ``"verify"``).  Used by the CLI for the per-
        step header line.
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
        success wins and the step counts as OK.
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


def run_step(
    step: InstallStep,
    *,
    env: Optional[Dict[str, str]] = None,
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

    for n, argv in enumerate(attempts):
        run_argv = list(argv)
        if prefix:
            # Same PIP_CACHE_DIR / activate.d treatment for every
            # attempt, primary or alternative.
            try:
                bypassed, _ = _bypass_conda_run(argv, prefix)
                run_argv = list(bypassed)
            except ValueError:
                pass
        if n and sink is not None:
            sink.write(f"    {step.label}: previous source was not "
                       f"accepted ({_why_rejected(step, rc, combined)}); "
                       f"trying the declared alternative\n")
            sink.flush()
        rc, combined = _builds.run_streaming(
            run_argv, env=env, sink=sink, timeout=timeout)
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
    return (conda, "run", "-n", env_name, "--no-capture-output",
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
        argv=(conda, "run", "-n", env_name, "--no-capture-output",
              *recipe.verify_argv),
        ignore_exit_code=recipe.verify_ignore_exit_code,
        expect_contains=recipe.verify_expect_contains,
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
    plain = [p for p in recipe.pip_packages if p.is_plain()]
    if plain:
        steps.append(InstallStep(
            label="pip install",
            argv=pip_argv(conda, env_name, *(p.spec() for p in plain)),
        ))
    steps.extend(pip_step_for(pkg, conda, env_name)
                 for pkg in recipe.pip_packages if not pkg.is_plain())

    # Phase 3: extra dispatch-into-env steps.
    for extra in recipe.extra_steps:
        argv = (conda, "run", "-n", env_name, "--no-capture-output",
                *extra)
        steps.append(InstallStep(label="extra", argv=argv))

    # Phase 4: verify (only if the recipe declares one).
    verify = verify_step_for(recipe, conda, env_name)
    if verify is not None:
        steps.append(verify)

    return steps


def plan_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
) -> Tuple[str, List[InstallStep]]:
    """Return ``(effective_name, steps)`` for the recipe.

    Pure planner -- does not run anything.  Useful for ``--dry-run``
    + for tests that assert on the command shape without subprocess
    side effects.

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
    effective = _effective_name(recipe, caps)
    return effective, _plan(recipe, effective, caps.conda_binary)


def _env_prefix(env_name: str, conda_binary: str) -> Optional[str]:
    """Return ``$CONDA_PREFIX`` for the named env, or ``None`` if not found.

    Four-tier resolution mirrors ``install-env.sh``'s bash
    ``_resolve_env_python``.  Multiple fallbacks because a single
    failure here cascades into the verify step running through
    ``<mgr> run`` (the buggy path we're trying to bypass) -- the
    user then sees ``exec: --: invalid option`` and the install
    "succeeds" but actually fails:

    1. ``<mgr> env list --json`` -- the canonical registry.
       Match by basename so envs in custom envs_dirs are caught.
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
    4. Direct disk probe of common envs_dir locations -- mamba's
       ``~/.conda/envs`` is the load-bearing fallback here because
       it's the default mamba writes to even when the conda binary
       being asked is from a different install root.
    """
    import json as _json
    # Strategy 1: registry.  ``Path(prefix).name == env_name`` so
    # envs at custom locations still match if the basename is right.
    try:
        cp = subprocess.run(
            [conda_binary, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        cp = None
    if cp is not None and cp.returncode == 0:
        try:
            envs = _json.loads(cp.stdout).get("envs", [])
        except (ValueError, KeyError):
            envs = []
        for prefix in envs:
            if Path(prefix).name == env_name:
                return prefix
    # Strategies 2 + 3: info --json.
    try:
        info_cp = subprocess.run(
            [conda_binary, "info", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        info: dict = {}
        if info_cp.returncode == 0:
            try:
                info = _json.loads(info_cp.stdout) or {}
            except ValueError:
                info = {}
        # Strategy 2: envs (full paths, not the search list).
        for prefix in info.get("envs", []) or []:
            if Path(prefix).name == env_name and Path(prefix).is_dir():
                return prefix
        # Strategy 3: envs_dirs (search list) -- look for the env
        # under each configured envs_dir.
        for envs_dir in info.get("envs_dirs", []) or []:
            candidate = Path(envs_dir) / env_name
            if candidate.is_dir():
                return str(candidate)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    # Strategy 4: disk probe.  mamba init's default envs_dir is
    # ``~/.conda/envs``, even when the conda binary lives elsewhere
    # (e.g. ``conda_binary = ~/miniconda3/condabin/conda`` and envs
    # end up under ``~/.conda/envs`` because mamba's defaults differ
    # from conda's).  Plus the conda binary's own install-root envs
    # dir as a secondary guess.
    home = Path.home()
    candidates = [home / ".conda" / "envs" / env_name]
    # Derive ``<install root>/envs/<name>`` from the conda binary's
    # path -- strip ``/condabin/...`` OR ``/bin/...`` to get root.
    conda_path = Path(conda_binary)
    for marker in ("condabin", "bin"):
        try:
            idx = conda_path.parts.index(marker)
        except ValueError:
            continue
        root = Path(*conda_path.parts[:idx])
        candidates.append(root / "envs" / env_name)
        break
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    return None


# --------------------------------------------------------------------- #
#  Env-state probe -- diagnose conda env state up front                  #
# --------------------------------------------------------------------- #


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
    """
    name: str
    listed_in_registry: bool
    dir_exists: bool
    has_conda_meta: bool
    prefix: Optional[str]

    @property
    def state_label(self) -> str:
        """One-word classification."""
        reg = self.listed_in_registry
        dir_ok = self.dir_exists and self.has_conda_meta
        if not reg and not self.dir_exists:
            return "FRESH"
        if reg and dir_ok:
            return "PRESENT"
        if not reg and dir_ok:
            return "ORPHAN"
        if reg and not self.dir_exists:
            return "GHOST"
        if self.dir_exists and not self.has_conda_meta:
            return "BROKEN"
        return "UNKNOWN"

    @property
    def can_resume(self) -> bool:
        """``conda create`` can be skipped and downstream phases run."""
        return self.state_label == "PRESENT"

    @property
    def needs_cleanup(self) -> bool:
        """User should run ``--clean`` or manually fix before installing."""
        return self.state_label in ("ORPHAN", "GHOST", "BROKEN")

    def describe(self) -> str:
        """Multi-line human description of the state + recommendation."""
        s = self.state_label
        lines = [
            f"  Env name:           {self.name}",
            f"  Registry lists it:  {'yes' if self.listed_in_registry else 'no'}",
            f"  Directory exists:   {'yes' if self.dir_exists else 'no'}",
            f"  conda-meta/ present:{' yes' if self.has_conda_meta else ' no'}",
        ]
        if self.prefix:
            lines.append(f"  Prefix path:        {self.prefix}")
        lines.append(f"  State:              {s}")
        if s == "FRESH":
            lines.append("  → conda create will run (fresh install).")
        elif s == "PRESENT":
            lines.append("  → conda create will be SKIPPED; install resumes from this env.")
        elif s == "ORPHAN":
            lines.append("  → ORPHAN: directory exists but conda's registry doesn't")
            lines.append("    track it.  conda create will refuse with `prefix already")
            lines.append("    exists`.  RECOMMENDED: re-run with --clean to wipe the")
            lines.append("    directory and start fresh.")
        elif s == "GHOST":
            lines.append("  → GHOST: conda's registry lists this env but the directory")
            lines.append("    is gone.  Fix manually with:")
            lines.append(f"      conda env remove -n {self.name} -y")
            lines.append("    or re-run with --clean which will do the same thing.")
        elif s == "BROKEN":
            lines.append("  → BROKEN: directory exists but is missing conda-meta/, so")
            lines.append("    it's not a real conda env.  Almost certainly residue from")
            lines.append("    a previous failed install.  RECOMMENDED: re-run with")
            lines.append("    --clean to wipe the directory and start fresh.")
        return "\n".join(lines)


def probe_env_state(env_name: str, conda_binary: str) -> EnvState:
    """Probe the conda env's current state.  Pure read; no side effects.

    Runs THREE independent checks (registry, conda-info envs_dirs,
    filesystem) and combines the results into an :class:`EnvState`.
    Cheap -- two ``conda`` subprocesses, ~100 ms each on a warm
    system, much less than the cost of a single failed
    ``conda create``.
    """
    import json as _json

    # Check 1: conda env list (the registry)
    listed = False
    prefix_from_registry: Optional[str] = None
    try:
        cp = subprocess.run(
            [conda_binary, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if cp.returncode == 0:
            envs = _json.loads(cp.stdout).get("envs", [])
            for prefix in envs:
                if Path(prefix).name == env_name:
                    listed = True
                    prefix_from_registry = prefix
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            ValueError, KeyError):
        pass

    # Checks 2 + 3: filesystem (conda's envs_dirs)
    dir_exists = False
    has_conda_meta = False
    prefix_from_fs: Optional[str] = None
    try:
        info_cp = subprocess.run(
            [conda_binary, "info", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if info_cp.returncode == 0:
            info = _json.loads(info_cp.stdout)
            for envs_dir in info.get("envs_dirs", []):
                candidate = Path(envs_dir) / env_name
                if candidate.is_dir():
                    dir_exists = True
                    prefix_from_fs = str(candidate)
                    if (candidate / "conda-meta").is_dir():
                        has_conda_meta = True
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            ValueError, KeyError):
        pass

    prefix = prefix_from_registry or prefix_from_fs
    return EnvState(
        name=env_name,
        listed_in_registry=listed,
        dir_exists=dir_exists,
        has_conda_meta=has_conda_meta,
        prefix=prefix,
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
class _Phase:
    """Where a recipe's steps get dispatched.

    Holds the one expensive thing -- the env prefix, which costs three to
    five ``<mgr> env list`` / ``info --json`` calls to resolve -- so every
    phase of a recipe shares one resolution instead of re-probing.
    ``conda create`` is the only step that needs no prefix; there is no
    env yet.  Everything after it is dispatched through the activate.d-
    sourcing bypass, which does.
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
_OUTCOME_WORD = {
    Outcome.OK: "OK",
    Outcome.RECOVERED: "OK via the declared alternative",
    Outcome.DEGRADED: "UNAVAILABLE -- optional, continuing",
    Outcome.FAILED: "FAILED",
    Outcome.SKIPPED: "SKIPPED",
}


def _report(tag: str, done: InstallStep) -> None:
    """Announce one finished step on stderr."""
    rc = "" if done.returncode is None else f" (rc={done.returncode})"
    sys.stderr.write(f"[{tag}] {done.label}: "
                     f"{_OUTCOME_WORD[done.outcome]}{rc}\n")
    # A step that never dispatched streamed nothing, so its reason has
    # only been recorded -- say it here or the user sees a bare verdict.
    if done.returncode is None and done.outcome is not Outcome.OK:
        first = (done.output or "").strip().splitlines()
        if first:
            sys.stderr.write(f"[{tag}]   {first[0]}\n")
    sys.stderr.flush()


def _create_decision(step: InstallStep, phase: _Phase, *,
                     skip_if_present: bool,
                     force_resume: bool) -> Optional[InstallStep]:
    """Whether ``conda create`` needs to run, as an outcome.

    Three answers, and all three are now states rather than a fabricated
    exit code plus a separate bool:

      * the env is usable   -> ``SKIPPED``, claiming no exit code;
      * the env is wreckage -> ``FAILED``, carrying ``--clean`` as the
        remedy (``conda create`` would refuse with "prefix already
        exists" and the state probe has already classified why);
      * otherwise           -> ``None``, meaning dispatch it.

    The probe is live (:func:`probe_env_state`) rather than read off the
    cached capabilities, which go stale right after ``--clean``.
    """
    if not skip_if_present:
        return None
    state = probe_env_state(phase.env_name, phase.conda_binary)
    # ``--force-resume``: the operator knows the env is usable even
    # though the probe says GHOST / ORPHAN / BROKEN -- typically mid
    # source-build, where the directory exists but conda-meta has not
    # been finalised yet.
    if state.can_resume or force_resume:
        why = ("already exists" if state.can_resume
               else f"--force-resume; state was {state.state_label}")
        return _undispatched(
            step, Outcome.SKIPPED,
            f"env `{phase.env_name}` {why}; skipping create")
    if state.needs_cleanup:
        return _undispatched(
            step, Outcome.FAILED,
            f"env `{phase.env_name}` is in state {state.state_label} "
            f"-- re-run with --clean to wipe before installing.")
    return None


def _run_phase(
    steps: Sequence[InstallStep],
    phase: _Phase,
    *,
    tag: str,
    executed: List[InstallStep],
    skip_create_if_present: bool = False,
    force_resume: bool = False,
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
        if step.label == "conda create":
            decided = _create_decision(
                step, phase, skip_if_present=skip_create_if_present,
                force_resume=force_resume)
            if decided is not None:
                executed.append(decided)
                _report(where, decided)
                if decided.outcome.stops_the_install:
                    return False
                continue
        elif phase.ensure_prefix() is None:
            # FAIL LOUD.  Dispatching the unbypassed argv would hit mamba
            # 1.x's ``exec --`` stub bug, and the user would be reading an
            # error about the wrong thing entirely.
            executed.append(_undispatched(
                step, Outcome.FAILED,
                f"could not resolve env prefix for `{phase.env_name}` "
                f"-- not dispatching {step.label}.  Run "
                f"`{phase.conda_binary} env list` to confirm the env is "
                f"there; if it is and we cannot find it, file an issue "
                f"with that output."))
            _report(where, executed[-1])
            return False
        sys.stderr.write(f"[{where}] {step.label}: starting\n")
        sys.stderr.flush()
        # ONE DOOR, and the outcome decides what happens next -- no
        # nested conditions over return codes, alternatives and
        # optionality, which is where branches kept going missing.
        done = run_step(step, prefix=phase.prefix, sink=sys.stderr)
        executed.append(done)
        _report(where, done)
        if done.outcome.stops_the_install:
            return False
    return True


def run_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
    skip_create_if_present: bool = True,
    rebuild: Optional[str] = None,
    build_on_warnings: Optional["_builds.ConfirmWarningsCallback"] = None,
    build_on_progress: Optional["_builds.ProgressCallback"] = None,
    build_skip_network_check: bool = False,
    force_resume: bool = False,
) -> InstallResult:
    """Execute the install plan, stopping at the first failed step.

    Parameters
    ----------
    skip_create_if_present
        When ``True`` (default) and the env already exists, the
        ``conda create`` step is reported as a no-op (returncode 0,
        output ``"env already exists; skipping create"``) and the
        remaining phases run normally.  This is what makes ``install``
        idempotent: re-running picks up new pip deps without trying
        to re-create the env.  Set ``False`` only in tests.
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
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = _effective_name(recipe, caps)
    planned = _plan(recipe, effective, caps.conda_binary)

    sys.stderr.write(
        f"[install] recipe `{recipe.name}` -> env `{effective}`\n"
    )
    sys.stderr.write(
        f"[install] env manager: {caps.conda_binary}\n"
    )
    sys.stderr.flush()

    # Cache the env prefix once -- recomputing per step would call
    # ``<mgr> env list / info --json`` 3-5 times PER RECIPE on mamba 2.x,
    # which can stretch a multi-recipe bootstrap into several minutes
    # of silent probing.  We re-resolve only AFTER conda create runs
    # (in case the env is brand new).
    cached_prefix: Optional[str] = _env_prefix(effective, caps.conda_binary)
    if cached_prefix is not None:
        sys.stderr.write(
            f"[install] env prefix: {cached_prefix}\n"
        )
        sys.stderr.flush()
    # NOTE: we DELIBERATELY do not pre-compute ``env_exists`` from caps
    # here.  The conda-create skip decision below uses ``probe_env_state``
    # live -- the cached caps view can be stale (notably right after
    # --clean, when ``get_capabilities()`` returns the bound snapshot
    # rather than re-detecting), and trusting it caused the
    # 2026-06-15 "env already exists; conda may have failed silently"
    # regression.  See the conda-create branch below for the live probe.
    executed: List[InstallStep] = []
    phase = _Phase(env_name=effective, conda_binary=caps.conda_binary,
                   prefix=cached_prefix)

    # Reorder: conda-create + pip + extra_steps + (build_spec) + verify.
    # Verify is pulled out of `planned` and run AFTER the build phase so
    # it checks the built binary rather than the env that will hold it.
    # Both groups go through the SAME runner -- verify's own loop is what
    # used to re-implement the prefix resolution, the `conda run` bypass,
    # the launch-failure branch and the output trim, and to leave every
    # verify step carrying no outcome at all.
    verify_steps = [s for s in planned if s.label == "verify"]
    pre_verify = [s for s in planned if s.label != "verify"]

    ok = _run_phase(pre_verify, phase, tag="install", executed=executed,
                    skip_create_if_present=skip_create_if_present,
                    force_resume=force_resume)

    # Build-spec phase: only if the recipe declares one AND nothing
    # failed before it.  `builds.run_build_spec` keeps its own executor
    # (it has sentinel-based resume, which no install step has), so what
    # happens here is an ADAPTER: its results become InstallSteps whose
    # outcome is decided by the same rule as everything else.
    build_result: Optional[_builds.BuildResult] = None
    if ok and recipe.build_spec is not None:
        if phase.ensure_prefix() is None:
            executed.append(_undispatched(
                InstallStep(label="build",
                            argv=("internal", "resolve-env-prefix")),
                Outcome.FAILED,
                f"could not resolve $CONDA_PREFIX for env {effective!r}; "
                f"conda may have failed silently."))
            _report("build", executed[-1])
            ok = False
        else:
            build_result = _builds.run_build_spec(
                recipe.build_spec, phase.prefix,
                conda_binary=caps.conda_binary,
                rebuild=rebuild,
                conda_specs=list(recipe.conda_specs),
                skip_network_check=build_skip_network_check,
                on_warnings=build_on_warnings,
                on_progress=build_on_progress,
            )
            if build_result.preflight_errors:
                executed.append(_undispatched(
                    InstallStep(label="build:preflight",
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
        _run_phase(verify_steps, phase, tag="verify", executed=executed)

    # DERIVED, not tracked alongside.  A separate `succeeded` bool was
    # what let the word printed to the user disagree with the verdict --
    # two failures announced themselves as "SKIPPED".  `stops_the_install`
    # and not `is_success` is the right predicate: a DEGRADED optional
    # package is honestly absent without the install having failed.
    succeeded = not any(s.outcome.stops_the_install for s in executed)
    if build_result is not None and not build_result.succeeded:
        # builds.py owns its own verdict; do not re-derive it from steps.
        succeeded = False

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
    "create_step_for",
    "conda_step_for",
    "pip_argv",
    "pip_step_for",
    "verify_step_for",
    "InstallStep",
    "Outcome",
    "run_step",
    "OUTPUT_LIMIT",
    # ...and the two entry points built on it.
    "InstallResult",
    "plan_install",
    "run_install",
]
