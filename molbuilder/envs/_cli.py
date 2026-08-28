"""Click subcommands for ``molbuilder envs ...``.

Three subcommands surface the recipe registry:

  * ``molbuilder envs list``    -- short table of every recipe + status
  * ``molbuilder envs doctor``  -- same as list + verify command run
  * ``molbuilder envs install <name>`` -- execute the recipe's plan

Kept in its own module so ``cli.py`` only has to register the group;
all the rendering lives next to the recipe / doctor / install code.
"""
from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, TextIO

import click

from ..diagnostics import get_capabilities, reset_capabilities
from . import advise as _advise
from . import builds as _builds
from . import doctor as _doctor
from . import install as _install
from . import validate as _validate
from .recipes import BUILTIN_RECIPES, recipe_by_name


# --------------------------------------------------------------------- #
#  Install log persistence                                               #
# --------------------------------------------------------------------- #
#
# Every ``molbuilder envs install`` run drops a full copy of its
# terminal output at ``~/.molbuilder/logs/install-<recipe>-<TS>.log``
# so the user can grep / diff / share install transcripts later
# without rerunning the build.  The path is reported at the start of
# the install and again at the end.


def _log_root() -> Path:
    """``~/.molbuilder/logs``, resolved WHEN ASKED, not at import.

    This was a module constant, which froze the real home into the
    module the moment anything imported it -- so a test that isolated
    ``HOME`` afterwards still wrote install logs into the developer's
    actual ``~/.molbuilder/logs`` (five zero-byte files per full-suite
    run, found 2026-08-28 by noticing them appear at the same second a
    suite started).  A path that depends on the environment is a
    QUESTION, and a question is asked when it is asked.
    """
    return Path(os.path.expanduser("~/.molbuilder/logs"))


def _resolve_install_log_path(recipe_name: str) -> Path:
    """Compose the log filename for one install run."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in recipe_name)
    return _log_root() / f"install-{safe}-{ts}.log"


class _TeeStream:
    """Forward writes to two streams (terminal + on-disk log).

    Used to tee ``sys.stderr`` during install so every click.echo
    plus every line streamed by ``run_streaming`` lands in both the
    user's terminal and the persisted log file -- no extra plumbing
    needed in run_install / run_build_spec / the per-phase callbacks.
    """

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        # Best-effort: writes to the persisted log MUST NOT mask
        # terminal output if the disk fills / file system blips.
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
        except Exception:
            pass
        return n

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8") or "utf-8"


@contextlib.contextmanager
def _tee_console_to(log_path: Path):
    """Context manager: tee BOTH ``sys.stdout`` and ``sys.stderr`` to
    ``log_path`` for the duration of the block.  Both streams are
    needed because click.echo defaults to stdout (recap lines, "install
    OK") while ``err=True`` calls and ``run_streaming``'s subprocess
    output go to stderr.  Best-effort -- if the log dir can't be
    created (read-only HOME, etc.) the install still runs and the
    user just doesn't get a persisted log."""
    fh = None
    orig_out = sys.stdout
    orig_err = sys.stderr
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(orig_out, fh)
        sys.stderr = _TeeStream(orig_err, fh)
    except OSError:
        # Bail silently -- install proceeds without a persisted log.
        fh = None
    try:
        yield
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass


@click.group("envs",
             context_settings={"help_option_names": ["-h", "--help"]})
def envs_group() -> None:
    """Inspect and install the conda envs molbuilder dispatches into.

    See docs/ops/installation.md for the prose recipes and the rationale
    for the four-env layout.
    """


# --------------------------------------------------------------------- #
#  list                                                                  #
# --------------------------------------------------------------------- #


@envs_group.command("list", short_help="show every recipe + whether the env exists")
def cmd_list() -> None:
    """One-line-per-recipe summary; ``doctor`` is the verbose form."""
    caps = get_capabilities()
    reports = _doctor.report_all(caps, run_verify=False)
    if not reports:
        click.echo("(no recipes registered)")
        return
    width = max(len(r.effective_name) for r in reports)
    for rep in reports:
        marker = "OK " if rep.present else "-- "
        click.echo(
            f"{marker} {rep.effective_name:<{width}}  "
            f"{rep.recipe.description}"
        )


# --------------------------------------------------------------------- #
#  doctor                                                                #
# --------------------------------------------------------------------- #


def _fix_cmd(action: str, recipe_name: str, *flags: str) -> str:
    """The ONE spelling of an env fix command -- the shell launcher form.

    ``bash scripts/install-env.sh <verb> ...`` works from a bare shell (it
    finds conda, requires the host env, and dispatches), which is exactly
    the situation a person with a broken env is in -- and it is the form
    every hint in this file already taught.  ``doctor`` alone said
    ``molbuilder envs install`` (user, 2026-08-20: a detected problem must
    carry its exact fix command; two spellings of the same fix is how a
    reader ends up touring the docs instead).
    """
    return " ".join(["bash scripts/install-env.sh", action,
                     recipe_name, *flags])


def _render_doctor(reports: Iterable[_doctor.EnvReport]) -> int:
    """Print the doctor report.  Returns a process exit code:
    0 when every present env verified ok (or is verify-skipped),
    1 when any verify failed.

    Missing envs are NOT a failure -- ``doctor`` is informational;
    it's the user's call whether to install them.  ``install`` is
    what acts.
    """
    reports = list(reports)
    if not reports:
        click.echo("(no recipes registered)")
        return 0

    any_failed = False
    for rep in reports:
        click.echo("")
        click.echo(f"==  {rep.effective_name}  ==")
        click.echo(f"    {rep.recipe.description}")
        if rep.effective_name != rep.recipe.name:
            click.echo(
                f"    (default name `{rep.recipe.name}` overridden "
                f"via molbuilder.json)"
            )
        if not rep.present:
            click.echo("    state:   MISSING")
            click.echo("    next:    "
                       + _fix_cmd("install", rep.recipe.name, "--yes"))
            continue
        click.echo("    state:   present")
        if rep.verify_ok is None:
            click.echo("    verify:  skipped")
        elif rep.verify_ok:
            click.echo("    verify:  OK")
        else:
            any_failed = True
            click.echo("    verify:  FAILED")
            if rep.verify_output.strip():
                indented = "\n".join(
                    "        " + ln
                    for ln in rep.verify_output.strip().splitlines()[:8]
                )
                click.echo(indented)
            click.echo("    next:    "
                       + _fix_cmd("repair", rep.recipe.name))
            click.echo("             (if repair finds nothing to fix, "
                       "rebuild from the recipe: "
                       + _fix_cmd("install", rep.recipe.name,
                                  "--clean", "--yes") + ")")
        # Package audit (real check, not just verify smoke test).
        # Required-missing -> FAILED + exits 1; optional-missing
        # (e.g. GPU-only cupy + gpu4pyscf) -> info-only, env still
        # usable for the non-optional code paths.
        if rep.package_audit is not None and rep.package_audit.checked:
            pa = rep.package_audit
            n_total = pa.n_conda_declared + pa.n_pip_declared
            required_issues = [i for i in pa.issues
                               if not i.kind.endswith("-optional")]
            optional_issues = [i for i in pa.issues
                               if i.kind.endswith("-optional")]
            n_required = len(required_issues)
            n_optional = len(optional_issues)
            n_ok = n_total - n_required - n_optional
            if n_required == 0 and n_optional == 0:
                click.echo(
                    f"    audit:   OK ({n_ok}/{n_total} declared "
                    f"packages present + version-matched)"
                )
            elif n_required == 0:
                # Only optional packages missing: env is usable; the
                # specific features those packages gate (GPU,
                # peptide builder, etc.) will be non-functional.
                click.echo(
                    f"    audit:   OK ({n_ok}/{n_total} ok; "
                    f"{n_optional} optional unavailable -- gated "
                    f"features will be disabled)"
                )
                for issue in optional_issues[:15]:
                    click.echo(
                        f"        [{issue.kind}] {issue.spec}  "
                        f"(installed: {issue.found})"
                    )
                click.echo("    enable:  "
                           + _fix_cmd("repair", rep.recipe.name,
                                      "--include-optional")
                           + "   (installs these; the gated features "
                             "turn on)")
            else:
                any_failed = True
                click.echo(
                    f"    audit:   FAILED ({n_ok}/{n_total} ok; "
                    f"{n_required} REQUIRED missing"
                    + (f", {n_optional} optional unavailable"
                       if n_optional else "")
                    + ")"
                )
                for issue in required_issues[:15]:
                    click.echo(
                        f"        [{issue.kind}] {issue.spec}  "
                        f"(installed: {issue.found})"
                    )
                if optional_issues:
                    click.echo(
                        f"        ({n_optional} additional optional "
                        f"unavailable -- not counted as failure)"
                    )
                if n_required > 15:
                    click.echo(
                        f"        ... and {n_required - 15} more required"
                    )
                _version_kinds = ("conda-version", "conda-build")
                _has_missing = any(i.kind not in _version_kinds
                                   for i in required_issues)
                _has_version = any(i.kind in _version_kinds
                                   for i in required_issues)
                # Bare `repair` installs the MISSING packages and skips
                # version/build mismatches by design -- so the command
                # offered must actually fix what was just listed.
                _flags = (("--include-version-fix",) if _has_version
                          else ())
                click.echo("    next:    "
                           + _fix_cmd("repair", rep.recipe.name, *_flags))

                if _has_version and not _has_missing:
                    click.echo("             (only version/build pins "
                               "differ; --include-version-fix is what "
                               "makes repair rebuild those)")

    click.echo("")
    if any_failed:
        click.echo("doctor: one or more envs failed verify or package "
                   "audit -- each carries its `next:` fix command above.",
                   err=True)
        return 1
    return 0


@envs_group.command("repair",
                    short_help="install missing packages flagged by audit")
@click.argument("name")
@click.option("--include-optional", is_flag=True,
              help="also retry installing optional packages (e.g. GPU "
                   "wheels for molbuilder-pySCF).  Default: skip.")
@click.option("--include-version-fix", is_flag=True,
              help="also attempt to install specs whose version / build "
                   "doesn't match the recipe pin.  Default: skip "
                   "(destructive; may rebuild many dependent packages).")
def cmd_repair(name: str, include_optional: bool,
               include_version_fix: bool) -> None:
    """Install missing packages reported by ``doctor``'s audit.

    Designed for non-interactive use on remote HPC nodes: no prompts,
    streams per-package progress to stderr, returns exit code 0 iff
    all REQUIRED missing packages got installed.

    What it does:
      * Runs the same audit doctor uses on the named recipe's env.
      * For ``conda-missing``: ``<mgr> install -n <env> <spec> -y``.
      * For ``pip-missing``: ``<env>/bin/python -m pip install <spec>``
        (through the same bypass wrapper used elsewhere, so PIP_CACHE_DIR
        + activate.d hooks apply).
      * Skips ``*-optional`` issues unless ``--include-optional``.
      * Skips ``conda-version`` / ``conda-build`` issues unless
        ``--include-version-fix`` (those rebuilds can take a long time
        on slow HPC filesystems, so they're opt-in).

    What it does NOT do:
      * Recreate the env (use ``install <recipe> --clean`` for that).
      * Run the recipe's verify_argv (use ``doctor`` for that).
      * Touch packages not declared in the recipe.

    Re-running is safe: if everything is already installed, repair
    exits 0 with "nothing to do".
    """
    caps = get_capabilities()
    recipe = recipe_by_name(name)
    if recipe is None:
        registered = ", ".join(r.name for r in BUILTIN_RECIPES)
        raise click.UsageError(
            f"unknown recipe `{name}`.  Registered: {registered}"
        )
    if caps.conda_binary is None:
        raise click.UsageError(
            "conda/mamba not found; cannot run repair.  See "
            "docs/ops/installation.md."
        )
    effective = _doctor._effective_name(recipe, caps)
    if not caps.env_available(effective):
        click.echo(
            f"env `{effective}` does not exist.  Install it first:",
            err=True,
        )
        click.echo(
            "    " + _fix_cmd("install", recipe.name, "--yes"),
            err=True,
        )
        sys.exit(2)
    prefix_str = _install._env_prefix(effective, caps.conda_binary)
    if prefix_str is None:
        click.echo(
            f"could not resolve env prefix for `{effective}`.  Run "
            f"`{caps.conda_binary} env list` to check.",
            err=True,
        )
        sys.exit(2)
    prefix = Path(prefix_str)
    click.echo(f"[repair] {effective}", err=True)
    click.echo(f"[repair]   env prefix: {prefix}", err=True)
    click.echo(f"[repair]   package manager: {caps.conda_binary}  "
               f"(from {caps.conda_binary_source})", err=True)
    audit = _doctor.audit_packages(prefix, recipe)
    # Partition issues by what we'll act on.
    to_install_conda: list = []
    to_install_pip: list = []
    skipped_optional: list = []
    skipped_version: list = []
    for issue in audit.issues:
        if issue.kind.endswith("-optional"):
            if include_optional:
                if issue.kind.startswith("conda-"):
                    to_install_conda.append(issue.spec)
                else:
                    to_install_pip.append(issue.spec)
            else:
                skipped_optional.append(issue)
            continue
        if issue.kind in ("conda-version", "conda-build"):
            if include_version_fix:
                to_install_conda.append(issue.spec)
            else:
                skipped_version.append(issue)
            continue
        if issue.kind == "conda-missing":
            to_install_conda.append(issue.spec)
        elif issue.kind == "pip-missing":
            to_install_pip.append(issue.spec)
    if not to_install_conda and not to_install_pip:
        if skipped_optional or skipped_version:
            click.echo(
                f"[repair]   nothing to fix among REQUIRED packages.  "
                f"{len(skipped_optional)} optional skipped (pass "
                f"--include-optional to retry), "
                f"{len(skipped_version)} version/build mismatches skipped "
                f"(pass --include-version-fix to address).",
                err=True,
            )
        else:
            click.echo("[repair]   audit clean -- nothing to do.", err=True)
        sys.exit(0)
    failures: list = []
    successes: list = []
    if to_install_conda:
        click.echo(
            f"[repair]   {len(to_install_conda)} conda package(s) "
            f"to install: {' '.join(to_install_conda)}", err=True,
        )
        argv = [
            caps.conda_binary, "install", "-n", effective, "-y",
            *to_install_conda,
        ]
        for ch in recipe.channels:
            argv.extend(["-c", ch])
        rc, _ = _builds.run_streaming(argv, sink=sys.stderr, timeout=1800)
        if rc == 0:
            successes.extend(to_install_conda)
            click.echo("[repair]   conda install: OK", err=True)
        else:
            failures.extend(to_install_conda)
            click.echo(f"[repair]   conda install: FAILED (rc={rc})", err=True)
            if rc in (137, -9):
                # SIGKILL mid-solve: the classic login-node memory cap
                # (seen on ASU Sol 2026-08-21 -- mamba's "Resolving
                # Environment" stage is the hungry part).  Say the fix,
                # not just the number (ops/installation.md's rule: a
                # defect line carries its remedy).
                click.echo(
                    "[repair]   rc=137 means the process was KILLED, "
                    "usually by a login node's memory cap during the "
                    "dependency solve.  Re-run this exact command inside "
                    "a short interactive allocation, e.g. on Sol:\n"
                    "[repair]     salloc -p htc -q public -t 00:30:00 "
                    "-c 4 --mem=16G\n"
                    "[repair]   then, in that shell, the same "
                    "`python -m molbuilder envs repair ...` again.",
                    err=True)
    if to_install_pip:
        click.echo(
            f"[repair]   {len(to_install_pip)} pip package(s) "
            f"to install: {' '.join(to_install_pip)}", err=True,
        )
        # Reuse install.py's pip-step pattern via the bypass wrapper so
        # PIP_CACHE_DIR + activate.d sourcing apply consistently.
        raw_argv = (
            caps.conda_binary, "run", "-n", effective,
            "--no-capture-output",
            "python", "-m", "pip", "install", *to_install_pip,
        )
        try:
            new_argv, _ = _install._bypass_conda_run(raw_argv, prefix_str)
            rc, _ = _builds.run_streaming(
                list(new_argv), sink=sys.stderr, timeout=1800,
            )
        except ValueError as exc:
            rc = None
            click.echo(f"[repair]   pip install: failed to set up "
                       f"wrapper ({exc})", err=True)
        if rc == 0:
            successes.extend(to_install_pip)
            click.echo("[repair]   pip install: OK", err=True)
        else:
            failures.extend(to_install_pip)
            click.echo(f"[repair]   pip install: FAILED (rc={rc})", err=True)
    # Re-audit so the user sees the new state.
    click.echo("[repair]   re-running audit...", err=True)
    audit2 = _doctor.audit_packages(prefix, recipe)
    remaining = [i for i in audit2.issues
                 if not i.kind.endswith("-optional")
                 and i.kind not in ("conda-version", "conda-build")]
    if remaining:
        for issue in remaining[:10]:
            click.echo(
                f"[repair]   still missing: {issue.spec} ({issue.kind})",
                err=True,
            )
    click.echo("", err=True)
    click.echo(
        f"[repair] summary: {len(successes)} installed, "
        f"{len(failures)} failed, "
        f"{len(skipped_optional)} optional skipped, "
        f"{len(skipped_version)} version-fix skipped.",
        err=True,
    )
    sys.exit(0 if not remaining else 1)


@envs_group.command("clean",
                    short_help="delete build-only directories from a "
                               "source-build env (frees disk; keeps "
                               "installed binaries)")
@click.argument("name")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="skip the confirmation prompt (use in batch / cron).")
@click.option("--dry-run", is_flag=True,
              help="show what would be deleted + sizes; delete nothing.")
@click.option("--keep-src", is_flag=True,
              help="keep ``src/`` (source clones).  Useful if you plan "
                   "to run ``install --rebuild=<component>`` later -- "
                   "rebuilding without sources triggers a fresh re-clone.")
def cmd_clean(name: str, auto_yes: bool, dry_run: bool,
              keep_src: bool) -> None:
    """Delete build-only directories from a source-build env's
    artifact root, freeing disk space on $HOME-mounted conda envs.

    Source-build envs (``molbuilder-siesta-gpu`` today) accumulate
    cmake build trees, source clones, ccache, and pip caches under
    ``$CONDA_PREFIX/opt/<artifact_subdir>/``.  The build phase needs
    them; at runtime they are dead weight (typically 3-5 GB combined).
    This command identifies + removes the build-only set, leaving the
    installed binaries (``elpa/``, ``siesta/``), sentinels, logs, and
    toolchain fingerprint intact.

    Safe to re-run: skipped dirs that don't exist anymore are a no-op.
    Safe to interrupt: each dir is independent; partial deletion leaves
    the rest untouched.

    To do a fresh install with the same env, use
    ``install <recipe> --clean`` instead -- that wipes the whole env.
    To rebuild a single component without losing the conda env, use
    ``install <recipe> --rebuild=<component>``.
    """
    import shutil as _shutil
    caps = get_capabilities()
    recipe = recipe_by_name(name)
    if recipe is None:
        registered = ", ".join(r.name for r in BUILTIN_RECIPES)
        raise click.UsageError(
            f"unknown recipe `{name}`.  Registered: {registered}"
        )
    if recipe.build_spec is None:
        raise click.UsageError(
            f"recipe `{recipe.name}` is conda-only (no build_spec); "
            f"there are no build directories to clean.  ``clean`` is "
            f"meaningful only for source-build envs like "
            f"`molbuilder-siesta-gpu`."
        )
    if caps.conda_binary is None:
        raise click.UsageError(
            "conda/mamba not found; cannot resolve env prefix."
        )
    effective = _doctor._effective_name(recipe, caps)
    if not caps.env_available(effective):
        click.echo(
            f"env `{effective}` does not exist -- nothing to clean.",
            err=True,
        )
        return
    prefix_str = _install._env_prefix(effective, caps.conda_binary)
    if prefix_str is None:
        click.echo(
            f"could not resolve env prefix for `{effective}`.  Run "
            f"`{caps.conda_binary} env list` to check.",
            err=True,
        )
        sys.exit(2)
    from .builds import resolve_paths
    paths = resolve_paths(recipe.build_spec, prefix_str)

    # Build-only set: present in build phase, dead weight at runtime.
    # Order matters only for the report; deletion is independent.
    candidates: list = [
        ("build/", paths.build,
         "cmake build trees + object files (usually the biggest)"),
        ("src/", paths.src,
         "source clones (re-cloned by --rebuild=<component>)"),
        (".ccache/", paths.root / ".ccache",
         "compiler cache (speeds up rebuilds; safe to drop)"),
        (".cache/", paths.root / ".cache",
         "pip + XDG cache (re-downloaded on next install)"),
        (".tmp/", paths.root / ".tmp",
         "cmake / openmpi tmp (always safe to drop)"),
    ]
    if keep_src:
        candidates = [c for c in candidates if c[0] != "src/"]

    # Anything in this list, do NOT touch.  Listed for the user-facing
    # report so they understand what survives.
    kept = [
        ("logs/", paths.logs, "install logs (useful for debugging)"),
        (".sentinels/", paths.sentinels,
         "resume markers (small; --force-resume reads them)"),
        (".toolchain-fingerprint", paths.fingerprint_file,
         "toolchain hash (used by --clean / --force-resume)"),
    ]
    # Per-component install dirs (elpa/, siesta/) -- the load-bearing
    # artifacts the activate hook publishes on PATH + LD_LIBRARY_PATH.
    for comp in recipe.build_spec.components:
        kept.append((
            f"{comp.name}/",
            paths.component_install(comp.name),
            "INSTALLED binary -- runtime needs this",
        ))

    def _du(p: Path) -> int:
        """Bytes used by a directory tree.  Returns 0 for missing /
        unreadable paths instead of raising."""
        if not p.exists():
            return 0
        total = 0
        try:
            for entry in p.rglob("*"):
                try:
                    if entry.is_file() and not entry.is_symlink():
                        total += entry.stat().st_size
                except OSError:
                    continue
        except OSError:
            pass
        return total

    def _human(n: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.1f} PiB"

    click.echo(f"[clean] {effective}", err=True)
    click.echo(f"[clean]   artifact root: {paths.root}", err=True)
    if not paths.root.exists():
        click.echo(
            f"[clean] artifact root does not exist -- nothing to clean.",
            err=True,
        )
        return
    click.echo("", err=True)
    click.echo("Build-only directories (CANDIDATES FOR DELETION):", err=True)
    total_bytes = 0
    present: list = []
    for label, path, desc in candidates:
        size = _du(path)
        total_bytes += size
        marker = "  " if path.exists() else "--"
        size_s = _human(size) if path.exists() else "absent"
        click.echo(
            f"  {marker} {label:<26} {size_s:>12}   {desc}",
            err=True,
        )
        if path.exists():
            present.append((label, path))
    click.echo("", err=True)
    click.echo(
        f"  TOTAL reclaimable: {_human(total_bytes)} "
        f"({len(present)} dirs present)",
        err=True,
    )
    click.echo("", err=True)
    click.echo("Will NOT touch (load-bearing or small):", err=True)
    for label, path, desc in kept:
        size = _du(path) if path.exists() else 0
        size_s = _human(size) if path.exists() else "absent"
        click.echo(
            f"     {label:<26} {size_s:>12}   {desc}",
            err=True,
        )

    if dry_run:
        click.echo("", err=True)
        click.echo("[clean] --dry-run: no files deleted.", err=True)
        return
    if not present:
        click.echo("", err=True)
        click.echo("[clean] nothing to delete (build dirs already absent).",
                   err=True)
        return
    if not auto_yes:
        click.echo("", err=True)
        if not click.confirm(
            f"Delete {len(present)} build-only "
            f"director{'y' if len(present) == 1 else 'ies'} "
            f"({_human(total_bytes)})?",
            default=False, err=True,
        ):
            click.echo("[clean] aborted by user.", err=True)
            sys.exit(2)

    click.echo("", err=True)
    failed: list = []
    for label, path in present:
        click.echo(f"[clean] rm -rf {path}", err=True)
        try:
            _shutil.rmtree(path)
        except OSError as exc:
            failed.append((label, exc))
            click.echo(f"[clean]   FAILED: {exc}", err=True)
    click.echo("", err=True)
    if failed:
        click.echo(
            f"[clean] {len(present) - len(failed)}/{len(present)} ok, "
            f"{len(failed)} failed.  Re-run with sudo or check perms.",
            err=True,
        )
        sys.exit(1)
    click.echo(
        f"[clean] freed ~{_human(total_bytes)} "
        f"({len(present)} dirs removed).",
        err=True,
    )
    click.echo(
        f"[clean] runtime binaries intact: `siesta --version` should "
        f"still work after ``conda activate {effective}``.",
        err=True,
    )


@envs_group.command("doctor",
                    short_help="report installed / missing / verified envs")
@click.option("--no-verify", is_flag=True,
              help="skip the per-env verify command (faster).")
def cmd_doctor(no_verify: bool) -> None:
    """Per-recipe report: name, present/missing, verify outcome.

    A missing env is reported but does not fail the command -- run
    ``molbuilder envs install <name>`` to add one.  A present env that
    fails its verify command exits with code 1 so the doctor command
    is usable in CI / startup health checks.
    """
    caps = get_capabilities()
    # WHICH manager, and WHY -- the one door's answer, first line of
    # every doctor run ("the script did not follow the correct
    # pathway" must never again be a silent condition; ASU Sol,
    # 2026-08-21).  A recorded-but-unusable manager is a defect line
    # with its remedy, and the run stops here rather than reporting
    # env states probed through nothing.
    if caps.conda_binary:
        click.echo(f"[doctor] package manager: {caps.conda_binary}  "
                   f"(from {caps.conda_binary_source})", err=True)
    else:
        click.echo(f"[doctor] package manager: NONE -- "
                   f"{caps.conda_binary_source or 'no manager found'}",
                   err=True)
        click.echo("[doctor]   fix: record it once as "
                   "`\"envs\": {\"manager\": \"/abs/path\"}` in "
                   "molbuilder.json, or load/install one "
                   "(on ASU Sol: `module load mamba` in the shell that "
                   "runs molbuilder).", err=True)
        sys.exit(1)
    reports = _doctor.report_all(caps, run_verify=not no_verify)
    sys.exit(_render_doctor(reports))


# --------------------------------------------------------------------- #
#  validate                                                              #
# --------------------------------------------------------------------- #


def _render_validation(report: "_validate.ValidationReport",
                       *, show_output_on_fail: bool) -> int:
    """Print the table + return the process exit code (0 / 1)."""
    if not report.probes:
        click.echo(
            f"`{report.recipe_name}` has no validator defined yet.  "
            f"(Today only molbuilder-siesta-gpu is wired.)",
            err=True,
        )
        return 0
    click.echo(f"Validating {report.recipe_name}:")
    longest = max(len(p.name) for p in report.probes)
    for probe in report.probes:
        tag = "PASS" if probe.passed else "FAIL"
        click.echo(f"  [{tag}] {probe.name.ljust(longest)}  {probe.detail}")
        if not probe.passed and show_output_on_fail and probe.output:
            for line in probe.output.splitlines()[-20:]:
                click.echo(f"        {line}")
    n_pass = sum(1 for p in report.probes if p.passed)
    n_total = len(report.probes)
    if report.all_passed:
        click.echo(f"all {n_total} checks passed")
        return 0
    click.echo(f"{n_pass}/{n_total} checks passed -- env not production-ready",
               err=True)
    return 1


@envs_group.command("advise",
                    short_help="recommend mpi_np/omp/mps for a recipe + this host")
@click.argument("name")
@click.option("--n-atoms", type=int, default=None,
              help="atom count for the eigenproblem (caps mpi_np "
                   "in the recommendation, same rule the runtime uses).")
@click.option("--n-orbitals", type=int, default=None,
              help="orbital count (used to estimate VRAM/rank).  If "
                   "omitted, the VRAM column shows 'n/a'.")
def cmd_advise(name: str,
               n_atoms: Optional[int],
               n_orbitals: Optional[int]) -> None:
    """Print a per-host preset table for one recipe.

    Today only ``siesta-gpu`` has a real advisor; other names return
    a "no advisor" notice.  The advisor probes the host (lscpu /
    nvidia-smi / nvidia-cuda-mps-control) and prints three presets
    side-by-side -- `default` (ELPA 2024.05 throughput optimum),
    `memory` (fewer ranks, more OMP), and `fallback` (single rank,
    no MPS).  The recommended preset for the detected host is echoed
    as ready-to-paste ``MOLBUILDER_*`` env-var exports.
    """
    if name not in ("siesta-gpu", "molbuilder-siesta-gpu"):
        click.echo(
            f"`{name}` has no advisor wired.  Today only `siesta-gpu` "
            f"is supported (it's the only recipe whose performance "
            f"depends on host topology).",
            err=True,
        )
        sys.exit(0)
    probe = _advise.probe_host()
    presets = _advise.recommend(probe, n_atoms=n_atoms, n_orbitals=n_orbitals)
    click.echo(_advise.format_report(
        probe, presets,
        n_atoms=n_atoms, n_orbitals=n_orbitals,
    ))


@envs_group.command("validate",
                    short_help="run post-install correctness probes")
@click.argument("name")
@click.option("--quiet-on-fail", is_flag=True,
              help="don't dump the failing probe's captured output "
                   "(useful for terse CI logs).")
def cmd_validate(name: str, quiet_on_fail: bool) -> None:
    """Run the post-install validator suite for one recipe.

    Where ``doctor`` only confirms the env is present and ``siesta
    --version`` exits 0, ``validate`` runs the upstream-recommended
    sanity tests (SIESTA ``ctest -L simple``, ELPA ``make check``,
    plus a probe for the silent CPU-fallback warning that
    ``nvidia-smi`` cannot detect).  See ``molbuilder/envs/validate.py``
    for the full rationale + sources.

    Exits 0 when every probe passes, 1 otherwise.  Today only
    ``molbuilder-siesta-gpu`` has probes defined; other recipes
    return 0 with a "no validator" notice.
    """
    recipe = recipe_by_name(name)
    if recipe is None:
        registered = ", ".join(r.name for r in BUILTIN_RECIPES)
        raise click.UsageError(
            f"unknown recipe `{name}`.  Registered: {registered}"
        )
    caps = get_capabilities()
    # Need the env to exist and its prefix to be resolvable.
    effective = caps.env_for_category(recipe.category) or recipe.name
    if effective not in caps.conda_envs:
        click.echo(
            f"env `{effective}` is not present.  Install it first:\n"
            f"  molbuilder envs install {name}",
            err=True,
        )
        sys.exit(2)
    env_prefix = _install._env_prefix(effective, caps.conda_binary)
    if env_prefix is None:
        click.echo(
            f"could not resolve $CONDA_PREFIX for env `{effective}` "
            f"(registry says present but path lookup failed)",
            err=True,
        )
        sys.exit(2)
    report = _validate.validate_recipe(name, env_prefix)
    sys.exit(_render_validation(report, show_output_on_fail=not quiet_on_fail))


# --------------------------------------------------------------------- #
#  install                                                               #
# --------------------------------------------------------------------- #


def _shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


@envs_group.command("install",
                    short_help="install (or repair) one recipe")
@click.argument("name")
@click.option("--dry-run", is_flag=True,
              help="print the planned commands; do not run them.")
@click.option("--check", is_flag=True,
              help="report present/verify; do not install.")
@click.option("--rebuild", "rebuild", default=None, metavar="COMPONENT",
              help="for source-build recipes (e.g. molbuilder-siesta-gpu): "
                   "wipe sentinels + build dir for a named component plus "
                   "everything downstream of it.  Pass `all` to rebuild "
                   "every component.  Default behaviour resumes from the "
                   "sentinel set.")
@click.option("--clean", is_flag=True,
              help="WIPE the conda env AND the source-build artifact "
                   "directory, then do a fresh install.  Removes the "
                   "conda env via ``conda env remove -n <name> -y`` "
                   "(every package is gone -- gcc, cmake, openmpi, "
                   "cuda toolkit, etc.) and deletes "
                   "$CONDA_PREFIX/opt/<artifact_subdir>/ (source clones, "
                   "build trees, installed siesta/transiesta/tbtrans "
                   "binaries, logs, sentinels).  Use this for a "
                   "guaranteed-clean start after a failed install, a "
                   "recipe upgrade, or when in doubt.  Source-build "
                   "recipes only.  Destructive: requires explicit "
                   "confirmation unless --yes is also passed.")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="proceed without asking for confirmation.  Required "
                   "for non-interactive runs (CI, headless installs).  "
                   "Source-build recipes ask for confirmation at three "
                   "points without this: before initial install "
                   "(~45 min commitment), before --rebuild=all or "
                   "--clean (destructive), and when preflight surfaces "
                   "a non-fatal warning (sm_80 fallback etc.).")
@click.option("--skip-network-check", is_flag=True,
              help="skip the git ls-remote reachability check.  Use "
                   "when running behind a firewall that blocks "
                   "ls-remote but allows clone.")
@click.option("--force-resume", is_flag=True,
              help="ignore the env-state probe's GHOST / ORPHAN / "
                   "BROKEN hard-stop and try to resume the build "
                   "anyway.  Use when you KNOW the env is partly "
                   "built and just needs the next phase to finish "
                   "(e.g. an in-progress source build that hit a "
                   "transient error).  Source-build recipes only.  "
                   "Skips conda create + sentinel-resume picks up "
                   "where it left off; the per-step retry in "
                   "build_argv (cmake --build || retry || retry-j1) "
                   "absorbs transient parallel-build races like the "
                   "flook lua.h ordering issue.")
def cmd_install(name: str, dry_run: bool, check: bool,
                rebuild: Optional[str],
                clean: bool,
                auto_yes: bool,
                skip_network_check: bool,
                force_resume: bool) -> None:
    """Run a recipe's install plan against the local conda.

    NAME is the recipe's canonical name (e.g., ``molbuilder-pySCF``).
    User-side overrides via ``molbuilder.json`` apply automatically;
    the effective env name is reported in the output.

    For source-build recipes (those carrying a ``build_spec``), pass
    ``--rebuild=<component>`` to force a rebuild of that component +
    everything downstream of it, or ``--rebuild=all`` to rebuild
    everything from scratch.
    """
    if dry_run and check:
        raise click.UsageError("--dry-run and --check are mutually exclusive")

    recipe = recipe_by_name(name)
    if recipe is None:
        registered = ", ".join(r.name for r in BUILTIN_RECIPES)
        raise click.UsageError(
            f"unknown recipe `{name}`.  Registered: {registered}"
        )

    if rebuild is not None:
        if recipe.build_spec is None:
            raise click.UsageError(
                f"--rebuild only applies to source-build recipes; "
                f"`{name}` is a conda-only recipe."
            )
        valid = ("all", "none") + tuple(
            c.name for c in recipe.build_spec.components
        )
        # ELSI is a SIESTA submodule (built inside SIESTA's cmake, not
        # as a separately-listable component) -- accept the alias and
        # remap so users coming from the old siesta-gpu-rebuild.sh
        # wrapper or SIESTA 5.4 INSTALL.md vocabulary don't trip.
        # The pre-2026-06-24 shell-side remap is gone; this is the
        # single source of truth for the rename.
        if (rebuild == "elsi"
                and name == "molbuilder-siesta-gpu"
                and "siesta" in valid):
            click.echo(
                "Note: ELSI is a SIESTA submodule -- rebuilding 'siesta' "
                "(which compiles ELSI as part of SIESTA's cmake).",
                err=True,
            )
            rebuild = "siesta"
        if rebuild not in valid:
            raise click.UsageError(
                f"--rebuild={rebuild!r} unknown; choices: {', '.join(valid)}"
            )

    if clean and recipe.build_spec is None:
        raise click.UsageError(
            f"--clean only applies to source-build recipes; "
            f"`{name}` is a conda-only recipe."
        )

    caps = get_capabilities()

    if check:
        reports = _doctor.report_all(caps, recipes=(recipe,))
        sys.exit(_render_doctor(reports))

    try:
        effective, plan = _install.plan_install(recipe, caps=caps)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if dry_run:
        click.echo(f"# dry-run plan for `{recipe.name}` "
                   f"(effective env: `{effective}`)")
        for step in plan:
            click.echo(f"# -- {step.label} --")
            click.echo(_shell_join(step.argv))
        if recipe.build_spec is not None:
            click.echo("")
            # If env exists, probe it.  Otherwise probe with $HOME
            # for the disk check; env-specific tools (gcc, openmpi)
            # show up as "(detected after conda create)".  This
            # avoids the misleading "env's gcc 11.4" line that would
            # otherwise just be the system gcc.
            import os as _os
            env_for_probe = None
            disk_path = _os.path.expanduser("~")
            if caps.env_available(effective):
                # We have an env -- find its prefix for an honest probe.
                from . import install as _install_mod
                env_for_probe = _install_mod._env_prefix(
                    effective, caps.conda_binary,
                )
                if env_for_probe:
                    disk_path = env_for_probe
            probe = _builds.probe_toolchain(env_for_probe or "/nonexistent")
            click.echo(_builds.format_install_summary(
                recipe.build_spec, probe, rebuild=rebuild,
            ))
            if env_for_probe is None:
                click.echo("Detection note (env not yet created):")
                click.echo("  * gcc / OpenMPI / env health checks run after "
                           "`conda create` completes;")
                click.echo("  * host-side probes (CUDA, GPU compute cap, "
                           "disk) are accurate now.")
                click.echo("")
            click.echo(_builds.format_preflight_report(
                _builds.preflight(
                    recipe.build_spec,
                    probe,
                    recipe.conda_packages,
                    env_prefix=disk_path,
                    check_network=False,  # dry-run avoids network ls-remote
                )
            ))
            click.echo("")
            click.echo("(--dry-run: no subprocess executed.  Remove "
                       "--dry-run to proceed.)")
        return

    click.echo(f"installing `{effective}` ({recipe.description})")
    if rebuild:
        click.echo(f"  --rebuild={rebuild}")
    if clean:
        click.echo(f"  --clean (REMOVE conda env + WIPE artifact dir, "
                   f"then fresh install)")

    # === Step 0: probe + diagnose conda env state up front ===
    # Resolve the conda env's state BEFORE any subprocess work runs.
    # Catches all the edge cases (orphan dirs from prior failed
    # installs, ghost registry entries, missing conda-meta) in one
    # cheap probe instead of failing 10 minutes into ``conda create``.
    click.echo("")
    click.echo("Conda env state check:")
    state = _install.probe_env_state(effective, caps.conda_binary)
    click.echo(state.describe())
    if state.needs_cleanup and not clean and not force_resume:
        click.echo("")
        click.echo("HARD STOP: env is in a state that conda create cannot")
        click.echo(f"recover from on its own ({state.state_label}).")
        click.echo("")
        click.echo(
            "If you know the env directory IS usable (e.g. mid-source-"
            "build), bypass this check with --force-resume:"
        )
        click.echo("")
        click.echo(
            "    " + _fix_cmd("install", name, "--force-resume", "--yes")
        )
        click.echo("")
        click.echo("Copy-paste to fix (wipes + reinstalls):")
        click.echo("")
        click.echo(
            "    " + _fix_cmd("install", name, "--clean", "--yes")
        )
        click.echo("")
        click.echo("Or the manual equivalent:")
        click.echo("")
        # Use the detected env-manager binary, not a hardcoded ``mamba``.
        # User might have only conda installed; ``mamba env remove``
        # would fail with "mamba: command not found" in that case.
        click.echo(f"    {caps.conda_binary} env remove -n {name} -y")
        click.echo(
            "    " + _fix_cmd("install", name, "--yes")
        )
        sys.exit(2)
    if state.state_label == "PRESENT" and not clean:
        click.echo("")
        click.echo("Install will RESUME on this env (conda create skipped,")
        click.echo("sentinel-protected build phases short-circuit when valid).")

    # For source-build recipes, detect existing artifact state up
    # front so the user knows whether this is a fresh install, a
    # resume, or a wipe.
    if recipe.build_spec is not None:
        env_prefix_for_state = (
            _install._env_prefix(effective, caps.conda_binary)
            if caps.env_available(effective) else None
        )
        if env_prefix_for_state:
            paths_for_state = _builds.resolve_paths(
                recipe.build_spec, env_prefix_for_state,
            )
            if paths_for_state.root.exists():
                stale = _builds.detect_stale_artifact_dirs(
                    recipe.build_spec, env_prefix_for_state,
                )
                click.echo("")
                click.echo("Existing artifact directory detected:")
                click.echo(f"  {paths_for_state.root}")
                if clean:
                    click.echo(
                        "  → --clean will WIPE this directory (sources, "
                        "build trees, installed binaries, logs, "
                        "sentinels) before starting."
                    )
                elif rebuild == "all":
                    click.echo(
                        "  → --rebuild=all will wipe per-component "
                        "install + build dirs (keeping src/ clones)."
                    )
                else:
                    click.echo(
                        "  → resuming: each component is probed at "
                        "install start (install dir + verify); ones "
                        "that pass are SKIPPED end-to-end.  Pass "
                        "--clean to wipe everything, or "
                        "--rebuild=<component> to force one component "
                        "+ everything downstream to rebuild."
                    )
                if stale:
                    click.echo(
                        f"  ⚠ stale entries (from a prior recipe "
                        f"version or failed install): {', '.join(stale)}"
                    )

    # Execute --clean BEFORE running run_install -- this is destructive,
    # so we ask for explicit confirmation independent of the post-summary
    # confirmation.  --clean does TWO things:
    #
    #   1. Remove the conda env entirely (``conda env remove -n <name>
    #      --all -y``) if it exists.  The downstream install then runs
    #      ``conda create`` to make a fresh env -- equivalent to the
    #      first-install state.
    #   2. Wipe the source-build artifact dir at
    #      ``$CONDA_PREFIX/opt/<artifact_subdir>/`` if it survives.
    #      Usually step 1 takes the dir with it (it lived inside the
    #      env's prefix), so this is belt-and-suspenders.
    if clean and recipe.build_spec is not None:
        env_exists_pre_clean = caps.env_available(effective)
        env_prefix = (
            _install._env_prefix(effective, caps.conda_binary)
            if env_exists_pre_clean else None
        )
        artifact_root = None
        if env_prefix:
            paths = _builds.resolve_paths(recipe.build_spec, env_prefix)
            if paths.root.exists():
                artifact_root = paths.root

        if env_exists_pre_clean or artifact_root:
            click.echo("")
            click.echo("=" * 64)
            click.echo("  --clean: ENV + ARTIFACTS WILL BE WIPED")
            click.echo("=" * 64)
            if env_exists_pre_clean:
                click.echo(f"  Conda env to REMOVE:")
                click.echo(f"    {effective}")
                if env_prefix:
                    click.echo(f"      (prefix: {env_prefix})")
            if artifact_root:
                click.echo(f"  Artifact directory to DELETE:")
                click.echo(f"    {artifact_root}")
                try:
                    for entry in sorted(artifact_root.iterdir()):
                        click.echo(f"      - {entry.name}/")
                except OSError:
                    pass
            click.echo("")
            click.echo("  This wipes the conda env (every package -- gcc,")
            click.echo("  cmake, openmpi, cuda toolkit, etc.) AND any")
            click.echo("  source-built artifacts (siesta/transiesta/tbtrans")
            click.echo("  binaries, build trees, logs, sentinels).  A fresh")
            click.echo("  install runs after, equivalent to first-time setup.")
            click.echo("")
            if not auto_yes:
                if not click.confirm(
                        "Proceed with wipe?", default=False,
                ):
                    click.echo("aborted by user (--clean declined)")
                    sys.exit(0)

            # Step 1: remove conda env.
            if env_exists_pre_clean:
                click.echo(f"removing conda env: {effective}")
                try:
                    subprocess.run(
                        [caps.conda_binary, "env", "remove",
                         "-n", effective, "-y"],
                        check=True,
                    )
                    click.echo(f"removed conda env {effective}")
                except subprocess.CalledProcessError as exc:
                    click.echo(f"FAILED to remove conda env: {exc}",
                               err=True)
                    click.echo(f"  (you may need to run this manually:",
                               err=True)
                    click.echo(
                        f"   conda env remove -n {effective} -y)",
                        err=True,
                    )
                    sys.exit(1)

            # Step 2: wipe artifact dir if it survived (usually it was
            # inside the env's prefix and went with step 1, but for
            # defensiveness).
            if artifact_root and artifact_root.exists():
                import shutil as _shutil
                _shutil.rmtree(artifact_root)
                click.echo(f"wiped {artifact_root}")

            # Refresh capabilities so the downstream install knows
            # the env is gone (conda create will run fresh).  IMPORTANT:
            # ``get_capabilities()`` returns the previously-bound
            # snapshot when one exists; without ``reset_capabilities()``
            # first, this is a no-op and the install proceeds with a
            # stale ``conda_envs`` that still lists the removed env.
            reset_capabilities()
            caps = get_capabilities()
            click.echo("")

    # For source-build recipes, surface the install summary + ask for
    # confirmation BEFORE any subprocess runs.  The 45-min commitment
    # + 12 GB disk footprint warrants the speedbump.
    if recipe.build_spec is not None and not auto_yes:
        # Best-effort probe BEFORE conda create has run: env may not
        # exist yet, in which case the probe returns mostly None.  We
        # use it only for the build-job count + cost summary.
        probe_for_summary = _builds.probe_toolchain(
            "/" if not caps.env_available(effective) else effective,
        )
        click.echo("")
        click.echo(_builds.format_install_summary(
            recipe.build_spec, probe_for_summary, rebuild=rebuild,
        ))
        if rebuild == "all":
            click.echo("WARNING: --rebuild=all will wipe every build dir "
                       "+ install dir; only `src/` is preserved.")
            click.echo("")
        if not click.confirm("Proceed?", default=True):
            click.echo("aborted by user.")
            sys.exit(0)
        click.echo("")

    # Hook the build executor's progress into the CLI's output.
    # State holds the running step count + total so the callback can
    # render "[N/total]" headers without a closure variable race.
    state = {"i": 0, "total": 0}

    def on_warnings(report: "_builds.PreflightReport") -> bool:
        click.echo("")
        click.echo(_builds.format_preflight_report(report))
        click.echo("")
        if auto_yes:
            return True
        return click.confirm("Proceed despite warnings?", default=True)

    def on_progress(event: str, step: "_builds.BuildStep",
                    _result) -> None:
        if event == "start":
            state["i"] += 1
            click.echo(_builds.format_progress_event(
                event, step, state["i"], state["total"],
            ))
        elif event == "skip":
            state["i"] += 1
            click.echo(_builds.format_progress_event(
                event, step, state["i"], state["total"],
            ))
        elif event in ("ok", "fail"):
            click.echo(_builds.format_progress_event(
                event, step, state["i"], state["total"],
            ))
            if event == "fail" and _result is not None and _result.output:
                tail = "\n".join(
                    "    " + ln
                    for ln in _result.output.strip().splitlines()[-12:]
                )
                click.echo(tail)

    # Pre-count total steps so the progress callback can render N/total.
    if recipe.build_spec is not None:
        state["total"] = sum(
            5 if c.verify_argv else 4 for c in recipe.build_spec.components
        )

    # Persist the full install transcript so the user can grep / diff /
    # share it later without re-running the build (which can take 30-45
    # min for siesta-gpu).  Path is reported up front (so it's grep-
    # able even if the install crashes) and again on completion.
    log_path = _resolve_install_log_path(recipe.name)
    click.echo(f"install log: {log_path}")

    with _tee_console_to(log_path):
        result = _install.run_install(
            recipe, caps=caps, rebuild=rebuild,
            build_on_warnings=on_warnings,
            build_on_progress=on_progress,
            build_skip_network_check=skip_network_check,
            force_resume=force_resume,
        )

        # If the build_spec executor short-circuited on preflight errors,
        # print them PROMINENTLY before the per-step recap.  Previously
        # the ``build:preflight`` step was silently filtered out by the
        # ``startswith("build:")`` skip rule so the user got "install
        # FAILED" with zero diagnostic info.  This is the failure mode
        # the user hit on 2026-06-15 when ELPA's empty repo_url caused a
        # check_repo_reachable failure that never reached the terminal.
        if (result.build_result is not None
                and result.build_result.preflight_errors):
            click.echo("", err=True)
            click.echo("=" * 64, err=True)
            click.echo("  BUILD PREFLIGHT FAILED -- install cannot proceed",
                       err=True)
            click.echo("=" * 64, err=True)
            for err_msg in result.build_result.preflight_errors:
                for line in err_msg.splitlines():
                    click.echo(f"  ! {line}", err=True)
            click.echo("", err=True)

        # Per-step recap.  Build steps (component.phase) were already
        # streamed live by on_progress, so we skip those.  But the
        # ``build:preflight`` pseudo-step is the EXCEPTION: preflight
        # never runs through on_progress (it short-circuits before
        # phases), so if we skipped it here too the user sees nothing.
        # The block above already printed preflight_errors loudly, so
        # the recap still skips it (avoiding duplication).
        for step in result.steps:
            if step.label.startswith("build:"):
                continue
            click.echo(f"-- {step.label} (rc={step.returncode})")
            if step.output.strip():
                tail = "\n".join(
                    "    " + ln
                    for ln in step.output.strip().splitlines()[-12:]
                )
                click.echo(tail)

        if result.succeeded:
            click.echo("install OK")
            if (result.build_result is not None
                    and result.build_result.activate_hook_written):
                click.echo(
                    f"  activate hook: $CONDA_PREFIX/etc/conda/activate.d/"
                    f"zz-{recipe.build_spec.artifact_subdir}.sh"
                )
                click.echo(
                    f"  binaries:      $CONDA_PREFIX/opt/"
                    f"{recipe.build_spec.artifact_subdir}/siesta/bin/"
                )
                click.echo(
                    f"  re-activate the env (`conda activate {effective}`) "
                    f"to pick up the new PATH + LD_LIBRARY_PATH."
                )
        else:
            click.echo("install FAILED -- see step output above", err=True)

    # Echo the log path AGAIN after the tee block closes so it lands
    # in the user's terminal even if scrollback ate the leading line.
    click.echo(f"install log saved: {log_path}")
    if not result.succeeded:
        sys.exit(1)


# --------------------------------------------------------------------- #
#  bootstrap -- install every conda-only recipe + run doctor             #
# --------------------------------------------------------------------- #


@envs_group.command(
    "bootstrap",
    short_help="install every conda-only recipe + run doctor at the end")
@click.option("--dry-run", is_flag=True,
              help="print the plan; do not install anything.")
@click.option("--skip-existing", is_flag=True, default=True,
              help="default: skip recipes whose env already exists.  "
                   "Pass ``--no-skip-existing`` to re-run install on "
                   "envs that are already present (idempotent re-pass).")
@click.option("--no-skip-existing", "skip_existing", flag_value=False,
              help="re-run install on envs that are already present.")
@click.option("--include-source-builds", is_flag=True,
              help="also bootstrap recipes that build from source (e.g. "
                   "``molbuilder-siesta-gpu``).  Default excludes them "
                   "because each takes ~30-45 minutes and a fresh "
                   "bootstrap usually doesn't want to commit to that "
                   "up front.  When --include-source-builds is passed, "
                   "the user is asked to confirm before each source "
                   "build starts unless --yes is also given.")
@click.option("--yes", "-y", "auto_yes", is_flag=True,
              help="proceed without interactive confirmation.  Required "
                   "for headless / CI / HPC-batch use.")
def cmd_bootstrap(dry_run: bool, skip_existing: bool,
                  include_source_builds: bool, auto_yes: bool) -> None:
    """Install every registered recipe, then run ``doctor`` for the
    summary.

    The bootstrap path is intended for first-run deployment on an
    HPC / supercomputer cluster (or any fresh machine) where the user
    wants the full molbuilder env stack present in one command.
    Existing envs are skipped by default so the command is idempotent
    and safe to re-run.

    Source-build recipes (e.g. ``molbuilder-siesta-gpu``) are NOT
    included by default because each takes 30-45 minutes.  Use
    ``--include-source-builds`` to opt in.

    At the end runs ``molbuilder envs doctor`` so the user sees the
    health of every env in one report.
    """
    caps = get_capabilities()

    # Pick recipes to install.  Order: conda-only first (cheap, fast),
    # then source-builds last (slow) only when opted in.
    conda_only   = [r for r in BUILTIN_RECIPES if r.build_spec is None]
    source_builds = [r for r in BUILTIN_RECIPES if r.build_spec is not None]

    # Progress visibility: the steps from here through ``run_install``
    # do multiple ``<mgr> env list`` / ``<mgr> info --json`` probes (env
    # presence, env-prefix resolution, env-state classification).  On
    # mamba 2.x with a populated registry these can each take several
    # seconds.  Print before EACH probe so the user can see exactly
    # what we're waiting on, not a silent multi-minute hang.
    click.echo(f"[bootstrap] env manager: {caps.conda_binary}", err=True)
    click.echo(
        f"[bootstrap] registered recipes: "
        f"{len(conda_only)} conda-only + {len(source_builds)} source-build",
        err=True,
    )

    plan: list = list(conda_only)
    if include_source_builds:
        plan.extend(source_builds)
    else:
        # Tell the user explicitly what's being skipped.
        if source_builds:
            names = ", ".join(r.name for r in source_builds)
            click.echo(
                f"(skipping source-build recipes: {names}; "
                f"pass --include-source-builds to opt in)",
                err=True,
            )

    if not plan:
        click.echo("(no recipes registered)")
        return

    # Filter out present envs by default.
    if skip_existing:
        click.echo("[bootstrap] checking which envs are already "
                   "present...", err=True)
        before = len(plan)
        new_plan: list = []
        for r in plan:
            env_name = caps.env_for_category(r.category) or r.name
            present = caps.env_available(env_name)
            click.echo(
                f"[bootstrap]   {env_name:<30}  "
                f"{'present (will skip + validate at end)' if present else 'missing (will install)'}",
                err=True,
            )
            if not present:
                new_plan.append(r)
        plan = new_plan
        skipped = before - len(plan)
        if skipped:
            click.echo(
                f"[bootstrap] skipping {skipped} env(s) already present; "
                f"doctor at the end will verify them.  Pass "
                f"--no-skip-existing to re-run install on them.",
                err=True,
            )

    if not plan:
        click.echo("All registered envs are already present.  "
                   "Running doctor to verify.")
    else:
        # Per-recipe banner.
        click.echo(f"bootstrap plan: {len(plan)} env(s)")
        for r in plan:
            click.echo(
                f"  - {r.name:<30}  {r.description}")
        if dry_run:
            click.echo("(dry-run: no install executed.)")
            return
        if not auto_yes and not click.confirm(
                f"Proceed with bootstrap of {len(plan)} env(s)?",
                default=True):
            click.echo("aborted.")
            sys.exit(0)
        # Run installs sequentially.  Failures are recorded but
        # bootstrap continues so the user gets a full report at the
        # end.
        failures: list = []
        for i, recipe in enumerate(plan, 1):
            click.echo("")
            click.echo("=" * 70)
            click.echo(f"[{i}/{len(plan)}] {recipe.name}")
            click.echo("=" * 70)
            log_path = _resolve_install_log_path(recipe.name)
            with _tee_console_to(log_path):
                try:
                    result = _install.run_install(recipe, caps=caps)
                except RuntimeError as e:
                    click.echo(f"  ERROR: {e}", err=True)
                    failures.append(
                        (recipe.name, "run_install raised", str(e)))
                    continue
            click.echo(f"  log: {log_path}")
            if not result.succeeded:
                failures.append((recipe.name, "install step failed",
                                 f"see {log_path}"))

        # Final summary banner.
        click.echo("")
        click.echo("=" * 70)
        if failures:
            click.echo(f"bootstrap finished with {len(failures)} failure(s):",
                       err=True)
            for name, why, hint in failures:
                click.echo(f"  - {name}: {why} -- {hint}", err=True)
        else:
            click.echo("bootstrap complete; every recipe installed.")

    # Refresh capabilities so doctor sees newly-created envs.
    from .. import diagnostics as _diag
    caps_after = _diag.detect()
    _diag.set_capabilities(caps_after)
    click.echo("")
    click.echo("=" * 70)
    click.echo("doctor (post-bootstrap smoke check):")
    click.echo("=" * 70)
    reports = _doctor.report_all(caps_after, run_verify=True)
    exit_code = _render_doctor(reports)

    # "What now?" -- show the most common next-step commands so the
    # user knows how to proceed once bootstrap finishes.  Includes
    # the GPU SIESTA install (commonly missed because source builds
    # are opt-in) + ``repair`` for any package gaps doctor's audit
    # surfaced + ``doctor`` itself for re-verification.  Always shown
    # at the end of bootstrap, even on success, so the GPU path is
    # discoverable from the bootstrap output alone.
    click.echo("")
    click.echo("=" * 70)
    click.echo("NEXT STEPS (copy-paste any of these):")
    click.echo("=" * 70)
    if not include_source_builds:
        click.echo("")
        click.echo("# Install GPU SIESTA (source-built, ~45 min, ~12 GB disk;")
        click.echo("# host env from this bootstrap is the prerequisite):")
        click.echo(
            _fix_cmd("install", "molbuilder-siesta-gpu", "--yes")
        )
    click.echo("")
    click.echo("# Repair any REQUIRED package gaps doctor's audit surfaced:")
    click.echo(_fix_cmd("repair", "<recipe-name>"))
    click.echo("")
    click.echo("# Re-verify env health after any change:")
    click.echo("bash scripts/install-env.sh doctor")
    click.echo("")
    click.echo("# Full --help:")
    click.echo("bash scripts/install-env.sh --help")
    click.echo("")

    if 'failures' in dir() and failures:
        # Already had install failures -- exit non-zero to signal CI.
        sys.exit(1)
    sys.exit(exit_code)


__all__ = ["envs_group"]
