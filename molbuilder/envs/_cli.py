"""Click subcommands for ``molbuilder envs ...``.

Three subcommands surface the recipe registry:

  * ``molbuilder envs list``    -- short table of every recipe + status
  * ``molbuilder envs doctor``  -- same as list + verify command run
  * ``molbuilder envs install <name>`` -- execute the recipe's plan

Kept in its own module so ``cli.py`` only has to register the group;
all the rendering lives next to the recipe / doctor / install code.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Iterable, Optional

import click

from ..diagnostics import get_capabilities
from . import builds as _builds
from . import doctor as _doctor
from . import install as _install
from .recipes import BUILTIN_RECIPES, recipe_by_name


@click.group("envs",
             context_settings={"help_option_names": ["-h", "--help"]})
def envs_group() -> None:
    """Inspect and install the conda envs molbuilder dispatches into.

    See docs/README_install.md for the prose recipes and the rationale
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
            click.echo("    install: molbuilder envs install "
                       f"{rep.recipe.name}")
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

    click.echo("")
    if any_failed:
        click.echo("doctor: one or more envs failed verify.  See "
                   "above + docs/README_install.md.", err=True)
        return 1
    return 0


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
    reports = _doctor.report_all(caps, run_verify=not no_verify)
    sys.exit(_render_doctor(reports))


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
                   "conda env via ``conda env remove -n <name> --all "
                   "-y`` (every package is gone -- gcc, cmake, openmpi, "
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
def cmd_install(name: str, dry_run: bool, check: bool,
                rebuild: Optional[str],
                clean: bool,
                auto_yes: bool,
                skip_network_check: bool) -> None:
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
                        "  → resuming: phases with valid sentinels "
                        "(matching toolchain fingerprint) will be "
                        "SKIPPED.  Pass --clean to wipe everything, "
                        "or --rebuild=all to wipe per-component dirs."
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
                         "-n", effective, "--all", "-y"],
                        check=True,
                    )
                    click.echo(f"removed conda env {effective}")
                except subprocess.CalledProcessError as exc:
                    click.echo(f"FAILED to remove conda env: {exc}",
                               err=True)
                    click.echo(f"  (you may need to run this manually:",
                               err=True)
                    click.echo(
                        f"   conda env remove -n {effective} --all -y)",
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
            # the env is gone (conda create will run fresh).
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

    result = _install.run_install(
        recipe, caps=caps, rebuild=rebuild,
        build_on_warnings=on_warnings,
        build_on_progress=on_progress,
        build_skip_network_check=skip_network_check,
    )
    for step in result.steps:
        if step.label.startswith("build:"):
            # Already streamed by on_progress; skip the trailing recap.
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
        return
    click.echo("install FAILED -- see step output above", err=True)
    sys.exit(1)


__all__ = ["envs_group"]
