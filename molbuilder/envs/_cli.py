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
import sys
from typing import Iterable, Optional

import click

from ..diagnostics import get_capabilities
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
def cmd_install(name: str, dry_run: bool, check: bool) -> None:
    """Run a recipe's install plan against the local conda.

    NAME is the recipe's canonical name (e.g., ``molbuilder-pySCF``).
    User-side overrides via ``molbuilder.json`` apply automatically;
    the effective env name is reported in the output.
    """
    if dry_run and check:
        raise click.UsageError("--dry-run and --check are mutually exclusive")

    recipe = recipe_by_name(name)
    if recipe is None:
        registered = ", ".join(r.name for r in BUILTIN_RECIPES)
        raise click.UsageError(
            f"unknown recipe `{name}`.  Registered: {registered}"
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
        return

    click.echo(f"installing `{effective}` ({recipe.description})")
    result = _install.run_install(recipe, caps=caps)
    for step in result.steps:
        click.echo(f"-- {step.label} (rc={step.returncode})")
        if step.output.strip():
            tail = "\n".join(
                "    " + ln
                for ln in step.output.strip().splitlines()[-12:]
            )
            click.echo(tail)
    if result.succeeded:
        click.echo("install OK")
        return
    click.echo("install FAILED -- see step output above", err=True)
    sys.exit(1)


__all__ = ["envs_group"]
