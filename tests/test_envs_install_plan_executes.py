"""Every step of every install plan, generated AND executed.

Why this file exists.  Two bugs shipped in one day because the *pieces*
were tested and the *whole* was not:

  * `_bypass_conda_run` logged the inner command with
    ``echo "...{shlex.quote(x)}"`` -- single quotes inside a
    double-quoted echo.  Any command containing a double quote closed
    the echo early and the rest was parsed as shell.  The first step
    with one died with ``syntax error near unexpected token '('`` --
    from the DIAGNOSTIC line, about a valid command.
  * the siesta-gpu verify message said "Re-run `molbuilder envs install
    siesta-gpu`".  Backticks inside a double-quoted shell string are
    COMMAND SUBSTITUTION: verify would have run an install, recursively.

Neither was visible in the Python that generates the shell.  Both were
obvious on reading the generated bash.  So these tests generate it,
lex it, syntax-check it, scan it for substitution hazards, and -- for
the steps that are safe to run -- EXECUTE them against a synthetic env
prefix and assert the end result.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install as I
from molbuilder.envs import recipes as R

_PREFIXED = "x86_64-conda-linux-gnu"


@pytest.fixture(autouse=True)
def _caps():
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary="/usr/bin/conda",
                                  conda_envs=frozenset()))


def _all_steps():
    """(recipe_name, label, argv) for every step of every install plan."""
    for recipe in R.BUILTIN_RECIPES:
        name, steps = I.plan_install(recipe)
        for st in steps:
            yield name, st.label, tuple(st.argv)


def _bypassed(argv):
    """The generated bash for ``argv``, or None when not bypassed."""
    try:
        new, _ = I._bypass_conda_run(argv, "/opt/envs/fake-prefix")
    except ValueError:
        return None          # conda create: plain argv, no shell wrapper
    return new[-1]


def test_every_generated_step_is_valid_bash():
    n = 0
    for name, label, argv in _all_steps():
        sh = _bypassed(argv)
        if sh is None:
            continue
        cp = subprocess.run(["bash", "-n", "-c", sh],
                            capture_output=True, text=True, timeout=30)
        assert cp.returncode == 0, (
            f"{name}/{label}: generated bash is invalid:\n{cp.stderr}")
        shlex.split(sh)      # independent lexer: quoting must balance
        n += 1
    assert n >= 3, f"expected several bypassed steps, saw {n}"


def test_no_step_smuggles_a_command_substitution_into_a_message():
    """Backticks and ``$(`` inside a message are executed, not printed.

    Allowed: the deliberate ``$(command -v gcc)`` in the toolchain
    warning, which is meant to run.  Everything else in prose is a
    mistake -- this catches the class, not the instance.
    """
    for name, label, argv in _all_steps():
        for arg in argv:
            assert "`" not in arg, (
                f"{name}/{label}: backtick in a shell argument -- inside "
                f"double quotes that is command substitution, not "
                f"punctuation:\n  {arg[:200]}")


def test_the_toolchain_shim_step_actually_creates_the_links(tmp_path):
    """End result, not shape.  Runs the real step against a synthetic
    prefix and asserts the bare names exist and point at the conda
    toolchain -- and that a pre-existing tool is never shadowed."""
    prefix = tmp_path / "env"
    binp = prefix / "bin"
    binp.mkdir(parents=True)
    for t in ("gcc", "g++", "gfortran", "ar", "ranlib", "ld", "nm", "strip"):
        p = binp / f"{_PREFIXED}-{t}"
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
    # A real tool already under its bare name: must survive untouched.
    real = binp / "ranlib"
    real.write_text("#!/bin/sh\necho real\n")
    real.chmod(0o755)

    recipe = R.recipe_by_name("molbuilder-siesta-gpu")
    assert recipe.extra_steps, "siesta-gpu must carry the shim step"
    argv = ("/usr/bin/conda", "run", "-n", "x", "--no-capture-output",
            *recipe.extra_steps[0])
    new_argv, _ = I._bypass_conda_run(argv, str(prefix))
    cp = subprocess.run(list(new_argv), capture_output=True, text=True,
                        timeout=120)
    assert cp.returncode == 0, f"shim step failed:\n{cp.stderr}"

    for bare in ("gcc", "g++", "gfortran", "cc", "c++", "ar", "ld", "nm"):
        link = binp / bare
        assert link.is_symlink(), f"{bare} was not created"
        assert os.readlink(link).startswith(_PREFIXED)
    assert not real.is_symlink(), "a pre-existing tool was shadowed"
    assert real.read_text().endswith("echo real\n")

    # Idempotent: a second run must create nothing and still succeed.
    cp2 = subprocess.run(list(new_argv), capture_output=True, text=True,
                         timeout=120)
    assert cp2.returncode == 0
    assert re.search(r"created 0 bare-name", cp2.stderr), cp2.stderr


def test_the_verify_step_runs_and_only_warns_about_a_foreign_gcc(tmp_path):
    """The toolchain check must NOT fail an env that merely predates the
    shims -- it warns and names the remedy.  Executed with a stub
    ``siesta`` so the real binary is not needed."""
    prefix = tmp_path / "env"
    binp = prefix / "bin"
    binp.mkdir(parents=True)
    stub = binp / "siesta"
    stub.write_text("#!/bin/sh\necho 'siesta 5.4.2'\n")
    stub.chmod(0o755)

    recipe = R.recipe_by_name("molbuilder-siesta-gpu")
    argv = ("/usr/bin/conda", "run", "-n", "x", "--no-capture-output",
            *recipe.verify_argv)
    new_argv, _ = I._bypass_conda_run(argv, str(prefix))
    cp = subprocess.run(list(new_argv), capture_output=True, text=True,
                        timeout=120)
    assert cp.returncode == 0, (
        f"verify must not fail an env without shims:\n{cp.stderr}")
    assert recipe.verify_expect_contains in cp.stdout
    assert "WARNING: bare gcc resolves to" in cp.stderr
    # ...and the remedy is printed, not executed.
    assert "molbuilder envs install siesta-gpu" in cp.stderr
