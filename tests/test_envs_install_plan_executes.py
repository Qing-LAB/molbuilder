"""Every step of every install plan, generated AND executed.

Why this file exists.  Two bugs shipped in one day because the *pieces*
were tested and the *whole* was not:

  * the activation wrapper (then `_bypass_conda_run`) logged the inner
    command with
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
from molbuilder.envs import builds as _B
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
    for recipe in R.builtin_recipes():
        # The opt-in steps too: they are generated shell like the rest.
        name, steps = I.plan_install(recipe, include_opt_in=True)
        for st in steps:
            yield name, st.label, tuple(st.argv)


def _wrapped(argv):
    """The generated bash for ``argv``, or None when there is none.

    The wrapper is the FALLBACK path now (`installation.md` M4 -- a manager
    whose `run` is broken), not the route every machine takes.  It is still
    generated shell, so it still gets read, lexed and executed here.
    """
    try:
        return _B.activation_wrapper(argv, "/opt/envs/fake-prefix")[-1]
    except ValueError:
        return None          # conda create: the manager's own command


def test_every_generated_step_is_valid_bash():
    n = 0
    for name, label, argv in _all_steps():
        sh = _wrapped(argv)
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


def test_the_browser_step_confines_itself_to_the_prefix(tmp_path):
    """END RESULT, not the string.  Runs the real chromium step against a
    synthetic prefix with a stub `playwright` and asserts where things land.

    The requirement is confinement (user, 2026-09-18: *"confined within conda
    env ... not required to have elevated privilege or contaminating the
    system"*).  `playwright install` defaults to `~/.cache/ms-playwright` --
    measured 1.3 GB there, outside every env and surviving `conda env
    remove`.

    Executing it proves the hook is written, the download is pointed inside the prefix,
    and $HOME is untouched.
    """
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    fake_home = tmp_path / "home"
    (fake_home / ".cache").mkdir(parents=True)
    # A stub python that records where playwright was told to put the browser.
    stub = prefix / "bin" / "python"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s" "$PLAYWRIGHT_BROWSERS_PATH" > "$0.browsers-path"\n'
        'mkdir -p "$PLAYWRIGHT_BROWSERS_PATH/chromium-stub"\n')
    stub.chmod(0o755)

    argv = _host_step("ms-playwright")
    env = dict(os.environ, HOME=str(fake_home))
    cp = subprocess.run(list(_B.activation_wrapper(argv, str(prefix))),
                        capture_output=True, text=True, timeout=120, env=env)
    assert cp.returncode == 0, f"browser step failed:\n{cp.stderr}"

    told = (prefix / "bin" / "python.browsers-path").read_text()
    assert told == str(prefix / "share" / "ms-playwright"), (
        f"the download was pointed at {told!r}, not inside the prefix")
    assert (prefix / "share" / "ms-playwright" / "chromium-stub").is_dir()
    assert not list((fake_home / ".cache").iterdir()), (
        "something was written to the user's cache instead of the env")

    # The hook is what keeps the path set for the PYTEST process, not just
    # this one -- and it must carry a LITERAL $CONDA_PREFIX so a cloned or
    # moved env stays valid.
    hook = prefix / "etc" / "conda" / "activate.d" / "playwright-browsers.sh"
    assert hook.is_file(), "no activate.d hook: the browser would be looked "\
                           "for outside the env it was downloaded into"
    assert "$CONDA_PREFIX/share/ms-playwright" in hook.read_text()


def _host_step(installs_into: str):
    """The argv the installer plans (`install.extra_steps_for`, opt-in
    included) for the host recipe's extra step that installs into
    ``$CONDA_PREFIX/share/<installs_into>``."""
    steps = I.extra_steps_for(R.recipe_by_name("molbuilder"), "/usr/bin/conda",
                              "x", include_opt_in=True)
    (step,) = [s for s in steps
               if any(f"share/{installs_into}" in a for a in s.argv)]
    return tuple(step.argv)


def test_the_js_test_packages_step_confines_itself_to_the_prefix(tmp_path):
    """END RESULT, not the string -- the browser step's twin
    (`env-framework.md` § 3.2a).  Runs the real step against a synthetic
    prefix, in the environment every install step runs in
    (`builds.env_for_step`), with a stub `npm` that records what it was
    asked -- a test downloads nothing; the real install is
    `--with-dev-tools`'s -- and asserts where things land: the lockfile's packages installed
    by `npm ci` with no install scripts, in the env's own `share/`, npm's
    cache inside the prefix, and $HOME untouched."""
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # A stub npm that records what it was asked, and -- as npm does --
    # writes its log under its cache, ``~/.npm`` unless told otherwise.
    stub = prefix / "bin" / "npm"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n%s\\n%s\\n" "$*" "$PWD" "$npm_config_cache" '
        '> "$0.asked"\n'
        'C="${npm_config_cache:-$HOME/.npm}"; mkdir -p "$C/_logs"; '
        'echo ran > "$C/_logs/debug.log"\n'
        'mkdir -p node_modules/jsdom\n')
    stub.chmod(0o755)

    argv = _host_step("molbuilder-js")
    env = _B.env_for_step(str(prefix), dict(os.environ, HOME=str(fake_home)))
    cp = subprocess.run(list(_B.activation_wrapper(argv, str(prefix))),
                        capture_output=True, text=True, timeout=120, env=env)
    assert cp.returncode == 0, f"JS packages step failed:\n{cp.stderr}"

    args, cwd, cache = (prefix / "bin" / "npm.asked").read_text().split("\n")[:3]
    there = prefix / R.JS_TEST_PACKAGES_HOME
    assert args.split()[0] == "ci" and "--ignore-scripts" in args.split(), (
        f"not the lockfile's install with no install scripts: {args!r}")
    assert cwd == str(there), f"npm ran in {cwd!r}, not inside the prefix"
    assert cache == str(prefix / "var" / "cache" / "npm"), (
        f"npm's cache was {cache!r}, not inside the prefix")
    pins = R._JS_TEST_PACKAGES
    for name in ("package.json", "package-lock.json"):
        assert (there / name).read_text() == (pins / name).read_text(), (
            f"{name} is not the pinned one")
    assert (there / "node_modules" / "jsdom").is_dir()
    assert not list(fake_home.iterdir()), (
        "something was written to the user's home instead of the env")


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
            *recipe.extra_steps[0].argv)
    new_argv = _B.activation_wrapper(argv, str(prefix))
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


def test_the_verify_step_only_warns_about_a_foreign_gcc(tmp_path):
    """The toolchain check must NOT fail an env that merely predates the
    shims -- it warns, names the remedy, and the verify goes on.

    The generated check itself, EXECUTED with a `gcc` outside the env; only
    that part of the command, because the rest of it runs the env's SIESTA, and
    no engine runs in a basic test.  A `molbuilder` on PATH writes down that it
    ran: a backticked remedy inside the double-quoted echo would EXECUTE, an
    install inside an install (this file's header)."""
    recipe = R.recipe_by_name("molbuilder-siesta-gpu")
    script = recipe.verify_argv[-1]
    start = script.index('case "$(command -v gcc)" in')
    check = script[start:script.index("esac;", start) + len("esac;")]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ran = tmp_path / "the-remedy-ran"
    for name, body in (("gcc", "exit 0"), ("molbuilder", f'touch "{ran}"')):
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    cp = subprocess.run(["bash", "-c", check + " echo went-on"],
                        capture_output=True, text=True, timeout=20,
                        env={"PATH": f"{bin_dir}:/usr/bin:/bin",
                             "CONDA_PREFIX": str(tmp_path / "env")})

    # A WARNING, NOT A GATE.
    assert cp.returncode == 0 and "went-on" in cp.stdout, (cp.stdout, cp.stderr)
    assert "WARNING: bare gcc resolves to" in cp.stderr, cp.stderr
    assert not ran.exists(), "the remedy was executed, not said"
    # ...and the remedy NAMES A REGISTERED RECIPE: `recipe_by_name` takes
    # canonical names only, so a remedy naming `siesta-gpu` hands you
    # "unknown recipe 'siesta-gpu'", exit 2.
    assert f"install {recipe.name}" in cp.stderr, cp.stderr
