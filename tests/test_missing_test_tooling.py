"""The one rule for what a test needs beyond the host env (`support.tooling`):
a test that needs the opt-in tooling skips, naming the command that installs
it (`process/testing.md` § 4) -- never errors, never a reason of its own --
and a machine without the tooling still collects every file; a test that
needs an env, the env manager or a GPU skips by the same one answer, before
any of its fixtures is set up (§ 3).

THE SILENT FAILURES it guards: on a machine without the browser tooling, a
browser test ERRORING with "fixture 'page' not found" -- a broken suite, to
a reader -- after its other fixtures, real runs among them, were set up; a
file that imports the browser library at its top failing the whole
collection; a test that only parametrizes a value named like a browser
fixture skipped for a browser it does not use; a test on an env without Node
skipping with a reason that names no way to install it; a test whose run's
env is absent having its other fixtures set up first, or its own body's need
not skipped for, or the env's reason reported before the browser's.

Each asked of the real suite in a pytest of its own, its outcomes read from
the suite's own outcome record (`tools.progress_plugin`, read by
`tools.testrun`).  THE TOOL IS MADE ABSENT FOR THAT RUN, never removed from
the env -- an env is never reshaped to suit a test (`testing.md` § 3): the
browser library and plugin made unimportable by an import finder and the
plugin not loaded, Node absent from an env -- a virtual env built on this
one, its packages all present and no Node in it -- and the env manager
absent from PATH.  The command each skip must name is the one
`process/testing.md` § 3 and § 4 name.

Category: API.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from molbuilder.envs.hints import install_cmd, opt_in_cmd
from support import tooling
from tools import testrun

REPO = Path(__file__).resolve().parents[1]
#: The command the contract names (`process/testing.md` § 4), as `doctor`
#: names it -- by the checkout's full path, whatever it holds
#: (`envs.hints.opt_in_cmd`; `ops/installation.md` § 2).
THE_COMMAND = opt_in_cmd("molbuilder")

#: A pytest with the browser tooling ABSENT, as on a plain install: an
#: import finder answers "no such module" for the library and its plugin,
#: and the plugin is not loaded.
_WITHOUT_THE_BROWSER = (
    "import importlib.abc, sys\n"
    "class _Absent(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name.split('.')[0] in ('playwright', 'pytest_playwright'):\n"
    "            raise ModuleNotFoundError(f'No module named {name!r}',\n"
    "                                      name=name)\n"
    "sys.meta_path.insert(0, _Absent())\n"
    "import pytest\n"
    "raise SystemExit(pytest.main(['-p', 'no:playwright'] + sys.argv[1:]))\n")


def _pytest(python, record, *args, without_the_browser=False, env=None):
    """A pytest of its own, run from the repo by ``python`` in ``env`` (this
    process's, by default), its outcomes recorded in ``record`` by the
    suite's own recorder: its exit code, and each test's ``(file, outcome,
    reason)``."""
    head = ([str(python), "-c", _WITHOUT_THE_BROWSER] if without_the_browser
            else [str(python), "-m", "pytest"])
    out = subprocess.run([*head, "-p", "no:cacheprovider",
                          "-p", "tools.progress_plugin",
                          f"--progress-file={record}", "-q", *args],
                         cwd=REPO, capture_output=True, text=True, timeout=600,
                         env=env)
    tests = [(e["nodeid"].split("::")[0], e["outcome"], e.get("reason") or "")
             for e in testrun._read_events(str(record))
             if e.get("event") == "test"]
    return out.returncode, tests, out.stdout + out.stderr


def test_the_browser_fixtures_are_the_plugins_own(pytestconfig):
    """A plugin that grows a fixture, missing from the rule's list, would
    leave a test asking for it to ERROR on a machine without the plugin:
    the rule's list is the plugin's public fixtures, asked where it is
    installed.  Fails on a fixture missing from the list."""
    if not tooling.browser_plugin(pytestconfig):
        pytest.skip(tooling.missing(tooling.BROWSER))
    plugin = pytestconfig.pluginmanager.get_plugin(tooling.BROWSER_PLUGIN)
    own = {name for name, value in vars(plugin).items()
           if not name.startswith("_")
           and (hasattr(value, "_fixture_function_marker")
                or type(value).__name__ == "FixtureFunctionDefinition")}
    assert own == set(tooling.BROWSER_FIXTURES), (
        f"the plugin's fixtures: {sorted(own)}")


def test_a_browser_test_skips_naming_the_command_without_the_browser(
        tmp_path):
    """A browser test erroring, or skipping with another reason, on a
    machine without the browser tooling: asked through ``page``
    (`test_pages_no_js_errors`) and through ``browser``
    (`test_inspector_registry_e2e` -- its fixture makes real runs, which the
    rule leaves unmade); and a test that parametrizes a ``page`` value of its
    own, needing no browser, skipped for one (`test_frame_set_units`).  Fails
    on the rule keyed on ``page`` alone and on parametrized names counted
    (the browser's reason reported before a need's:
    :func:`test_what_a_test_needs_of_the_machine_skips_by_the_one_rule`)."""
    code, tests, said = _pytest(
        sys.executable, tmp_path / "record.jsonl",
        "tests/test_pages_no_js_errors.py",
        "tests/test_inspector_registry_e2e.py::"
        "test_a_file_opens_in_the_viewer_its_kind_names",
        "tests/test_frame_set_units.py::"
        "test_a_page_says_the_refusal_where_the_frames_would_be",
        without_the_browser=True)
    assert code == 0, said[-3000:]
    skipped = [(f, why) for f, outcome, why in tests if outcome == "skipped"]
    assert {f for f, _ in skipped} == {
        "tests/test_pages_no_js_errors.py",
        "tests/test_inspector_registry_e2e.py"}, tests
    assert all(THE_COMMAND in why for _, why in skipped), skipped
    assert {f for f, outcome, _ in tests if outcome == "passed"} == {
        "tests/test_frame_set_units.py"}, tests


def test_the_suite_collects_without_the_browser(tmp_path):
    """`process/testing.md` § 4: "a machine without the tooling collects
    every file" -- a test file that imported the browser library at its top
    would fail the whole collection on a plain install, and one that skipped
    itself whole would hide its tests behind a reason of its own.  Fails on
    either."""
    record = tmp_path / "record.jsonl"
    code, _, said = _pytest(sys.executable, record, "--collect-only", "tests",
                            without_the_browser=True)
    assert code == 0, said[-3000:]
    # ...and no file skipped itself whole for the browser -- a skip of its
    # own, with a reason of its own, beside the rule.  (A file may skip
    # whole for an optional package, `test_peptide` for PeptideBuilder.)
    own = [e for e in testrun._read_events(str(record))
           if e.get("event") == "collect" and e.get("outcome") == "skipped"
           and "playwright" in (e.get("reason") or "")]
    assert not own, own


def test_a_node_test_skips_naming_the_command_in_an_env_without_node(
        tmp_path):
    """A Node test on an env without Node skipping with a reason of its own,
    which names no way to install it: every Node test of a file, run in an
    env that has none, skips naming the command.  Fails on the harness's
    skip giving a reason of its own."""
    env = tmp_path / "env-without-node"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                    "--without-pip", str(env)], check=True, timeout=120)
    assert not (env / "bin" / "node").exists()
    code, tests, said = _pytest(env / "bin" / "python",
                                tmp_path / "record.jsonl",
                                "tests/test_molbuilder_runtime_js.py")
    assert code == 0, said[-3000:]
    assert tests and all(outcome == "skipped" and THE_COMMAND in why
                         for _, outcome, why in tests), tests


def _without_an_env_manager() -> dict:
    """This process's environment with no conda, mamba or micromamba to be
    found -- PATH without the folders that hold one, and neither manager's
    variable -- so no env is listed, as on a machine with none; the envs
    themselves untouched."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("CONDA_EXE", "MAMBA_EXE")}
    env["PATH"] = os.pathsep.join(
        d for d in env["PATH"].split(os.pathsep)
        if not any((Path(d) / m).exists()
                   for m in ("conda", "mamba", "micromamba")))
    return env


def test_what_a_test_needs_of_the_machine_skips_by_the_one_rule(tmp_path):
    """`process/testing.md` § 3: what a test needs of the machine beyond the
    host env is stated once, and an unmet need skips by one rule -- for a
    test asking for a fixture declared to make a real run in an env, directly
    or through another fixture, before any of its fixtures is set up; for a
    test stating a need of its own body (`support.tooling.needs`); and after
    the browser's reason, for a test that also drives a browser (§ 4).  Asked
    of a suite run by the suite's own rules (`support.suite.linked`), with
    neither the browser nor an env manager to be found for that run; the
    run's fixture fails if it is ever set up, a body stating its need if it
    ever runs; each env's reason names the command that installs it.  Fails
    on the hook's skip removed, on a test's own need not skipped for, on a
    reason naming no command, and on the env's reason reported before the
    browser's."""
    from support.suite import linked
    suite = linked(tmp_path / "suite", {"test_needs.py": (
        "import pytest\n"
        "from support.real_runs import makes_a_real_run\n"
        "from support.tooling import needs\n"
        "@makes_a_real_run('module', 'molbuilder-siesta')\n"
        "def run():\n"
        "    raise AssertionError('set up, its need unmet')\n"
        "@pytest.fixture\n"
        "def read(run):\n"
        "    return run\n"
        "def test_asks_for_the_run(run):\n"
        "    pass\n"
        "def test_asks_for_it_through_another(read):\n"
        "    pass\n"
        "@needs('molbuilder-pySCF')\n"
        "def test_states_its_own_need():\n"
        "    raise AssertionError('ran, its need unmet')\n"
        "def test_drives_a_browser_on_the_run(page, run):\n"
        "    pass\n"
        "@needs('molbuilder-pySCF')\n"
        "def test_states_its_own_need_and_drives_a_browser(page):\n"
        "    pass\n")})
    code, tests, said = _pytest(sys.executable, tmp_path / "record.jsonl",
                                str(suite), without_the_browser=True,
                                env=_without_an_env_manager())
    assert code == 0, said[-3000:]
    why = {node.split("::")[-1]: (outcome, reason)
           for node, outcome, reason in (
               (e["nodeid"], e["outcome"], e.get("reason") or "")
               for e in testrun._read_events(str(tmp_path / "record.jsonl"))
               if e.get("event") == "test")}
    assert set(why) == {
        "test_asks_for_the_run", "test_asks_for_it_through_another",
        "test_states_its_own_need", "test_drives_a_browser_on_the_run",
        "test_states_its_own_need_and_drives_a_browser"}, why
    assert all(outcome == "skipped" for outcome, _ in why.values()), why
    for name, env in (("test_asks_for_the_run", "molbuilder-siesta"),
                      ("test_asks_for_it_through_another", "molbuilder-siesta"),
                      ("test_states_its_own_need", "molbuilder-pySCF")):
        assert install_cmd(env) in why[name][1], why
    for name in ("test_drives_a_browser_on_the_run",
                 "test_states_its_own_need_and_drives_a_browser"):
        assert THE_COMMAND in why[name][1], why


@pytest.mark.parametrize("package, required", [
    ("flask", True), ("rdkit", True), ("openbabel", True), ("ase", True),
    ("git", True),
    ("PeptideBuilder", False), ("threedna", False), ("amber", False),
])
def test_what_the_host_env_requires_is_never_skipped_for(package, required):
    """`process/testing.md` § 3: a package the host env requires -- a
    `conda` line of `host-env.txt`, read through the host recipe -- is never
    skipped for; an optional one (a `pip` line) or a tool outside the env
    may be.  A package moved between the two kinds moves the rule with it.
    Fails on a required package read as optional, which would let a broken
    env read green."""
    assert tooling.host_requires(package) is required
