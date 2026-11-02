"""The suite never reaches the person's machine (`docs/process/testing.md` § 2b).

GOAL: a test -- of any scope, at collection, in a child process, or a helper
someone runs by hand -- writes nothing into the person's home, config, state,
environments or projects folders, and connects to nothing off this machine.

WHY: on 2026-10-30 a helper of this suite, run by hand outside pytest, wrote
its test machine record into the person's real
`~/.config/molbuilder/environment.json`, and every real run on their machine
failed until they probed it again; the audit that followed found module
fixtures, collection-time checks and child processes running before or outside
the per-test isolation, and tests posting to Slack and solving against
conda-forge (`docs/archive/audits/2026-11-02-test-audit/`).

PREVENTS: a write, a delete or a connection that lands on the person's machine
or off it, and test code that loads with no sandbox around it.

Every probe is built so that nothing happens even where a guard is missing: a
path under a folder that does not exist, an address no network routes.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

import conftest
from molbuilder.config_dir import config_dir, state_dir

#: A name nothing on any machine has.
NOWHERE = f"molbuilder-guard-probe-{uuid.uuid4().hex}"

#: A write or delete each guard must refuse -- the person's places, each
#: probed under a folder that does not exist.
REFUSED = [
    ("a file in your home",
     lambda: open(conftest._REAL_HOME / NOWHERE / "f", "w")),
    ("a delete in your config folder",
     lambda: os.remove(conftest._REAL_CONFIG_DIR / NOWHERE)),
    ("a folder in the checkout's projects library",
     lambda: os.mkdir(conftest._REPO / "projects" / "pseudopotential"
                      / NOWHERE / "x")),
    ("a tree removed beside your environment",
     lambda: shutil.rmtree(Path(sys.prefix).parent / NOWHERE)),
    ("a connection off this machine",
     lambda: socket.create_connection(("192.0.2.1", 9), timeout=1)),
]


@pytest.mark.parametrize("what, act", REFUSED, ids=[r[0] for r in REFUSED])
def test_a_reach_past_the_sandbox_is_refused(what, act):
    try:
        with pytest.raises(conftest.ReachedOutOfTheSandbox):
            act()
    finally:
        conftest._REACHED.clear()     # this test's own reach, not a finding


@pytest.fixture(scope="module")
def module_folders():
    """What a module-scoped fixture sees -- the scope the per-test isolation
    never covered."""
    return [Path.home(), Path(config_dir()), Path(state_dir())]


def test_every_scope_is_given_the_sandbox_s_folders(module_folders):
    temp = Path(tempfile.gettempdir()).resolve()
    for got in module_folders + [Path.home(), Path(config_dir()),
                                 Path(state_dir())]:
        assert temp in got.resolve().parents, got


def _python(code: str) -> subprocess.CompletedProcess:
    """``code`` in a child python started from the checkout as a person or
    an agent would start a script: no pytest running around it."""
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_VERSION"}
    return subprocess.run([sys.executable, "-c", code], cwd=conftest._REPO,
                          env=env, capture_output=True, text=True, timeout=120)


#: Test code imported by a script -- the 2026-10-30 road, last row its shape.
SCRIPTS = [
    ("the helpers", "import sys; sys.path.insert(0, 'tests'); "
                    "import support.road"),
    ("the conftest", "import sys; sys.path.insert(0, 'tests'); "
                     "import conftest"),
    ("the engine road", "import sys; sys.path.insert(0, 'tests'); "
                        "import _road"),
    ("a road row run by hand", "import sys; sys.path.insert(0, 'tests'); "
                               "import pytest; "
                               "from support.road import run_road_case"),
]


@pytest.mark.parametrize("what, code", SCRIPTS, ids=[r[0] for r in SCRIPTS])
def test_test_code_refuses_to_load_outside_pytest(what, code):
    got = _python(code)
    assert got.returncode != 0, got.stdout
    assert "runs only inside pytest" in got.stderr, got.stderr[-2000:]


def test_a_child_process_request_off_this_machine_goes_nowhere():
    """A tool a test starts -- conda, pip, a script -- sends its HTTP through
    the proxy the sandbox names, which refuses at once."""
    got = _python("import urllib.request\n"
                  "urllib.request.urlopen('http://192.0.2.1/', timeout=20)")
    assert got.returncode != 0
    assert "refused" in got.stderr.lower(), got.stderr[-2000:]
