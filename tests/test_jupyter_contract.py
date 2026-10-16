"""The notebook feature's contract.

`plan.md` § 5n, J10: 1,610 lines across five files, plus the notebook half of
`serve_daemon` and six CLI verbs, and **nothing under `tests/` reached any of
it**. Every one of the nine defects fixed on 2026-09-14 was found by eye or by
running the live server, and one of them — 40 lines of Python inside a string
literal — shipped a `NameError` on every notebook start because no linter,
import or test could see it. That is the root of the whole row: nothing had
ever forced a seam on this code.

Each one asserts a rule a DOCUMENT states, and each was mutation-tested —
the code broken, the right test watched to fail, the code restored:

1. the schema gate                  `jupyter.md` § 4.1 · `data/jupyter.toml`
2. an unknown section, refused by name                  `data/jupyter.toml`
3. every authored setting reaches the command line      `jupyter.md` § 4
4. a notebook saved leaves no checkpoint beside it    `jupyter.md` § 4.2
5. the control routes' 404 and 403                      `jupyter.md` § 5.2
6. the stop grace stays under the supervisor's          `jupyter.py`
7. a start forgets the workspace and keeps the settings `jupyter.md` § 4.1
8. one server's start keeps another server's layout   `jupyter.md` § 4.1
"""
from __future__ import annotations

import inspect
import json

import pytest

from molbuilder import jupyter as J


@pytest.fixture
def rules_file(tmp_path, monkeypatch):
    """The shipped table, copied somewhere a test may corrupt it.

    Returns a writer: hand it the file's text and `load_rules()` will read
    that instead. The real `data/jupyter.toml` is never touched.
    """
    real = J.rules_path().read_text(encoding="utf-8")
    target = tmp_path / "jupyter.toml"

    def write(text=real):
        target.write_text(text, encoding="utf-8")
        monkeypatch.setattr(J, "rules_path", lambda: target)
        return target
    write()
    return write


# --------------------------------------------------------------------- #
#  1-2. The gate, and the refusals that name what exists                 #
# --------------------------------------------------------------------- #

def test_a_wrong_schema_stamp_is_refused(rules_file):
    """The stamp is the artifact's NAME as well as its version.

    `persist.check_schema`'s own history is the reason: it compared only
    majors once, so any `@1` artifact parsed as any other.
    """
    rules_file(J.rules_path().read_text(encoding="utf-8")
               .replace("molbuilder/jupyter@1", "molbuilder/warm-files@1"))
    with pytest.raises(ValueError) as exc:
        J.load_rules()
    assert "molbuilder/jupyter" in str(exc.value)


def test_an_unknown_section_is_refused_by_naming_the_ones_that_exist(
        rules_file):
    """A typo must not disable a setting in silence.

    That is the whole reason this table exists -- `port_retries` went
    missing for a week inside a list literal -- so the refusal names the
    sections a person can choose from rather than only the one they got
    wrong (`warmfiles`' own refusal style).
    """
    rules_file(J.rules_path().read_text(encoding="utf-8")
               + '\n[kernels]\nfoo = 1\n')
    with pytest.raises(J.JupyterRulesError) as exc:
        J.load_rules()
    said = str(exc.value)
    assert "kernels" in said
    for legal in ("server", "lab"):
        assert legal in said, f"the refusal does not name [{legal}]"


# --------------------------------------------------------------------- #
#  3. Every authored setting reaches the command line                    #
# --------------------------------------------------------------------- #

def test_every_authored_setting_reaches_the_argv():
    """THE ASSERTION A DROPPED ROW TRIPS.

    `--ServerApp.port_retries=0` was absent for a week while nothing could
    notice, because the settings were a list literal and a list has no shape
    to be missing from. This walks the table rather than naming settings, so
    a row added to `data/jupyter.toml` is covered the day it is added and a
    row that stops being emitted fails here.
    """
    rules = J.load_rules()
    argv = J.notebook_argv("conda", "env", host="127.0.0.1", port=6007,
                           root_dir="/p", cert=None, key=None,
                           lab_dirs={})
    assert rules.server, "the table declares no [server] settings at all"
    for trait, value in rules.server.items():
        spelled = f"--{trait}={value!r}" if isinstance(value, bool) \
            else f"--{trait}={value}"
        assert spelled in argv, f"{trait} never reached the command line"


def test_a_runtime_setting_reaches_the_argv_through_the_same_emitter():
    """The computed half goes out the same door as the authored half: a
    value only this process knows comes out in the same `--trait=value`
    spelling.
    """
    argv = J.notebook_argv("conda", "env", host="1.2.3.4", port=6007,
                           root_dir="/p", cert=None, key=None,
                           lab_dirs={"LabApp.workspaces_dir": "/w"})
    # `ip` and `workspaces_dir` are computed at run time; the token is not
    # emitted at all -- see the test below.
    assert "--ServerApp.ip=1.2.3.4" in argv
    assert "--LabApp.workspaces_dir=/w" in argv


def test_the_notebook_token_never_reaches_the_command_line():
    """The token rides `JUPYTER_TOKEN`, because `/proc` keeps `environ`
    owner-only and `cmdline` world-readable.

    Asserted over the WHOLE argv rather than one spelling: what matters is the
    value being absent, so a row reintroducing it under another trait name has
    to fail here too.
    """
    secret = "tok-must-not-appear-on-the-command-line"
    argv = J.notebook_argv("conda", "env", host="127.0.0.1", port=6007,
                           root_dir="/p", cert=None, key=None, lab_dirs={})
    assert not any(secret in part for part in argv), argv
    assert not any("token" in part.lower() for part in argv), argv


# --------------------------------------------------------------------- #
#  4. A notebook saved leaves no checkpoint beside it                    #
# --------------------------------------------------------------------- #

def _free_serve_port() -> int:
    """A serve port whose notebook port (`jupyter_port`) is free here."""
    import socket
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", J.jupyter_port(port)))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port pair")


@pytest.mark.slow
def test_a_notebook_saved_in_the_projects_tree_leaves_no_checkpoint(
        isolated_projects_root, tmp_path):
    """`jupyter.md` § 4.2: the framed server writes no `.ipynb_checkpoints/`
    beside a notebook it saves -- a checkpoint asked for answers
    ``no-checkpoint``, the list comes back empty, and a restore is refused
    rather than quietly doing nothing.

    THE SILENT FAILURE: a directory in every folder somebody opened a
    notebook in, swept up by result scans and carried to the cluster with
    every copy -- which is what comes back, in silence, when the class that
    answers the checkpoints is lost from the config (traitlets logs the
    config file's error and starts on its defaults).

    The notebook server started as the supervisor starts it (`shepherd_argv`,
    the env entered through the manager), rooted at the projects tree; the
    notebook saved and checkpointed through Jupyter's own contents API, as
    Lab saves one.

    MUTATIONS THIS MUST FAIL AGAINST: the config file not handed to the
    server; the class wired into no manager; a restore that quietly does
    nothing; a restore the class no longer answers (the server's own
    error, not a refusal)."""
    import os
    import signal
    import subprocess
    import time
    import urllib.error
    import urllib.request
    from _road import env_available
    if not env_available("molbuilder-jupyternb"):
        pytest.skip("needs the molbuilder-jupyternb env")
    serve_port = _free_serve_port()
    folder = isolated_projects_root / "P" / "notes"
    folder.mkdir(parents=True)
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / "state"))
    shepherd = subprocess.Popen(
        J.shepherd_argv(serve_port, host="127.0.0.1"), env=env,
        stdout=open(tmp_path / "notebook.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        for _ in range(120):
            if J.answering(serve_port) or shepherd.poll() is not None:
                break
            time.sleep(0.5)
        assert J.answering(serve_port), (
            (tmp_path / "notebook.log").read_text()[-3000:])
        runtime = J.read_runtime(serve_port)
        base = f"{runtime['base']}/api/contents/P/notes/a.ipynb"

        def ask(method, url, body=None):
            req = urllib.request.Request(
                url, method=method,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Authorization": f"token {runtime['token']}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read() or b"null")
            except urllib.error.HTTPError as e:
                return e.code, None

        notebook = {"cells": [], "metadata": {}, "nbformat": 4,
                    "nbformat_minor": 5}
        saved, _ = ask("PUT", base, {"type": "notebook", "content": notebook})
        made, checkpoint = ask("POST", f"{base}/checkpoints")
        listed, checkpoints = ask("GET", f"{base}/checkpoints")
        restored, _ = ask("POST", f"{base}/checkpoints/no-checkpoint")
    finally:
        os.killpg(shepherd.pid, signal.SIGTERM)
        try:
            shepherd.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(shepherd.pid, signal.SIGKILL)
    assert saved in (200, 201), saved
    assert (made, (checkpoint or {}).get("id")) == (201, "no-checkpoint"), (
        made, checkpoint)
    assert (listed, checkpoints) == (200, []), (listed, checkpoints)
    assert 400 <= restored < 500, restored
    left = sorted(str(q.relative_to(isolated_projects_root))
                  for q in isolated_projects_root.rglob("*"))
    assert left == ["P", "P/notes", "P/notes/a.ipynb"], left


# --------------------------------------------------------------------- #
#  5. The control routes                                                 #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("route", ["/api/jupyter/start", "/api/jupyter/stop"])
def test_no_supervisor_answers_404_on_both_control_routes(route, monkeypatch):
    """404, not a missing route and not a 403.

    A misconfiguration must read as *the button is missing* -- which is what
    it is -- and never as *anyone may start a kernel here* (`jupyter.md`
    § 5.2). This test could not be written at all until 2026-09-15: the
    routes were registered behind an environment variable read at import, so
    building an app without one produced an app without them.
    """
    monkeypatch.delenv("MOLBUILDER_SUPERVISED", raising=False)
    from molbuilder.web.app import create_app
    app = create_app(config={})
    app.config["MOLBUILDER_SERVE_PORT"] = 8000
    r = app.test_client().post(route)
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


@pytest.mark.parametrize("route", ["/api/jupyter/start", "/api/jupyter/stop"])
def test_a_caller_who_may_not_control_gets_403_not_404(route, monkeypatch):
    """With a supervisor present, the refusal changes meaning.

    404 says *there is no button*; 403 says *there is one and it is not
    yours*. Both routes answer with the SAME sentence -- they answered with
    two different ones until 2026-09-15, and the shorter told a person
    nothing they could act on (`plan.md` § 5n, J4).
    """
    from molbuilder.web.app import create_app
    from molbuilder.web.blueprints import jupyter as bp
    monkeypatch.setattr(bp, "_supervised", lambda: True)
    monkeypatch.setattr(bp, "_may_control", lambda: False)
    app = create_app(config={})
    app.config["MOLBUILDER_SERVE_PORT"] = 8000
    r = app.test_client().post(route)
    assert r.status_code == 403
    assert "admin" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  6. The two grace periods, which must not meet                         #
# --------------------------------------------------------------------- #

def test_the_shepherds_grace_stays_under_the_supervisors():
    """`jupyter.md` § 3.3: the shepherd's polite phase ends before
    reconciliation stops waiting for it.

    The two were both 5.0, so reconciliation's poll timed out at the same
    instant the shepherd would have exited and reported *"(forced)"* for
    every perfectly polite stop. It was fixed by changing one number, and
    nothing stopped the other moving back.  No run shows it -- a polite stop
    ends in well under a second unless the server stalls the whole grace --
    so the two numbers are asked (`process/testing.md` § 3a).
    """
    from molbuilder import serve_daemon
    supervisor = inspect.signature(
        serve_daemon.stop_by_pidfile).parameters["grace_s"].default
    assert J._STOP_GRACE_S < supervisor, (
        f"the shepherd's polite phase ({J._STOP_GRACE_S}s) must finish "
        f"strictly before reconciliation gives up on it ({supervisor}s), "
        f"or every clean stop is reported as forced")


# --------------------------------------------------------------------- #
#  7. A start forgets the layout and keeps the settings                  #
# --------------------------------------------------------------------- #

def test_a_start_empties_the_workspace_and_keeps_user_settings(
        tmp_path, monkeypatch):
    """The user's rule, 2026-09-15: *"persistent when switching tab, but do
    not need this when server get shutdown and restarted."*

    **`user-settings/` is the mutation that matters.** The directory to
    empty is named in a data file, and pointing that row at the wrong
    directory would silently discard a person's own Lab settings instead of
    a window layout. `_forget_workspace`'s guard refuses a name that escapes
    the Lab home but cannot catch a wrong-but-valid one -- this is what
    catches that, and it is the only thing that does.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    out = J.prepare_lab_home(8000)
    ws = __import__("pathlib").Path(out["LabApp.workspaces_dir"])
    home = ws.parent.parent
    assert not J.workspace_saved(8000), "a fresh Lab home has no layout"

    # Lab saves a layout; a person changes a setting.
    (ws / "default.jupyterlab-workspace").write_text('{"data": {}}')
    (ws / "nested").mkdir()
    (ws / "nested" / "more").write_text("x")
    mine = home / "user-settings" / "mine.jupyterlab-settings"
    mine.write_text('{"theme": "chosen by a person"}')
    assert J.workspace_saved(8000)

    J.prepare_lab_home(8000)      # the next notebook server starts

    assert not J.workspace_saved(8000)
    assert list(ws.iterdir()) == [], "the layout outlived its server"
    assert mine.read_text() == '{"theme": "chosen by a person"}', (
        "a start discarded the person's own Lab settings")
    # And the overrides ARE rewritten, so the answer to "why does my framed
    # Lab look like this" is always the table and never a stale file.
    written = json.loads(
        (home / "settings" / "overrides.json").read_text(encoding="utf-8"))
    assert written == J.load_rules().overrides


def test_one_servers_start_does_not_forget_another_servers_layout(
        tmp_path, monkeypatch):
    """TWO MOLBUILDERS DO NOT SHARE A LAYOUT.

    Found in review 2026-09-15, hours after J13 shipped: the Lab home is
    deliberately not port-keyed, and J13 put per-server SESSION state into
    it. Starting B's notebook emptied A's workspace, so A's next tab switch
    lost the notebook whose kernel was still running -- J13's own bug, back.
    And B's saved workspace made A's first framing report `workspace_saved`
    true, so A opened at the projects root instead of the selected folder.

    `settings/` and `user-settings/` stay SHARED, which is the half of the
    original reasoning that is still right, so this asserts both halves.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    a = __import__("pathlib").Path(
        J.prepare_lab_home(8000)["LabApp.workspaces_dir"])
    (a / "default.jupyterlab-workspace").write_text('{"data": {"A": 1}}')
    shared = a.parent.parent / "user-settings" / "mine.jupyterlab-settings"
    shared.write_text("shared")

    b = __import__("pathlib").Path(
        J.prepare_lab_home(9000)["LabApp.workspaces_dir"])

    assert a != b, "two servers were handed the same workspace directory"
    assert J.workspace_saved(8000), "B's start forgot A's layout"
    assert not J.workspace_saved(9000), "B inherited A's layout"
    assert shared.read_text() == "shared", (
        "the SHARED half of the Lab home must survive -- the defaults and a "
        "person's own settings do not differ between servers")


def test_the_suite_cannot_see_the_developers_own_servers():
    """THE ISOLATION THAT MAKES EVERY OTHER TEST HERE MEAN SOMETHING.

    `config_dir.ports_with_pidfile()` globs `$XDG_RUNTIME_DIR/molbuilder`,
    and five call sites reach it as of 2026-09-15 -- both status surveys,
    both "but a server IS running on ..." hints, and `port_clash`, which
    `serve start` calls before detaching. One of those then makes a real
    HTTP request to every port it finds.

    `conftest` pins `XDG_RUNTIME_DIR` for exactly this, and it did not until
    2026-09-15: redirecting HOME does not reach that variable, so a test
    could read the developer's live `serve-<port>.pid` and prove nothing
    (`plan.md` § 5n.8). This asserts the guarantee instead of trusting it.
    """
    from molbuilder.config_dir import ports_with_pidfile, runtime_dir
    root = str(runtime_dir())
    assert "/run/user/" not in root, (
        f"the suite is looking at the session runtime root ({root}); "
        f"conftest must pin XDG_RUNTIME_DIR")
    assert ports_with_pidfile() == [] and ports_with_pidfile("jupyter") == [], (
        "the suite can see pidfiles it did not write")
