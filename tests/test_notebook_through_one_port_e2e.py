"""The notebook through molbuilder's own port -- `web/jupyter.md` § 2, § 3.4
and § 6 -- on the road: `molbuilder serve start`, `molbuilder jupyter start`,
`molbuilder serve restart`, and a real browser on molbuilder's one port.

The silent failures: a JupyterNB tab that hides a notebook it could show --
the tab framed Jupyter on a port of its own and covered a working frame with
"cannot reach it" when its own check was blocked by the page's security
policy (every version from 2026-09-15, measured 2026-10-17); a kernel
listening on a network port; and a notebook left running on code its server
no longer runs.  Read in the browser and through the pass-through: the tab
frames JupyterLab from the page's own origin and a cell run in it answers
`2` -- over HTTP and over HTTPS, the kernel's WebSocket carried by
molbuilder's port; the notebook's own port answering this machine only; and
a restart of the server stopping its notebook.

Run where the opt-in `molbuilder-jupyternb` env is installed; skipped
elsewhere.
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from _road import env_available

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.skipif(
    not env_available("molbuilder-jupyternb"),
    reason="needs the opt-in molbuilder-jupyternb env")

_REPO = Path(__file__).resolve().parents[1]

#: A self-signed certificate is what the HTTPS stack serves; the test's own
#: requests trust it as the browser is told to.
_TRUSTING = ssl.create_default_context()
_TRUSTING.check_hostname = False
_TRUSTING.verify_mode = ssl.CERT_NONE


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """The browser trusts the HTTPS stack's self-signed certificate."""
    return {**browser_context_args, "ignore_https_errors": True}


def _molbuilder(env, log, *words, timeout=120):
    """One `molbuilder` verb, as a person types it, in this test's own
    configuration -- ``(return code, what it said)``.  Its output goes to
    ``log``, a FILE: `serve start` detaches a server that keeps what it was
    handed, and a pipe held open would never end."""
    with open(log, "a") as out:
        start = out.tell()
        rc = subprocess.run([sys.executable, "-m", "molbuilder", *words],
                            cwd=_REPO, env=env, stdout=out,
                            stderr=subprocess.STDOUT, timeout=timeout,
                            ).returncode
    with open(log) as said:
        said.seek(start)
        return rc, said.read()


def _ask(url, *, method="GET", body=None):
    """``(status, body)`` of one request to molbuilder -- no token: the
    pass-through presents the notebook's own (`jupyter.md` § 2)."""
    req = urllib.request.Request(
        url, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=30, context=_TRUSTING) as r:
        return r.status, r.read()


def _status(base):
    return json.loads(_ask(f"{base}/api/jupyter/status")[1])


def _until(ask, *, within_s, what, evidence=None):
    """``ask()`` until it answers truthily, or fail naming ``what`` -- with
    what ``evidence()`` says, when given: a wait that ends in a failure
    keeps the reason it has."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            answer = ask()
        except OSError:
            answer = None
        if answer:
            return answer
        time.sleep(0.5)
    pytest.fail(f"{what} within {within_s} s"
                + (f"\n{evidence()}" if evidence else ""))


@contextmanager
def _stack(root: Path, scheme: str):
    """molbuilder started as a person starts it -- `serve start`, over
    ``scheme``, then `jupyter start` -- with a configuration, state,
    projects tree and notebook home of its own: no sign-in, on loopback,
    where this machine's browser may control the notebook (`jupyter.md`
    § 6).  Yields its address and its environment; `serve stop` takes both
    down after, and says so."""
    from test_jupyter_contract import _free_serve_port
    cfg, run = root / "cfg", root / "run"
    for d in (cfg, run):
        d.mkdir(mode=0o700)
    projects = root / "projects"
    projects.mkdir()
    (cfg / "molbuilder.json").write_text(
        json.dumps({"paths": {"projects": str(projects)}}))
    (cfg / "molbuilder.json").chmod(0o600)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MOLBUILDER_")}
    env.update(MOLBUILDER_CONFIG_DIR=str(cfg), XDG_RUNTIME_DIR=str(run),
               XDG_STATE_HOME=str(root / "state"),
               IPYTHONDIR=str(root / "ipython"),
               JUPYTER_RUNTIME_DIR=str(root / "jupyter-runtime"),
               JUPYTER_DATA_DIR=str(root / "jupyter-data"))
    tls = []
    if scheme == "https":
        cert, key = root / "cert.pem", root / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                        "-nodes", "-days", "1", "-subj", "/CN=127.0.0.1",
                        "-keyout", str(key), "-out", str(cert)],
                       check=True, capture_output=True)
        tls = ["--cert", str(cert), "--key", str(key)]
    port = _free_serve_port()
    base = f"{scheme}://127.0.0.1:{port}"
    log = root / "verbs.log"
    try:
        rc, said = _molbuilder(env, log, "serve", "start", "--port",
                               str(port), *tls)
        assert rc == 0, said
        _until(lambda: _ask(f"{base}/api/health")[0] == 200,
               within_s=60, what="molbuilder answering")
        rc, said = _molbuilder(env, log, "jupyter", "start", "--port",
                               str(port))
        assert rc == 0, said

        def notebook_said():
            """The server's last status answer and the end of the
            notebook's own log (`config_dir.jupyter_log`, under this
            stack's state folder) and of the verbs."""
            from unittest import mock
            from molbuilder.config_dir import jupyter_log
            with mock.patch.dict(os.environ,
                                 {"XDG_STATE_HOME": env["XDG_STATE_HOME"]}):
                nb_log = jupyter_log(port)
            try:
                st = _status(base)
            except OSError as exc:
                st = f"unreadable: {exc}"
            nb = (nb_log.read_text(errors="replace")[-4000:]
                  if nb_log.is_file() else "(none written)")
            return (f"status: {st}\nverbs:\n{log.read_text()[-2000:]}\n"
                    f"notebook log {nb_log}:\n{nb}")

        _until(lambda: (lambda st: st["running"] and st["answering"])(
            _status(base)), within_s=180, what="the notebook answering",
            evidence=notebook_said)
        yield {"base": base, "port": port, "env": env, "log": log}
    finally:
        rc, said = _molbuilder(env, log, "serve", "stop", "--port",
                               str(port))
        assert rc == 0, said


@pytest.fixture(scope="module", params=["http", "https"])
def one_port(request, tmp_path_factory):
    with _stack(tmp_path_factory.mktemp(f"one-port-{request.param}"),
                request.param) as stack:
        yield stack


def test_the_tab_frames_the_notebook_from_its_own_origin_and_a_cell_answers(
        page, one_port):
    """`jupyter.md` § 2: the JupyterNB tab frames JupyterLab at
    ``/jupyter/`` on the page's own address, and a cell run in it answers --
    every request and the kernel's WebSocket carried by molbuilder's one
    port, the token presented by molbuilder and never by the page.

    MUTATIONS THIS MUST FAIL AGAINST: the frame pointed at Jupyter's own
    port (the page's origin no longer); the WebSocket not joined (the cell
    never answers)."""
    base = one_port["base"]
    notebook = {"cells": [{"cell_type": "code", "execution_count": None,
                           "id": "one", "metadata": {}, "outputs": [],
                           "source": "1+1"}],
                "metadata": {"kernelspec": {"name": "python3",
                                            "display_name": "Python 3",
                                            "language": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    saved, _ = _ask(f"{base}/jupyter/api/contents/one.ipynb", method="PUT",
                    body={"type": "notebook", "content": notebook})
    assert saved in (200, 201), saved

    page.goto(f"{base}/jupyternb")
    lab = page.frame_locator("#nb-frame")
    lab.locator(".jp-LabShell").wait_for(timeout=90_000)
    framed = page.evaluate("document.getElementById('nb-frame').src")
    assert framed.startswith(f"{base}/jupyter/lab"), framed

    page.evaluate("(u) => { document.getElementById('nb-frame').src = u; }",
                  f"{base}/jupyter/lab/tree/one.ipynb")
    cell = lab.locator(".jp-Notebook .jp-Cell .cm-content").first
    cell.wait_for(timeout=90_000)

    # THE CELL IS RUN ONCE THE KERNEL IS CONNECTED AND IDLE, as Jupyter
    # reports it through the pass-through: a cell run before then is
    # Lab's to queue or drop, and says nothing about the connection.
    def connected_and_idle():
        kernels = json.loads(_ask(f"{base}/jupyter/api/kernels")[1])
        return any(k.get("connections", 0) >= 1
                   and k.get("execution_state") == "idle" for k in kernels)
    _until(connected_and_idle, within_s=90,
           what="the notebook's kernel connected and idle")
    cell.click()
    page.keyboard.press("Shift+Enter")
    out = lab.locator(".jp-OutputArea-output").first
    out.wait_for(timeout=120_000)
    assert out.inner_text().strip() == "2", out.inner_text()


def _an_address_off_loopback():
    """This machine's address on its outward interface, or ``None``."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))          # TEST-NET-1: nothing is sent
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") else ip


def test_the_notebooks_own_port_answers_this_machine_only(one_port):
    """`jupyter.md` § 2, § 6: Jupyter listens on ``127.0.0.1`` -- its own
    port answers there, and is refused on this machine's outward address,
    so no kernel is on a network port.

    MUTATION THIS MUST FAIL AGAINST: Jupyter bound to every interface."""
    from molbuilder.jupyter import jupyter_port
    nb = jupyter_port(one_port["port"])
    with socket.create_connection(("127.0.0.1", nb), timeout=5):
        pass
    outward = _an_address_off_loopback()
    if outward is None:
        pytest.skip("this machine has no address off loopback to ask")
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((outward, nb), timeout=5).close()


def test_a_restart_of_the_server_stops_its_notebook(tmp_path):
    """`jupyter.md` § 3.4: `serve restart` stops a running notebook -- a
    restart is how new code takes effect, and a notebook left running would
    be one the server no longer started -- and the server answers again.

    MUTATION THIS MUST FAIL AGAINST: the supervisor's restart leaving the
    notebook running."""
    with _stack(tmp_path, "http") as stack:
        base = stack["base"]
        rc, said = _molbuilder(stack["env"], stack["log"], "serve",
                               "restart", "--port", str(stack["port"]))
        assert rc == 0, said
        st = _until(lambda: (lambda s: s if not s["running"] else None)(
            _status(base)), within_s=60, what="the notebook stopped")
        assert st["answering"] is False, st
