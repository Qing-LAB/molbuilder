"""THE way a test starts a real HTTP server for the app.

**One implementation, and each module keeps its own fixture scope.**  The
scopes are not uniform today — most are ``module``, a few are function — and a
scope is a decision about how much state a file's tests share, not something a
de-duplication gets to change on their behalf.  So this is a context manager
rather than a fixture: a module writes

    @pytest.fixture(scope="module")
    def flask_server(isolated_projects_root_module):
        with serve() as base_url:
            yield base_url

-- a per-test server asks ``isolated_projects_root`` -- and the server
itself is here, once; and the port a server started as a person starts it
is given (:func:`free_serve_port`).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def serve():
    """Run the app on an ephemeral port; yield its base URL.

    The app is ``create_app(config={})``'s: the sections it reads at start
    -- sign-in, admin, rate limit -- from no file, so no sign-in stands in
    a page's way.  The port is chosen by the OS — 0 — so concurrent test
    modules never collide on a fixed number, and the thread is a daemon so a
    hung shutdown cannot keep the interpreter alive.

    **IT SERVES THE PROJECTS TREE THE ONE DOOR ALREADY POINTS AT, and
    makes none**: the fixture that serves asks for the tree first --
    ``isolated_projects_root_module`` for a module's server,
    ``isolated_projects_root`` for a per-test one (`tests/conftest.py`,
    `point_the_one_door_at`) -- and with the door pointing nowhere it
    refuses, because unset is the developer's real tree
    (`process/testing.md` § 2a).  A test that serves a real run's tree
    (`support.real_runs`) asks for neither: it points the door at that run's
    ``tree`` first -- ``point_the_one_door_at(run.tree.parent,
    monkeypatch.setenv)``, the call the run's tree was made by -- and serves
    in the test.  A test that points the door elsewhere under a module's
    server still wins: the root is read at call time, and the test's
    monkeypatch is undone before the module's.

    **A documents tree of its own**: the documents route regenerates
    `toc.json` as it answers (`blueprints.docs._build_toc_tree`), so a page
    served from the checkout would write there, and would depend on what the
    file lists.  A test asks the route; `test_docs_tab` pins what it does.
    """
    from werkzeug.serving import make_server

    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.web.app import create_app
    from molbuilder.web.blueprints import docs as docs_blueprint

    if not os.environ.get(PROJECTS_ROOT_ENV):
        raise RuntimeError(
            f"serve: {PROJECTS_ROOT_ENV} is unset, which is the developer's "
            f"real projects tree -- the fixture that serves asks for "
            f"`isolated_projects_root_module` (a module's server) or "
            f"`isolated_projects_root` (a per-test one) first")

    the_docs = docs_blueprint._docs_root
    own_docs = tempfile.mkdtemp(prefix="served-docs-")
    served_docs = shutil.copytree(the_docs(), os.path.join(own_docs, "docs"))
    docs_blueprint._docs_root = lambda: Path(served_docs)

    server = make_server("127.0.0.1", 0, create_app(config={}),
                         threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        docs_blueprint._docs_root = the_docs
        shutil.rmtree(own_docs, ignore_errors=True)


def free_serve_port() -> int:
    """A port for `serve start` whose notebook port (`jupyter.jupyter_port`)
    is free here too -- each found free on loopback, the pair asked again
    when the notebook's is taken."""
    import socket

    from molbuilder.jupyter import jupyter_port
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", jupyter_port(port)))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port pair")
