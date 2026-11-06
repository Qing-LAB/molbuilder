"""THE way a test starts a real HTTP server for the app.

**One implementation, and each module keeps its own fixture scope.**  The
scopes are not uniform today — most are ``module``, a few are function — and a
scope is a decision about how much state a file's tests share, not something a
de-duplication gets to change on their behalf.  So this is a context manager
rather than a fixture: a module writes

    @pytest.fixture(scope="module")          # or plain @pytest.fixture
    def flask_server():
        with serve() as base_url:
            yield base_url

and the server itself is here, once.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def serve(config: dict | None = None):
    """Run the app on an ephemeral port; yield its base URL.

    ``config`` is passed straight to ``create_app`` (``{}`` by default, which
    is what every caller used).  The port is chosen by the OS — 0 — so
    concurrent test modules never collide on a fixed number, and the thread is
    a daemon so a hung shutdown cannot keep the interpreter alive.

    **A projects tree of its own** unless the test already pointed the one
    door somewhere (`isolated_projects_root`): every page lists the
    sidebar's root as it boots, and the default is the developer's real
    tree (`process/testing.md` § 2a).  A test that isolates later, per
    function, still wins: the root is read at call time.

    **A documents tree of its own**: the documents route regenerates
    `toc.json` as it answers (`blueprints.docs._build_toc_tree`), so a page
    served from the checkout would write there, and would depend on what the
    file lists.  A test asks the route; `test_docs_tab` pins what it does.
    """
    from _pytest.monkeypatch import MonkeyPatch
    from werkzeug.serving import make_server

    from molbuilder.projects import PROJECTS_ROOT_ENV, PROJECTS_ROOT_NAME
    from molbuilder.web.app import create_app
    from molbuilder.web.blueprints import docs as docs_blueprint

    variables = MonkeyPatch()
    own = None
    if not os.environ.get(PROJECTS_ROOT_ENV):
        own = tempfile.mkdtemp(prefix="served-tree-")
        tree = os.path.join(own, PROJECTS_ROOT_NAME)
        os.mkdir(tree)
        variables.setenv(PROJECTS_ROOT_ENV, tree)

    the_docs = docs_blueprint._docs_root
    own_docs = tempfile.mkdtemp(prefix="served-docs-")
    served_docs = shutil.copytree(the_docs(), os.path.join(own_docs, "docs"))
    docs_blueprint._docs_root = lambda: Path(served_docs)

    server = make_server("127.0.0.1", 0, create_app(config=config or {}),
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
        variables.undo()
        if own is not None:
            shutil.rmtree(own, ignore_errors=True)
