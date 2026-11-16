"""THE way a test starts a real HTTP server for the app.

A test that needs the app serving asks for ``app_at`` (`tests/conftest.py`,
`process/testing.md` § 2a), the one fixture over :func:`serve`: module-scoped,
standing on the module's tree, the projects root read at each request so a
test pointing the door at a tree of its own is answered from it.  This module
keeps the server itself, once, as a context manager: for a test that needs a
DIFFERENT app -- sign-in on, a rate limit (``serve(config)``) -- for a test
serving a real run's tree, which it points the door at first, and for the
e2e files that still start their own until F10 runs them; and the port a
server started as a person starts it (:func:`free_serve_port`).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def serve(config=None):
    """Run the app on an ephemeral port; yield its base URL.

    The app is ``create_app(config=config)``'s: the sections it reads at
    start -- sign-in, admin, rate limit -- from no file; none given, no
    sign-in stands in a page's way, and a test that is about signing in hands
    its ``auth`` section in, or its rate limit (`process/testing.md` § 3b:
    the real thing).  The port is chosen by the OS — 0 — so concurrent test
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


#: WHERE A TEST'S `serve start` LISTENS: even ports below the system's
#: ephemeral range (32768 and up on Linux).  Its notebook takes the port after
#: (`jupyter.jupyter_port`), which nothing else in a parallel run can be
#: handed: every port asked of the system (``bind(..., 0)``, the in-process
#: servers, the browsers) comes from the ephemeral range, and every other
#: test's serve port is even.  Found 2026-11-15: a free pair picked from the
#: ephemeral range lost its notebook port to another worker's server a minute
#: later, and the notebook refused to start.
_SERVE_PORTS = range(20000, 32000, 2)


def free_serve_port() -> int:
    """A port for `serve start` whose notebook port (`jupyter.jupyter_port`)
    is free here too -- an even one below the ephemeral range
    (:data:`_SERVE_PORTS`), the pair found free on loopback."""
    import random
    import socket

    from molbuilder.jupyter import jupyter_port
    for port in random.sample(_SERVE_PORTS, 50):
        try:
            for each in (port, jupyter_port(port)):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", each))
        except OSError:
            continue
        return port
    raise RuntimeError("no free port pair")
