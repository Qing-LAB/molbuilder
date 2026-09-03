"""THE way a test starts a real HTTP server for the app.

Twenty test modules each defined a ``flask_server`` fixture, and sixteen of
them were byte-identical in two groups: make a `werkzeug` server on an
ephemeral port, run it in a daemon thread, yield ``http://127.0.0.1:<port>``,
shut it down.  The other four differed only in spelling.

**One implementation, and each module keeps its own fixture scope.**  The
scopes are not uniform today — most are ``module``, a few are function — and a
scope is a decision about how much state a file's tests share, not something a
de-duplication gets to change on their behalf.  So this is a context manager
rather than a fixture: a module writes

    @pytest.fixture(scope="module")          # or plain @pytest.fixture
    def flask_server():
        with serve() as base_url:
            yield base_url

and the fourteen lines that used to sit under it are here, once.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


@contextmanager
def serve(config: dict | None = None):
    """Run the app on an ephemeral port; yield its base URL.

    ``config`` is passed straight to ``create_app`` (``{}`` by default, which
    is what every caller used).  The port is chosen by the OS — 0 — so
    concurrent test modules never collide on a fixed number, and the thread is
    a daemon so a hung shutdown cannot keep the interpreter alive.
    """
    from werkzeug.serving import make_server

    from molbuilder.web.app import create_app

    server = make_server("127.0.0.1", 0, create_app(config=config or {}),
                         threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
