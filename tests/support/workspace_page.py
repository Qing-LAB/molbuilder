"""A page's workspace in one browser tab, as a node test plays it -- ONE door
(`web/workspace.md` § 5, `web/runtime.md` § 3, `process/testing.md` § 4).

The page loads what a page keeps in this browser tab first (`page-prefs.js`,
where the tab's own id lives; `_app_header.html`), then the header's other
scripts a test names (``header``, the notification bar among them), then the
workspace's one door (`workspace/dispatcher.js`) -- on Node's own storage,
the tab's own (`_node_esm.tab_storage`): a test that carries one opening's
storage into the next plays that tab reloaded.  On the page: ``ws``, the workspace's door, and
``writes()``, its writes on the wire when the app answers (`_node_esm.served`).

MolView's page builds on this one (`support.molview_page`).
"""
from __future__ import annotations

from pathlib import Path

from tests._node_esm import run_node, served, tab_storage

_LIB = Path(__file__).resolve().parents[2] / "molbuilder" / "web" / "static" / "lib"
#: What a page keeps in this browser tab, and the workspace's one door.
KEPT = _LIB / "page-prefs.js"
WS = _LIB / "workspace" / "dispatcher.js"

#: The page's script once the workspace is loaded.
PRELUDE = """
/* The workspace a page hands its viewers (workspace.md § 5): the real one,
 * its files on the app's own routes, this browser tab its own. */
const ws = globalThis.molbuilder.workspace;
const writes = () => __calls.filter(
    (c) => c.url.endsWith("/api/workspace-storage/write"));
"""


def run(app_at: str | None, snippet: str, *, tab=None, header=(),
        globals_js: str = ""):
    """``snippet`` on the page in one browser tab: ``app_at`` -- the app
    answering its requests (``None``: a snippet that sends none); ``tab`` --
    what the tab kept in an earlier opening; ``header`` -- the header's
    other scripts, loaded after what the page keeps and before the
    workspace; ``globals_js`` -- what a test puts on the page before any of
    its scripts."""
    return run_node([KEPT, *header, WS], PRELUDE + snippet,
                    globals_js=(tab_storage(tab)
                                + (served(app_at) if app_at else "")
                                + globals_js),
                    storage=True)
