"""MolView's page as a node test plays it -- ONE door for every MolView test
under node (`web/molview.md` § 13.1, `process/testing.md` § 4):

- the viewer through the page's host (`molview_host.mjs`: ``createModel``,
  ``mount``, ``ask`` -- the one reader of a server's answer a page hands
  its viewer), the app's own routes answering (`_node_esm.served`), each
  request kept in ``__calls`` and each answer in ``__answered``;
- the structures a test opens, saved by the road's saver and named
  (`support.road.saved_by_name`), opened by path as a tab opens a file:
  ``opened(target,
  name)`` -- ``target`` a model or a mounted viewer's ``data`` -- and
  ``loaded(name, opts)``, a model holding it, the wire's record starting
  after it;
- with ``workspace=True``, the page's workspace in one browser tab
  (`support.workspace_page`: ``ws``, ``writes()``), and the view lane's
  gathered write awaited by the lane's own delay (``settled()``, § 11.2b).

A test's own page script -- its helpers, a viewer mounted, the view lane
attached by hand -- is the test file's, put before its snippet.

What node cannot run -- the drawing library, a DOM -- a test adds to the
page itself (`molview_3dmol_standin.js`, `molview_dom_standin.js`), through
:func:`run`'s ``globals_js``, which runs before any of the page's modules.
"""
from __future__ import annotations

import json
from pathlib import Path

from support import workspace_page
from tests._node_esm import lifted, run_node, served

REPO = Path(__file__).resolve().parents[2]
_LIB = REPO / "molbuilder" / "web" / "static" / "lib"
#: The page that hosts MolView, as a test plays it: it hands the viewer the
#: one reader of a server's answer.
HOST = Path(__file__).resolve().parent / "molview_host.mjs"
UICTX = _LIB / "molview" / "ui-context.js"

#: THE VIEW LANE'S OWN WRITE DELAY -- § 11.2b: changes are gathered and
#: written once; a test waits that long, then for the write on the wire,
#: never a number of its own.
LANE_DELAY = lifted(UICTX, "const WRITE_DELAY_MS = ", ";")


def prelude(pairs: dict, *, workspace: bool = False) -> str:
    """The page's script before a test's own (see the module's header)."""
    page = f"""
const {{ createModel, mount, ask }} = await import({json.dumps(HOST.as_uri())});
const PAIRS = {json.dumps(pairs)};

/* A structure opened as a tab opens one -- by path, the server reading the
 * pair -- and the wire quiet after it: the load answered and, in an editable
 * viewer handed a workspace, point 0 written. */
async function opened(target, name, extra) {{
    const got = await target.installMolecule(
        Object.assign({{ path: PAIRS[name] }}, extra || {{}}));
    await __quiet();
    return got;
}}

/* A model holding `name`, the wire's record starting after it. */
async function loaded(name, opts) {{
    const m = createModel(opts || {{}});
    await opened(m, name);
    __calls.length = 0;
    __answered.length = 0;
    return m;
}}
"""
    if not workspace:
        return page
    page += f"""
/* The view lane's gathered write, then the wire quiet: written and
 * answered. */
{LANE_DELAY};
async function settled() {{
    await new Promise((r) => setTimeout(r, WRITE_DELAY_MS));
    await __quiet();
}}
"""
    return page


def run(app_at: str, pairs: dict, snippet: str, *, workspace: bool = False,
        globals_js: str = "", given: dict | None = None):
    """``snippet`` on the page, the app at ``app_at`` answering.
    ``workspace`` -- as :func:`prelude`; ``globals_js`` -- what a test puts
    on the page before any of its modules (the stand-ins for what node
    cannot run); ``given`` -- values a test worked out in Python, as
    constants of the same names."""
    consts = "".join(f"const {k} = {json.dumps(v)};\n"
                     for k, v in (given or {}).items())
    script = prelude(pairs, workspace=workspace) + consts + snippet
    if workspace:
        return workspace_page.run(app_at, script, globals_js=globals_js)
    return run_node([], script, globals_js=served(app_at) + globals_js)
