"""The page around MolView's view lane, as a node test plays it (`web/
molview.md` § 11.2b, § 13.1): the real model and the real lane, the app's own
routes answering (`_node_esm.served`), the real workspace in one browser tab on
Node's own storage (`_node_esm.tab_storage`, ``run_node(..., storage=True)``);
the one stand-in the pose pair at the engine seam, the drawing library being
one node cannot run.  `test_molview_ui_context` drives it.  The lane is
attached as `mount.js` attaches it -- its `hasTruthLane` rule, read-only
viewers alone carrying a selection -- by hand, since the mount itself needs
a DOM: the page part of the tests stretch drives the real mount.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import lifted

REPO = Path(__file__).resolve().parents[2]
_LIB = REPO / "molbuilder" / "web" / "static" / "lib"
#: The page that hosts MolView, as a test plays it: it hands the viewer the
#: one reader of a server's answer (`molview_host.mjs`).
HOST = Path(__file__).resolve().parent / "molview_host.mjs"
UICTX = _LIB / "molview" / "ui-context.js"
#: The workspace's one door, and what a page keeps in this browser tab --
#: where the tab's own id lives (`runtime.md` § 3).
WS = _LIB / "workspace" / "dispatcher.js"
KEPT = _LIB / "page-prefs.js"

#: THE LANE'S OWN WRITE DELAY -- § 11.2b: changes are gathered and written
#: once; a test waits that long, then for the write on the wire, rather than
#: a number of its own.
LANE_DELAY = lifted(UICTX, "const WRITE_DELAY_MS = ", ";")


def prelude(pairs) -> str:
    """The page's script before a test's own: ``pairs`` -- name to path, the
    structures a test opens by name (``opened(viewer, name)``) -- and the
    lane's doors: ``session``, ``settled``, ``writes``, ``ws``."""
    return f"""
const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
const {{ attachUiContext }} =
    await import({json.dumps(UICTX.resolve().as_uri())});
await import({json.dumps(KEPT.resolve().as_uri())});
await import({json.dumps(WS.resolve().as_uri())});
const ws = globalThis.molbuilder.workspace;
const PAIRS = {json.dumps({k: str(v) for k, v in pairs.items()})};

// The pose pair, a stand-in at the engine seam (§ 9.7's two bounded
// askings): node cannot run the drawing library.
function makeEngine(pose) {{
    return {{
        pose: pose || null,
        pointed: [],
        getCamera() {{ return this.pose ? this.pose.slice() : null; }},
        setCamera(p) {{ this.pointed.push(p.slice()); return true; }},
    }};
}}

// The lane's gathered write, then the wire quiet: written and answered.
const settled = async () => {{
    {LANE_DELAY};
    await new Promise((r) => setTimeout(r, WRITE_DELAY_MS));
    await __quiet();
}};

async function session(opts) {{
    const owner = (opts && opts.owner) || "s";
    const model = createModel(Object.assign({{ owner, workspace: ws }},
                                            (opts && opts.model) || {{}}));
    const engine = makeEngine((opts && opts.pose) || null);
    const off = attachUiContext({{
        model, engine, workspace: ws, owner,
        canvasEl: null,
        hasTruthLane: !(opts && opts.model && opts.model.mode === "readonly"),
    }});
    return {{ model, engine, off }};
}}

async function opened(one, name) {{
    await one.model.installMolecule({{ path: PAIRS[name] }});
    await __quiet();
}}

const writes = () => __calls.filter(
    (c) => c.url.endsWith("/api/workspace-storage/write"));
"""
