"""The trajectory viewer's rebuild reaches its read-only viewer.

A live run's poll is applied as an append when it provably extends the movie,
and as a REBUILD otherwise -- the first trajectory from a single-frame load,
or a frame already shown that changed, the step in flight now finished
(`web/trajectory.md` § 5).  The rebuild installs the whole series again into
the viewer the tab mounted read-only, and a read-only MolView takes a second
install only when its host says it means it (`web/molview.md` § 9.4,
``enforce``).  Without it the install was refused in silence: the frame
counter grew while the movie kept the old frames.

**This RUNS the real code on both sides**: the viewer's ``rebuildModel``,
lifted from the shipped module, against MolView's real model in read-only
mode -- its server answered by `test_molview_model`'s stand-in -- so what is
asserted is the frames MolView holds afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node
from tests.test_molview_model import MODEL, SERVER

CORE = (Path(__file__).resolve().parents[1]
        / "molbuilder/web/static/lib/trajectory/core.js")


def _rebuild_model() -> str:
    src = CORE.read_text(encoding="utf-8")
    i = src.index("    async function rebuildModel(seekIdx) {")
    return src[i:src.index("    // (No local showFrame()", i)]


#: What `rebuildModel` reaches outside itself, at the edge: the mounted
#: viewer (MolView's real model, read-only), the tab's state, and the page.
HARNESS = """
const { createModel } = await import(%s);
const m = createModel({ mode: "readonly" });
const _mv = {};
const _mvReady = Promise.resolve();
const _mvdata = () => m;
const $ = () => null;
const said = [];
function setStatus(text) { said.push(text); }
function buildForcesPerFrame() { return null; }
globalThis.__payload = undefined;
globalThis.__nextPayload = { atoms: [__atomRow(0, "C", 0), __atomRow(1, "O", 1)] };
/* A run's frames as the server parses them: `[element, x, y, z]` a row. */
const frames = (n, moved) => Array.from({ length: n }, (_, i) => [
    ["C", 0.1 * i, 0, 0],
    ["O", 1 + 0.1 * i + (i === moved ? 0.5 : 0), 0, 0],
]);
const state = { structure: { elements: ["C", "O"] }, path: "/p/run.out",
                data: { frames: frames(4) } };
"""


def test_a_rebuild_replaces_the_movie_in_a_read_only_viewer():
    """Four frames shown; the next poll brings six, and the fourth --
    the one in flight at the last read -- moved: the viewer holds the six,
    the fourth where the run now says it is."""
    out = run_node([], HARNESS % json.dumps(MODEL.resolve().as_uri())
                   + _rebuild_model() + """
        await rebuildModel(3);
        const first = m.frameCount();
        state.data = { frames: frames(6, 3) };
        await rebuildModel(5);
        const held = m.getStructure().frames;
        console.log(JSON.stringify({
            first, then: held.length, fourth: held[3][1][0], said }));
    """, globals_js=SERVER)
    assert out["first"] == 4, out
    assert out["then"] == 6, (
        "the rebuild did not reach the read-only viewer: it still holds "
        f"{out['then']} frames")
    assert abs(out["fourth"] - 1.8) < 1e-9, (
        "the frame that moved shows where it was, not where the run says")
    assert out["said"] == [], out
