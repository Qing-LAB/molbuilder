"""embedIo -- the ONE seal over the 3Dmol embed handle (molview-render-streamline.md §9.1).

Node unit test: embedIo is a THIN translator. We drive it with a STUB handle that records
every call, and assert it maps the engine's plain data (§7.3 ProcessedFrame) onto the right
embed doors -- the multi-frame load sequence, the native swap, the append path, the overlay
forwarding, and busy. No 3Dmol here; the handle is a spy.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/engine/embed-io.js"


_BOOT = """
            // A spy handle: every door records (name, args); reads return canned values.
            function makeHandle(reads) {
                reads = reads || {};
                const calls = [];
                const rec = (name) => (...args) => { calls.push({ name, args }); };
                return {
                    _calls: calls,
                    setStructure:      rec("setStructure"),
                    setAnimation:      rec("setAnimation"),
                    setAnimationFrame: rec("setAnimationFrame"),
                    appendFrames:      rec("appendFrames"),
                    appendFrameArrows: rec("appendFrameArrows"),
                    setLabels:         rec("setLabels"),
                    setAtomLabels:     rec("setAtomLabels"),
                    setOverlays:       rec("setOverlays"),
                    setArrows:         rec("setArrows"),
                    setCell:           rec("setCell"),
                    setAxes:           rec("setAxes"),
                    setBusy:           rec("setBusy"),
                    getFrameCount:     () => reads.frameCount,
                    getAnimationFrame: () => reads.currentFrame,
                    getAnimationKind:  () => reads.kind,
                };
            }
            const embedIo = globalThis.molbuilder.molview.engine.embedIo;
            // Two processed frames: 2 atoms each. (positions differ per frame.)
            const F = [
                { positions: [[0,0,0],[1,0,0]], elements: ["C","H"],
                  arrows: [{start:[0,0,0],end:[1,0,0]}], labels: {atoms:"all"}, halos: {atoms:[0]} },
                { positions: [[0,0,1],[1,0,1]], elements: ["C","H"],
                  arrows: [{start:[0,0,1],end:[1,0,1]}], labels: {atoms:"all"}, halos: {atoms:[0]} },
            ];
            const names = (h) => h._calls.map(c => c.name);
            const byName = (h, n) => h._calls.filter(c => c.name === n);
"""


def _run_node(snippet: str) -> object:
    # ES-module harness (tests/_node_esm): imports the embed-io module (ESM), which publishes
    # its global via the transitional shim; the bootstrap + snippet drive it through the global.
    return run_node([MODULE], _BOOT + snippet)


def test_multiframe_load_is_setStructure_then_setAnimation():
    out = _run_node("""
        const h = makeHandle();
        embedIo.create(h).loadFrames({ frames: F, cellBox: null });
        const anim = byName(h, "setAnimation")[0].args[0];
        console.log(JSON.stringify({
            order: names(h),
            f0xyz: byName(h, "setStructure")[0].args[0].xyz.split("\\n")[0],  // atom count line
            animKind: anim.kind,
            animFrames: anim.frames,                 // per-frame positions
            arrowsPerFrame: anim.arrowsPerFrame.length,
            frameStrip: anim.frameStrip,
        }));
    """)
    # frame0 establishes identity via setStructure, THEN the native movie is built.
    assert out["order"][0] == "setStructure"
    assert "setAnimation" in out["order"]
    assert out["f0xyz"] == "2"                       # 2 atoms
    assert out["animKind"] == "trajectory"
    assert out["animFrames"] == [[[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 1]]]
    assert out["arrowsPerFrame"] == 2                # one arrow set per frame, baked in
    assert out["frameStrip"] is False               # MolView owns the frame bar, not the embed
    # loadFrames is COORDINATES ONLY -- overlays (labels/halos/arrows) are the engine's job,
    # applied right after. loadFrames must not touch those doors.
    assert "setLabels" not in out["order"]
    assert "setOverlays" not in out["order"]
    assert "setArrows" not in out["order"]


def test_single_frame_load_is_plain_structure_no_movie():
    out = _run_node("""
        const h = makeHandle();
        embedIo.create(h).loadFrames({ frames: [F[0]] });
        console.log(JSON.stringify({ order: names(h) }));
    """)
    # one frame = a static structure: setStructure ONLY (no movie, no frame bar). Overlays --
    # including this frame's arrows -- are applied by the engine afterwards, not here.
    assert out["order"] == ["setStructure"]


def test_swap_frame_is_native_no_rebuild():
    out = _run_node("""
        const h = makeHandle();
        embedIo.create(h).swapFrame(3);
        console.log(JSON.stringify({ calls: h._calls }));
    """)
    assert out["calls"] == [{"name": "setAnimationFrame", "args": [3]}]  # ONLY the native swap


def test_append_extends_movie_with_new_frames_and_arrows():
    out = _run_node("""
        const h = makeHandle();
        embedIo.create(h).appendFrames({ frames: [F[1]] });
        console.log(JSON.stringify({
            appendFrames: byName(h, "appendFrames")[0].args[0],
            appendArrows: byName(h, "appendFrameArrows")[0].args[0].length,
        }));
    """)
    assert out["appendFrames"] == [[[0, 0, 1], [1, 0, 1]]]   # new frame's positions only
    assert out["appendArrows"] == 1                          # its arrows appended alongside


def test_setFrameArrows_rebakes_in_place_without_coord_reload():
    out = _run_node("""
        const h = makeHandle();
        embedIo.create(h).setFrameArrows([[{start:[0,0,0],end:[0,1,0]}], []]);
        const anim = byName(h, "setAnimation");
        console.log(JSON.stringify({
            names: names(h),
            arrowsPerFrame: anim.length ? anim[0].args[0].arrowsPerFrame.length : -1,
        }));
    """)
    # a partial setAnimation({arrowsPerFrame}) ONLY -- no setStructure / addModelsAsFrames.
    assert out["names"] == ["setAnimation"]
    assert "setStructure" not in out["names"]
    assert out["arrowsPerFrame"] == 2              # both frames' arrows re-handed


def test_apply_overlays_forwards_only_present_fields():
    out = _run_node("""
        const h = makeHandle();
        const io = embedIo.create(h);
        io.applyOverlays({ labels: [{position:[0,0,0],text:"1"}], halos: null });  // arrows/cell/axes absent
        console.log(JSON.stringify({ names: names(h) }));
    """)
    # only the two provided doors are touched; an absent field never clears another overlay.
    assert "setAtomLabels" in out["names"]     # labels -> the explicit-text door
    assert "setOverlays" in out["names"]
    assert "setArrows" not in out["names"]
    assert "setCell" not in out["names"]
    assert "setAxes" not in out["names"]


def test_setBusy_and_reads_delegate():
    out = _run_node("""
        const h = makeHandle({ frameCount: 7, currentFrame: 3, kind: "trajectory" });
        const io = embedIo.create(h);
        io.setBusy("Updating…");
        console.log(JSON.stringify({
            busy: byName(h, "setBusy")[0].args[0],
            frameCount: io.frameCount(),
            currentFrame: io.currentFrame(),
            kind: io.animationKind(),
        }));
    """)
    assert out["busy"] == "Updating…"
    assert out["frameCount"] == 7
    assert out["currentFrame"] == 3
    assert out["kind"] == "trajectory"


def test_create_requires_a_handle():
    out = _run_node("""
        let threw = false;
        try { embedIo.create(null); } catch (e) { threw = true; }
        console.log(JSON.stringify({ threw }));
    """)
    assert out["threw"] is True
