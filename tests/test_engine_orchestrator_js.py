"""engine -- the orchestrator / single render place (molview-render-streamline.md §5, §8, §9).

Node unit test: we drive the engine with a STUB embedIo (records every primitive call) and a
STUB store (getState + subscribe + a setter that fires subscribers), plus the REAL process +
index helper. We assert the §8 tier chosen for each change:
  setData          -> STRUCTURAL REGEN (busy + loadFrames of all frames)
  showFrame        -> NATIVE SWAP (swapFrame + overlay re-apply; no busy, no reload)
  toggle showIndex -> OVERLAY REFRESH (applyOverlays; no reload, no busy)
  toggle isolate   -> STRUCTURAL REGEN (reload -- drawn atom set changed)
  appendFrames     -> APPEND (validate + extend; hard error on atom-count mismatch)
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODS = [
    ROOT / "molbuilder/web/static/lib/molview/_atom-index.js",
    ROOT / "molbuilder/web/static/lib/molview/engine/process.js",
    ROOT / "molbuilder/web/static/lib/molview/engine/engine.js",
]


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = ("global.window = global;\n"
            + "\n".join(m.read_text() for m in MODS) + "\n"
            + """
            // Stub embedIo: record every primitive call by name.
            function makeIo() {
                const calls = [];
                const rec = (name) => (...args) => { calls.push({ name, args }); };
                return { _calls: calls, _names: () => calls.map(c => c.name),
                    loadFrames: rec("loadFrames"), swapFrame: rec("swapFrame"),
                    appendFrames: rec("appendFrames"), applyOverlays: rec("applyOverlays"),
                    setBusy: rec("setBusy"),
                    frameCount: () => 0, currentFrame: () => 0, animationKind: () => null };
            }
            // Stub store: getState + subscribe + a _set that patches state and fires subscribers.
            function makeStore(init) {
                let s = Object.assign({ indices: [], isolate: false, showIndex: false,
                    showForces: false, showCell: false, showAxis: false }, init || {});
                const subs = [];
                return { getState: () => s, subscribe: (fn) => { subs.push(fn);
                    return () => { const i = subs.indexOf(fn); if (i>=0) subs.splice(i,1); }; },
                    _set: (p) => { s = Object.assign({}, s, p); subs.forEach(fn => fn(s)); } };
            }
            const engineNs = global.molbuilder.molview.engine;
            // 3-atom, 2-frame data set.
            const DATA = {
                frames: [ [[0,0,0],[1,0,0],[2,0,0]], [[0,0,1],[1,0,1],[2,0,1]] ],
                elements: ["C","H","O"], annotations: [{}, {label:"bridge"}, {}],
                cell: null, forcesPerFrame: null,
            };
            const mk = (init) => { const io = makeIo(), store = makeStore(init);
                const e = engineNs.create({}, { embedIo: io, store: store });
                return { io, store, e }; };
            """
            + snippet)
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_setData_is_structural_regen_with_busy():
    out = _run_node("""
        const { io, e } = mk();
        e.setData(DATA);
        const load = io._calls.find(c => c.name === "loadFrames");
        console.log(JSON.stringify({
            names: io._names(),
            nFrames: load.args[0].frames.length,
            frameCount: e.frameCount(),
        }));
    """)
    # full load -> busy on, load ALL frames, busy off.
    assert out["names"][0] == "setBusy"                 # busy raised first
    assert "loadFrames" in out["names"]
    assert out["nFrames"] == 2                          # both frames processed + loaded
    assert out["frameCount"] == 2


def test_setData_busy_raised_then_cleared():
    out = _run_node("""
        const { io, e } = mk();
        e.setData(DATA);
        const busy = io._calls.filter(c => c.name === "setBusy").map(c => c.args[0]);
        console.log(JSON.stringify({ busy }));
    """)
    assert out["busy"][0] is not None      # scrim on (a message)
    assert out["busy"][-1] is None         # scrim cleared when ready


def test_showFrame_is_native_swap_no_busy_no_reload():
    out = _run_node("""
        const { io, e } = mk();
        e.setData(DATA);
        io._calls.length = 0;              // ignore the load; watch only the swap
        e.showFrame(1);
        console.log(JSON.stringify({ names: io._names(), current: e.currentFrame() }));
    """)
    assert "swapFrame" in out["names"]
    assert "applyOverlays" in out["names"]     # overlays follow the shown frame
    assert "loadFrames" not in out["names"]    # NO movie rebuild
    assert "setBusy" not in out["names"]       # NO busy for a pure swap
    assert out["current"] == 1


def test_toggle_showIndex_is_overlay_refresh_no_reload():
    out = _run_node("""
        const { io, store, e } = mk();
        e.setData(DATA);
        io._calls.length = 0;
        store._set({ showIndex: true });   // a flag write fires the engine's subscription
        console.log(JSON.stringify({ names: io._names() }));
    """)
    assert out["names"] == ["applyOverlays"]   # overlay refresh ONLY -- no reload, no busy


def test_toggle_isolate_is_structural_regen():
    out = _run_node("""
        const { io, store, e } = mk();
        e.setData(DATA);
        io._calls.length = 0;
        store._set({ indices: [1], isolate: true });   // drawn atom set changes
        console.log(JSON.stringify({ names: io._names() }));
    """)
    assert "loadFrames" in out["names"]        # movie rebuilt -- trajectory SURVIVES the isolate
    assert "setBusy" in out["names"]           # structural regen raises busy


def test_selection_change_while_not_isolating_is_overlay_only():
    out = _run_node("""
        const { io, store, e } = mk();
        e.setData(DATA);
        io._calls.length = 0;
        store._set({ indices: [2] });          // isolate OFF -> only the selection halo changes
        console.log(JSON.stringify({ names: io._names() }));
    """)
    assert "loadFrames" not in out["names"]    # no reload -- just a halo change
    assert out["names"] == ["applyOverlays"]


def test_append_valid_extends_movie_without_moving_frame():
    out = _run_node("""
        const { io, e } = mk();
        e.setData(DATA);
        io._calls.length = 0;
        e.appendFrames([ [[0,0,2],[1,0,2],[2,0,2]] ]);   // one new 3-atom frame
        console.log(JSON.stringify({
            names: io._names(),
            appended: io._calls.find(c => c.name === "appendFrames").args[0].frames.length,
            frameCount: e.frameCount(), current: e.currentFrame(),
        }));
    """)
    assert "appendFrames" in out["names"]
    assert "loadFrames" not in out["names"]    # extend, not reload (§6.2)
    assert out["appended"] == 1
    assert out["frameCount"] == 3
    assert out["current"] == 0                 # append does NOT move the shown frame


def test_append_atom_count_mismatch_is_hard_error():
    out = _run_node("""
        const { e } = mk();
        e.setData(DATA);
        let threw = false, msg = "";
        try { e.appendFrames([ [[0,0,0],[1,0,0]] ]); }   // 2 atoms, expected 3
        catch (err) { threw = true; msg = err.message; }
        console.log(JSON.stringify({ threw, hasCount: /expected 3/.test(msg) }));
    """)
    assert out["threw"] is True                # never coerce -- hard error
    assert out["hasCount"] is True


def test_append_before_load_is_hard_error():
    out = _run_node("""
        const { e } = mk();
        let threw = false;
        try { e.appendFrames([ [[0,0,0]] ]); } catch (_) { threw = true; }
        console.log(JSON.stringify({ threw }));
    """)
    assert out["threw"] is True                # no identity to append to


# ---- the async paint-yield path (browser has requestAnimationFrame; node doesn't, so we
#      inject a controllable one to exercise the busy scrim + burst coalescing) ------------- #

_RAF = """
    let _q = [], _id = 0;
    global.requestAnimationFrame = (fn) => { _q.push({ id: ++_id, fn }); return _id; };
    global.cancelAnimationFrame  = (id) => { _q = _q.filter(x => x.id !== id); };
    // The regen uses a NESTED double rAF, so draining re-fills the queue -- loop till empty.
    function flushRaf() { let g = 0; while (_q.length && g++ < 100) { _q.shift().fn(); } }
"""


def test_paint_yield_shows_busy_before_the_blocking_load():
    out = _run_node(_RAF + """
        const { io, e } = mk();
        e.setData(DATA);                       // schedules the regen behind a paint yield
        const before = { names: io._names(),
                         busy: io._calls.filter(c => c.name === "setBusy").map(c => c.args[0]) };
        flushRaf();                            // the paint happened -> now the heavy load runs
        const after = io._names();
        console.log(JSON.stringify({ before, after }));
    """)
    # BEFORE the paint: busy scrim is on, but loadFrames has NOT run (that's the whole point --
    # the scrim paints before the freeze).
    assert out["before"]["busy"] == ["Updating view…"]
    assert "loadFrames" not in out["before"]["names"]
    # AFTER the paint yield: the load runs and busy clears.
    assert "loadFrames" in out["after"]
    assert out["after"][-1] == "setBusy"       # last call clears busy (null)


def test_burst_of_structural_changes_coalesces_to_one_load():
    out = _run_node(_RAF + """
        const { io, store, e } = mk();
        e.setData(DATA);                       // regen #1 scheduled
        store._set({ indices: [1], isolate: true });   // regen #2 -> cancels #1, reschedules
        store._set({ indices: [1, 2], isolate: true }); // regen #3 -> cancels #2, reschedules
        flushRaf();
        const loads = io._calls.filter(c => c.name === "loadFrames");
        console.log(JSON.stringify({
            loadCount: loads.length,
            drawn: loads.length ? loads[0].args[0].frames[0].positions.length : -1,
        }));
    """)
    # a burst collapses to ONE load of the LATEST state (isolate on, selection {1,2} -> 2 atoms).
    assert out["loadCount"] == 1
    assert out["drawn"] == 2
