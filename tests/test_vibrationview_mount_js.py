"""VibrationView mount -- the concealed normal-mode viewer wrapper (vibrationview.md §1/§4).

Node unit test with a STUBBED viewer embed (records the calls VibrationView makes).  Pins the
Phase-2 seal (task #51): VibrationView OWNS the animation -- the phase clock, play/pause
state, amplitude/speed, and the per-tick math -- and drives the embed's generic
``setAtomCoords`` door per rAF; export capture goes through the external-animator provider
(``setAnimationProvider({frameCoords, restCoords, cycleSec})``).  The embed's old
``setAnimation({kind:"vibration"})`` was deleted with the move; a mode requested before the
viewer is ready is still deferred until the baseline is drawn.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
# vibrationview is a native ES module: it IMPORTS mode-math + the shared xyz writer itself,
# so the harness loads only the door module (imports resolve relatively on disk).
MODULES = [
    ROOT / "molbuilder/web/static/lib/vibrationview/vibrationview.js",
]


def _run_node(snippet: str) -> object:
    return run_node(MODULES, snippet)


# A stubbed embed: captures onReady (so the test controls readiness) + records every call
# VibrationView makes onto the handle.  A manual rAF queue stands in for the browser clock
# so the test steps the OWNED loop deterministically.
_HARNESS = """
    let capturedOnReady = null;
    const calls = [];
    let provider = null;
    const h = {
        setStructure:         (o) => calls.push(['setStructure', o.xyz.split('\\n')[0]]),
        setOverlays:          (o) => calls.push(['setOverlays', o ? o.atoms[0].indices : null]),
        setAtomCoords:        (c) => calls.push(['setAtomCoords', c]),
        setAnimationProvider: (p) => { provider = p; calls.push(['setAnimationProvider', !!p]); },
        refit:                () => calls.push(['refit']),
        dispose:              () => calls.push(['dispose']),
    };
    global.molbuilder.viewer = { embed: (host, opts) => { capturedOnReady = opts.onReady; return h; } };
    // Deterministic rAF: queue + manual pump (the loop is vibrationview-owned now).
    let rafQ = [];
    global.requestAnimationFrame = (fn) => { rafQ.push(fn); return rafQ.length; };
    global.cancelAnimationFrame  = (id) => {};
    function pumpRaf(n) { for (let i = 0; i < n && rafQ.length; i++) { rafQ.shift()(); } }
    const host = {};
    const mount = global.molbuilder.vibrationview.mount;
"""


def test_showMode_scatters_registers_provider_and_ticks_setAtomCoords():
    out = _run_node(_HARNESS + """
        const geom = { elements:['O','H','H'], positions:[[0,0,0],[1,0,0],[0,1,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[1,2], frozenAtomIdx:[0],
                                  amplitude:0.2, speedHz:1.5 });
        capturedOnReady(h);                                    // viewer ready -> baseline + grey
        vib.showMode({ index: 3, displacements: [[1,0,0],[0,1,0]] });  // free rows for atoms 1,2
        pumpRaf(1);                                            // one owned-loop tick
        // Provider frame 0 of an N-frame capture = phase 0 = amplitude * cos(0) = +amplitude.
        const f0 = provider.frameCoords(0, 30, 1.0);
        console.log(JSON.stringify({
            calls: calls.filter(c => c[0] !== 'setAtomCoords').concat(
                   [['ticked', calls.some(c => c[0] === 'setAtomCoords')]]),
            f0, cycleSec: provider.cycleSec, rest: provider.restCoords(),
            mode: vib.getMode(), playing: vib.isPlaying() }));
    """)
    # equilibrium drawn (3-atom xyz header "3"); frozen atom 0 greyed
    assert ["setStructure", "3"] in out["calls"]
    assert ["setOverlays", [0]] in out["calls"]
    assert ["setAnimationProvider", True] in out["calls"]
    assert ["ticked", True] in out["calls"]        # the OWNED loop drove setAtomCoords
    # phase-0 capture frame: eq + 0.2 * scattered displacement (frozen atom 0 = zero row).
    assert out["f0"] == [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]]
    assert abs(out["cycleSec"] - 1 / 1.5) < 1e-9   # one cosine cycle = 1/speedHz
    assert out["rest"] == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]   # equilibrium restore
    assert out["mode"] == 3
    assert out["playing"] is True


def test_showMode_before_ready_is_deferred_until_baseline_drawn():
    out = _run_node(_HARNESS + """
        const geom = { elements:['H','H'], positions:[[0,0,0],[1,0,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[0,1] });
        vib.showMode({ index:1, displacements:[[1,0,0],[0,0,1]] });   // BEFORE ready -> deferred
        const beforeReady = calls.slice();
        capturedOnReady(h);                                          // ready -> baseline + flush
        console.log(JSON.stringify({ beforeReady, afterKinds: calls.map(c => c[0]) }));
    """)
    assert out["beforeReady"] == []                          # nothing drawn before ready
    assert "setStructure" in out["afterKinds"]               # baseline drawn on ready
    assert "setAnimationProvider" in out["afterKinds"]       # then the deferred mode applied


def test_amplitude_and_speed_are_live_variable_writes():
    """The D1/N1 slider regression home (moved from the embed with task #51):
    a slider drag must NEVER stop the loop.  Amplitude/speed are plain
    variable writes the owned tick reads next frame -- no embed call at all,
    no re-registration, and the loop keeps running."""
    out = _run_node(_HARNESS + """
        const geom = { elements:['H','H'], positions:[[0,0,0],[1,0,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[0,1], speedHz: 1.0 });
        capturedOnReady(h);
        vib.showMode({ index:1, displacements:[[1,0,0],[0,0,1]] });
        calls.length = 0;                                    // clear; only capture the control edits
        vib.setAmplitude(0.5);
        vib.setSpeed(2.0);
        const controlCalls = calls.slice();
        const stillPlaying = vib.isPlaying();
        // The next capture frame reflects the new amplitude immediately.
        const f0 = provider.frameCoords(0, 30, 1.0);
        console.log(JSON.stringify({ controlCalls, stillPlaying, f0,
                                     cycleAfter: provider.cycleSec }));
    """)
    assert out["controlCalls"] == []                # pure variable writes -- no embed traffic
    assert out["stillPlaying"] is True              # a slider drag never stops the loop
    assert out["f0"] == [[0.5, 0, 0], [1, 0, 0.5]]  # amplitude 0.5 at phase 0
    # cycleSec was captured at provider registration (speed 1.0); the live speed
    # variable drives the RUNNING loop -- the provider re-registers per mode.
    assert abs(out["cycleAfter"] - 1.0) < 1e-9


def test_showMode_carries_geometry_and_redraws_only_on_change():
    """The spectra flow: the viewer is mounted WITHOUT a geometry; each showMode carries the
    equilibrium structure + free/frozen (a mode is defined against its structure).  The
    baseline is drawn on the first mode, NOT redrawn for the next mode of the SAME structure
    (browsing modes), but redrawn when the structure changes (a different result)."""
    out = _run_node(_HARNESS + """
        const vib = mount(host, {});                            // no geometry at mount
        capturedOnReady(h);                                     // ready, nothing to draw yet
        const g1 = { elements:['H','H'], positions:[[0,0,0],[1,0,0]] };
        vib.showMode({ index:1, displacements:[[1,0,0],[0,0,1]], geometry:g1, freeAtomIdx:[0,1] });
        const afterFirst = calls.map(c => c[0]); calls.length = 0;
        vib.showMode({ index:2, displacements:[[0,1,0],[0,0,1]], geometry:g1, freeAtomIdx:[0,1] });
        const afterSameGeom = calls.map(c => c[0]); calls.length = 0;
        const g2 = { elements:['O','H','H'], positions:[[0,0,0],[1,0,0],[0,1,0]] };
        vib.showMode({ index:5, displacements:[[1,0,0],[0,1,0],[0,0,1]], geometry:g2, freeAtomIdx:[0,1,2] });
        const afterNewGeom = calls.map(c => c[0]);
        console.log(JSON.stringify({ afterFirst, afterSameGeom, afterNewGeom }));
    """)
    assert "setStructure" in out["afterFirst"]
    assert "setAnimationProvider" in out["afterFirst"]
    assert "setStructure" not in out["afterSameGeom"]   # same structure -> no rebuild
    assert "setAnimationProvider" in out["afterSameGeom"]  # but the new mode still animates
    assert "setStructure" in out["afterNewGeom"]        # new structure -> redraw the baseline


def test_play_pause_own_the_loop_and_dispose_clears_the_provider():
    out = _run_node(_HARNESS + """
        const geom = { elements:['H'], positions:[[0,0,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[0] });
        capturedOnReady(h);
        vib.showMode({ index:1, displacements:[[1,0,0]] });
        const playingAfterShow = vib.isPlaying();
        vib.pause();
        const playingAfterPause = vib.isPlaying();
        calls.length = 0;
        pumpRaf(3);                                    // paused -> the loop must NOT tick
        const tickedWhilePaused = calls.some(c => c[0] === 'setAtomCoords');
        vib.play();
        pumpRaf(1);
        const tickedAfterPlay = calls.some(c => c[0] === 'setAtomCoords');
        vib.dispose();
        console.log(JSON.stringify({
            playingAfterShow, playingAfterPause, tickedWhilePaused, tickedAfterPlay,
            providerCleared: calls.some(c => c[0] === 'setAnimationProvider' && c[1] === false),
            disposed: calls.some(c => c[0] === 'dispose') }));
    """)
    assert out["playingAfterShow"] is True     # showMode starts the owned loop
    assert out["playingAfterPause"] is False   # pause is vibrationview state, not embed state
    assert out["tickedWhilePaused"] is False   # a paused loop drives no coords
    assert out["tickedAfterPlay"] is True      # play resumes the owned loop
    assert out["providerCleared"] is True      # dispose deregisters the export provider
    assert out["disposed"] is True             # ...then disposes the embed
