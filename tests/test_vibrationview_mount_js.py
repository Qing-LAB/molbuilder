"""VibrationView mount -- the concealed normal-mode viewer wrapper (vibrationview.md §1/§4).

Node unit test with a STUBBED viewer embed (records the calls VibrationView makes).  Pins the
Phase-1 wrap: showMode scatters the eigenvector + drives the embed's setAnimation({kind:
"vibration"}); a mode requested before the viewer is ready is deferred until the baseline is
drawn; amplitude / speed are LIVE partial updates; play/pause/dispose delegate to the embed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    ROOT / "molbuilder/web/static/lib/vibrationview/mode-math.js",
    ROOT / "molbuilder/web/static/lib/vibrationview/vibrationview.js",
]


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = ("global.window = global;\n"
            + "\n".join(m.read_text() for m in MODULES) + "\n" + snippet)
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# A stubbed embed: captures onReady (so the test controls readiness) + records every call
# VibrationView makes onto the handle.
_HARNESS = """
    let capturedOnReady = null;
    const calls = [];
    const h = {
        setStructure:       (o) => calls.push(['setStructure', o.xyz.split('\\n')[0]]),
        setOverlays:        (o) => calls.push(['setOverlays', o ? o.atoms[0].indices : null]),
        setAnimation:       (o) => calls.push(['setAnimation', o]),
        playAnimation:      () => calls.push(['play']),
        pauseAnimation:     () => calls.push(['pause']),
        isAnimationPlaying: () => true,
        refit:              () => calls.push(['refit']),
        dispose:            () => calls.push(['dispose']),
    };
    global.molbuilder.viewer = { embed: (host, opts) => { capturedOnReady = opts.onReady; return h; } };
    const host = {};
    const mount = global.molbuilder.vibrationview.mount;
"""


def test_showMode_scatters_and_drives_the_embed_vibration():
    out = _run_node(_HARNESS + """
        const geom = { elements:['O','H','H'], positions:[[0,0,0],[1,0,0],[0,1,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[1,2], frozenAtomIdx:[0],
                                  amplitude:0.2, speedHz:1.5 });
        capturedOnReady(h);                                    // viewer ready -> baseline + grey
        vib.showMode({ index: 3, displacements: [[1,0,0],[0,1,0]] });  // free rows for atoms 1,2
        console.log(JSON.stringify({ calls, mode: vib.getMode() }));
    """)
    # equilibrium drawn (3-atom xyz header "3"); frozen atom 0 greyed
    assert ["setStructure", "3"] in out["calls"]
    assert ["setOverlays", [0]] in out["calls"]
    # the vibration handed to the embed: the free eigenvector scattered to global order
    anim = [c[1] for c in out["calls"] if c[0] == "setAnimation"][-1]
    assert anim["kind"] == "vibration"
    assert anim["displacements"] == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]   # frozen0 zero; free->1,2
    assert anim["amplitude"] == 0.2 and anim["speedHz"] == 1.5
    assert out["mode"] == 3


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
    assert "setAnimation" in out["afterKinds"]               # then the deferred mode applied


def test_amplitude_and_speed_are_live_partial_updates():
    out = _run_node(_HARNESS + """
        const geom = { elements:['H','H'], positions:[[0,0,0],[1,0,0]] };
        const vib = mount(host, { geometry: geom, freeAtomIdx:[0,1] });
        capturedOnReady(h);
        vib.showMode({ index:1, displacements:[[1,0,0],[0,0,1]] });
        calls.length = 0;                                    // clear; only capture the control edits
        vib.setAmplitude(0.5);
        vib.setSpeed(2.0);
        console.log(JSON.stringify(calls));
    """)
    assert ["setAnimation", {"amplitude": 0.5}] in out       # no structure rebuild -- partial update
    assert ["setAnimation", {"speedHz": 2.0}] in out


def test_play_pause_dispose_delegate_to_the_embed():
    out = _run_node(_HARNESS + """
        const geom = { elements:['H'], positions:[[0,0,0]] };
        const vib = mount(host, { geometry: geom });
        capturedOnReady(h);
        vib.play(); vib.pause(); const playing = vib.isPlaying(); vib.dispose();
        console.log(JSON.stringify({
            seq: calls.filter(c => ['play','pause','dispose'].includes(c[0])).map(c => c[0]),
            playing }));
    """)
    assert out["playing"] is True
    assert out["seq"] == ["play", "pause", "pause", "dispose"]   # dispose pauses first, then disposes
