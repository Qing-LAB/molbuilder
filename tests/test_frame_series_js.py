"""Frame series -- the coordinate time axis (workspace-contract.md §1.5).

Node unit test of the PURE frame-series core: it holds frames[] + optional forces[] +
currentFrame, enforces the SAME-ATOMS invariant (a mismatched frame is a HARD ERROR, not
coerced), and every read returns a defensive copy.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "molbuilder/web/static/lib/molview/_frame-series.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = "global.window = global;\n" + MOD.read_text() + "\n" + snippet
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_NEW = "const s = global.molbuilder.molview._createFrameSeries();"


def test_reset_populates_frames_and_defaults_to_frame_0():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0],[1,0,0]], [[0,1,0],[1,1,0]], [[0,2,0],[1,2,0]] ]);
        console.log(JSON.stringify({
            frameCount: s.frameCount(), atomCount: s.atomCount(),
            currentFrame: s.currentFrame(), cur: s.currentCoords() }));
    """)
    assert out["frameCount"] == 3 and out["atomCount"] == 2
    assert out["currentFrame"] == 0
    assert out["cur"] == [[0, 0, 0], [1, 0, 0]]          # frame 0 by default


def test_setFrame_selects_and_out_of_range_throws():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0]], [[9,0,0]], [[8,0,0]] ]);
        s.setFrame(2);
        const cur = s.currentCoords(); const idx = s.currentFrame();
        let threw = false;
        try { s.setFrame(5); } catch (_) { threw = true; }
        console.log(JSON.stringify({ idx, cur, threw }));
    """)
    assert out["idx"] == 2 and out["cur"] == [[8, 0, 0]]
    assert out["threw"] is True                          # out of range -> hard error


def test_addFrame_appends_and_mismatched_count_is_a_hard_error():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0],[1,0,0]] ]);       // 2 atoms established
        s.addFrame([[0,1,0],[1,1,0]]);        // OK -> 2 frames
        const okCount = s.frameCount();
        let threw = false;
        try { s.addFrame([[0,0,0],[1,0,0],[2,0,0]]); } catch (_) { threw = true; }  // 3 atoms!
        console.log(JSON.stringify({ okCount, afterThrow: s.frameCount(), threw }));
    """)
    assert out["okCount"] == 2
    assert out["threw"] is True                          # 3 atoms vs 2 -> rejected
    assert out["afterThrow"] == 2                        # not appended


def test_forces_ride_per_frame_and_must_match_atom_count():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0],[1,0,0]] ], { forces: [ [[0.1,0,0],[0,0.2,0]] ] });
        const f0 = s.getForces(0);
        let threw = false;
        try { s.addFrame([[0,1,0],[1,1,0]], { forces: [[9,9,9]] }); } catch (_) { threw = true; }
        console.log(JSON.stringify({ f0, threw, count: s.frameCount() }));
    """)
    assert out["f0"] == [[0.1, 0, 0], [0, 0.2, 0]]
    assert out["threw"] is True                          # 1 force vs 2 atoms -> rejected
    assert out["count"] == 1


def test_reloadFrames_replaces_the_whole_set():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0]], [[1,0,0]] ]);    // 2 frames, 1 atom
        s.reloadFrames([ [[5,0,0],[6,0,0]], [[7,0,0],[8,0,0]], [[9,0,0],[9,1,0]] ]);  // 3 frames, 2 atoms
        console.log(JSON.stringify({ frameCount: s.frameCount(), atomCount: s.atomCount(),
                                     currentFrame: s.currentFrame() }));
    """)
    assert out["frameCount"] == 3 and out["atomCount"] == 2
    assert out["currentFrame"] == 0                      # reload resets to frame 0


def test_reads_return_defensive_copies():
    out = _run_node(_NEW + """
        s.reset([ [[0,0,0],[1,0,0]] ]);
        const a = s.getFrame(0);
        a[0][0] = 999;                        // mutate the returned copy
        console.log(JSON.stringify({ mutated: a[0][0], internal: s.getFrame(0)[0][0] }));
    """)
    assert out["mutated"] == 999 and out["internal"] == 0    # internal untouched -> copy
