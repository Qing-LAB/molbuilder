"""FrameSet -- the module-owned dynamic-coordinate table + time-index layer
(atom-annotations.md § 6.3, render-pipeline layer 1).

Node unit test: static (1 frame) coerces every t to 0; a trajectory clamps t to
[0, nframes-1]; setFrame drives currentFrame; identity is fixed across frames.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/frameset.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    full = "global.window = global;\n" + MODULE.read_text() + "\n" + snippet
    proc = subprocess.run([node, "--input-type=commonjs", "-e", full],
                          capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_MK = "const mk = global.molbuilder.molview.createFrameSet;\n"


def test_static_single_frame_coerces_everything_to_zero():
    # The 1-frame proof: a static structure is a 1-frame FrameSet; t is
    # irrelevant and always resolves to frame 0.
    out = _run_node(_MK + """
        const fs = mk([[[0,0,0],[1,0,0]]]);   // 1 frame, 2 atoms
        console.log(JSON.stringify({
            nframes: fs.nframes, natoms: fs.natoms, isStatic: fs.isStatic,
            c_neg: fs.coerce(-3), c_big: fs.coerce(99), c_nan: fs.coerce("x"),
            coordsAt5: fs.coordsAt(5),
        }));
    """)
    assert out["nframes"] == 1 and out["natoms"] == 2 and out["isStatic"] is True
    assert out["c_neg"] == 0 and out["c_big"] == 0 and out["c_nan"] == 0
    assert out["coordsAt5"] == [[0, 0, 0], [1, 0, 0]]   # frame 0 regardless of t


def test_trajectory_clamps_time_index_and_tracks_current():
    out = _run_node(_MK + """
        const fs = mk([
            [[0,0,0]], [[1,0,0]], [[2,0,0]],   // 3 frames, 1 atom, moving in x
        ]);
        const before = fs.currentFrame;
        const set2 = fs.setFrame(2);
        console.log(JSON.stringify({
            nframes: fs.nframes, isStatic: fs.isStatic,
            c_neg: fs.coerce(-1), c_over: fs.coerce(9), c_mid: fs.coerce(1),
            c_frac: fs.coerce(1.9),                 // floor -> 1
            before: before, set2: set2, current: fs.currentFrame,
            coordsAt1: fs.coordsAt(1), coordsNow: fs.coords(),
        }));
    """)
    assert out["nframes"] == 3 and out["isStatic"] is False
    assert out["c_neg"] == 0 and out["c_over"] == 2       # clamp
    assert out["c_mid"] == 1 and out["c_frac"] == 1       # floor
    assert out["before"] == 0 and out["set2"] == 2 and out["current"] == 2
    assert out["coordsAt1"] == [[1, 0, 0]]
    assert out["coordsNow"] == [[2, 0, 0]]                # currentFrame == 2


def test_nonfinite_t_holds_current_frame():
    out = _run_node(_MK + """
        const fs = mk([[[0,0,0]], [[1,0,0]], [[2,0,0]]]);
        fs.setFrame(2);
        console.log(JSON.stringify({ held: fs.coerce(undefined) }));
    """)
    assert out["held"] == 2   # non-finite t keeps the current frame, not 0


def test_inconsistent_atom_count_rejected():
    out = _run_node(_MK + """
        let err = null;
        try { mk([[[0,0,0],[1,0,0]], [[0,0,0]]]); }   // frame 1 drops an atom
        catch (e) { err = e.message; }
        console.log(JSON.stringify({ err: err }));
    """)
    assert out["err"] and "fixed across frames" in out["err"]


def test_empty_frames_rejected():
    out = _run_node(_MK + """
        let err = null;
        try { mk([]); } catch (e) { err = e.message; }
        console.log(JSON.stringify({ err: err }));
    """)
    assert out["err"] and "non-empty" in out["err"]
