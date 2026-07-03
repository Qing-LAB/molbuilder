"""k-grid tiling -- render-pipeline layer 3 (atom-annotations.md § 6.3).

Node unit test for the pure compute: identity for [1,1,1], correct lattice
offsets, image ordering (original cell first), sourceIndex back-mapping, dim
clamping, and the cell requirement when tiling.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molview/kgrid.js"


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


_MK = "const tile = global.molbuilder.molview.tileKgrid;\n"


def test_identity_no_tiling_needs_no_cell():
    out = _run_node(_MK + """
        const r = tile([[0,0,0],[1,0,0]], null, [1,1,1]);
        console.log(JSON.stringify(r));
    """)
    assert out["nimages"] == 1
    assert out["positions"] == [[0, 0, 0], [1, 0, 0]]
    assert out["sourceIndex"] == [0, 1]


def test_2x1x1_tiles_by_lattice_vector_a():
    out = _run_node(_MK + """
        const r = tile([[0,0,0],[1,0,0]],
                       [[10,0,0],[0,10,0],[0,0,10]], [2,1,1]);
        console.log(JSON.stringify(r));
    """)
    assert out["nimages"] == 2
    # original cell (0,0,0) first, then image shifted by a=[10,0,0].
    assert out["positions"] == [[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]]
    assert out["sourceIndex"] == [0, 1, 0, 1]   # each image maps back to its unit atom


def test_2x2x1_image_count_and_offsets():
    out = _run_node(_MK + """
        const r = tile([[0,0,0]], [[5,0,0],[0,7,0],[0,0,9]], [2,2,1]);
        console.log(JSON.stringify(r));
    """)
    assert out["nimages"] == 4
    # (i,j,k) with k fixed 0: (0,0),(0,1)->+b,(1,0)->+a,(1,1)->+a+b
    assert out["positions"] == [[0, 0, 0], [0, 7, 0], [5, 0, 0], [5, 7, 0]]
    assert out["sourceIndex"] == [0, 0, 0, 0]


def test_dims_are_floored_and_clamped():
    out = _run_node(_MK + """
        const cell = [[1,0,0],[0,1,0],[0,0,1]];
        console.log(JSON.stringify({
            zero:  tile([[0,0,0]], cell, [0,1,1]).nimages,   // 0 -> 1
            neg:   tile([[0,0,0]], cell, [-3,1,1]).nimages,  // <1 -> 1
            frac:  tile([[0,0,0]], cell, [2.9,1,1]).nimages, // floor -> 2
        }));
    """)
    assert out["zero"] == 1 and out["neg"] == 1 and out["frac"] == 2


def test_tiling_without_cell_raises():
    out = _run_node(_MK + """
        let err = null;
        try { tile([[0,0,0]], null, [2,1,1]); } catch (e) { err = e.message; }
        console.log(JSON.stringify({ err: err }));
    """)
    assert out["err"] and "cell" in out["err"]
