"""VibrationView mode math -- the eigenvector free-atom -> global scatter (vibrationview.md §2/§5).

Node unit test of the pure scatterDisplacements: a FREE-length eigenvector scatters to global
atom order via freeAtomIdx (frozen atoms -> [0,0,0], so they never move); a GLOBAL-length
vector passes through unchanged; out-of-range free indices are dropped.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "molbuilder/web/static/lib/vibrationview/mode-math.js"


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


def test_free_length_eigenvector_scatters_to_global_frozen_zero():
    # 4 atoms; atoms 1 and 2 are FREE (0 and 3 frozen).  A 2-row free eigenvector must land on
    # global atoms 1 and 2; frozen atoms 0 and 3 stay [0,0,0] (anchored, no motion).
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        const disp = [[1,0,0],[0,2,0]];   // free rows: atom1, atom2
        console.log(JSON.stringify(s(disp, [1,2], 4)));
    """)
    assert out == [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 0]]


def test_global_length_vector_passes_through():
    # displacements.length == natoms -> already global order, used directly.
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        console.log(JSON.stringify(s([[1,1,1],[2,2,2],[3,3,3]], [0,1,2], 3)));
    """)
    assert out == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]


def test_out_of_range_free_index_is_dropped():
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        console.log(JSON.stringify(s([[9,9,9]], [7], 3)));   // free idx 7 >= natoms -> dropped
    """)
    assert out == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_no_free_map_and_non_global_length_yields_all_zero():
    # Can't place free-length rows without a map -> defensive all-zero (no motion) rather than
    # mis-indexed motion.
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        console.log(JSON.stringify(s([[1,1,1]], null, 3)));
    """)
    assert out == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
