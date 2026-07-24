"""VibrationView mode math -- the eigenvector free-atom -> global scatter (vibrationview.md §2/§5).

Node unit test of the pure scatterDisplacements: a FREE-length eigenvector scatters to global
atom order via freeAtomIdx (frozen atoms -> [0,0,0], so they never move); a GLOBAL-length
vector passes through unchanged; out-of-range free indices are dropped.
"""
from pathlib import Path

from _node_esm import run_node

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "molbuilder/web/static/lib/vibrationview/mode-math.js"


def _run_node(snippet: str) -> object:
    # mode-math is a native ES module; the shared harness imports it (its TEST-SEAM window
    # publish makes globalThis.molbuilder.vibrationview.scatterDisplacements readable).
    return run_node([MOD], snippet)


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
    # No free map + one row per atom -> already global order, used directly.
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        console.log(JSON.stringify(s([[1,1,1],[2,2,2],[3,3,3]], null, 3)));
    """)
    assert out == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]


def test_free_map_is_authoritative_even_when_all_atoms_free():
    # All 3 atoms free but the map is a PERMUTATION [2,0,1]: free row 0 -> global 2, row 1 ->
    # global 0, row 2 -> global 1.  The map must win -- a naive "length==natoms -> use as
    # global" would leave the rows mis-ordered (the original spectra scatter always followed
    # the map).
    out = _run_node("""
        const s = global.molbuilder.vibrationview.scatterDisplacements;
        console.log(JSON.stringify(s([[1,0,0],[0,1,0],[0,0,1]], [2,0,1], 3)));
    """)
    assert out == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


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
