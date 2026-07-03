"""Validates spec.md § 5.1 invariant 3 — the spectra free-atom-row → global-atom
displacement scatter (``spectraInspector._scatterModeDisplacements``).

Eigenvectors are shape (n_free, 3), indexed by FREE-atom row.  The scatter must
place row k on GLOBAL atom free_atom_idxs[k] and leave every non-free (frozen)
atom at zero — applying `eigenvector[i]` to global atom `i` would displace the
wrong atoms whenever a frozen atom exists.
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/spectra/core.js"


def _run_node(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    # spectra/core.js is ~3k lines -- too big for `node -e` (ARG_MAX), so write
    # the combined script to a temp .js file and run that.
    full = "global.window = global;\n" + MODULE.read_text() + "\n" + snippet
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
        tf.write(full)
        tmp = tf.name
    try:
        proc = subprocess.run([node, tmp], capture_output=True, text=True,
                              timeout=15)
    finally:
        os.unlink(tmp)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n"
                    f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_scatter_maps_free_rows_to_global_atoms_frozen_zero():
    # n=3 atoms; atoms 0 and 2 free, atom 1 frozen.  The eigenvector has 2 rows
    # (one per FREE atom), NOT 3.
    out = _run_node("""
        const f = global.molbuilder.spectraInspector._scatterModeDisplacements;
        console.log(JSON.stringify(f([0, 2], [[0.7, 0, 0], [-0.7, 0, 0]], 3)));
    """)
    # free row 0 -> global atom 0; free row 1 -> global atom 2; atom 1 stays 0.
    assert out == [[0.7, 0, 0], [0, 0, 0], [-0.7, 0, 0]]


def test_scatter_all_free_preserves_order():
    out = _run_node("""
        const f = global.molbuilder.spectraInspector._scatterModeDisplacements;
        console.log(JSON.stringify(f([0, 1], [[1, 0, 0], [0, 2, 0]], 2)));
    """)
    assert out == [[1, 0, 0], [0, 2, 0]]


def test_scatter_skips_out_of_range_free_index():
    # The backend now rejects an out-of-range free index at construction, but
    # the frontend must not crash / write out of bounds -- it skips it.
    out = _run_node("""
        const f = global.molbuilder.spectraInspector._scatterModeDisplacements;
        console.log(JSON.stringify(f([0, 5], [[1, 0, 0], [9, 9, 9]], 2)));
    """)
    assert out == [[1, 0, 0], [0, 0, 0]]
