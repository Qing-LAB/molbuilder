"""Structure inspector: extended-XYZ Lattice= parser (atom-annotations.md § 6.3 B0).

molbuilder writes the cell on the .xyz comment line as Lattice="...9 nums...".
The embed doesn't parse it, so the inspector does + passes opts.lattice, which is
what makes getLattice() work (k-grid can tile) + the cell wireframe show.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "molbuilder/web/static/lib/inspectors/structure.js"


def _run(snippet: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    harness = (
        "global.window = global;\n"
        "global.document = { createElement: () => ({ style:{}, "
        "classList:{add(){},toggle(){},remove(){}}, setAttribute(){}, "
        "appendChild(){}, addEventListener(){} }), addEventListener(){} };\n"
        + STRUCT.read_text() + "\n"
        "const f = global.molbuilder.inspectors.structureInspector._parseExtxyzLattice;\n"
        + snippet
    )
    p = subprocess.run(["node", "--input-type=commonjs", "-e", harness],
                       capture_output=True, text=True, timeout=15)
    if p.returncode != 0:
        pytest.fail(f"node exited {p.returncode}\n{p.stderr}\n{p.stdout}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_parses_extxyz_lattice_to_3x3():
    out = _run(r'''
        const text = ["2",
          'Lattice="10 0 0 0 11 0 0 0 22" Properties=species:S:1:pos:R:3',
          "C 0 0 0", "H 1 0 0"].join("\n");
        console.log(JSON.stringify(f(text)));
    ''')
    assert out == [[10, 0, 0], [0, 11, 0], [0, 0, 22]]


def test_plain_xyz_without_lattice_returns_null():
    out = _run(r'''
        const text = ["2", "just a comment", "C 0 0 0", "H 1 0 0"].join("\n");
        console.log(JSON.stringify(f(text)));
    ''')
    assert out is None


def test_malformed_lattice_returns_null():
    # Only 3 numbers -> not a 3x3 cell -> null (don't hand the embed junk).
    out = _run(r'''
        const text = ["2", 'Lattice="10 0 0"', "C 0 0 0"].join("\n");
        console.log(JSON.stringify(f(text)));
    ''')
    assert out is None
