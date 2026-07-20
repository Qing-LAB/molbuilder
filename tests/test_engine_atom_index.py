"""Single-point engine atom-index translation (engine_atom_index.py) +
the display↔engine consistency binding (data-vocabulary.md § 3.1)."""
from pathlib import Path

from molbuilder import engine_atom_index as eai
from _node_esm import run_node


def test_siesta_and_geometric_are_1based():
    for i in (0, 4, 41):
        assert eai.siesta_atom_index(i) == i + 1
        assert eai.geometric_atom_index(i) == i + 1


def test_pyscf_mol_atom_is_0based_identity():
    for i in (0, 4, 41):
        assert eai.pyscf_atom_index(i) == i


_JS = (Path(__file__).resolve().parent.parent
       / "molbuilder/web/static/lib/molview/_atom-index.js")


def test_frontend_display_matches_engine_atom_number():
    """The user's mental model must survive the round-trip: the 1-based atom id
    the frontend DISPLAYS must equal the atom number the engine translator
    emits into the files the user reads (SIESTA .fdf / geomeTRIC $freeze).
    Binds the JS toDisplay to the Python engine index so they can't drift.

    Loaded through the shared ES-module harness (tests/_node_esm), reading toDisplay through the
    global the file publishes -- classic IIFE or native ESM alike."""
    for i in (0, 4, 41):
        js = run_node(
            [_JS],
            f"console.log(JSON.stringify(globalThis.molbuilder.atomIndexModel.toDisplay({i})));")
        assert js == eai.siesta_atom_index(i) == eai.geometric_atom_index(i), (
            f"frontend display {js} != engine atom number for internal {i}")
