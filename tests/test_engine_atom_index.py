"""Single-point engine atom-index translation (engine_atom_index.py) +
the display↔engine consistency binding (model/overview.md § 2)."""
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


def test_dispatch_matches_named_forward_functions():
    """to_engine_index(i, engine) must equal the per-engine FACT function, so
    the parametrized surface and the named surface cannot drift."""
    for i in (0, 4, 41):
        assert eai.to_engine_index(i, "siesta") == eai.siesta_atom_index(i)
        assert eai.to_engine_index(i, "geometric") == eai.geometric_atom_index(i)
        assert eai.to_engine_index(i, "pyscf") == eai.pyscf_atom_index(i)


def test_from_engine_index_is_the_inverse():
    """The inverse (engine number -> internal) — the return leg of the
    round-trip — and full round-trip identity for every engine."""
    assert eai.from_engine_index(5, "siesta") == 4       # SIESTA 1-based
    assert eai.from_engine_index(5, "geometric") == 4
    assert eai.from_engine_index(4, "pyscf") == 4         # PySCF 0-based identity
    for engine in ("siesta", "geometric", "pyscf"):
        for i in (0, 4, 41):
            assert eai.from_engine_index(eai.to_engine_index(i, engine), engine) == i


def test_unknown_engine_raises():
    import pytest
    for fn in (eai.to_engine_index, eai.from_engine_index):
        with pytest.raises(ValueError):
            fn(0, "gaussian")


def test_default_engine_is_siesta():
    for i in (0, 4, 41):
        assert eai.to_engine_index(i) == eai.siesta_atom_index(i)
        assert eai.from_engine_index(eai.siesta_atom_index(i)) == i


_JS = (Path(__file__).resolve().parent.parent
       / "molbuilder/web/static/lib/molview/_atom-index.js")


