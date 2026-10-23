"""Single-point conversion of an atom index to each engine's own numbering
(engine_atom_index.py) + the display↔engine consistency binding
(model/overview.md § 2: *convert* changes where counting starts)."""
from molbuilder import engine_atom_index as eai


def test_siesta_and_geometric_are_1based():
    for i in (0, 4, 41):
        assert eai.siesta_numbering(i) == i + 1
        assert eai.geometric_numbering(i) == i + 1


def test_pyscf_mol_atom_is_0based_identity():
    for i in (0, 4, 41):
        assert eai.pyscf_numbering(i) == i


def test_dispatch_matches_named_forward_functions():
    """convert_to_engine_numbering(i, engine) must equal the per-engine FACT
    function, so the parametrized surface and the named surface cannot
    drift."""
    for i in (0, 4, 41):
        assert eai.convert_to_engine_numbering(i, "siesta") == eai.siesta_numbering(i)
        assert eai.convert_to_engine_numbering(i, "geometric") == eai.geometric_numbering(i)
        assert eai.convert_to_engine_numbering(i, "pyscf") == eai.pyscf_numbering(i)


def test_convert_from_engine_numbering_is_the_inverse():
    """The inverse (engine number -> internal) — the return leg of the
    round-trip — and full round-trip identity for every engine."""
    assert eai.convert_from_engine_numbering(5, "siesta") == 4       # SIESTA 1-based
    assert eai.convert_from_engine_numbering(5, "geometric") == 4
    assert eai.convert_from_engine_numbering(4, "pyscf") == 4         # PySCF 0-based identity
    for engine in ("siesta", "geometric", "pyscf"):
        for i in (0, 4, 41):
            assert eai.convert_from_engine_numbering(eai.convert_to_engine_numbering(i, engine), engine) == i


def test_unknown_engine_raises():
    import pytest
    for fn in (eai.convert_to_engine_numbering, eai.convert_from_engine_numbering):
        with pytest.raises(ValueError):
            fn(0, "gaussian")





