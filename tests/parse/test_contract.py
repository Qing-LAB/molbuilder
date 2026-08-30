"""``parse.contract.contract_of`` — the recorded-contract extractor
(structure-info-plan.md I5): one deck in the directory defines the
answer; anything else is None, never a guess."""
from __future__ import annotations

from molbuilder.parse.contract import contract_of

_DECK = """SystemLabel Relax
MeshCutoff 250.0 Ry
PAO.BasisSize SZ
PAO.EnergyShift 0.02 Ry
XC.functional GGA
XC.authors PBE
ElectronicTemperature 200.0 K
%block kgrid_Monkhorst_Pack
  4 0 0 0.0
  0 4 0 0.0
  0 0 2 0.0
%endblock kgrid_Monkhorst_Pack
"""


def test_one_siesta_deck_answers_the_contract(tmp_path):
    (tmp_path / "Relax.fdf").write_text(_DECK)
    out = contract_of(tmp_path)
    assert out["engine"] == "siesta"
    assert out["source"] == "Relax.fdf"
    assert len(out["source_sha256"]) == 64
    c = out["contract"]
    assert c["basis_size"] == "SZ"
    assert c["siesta_mesh_cutoff_ry"] == 250.0
    assert c["xc_authors"] == "PBE"
    assert c["k_mesh_transverse"] == [4, 4, 2]
    assert c["electronic_temperature_k"] == 200.0


def test_no_deck_is_none(tmp_path):
    assert contract_of(tmp_path) is None


def test_two_decks_are_none_never_a_guess(tmp_path):
    (tmp_path / "a.fdf").write_text(_DECK)
    (tmp_path / "b.fdf").write_text(_DECK)
    assert contract_of(tmp_path) is None


def test_a_deck_stating_nothing_is_none(tmp_path):
    (tmp_path / "empty.fdf").write_text("SystemLabel x\n")
    assert contract_of(tmp_path) is None


def test_the_contract_keys_are_the_contracted_vocabulary(tmp_path):
    """The recorded block's keys ARE TransportConfig's CONTRACT_FIELDS
    -- one agreed vocabulary (user, 2026-08-29: 'these names should be
    agreed on contracts so you don't drift or hallucinate').  A key
    outside the set is drift at the source: every consumer downstream
    (the sealed fill, the pane, the pair) would carry a name nothing
    reads."""
    from molbuilder.transport.stages import CONTRACT_FIELDS
    (tmp_path / "Relax.fdf").write_text(_DECK)
    out = contract_of(tmp_path)
    stray = set(out["contract"]) - set(CONTRACT_FIELDS)
    assert not stray, (
        f"contract_of emits {sorted(stray)} outside CONTRACT_FIELDS "
        f"{sorted(CONTRACT_FIELDS)} -- the vocabulary is the contract")
