"""``parse.contract.contract_of`` — the recorded-contract extractor.

`model/parse.md` § 5b: the run's own deck, handed by its caller, defines the
answer; a deck that states nothing, or none, is `None`, never a guess."""
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
    out = contract_of(tmp_path / "Relax.fdf")
    assert out["engine"] == "siesta"
    assert out["source"] == "Relax.fdf"
    assert len(out["source_sha256"]) == 64
    c = out["contract"]
    assert c["basis_size"] == "SZ"
    assert c["mesh_cutoff"] == 250.0
    assert c["xc_authors"] == "PBE"
    assert c["kgrid"] == [4, 4, 2]
    assert c["electronic_temperature"] == 200.0


def test_no_deck_is_none(tmp_path):
    assert contract_of(None) is None
    assert contract_of(tmp_path / "absent.fdf") is None


def test_a_deck_stating_nothing_is_none(tmp_path):
    (tmp_path / "empty.fdf").write_text("SystemLabel x\n")
    assert contract_of(tmp_path / "empty.fdf") is None


def test_the_record_speaks_the_catalogues_names(tmp_path):
    """Every key the recorded block carries is the calculation's own name for
    that setting (`model/parse.md` § 5b; user, 2026-10-02: "i'd rather unify
    the names, rather than having a drift from contract") -- so each reader
    takes it as it stands, and a key the calculation does not have is drift
    at the source: the citation fill and the relaxation check would carry a
    name nothing reads."""
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    names = {f.name for f in dataclasses.fields(SiestaConfig)}
    (tmp_path / "Relax.fdf").write_text(_DECK)
    out = contract_of(tmp_path / "Relax.fdf")
    stray = set(out["contract"]) - names
    assert not stray, (
        f"contract_of records {sorted(stray)}, which are not the settings' "
        f"own names")
