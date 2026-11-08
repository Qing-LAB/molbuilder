"""``parse.contract.contract_of`` — the recorded-contract extractor.

`model/parse.md` § 5b: the run's own deck, handed by its caller, defines the
answer; a deck that states nothing, or none, is `None`, never a guess.  The deck
is one `jobset prep` wrote."""
from __future__ import annotations

from molbuilder.parse.contract import contract_of


def _prepared_deck(tmp_path, monkeypatch, **template):
    """The deck prep wrote for the road's H2, ``template``'s values the
    person's."""
    from support.road import (described, jobset, road_tree, set_by_the_person,
                              stage_file)
    bundle = described(road_tree(tmp_path, monkeypatch), stage_strategy="")
    if template:
        set_by_the_person(bundle, template)
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    return stage_file(bundle, "coarse", ".fdf")


def test_one_siesta_deck_answers_the_contract(tmp_path, monkeypatch):
    deck = _prepared_deck(tmp_path, monkeypatch, basis_size="SZ",
                          mesh_cutoff=250.0, xc_authors="PBEsol",
                          kgrid=(4, 4, 2), electronic_temperature=200.0)
    out = contract_of(deck)
    assert out["engine"] == "siesta"
    assert out["source"] == deck.name
    assert len(out["source_sha256"]) == 64
    c = out["contract"]
    assert c["basis_size"] == "SZ"
    assert c["mesh_cutoff"] == 250.0
    assert c["xc_authors"] == "PBEsol"
    assert c["kgrid"] == [4, 4, 2]
    assert c["electronic_temperature"] == 200.0


def test_no_deck_or_one_stating_nothing_is_none(tmp_path):
    """The one example of *nothing to read*: no answer, never a guess."""
    assert contract_of(None) is None
    assert contract_of(tmp_path / "absent.fdf") is None
    (tmp_path / "empty.fdf").write_text("SystemLabel x\n")
    assert contract_of(tmp_path / "empty.fdf") is None


def test_a_deck_prep_writes_answers_everything_a_citation_takes(
        tmp_path, monkeypatch):
    """The reader's half of `test_what_a_citation_takes_is_read_back`
    (`tests/test_template_citation_marker.py`): the three lists name what a
    citation reads back, and this is the reader.  A name in a list that the
    reader does not give is the citation silently short, the same as a name in
    no list: the setting takes its default, and every run still finishes.

    And every key the recorded block carries is the calculation's own name for
    that setting (`model/parse.md` § 5b; user, 2026-10-02: "i'd rather unify
    the names, rather than having a drift from contract") -- so each reader
    takes it as it stands, and a key the calculation does not have is drift
    at the source: the citation fill and the relaxation check would carry a
    name nothing reads."""
    import dataclasses

    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.contract import (K_MESH_RECORD_KEYS, RECORDED_FIELDS,
                                           SCF_RECORD_KEYS)
    answered = set(contract_of(_prepared_deck(tmp_path,
                                              monkeypatch))["contract"])
    assert {*RECORDED_FIELDS, *SCF_RECORD_KEYS, *K_MESH_RECORD_KEYS} \
        - answered == set()
    stray = answered - {f.name for f in dataclasses.fields(SiestaConfig)}
    assert not stray, (
        f"contract_of records {sorted(stray)}, which are not the settings' "
        f"own names")
