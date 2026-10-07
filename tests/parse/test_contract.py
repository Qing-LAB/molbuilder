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


# `test_two_decks_are_none_never_a_guess` retired 2026-10-04 (W56 review,
# ruling 1): nothing searches a folder for its decks -- the caller hands the
# run's own (`runs.declared`).


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


# --------------------------------------------------------------------- #
#  engine_of — WHICH ENGINE RAN                                         #
# --------------------------------------------------------------------- #
#
# `running-a-job.md` § 4.2 owns the rule and the resolution order.  What
# these pin is the ORDER, because the order is the whole design: the
# engine is DECLARED when the deck is generated (the only moment it is
# known for certain) and the file-cluster sniff is a fallback for
# directories molbuilder did not write.  A test that only checked "a
# .fdf means siesta" would pass just as well against the constant this
# replaced -- `decode_run_dir` answered `engine="siesta"` for every
# directory until 2026-09-04, including every PySCF run.

from molbuilder.parse.contract import engine_of      # noqa: E402

# THREE TESTS WERE RETIRED HERE 2026-09-04, the day they were written.
# They asserted what a directory says when it holds two engines' files at
# once -- built by hand, because the product cannot produce one: engines
# never share a run directory (user, and measured: 0 of 113 real run
# directories disagree). To justify them I invented a workflow -- "you set
# a folder up for SIESTA, then change your mind and set the same folder up
# for PySCF" -- which nobody does. `engine_of` still answers "unknown"
# rather than guessing if it ever meets one; that costs two lines and needs
# no test dramatising a user who does not exist.


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 4 tests here named a folder's engine from a PySCF log, progress
# logs or a run script written by hand (`process/testing.md` § 6).


def test_a_bare_py_file_is_not_an_engine_signal(tmp_path):
    """A person's own script can sit beside a run -- and the monitor's
    modules did, as `.py` files, until they travelled in one file
    (2026-09-26).  Any python file would match, so "there is a `.py` here"
    says nothing about which engine ran.
    """
    (tmp_path / "mb_monitor.py").write_text("print(1)\n")
    assert engine_of(tmp_path) == "unknown"


# --------------------------------------------------------------------- #
#  The relaxation record (model/parse.md § 5b.1) -- measured fixture     #
# --------------------------------------------------------------------- #

# Moved to the e2e tier 2026-10-06 (user: "when a test need siesta's output
# why is it not part of a e2e test?"): the relaxation record is read off a
# relaxation run on the road with the real SIESTA,
# `tests/test_siesta_relax_run_e2e.py`; the force-constant output cut to its
# first steps that showed a run with no record was retired with it
# (`process/testing.md` § 6).


