"""``parse.contract.contract_of`` — the recorded-contract extractor.

`model/parse.md` § 5b: one deck in the directory defines the answer; anything
else is `None`, never a guess.  (Transport asks the same question as a
REFUSAL, `transport.md` § 3.1 -- there you are citing the directory on
purpose; here the caller is enriching a result it already has.)"""
from __future__ import annotations

from pathlib import Path

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
    """The recorded block's keys are the record's own vocabulary -- the
    one table that maps them to the catalogue's names, and the k-point
    keys it carries apart (user, 2026-08-29: 'these names should be agreed
    on contracts so you don't drift or hallucinate').  A key outside it is
    drift at the source: every consumer downstream (the citation fill, the
    relaxation check) would carry a name nothing reads."""
    from molbuilder.parse.contract import (K_MESH_RECORD_KEYS,
                                           RECORD_TO_SIESTA_FIELD)
    vocabulary = set(RECORD_TO_SIESTA_FIELD) | set(K_MESH_RECORD_KEYS)
    (tmp_path / "Relax.fdf").write_text(_DECK)
    out = contract_of(tmp_path)
    stray = set(out["contract"]) - vocabulary
    assert not stray, (
        f"contract_of emits {sorted(stray)} outside the record's "
        f"vocabulary {sorted(vocabulary)}")


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


def _prov(engine: str) -> str:
    """A PROVENANCE block from the REAL EMITTER, never typed here.

    These cases build directory shapes a real run cannot easily be made
    to have (two decks disagreeing, litter beside a fresh deck), so they
    have to construct the file. What they must NOT do is invent its
    FORMAT: a hand-typed block is a literal, and a literal cannot
    regress. Every test of this mechanism fed the reader a typed string
    until 2026-09-04, which is how the emitters came to have no coverage
    at all -- an adversarial review deleted the declaration from both
    writers and watched 853 tests stay green.

    Going through `emit_provenance` means the day the block's shape
    changes, these fail too, instead of quietly testing a format nothing
    writes any more.
    """
    from molbuilder.script_emit import emit_provenance

    return emit_provenance(generator_version="git deadbee",
                           generated_at="2026-09-04T00:00:00-07:00",
                           engine=engine) + "\n"


def test_a_pyscf_run_directory_says_pyscf(tmp_path):
    """The regression this whole mechanism exists for."""
    (tmp_path / "co2.pyscf.log").write_text("...\n")
    assert engine_of(tmp_path) == "pyscf"


def test_the_molwatch_header_outranks_the_file_cluster(tmp_path):
    """A `.fdf` present would sniff as SIESTA; the log's own header wins.

    This is the rung that carries every run prepared before the
    PROVENANCE key shipped -- both generators have written
    `# engine: <name>` at file-emission time for far longer.
    """
    (tmp_path / "j.fdf").write_text("SystemLabel j\n")
    (tmp_path / "j.molwatch.log").write_text("# engine: pyscf\n# step 0\n")
    assert engine_of(tmp_path) == "pyscf"



def test_the_wrapper_alone_carries_a_transiesta_run(tmp_path):
    """TranSIESTA is the same engine as SIESTA -- a different TASK.

    A transport deck gets no PROVENANCE at all (`jobset/prep.py` writes
    it with a bare `write_text`, bypassing `prepare_deck`), so the
    `.run.sh` is the ONLY artifact declaring the engine. Hence the
    second assertion, and it is the point of this test: with the deck
    removed the answer must still be `siesta`, which is what proves the
    WRAPPER was read.

    Without it this test was vacuous -- an earlier version asserted only
    the first line, which passes against a `.fdf` cluster sniff, against
    a provenance rung that ignores `*.run.sh`, and against the literal
    constant `"siesta"` this whole mechanism replaced. It was caught in
    an adversarial review the day it was written.
    """
    deck = tmp_path / "j.fdf"
    deck.write_text("%block TS.Elec.Left\n%endblock TS.Elec.Left\n")
    (tmp_path / "j.run.sh").write_text(_prov("siesta"))
    assert engine_of(tmp_path) == "siesta"

    deck.unlink()                       # nothing left to sniff
    assert engine_of(tmp_path) == "siesta", (
        "with no deck to sniff, only the wrapper's PROVENANCE can answer "
        "-- if this fails, the provenance rung is not reading *.run.sh "
        "and a transport run has no declaration at all")


def test_a_bare_py_file_is_not_an_engine_signal(tmp_path):
    """A person's own script can sit beside a run -- and the monitor's
    modules did, as `.py` files, until they travelled in one file
    (2026-09-26).  Any python file would match, so "there is a `.py` here"
    says nothing about which engine ran.
    """
    (tmp_path / "mb_monitor.py").write_text("print(1)\n")
    assert engine_of(tmp_path) == "unknown"


def test_molwatch_is_a_format_and_is_refused_as_an_engine(tmp_path):
    """A log with no `# engine:` header parses to `source_format`
    ``"molwatch"``.  That is the FORMAT, and reading it as the engine is
    the substitution § 4.2 forbids -- it is how the wire reported
    ``format: "molwatch"`` for every molbuilder run."""
    (tmp_path / "j.molwatch.log").write_text("# engine: molwatch\n")
    assert engine_of(tmp_path) == "unknown"


# --------------------------------------------------------------------- #
#  The relaxation record (model/parse.md § 5b.1) -- measured fixture     #
# --------------------------------------------------------------------- #

_RELAX = Path(__file__).resolve().parents[1] / "fixtures" / "siesta_relax"


def test_a_finished_siesta_relaxation_answers_its_record():
    """`tests/fixtures/siesta_relax`: H2 with the first atom held, relaxed
    on SIESTA 5.4.2 through the jobset road (Broyden to 0.01 eV/Å, four
    geometry steps; the `.out` says so).  The record reads the run's own
    tolerance, the last reported forces over every atom and over the one
    moved atom, the held set, the verdict, and pins itself to the final
    geometry."""
    from molbuilder.parse.contract import relaxation_of
    from molbuilder.structure import Structure

    rec = relaxation_of(_RELAX / "01_relax" / "run-0")
    assert rec is not None
    assert rec["engine"] == "siesta"
    assert rec["source"].endswith(".out")
    assert rec["n_steps"] == 4
    assert rec["force_tolerance_ev_ang"] == 0.01
    # the last step's `siesta: Atomic forces` block: the held atom carries
    # the constraint force, the moved one is what SIESTA judged
    assert abs(rec["max_force_ev_ang"] - 0.004887) < 2e-6
    assert abs(rec["max_force_free_ev_ang"] - 0.001042) < 2e-6
    assert rec["held_atom_idxs"] == [0]
    assert rec["held_atom_keys"] == ["H 5.000000 5.000000 5.000000"]
    assert rec["converged"] is True
    assert rec["run_state"] == "ended"
    # the fingerprint is the FINAL geometry's -- the coordinates the run
    # printed last, not the input's
    final = Structure(elements=["H", "H"],
                      positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 5.774583]])
    assert rec["geometry_sha256"] == final.geometry_fingerprint()
    # the same atoms listed the other way round -- a deck's held-first copy
    # -- fingerprint the same (§ 5b.1), so the gate can judge that copy
    swapped = Structure(elements=["H", "H"],
                        positions=[[5.0, 5.0, 5.774583], [5.0, 5.0, 5.0]])
    assert swapped.geometry_fingerprint() == rec["geometry_sha256"]


def test_a_run_that_relaxed_nothing_has_no_record():
    """A force-constant run echoes no force tolerance and moves nothing on
    purpose (`tests/fixtures/siesta_fc`): no record -- `None`, never a
    guess.  (A directory with only a deck is pinned by
    `test_structure_info_bridge.py`.)"""
    from molbuilder.parse.contract import relaxation_of
    fc = Path(__file__).resolve().parents[1] / "fixtures" / "siesta_fc"
    assert relaxation_of(fc) is None

