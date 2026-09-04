"""``parse.contract.contract_of`` — the recorded-contract extractor.

`model/parse.md` § 5b: one deck in the directory defines the answer; anything
else is `None`, never a guess.  (Transport asks the same question as a
REFUSAL, `transport.md` § 3.1 -- there you are citing the directory on
purpose; here the caller is enriching a result it already has.)"""
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

_PROV = ("# === molbuilder provenance BEGIN ===\n"
         "#   engine               {e}\n"
         "#   generator-version    git deadbee\n"
         "# === molbuilder provenance END ===\n")


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


def test_provenance_outranks_everything_below_it(tmp_path):
    """The declaration is the SOLE source of truth when it is present."""
    (tmp_path / "j.fdf").write_text("SystemLabel j\n")
    (tmp_path / "j.molwatch.log").write_text("# engine: siesta\n")
    (tmp_path / "j.run.sh").write_text(_PROV.format(e="pyscf"))
    assert engine_of(tmp_path) == "pyscf"


def test_a_transiesta_run_is_engine_siesta(tmp_path):
    """TranSIESTA is the same engine as SIESTA -- a different TASK.

    It emits no deck PROVENANCE (`job-contracts.md` § 3.1's per-engine
    table), which is exactly why the wrapper carries the declaration
    too: `.run.sh` is the one artifact every prepared run has, whatever
    the engine and whatever the task.
    """
    (tmp_path / "j.fdf").write_text(
        "%block TS.Elec.Left\n%endblock TS.Elec.Left\n")
    (tmp_path / "j.run.sh").write_text(_PROV.format(e="siesta"))
    assert engine_of(tmp_path) == "siesta"


def test_a_bare_py_file_is_not_an_engine_signal(tmp_path):
    """`mb_monitor.py` and `config_dir.py` ship beside every flat run.

    The same foot-gun `JobDirParser.can_parse` documents for its own
    claim rule: any python file would match.
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


def test_a_directory_that_contradicts_itself_answers_unknown(tmp_path):
    """Same rule as `contract_of` above (§ 5b): a directory that says two
    things cannot be made to say one by picking.  Engines never share a
    run directory, so this is a real anomaly the caller should see."""
    (tmp_path / "a.run.sh").write_text(_PROV.format(e="siesta"))
    (tmp_path / "b.run.sh").write_text(_PROV.format(e="pyscf"))
    assert engine_of(tmp_path) == "unknown"
