"""The warm-file rules loader — `job-contracts.md` § 4.2a's one reader.

From the contract: the closed vocabulary, the [base]+sections hierarchy,
the growth-rule refusal, and the byte-faith pins.
"""
from __future__ import annotations

import pytest

from molbuilder import warmfiles as W


# --------------------------------------------------------------------- #
#  Byte-faith: the files SAY this vocabulary, spelled here as LITERALS   #
# --------------------------------------------------------------------- #
#
# B-2 (final review, 2026-08-13): a byte-faith pin needs one side the
# test's own bytes.

#: siesta/warm-files.toml, every section, in row order (order is
#: load-bearing: Job.warm order, the banner order).
_SIESTA_INVENTORY = (".XV", ".DM",
                     ".LWF", ".ZM", ".BONDS", ".PARTIAL",
                     ".EIG", ".HSX", ".WFSX", ".STRUCT_NEXT_ITER",
                     ".CG", ".TSHS", ".TSDE",
                     # the force-constant run's product and its constrained twin
                     ".FC", ".FCC",
                     # an optical call's product (engines/vibration.md 5.11)
                     ".EPSIMG")

#: pyscf/warm-files.toml, same rule.
_PYSCF_INVENTORY = (".chk", "_optimized.xyz")


def test_siesta_inventory_is_the_declared_vocabulary():
    assert W.warm_list("siesta").suffixes == _SIESTA_INVENTORY


def test_pyscf_inventory_is_the_declared_vocabulary():
    assert W.warm_list("pyscf").suffixes == _PYSCF_INVENTORY


def test_siesta_optimization_carries_exactly_what_the_table_declares():
    """The carry rows are § 2.3.4's table, in its row order, with the pair
    condition on the optimiser history: the warm-start inputs, each with the
    keyword that reads it.  SIESTA's history is a run's own and carries
    nowhere (`job-contracts.md` § 2.2)."""
    rows = [r for r in W.warm_list("siesta", "optimization").rules if r.carry]
    assert [(r.suffix, r.requires_same, r.honoured_by) for r in rows] == [
        (".XV", None, "MD.UseSaveXV"),
        (".DM", None, "DM.UseSaveDM"),
        (".CG", "optimizer", "MD.UseSaveCG"),
    ]


def test_transport_has_its_own_vocabulary_and_no_optimizer_history():
    """P5 (archive/2026-09-01-transport-design.md 4.3) split the two TS rows: `.TSDE` is
    the NEGF density -- THE warm state of a continued device run and the
    file the bias chain hands along -- so it carries; `.TSHS` stays
    inventory-only ON PURPOSE (a product of its own deck; the device's
    copies arrive by the structural gather, never by continuation, so a
    carried one could mask a changed deck)."""
    rows = {r.suffix: r for r in W.warm_list("siesta", "transport").rules}
    assert ".TSHS" in rows and ".TSDE" in rows
    assert ".CG" not in rows
    assert rows[".TSDE"].carry == "when-continuing"
    assert rows[".TSDE"].honoured_by is None, (
        "TranSIESTA reads .TSDE by presence -- an honoured_by would put "
        "it into the present-but-not-honoured check and refuse a "
        "legitimate continuation")
    assert rows[".TSHS"].carry is None
    # AND THE ENGINE'S SCRATCH (`engines/transport.md` § 2a.11): the leads'
    # surface Green's functions a device run holds once per voltage, written
    # by the first point there and read by the rest -- never carried, never
    # taken over into a new run; not restart rows, a section-level fact.
    assert W.is_scratch("AuBDT.v0.2.TSGFL") and W.is_scratch("AuBDT.v0.2.TSGFR")
    assert not W.is_scratch("AuBDT.TSDE") and not W.is_scratch("AuBDT.TSHS")
    # ...AND THE FOLDER OF THE RUN'S OWN IT LIVES IN -- the fourth section
    # fact, stated only with `scratch` (job-contracts.md § 4.2a).
    assert W.warm_list("siesta", "transport").scratch_dir == "tsgf"
    assert W.warm_list("siesta", "optimization").scratch_dir is None


def test_an_optical_calls_scratch_lies_beside_it_and_its_spectrum_is_kept():
    """SIESTA's breakdown of an optical call's spectrum by transition,
    `.contrib.EPSIMG`, is the engine's scratch: no copy carries it -- not a
    warm launch's take-over, not the checkpoint, not a download -- and it
    lies beside the call, with no folder of its own.  The spectrum itself,
    `.EPSIMG`, is the call's product, carried with every done call
    (`engines/vibration.md` § 5.11 § 6, § 8; `job-contracts.md` § 4.2a).
    The families are globs over the shipped vocabulary: one read wrong
    drops the product from every saved state, or saves the scratch in
    each, and nothing says so."""
    assert W.is_scratch("CO2.contrib.EPSIMG")
    assert W.never_carried("CO2.contrib.EPSIMG")
    assert not W.is_scratch("CO2.EPSIMG")
    assert not W.never_carried("CO2.EPSIMG")
    assert W.warm_list("siesta", "optical").scratch_dir is None


# --------------------------------------------------------------------- #
#  The growth rule's refusal                                             #
# --------------------------------------------------------------------- #

def test_an_unknown_calculation_is_refused_naming_the_sections():
    with pytest.raises(W.WarmFilesError) as e:
        W.warm_list("siesta", "spectroscopy").rules
    msg = str(e.value)
    assert "spectroscopy" in msg
    assert "optimization" in msg and "transport" in msg
    assert "new section" in msg


def test_base_is_not_a_calculation_type():
    with pytest.raises(W.WarmFilesError):
        W.warm_list("siesta", "base").rules


# --------------------------------------------------------------------- #
#  The closed vocabulary, enforced                                       #
# --------------------------------------------------------------------- #

def _write_rules(tmp_path, monkeypatch, text):
    f = tmp_path / "warm-files.toml"
    f.write_text(text)
    monkeypatch.setattr(W, "_rules_path", lambda engine: f)
    return f


_HEAD = 'schema = "molbuilder/warm-files@1"\nengine = "x"\n'


def test_a_fourth_key_is_refused_as_the_design_signal(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[base.file]]\nsuffix = ".A"\nwhen = "always"\n')
    with pytest.raises(W.WarmFilesError, match="closed"):
        W.warm_list("x")


def test_a_carry_value_outside_the_vocabulary_is_refused(tmp_path,
                                                         monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[base.file]]\nsuffix = ".A"\ncarry = "always"\n')
    with pytest.raises(W.WarmFilesError, match="when-continuing"):
        W.warm_list("x")


def test_one_suffix_lives_in_one_section(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[base.file]]\nsuffix = ".A"\n'
                 '[[opt.file]]\nsuffix = ".A"\n')
    with pytest.raises(W.WarmFilesError, match="one row per file"):
        W.warm_list("x")


def test_a_file_without_base_reads_as_truncated(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[opt.file]]\nsuffix = ".A"\n')
    with pytest.raises(W.WarmFilesError, match="base"):
        W.warm_list("x")


def test_the_engine_key_must_match_the_package(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 'schema = "molbuilder/warm-files@1"\nengine = "y"\n'
                 '[base]\n')
    with pytest.raises(W.WarmFilesError, match="agree"):
        W.warm_list("x")


def test_a_wrong_schema_major_is_refused(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 'schema = "molbuilder/warm-files@2"\nengine = "x"\n[base]\n')
    with pytest.raises(ValueError):
        W.warm_list("x")


def test_a_missing_file_names_the_expected_home():
    with pytest.raises(W.WarmFilesError, match="warm-files.toml"):
        W.warm_list("no_such_engine")


# --------------------------------------------------------------------- #
#  The two files that must name the same keywords                       #
# --------------------------------------------------------------------- #

def test_the_warm_rules_and_the_catalogue_name_the_same_keywords():
    """``honoured_by`` and ``[item.restart].expands`` are two questions with
    one answer.

    They are not duplicates and must not be collapsed: the rules file says
    *which keyword honours THIS FILE* (a per-file mapping, which is what
    ``prep`` needs to decide a carry), and the catalogue says *which keywords
    this PARAMETER writes* (what the deck writer needs to emit the group).
    Different shapes, different consumers.

    What they cannot do is disagree about the SET. `restart` is one field
    expanding into a group; if the rules file honoured a keyword the deck
    never writes, a carried file would sit unread, and if the deck wrote one
    no file claims, a keyword would be set for state nothing carries — both
    halves of `run-identity.md` § 4's silent pair, from a mismatch no test
    would have caught.
    """
    from molbuilder.script_emit import parameter
    declared = set(parameter("restart", "siesta").writes)
    honoured = {r.honoured_by
                for r in W.warm_list("siesta", "optimization").rules
                if r.honoured_by}
    assert declared == honoured, (
        f"the catalogue says `restart` writes {sorted(declared)} and the warm "
        f"rules say {sorted(honoured)} honour a carried file -- one field, "
        f"one group, and these are the two files that spell it")


def test_the_restart_group_object_is_not_a_second_declaration():
    """`SIESTA_RESTART_GROUP.keys` is DERIVED, and this is what says so.

    A literal tuple can drift into a different order from the catalogue's
    `expands` and the rules file's rows.  Pinning identity-with-the-catalogue rather than a hardcoded expectation is
    the point: a test naming the three keywords here would be a fourth copy.
    """
    from molbuilder.config.siesta import SIESTA_RESTART_GROUP
    from molbuilder.script_emit import parameter
    assert tuple(SIESTA_RESTART_GROUP.keys) == parameter("restart",
                                                         "siesta").writes
