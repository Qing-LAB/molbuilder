"""The warm-file rules loader — `job-contracts.md` § 4.2a's one reader.

Written with the module (U1, 2026-08-13), from the contract: the closed
vocabulary, the [base]+sections hierarchy, the growth-rule refusal, and
— the byte-faith pins — the seed files reproducing EXACTLY the tuples
they will retire, which is what turns U2/U3 into provable re-plumbing.
"""
from __future__ import annotations

import pytest

from molbuilder import warmfiles as W


# --------------------------------------------------------------------- #
#  Byte-faith: the files SAY this vocabulary, spelled here as LITERALS   #
# --------------------------------------------------------------------- #
#
# B-2 (final review, 2026-08-13): these two pinned the loader against
# runwrap's tuples -- which U3 made ALIASES OF THE SAME LOADER, so both
# sides moved together and deleting every inventory-only row from the
# TOML stayed green while the wrapper banner silently shrank.  A
# byte-faith pin needs one side the test's own bytes.

#: siesta/warm-files.toml, every section, in row order (order is
#: load-bearing: Job.warm order, the banner order).
_SIESTA_INVENTORY = (".XV", ".DM", ".LWF", ".ZM", ".Bonds", ".PARTIAL",
                     ".EIG", ".HSX", ".WFSX", ".STRUCT_NEXT_ITER",
                     ".CG", ".TSHS", ".TSDE")

#: pyscf/warm-files.toml, same rule.
_PYSCF_INVENTORY = (".chk", "_optimized.xyz", "_geom_optim.xyz",
                    "_geom_optim.tmp", "_geom.tmp")


def test_siesta_inventory_is_the_declared_vocabulary():
    assert W.inventory("siesta") == _SIESTA_INVENTORY


def test_pyscf_inventory_is_the_declared_vocabulary():
    assert W.inventory("pyscf") == _PYSCF_INVENTORY


def test_runwrap_banner_tuples_are_the_loader_alias():
    """The U3 re-plumbing itself: runwrap's tuples ARE the loader's
    answer.  Meaningful again now that the loader's answer is pinned to
    literals above -- equality to a pinned side is no tautology."""
    from molbuilder.runwrap import _PYSCF_WARM_SUFFIXES, _SIESTA_WARM_SUFFIXES
    assert _SIESTA_WARM_SUFFIXES == _SIESTA_INVENTORY
    assert _PYSCF_WARM_SUFFIXES == _PYSCF_INVENTORY


def test_siesta_optimization_carries_exactly_the_declared_trio():
    """The carry rows are § 2.3.4's three, in its row order, with the
    pair condition on the history — the exact declaration U2 renders."""
    rows = [r for r in W.rules_for("siesta", "optimization") if r.carry]
    assert [(r.suffix, r.requires_same, r.honoured_by) for r in rows] == [
        (".XV", None, "MD.UseSaveXV"),
        (".DM", None, "DM.UseSaveDM"),
        (".CG", "optimizer", "MD.UseSaveCG"),
    ]


def test_transport_has_its_own_vocabulary_and_no_optimizer_history():
    rows = W.rules_for("siesta", "transport")
    suffixes = [r.suffix for r in rows]
    assert ".TSHS" in suffixes and ".TSDE" in suffixes
    assert ".CG" not in suffixes
    # transport carries nothing between stages today: inventory-only
    assert not any(r.carry for r in rows if r.suffix in (".TSHS", ".TSDE"))


def test_an_empty_section_is_base_only():
    """[vibration] with no rows IS a statement — the smallest expansion
    the § 4.2a growth rule allows."""
    assert [r.suffix for r in W.rules_for("pyscf", "vibration")] == [".chk"]


# --------------------------------------------------------------------- #
#  The growth rule's refusal                                             #
# --------------------------------------------------------------------- #

def test_an_unknown_calculation_is_refused_naming_the_sections():
    with pytest.raises(W.WarmFilesError) as e:
        W.rules_for("siesta", "spectroscopy")
    msg = str(e.value)
    assert "spectroscopy" in msg
    assert "optimization" in msg and "transport" in msg
    assert "new section" in msg


def test_base_is_not_a_calculation_type():
    with pytest.raises(W.WarmFilesError):
        W.rules_for("siesta", "base")


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
        W.load_warm_files("x")


def test_a_carry_value_outside_the_vocabulary_is_refused(tmp_path,
                                                         monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[base.file]]\nsuffix = ".A"\ncarry = "always"\n')
    with pytest.raises(W.WarmFilesError, match="when-continuing"):
        W.load_warm_files("x")


def test_one_suffix_lives_in_one_section(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[base.file]]\nsuffix = ".A"\n'
                 '[[opt.file]]\nsuffix = ".A"\n')
    with pytest.raises(W.WarmFilesError, match="one row per file"):
        W.load_warm_files("x")


def test_a_file_without_base_reads_as_truncated(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 _HEAD + '[[opt.file]]\nsuffix = ".A"\n')
    with pytest.raises(W.WarmFilesError, match="base"):
        W.load_warm_files("x")


def test_the_engine_key_must_match_the_package(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 'schema = "molbuilder/warm-files@1"\nengine = "y"\n'
                 '[base]\n')
    with pytest.raises(W.WarmFilesError, match="agree"):
        W.load_warm_files("x")


def test_a_wrong_schema_major_is_refused(tmp_path, monkeypatch):
    _write_rules(tmp_path, monkeypatch,
                 'schema = "molbuilder/warm-files@2"\nengine = "x"\n[base]\n')
    with pytest.raises(ValueError):
        W.load_warm_files("x")


def test_a_missing_file_names_the_expected_home():
    with pytest.raises(W.WarmFilesError, match="warm-files.toml"):
        W.load_warm_files("no_such_engine")


# --------------------------------------------------------------------- #
#  U4 — the honoured_by agreement check (mandatory, § 4.2a)              #
# --------------------------------------------------------------------- #

def _siesta_deck(restart: str) -> str:
    import numpy as np
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_fdf
    from molbuilder.structure import Structure
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    return render_fdf(struct, SiestaConfig(system_label="JOB",
                                           restart=restart))


def test_a_continuing_deck_emits_every_declared_keyword():
    """§ 4.2a: the check that makes a drifting rules file catchable.  A
    rules file whose keywords the emitter stopped gating is § 4's silent
    pair reborn as config drift — a file carried in and never read."""
    import re
    deck = _siesta_deck("continue")
    rules = [r for r in W.rules_for("siesta", "optimization")
             if r.honoured_by]
    assert rules, "the optimization vocabulary declares no keywords?"
    for rule in rules:
        assert re.search(rf"^{re.escape(rule.honoured_by)}\s+\.true\.",
                         deck, re.M), (
            f"{rule.suffix}: the rules file declares {rule.honoured_by!r} "
            f"honours it, and a continuing deck does not set that keyword "
            f"— the carry would be 'present but not honoured'")


def test_a_clean_deck_honours_nothing():
    """The other silent half: a clean deck emitting UseSave* would read
    warm state a stage was told to ignore."""
    import re
    deck = _siesta_deck("clean")
    for rule in W.rules_for("siesta", "optimization"):
        if rule.honoured_by:
            assert not re.search(
                rf"^{re.escape(rule.honoured_by)}\s+\.true\.", deck,
                re.M), (
                f"a clean deck sets {rule.honoured_by} .true. — warm "
                f"state would be read by a stage told to start clean")
