"""Tests for molbuilder.validation.chemistry.

Per docs/protocols/test-strategy.md § 3 (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _peptide_struct, _vacuum_cell




# --------------------------------------------------------------------- #
#  Peptide protonation: warn when neutral build vs charged side chains  #
# --------------------------------------------------------------------- #




def test_peptide_with_asp_glu_warns_on_zero_charge():
    """ARNDCEQGHI has 1 Arg(+1), 1 Asp(-1), 1 Glu(-1), 1 His (skipped),
    others neutral.  Expected pH-7 charge = -1.  cfg.charge = 0
    (default) should warn."""
    s = _peptide_struct(["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE"])
    cfg = PySCFConfig()  # charge default = None -> auto-detect -> 0 for non-nucleic
    issues = validate(s, cfg)
    msgs = [i for i in issues if i.where == "config.charge"]
    assert len(msgs) == 1
    assert "-1" in msgs[0].message
    assert "neutral" in msgs[0].message.lower()



def test_peptide_with_explicit_charge_no_warn():
    """When the user explicitly sets cfg.charge = -1, the validator
    treats them as having opted in and stays silent."""
    s = _peptide_struct(["ALA","ARG","ASP","GLU","GLY"])
    cfg = PySCFConfig(charge=-1)
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.charge"] == []



def test_peptide_neutral_residues_no_warn():
    """A peptide of only neutral side chains (G, A, V, L, I) has
    estimated pH-7 charge = 0.  No warning."""
    s = _peptide_struct(["GLY","ALA","VAL","LEU","ILE"])
    cfg = PySCFConfig()
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.charge"] == []



def test_non_peptide_skips_protonation_check():
    """A nucleic-acid structure (no AA residue names) shouldn't trigger
    the peptide protonation check at all -- expected_pH7_peptide_charge
    returns None for non-peptides."""
    n = 5
    pos = np.column_stack([np.arange(n) * 3.0, np.zeros(n), np.zeros(n)])
    s = Structure(elements=["C"] * n, positions=pos,
                  residue_names=["DA","DT","DG","DC","DA"],
                  residue_ids=list(range(1, n + 1)))
    cfg = PySCFConfig()
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.charge"] == []



class TestSuggestSpinTotal:
    """The chemistry helper that the validator uses.  Pinning it
    directly because the validator's tests are integration-y."""

    def test_iron_recommends_high_spin(self):
        from molbuilder.chemistry import suggest_spin_total
        preferred, alts = suggest_spin_total(["Fe"])
        assert preferred == 4.0   # Fe(II) high-spin / deoxy-heme / bis-thiolate
        # All six registered Fe entries should appear in alternatives.
        values = sorted({v for v, _ in alts})
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_copper_recommends_one(self):
        from molbuilder.chemistry import suggest_spin_total
        preferred, _ = suggest_spin_total(["Cu"])
        assert preferred == 1.0   # Cu(II) d⁹

    def test_no_metals_returns_safe_nonzero(self):
        """Even for "no metals", return non-zero -- a zero starting
        guess re-triggers the propor failure for any future metal."""
        from molbuilder.chemistry import suggest_spin_total
        preferred, alts = suggest_spin_total([])
        assert preferred > 0
        assert alts == []

    def test_multiple_metals_picks_max(self):
        """Mixed Cu+Fe: pick the LARGER per-element starting value
        (Fe's 4.0 wins over Cu's 1.0).  Rationale: SIESTA's propor
        failure is "can't split zero spin"; ramping DOWN from a non-
        zero guess is safe, ramping UP from zero is what abort-ed."""
        from molbuilder.chemistry import suggest_spin_total
        preferred, _ = suggest_spin_total(["Cu", "Fe"])
        assert preferred == 4.0



# --------------------------------------------------------------------- #
#  Phase 1d: validator + analyzer single-source-of-truth invariant      #
# --------------------------------------------------------------------- #


class TestCheckOpenShellMetalUsesAnalyzer:
    """Pins the contract added in Phase 1d: ``check_open_shell_metal``
    reads its conclusions from ``ChemistryAnalysis``, not from a
    separately-imported ``detect_open_shell_metals``.  Single source
    of truth for the chemistry — validator and ``/api/structure/analyze``
    cannot disagree by construction.  See
    ``docs/protocols/scientific-validation.md`` § 5.3.
    """

    def _hemeC_like(self):
        """Minimal Fe-bearing fixture.  Geometry doesn't matter; the
        check fires on element identity."""
        import numpy as np
        from molbuilder.structure import Structure
        return Structure(
            elements      = ["Fe", "N", "N", "N", "N"],
            positions     = np.zeros((5, 3)),
            atom_names    = ["FE", "N1", "N2", "N3", "N4"],
            residue_ids   = [1] * 5,
            residue_names = ["HEM"] * 5,
            chain_ids     = ["A"] * 5,
        )

    def test_validator_reads_metals_from_analyze_structure(self, monkeypatch):
        """Monkey-patch ``analyze_structure`` to return a fake
        ``ChemistryAnalysis`` with ``metals=[]`` even though the real
        chemistry would say ``["Fe"]``.  If the validator reads from
        the analyzer (Phase 1d), the warn DOES NOT fire.  Pre-Phase-1d
        it would have fired because the validator called
        ``detect_open_shell_metals`` directly.
        """
        from molbuilder import chemistry as ch
        from molbuilder.validation import check_open_shell_metal

        def _fake_analysis(struct):
            # Force empty metals — pretend it's pure organic.
            return ch.ChemistryAnalysis(
                n_atoms             = struct.n_atoms,
                elements            = sorted(set(e.capitalize() for e in struct.elements)),
                n_electrons_neutral = 0,
                metals              = [],   # ← the lie
                metal_hints         = [],
                suggested_charge    = 0,
                suggested_spin      = 0,
                suggested_treatment = "closed",
                rationale           = "(stub for the test)",
                warnings            = [],
            )
        monkeypatch.setattr(ch, "analyze_structure", _fake_analysis)
        # Also patch the binding inside molbuilder.validation if the
        # function was imported by name there.
        from molbuilder import validation as val_mod
        if hasattr(val_mod, "analyze_structure"):
            monkeypatch.setattr(val_mod, "analyze_structure", _fake_analysis)

        issues = check_open_shell_metal(
            self._hemeC_like(),
            is_closed_shell=True,
            engine_label="PySCF",
        )
        assert issues == [], (
            "Validator fired the open-shell-metal WARN even though "
            "the analyzer reported metals=[].  Phase 1d contract "
            "broken: validator must read from ChemistryAnalysis, "
            "not call detect_open_shell_metals directly."
        )

    def test_validator_includes_analyzer_rationale_in_message(self):
        """When the warn fires, its message must include the
        analyzer's rationale text — concrete proof that the validator
        is consuming ChemistryAnalysis output, not assembling its own
        rationale inline.
        """
        from molbuilder.validation import check_open_shell_metal
        issues = check_open_shell_metal(
            self._hemeC_like(),
            is_closed_shell=True,
            engine_label="PySCF",
        )
        assert len(issues) == 1
        msg = issues[0].message
        # The analyzer's rationale carries the engine-agnostic phrase
        # "open-shell treatment".  If the validator built its own
        # message inline, this assertion wouldn't hold.
        assert "open-shell treatment" in msg, (
            "Validator message does not include the analyzer rationale "
            "— evidence it isn't reading from ChemistryAnalysis."
        )

    def test_au_bdt_au_closed_shell_does_NOT_warn(self):
        """The 2026-06-13 noble-metal cluster-context fix: an Au-BDT-
        Au junction (4 Au atoms + benzene-1,4-dithiol) with closed-
        shell SCF (the published-literature standard for Au transport
        calculations) MUST NOT fire the open-shell-mismatch warning.

        Pre-fix the validator checked ``analysis.metals`` (non-empty
        → warn) and contradicted the detection chip that ALSO read
        from analyze_structure but correctly displayed "closed-shell
        singlet" — same form, two surfaces, two contradictory verdicts.

        Post-fix the validator reads ``analysis.suggested_treatment``
        which is "closed" for Au_4 + ligand systems; warning silenced."""
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.validation import check_open_shell_metal
        # 4 Au + 2 S + 6 C + 4 H — 16 atoms, even electron count.
        elements = ["Au"]*4 + ["S"]*2 + ["C"]*6 + ["H"]*4
        struct = Structure(
            elements      = elements,
            positions     = np.zeros((len(elements), 3)),
            atom_names    = [f"A{i}" for i in range(len(elements))],
            residue_ids   = [1] * len(elements),
            residue_names = ["JCT"] * len(elements),
            chain_ids     = ["A"] * len(elements),
        )
        issues = check_open_shell_metal(
            struct,
            is_closed_shell=True,
            engine_label="SIESTA",
        )
        assert issues == [], (
            "Validator fires the open-shell-mismatch warning for an "
            "Au-BDT-Au junction with closed-shell SCF.  The analyzer "
            "correctly suggests closed-shell singlet for noble-metal "
            "clusters ≥ 4 atoms (Stoner criterion fails, s-band "
            "delocalises); the validator must respect that decision "
            "rather than checking analysis.metals being non-empty. "
            f"Got: {[i.message[:200] for i in issues]}")

    def test_single_au_atom_still_warns_open_shell(self):
        """Single Au atom (no extended-metallic-bonding context) IS
        open-shell per the atomic ground state 5d¹⁰ 6s¹.  Validator
        MUST still warn if user picks closed-shell for it — the noble-
        metal cluster-context override only applies at cluster size
        ≥ 4 atoms."""
        import numpy as np
        from molbuilder.structure import Structure
        from molbuilder.validation import check_open_shell_metal
        struct = Structure(
            elements      = ["Au"],
            positions     = np.zeros((1, 3)),
            atom_names    = ["AU1"],
            residue_ids   = [1],
            residue_names = ["AU"],
            chain_ids     = ["A"],
        )
        issues = check_open_shell_metal(
            struct,
            is_closed_shell=True,
            engine_label="PySCF",
        )
        assert len(issues) == 1, (
            "Validator should warn for a single Au atom in closed-shell "
            "config (atomic ground state IS open-shell doublet; cluster-"
            "context override needs ≥ 4 atoms).  "
            f"Got {len(issues)} issue(s).")
