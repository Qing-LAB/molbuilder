"""Tests for molbuilder.chemistry: phosphate charge + protonation.

We synthesize phosphate diesters of various protonation states (rather
than depending on rdkit / tleap) so the test runs in any environment.
The synthetic geometries use realistic bond lengths -- P-O ~1.5 A,
non-bridging O-O ~2.45 A -- so the proximity-based adjacency works
correctly without spuriously bonding the two non-bridging Os to each
other.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from molbuilder.chemistry import (
    formal_charge_from_phosphates,
    protonate_phosphate_oxygens,
)
from molbuilder.structure import Structure


def _diester(*, op1_h: bool, op2_h: bool) -> Structure:
    """Synthetic R-O-P(O*)(O*)-O-R' with optional Hs on OP1, OP2.

    Geometry is correct sp3 around P (O-P-O ~109 deg).  This avoids the
    pitfall where two non-bridging Os end up close enough to be picked
    up as bonded by the distance-based adjacency code.
    """
    base = [
        ("C", "C5'", -2.5, 0.0, 0.0),
        ("O", "O5'", -1.4, 0.0, 0.0),
        ("P", "P",    0.0, 0.0, 0.0),
        ("O", "OP1",  0.0, 1.5, 0.0),
        ("O", "OP2",  0.0,-0.8, 1.3),
        ("O", "O3'",  1.4, 0.0, 0.0),
        ("C", "C3'",  2.5, 0.0, 0.0),
    ]
    if op1_h:
        base.append(("H", "HOP1", 0.45, 2.36, 0.0))
    if op2_h:
        base.append(("H", "HOP2", 0.30,-1.55, 1.95))
    return Structure(
        elements=[r[0] for r in base],
        positions=np.array([[r[2], r[3], r[4]] for r in base], dtype=float),
        atom_names=[r[1] for r in base],
        residue_ids=[1] * len(base),
        residue_names=["DA"] * len(base),
        chain_ids=["A"] * len(base),
    )


# --------------------------------------------------------------------- #
#  Charge detection                                                     #
# --------------------------------------------------------------------- #


def test_charge_fully_deprotonated():
    s = _diester(op1_h=False, op2_h=False)
    assert formal_charge_from_phosphates(s) == -1


def test_charge_fully_protonated():
    s = _diester(op1_h=True, op2_h=True)
    assert formal_charge_from_phosphates(s) == 0


@pytest.mark.parametrize("op1_h, op2_h", [(True, False), (False, True)])
def test_charge_partially_protonated(op1_h, op2_h):
    """Either OP1-H or OP2-H present (not both) -> formally neutral."""
    s = _diester(op1_h=op1_h, op2_h=op2_h)
    assert formal_charge_from_phosphates(s) == 0


# --------------------------------------------------------------------- #
#  Protonation                                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("op1_h, op2_h", [(True, False), (False, True)])
def test_protonate_idempotent_when_already_neutral(op1_h, op2_h):
    """Already-neutral phosphate must NOT have an extra H tacked on."""
    s = _diester(op1_h=op1_h, op2_h=op2_h)
    n_before = s.n_atoms
    s2, n = protonate_phosphate_oxygens(s)
    assert n == 0
    assert s2.n_atoms == n_before


def test_protonate_adds_one_H_for_charge_minus_one():
    s = _diester(op1_h=False, op2_h=False)
    s2, n = protonate_phosphate_oxygens(s)
    assert n == 1
    assert formal_charge_from_phosphates(s2) == 0
    # Idempotent: a second pass is a no-op
    _, n2 = protonate_phosphate_oxygens(s2)
    assert n2 == 0


def test_protonate_geometry():
    """The new H sits at 0.96 A from its O at 109.47 deg from P-O axis."""
    s = _diester(op1_h=False, op2_h=False)
    s2, _ = protonate_phosphate_oxygens(s)
    p_pos = s2.positions[2]
    # The implicit P=O is OP1 (alphabetically first); H goes on OP2 (idx 4).
    op2_pos = s2.positions[4]
    h_pos   = s2.positions[-1]
    d = float(np.linalg.norm(h_pos - op2_pos))
    assert abs(d - 0.96) < 0.01, f"O-H = {d:.3f} A"
    v_op = p_pos - op2_pos; v_op /= np.linalg.norm(v_op)
    v_oh = h_pos  - op2_pos; v_oh /= np.linalg.norm(v_oh)
    ang = math.degrees(math.acos(float(np.dot(v_op, v_oh))))
    assert abs(ang - 109.47) < 0.5, f"P-O-H angle = {ang:.2f} deg"


def test_terminal_phosphate_dianion():
    """3 non-bridging Os, all bare -> charge -2, protonate adds 2 H."""
    elements = ["C", "O", "P", "O", "O", "O"]
    positions = np.array([
        [-2.5, 0.0, 0.0], [-1.4, 0.0, 0.0], [0.0, 0.0, 0.0],
        [ 0.0, 1.5, 0.0], [ 0.0,-0.8, 1.3], [ 0.0,-0.8,-1.3],
    ])
    s = Structure(elements=elements, positions=positions,
                  atom_names=["C5'", "O5'", "P", "OP1", "OP2", "OP3"])
    assert formal_charge_from_phosphates(s) == -2
    s2, n = protonate_phosphate_oxygens(s)
    assert n == 2
    assert formal_charge_from_phosphates(s2) == 0


def test_no_phosphate_no_op():
    """Peptide-like structure (no P) is unchanged."""
    elements = ["C", "C", "N", "O", "H"]
    positions = np.array([[0,0,0],[1.5,0,0],[2.0,1.0,0],[1.5,-1,0],[0,-1,0]],
                         dtype=float)
    s = Structure(elements=elements, positions=positions,
                  atom_names=["C","C","N","O","H"])
    assert formal_charge_from_phosphates(s) == 0
    _, n = protonate_phosphate_oxygens(s)
    assert n == 0


def test_empty_structure():
    s = Structure(elements=[], positions=np.zeros((0, 3)))
    assert formal_charge_from_phosphates(s) == 0
    _, n = protonate_phosphate_oxygens(s)
    assert n == 0


# --------------------------------------------------------------------- #
#  _drop_overlapping_hydrogens: ghost-H removal                         #
# --------------------------------------------------------------------- #


def test_drop_overlapping_hydrogens_h_on_heavy_drops_only_the_h():
    """The dominant failure mode: an H placed at its parent heavy
    atom's coordinates by RDKit's AddHs(addCoords=True).  The ghost
    H is dropped; the heavy atom stays."""
    from molbuilder.chemistry import _drop_overlapping_hydrogens
    s = Structure(
        elements=["C", "H", "H"],
        positions=np.array([
            [0.0, 0.0, 0.0],     # heavy C
            [0.0, 0.0, 0.0],     # ghost H (zero distance from C)
            [1.10, 0.0, 0.0],    # real C-H bond
        ], dtype=float),
    )
    out = _drop_overlapping_hydrogens(s)
    assert out.elements == ["C", "H"], out.elements
    # The kept H must be the one at 1.10 A, not the ghost.
    assert float(out.positions[1, 0]) == pytest.approx(1.10)


def test_drop_overlapping_hydrogens_h_h_pair_keeps_one():
    """Pre-fix: a symmetric pass marked BOTH H atoms of an H-H ghost
    pair as overlapping (each saw the other within 0.05 A) and dropped
    them both -- silently removing real protons.  The fix tracks
    which atoms are already-marked-dropped so the second H can't
    cause the first to be dropped.  Net effect: one H survives."""
    from molbuilder.chemistry import _drop_overlapping_hydrogens
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.00, 0.0, 0.0],    # parent O
            [0.96, 0.0, 0.0],    # H1
            [0.96, 0.0, 0.0],    # H2 -- ghost copy of H1 (zero distance)
        ], dtype=float),
    )
    out = _drop_overlapping_hydrogens(s)
    # Pre-fix: 0 H survived.  Post-fix: exactly 1 H survives.
    assert out.elements.count("H") == 1, (
        f"H-H ghost pair: expected 1 H to survive, got {out.elements}"
    )
    # The O is never touched (heavy atoms are not candidates).
    assert "O" in out.elements


def test_drop_overlapping_hydrogens_no_overlap_returns_struct_unchanged():
    """Identity case: no ghosts, no removals -- the function should
    return the original Structure object (cheap shortcut)."""
    from molbuilder.chemistry import _drop_overlapping_hydrogens
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.24, 0.927, 0.000],
        ], dtype=float),
    )
    out = _drop_overlapping_hydrogens(s)
    # Same object (the function returns `struct` early when keep.all()).
    assert out is s


# --------------------------------------------------------------------- #
#  Spin/charge parity + open-shell metal detection                      #
#  (2026-05-22 hemeC-dithiol incident -- both helpers added to surface  #
#  the silent-default failures.  Tests pin the contracts shared by      #
#  _validate_pyscf, _validate_siesta, and the spectra preflight.)       #
# --------------------------------------------------------------------- #


class TestTotalElectrons:
    def test_water_neutral(self):
        from molbuilder.chemistry import total_electrons
        s = Structure(elements=["O", "H", "H"],
                      positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))
        assert total_electrons(s, charge=0) == 10        # 8 + 1 + 1

    def test_charge_subtracts(self):
        from molbuilder.chemistry import total_electrons
        s = Structure(elements=["O", "H", "H"],
                      positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))
        assert total_electrons(s, charge=1)  == 9        # OH+ cation
        assert total_electrons(s, charge=-2) == 12       # O²⁻ anion

    def test_unknown_element_raises_keyerror(self):
        from molbuilder.chemistry import total_electrons
        s = Structure(elements=["O", "Xy"],     # Xy not a real element
                      positions=np.array([[0, 0, 0], [1, 0, 0]]))
        with pytest.raises(KeyError, match="unknown element symbol"):
            total_electrons(s, charge=0)


class TestCheckSpinChargeParity:
    def _h2o(self):
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))

    def test_water_neutral_singlet_is_consistent(self):
        from molbuilder.chemistry import check_spin_charge_parity
        # 10 electrons, spin=0 -> parity matches.
        assert check_spin_charge_parity(self._h2o(), 0, 0) is None

    def test_water_neutral_doublet_is_inconsistent(self):
        """10 electrons + spin=1 (one unpaired) is parity-impossible."""
        from molbuilder.chemistry import check_spin_charge_parity
        msg = check_spin_charge_parity(self._h2o(), 0, 1)
        assert msg is not None
        assert "parity" in msg.lower()
        assert "even" in msg.lower()

    def test_water_cation_doublet_is_consistent(self):
        """OH₂⁺ = 9 electrons, spin=1 (radical) -> parity matches."""
        from molbuilder.chemistry import check_spin_charge_parity
        assert check_spin_charge_parity(self._h2o(), 1, 1) is None

    def test_negative_spin_rejected(self):
        from molbuilder.chemistry import check_spin_charge_parity
        msg = check_spin_charge_parity(self._h2o(), 0, -2)
        assert msg is not None
        assert "negative" in msg.lower()

    def test_non_integer_spin_rejected(self):
        """Reject float spin BEFORE the parity arithmetic so we don't
        emit a useless 'change spin to 2.5 / 0.5' suggestion (regression
        from code-review round 2026-05-23)."""
        from molbuilder.chemistry import check_spin_charge_parity
        msg = check_spin_charge_parity(self._h2o(), 0, 1.5)
        assert msg is not None
        assert "non-negative int" in msg

    def test_bool_spin_rejected(self):
        """bool is technically a subclass of int in Python but is
        meaningless for 2S; reject it explicitly."""
        from molbuilder.chemistry import check_spin_charge_parity
        msg = check_spin_charge_parity(self._h2o(), 0, True)
        assert msg is not None
        assert "non-negative int" in msg


class TestDetectOpenShellMetals:
    def test_d10_metals_excluded(self):
        """Zn / Cd / Hg are d¹⁰ closed-shell; the workflow's closed-
        shell SCF works for them and we MUST NOT false-positive the
        open-shell warning."""
        from molbuilder.chemistry import detect_open_shell_metals
        for el in ("Zn", "Cd", "Hg"):
            s = Structure(elements=[el, "Cl", "Cl"],
                          positions=np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]]))
            assert detect_open_shell_metals(s) == []

    def test_main_group_metals_excluded(self):
        from molbuilder.chemistry import detect_open_shell_metals
        for el in ("Na", "Mg", "Ca", "Al"):
            s = Structure(elements=[el], positions=np.array([[0, 0, 0]]))
            assert detect_open_shell_metals(s) == []

    def test_first_row_transition_metals_detected(self):
        from molbuilder.chemistry import detect_open_shell_metals
        for el in ("Fe", "Mn", "Co", "Ni", "Cu", "Cr", "V", "Ti", "Sc"):
            s = Structure(elements=[el], positions=np.array([[0, 0, 0]]))
            assert detect_open_shell_metals(s) == [el]

    def test_pdb_uppercase_normalised(self):
        """PDB writes element symbols uppercased (FE not Fe).
        detect_open_shell_metals must capitalize-match so a PDB-
        loaded Structure isn't silently missed."""
        from molbuilder.chemistry import detect_open_shell_metals
        s = Structure(elements=["FE", "N", "N"],
                      positions=np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]]))
        assert detect_open_shell_metals(s) == ["Fe"]


class TestExplainMetalSpin:
    def test_fe_high_spin_quintet(self):
        from molbuilder.chemistry import explain_metal_spin
        msg = explain_metal_spin("Fe", 4)
        assert msg is not None
        assert "Fe(II)" in msg
        assert "high-spin" in msg

    def test_fe_high_spin_ferric(self):
        from molbuilder.chemistry import explain_metal_spin
        msg = explain_metal_spin("Fe", 5)
        assert msg is not None
        assert "Fe(III)" in msg
        assert "5/2" in msg

    def test_unknown_combo_returns_none(self):
        """No entry for (Fe, 99) -- silent None rather than a
        misleading made-up hint."""
        from molbuilder.chemistry import explain_metal_spin
        assert explain_metal_spin("Fe", 99) is None

    def test_pdb_uppercase_normalised(self):
        """Same normalisation contract as detect_open_shell_metals."""
        from molbuilder.chemistry import explain_metal_spin
        assert explain_metal_spin("FE", 4) is not None


# --------------------------------------------------------------------- #
#  PySCF ECP (pseudopotential) resolution                              #
#  Shared between Build's pyscf/input.py and spectra/pyscf_script.py.   #
#  Tests below pin the cross-engine rule so the two emitters can't     #
#  drift on ECP handling.                                              #
# --------------------------------------------------------------------- #


class TestResolvePyscfEcp:
    def _fe(self):
        return Structure(elements=["Fe", "N", "N"],
                         positions=np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]]))

    def _pt(self):
        return Structure(elements=["Pt", "Cl", "Cl"],
                         positions=np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]]))

    def _organic(self):
        return Structure(elements=["C", "H", "H", "H", "H"],
                         positions=np.array([[0, 0, 0], [1, 0, 0],
                                             [-1, 0, 0], [0, 1, 0], [0, -1, 0]]))

    def test_def2_bundles_own_ecp_returns_none(self):
        from molbuilder.chemistry import resolve_pyscf_ecp
        # def2-* auto-applies Stuttgart ECP for heavy atoms; emitting
        # lanl2dz on top would double-count.  Return None to skip the
        # ecp= kwarg.
        assert resolve_pyscf_ecp(self._fe(), None, "def2-SVP")  is None
        assert resolve_pyscf_ecp(self._pt(), None, "def2-SVP")  is None
        assert resolve_pyscf_ecp(self._pt(), None, "def2-TZVP") is None
        # def2 name spellings: hyphen, underscore, no-separator
        # all need to be matched.
        assert resolve_pyscf_ecp(self._pt(), None, "def2_SVP")  is None
        assert resolve_pyscf_ecp(self._pt(), None, "def2svp")   is None
        assert resolve_pyscf_ecp(self._pt(), None, "DEF2-SVP")  is None   # case-insensitive

    def test_non_def2_heavy_auto_picks_lanl2dz(self):
        from molbuilder.chemistry import resolve_pyscf_ecp
        assert resolve_pyscf_ecp(self._pt(), None, "cc-pVDZ") == "lanl2dz"

    def test_non_def2_light_returns_none(self):
        """Fe (Z=26) is light enough that all-electron cc-pVDZ is
        correct.  The threshold is Z > 36 (post-Kr)."""
        from molbuilder.chemistry import resolve_pyscf_ecp
        assert resolve_pyscf_ecp(self._fe(), None, "cc-pVDZ")     is None
        assert resolve_pyscf_ecp(self._organic(), None, "cc-pVDZ") is None
        assert resolve_pyscf_ecp(self._organic(), None, "6-31G*") is None

    def test_explicit_string_wins(self):
        from molbuilder.chemistry import resolve_pyscf_ecp
        # User-set value bypasses both auto branches.
        assert resolve_pyscf_ecp(self._pt(), "lanl2dz",  "def2-SVP") == "lanl2dz"
        assert resolve_pyscf_ecp(self._pt(), "stuttgart", "cc-pVDZ") == "stuttgart"
        assert resolve_pyscf_ecp(self._organic(), "lanl2dz", "cc-pVDZ") == "lanl2dz"

    def test_explicit_empty_string_disables(self):
        from molbuilder.chemistry import resolve_pyscf_ecp
        # Treating "" / "none" identically prevents the Python-API
        # case ``ecp="none"`` from reaching gto.M(ecp="none") and
        # raising "Unable to parse the input ECP data" at SCF time.
        assert resolve_pyscf_ecp(self._pt(), "",      "cc-pVDZ") is None
        assert resolve_pyscf_ecp(self._pt(), "none",  "cc-pVDZ") is None
        assert resolve_pyscf_ecp(self._pt(), "NONE",  "cc-pVDZ") is None
        assert resolve_pyscf_ecp(self._pt(), "  none ", "cc-pVDZ") is None

    def test_dict_per_element_passes_through(self):
        """Per-element ECP dicts let the user mix-and-match across
        heavy atoms (e.g. lanl2dz on Pt, stuttgart on Mo)."""
        from molbuilder.chemistry import resolve_pyscf_ecp
        spec = {"Pt": "lanl2dz"}
        assert resolve_pyscf_ecp(self._pt(), spec, "cc-pVDZ") == spec
