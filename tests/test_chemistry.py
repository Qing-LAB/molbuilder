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
