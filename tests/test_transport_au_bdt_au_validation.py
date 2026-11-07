"""Au-BDT-Au validation fixture for the Transport B.3 engine.

The canonical "fruit fly" of molecular electronics: benzene-1,4-
dithiolate between gold electrodes.  Fixture provenance + bond-
length sources are documented in
``tests/data/au_bdt_au.README.md``.

These tests pin:

1. The committed XYZ + sidecar load cleanly into a Structure
   with the right region labels.
2. The committed geometry CAN be regenerated from textbook bond
   lengths (Bilic-Reimers 2002 S-Au=2.38, etc.), so a future
   bond-length update + fixture refresh is auditable.

What prep writes for this junction, relaxed for real, is
``test_transport_decks_e2e.py``'s.

What's deliberately NOT tested:

* The actual TranSIESTA T(E_F) value.  That would require:
  a SIESTA-MPI run, geometry relaxation first, separate
  electrode .TSHS generation.  When all three are available the
  user can run the emitted .fdf and compare against Reed 2006 /
  Stokbro 2003 (G(E_F) ~ 0.01 G_0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- #
#  Bond lengths from the literature                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BondLengths:
    """Textbook bond lengths used to construct the fixture geometry.

    Cited in tests/data/au_bdt_au.README.md.  Updating any of these
    values requires regenerating the XYZ fixture (the regeneration
    test below catches the mismatch automatically).
    """
    cc_aromatic: float  = 1.40    # Å, textbook benzene
    ch_aromatic: float  = 1.09    # Å, textbook
    cs_aromatic: float  = 1.78    # Å, NIST WebBook
    s_au_atop:   float  = 2.38    # Å, Bilic & Reimers 2002
    au_au_fcc:   float  = 2.88    # Å, Au bulk lattice 4.078 / sqrt(2)


BL = BondLengths()


# --------------------------------------------------------------------- #
#  Construct the fixture geometry from first principles                 #
# --------------------------------------------------------------------- #


def _build_au_bdt_au_coords():
    """Return list of (element, (x, y, z)) tuples in the canonical
    L-electrode -> bridge -> R-electrode order -- GEOMETRIC order:
    the L block is the LOWER one and the list ascends along z
    (`engines/transport.md` § 5's label-convention box -- warned, never
    enforced -- the one convention:
    L-electrode is the LOW-z lead).

    The benzene ring sits in the y-z plane (x = 0), with the S-S
    axis along z.  S atoms sit on the z-axis at z = ±(C-C + C-S) =
    ±3.18 Å.  Au atoms form a simple atomic chain along z, atop-
    bonded to S.
    """
    atoms = []

    # ---- L-electrode: 3 Au atoms in a +z chain ----
    # Au_left_1 is atop-bonded to S1 at the canonical S-Au distance.
    s1_z       = BL.cc_aromatic + BL.cs_aromatic            # = 3.18
    au_left_1z = s1_z + BL.s_au_atop                         # = 5.56
    au_left_2z = au_left_1z + BL.au_au_fcc                   # = 8.44
    au_left_3z = au_left_2z + BL.au_au_fcc                   # = 11.32
    atoms.append(("Au", (0.0, 0.0, au_left_3z)))
    atoms.append(("Au", (0.0, 0.0, au_left_2z)))
    atoms.append(("Au", (0.0, 0.0, au_left_1z)))

    # ---- Bridge: S1 + 6 C (benzene) + 4 H + S2 ----
    # S1 atop the C1=C4 axis (= the z-axis).
    atoms.append(("S", (0.0, 0.0, +s1_z)))

    # Six benzene carbons at angles 90°, 30°, -30°, -90°, -150°, +150°
    # about the ring centre (origin).  This places C1, C4 along z
    # (i.e. at angles +90° and -90° respectively) so the S-S axis
    # aligns with the transport direction.
    r = BL.cc_aromatic
    # Carbon positions in the y-z plane.
    c_angles_deg = [90, 30, -30, -90, -150, 150]
    c_positions  = []
    for a_deg in c_angles_deg:
        a = math.radians(a_deg)
        c_positions.append((0.0, r * math.cos(a), r * math.sin(a)))
    # The order produced by c_angles_deg is C1 (top), C2, C3, C4
    # (bottom), C5, C6 -- which is exactly the para-ortho-meta
    # walk around the ring.  Match the README ordering by appending
    # in that order.
    for cp in c_positions:
        atoms.append(("C", cp))

    # Four H atoms at C2, C3, C5, C6 (NOT C1, C4 — those are S-substituted).
    # Each H is placed radially outward from the ring centre.
    h_carbon_indices = [1, 2, 4, 5]   # 0=C1, 1=C2, ..., 5=C6
    for ci in h_carbon_indices:
        cx, cy, cz = c_positions[ci]
        norm = math.hypot(cy, cz)
        ux, uy, uz = 0.0, cy / norm, cz / norm
        h_pos = (cx + ux * BL.ch_aromatic,
                 cy + uy * BL.ch_aromatic,
                 cz + uz * BL.ch_aromatic)
        atoms.append(("H", h_pos))

    # S2 — mirror of S1.
    atoms.append(("S", (0.0, 0.0, -s1_z)))

    # ---- R-electrode: 3 Au atoms in a -z chain ----
    au_right_1z = -au_left_1z
    au_right_2z = -au_left_2z
    au_right_3z = -au_left_3z
    atoms.append(("Au", (0.0, 0.0, au_right_1z)))
    atoms.append(("Au", (0.0, 0.0, au_right_2z)))
    atoms.append(("Au", (0.0, 0.0, au_right_3z)))

    # The mirror: the walk above builds top-down (L first at high z);
    # negating z makes the SAME chemistry ascend -- L at the bottom,
    # every index and label untouched.
    return [(el, (x, y, -z)) for (el, (x, y, z)) in atoms]


# --------------------------------------------------------------------- #
#  Fixture-load tests                                                   #
# --------------------------------------------------------------------- #


_HERE = Path(__file__).parent
_FIX_XYZ      = _HERE / "data" / "au_bdt_au.xyz"
_FIX_SIDECAR  = _HERE / "data" / "au_bdt_au.molstruct.json"


def test_fixture_pair_opens_through_the_read_door():
    """The committed pair opens through the read door with the documented
    composition (3 Au + 12 bridge + 3 Au = 18 atoms) and the L-electrode /
    bridge / R-electrode labels, contiguous as Brandbyge 2002 § III and the
    preflight require.  Its sidecar pins its XYZ (`structure-molstruct.md`
    § 3), and the door compares the pin -- so a stray XYZ edit without
    writing the pair again is refused there rather than carried into a
    deck."""
    from molbuilder.sidecars.molstruct import load
    from molbuilder.workingcopy_structure import StructureCodec
    assert load(_FIX_SIDECAR).get("geometry_hash"), "the fixture pins nothing"
    struct = StructureCodec().load(_FIX_XYZ)
    assert struct.n_atoms == 18
    counts = {e: struct.elements.count(e) for e in set(struct.elements)}
    assert counts == {"Au": 6, "S": 2, "C": 6, "H": 4}
    assert struct.regions == {"L-electrode": [0, 1, 2],
                              "bridge": list(range(3, 15)),
                              "R-electrode": [15, 16, 17]}, struct.regions


# --------------------------------------------------------------------- #
#  Regeneration audit                                                   #
# --------------------------------------------------------------------- #


def test_fixture_regenerated_from_textbook_chemistry():
    """The committed XYZ MUST match coordinates computed from the
    bond lengths in ``BondLengths``.  Pin so a future bond-length
    update without fixture refresh surfaces as a position mismatch.

    Tolerance: 0.001 Å.  Below that we're catching floating-point
    formatting drift, not chemistry.
    """
    from molbuilder.workingcopy_structure import StructureCodec
    committed = StructureCodec().load(_FIX_XYZ)
    rebuilt   = _build_au_bdt_au_coords()
    assert len(committed.elements) == len(rebuilt) == 18

    for i, (sym, pos) in enumerate(rebuilt):
        assert committed.elements[i] == sym, (
            f"atom {i} element drift: committed={committed.elements[i]} "
            f"rebuilt={sym}"
        )
        d = np.linalg.norm(np.asarray(committed.positions[i])
                           - np.asarray(pos))
        assert d < 1e-3, (
            f"atom {i} ({sym}) position drift {d:.4f} Å > 1e-3 — "
            f"committed XYZ no longer matches the textbook bond "
            f"lengths in BondLengths.  Regenerate the fixture or "
            f"update the bond-length constants."
        )


