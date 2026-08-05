"""Au-BDT-Au validation fixture for the Transport B.3 engine.

The canonical "fruit fly" of molecular electronics: benzene-1,4-
dithiolate between gold electrodes.  Fixture provenance + bond-
length sources are documented in
``tests/data/au_bdt_au.README.md``.

These tests pin:

1. The committed XYZ + sidecar load cleanly into a Structure
   with the right region labels.
2. ``TransiestaEngine.preflight`` returns NO errors on this
   labeled, correctly-ordered structure (sanity check for the
   atom-ordering check from BLOCKER fix 033ae1b).
3. ``TransiestaEngine.render_script`` emits a runnable .fdf
   carrying every required keyword + the right
   ``TS.NumUsedAtomsLeft / Right`` counts (3 and 3).
4. The committed geometry CAN be regenerated from textbook bond
   lengths (Bilic-Reimers 2002 S-Au=2.38, etc.), so a future
   bond-length update + fixture refresh is auditable.

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
import pytest


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
    L-electrode -> bridge -> R-electrode order.

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

    return atoms


# --------------------------------------------------------------------- #
#  Fixture-load tests                                                   #
# --------------------------------------------------------------------- #


_HERE = Path(__file__).parent
_FIX_XYZ      = _HERE / "data" / "au_bdt_au.xyz"
_FIX_SIDECAR  = _HERE / "data" / "au_bdt_au.molstruct.json"


def test_xyz_fixture_loads_with_18_atoms():
    """Sanity: the committed XYZ parses + has the documented
    composition (3 Au + 12 bridge + 3 Au = 18 atoms)."""
    from molbuilder.structure import Structure
    struct = Structure.from_xyz(_FIX_XYZ.read_text())
    assert struct.n_atoms == 18
    counts = {e: struct.elements.count(e) for e in set(struct.elements)}
    assert counts == {"Au": 6, "S": 2, "C": 6, "H": 4}


def test_sidecar_carries_correct_region_labels():
    """The .molstruct.json sidecar must contain L-electrode / bridge
    / R-electrode regions with the contiguous indices Brandbyge
    2002 § III + our preflight check require."""
    from molbuilder.sidecars.molstruct import load
    data = load(_FIX_SIDECAR)   # returns a dict
    regions = data["regions"]
    assert regions["L-electrode"] == [0, 1, 2]
    assert regions["bridge"]      == list(range(3, 15))
    assert regions["R-electrode"] == [15, 16, 17]


def test_fixture_sidecar_hash_matches_xyz():
    """The sidecar's structure_hash must match the SHA-256 of the
    XYZ bytes.  Pin so a stray XYZ edit without regenerating the
    sidecar surfaces as a hash mismatch (stale-sidecar guard from
    the molstruct_json contract)."""
    from molbuilder.sidecars.molstruct import load, sha256_of_file
    data = load(_FIX_SIDECAR)
    assert data["structure_hash"] == sha256_of_file(_FIX_XYZ)


# --------------------------------------------------------------------- #
#  Engine integration                                                   #
# --------------------------------------------------------------------- #


def _struct_with_sidecar():
    """Load the XYZ and apply the sidecar regions to the Structure.
    Mirrors what the web blueprint does on /api/transport/render."""
    from molbuilder.structure import Structure
    from molbuilder.sidecars.molstruct import load, apply_to_structure
    struct = Structure.from_xyz(_FIX_XYZ.read_text())
    data = load(_FIX_SIDECAR)
    # ``apply_to_structure`` mutates struct.regions + struct.frozen_atoms
    # in place — same code path the web blueprint takes.
    apply_to_structure(struct, data)
    return struct


def test_preflight_clean_on_au_bdt_au_fixture():
    """The whole point of the BLOCKER fix from 033ae1b: a
    correctly-ordered, properly-labeled Au-BDT-Au junction passes
    preflight with no errors.  Pin so a future preflight refactor
    that over-fires on this canonical system surfaces.

    Au has no entry in the open-shell-metals set (5d¹⁰ closed),
    so the shared open-shell-metal check also returns clean —
    confirming the cross-engine consistency rule from science.md
    § 2.4 doesn't false-positive on closed-shell transition
    metals.
    """
    from molbuilder.config.transport import TransportConfig
    from molbuilder.transport import get_engine
    cfg = TransportConfig(job_name="au_bdt_au_test")
    issues = get_engine("transiesta").preflight(
        _struct_with_sidecar(), cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert not errs, (
        f"clean Au-BDT-Au junction triggered preflight errors: "
        f"{[e.message[:80] for e in errs]}"
    )


def test_render_script_emits_correct_atom_counts():
    """The emitted .fdf must carry ``used-atoms 3`` inside both
    electrode blocks — derived from the sidecar's region sizes,
    NOT from a hardcoded constant.  Pin so a future refactor that
    hardcodes these silently passes for OUR fixture but breaks on
    user structures.

    2026-06-18: modern TranSIESTA syntax (SIESTA 4.1+ / 5.x) uses
    ``%block TS.Elec.<name>`` with a ``used-atoms <N>`` line
    instead of the legacy ``TS.NumUsedAtomsLeft = N`` / Right
    flat keys.  Empirically verified against SIESTA 5.4.2; see
    ``tests/test_transiesta_siesta_smoke_l4.py``.
    """
    from molbuilder.config.transport import TransportConfig
    from molbuilder.transport import get_engine
    cfg = TransportConfig(job_name="au_bdt_au_test")
    script = get_engine("transiesta").render_script(
        _struct_with_sidecar(), cfg)
    # Au-BDT-Au has 3 atoms in each electrode region — both modern
    # ``used-atoms`` lines must say 3.  Pinning both occurrences so a
    # future refactor that hardcodes one but breaks the other surfaces.
    assert script.count("used-atoms         3") == 2, (
        "expected ``used-atoms 3`` on EACH electrode block (2 total) "
        "in the modern TS.Elec.<name> emission; got "
        f"{script.count('used-atoms         3')} occurrences"
    )
    # SystemLabel echoes the job_name.
    assert "SystemLabel            au_bdt_au_test" in script
    # NEGF block emitted (modern syntax — only the top-level
    # SolutionMethod, NOT TS.SolutionMethod which 5.4.2 rejects
    # with "Unrecognized TranSiesta solution method").
    assert "SolutionMethod         transiesta" in script
    # TBtrans block for transmission.
    assert "TS.TBT.NumE            401" in script   # default 401 points


def test_render_script_has_correct_species_block():
    """ChemicalSpeciesLabel must list Au (Z=79), C (Z=6), H (Z=1),
    S (Z=16) — every element present in the fixture."""
    from molbuilder.config.transport import TransportConfig
    from molbuilder.transport import get_engine
    cfg = TransportConfig(job_name="au_bdt_au_test")
    script = get_engine("transiesta").render_script(
        _struct_with_sidecar(), cfg)
    block_start = script.index("%block ChemicalSpeciesLabel")
    block_end   = script.index("%endblock ChemicalSpeciesLabel")
    block = script[block_start:block_end]
    for sym, z in [("Au", 79), ("C", 6), ("H", 1), ("S", 16)]:
        assert f"  {z}  {sym}" in block or f"{z}  {sym}" in block, (
            f"missing ChemicalSpeciesLabel entry for {sym} (Z={z})"
        )


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
    from molbuilder.structure import Structure
    committed = Structure.from_xyz(_FIX_XYZ.read_text())
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


def test_bond_length_inventory_matches_readme_sources():
    """Pin BL values against the README citations so a stealth
    bond-length change carries a corresponding README update.
    A future refactor can update both in lock-step; this test
    surfaces the mismatch if the README and the dataclass drift.
    """
    readme = (_HERE / "data" / "au_bdt_au.README.md").read_text()
    # Bilic-Reimers gets cited specifically; the other rows aren't
    # tied to a single paper so we don't pin them by author.
    assert "2.38" in readme and "Bilic" in readme
    assert "1.40" in readme and "benzene" in readme.lower()
    assert "1.78" in readme
    assert "2.88" in readme
    # And the actual BL values match the README's table.
    assert BL.s_au_atop == 2.38
    assert BL.cc_aromatic == 1.40
    assert BL.cs_aromatic == 1.78
    assert BL.au_au_fcc == 2.88
    assert BL.ch_aromatic == 1.09
