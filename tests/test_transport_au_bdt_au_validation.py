"""Au-BDT-Au validation fixture for the Transport B.3 engine.

The canonical "fruit fly" of molecular electronics: benzene-1,4-
dithiolate between gold electrodes.  Fixture provenance + bond-
length sources are documented in
``tests/data/au_bdt_au.README.md``.

These tests pin:

1. The committed XYZ + sidecar load cleanly into a Structure
   with the right region labels.
2. ``validate(struct, SiestaConfig(), calculation="transport")`` — the gate
   a real prep runs — returns NO errors on this labeled, correctly-ordered
   structure.
3. The deck the live path renders carries every required keyword and
   ``TS.NumUsedAtomsLeft / Right`` counts DERIVED from the structure's own
   regions (checked on an asymmetric junction, so a hardcoded count fails).

*(2 and 3 drove ``TransiestaEngine.preflight`` and ``.render_script`` until
2026-09-17; both are deleted — the first dispatched for nothing, the second
was a second writer of a deck the framework already writes.)*
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
    L-electrode -> bridge -> R-electrode order -- GEOMETRIC order:
    the L block is the LOWER one and the list ascends along z
    (`engines/transport.md` § 5's label-convention box -- warned, never
    enforced -- the one convention:
    L-electrode is the LOW-z lead;
    mirrored 2026-08-29 when the preflight learned to check geometry,
    which exposed the old fixture as L-on-top-listed-first).

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


def _live_device_deck(label="au_bdt_au_test"):
    """The device deck through the LIVE path -- the same call `prep` makes.

    These three checks rendered through `TransiestaEngine.render_script` until
    2026-09-17.  That was a SECOND writer of this deck and is deleted; the
    checks are about the deck a person gets, so they follow the framework.
    """
    from molbuilder import script_emit as _sc
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import spec_for
    struct = _struct_with_sidecar()
    cfg = SiestaConfig(system_label=label)
    spec = spec_for(struct, cfg, stage_token="device", calculation="transport")
    return _sc.render_deck(spec, struct, cfg, verbose=cfg.verbose_comments)


def test_the_canonical_junction_validates_clean():
    """A correctly-ordered, properly-labeled Au-BDT-Au junction raises no
    error from the gate a real prep runs.

    Asked THROUGH THE LIVE DOOR: `validate(struct, SiestaConfig(),
    calculation="transport")` is what `prepare_deck` calls for every rung,
    so this pins that the canonical system does not trip the SIESTA
    validator, the shared checks, or `_validate_transport_kind`.

    Au has no entry in the open-shell-metals set (5d¹⁰ closed), so the
    shared open-shell-metal check also returns clean — confirming the
    cross-engine rule in `science/chemistry-correctness.md` § 2 does not
    false-positive on closed-shell transition metals.

    *(This drove `TransiestaEngine.preflight` and a bare `TransportConfig()`
    until 2026-09-17.  That class is deleted — it was registered under
    `TransportConfig` and every rung resolves a `SiestaConfig`, so it
    dispatched for nothing, and the one surface that did validate a
    `TransportConfig` was `/api/transport/render`, deleted the same day.
    Asking the dead checker told us nothing about what a prep sees.)*
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    issues = validate(_struct_with_sidecar(), SiestaConfig(),
                      calculation="transport")
    errs = [i for i in issues if i.severity == "error"]
    assert not errs, (
        f"clean Au-BDT-Au junction triggered validation errors: "
        f"{[(e.where, e.message[:70]) for e in errs]}"
    )


def test_each_electrode_block_declares_its_REGION_SIZE():
    """Each ``%block TS.Elec.<name>`` declares ``used-atoms`` equal to that
    electrode region's real size.

    **ASYMMETRIC ON PURPOSE (2 and 4), and that is the whole test.** The
    Au-BDT-Au fixture has three atoms in each lead, so a hardcoded ``3`` in
    the emitter satisfies any check written against it -- measured: mutating
    the emitter to a literal 3 left this test green when it used the fixture.
    A junction whose leads differ in size is the only shape that can tell a
    derived count from a constant.

    Its predecessor asserted the literal string ``"used-atoms         3"``
    twice, which pinned the emitter's column spacing as well and still could
    not have caught the mutation.
    """
    import numpy as np

    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.structure import Structure
    from molbuilder.parse.fdf import _parse_fdf
    from molbuilder import script_emit as _sc
    from molbuilder.siesta.input import spec_for

    # [L x2][bridge x2][R x4] along z, contiguous and ordered as the engine
    # reads them (lower lead first -- the -A3 end).
    n_l, n_b, n_r = 2, 2, 4
    n = n_l + n_b + n_r
    struct = Structure(
        elements=["Au"] * n,
        positions=np.array([[0.0, 0.0, 2.0 * i] for i in range(n)]),
        regions={"L-electrode": list(range(n_l)),
                 "bridge":      list(range(n_l, n_l + n_b)),
                 "R-electrode": list(range(n_l + n_b, n))},
    )
    cfg = SiestaConfig(system_label="asym")
    spec = spec_for(struct, cfg, stage_token="device", calculation="transport")
    deck = _sc.render_deck(spec, struct, cfg, verbose=cfg.verbose_comments)

    _scalars, blocks = _parse_fdf(deck)
    elec = {k: rows for k, rows in blocks.items()
            if k.startswith("tselec") and k != "tselecs"}
    assert len(elec) == 2, f"expected two electrode blocks, got {sorted(elec)}"

    got = sorted(int(dict((r[0].lower(), r[1]) for r in rows)["used-atoms"])
                 for rows in elec.values())
    assert got == sorted([n_l, n_r]), (
        f"electrode blocks declare used-atoms {got}; the structure's leads "
        f"hold {sorted([n_l, n_r])} atoms -- a count that does not track the "
        f"regions is hardcoded")


def test_the_device_deck_states_its_identity_and_method():
    """SystemLabel is the config's label and the solver is the NEGF one.

    Read back through the fdf parser rather than matched as substrings, so
    the emitter's column spacing is not part of the contract.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.fdf import _parse_fdf
    cfg = SiestaConfig(system_label="au_bdt_au_test")
    scalars, _blocks = _parse_fdf(_live_device_deck(cfg.system_label))
    assert scalars.get("systemlabel") == [cfg.system_label]
    # `SolutionMethod transiesta`, NOT `TS.SolutionMethod` -- 5.4.2 rejects
    # the latter with "Unrecognized TranSiesta solution method" (measured).
    assert scalars.get("solutionmethod") == ["transiesta"]
    assert "tssolutionmethod" not in scalars


def test_the_transmission_window_carries_the_configured_point_count():
    """TBtrans' contour window is a BLOCK, and its ``points`` row must be the
    value the config asks for -- compared against the config, not a literal."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.fdf import _parse_fdf
    cfg = SiestaConfig(system_label="au_bdt_au_test")
    _scalars, blocks = _parse_fdf(_live_device_deck(cfg.system_label))
    window = blocks.get("tbtcontourwindow")
    assert window, "no %block TBT.Contour.window in the device deck"
    rows = dict((r[0].lower(), r[1:]) for r in window)
    assert rows.get("points") == [str(cfg.transmission_n_points)]
    assert rows.get("part") == ["line"], (
        "tbtrans refuses anything but a line part for this contour")


def test_species_block_carries_every_element_with_its_true_Z():
    """``ChemicalSpeciesLabel`` must list every element in the structure with
    the atomic number the chemistry layer gives it.

    Both sides derived: the elements come from the fixture, the Z from
    ``chemistry.atomic_number``. This asserted four hand-written
    ``(Z, symbol)`` pairs until 2026-09-17 -- which would keep passing for
    this fixture while being wrong for any other structure.
    """
    from molbuilder.chemistry import atomic_number
    from molbuilder.parse.fdf import _parse_fdf
    struct = _struct_with_sidecar()
    _scalars, blocks = _parse_fdf(_live_device_deck())
    rows = blocks.get("chemicalspecieslabel")
    assert rows, "no %block ChemicalSpeciesLabel in the device deck"
    got = {sym: int(z) for _idx, z, sym in rows}
    expected = {el: atomic_number(el) for el in set(struct.elements)}
    assert got == expected


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
