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
from molbuilder.runfiles import RunNames


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
#  Engine integration                                                   #
# --------------------------------------------------------------------- #


def _struct_with_sidecar():
    """The committed pair, its labels applied -- through the one read door."""
    from molbuilder.workingcopy_structure import StructureCodec
    return StructureCodec().load(_FIX_XYZ)


def _live_device_deck(label="au_bdt_au_test", rung="device", point=None,
                      **over):
    """A NEGF rung's deck through the LIVE path -- the same call `prep` makes:
    the device's by default, the transmission's when a check is about what
    tbtrans reads (the two are separate texts, `transport.md` § 6.1b); at
    ``point`` as `prep` renders each point's (`spec_for(..., point=)`, the
    config at its voltage).
    """
    import dataclasses
    from molbuilder import script_emit as _sc
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.transport_rung import at_point
    from molbuilder.siesta.input import spec_for
    from molbuilder.template import role_answers
    struct = _struct_with_sidecar()
    # ...with the rung's own answers laid on, as `resolve` lays them for
    # every prep (`engines/template.md` § 6.4): the device's `transiesta`
    # is the rung's answer, not a default the spec supplies.
    cfg = at_point(dataclasses.replace(
        SiestaConfig(system_label=label),
        **role_answers("siesta", "transport", rung), **over), point)
    token = {"device": "04_device", "transmission": "05_transmission"}[rung]
    spec = spec_for(struct, cfg, calculation="transport",
                    names=RunNames.of(label, token, "hierarchical"),
                    point=point)
    return _sc.render_deck(spec, struct, cfg, verbose=cfg.verbose_comments)


def test_the_canonical_junction_validates_clean():
    """A correctly-ordered, properly-labeled Au-BDT-Au junction raises no
    error from the gate a real prep runs.

    Asked THROUGH THE LIVE DOOR: `validate(struct, SiestaConfig(),
    calculation="transport")` is what `prepare_deck` calls for every rung,
    so this pins that the canonical system does not trip the SIESTA
    validator, the shared checks, or `_validate_transport_kind`.

    And NO SPIN FINDING of any severity: gold is a noble s1 metal, not an
    open-d one, and the junction repeats, so the electronic state resolves
    it restricted (`science/chemistry-correctness.md` § 2a.1b) -- the gold
    lead once told to go open-shell, which is what the README beside the
    fixture says this system shows.
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
    spin = [i for i in issues if i.where in ("config.spin_treatment",
                                             "config.unpaired_electrons")]
    assert not spin, [(i.severity, i.message[:90]) for i in spin]


def test_each_electrode_block_declares_its_REGION_SIZE():
    """Each ``%block TS.Elec.<name>`` declares ``used-atoms`` equal to that
    electrode region's real size.

    **ASYMMETRIC ON PURPOSE (2 and 4), and that is the whole test.** The
    Au-BDT-Au fixture has three atoms in each lead, so a hardcoded ``3`` in
    the emitter satisfies any check written against it -- measured: mutating
    the emitter to a literal 3 left this test green when it used the fixture.
    A junction whose leads differ in size is the only shape that can tell a
    derived count from a constant.
    """
    import numpy as np

    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.structure import Structure
    from molbuilder.parse.fdf import read_fdf
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
    spec = spec_for(struct, cfg, calculation="transport",
                    names=RunNames.of("asym", "04_device", "hierarchical"))
    deck = _sc.render_deck(spec, struct, cfg, verbose=cfg.verbose_comments)

    _scalars, blocks = read_fdf(deck)
    elec = {k: rows for k, rows in blocks.items()
            if k.startswith("tselec") and k != "tselecs"}
    assert len(elec) == 2, f"expected two electrode blocks, got {sorted(elec)}"

    got = sorted(int(dict((r[0].lower(), r[1]) for r in rows)["used-atoms"])
                 for rows in elec.values())
    assert got == sorted([n_l, n_r]), (
        f"electrode blocks declare used-atoms {got}; the structure's leads "
        f"hold {sorted([n_l, n_r])} atoms -- a count that does not track the "
        f"regions is hardcoded")


def _electrode_rows(deck: str, lead: str) -> dict:
    """A lead's ``%block TS.Elec.<lead>`` as ``{line: value}``, through the
    fdf reader."""
    from molbuilder.parse.fdf import read_fdf
    _scalars, blocks = read_fdf(deck)
    return {r[0].lower(): r[1] for r in blocks[f"tselec{lead.lower()}"]}


def test_the_leads_greens_functions_are_the_runs_once_per_voltage():
    """`engines/transport.md` § 2a.11, *What a run's points share*: a device
    point's deck names each lead's surface Green's function in its RUN's
    scratch folder (`tsgf/`, the vocabulary's), the file named on the
    voltage, and asks the engine to reuse it -- so two frames at one
    voltage name the same file, two voltages never do, a point with no
    level names the run's own folder, and the transmission names none
    (tbtrans writes no such file).

    MUTATION THIS MUST FAIL AGAINST: the path left at the point (every
    frame writes its own 49 GiB pair); the voltage dropped from the name
    (the second voltage reads the first's file and the engine dies on the
    chemical potential); the folder dropped (the pair lies loose in the
    run); `GF.ReUse` not written; a `GF` line in the transmission's
    blocks."""
    from pathlib import PurePosixPath

    from molbuilder.transport.stages import Point

    def named(point):
        deck = _live_device_deck(point=point)
        rows = {lead: _electrode_rows(deck, lead) for lead in ("L", "R")}
        assert all(r["gf.reuse"] == "true" for r in rows.values()), rows
        here = PurePosixPath("run-0") / (point.rel if point else "")
        # The path as the engine resolves it from the point's folder.
        return {lead: PurePosixPath(
            *(p for p in (here / r["gf"]).parts if p != "."))
            for lead, r in rows.items()}

    def resolved(p: PurePosixPath) -> str:
        out = []
        for part in p.parts:
            if part == "..":
                out.pop()
            else:
                out.append(part)
        return "/".join(out)

    f0 = {k: resolved(v) for k, v in named(Point(frame=0, volts=0.2, of=3)).items()}
    f1 = {k: resolved(v) for k, v in named(Point(frame=1, volts=0.2, of=3)).items()}
    assert f0 == f1 == {"L": "run-0/tsgf/au_bdt_au_test.v0.2.TSGFL",
                        "R": "run-0/tsgf/au_bdt_au_test.v0.2.TSGFR"}, (f0, f1)
    other = {k: resolved(v) for k, v in named(Point(frame=1, volts=0.0, of=3)).items()}
    assert other == {"L": "run-0/tsgf/au_bdt_au_test.v0.TSGFL",
                     "R": "run-0/tsgf/au_bdt_au_test.v0.TSGFR"}, other
    alone = {k: resolved(v) for k, v in named(None).items()}
    assert alone == {"L": "run-0/tsgf/au_bdt_au_test.v0.TSGFL",
                     "R": "run-0/tsgf/au_bdt_au_test.v0.TSGFR"}, alone
    for lead in ("L", "R"):
        rows = _electrode_rows(_live_device_deck(rung="transmission"), lead)
        assert "gf" not in rows and "gf.reuse" not in rows, rows


def test_the_device_deck_writes_the_bias_windows_line_and_the_file_switch():
    """`engines/transport.md` § 2a.11, *The levers*: the device deck writes
    the bias window's line at every point -- `TS.Contours.nEq` naming
    `line`, and `TS.Contour.nEq.line` with the window the cutoff item gives
    and the spacing the card sets -- and the file switch,
    `TS.Elecs.Out-of-core`, TranSIESTA's own defaults written rather than
    left to it.  A spacing or a switch set on the card reaches the deck.

    MUTATION THIS MUST FAIL AGAINST: the block not written (the engine's
    silent 0.01 eV); the window typed with a number of its own rather than
    the cutoff item's; the switch left to the engine."""
    from molbuilder.parse.fdf import fdf_key, fdf_true, read_fdf
    from molbuilder.transport.stages import Point

    def read(**over):
        scalars, blocks = read_fdf(_live_device_deck(
            point=Point(frame=None, volts=0.2), **over))
        return scalars, blocks

    scalars, blocks = read()
    assert blocks[fdf_key("TS.Contours.nEq")] == [["line"]], blocks
    rows = {r[0].lower(): r[1:] for r in blocks[fdf_key("TS.Contour.nEq.line")]}
    assert rows["part"] == ["line"] and rows["method"] == ["mid-rule"], rows
    assert rows["delta"] == ["0.01", "eV"], rows
    assert " ".join(rows["from"]) == "- 5 kT - |V|/2 to |V|/2 + 5 kT", rows
    assert fdf_true(scalars[fdf_key("TS.Elecs.Out-of-core")]), scalars

    scalars, blocks = read(negf_neq_spacing_ev=0.02,
                           negf_neq_fermi_cutoff_kt=7.5,
                           ts_elecs_out_of_core=False)
    rows = {r[0].lower(): r[1:] for r in blocks[fdf_key("TS.Contour.nEq.line")]}
    assert rows["delta"] == ["0.02", "eV"], rows
    assert " ".join(rows["from"]) == "- 7.5 kT - |V|/2 to |V|/2 + 7.5 kT", rows
    assert not fdf_true(scalars[fdf_key("TS.Elecs.Out-of-core")]), scalars


def test_the_file_estimate_is_the_engines_own():
    """`engines/transport.md` § 2a.11, *What it costs*: the size a lead's
    surface Green's function file will have, from the device deck as prep
    writes it and the lead's orbital count, is what TranSIESTA itself
    prints at the run's start -- two runs of ours, read off their outputs:
    the Au–BDT–Au junction at 0 V (1425 orbitals, 5×5×1, 123 poles at 300 K:
    *49.170 GB*) and the test tree's chain at 0.2 V (208 orbitals, its lead
    run's own count, one k-point, 123 + 123 poles and a 46-point line:
    *194.092 MB*; SIESTA's units are 1024-based, and its print adds a few
    KiB of header to the matrices).

    MUTATION THIS MUST FAIL AGAINST: the k-points counted as the grid's
    (25, not 13); the +2 dropped; the second reservoir's poles or the line
    left out under bias; the window's kT taken at another temperature."""
    from molbuilder.transport.gf_files import estimate_from
    from molbuilder.transport.stages import Point
    gib, mib = 1024 ** 3, 1024 ** 2
    at_0v = _live_device_deck(point=None, kgrid=(5, 5, 1),
                              negf_eq_pole_ev=10.0, electronic_temperature=300.0)
    assert abs(estimate_from(at_0v, 1425) / gib - 49.170) < 0.001
    at_bias = _live_device_deck(point=Point(frame=None, volts=0.2),
                                kgrid=(1, 1, 1), negf_eq_pole_ev=10.0,
                                electronic_temperature=300.0)
    # SIESTA's print adds its header -- the cell, the coordinates, the
    # k-points, the energies -- a few KiB beside the matrices: 6 KiB here.
    assert abs(estimate_from(at_bias, 208) / mib - 194.092) < 0.01
    # THE GRID AS TRANSIESTA PLACES IT (`find_kgrid.F`): an even grid holds
    # Gamma and the zone boundary unshifted -- 4×4×1 is 10 points, 8 with
    # the half-step shift.
    from molbuilder.transport.gf_files import k_points_time_reversed
    assert k_points_time_reversed([(4, 0.0), (4, 0.0), (1, 0.0)]) == 10
    assert k_points_time_reversed([(4, 0.5), (4, 0.5), (1, 0.0)]) == 8


def test_prep_refuses_a_spacing_that_leaves_the_bias_window_under_two_points():
    """`engines/transport.md` § 2a.11, *The levers*: a device point under
    bias whose spacing leaves its window fewer than two points is refused
    at prep's settings gate, naming the spacing; 0 V has no window and no
    check.  The floor is ours -- TranSIESTA stops at none.

    MUTATION THIS MUST FAIL AGAINST: the gate dropped; the window counted
    without the cutoff's kT; 0 V checked."""
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset.transport_rung import at_point
    from molbuilder.template import role_answers
    from molbuilder.transport.stages import Point
    from molbuilder.validation import validate
    struct = _struct_with_sidecar()

    def findings(volts, **over):
        cfg = at_point(dataclasses.replace(
            SiestaConfig(system_label="au_bdt_au_test"),
            **role_answers("siesta", "transport", "device"), **over),
            Point(frame=None, volts=volts))
        return [i for i in validate(struct, cfg, calculation="transport",
                                    rung="device")
                if i.where == "config.negf_neq_spacing_ev"]

    # 0.001 V at 10 K: a 0.0096 eV window -- 0.1 eV leaves it no point.
    refused = findings(0.001, negf_neq_spacing_ev=0.1,
                       electronic_temperature=10.0)
    assert [i.severity for i in refused] == ["error"], refused
    assert "0 point(s)" in refused[0].message and "two or more" in refused[0].message
    assert findings(0.001, negf_neq_spacing_ev=0.001,
                    electronic_temperature=10.0) == []
    assert findings(0.0, negf_neq_spacing_ev=0.1,
                    electronic_temperature=10.0) == []


def test_the_device_deck_states_its_identity_and_method():
    """SystemLabel is the config's label, the solver is the NEGF one, and
    the NEGF inversion is the config's own -- two keywords, two values.

    Read back through the fdf parser rather than matched as substrings, so
    the emitter's column spacing is not part of the contract.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.fdf import read_fdf
    cfg = SiestaConfig(system_label="au_bdt_au_test")
    scalars, _blocks = read_fdf(_live_device_deck(cfg.system_label))
    assert scalars.get("systemlabel") == [cfg.system_label]
    # `SolutionMethod` switches the SCF to NEGF; `TS.SolutionMethod` is how
    # the Green's function is inverted -- 5.4.2 stops on `transiesta` there
    # ("Unrecognized TranSiesta solution method", measured).
    assert scalars.get("solutionmethod") == ["transiesta"]
    assert scalars.get("tssolutionmethod") == [cfg.ts_solution_method]


def test_the_transmission_window_carries_the_configured_point_count():
    """TBtrans' contour window is a BLOCK, and its ``points`` row must be the
    value the config asks for -- compared against the config, not a literal."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.fdf import read_fdf
    cfg = SiestaConfig(system_label="au_bdt_au_test")
    _scalars, blocks = read_fdf(_live_device_deck(cfg.system_label,
                                                    rung="transmission"))
    window = blocks.get("tbtcontourwindow")
    assert window, "no %block TBT.Contour.window in the transmission deck"
    rows = dict((r[0].lower(), r[1:]) for r in window)
    assert rows.get("points") == [str(cfg.transmission_n_points)]
    assert rows.get("part") == ["line"], (
        "tbtrans refuses anything but a line part for this contour")
    # The bounds are the x-axis of T(E).  Pinned against the CONFIG, not
    # against E_F: whether a `%block TBT.Contour` line's energies are
    # absolute or E_F-relative is an open question (`transport/record.py`),
    # so this says what the deck
    # carries, nothing about what it means.
    from _deck import fdf_energy_window
    lo, hi = fdf_energy_window(_live_device_deck(cfg.system_label,
                                                 rung="transmission"),
                               "TBT.Contour.window")
    assert lo == pytest.approx(cfg.transmission_emin_ev)
    assert hi == pytest.approx(cfg.transmission_emax_ev)


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


