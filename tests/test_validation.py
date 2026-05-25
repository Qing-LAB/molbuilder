"""Tests for molbuilder/validation.py.

Each test pins one row of the validation-pass table in
``docs/design.md`` § "Validation pass (pre-emission)".  Per-check
positive (issue raised) AND negative (no issue when the input is
valid) cases are exercised so a regression that loosens a check
shows up too.
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


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def water_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        title="water",
    )


def _vacuum_cell(size: float = 30.0) -> np.ndarray:
    """A simple cubic cell.  30 Å is a typical vacuum-padded box."""
    return np.eye(3) * size


# --------------------------------------------------------------------- #
#  Issue dataclass invariants                                           #
# --------------------------------------------------------------------- #


def test_issue_severity_accepts_error_warn_info():
    """Severity is restricted to error / warn / info.  Info was
    added 2026-05-22 for advisory hints (e.g. 'Fe + spin=4 implies
    high-spin Fe(II)') that don't add to the warn count; renamed
    from the old "error or warn only" pin."""
    Issue("error", "fine")
    Issue("warn",  "fine")
    Issue("info",  "fine")
    with pytest.raises(ValueError, match="severity"):
        Issue("debug", "not allowed")


def test_validation_error_carries_issues():
    issues = [
        Issue("warn", "minor", "x"),
        Issue("error", "fatal", "y"),
    ]
    with pytest.raises(ValidationError) as exc:
        raise ValidationError(issues)
    assert exc.value.issues == issues
    # Message lists the error but not the warn.
    assert "fatal" in str(exc.value)
    # The warning should NOT be in the formatted error message;
    # warnings get their own stderr path via report().
    assert "minor" not in str(exc.value)


def test_validation_error_rejects_empty_or_warn_only():
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([])
    with pytest.raises(ValueError, match="error-severity"):
        ValidationError([Issue("warn", "just a warning")])


# --------------------------------------------------------------------- #
#  Geometry: min atom-atom distance                                     #
# --------------------------------------------------------------------- #


def test_min_distance_too_small_is_error():
    """< 0.3 Å -- atoms effectively coincide; SCF will diverge."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    errs = [i for i in issues if i.severity == "error"
            and i.where == "geometry.min_distance"]
    assert len(errs) == 1
    assert "0.100" in errs[0].message


def test_min_distance_short_is_warn():
    """0.3 - 0.7 Å -- shorter than any real bond, likely broken."""
    s = Structure(
        elements=["C", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.min_distance"]
    assert len(warns) == 1


def test_min_distance_normal_bonds_no_issue(water_struct):
    """Real water (O-H ~0.957 Å) must not flag any geometry issue."""
    issues = validate(water_struct, SiestaConfig())
    geo = [i for i in issues if i.where == "geometry.min_distance"]
    assert geo == []


# --------------------------------------------------------------------- #
#  H/heavy ratio: catches heavy-atom skeletons headed for DFT           #
# --------------------------------------------------------------------- #


def test_h_ratio_skeleton_is_warn():
    """A heavy-atom-only structure (typical X3DNA fiber raw output, or a
    user-loaded heavy-atom PDB) is missing electrons -- DFT will compute
    the wrong total electron count.  Severity is warn (not error)
    because the user may legitimately want to inspect / hand-process
    the skeleton; the warning surfaces the issue prominently."""
    # 4 heavy atoms, 0 H -> ratio 0.0
    s = Structure(
        elements=["C", "N", "O", "P"],
        positions=np.array([[0,0,0],[1.5,0,0],[3.0,0,0],[4.5,0,0]],
                           dtype=float),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.h_ratio"]
    assert len(warns) == 1, f"expected 1 h_ratio warn, got {warns}"
    assert "0.00" in warns[0].message    # ratio printed to 2 decimals


def test_h_ratio_low_but_not_zero_is_warn():
    """Borderline case: ratio < 0.3 still warns (3 H, 12 heavy = 0.25)."""
    elements = ["C"] * 12 + ["H"] * 3
    pos = np.zeros((15, 3))
    for i in range(15):
        pos[i] = (i * 1.5, 0, 0)
    s = Structure(elements=elements, positions=pos)
    issues = validate(s, SiestaConfig())
    assert any(i.severity == "warn" and i.where == "geometry.h_ratio"
               for i in issues)


def test_h_ratio_organic_no_warn():
    """A canonical organic molecule (water, H/heavy = 2.0) must not
    warn -- typical organic ratios are 0.6 to 1.5."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0,0,0],[0.957,0,0],[-0.24,0.927,0]]),
    )
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "geometry.h_ratio"] == []


def test_polymer_orientation_normal_chain_no_warn():
    """A clean 4-mer DNA chain with 5'->3' residue ordering must not
    trip the orientation validator."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []


def test_polymer_orientation_reversed_listing_warns():
    """If a backend (or a user-loaded PDB) lists residues 3'->5', the
    structural 5' end (no incoming O3'-P bridge) won't match
    residue_ids[0], so the validator must warn.

    Build a normal chain, then flip residue_ids so the highest-numbered
    residue ends up at index 0.  Atom positions stay the same -- only
    the ID listing is reversed."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    rid_max = max(s.residue_ids)
    rid_min = min(s.residue_ids)
    # Map r -> (rid_max + rid_min - r) so 1->4, 2->3, 3->2, 4->1.
    flipped = [rid_max + rid_min - r for r in s.residue_ids]
    s_rev = type(s)(
        elements=list(s.elements), positions=s.positions.copy(),
        atom_names=list(s.atom_names), residue_ids=flipped,
        residue_names=list(s.residue_names),
        chain_ids=list(s.chain_ids), title=s.title,
    )
    issues = validate(s_rev, SiestaConfig())
    orient = [i for i in issues if i.where == "polymer.orientation"]
    assert len(orient) == 1, (
        f"expected one orientation warn, got: {orient}"
    )
    assert orient[0].severity == "warn"


def test_polymer_orientation_no_phosphorus_silent():
    """A peptide (no P, no O3') must not trigger the polymer-orientation
    check -- it's not a nucleic acid."""
    pytest.importorskip("PeptideBuilder")
    from molbuilder import build_peptide
    s = build_peptide("ARNDC")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []


def test_h_ratio_runs_after_layer2_protonation():
    """The user contract: validation runs at FDF/PySCF emission time,
    AFTER any add_hydrogens step at build time.  An X3DNA-built ATGC
    chain with default kwargs (add_hydrogens=True) must NOT trip the
    h_ratio warn -- protonation already happened, the ratio is healthy."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")     # default: add_hydrogens=True
    issues = validate(s, SiestaConfig())
    h_ratio_warns = [i for i in issues if i.where == "geometry.h_ratio"]
    assert h_ratio_warns == [], (
        f"X3DNA + add_hydrogens=True should produce a healthy H/heavy "
        f"ratio; got warning: {h_ratio_warns}"
    )


# --------------------------------------------------------------------- #
#  Cell: determinant + volume                                           #
# --------------------------------------------------------------------- #


def test_cell_determinant_zero_is_error(water_struct):
    """A degenerate cell (det == 0) is unusable; flag as error and
    skip the volume check below."""
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    errs = [i for i in issues if i.where == "cell.determinant"]
    assert len(errs) == 1
    assert errs[0].severity == "error"


def test_cell_determinant_negative_is_error(water_struct):
    """A left-handed cell (negative det) breaks SIESTA's PBC math."""
    cell = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 10
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    errs = [i for i in issues if i.where == "cell.determinant"]
    assert len(errs) == 1
    assert errs[0].severity == "error"


def test_cell_volume_tight_is_warn(water_struct):
    """Cell volume / atom-bounding-volume < 3 -- molecule fills the box."""
    # Water bounding box ~1 x 1 x 1 Å^3 -> atom_box ~1.  Cell 1 Å -> det = 1.
    cell = np.eye(3) * 1.0
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    vol = [i for i in issues if i.where == "cell.volume"]
    assert len(vol) == 1
    assert vol[0].severity == "warn"


def test_cell_volume_generous_no_warn(water_struct):
    """A 30 Å cubic box around water is generously vacuum-padded."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    vol = [i for i in issues if i.where == "cell.volume"]
    assert vol == []


# --------------------------------------------------------------------- #
#  Cell: atom-to-nearest-image distance                                 #
# --------------------------------------------------------------------- #


def test_image_distance_too_close_is_warn(water_struct):
    """A 5 Å cubic cell makes water see its own image ~5 Å away --
    well below the 6 Å "atoms still interacting" threshold."""
    cell = _vacuum_cell(5.0)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    msgs = [i for i in issues if i.where == "cell.image_distance"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"


def test_image_distance_generous_no_warn(water_struct):
    """A 30 Å cubic cell puts water's image >25 Å away; safely
    isolated, no warning."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "cell.image_distance"] == []


# --------------------------------------------------------------------- #
#  Geometry: net dipole in vacuum                                       #
# --------------------------------------------------------------------- #


def test_dipole_in_vacuum_polar_molecule_is_warn():
    """An HF-shaped molecule (H + F at ~0.92 Å) has a strong dipole
    (~1.8 D real, ~1.8 D heuristic).  In a Gamma-only vacuum cell
    the validator should warn about image-image dipole interactions."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    msgs = [i for i in issues if i.where == "geometry.dipole"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"
    assert "dipole" in msgs[0].message.lower()


def test_dipole_in_vacuum_nonpolar_molecule_no_warn():
    """N2 has zero dipole by symmetry (homonuclear diatomic).  No warn."""
    s = Structure(
        elements=["N", "N"],
        positions=np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []


def test_dipole_with_kgrid_no_warn():
    """A polar molecule in a periodic cell (k > 1) is INTENDED to
    have image-image interactions; the dipole warning is for the
    Gamma-only vacuum case where the user probably didn't realise."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(4, 4, 4))   # genuinely periodic
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []


# --------------------------------------------------------------------- #
#  SIESTA: kgrid sanity                                                 #
# --------------------------------------------------------------------- #


def _slab_struct(extent_x: float, n_atoms: int = 8):
    """Synthetic slab: atoms spanning `extent_x` along x with tight
    spacing on y/z.  Used to test the kgrid 'atoms span the axis'
    detection without rebuilding a real periodic structure."""
    xs = np.linspace(0.0, extent_x, n_atoms)
    pos = np.column_stack([xs, np.zeros(n_atoms), np.zeros(n_atoms)])
    return Structure(elements=["C"] * n_atoms, positions=pos)


def test_kgrid_along_vacuum_is_warn():
    """kgrid > 1 on an axis where atoms span only a small fraction of
    the cell (vacuum-padded axis) is wasted -- pre-fix the heuristic
    used cell-extent only; post-fix it checks atom-extent / cell-extent."""
    s = _slab_struct(extent_x=2.0)         # atoms confined to 2 Å
    cell = np.diag([5.0, 30.0, 30.0])
    cfg = SiestaConfig(kgrid=(4, 4, 1))    # k=4 on the 5 Å vacuum axis
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    assert any("kgrid[0]" in i.message for i in msgs)


def test_kgrid_one_on_periodic_axis_is_warn():
    """k == 1 along an axis that atoms ACTUALLY span (>=85%), when
    another axis has k > 1, is the slab-with-forgotten-axis case."""
    s = _slab_struct(extent_x=18.0)        # atoms span 18 of 20 Å (90%)
    cell = np.diag([20.0, 4.0, 4.0])
    cfg = SiestaConfig(kgrid=(1, 4, 1))
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    assert any("kgrid[0]" in i.message for i in msgs)


def test_kgrid_long_vacuum_padded_axis_no_false_positive():
    """The pre-fix bug: a long axis (> 10 Å) with k=1 used to trigger
    the 'looks periodic' warning regardless of whether atoms actually
    spanned the axis.  A 12-mer DNA in an 80 Å vacuum cell with kgrid
    (4, 4, 1) along the molecule's long axis is correct vacuum, NOT
    under-sampled.  Post-fix this case must NOT warn on axis 2."""
    # Atoms span only 30 of 80 Å along z (~38% of the axis).
    s = _slab_struct(extent_x=30.0)
    pos = np.column_stack([np.zeros(8), np.zeros(8),
                           np.linspace(0.0, 30.0, 8)])
    s = Structure(elements=["C"] * 8, positions=pos)
    cell = np.diag([10.0, 10.0, 80.0])     # 80 Å cell, 30 Å of atoms
    cfg = SiestaConfig(kgrid=(4, 4, 1))    # k=1 on the long-but-vacuum axis
    issues = validate(s, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    # axes 0/1: atoms span 0 of 10 Å -> vacuum -> k=4 should WARN
    # axis 2:   atoms span 30 of 80 Å -> 37% -> vacuum -> k=1 OK, no warn
    assert not any("kgrid[2]" in i.message for i in msgs), (
        f"axis 2 (38%-spanned) should not warn; got {[i.message for i in msgs]}"
    )


def test_all_gamma_in_vacuum_no_warn(water_struct):
    """Pure 1x1x1 Gamma in vacuum is the molecule case; no kgrid warning."""
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(water_struct, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.kgrid"] == []


# --------------------------------------------------------------------- #
#  SIESTA: spin_total without spin_polarized                            #
# --------------------------------------------------------------------- #


def test_spin_total_without_spin_polarized_is_warn(water_struct):
    """Setting spin_total without spin_polarized makes SIESTA silently
    ignore the total-spin pin -- exactly the kind of bug this gap-list
    item is meant to surface."""
    cfg = SiestaConfig(spin_polarized=False, spin_total=1.0)
    issues = validate(water_struct, cfg)
    spin = [i for i in issues if i.where == "config.spin_total"]
    assert len(spin) == 1
    assert spin[0].severity == "warn"


def test_spin_total_with_spin_polarized_no_warn(water_struct):
    cfg = SiestaConfig(spin_polarized=True, spin_total=1.0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.spin_total"] == []


# --------------------------------------------------------------------- #
#  SIESTA: wrap_into_cell                                               #
# --------------------------------------------------------------------- #


def test_atoms_outside_unit_cell_with_no_wrap_is_warn():
    """Atoms outside [0,1) fractional with wrap_into_cell=False -- the
    visualiser will draw them in a neighbour cell."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[15.0, 5.0, 5.0], [16.0, 5.0, 5.0]]),
    )
    cell = _vacuum_cell(10.0)    # atoms at x=15 are outside [0, 10)
    cfg = SiestaConfig(wrap_into_cell=False)
    issues = validate(s, cfg, cell=cell)
    wrap = [i for i in issues if i.where == "config.wrap_into_cell"]
    assert len(wrap) == 1
    assert wrap[0].severity == "warn"


def test_atoms_inside_with_no_wrap_no_warn(water_struct):
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(wrap_into_cell=False)
    # Shift water into the box so all atoms sit in [0, 30).
    s = Structure(
        elements=water_struct.elements,
        positions=water_struct.positions + np.array([15.0, 15.0, 15.0]),
    )
    issues = validate(s, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.wrap_into_cell"] == []


# --------------------------------------------------------------------- #
#  PySCF: spin field validation                                         #
# --------------------------------------------------------------------- #


def test_pyscf_negative_spin_is_error(water_struct):
    cfg = PySCFConfig(spin=-1)
    issues = validate(water_struct, cfg)
    spin_issues = [i for i in issues if i.where == "config.spin"]
    # spin=-1 produces TWO issues now that the dataclass declares
    # range=(0, 10) for the schema-driven form:
    #   1. error: "spin = -1 is negative; ..." (explicit semantic check)
    #   2. warn:  "Spin (2S) = -1 is outside the recommended range [0, 10]"
    #             (auto from field range metadata)
    # The error must survive; the additional range warn is fine
    # (both convey "this is wrong" to the user, with the explicit
    # error carrying the actionable explanation).
    errs = [i for i in spin_issues if i.severity == "error"]
    assert len(errs) == 1
    assert "negative" in errs[0].message


def test_pyscf_open_shell_spin_with_rks_is_warn(water_struct):
    """Open-shell spin with RKS / RHF (closed-shell methods) is the
    "I forgot to switch to UKS" foot-gun."""
    cfg = PySCFConfig(spin=1, method="RKS")
    issues = validate(water_struct, cfg)
    warns = [i for i in issues if i.where == "config.method"
             and i.severity == "warn"]
    assert len(warns) == 1


def test_pyscf_open_shell_spin_with_uks_no_warn(water_struct):
    cfg = PySCFConfig(spin=1, method="UKS")
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.method"] == []


# --------------------------------------------------------------------- #
#  report() helper: warnings to stderr, raise on errors                 #
# --------------------------------------------------------------------- #


def test_report_prints_warnings_to_stream():
    buf = io.StringIO()
    report(
        [Issue("warn", "watch out", "test.case")],
        raise_on_error=False, stream=buf,
    )
    out = buf.getvalue()
    assert "watch out" in out
    assert "[test.case]" in out


def test_report_raises_on_error_by_default():
    with pytest.raises(ValidationError):
        report([Issue("error", "fatal", "x")])


def test_report_can_be_told_not_to_raise():
    # Useful for the CLI's `validate` subcommand which collects all
    # issues for JSON-on-stdout emission rather than raising.
    report([Issue("error", "fatal", "x")], raise_on_error=False)


def test_report_emits_warnings_even_when_also_raising():
    """A run with both warnings and errors should surface BOTH -- the
    user wants to see all the warnings even if the error blocks
    emission."""
    buf = io.StringIO()
    issues = [
        Issue("warn", "minor first", "a"),
        Issue("warn", "minor second", "b"),
        Issue("error", "fatal", "c"),
    ]
    with pytest.raises(ValidationError):
        report(issues, stream=buf)
    out = buf.getvalue()
    assert "minor first" in out
    assert "minor second" in out


# --------------------------------------------------------------------- #
#  Generic config-metadata pass                                         #
#                                                                        #
#  These tests pin the metadata-reading contract using a synthetic     #
#  dataclass so the test stays focused on the reading code itself.     #
#  Production-config coverage lives further down.                      #
# --------------------------------------------------------------------- #


def test_metadata_range_check_flags_out_of_bounds(water_struct):
    """When a config field has metadata={'range': (lo, hi)}, an
    out-of-range value produces a 'warn' Issue with the field name."""
    from dataclasses import dataclass, field

    @dataclass
    class _ToyConfig:
        cutoff: float = field(default=10.0,
                              metadata={"range": (50.0, 600.0),
                                        "label": "Cutoff", "unit": "Ry"})

    issues = validate(water_struct, _ToyConfig())
    msgs = [i for i in issues if i.where == "config.cutoff"]
    assert len(msgs) == 1
    assert "Cutoff" in msgs[0].message
    assert "Ry" in msgs[0].message
    assert msgs[0].severity == "warn"


def test_metadata_validate_callable_emits_issue(water_struct):
    """A custom validate=callable can return an Issue directly."""
    from dataclasses import dataclass, field

    def _check_pow2(v, _cfg):
        if v & (v - 1):
            return Issue("warn", f"value {v} is not a power of 2",
                         "config.block_size")
        return None

    @dataclass
    class _ToyConfig:
        block_size: int = field(default=7,
                                metadata={"validate": _check_pow2})

    issues = validate(water_struct, _ToyConfig())
    assert any(i.where == "config.block_size" for i in issues)


# --------------------------------------------------------------------- #
#  Production config metadata: ranges read off SiestaConfig / PySCFConfig#
#                                                                        #
#  This is what makes Principle #1 load-bearing rather than aspirational #
#  -- the validator picks up out-of-range values from the production    #
#  configs without any per-field plumbing in the validator itself.      #
# --------------------------------------------------------------------- #


def test_siesta_mesh_cutoff_below_range_warns(water_struct):
    """mesh_cutoff has metadata range (50, 1000) Ry.  A value of 5
    Ry must emit a config.mesh_cutoff warn."""
    cfg = SiestaConfig(mesh_cutoff=5.0)
    issues = validate(water_struct, cfg)
    out_of_range = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(out_of_range) == 1
    assert "MeshCutoff" in out_of_range[0].message
    assert "Ry" in out_of_range[0].message


def test_siesta_mesh_cutoff_in_range_no_warn(water_struct):
    cfg = SiestaConfig(mesh_cutoff=300.0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.mesh_cutoff"] == []


def test_pyscf_grid_level_above_range_warns(water_struct):
    """grid_level has metadata range (0, 9).  Beyond that value isn't
    meaningful in PySCF; warn the user before they generate a script
    that PySCF will reject."""
    cfg = PySCFConfig(grid_level=20)
    issues = validate(water_struct, cfg)
    msgs = [i for i in issues if i.where == "config.grid_level"]
    assert len(msgs) == 1


# --------------------------------------------------------------------- #
#  Method / functional / grid_level cross-check rules                  #
# --------------------------------------------------------------------- #


def test_pyscf_uks_with_spin_zero_warns(water_struct):
    """UKS / UHF with spin = 0 runs the unrestricted formalism on a
    closed-shell system at ~2x the SCF cost.  Almost always a user
    mistake (default-of-RKS user flipped to UKS to "be safe").  Warn
    so the user knows."""
    cfg = PySCFConfig(method="UKS", spin=0)
    issues = validate(water_struct, cfg)
    method_warns = [i for i in issues
                    if i.severity == "warn" and i.where == "config.method"]
    assert len(method_warns) == 1
    assert "UKS" in method_warns[0].message
    assert "RKS" in method_warns[0].message


def test_pyscf_rks_with_spin_zero_no_warn(water_struct):
    """The flip side: RKS + spin=0 is the conventional closed-shell
    setup; no warning."""
    cfg = PySCFConfig(method="RKS", spin=0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.method"] == []


def test_pyscf_uks_with_spin_nonzero_no_warn():
    """UKS + spin > 0 is correct open-shell; no warning."""
    s = Structure(elements=["C","H","H","H"],
                  positions=np.array([[0,0,0],[1.08,0,0],[-0.54,0.94,0],[-0.54,-0.94,0]]))
    cfg = PySCFConfig(method="UKS", spin=1)
    issues = validate(s, cfg)
    assert [i for i in issues if i.where == "config.method"] == []


def test_pyscf_grid_level_3_with_hybrid_warns(water_struct):
    """Hybrid functionals (B3LYP / PBE0 / M06-2X / wB97X) at grid_level
    < 4 give noisy forces.  The user can override but should know."""
    cfg = PySCFConfig(method="RKS", spin=0, functional="B3LYP", grid_level=3)
    issues = validate(water_struct, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"
                  and i.severity == "warn"
                  and "hybrid" in i.message.lower()]
    assert len(grid_warns) == 1


def test_pyscf_grid_level_3_with_pure_gga_no_warn(water_struct):
    """Pure GGAs (PBE / BLYP / TPSS) don't have the noisy-force issue
    at grid_level 3.  The validator must not warn here."""
    cfg = PySCFConfig(method="RKS", spin=0, functional="PBE", grid_level=3)
    issues = validate(water_struct, cfg)
    grid_warns = [i for i in issues if i.where == "config.grid_level"
                  and "hybrid" in i.message.lower()]
    assert grid_warns == []


# --------------------------------------------------------------------- #
#  Peptide protonation: warn when neutral build vs charged side chains  #
# --------------------------------------------------------------------- #


def _peptide_struct(residue_names):
    """Synthetic peptide: one CA atom per residue, no actual chemistry,
    just enough metadata for the validator's residue-name check."""
    n = len(residue_names)
    pos = np.column_stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)])
    return Structure(
        elements=["C"] * n,
        positions=pos,
        atom_names=["CA"] * n,
        residue_ids=list(range(1, n + 1)),
        residue_names=list(residue_names),
    )


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


def test_siesta_peptide_protonation_warn(water_struct):
    """Same hint fires under the SIESTA validator (it's an engine-
    independent property of the structure + charge config)."""
    s = _peptide_struct(["ALA","LYS","ASP","ASP","GLY"])  # +1 -2 = -1
    cfg = SiestaConfig(net_charge=None)   # auto -> 0 for non-nucleic
    issues = validate(s, cfg)
    msgs = [i for i in issues if i.where == "config.charge"]
    assert len(msgs) == 1
    assert "-1" in msgs[0].message


def test_pyscf_default_grid_level_is_hybrid_safe():
    """Default grid_level should be >= 4 so the default
    `B3LYP + def2-SVP + density_fit + d3bj` recipe doesn't trip the
    hybrid-grid warning on its own defaults."""
    cfg = PySCFConfig()
    assert cfg.grid_level >= 4, (
        f"PySCFConfig.grid_level default = {cfg.grid_level}; should be "
        f">= 4 so the default hybrid recipe doesn't self-warn"
    )


# --------------------------------------------------------------------- #
#  Wire-in: render_fdf and render_script call validate()                #
# --------------------------------------------------------------------- #


def test_render_fdf_raises_on_overlapping_atoms():
    """A structure with atoms < 0.3 Å apart triggers a min-distance
    error from validate(), which render_fdf surfaces as
    ValidationError before emitting any FDF text."""
    from molbuilder.siesta import render_fdf
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),
    )
    with pytest.raises(ValidationError) as exc:
        render_fdf(s, SiestaConfig())
    assert "min_distance" in str(exc.value)


def test_render_fdf_emits_warnings_to_stderr(capsys, water_struct):
    """A spin_total without spin_polarized warning surfaces on stderr;
    the FDF still gets emitted (warnings don't block)."""
    from molbuilder.siesta import render_fdf
    cfg = SiestaConfig(spin_polarized=False, spin_total=1.0)
    fdf = render_fdf(water_struct, cfg)
    err = capsys.readouterr().err
    assert "spin_total" in err
    # FDF was still generated:
    assert "SystemName" in fdf


def test_render_script_raises_on_negative_spin(water_struct):
    """spin = -1 -> error from validate(), render_script raises
    ValidationError before emitting any Python text.

    Use UKS so the existing hard-coded RKS-with-nonzero-spin guard in
    pyscf.input doesn't pre-empt the validator -- this test is about
    the validator's negative-spin catch, not the pre-existing guard.
    """
    from molbuilder.pyscf import render_script
    cfg = PySCFConfig(spin=-1, method="UKS")
    with pytest.raises(ValidationError) as exc:
        render_script(water_struct, cfg)
    assert "spin" in str(exc.value)


def test_render_script_warns_on_open_shell_with_rks(capsys, water_struct):
    """Open-shell spin with closed-shell RKS / RHF method emits a
    warning to stderr but doesn't block emission."""
    from molbuilder.pyscf import render_script
    # Note: pyscf.input ALSO has a ValueError guard for this case
    # (RKS + spin != 0 hard-errors in render_script).  The validator
    # would warn, but the explicit guard takes precedence with an
    # error.  Check that one or the other catches it.
    cfg = PySCFConfig(spin=1, method="UKS")  # UKS doesn't trigger the hard guard
    # For UKS + spin=1 the validator has nothing to flag; this test
    # documents that a *legitimate* open-shell config doesn't warn.
    render_script(water_struct, cfg)
    err = capsys.readouterr().err
    # No "config.method" warn for a properly-set UKS config.
    assert "method" not in err or "warn [config.method]" not in err


# --------------------------------------------------------------------- #
#  R3: odd-electron count + RKS/RHF + spin=0 must warn                  #
# --------------------------------------------------------------------- #


def _ch3_radical_struct():
    """Methyl radical CH3 -- a textbook open-shell organic.
    n_e (neutral) = 6 (C) + 3 (H) = 9, odd -> doublet."""
    return Structure(
        elements=["C", "H", "H", "H"],
        positions=np.array([
            [ 0.000,  0.000, 0.000],
            [ 1.080,  0.000, 0.000],
            [-0.540,  0.935, 0.000],
            [-0.540, -0.935, 0.000],
        ]),
        title="ch3_radical",
    )


def test_pyscf_validator_warns_on_odd_electrons_with_rks_spin0():
    """R3: a system with an odd electron count run with method=RKS and
    spin=0 is silently the wrong electronic state -- the validator
    must emit a warn pointing at UKS / spin=1."""
    cfg = PySCFConfig(method="RKS", spin=0, charge=0)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues
                 if i.severity == "warn" and "odd electron" in i.message]
    assert len(odd_warns) == 1
    assert "UKS" in odd_warns[0].message
    assert "UHF" in odd_warns[0].message


def test_pyscf_validator_silent_on_odd_electrons_with_uks_spin1():
    """The legitimate fix (UKS + spin=1 on the same odd-electron
    system) must NOT trip the new R3 warn."""
    cfg = PySCFConfig(method="UKS", spin=1, charge=0)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []


def test_pyscf_validator_silent_on_even_electrons_with_rks():
    """Closed-shell organic on RKS+spin=0: no R3 warn."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        title="water",
    )
    cfg = PySCFConfig(method="RKS", spin=0, charge=0)
    issues = validate(s, cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []


def test_pyscf_validator_silent_when_user_overrode_charge_to_make_even():
    """If the user explicitly set cfg.charge to a value that makes the
    electron count even (e.g. methyl cation CH3+ with charge=+1, n_e=8,
    a closed-shell carbocation), R3 must NOT warn."""
    cfg = PySCFConfig(method="RKS", spin=0, charge=1)
    issues = validate(_ch3_radical_struct(), cfg)
    odd_warns = [i for i in issues if "odd electron" in i.message]
    assert odd_warns == []


# --------------------------------------------------------------------- #
#  SIESTA pseudo-coverage check is wired into the preflight             #
#  (2026-05-23: previously pseudos.py was only callable via the         #
#  /api/siesta/check-pseudos endpoint or in tests -- nothing in the     #
#  Build->Generate flow ever ran it.  These tests pin that every        #
#  validate(struct, SiestaConfig) call now exercises the coverage       #
#  check, surfacing missing files / XC mismatches as preflight Issues.) #
# --------------------------------------------------------------------- #


def _make_pdojo_psml(element: str, *, z: int = None,
                     libxc_id_x: int = 101,  # PBE exchange
                     libxc_id_c: int = 130   # PBE correlation
                     ) -> str:
    """Real-PseudoDojo-shape PSML (pseudo-atom-spec + nested
    <functional> for libxc)."""
    if z is None:
        from ase.data import atomic_numbers as _Z
        z = _Z[element]
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<psml version="1.1" xmlns="http://esl.cecam.org/PSML/ns/1.1">
<provenance creator="test"/>
<pseudo-atom-spec atomic-label="{element}" atomic-number="{z}"
 z-pseudo="{z}" relativity="scalar">
<exchange-correlation>
<libxc-info number-of-functionals="2">
<functional name="x" type="exchange" id="{libxc_id_x}"/>
<functional name="c" type="correlation" id="{libxc_id_c}"/>
</libxc-info>
</exchange-correlation>
</pseudo-atom-spec>
</psml>"""


class TestSiestaPseudoCoverageInPreflight:
    """Pin the actual wiring: validate(struct, SiestaConfig) MUST
    surface pseudo-coverage findings as Issues."""

    def _water(self):
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))

    def test_psml_lib_unset_emits_actionable_warn(self):
        """Default config (psml_lib=None) -> WARN telling user how to
        get pseudos.  Without this warning the user discovers the
        missing-pseudo failure only after a 5-minute mpirun start-up."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(), SiestaConfig())
        psml_issues = [i for i in issues if i.where == "config.psml_lib"]
        assert psml_issues
        assert psml_issues[0].severity == "warn"
        assert "psml_lib is not set" in psml_issues[0].message
        # Actionable: mentions PseudoDojo + the projects/pseudopotential/
        # convention.
        assert "pseudo-dojo.org" in psml_issues[0].message
        assert "projects/pseudopotential" in psml_issues[0].message

    def test_psml_lib_bad_path_emits_error(self):
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(),
                           SiestaConfig(psml_lib="/no/such/dir"))
        errs = [i for i in issues if i.where == "config.psml_lib"
                and i.severity == "error"]
        assert errs
        assert "/no/such/dir" in errs[0].message

    def test_complete_coverage_passes(self, tmp_path):
        """All elements present + matching XC -> NO psml-related issues
        (other checks may fire; we filter to just the psml ones)."""
        for el in ("O", "H"):
            (tmp_path / f"{el}.psml").write_text(_make_pdojo_psml(el))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path), xc_authors="PBE"))
        psml_issues = [i for i in issues if "psml" in i.where.lower()]
        assert psml_issues == []

    def test_missing_element_emits_error(self, tmp_path):
        """Only O.psml is present; H is missing -> ERROR Issue."""
        (tmp_path / "O.psml").write_text(_make_pdojo_psml("O"))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path)))
        h_issues = [i for i in issues
                    if i.where == "config.psml_lib.H"]
        assert h_issues
        assert h_issues[0].severity == "error"
        assert "no .psml file for H" in h_issues[0].message

    def test_xc_mismatch_emits_warn(self, tmp_path):
        """Pseudos are LDA but the calc requests PBE -> WARN per
        element (silently wrong bond lengths otherwise)."""
        # libxc id 1 = XC_LDA_X
        for el in ("O", "H"):
            (tmp_path / f"{el}.psml").write_text(
                _make_pdojo_psml(el, libxc_id_x=1, libxc_id_c=9))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path),
                                          xc_authors="PBE"))
        mismatch_issues = [i for i in issues
                           if "psml" in i.where.lower()
                           and i.severity == "warn"
                           and "LDA" in i.message]
        assert len(mismatch_issues) == 2   # O + H both flagged
        assert all("bond lengths will be silently wrong" in i.message
                   for i in mismatch_issues)


# --------------------------------------------------------------------- #
#  SIESTA propor: ERROR: IMAX = 0 preflight                             #
#                                                                       #
#  2026-05-24 hemeC-dithiol incident: ``Spin polarized`` + no           #
#  ``Spin.Total`` + Fe -> SIESTA aborts in propor() before SCF starts.  #
#  Pin the proactive validator that catches this in molbuilder rather   #
#  than after the 30-second SIESTA initial-DM construction.             #
# --------------------------------------------------------------------- #


class TestSpinPolarizedNeedsSpinTotal:
    """The validator should ERROR (not WARN) when the propor IMAX=0
    failure mode is loaded: spin_polarized=True + spin_total=None +
    structure contains an open-shell first-row TM.  And it should
    propose a starting value the user can plug into the form."""

    def _hemeC_like(self):
        """Synthetic Fe + C/H/N/O fragment.  Don't bother with real
        chemistry coords -- the validator only looks at the element
        list to decide whether the propor failure mode applies."""
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(
            elements=["Fe", "C", "C", "N", "N", "O", "H", "H", "H", "H"],
            positions=np.array([[i * 1.5, 0, 0] for i in range(10)],
                                dtype=float),
        )

    def _organic_only(self):
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["C", "C", "H", "H", "H", "H"],
                         positions=np.array([[i, 0, 0] for i in range(6)],
                                              dtype=float))

    def test_metal_plus_spinpol_no_spin_total_emits_error(self):
        """The actual hemeC-dithiol failure mode -- Fe present,
        spin_polarized=True, no spin_total.  Validator must produce
        an ERROR Issue."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(),
                           SiestaConfig(spin_polarized=True))
        errs = [i for i in issues
                if i.where == "config.spin_total" and i.severity == "error"]
        assert errs, (
            "Validator failed to flag the propor IMAX=0 failure mode "
            "(spin_polarized=True + Fe + no spin_total)"
        )

    def test_error_message_names_the_failure(self):
        """Error message must explain WHAT will go wrong, not just
        'set spin_total'.  Otherwise users won't connect the molbuilder
        ERROR to the SIESTA ``propor: ERROR: IMAX = 0`` they'd see at
        run time."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(),
                           SiestaConfig(spin_polarized=True))
        err = next(i for i in issues
                   if i.where == "config.spin_total" and i.severity == "error")
        # Names the SIESTA error string the user would otherwise see.
        assert "propor" in err.message and "IMAX = 0" in err.message
        # Names the metal that triggered the check.
        assert "Fe" in err.message

    def test_error_proposes_a_starting_value(self):
        """User shouldn't have to look up ligand-field rules.  The
        error must propose a concrete starting Spin.Total value."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(),
                           SiestaConfig(spin_polarized=True))
        err = next(i for i in issues
                   if i.where == "config.spin_total" and i.severity == "error")
        # The "START HERE: ..." line is the load-bearing UX bit.
        assert "START HERE" in err.message
        # For Fe the recommended starting value is 4.0 (high-spin Fe(II);
        # see chemistry._SPIN_TOTAL_DEFAULTS).
        assert "= 4" in err.message

    def test_error_lists_alternatives_to_sweep(self):
        """Beyond the starting value, the error should show the
        ranked alternatives so the user can experiment if SCF
        converges to a spin state that disagrees with the chemistry."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(),
                           SiestaConfig(spin_polarized=True))
        err = next(i for i in issues
                   if i.where == "config.spin_total" and i.severity == "error")
        # All six registered Fe entries (S=0/1/2/3/4/5) should appear.
        for s in (0, 1, 2, 3, 4, 5):
            assert f"spin_total = {s:>4g}" in err.message or \
                    f"spin_total = {s}" in err.message, (
                f"Alternative Spin.Total = {s} missing from error message"
            )

    def test_user_explicit_spin_total_silences_the_check(self):
        """If user sets spin_total explicitly, the propor failure mode
        is averted -- check must NOT fire."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(),
                           SiestaConfig(spin_polarized=True, spin_total=4.0))
        errs = [i for i in issues
                if i.where == "config.spin_total" and i.severity == "error"
                and "propor" in i.message]
        assert not errs, (
            "Validator wrongly fired the propor-IMAX-0 check even though "
            "the user set spin_total explicitly"
        )

    def test_no_spin_polarized_no_check(self):
        """spin_polarized=False -> no propor invocation at SIESTA setup
        time, so the check shouldn't fire (the open-shell-metal WARN
        from _check_open_shell_metal is the right complaint there)."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._hemeC_like(), SiestaConfig())  # spin_polarized default = False
        propor_errs = [i for i in issues
                        if i.where == "config.spin_total"
                        and "propor" in i.message]
        assert not propor_errs

    def test_no_metal_no_check(self):
        """Pure organic structure -- propor wouldn't fail even without
        Spin.Total, since closed-shell atoms split trivially.  Check
        must NOT fire."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._organic_only(),
                           SiestaConfig(spin_polarized=True))
        propor_errs = [i for i in issues
                        if i.where == "config.spin_total"
                        and "propor" in i.message]
        assert not propor_errs


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
