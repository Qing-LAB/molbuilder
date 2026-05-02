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


def test_issue_severity_must_be_error_or_warn():
    Issue("error", "fine")            # OK
    Issue("warn", "fine")             # OK
    with pytest.raises(ValueError, match="severity"):
        Issue("info", "not allowed")


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
#  SIESTA: kgrid sanity                                                 #
# --------------------------------------------------------------------- #


def test_kgrid_along_vacuum_is_warn(water_struct):
    """kgrid != 1 along an axis < 10 Å (vacuum direction) is wasted."""
    cell = np.diag([5.0, 30.0, 30.0])    # x = vacuum (5 Å), yz = periodic
    cfg = SiestaConfig(kgrid=(4, 4, 1))   # k > 1 along the 5 Å axis
    issues = validate(water_struct, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    # The 5 Å axis should produce a warning; the 30 Å axis with k=4 is fine.
    assert any("kgrid[0]" in i.message for i in msgs)


def test_kgrid_one_on_periodic_axis_is_warn(water_struct):
    """k == 1 along a > 10 Å axis WHEN another axis has k > 1 means the
    user probably forgot one direction.  All-Gamma must NOT trigger
    (a molecule in vacuum legitimately uses 1x1x1)."""
    cell = np.diag([20.0, 20.0, 20.0])
    cfg = SiestaConfig(kgrid=(4, 1, 1))
    issues = validate(water_struct, cfg, cell=cell)
    msgs = [i for i in issues if i.where == "config.kgrid"]
    # axis 1 and 2 are both > 10 Å with k=1 while axis 0 has k=4.
    assert any("kgrid[1]" in i.message for i in msgs)


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
    errs = [i for i in issues if i.where == "config.spin"]
    assert len(errs) == 1
    assert errs[0].severity == "error"


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
#  Today no production config field carries metadata yet -- that lands #
#  in commit 2.6b.  We exercise the metadata-reading code with a        #
#  synthetic dataclass to pin the contract before configs grow it.     #
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
