"""L1 tests for molbuilder.script_bundle (PR-A scope).

PR-A defines the dataclass shape, error type, and reserved API
seats; the assembler + materializer implementations land in
PR-B/C/D.  These tests pin the contract surface so the later PRs
can't accidentally widen it.

Contract: ``docs/protocols/bundle-contract.md``.
"""
from __future__ import annotations

from dataclasses import is_dataclass, fields
from pathlib import Path

import pytest

from molbuilder import script_bundle as sb
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Public API surface                                                   #
# --------------------------------------------------------------------- #


def test_module_exports_pinned_names():
    """``__all__`` is the contract surface.  Widening it requires a
    doc + test update."""
    assert sorted(sb.__all__) == sorted([
        "BundleError",
        "RunBundle",
        "assemble_from_run_dir",
        "write_bundle_as_handoff",
    ])


def test_bundle_error_is_exception():
    assert issubclass(sb.BundleError, Exception)


# --------------------------------------------------------------------- #
#  RunBundle dataclass shape (bundle-contract.md § 3)                   #
# --------------------------------------------------------------------- #


def test_run_bundle_is_frozen_dataclass():
    assert is_dataclass(sb.RunBundle)
    # Frozen: assigning after construction must raise.
    s = Structure(elements=["H"], positions=[[0.0, 0.0, 0.0]])
    b = sb.RunBundle(
        structure=s,
        regions={},
        frozen_atoms=[],
        user_custom_lines=[],
        provenance={},
        source_script=Path("/tmp/none.fdf"),
        source_engine="siesta",
        final_coords_from="fdf-initial",
        notes=[],
    )
    with pytest.raises(Exception):
        b.regions = {"x": [0]}    # frozen=True -> FrozenInstanceError


def test_run_bundle_field_names_match_contract():
    """Bundle-contract.md § 3 lists exactly these fields.  Any
    addition / removal must update the doc + this test together."""
    got = {f.name for f in fields(sb.RunBundle)}
    expected = {
        "structure",
        "regions",
        "frozen_atoms",
        "user_custom_lines",
        "provenance",
        "source_script",
        "source_engine",
        "final_coords_from",
        "notes",
    }
    assert got == expected


def test_run_bundle_notes_defaults_to_list():
    """The ``notes`` field uses ``field(default_factory=list)`` so
    every constructed bundle has a real list, never ``None``.  The
    bundle-contract pins this for diagnostic flow-through."""
    s = Structure(elements=["H"], positions=[[0.0, 0.0, 0.0]])
    b = sb.RunBundle(
        structure=s,
        regions={},
        frozen_atoms=[],
        user_custom_lines=[],
        provenance={},
        source_script=Path("/tmp/x.fdf"),
        source_engine="siesta",
        final_coords_from="fdf-initial",
        # notes deliberately omitted -- default_factory should kick in
    )
    assert b.notes == []
    assert isinstance(b.notes, list)


# --------------------------------------------------------------------- #
#  API seats (PR-A stubs raise; PR-B/C/D fill them)                     #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  assemble_from_run_dir -- SIESTA branch (PR-B)                        #
# --------------------------------------------------------------------- #


_BOHR = 0.5291772108  # Å per Bohr


def _h2_xv_text() -> str:
    """Minimal valid .XV for a 2-atom H structure."""
    return (
        "  10.0   0.0   0.0   0.0 0.0 0.0\n"
        "   0.0  10.0   0.0   0.0 0.0 0.0\n"
        "   0.0   0.0  10.0   0.0 0.0 0.0\n"
        "  2\n"
        "  1   1   0.000   0.000   0.000   0.0 0.0 0.0\n"
        "  1   1   1.500   0.000   0.000   0.0 0.0 0.0\n"
    )


def _h2_fdf_text(*, atom_md_block: str = "") -> str:
    return (
        "SystemLabel h2\n"
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "    0.000   0.000   0.000   1\n"
        "    0.740   0.000   0.000   1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
        + atom_md_block
    )


def _atom_md(*, regions=None, frozen=None, n_atoms_total=2) -> str:
    """Emit an ATOM-METADATA block via the canonical emitter."""
    from molbuilder import script_contract as _sc
    block = _sc.emit_atom_metadata(
        regions=regions or {},
        frozen_atoms=frozen or [],
        n_atoms_total=n_atoms_total,
    )
    return ("" if block is None else "\n" + block + "\n")


def test_assemble_from_run_dir_siesta_with_xv(tmp_path):
    """The common case: .fdf + .XV present, ATOM-METADATA carries labels.
    Bundle reflects the converged geometry from .XV with the .fdf's labels."""
    (tmp_path / "h2.fdf").write_text(
        _h2_fdf_text(
            atom_md_block=_atom_md(
                regions={"L-electrode": [0]},
                frozen=[1],
                n_atoms_total=2,
            ),
        )
    )
    (tmp_path / "h2.XV").write_text(_h2_xv_text())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.source_engine == "siesta"
    assert bundle.final_coords_from == "xv"
    assert bundle.structure.elements == ["H", "H"]
    # Position from .XV (1.5 Bohr -> Å), NOT from .fdf (0.74 Å).
    import numpy as np
    np.testing.assert_allclose(
        bundle.structure.positions[1, 0], 1.5 * _BOHR, atol=1e-9,
    )
    assert bundle.regions == {"L-electrode": [0]}
    assert bundle.frozen_atoms == [1]
    assert bundle.source_script.name == "h2.fdf"


def test_assemble_from_run_dir_siesta_uses_system_label_for_xv(tmp_path):
    """Audit BLOCKER 3: SIESTA writes ``<SystemLabel>.XV``, not
    ``<basename>.XV``.  When the .fdf basename has a stage suffix
    that the SystemLabel doesn't (the molbuilder generator's
    convention), the bundle must find the .XV via SystemLabel."""
    # .fdf named with a stage suffix the SystemLabel doesn't have.
    (tmp_path / "h2-stage2.fdf").write_text(
        "SystemLabel h2\n"
        + _h2_fdf_text(
            atom_md_block=_atom_md(
                regions={"L-electrode": [0]},
                n_atoms_total=2,
            ),
        )
    )
    # .XV named after SystemLabel, NOT the .fdf basename.
    (tmp_path / "h2.XV").write_text(_h2_xv_text())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "xv"
    import numpy as np
    np.testing.assert_allclose(
        bundle.structure.positions[1, 0], 1.5 * _BOHR, atol=1e-9,
    )


def test_assemble_from_run_dir_siesta_warns_on_left_handed_xv(tmp_path):
    """A .XV with negative-determinant cell silently mirrors the
    structure through any Fractional projection.  Bundle must add
    a loud notes entry so the user sees it in the result panel.
    Audit follow-up: deferral #4."""
    (tmp_path / "h2.fdf").write_text(_h2_fdf_text())
    # Negate the third cell row of the .XV -> det < 0.
    (tmp_path / "h2.XV").write_text(
        "  10.0   0.0   0.0   0.0 0.0 0.0\n"
        "   0.0  10.0   0.0   0.0 0.0 0.0\n"
        "   0.0   0.0 -10.0   0.0 0.0 0.0\n"
        "  2\n"
        "  1   1   0.000   0.000   0.000   0.0 0.0 0.0\n"
        "  1   1   1.500   0.000   0.000   0.0 0.0 0.0\n"
    )
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "xv"  # still bundles
    combined = "\n".join(bundle.notes)
    assert "LEFT-HANDED" in combined
    assert "chirality" in combined.lower()


def test_assemble_from_run_dir_siesta_warns_on_left_handed_fdf_lattice(tmp_path):
    """Same warning surfaces when the .fdf's LatticeVectors block
    is left-handed, even when the .XV is fine."""
    fdf_with_lh_lattice = (
        "SystemLabel h2\n"
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "%block LatticeVectors\n"
        "  1.0 0.0  0.0\n"
        "  0.0 1.0  0.0\n"
        "  0.0 0.0 -1.0\n"
        "%endblock LatticeVectors\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "    0.000 0.000 0.000 1\n"
        "    0.740 0.000 0.000 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
    )
    (tmp_path / "h2.fdf").write_text(fdf_with_lh_lattice)
    (tmp_path / "h2.XV").write_text(_h2_xv_text())  # right-handed
    bundle = sb.assemble_from_run_dir(tmp_path)
    combined = "\n".join(bundle.notes)
    assert "LEFT-HANDED" in combined


def test_assemble_from_run_dir_siesta_xv_unreadable_warn_combined(tmp_path):
    """Audit IMPORTANT 5: when .XV exists but read_xv fails, the
    fallback note MUST loudly say 'NOT converged geometry' so the
    user knows the bundle is initial-coords-only.  Pre-fix the
    second half of the message was suppressed."""
    (tmp_path / "h2.fdf").write_text(_h2_fdf_text())
    (tmp_path / "h2.XV").write_text("garbage that cannot be parsed\n")
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "fdf-initial"
    combined = "\n".join(bundle.notes)
    assert "NOT converged geometry" in combined
    # Original parse-error reason also present, so the user can
    # diagnose what went wrong with the .XV file.
    assert "h2.XV" in combined


def test_assemble_from_run_dir_siesta_fdf_initial_fallback(tmp_path):
    """No .XV -> bundle falls back to .fdf initial coords + notes."""
    (tmp_path / "h2.fdf").write_text(_h2_fdf_text())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "fdf-initial"
    # Position from .fdf initial coords (0.74 Å, not converted).
    import numpy as np
    np.testing.assert_allclose(bundle.structure.positions[1, 0], 0.74)
    assert any("not converged geometry" in n.lower() or "initial-coords" in n.lower()
               for n in bundle.notes)


def test_assemble_from_run_dir_siesta_no_labels(tmp_path):
    """Run dir without ATOM-METADATA -> bundle assembles with empty
    regions/frozen.  Empty != absent in dataclass terms, but the
    bundle ALWAYS surfaces concrete dicts/lists (see § 3)."""
    (tmp_path / "h2.fdf").write_text(_h2_fdf_text())
    (tmp_path / "h2.XV").write_text(_h2_xv_text())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.regions == {}
    assert bundle.frozen_atoms == []


def test_assemble_from_run_dir_siesta_n_atoms_mismatch(tmp_path):
    """ATOM-METADATA n_atoms_total != .XV atom count -> BundleError."""
    (tmp_path / "h2.fdf").write_text(
        _h2_fdf_text(
            atom_md_block=_atom_md(
                regions={"r": [0, 1, 2]},
                n_atoms_total=99,   # lies about the system size
            ),
        )
    )
    (tmp_path / "h2.XV").write_text(_h2_xv_text())
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "n_atoms_total" in str(exc.value)
    assert "different runs" in str(exc.value)


def test_assemble_from_run_dir_errors_when_no_script(tmp_path):
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "no engine script" in str(exc.value).lower()


def test_assemble_from_run_dir_errors_when_dir_missing(tmp_path):
    with pytest.raises(sb.BundleError):
        sb.assemble_from_run_dir(tmp_path / "does-not-exist")


def test_assemble_from_run_dir_errors_when_both_engines_present(tmp_path):
    """Both .fdf and .py -> ambiguous, raise."""
    (tmp_path / "h2.fdf").write_text(_h2_fdf_text())
    (tmp_path / "h2.py").write_text("# placeholder pyscf script\n")
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "ambiguous" in str(exc.value).lower()


# --------------------------------------------------------------------- #
#  assemble_from_run_dir -- PySCF branch (PR-C)                         #
# --------------------------------------------------------------------- #


def _h2_py_text(*, with_atom_md: bool = True) -> str:
    """Minimal molbuilder-style PySCF script.  Same shape as the
    pyscf generator emits (JOB literal + gto.M atom block)."""
    base = (
        'JOB = "h2-test"\n'
        "from pyscf import gto\n"
        "mol = gto.M(\n"
        "    atom = '''\n"
        "    H    0.00000000    0.00000000    0.00000000\n"
        "    H    0.74000000    0.00000000    0.00000000\n"
        "    ''',\n"
        "    basis = 'def2-SVP',\n"
        ")\n"
    )
    if with_atom_md:
        base += _atom_md(regions={"R-electrode": [1]}, frozen=[0], n_atoms_total=2)
    return base


def _h2_optimized_xyz(disp: float = 0.80) -> str:
    """Final converged geometry as PySCF would have written.
    Default disp differs from the .py initial (0.74) so tests can
    tell which path the bundle took."""
    return (
        "2\n"
        "Optimized geometry (PySCF)\n"
        f"H 0.0 0.0 0.0\n"
        f"H {disp} 0.0 0.0\n"
    )


def test_assemble_from_run_dir_pyscf_with_optimized_xyz(tmp_path):
    """Common case: .py + <JOB>_optimized.xyz.  Bundle reflects
    the converged geom from .xyz, with labels from the .py."""
    (tmp_path / "h2.py").write_text(_h2_py_text())
    (tmp_path / "h2-test_optimized.xyz").write_text(_h2_optimized_xyz(0.80))
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.source_engine == "pyscf"
    assert bundle.final_coords_from == "py-opt"
    # 0.80 from _optimized.xyz, NOT 0.74 from .py initial.
    import numpy as np
    np.testing.assert_allclose(bundle.structure.positions[1, 0], 0.80)
    assert bundle.regions == {"R-electrode": [1]}
    assert bundle.frozen_atoms == [0]
    assert bundle.source_script.name == "h2.py"


def test_assemble_from_run_dir_pyscf_opt_xyz_unreadable_warn_combined(tmp_path):
    """Mirror of the SIESTA "XV unreadable" test: when the
    <JOB>_optimized.xyz exists but is malformed, the fallback note
    must say 'NOT converged geometry'.  Audit IMPORTANT 6."""
    (tmp_path / "h2.py").write_text(_h2_py_text())
    (tmp_path / "h2-test_optimized.xyz").write_text("not actually xyz\n")
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "py-initial"
    combined = "\n".join(bundle.notes)
    assert "NOT converged geometry" in combined


def test_assemble_from_run_dir_pyscf_initial_fallback(tmp_path):
    """No <JOB>_optimized.xyz -> .py initial-coords fallback + note."""
    (tmp_path / "h2.py").write_text(_h2_py_text())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "py-initial"
    import numpy as np
    np.testing.assert_allclose(bundle.structure.positions[1, 0], 0.74)
    assert any("not converged geometry" in n.lower() or "initial-coords" in n.lower()
               for n in bundle.notes)


def test_assemble_from_run_dir_pyscf_glob_fallback_when_job_missing(tmp_path):
    """If the .py has no JOB literal but exactly one *_optimized.xyz
    exists, fall back to the glob match.  Documents the lenient
    branch in bundle-contract.md § 4.1 PySCF table."""
    # Drop the JOB line from the canonical script.
    text = _h2_py_text().replace('JOB = "h2-test"\n', "# no JOB line\n")
    (tmp_path / "h2.py").write_text(text)
    (tmp_path / "anywhere_optimized.xyz").write_text(_h2_optimized_xyz(0.90))
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "py-opt"
    import numpy as np
    np.testing.assert_allclose(bundle.structure.positions[1, 0], 0.90)


def test_assemble_from_run_dir_pyscf_multiple_optimized_falls_back(tmp_path):
    """Multiple *_optimized.xyz + JOB doesn't pick: the glob is
    ambiguous, so the bundle falls back to .py initial + records
    the ambiguity in notes."""
    text = _h2_py_text().replace('JOB = "h2-test"\n', "# no JOB line\n")
    (tmp_path / "h2.py").write_text(text)
    (tmp_path / "a_optimized.xyz").write_text(_h2_optimized_xyz(0.80))
    (tmp_path / "b_optimized.xyz").write_text(_h2_optimized_xyz(0.90))
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.final_coords_from == "py-initial"
    assert any("multiple *_optimized.xyz" in n for n in bundle.notes)


def test_assemble_from_run_dir_pyscf_n_atoms_mismatch(tmp_path):
    """ATOM-METADATA n_atoms_total != _optimized.xyz atom count
    -> BundleError, same as the SIESTA branch."""
    py_text = _h2_py_text(with_atom_md=False) + _atom_md(
        regions={"x": [0]}, n_atoms_total=99,
    )
    (tmp_path / "h2.py").write_text(py_text)
    (tmp_path / "h2-test_optimized.xyz").write_text(_h2_optimized_xyz())
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "n_atoms_total" in str(exc.value)


def test_assemble_from_run_dir_pyscf_no_labels(tmp_path):
    """No ATOM-METADATA -> empty regions/frozen, bundle assembles."""
    (tmp_path / "h2.py").write_text(_h2_py_text(with_atom_md=False))
    (tmp_path / "h2-test_optimized.xyz").write_text(_h2_optimized_xyz())
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.regions == {}
    assert bundle.frozen_atoms == []
    assert bundle.final_coords_from == "py-opt"


def test_assemble_from_run_dir_pyscf_handwritten_script_errors(tmp_path):
    """A non-molbuilder .py without the triple-quoted atom block
    raises BundleError so the user knows to re-render."""
    (tmp_path / "h2.py").write_text(
        'JOB = "weird"\n'
        "from pyscf import gto\n"
        "mol = gto.M(atom = [['H', (0,0,0)], ['H', (0.74,0,0)]], basis='sto-3g')\n"
    )
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "triple-quoted" in str(exc.value)


def test_assemble_from_run_dir_picks_largest_py_among_multiple(tmp_path):
    """Mirror of the SIESTA-staged-run test, for .py.  Two .py files,
    pick the one with the larger atom-metadata n_atoms_total."""
    (tmp_path / "h2-stage1.py").write_text(
        _h2_py_text(with_atom_md=False) + _atom_md(
            regions={"small": [0]}, n_atoms_total=2,
        )
    )
    # Build a 50-atom .py with matching atom-md.
    big_lines = "\n".join(
        f"    H    {i*1.0:.3f}    0.0    0.0" for i in range(50)
    )
    big_py = (
        'JOB = "h2_big"\n'
        "from pyscf import gto\n"
        "mol = gto.M(\n"
        "    atom = '''\n"
        + big_lines + "\n"
        "    ''',\n"
        "    basis = 'def2-SVP',\n"
        ")\n"
        + _atom_md(regions={"big": list(range(50))}, n_atoms_total=50)
    )
    (tmp_path / "h2-stage2.py").write_text(big_py)
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.source_script.name == "h2-stage2.py"
    assert bundle.regions == {"big": list(range(50))}
    assert any("multiple .py" in n for n in bundle.notes)


def test_assemble_from_run_dir_picks_largest_fdf_among_multiple(tmp_path):
    """Staged-run case: two .fdf, pick the one with the larger
    atom-metadata n_atoms_total."""
    (tmp_path / "h2-stage1.fdf").write_text(
        _h2_fdf_text(atom_md_block=_atom_md(
            regions={"small": [0]}, n_atoms_total=2,
        ))
    )
    # A second .fdf claiming the run had grown to 100 atoms; pick it
    # even though it has no matching .XV (so we'll need fdf-initial
    # fallback -- but to validate we'd also need the actual coords
    # to match).  For this test, build a 100-atom .fdf via the
    # initial-coords block and align atom-md.
    big_coord_lines = "\n".join(
        f"    {i*1.0:.3f}   0.0   0.0   1" for i in range(100)
    )
    big_fdf = (
        "SystemLabel h2_big\n"
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        + big_coord_lines + "\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
        + _atom_md(regions={"big": list(range(100))}, n_atoms_total=100)
    )
    (tmp_path / "h2-stage2.fdf").write_text(big_fdf)
    bundle = sb.assemble_from_run_dir(tmp_path)
    assert bundle.source_script.name == "h2-stage2.fdf"
    assert bundle.regions == {"big": list(range(100))}
    assert any("multiple .fdf" in n for n in bundle.notes)


# --------------------------------------------------------------------- #
#  write_bundle_as_handoff (PR-D)                                       #
# --------------------------------------------------------------------- #


def _h2_bundle():
    """Build a minimal RunBundle in-memory for materializer tests."""
    from pathlib import Path as _Path
    s = Structure(elements=["H", "H"],
                  positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    return sb.RunBundle(
        structure=s,
        regions={"L-electrode": [0], "bridge": [1]},
        frozen_atoms=[1],
        user_custom_lines=[],
        provenance={"generator-version": "test"},
        source_script=_Path("/tmp/origin.fdf"),
        source_engine="siesta",
        final_coords_from="xv",
        notes=[],
    )


def test_write_bundle_as_handoff_writes_both_files(tmp_path):
    bundle = _h2_bundle()
    xyz_path, sidecar_path = sb.write_bundle_as_handoff(
        bundle, tmp_path, stem="handoff",
    )
    assert xyz_path  == tmp_path / "handoff.xyz"
    assert sidecar_path == tmp_path / "handoff.molstruct.json"
    assert xyz_path.exists() and sidecar_path.exists()
    # XYZ comment carries the source provenance.
    text = xyz_path.read_text()
    assert "bundled from origin.fdf" in text
    assert "(siesta/xv)" in text


def test_write_bundle_as_handoff_sidecar_carries_labels(tmp_path):
    from molbuilder.parsers import molstruct_json as _msj
    bundle = _h2_bundle()
    _, sidecar_path = sb.write_bundle_as_handoff(
        bundle, tmp_path, stem="h2",
    )
    data = _msj.load(sidecar_path)
    assert data["regions"] == {"L-electrode": [0], "bridge": [1]}
    assert data["frozen_atoms"] == [1]
    assert data["n_atoms_total"] == 2
    # Hash matches the .xyz on disk (apply_sidecar_if_possible will
    # verify the same way).
    expected_hash = _msj.sha256_of_file(tmp_path / "h2.xyz")
    assert data["structure_hash"] == expected_hash


def test_write_bundle_as_handoff_refuses_existing_target(tmp_path):
    """overwrite=False (default) raises when the .xyz OR the sidecar
    already lives in target_dir under the same stem."""
    bundle = _h2_bundle()
    (tmp_path / "h2.xyz").write_text("existing\n")
    with pytest.raises(sb.BundleError) as exc:
        sb.write_bundle_as_handoff(bundle, tmp_path, stem="h2")
    assert "exists" in str(exc.value)


def test_write_bundle_as_handoff_overwrite_true_replaces(tmp_path):
    bundle = _h2_bundle()
    (tmp_path / "h2.xyz").write_text("placeholder\n")
    (tmp_path / "h2.molstruct.json").write_text("{}\n")
    xyz_path, _ = sb.write_bundle_as_handoff(
        bundle, tmp_path, stem="h2", overwrite=True,
    )
    # Replaced — placeholder text is gone.
    assert "placeholder" not in xyz_path.read_text()


def test_write_bundle_as_handoff_creates_missing_target_dir(tmp_path):
    bundle = _h2_bundle()
    target = tmp_path / "nested" / "deeper"
    xyz_path, sidecar_path = sb.write_bundle_as_handoff(
        bundle, target, stem="h2",
    )
    assert target.exists()
    assert xyz_path.exists() and sidecar_path.exists()


# --------------------------------------------------------------------- #
#  L3 round-trip: assemble -> write -> reload -> labels recovered       #
# --------------------------------------------------------------------- #


def test_l3_roundtrip_siesta_labels_survive_handoff(tmp_path):
    """End-to-end: a SIESTA run dir with .fdf+ATOM-METADATA + .XV is
    bundled, materialized to a target dir, then re-loaded via the
    canonical .xyz load path.  Labels survive intact.  Proves the
    in-body-wins-over-sidecar rule is end-to-end consistent: the
    new sidecar IS the only label source after handoff."""
    from molbuilder.parsers import molstruct_json as _msj

    # Source SIESTA dir.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "h2.fdf").write_text(
        _h2_fdf_text(
            atom_md_block=_atom_md(
                regions={"L-electrode": [0]},
                frozen=[1],
                n_atoms_total=2,
            ),
        )
    )
    (src_dir / "h2.XV").write_text(_h2_xv_text())

    # Assemble -> materialize.
    bundle = sb.assemble_from_run_dir(src_dir)
    dst_dir = tmp_path / "dst"
    xyz_path, sidecar_path = sb.write_bundle_as_handoff(
        bundle, dst_dir, stem="handoff",
    )

    # Re-load: pretend this is the next workflow stage opening the
    # bundle.  XYZ -> Structure, then sidecar applies regions/frozen.
    s = Structure.from_xyz(xyz_path.read_text())
    sidecar_data = _msj.load(sidecar_path)
    _msj.apply_to_structure(s, sidecar_data)
    assert s.regions == {"L-electrode": [0]}
    assert s.frozen_atoms == [1]


def test_l3_roundtrip_pyscf_labels_survive_handoff(tmp_path):
    """Mirror of the SIESTA L3, for the PySCF assembler branch."""
    from molbuilder.parsers import molstruct_json as _msj

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "h2.py").write_text(_h2_py_text())  # JOB="h2-test" + atom-md
    (src_dir / "h2-test_optimized.xyz").write_text(_h2_optimized_xyz(0.80))

    bundle = sb.assemble_from_run_dir(src_dir)
    dst_dir = tmp_path / "dst"
    xyz_path, sidecar_path = sb.write_bundle_as_handoff(
        bundle, dst_dir, stem="handoff",
    )

    s = Structure.from_xyz(xyz_path.read_text())
    sidecar_data = _msj.load(sidecar_path)
    _msj.apply_to_structure(s, sidecar_data)
    # _h2_py_text labels: regions={"R-electrode":[1]}, frozen=[0].
    assert s.regions == {"R-electrode": [1]}
    assert s.frozen_atoms == [0]
    # Optimized coords (0.80) survived too.
    import numpy as np
    np.testing.assert_allclose(s.positions[1, 0], 0.80)


