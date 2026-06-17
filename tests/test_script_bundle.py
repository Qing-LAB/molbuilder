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


def test_assemble_from_run_dir_pyscf_still_deferred(tmp_path):
    """PR-C lands PySCF.  Until then, surface a clear error citing
    the follow-on task."""
    (tmp_path / "h2.py").write_text("# pyscf placeholder\n")
    with pytest.raises(sb.BundleError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "PR-C" in str(exc.value) or "#490" in str(exc.value)


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


def test_write_bundle_as_handoff_is_reserved_stub(tmp_path):
    """Same reasoning as the assembler stub: the seat exists, the
    impl arrives in PR-D."""
    s = Structure(elements=["H"], positions=[[0.0, 0.0, 0.0]])
    bundle = sb.RunBundle(
        structure=s,
        regions={},
        frozen_atoms=[],
        user_custom_lines=[],
        provenance={},
        source_script=Path("/tmp/x.fdf"),
        source_engine="siesta",
        final_coords_from="fdf-initial",
        notes=[],
    )
    with pytest.raises(NotImplementedError) as exc:
        sb.write_bundle_as_handoff(bundle, tmp_path, stem="x")
    assert "PR-D" in str(exc.value) or "#491" in str(exc.value)
