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


def test_assemble_from_run_dir_is_reserved_stub(tmp_path):
    """The signature is reserved by PR-A; the implementation lands
    in PR-B/C.  Calling today must raise NotImplementedError -- not
    a silent stub-returning-None that masquerades as "no bundle"."""
    with pytest.raises(NotImplementedError) as exc:
        sb.assemble_from_run_dir(tmp_path)
    assert "PR-B" in str(exc.value) or "#489" in str(exc.value)


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
