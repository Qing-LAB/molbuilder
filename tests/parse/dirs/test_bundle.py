"""L3 tests for Phase G BundleDirParser.

Pins:
  * BundleDirParser is NOT in the auto-dispatch registry (would
    collide with JobDirParser on shared "directory with .fdf"
    claims; bundle vs job is a different user intent).
  * Explicit BundleDirParser.parse() returns a typed BundleResult
    with structure + regions + frozen_atoms + notes from the
    legacy assemble_from_run_dir round-trip.
  * BundleError from the legacy assembler (ambiguous engines,
    missing scripts, atom-count mismatch) propagates unchanged.
  * BundleResult is frozen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parse import BundleResult
from molbuilder.parse.dirs.bundle import BundleDirParser, BundleError
from molbuilder.parse.registry import _registered_dir_parsers


# NO PATH INTO THE REAL TREE, and no skip helper.  Both are gone (2026-08-03).
#
# A `pytest.skip("fixture dir absent")` is the dangerous half: on any machine
# without that directory the test SKIPS and the run still reads green, so the
# suite reports coverage it does not have.  Every run directory below is built
# in the test from a junction defined in source.


# Dispatch isolation ----------------------------------------------- #


def test_bundle_dir_parser_not_in_dispatch_registry():
    """BundleDirParser is explicit-dispatch; it must NOT appear
    in the auto-registered DirParser pool (else parse_dir on any
    job directory would raise AmbiguousFormatError with
    JobDirParser as the competing claimant)."""
    names = {p.name for p in _registered_dir_parsers()}
    assert "bundle-dir" not in names, (
        "BundleDirParser must not auto-register; bundle vs job "
        "dispatch ambiguity would deadlock parse_dir()")


def test_bundle_can_parse_claims_fdf_dir(tmp_path: Path):
    """can_parse mirrors the legacy assembler's precondition
    (at least one .fdf or .py present)."""
    (tmp_path / "test.fdf").write_text("SystemLabel test\n")
    assert BundleDirParser.can_parse(tmp_path)


def test_bundle_can_parse_rejects_empty_dir(tmp_path: Path):
    assert not BundleDirParser.can_parse(tmp_path)


# Constructed round trip --------------------------------------------- #
#
# NO EXTERNAL FIXTURE.  This used to parse projects/BDT/optimization/
# TJ-BDT-Au111 -- a real run directory -- and assert against numbers captured
# from whatever the code produced the day it was written.  That is an
# unversioned assumption dressed as a test: nothing re-checks the directory,
# and when the metadata format moved it went stale.  It then failed for a
# reason ("your fixture predates the label store") that looks exactly like the
# reason it must not be used for ("your parser is broken"), and telling those
# apart cost a bisect.
#
# So the run directory is BUILT: render the real generator's output for a
# junction defined in source, and read it back through the real parser.  What
# it proves is stronger than the old test's numbers -- that the emit and parse
# halves of the bundle API agree TODAY.


def _run_dir(tmp_path: Path) -> Path:
    """A SIESTA run directory, generated the way the application generates one."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_fdf
    from support.junction import build_junction

    run = tmp_path / "junction-run"
    run.mkdir()
    (run / "junction.fdf").write_text(
        render_fdf(build_junction(), SiestaConfig(system_label="junction")),
        encoding="utf-8")
    return run


def test_parse_returns_a_bundleresult_carrying_what_the_generator_wrote(tmp_path):
    """End-to-end over the emit/parse seam: a generated script yields a
    complete BundleResult, with the label store intact."""
    from support.junction import N_ATOMS, frozen, regions

    result = BundleDirParser.parse(_run_dir(tmp_path))

    assert isinstance(result, BundleResult)
    assert result.result_kind == "bundle"
    assert result.parser_name == "bundle-dir"
    assert result.structure is not None
    assert len(result.structure.elements) == N_ATOMS
    assert set(result.regions) == set(regions())
    assert Path(result.source).is_absolute()
    assert isinstance(result.notes, list)


def test_the_frozen_set_survives_the_generator_and_comes_back(tmp_path):
    """THE REGRESSION THIS FILE EXISTS FOR.

    The bundle path is how a FINISHED RUN is read back into a structure. When
    the atom-metadata block claimed a version it was not written in, this came
    back as an empty list while every label beside it survived -- so a junction
    reopened with its pinned electrodes forgotten, and nothing said a word.
    """
    from support.junction import frozen

    result = BundleDirParser.parse(_run_dir(tmp_path))

    assert list(result.frozen_atoms) == frozen(), (
        f"the frozen set did not survive the round trip: {result.frozen_atoms}")
    # No METADATA diagnostic: a freshly generated block states the version it
    # is written in, so the reader has nothing to warn about. (A note about the
    # absent `.XV` is expected and correct -- this run has no converged
    # geometry, and saying so is the parser doing its job.)
    assert not [n for n in result.notes if "schema_version" in n], (
        f"a freshly generated script raised a metadata diagnostic: "
        f"{result.notes}")



def test_a_script_with_no_atom_metadata_gives_empty_labels(tmp_path: Path):
    """A .fdf carrying no ATOM-METADATA block yields empty regions +
    frozen_atoms -- an empty dict, NOT None -- and still a valid result.

    BUILT, not found.  This read a real run directory
    (projects/BDT/optimization/BDT-withAuJunction) whose only relevant
    property was that it predates the block.  Depending on someone's
    scientific record for "a file without a feature" is a guess about
    relevance dressed as a fixture: it skips silently on any other machine,
    and it changes meaning the day that directory is regenerated or deleted.
    Stripping the block from a generated script states the property directly.
    """
    generated = (_run_dir(tmp_path) / "junction.fdf").read_text(encoding="utf-8")
    start = generated.index("# === molbuilder atom-metadata BEGIN ===")
    end = generated.index("# === molbuilder atom-metadata END ===") + len(
        "# === molbuilder atom-metadata END ===")
    stripped = generated[:start] + generated[end:]
    assert "atom-metadata" not in stripped

    bare = tmp_path / "no-labels"
    bare.mkdir()
    (bare / "junction.fdf").write_text(stripped, encoding="utf-8")

    result = BundleDirParser.parse(bare)
    assert isinstance(result, BundleResult)
    assert result.regions == {}
    assert result.frozen_atoms == []
    assert result.structure is not None, "coordinates still come from the .fdf"


# BundleError propagation ------------------------------------------ #


def test_bundle_error_propagates_on_empty_dir(tmp_path: Path):
    """Legacy assembler raises BundleError on a dir with no
    engine scripts; the DirParser wrapper must propagate
    unchanged (NOT swallow + return empty result)."""
    with pytest.raises(BundleError):
        BundleDirParser.parse(tmp_path)


def test_bundle_error_propagates_on_ambiguous_engines(tmp_path: Path):
    """Both .fdf and .py in one dir is ambiguous; assembler
    raises BundleError."""
    (tmp_path / "calc.fdf").write_text("SystemLabel test\n")
    (tmp_path / "calc.py").write_text("# pyscf\n")
    with pytest.raises(BundleError):
        BundleDirParser.parse(tmp_path)


# Frozen + envelope ---------------------------------------------- #


def test_bundleresult_is_frozen(tmp_path: Path):
    from dataclasses import FrozenInstanceError
    result = BundleDirParser.parse(_run_dir(tmp_path))
    with pytest.raises(FrozenInstanceError):
        result.notes = []   # noqa


def test_bundleresult_parser_name_is_slug(tmp_path: Path):
    """Envelope convention: parser_name is the slug 'bundle-dir',
    not the literal classname."""
    result = BundleDirParser.parse(_run_dir(tmp_path))
    assert result.parser_name == "bundle-dir"
    assert result.parser_name == BundleDirParser.name
