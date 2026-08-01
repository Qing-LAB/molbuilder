"""The fdf script generator, round-tripped: build -> render -> read back.

A DIFFERENT ROUND TRIP FROM THE SIDECAR'S.  ``test_junction_sidecar_roundtrip``
covers the ``.xyz`` + ``.molstruct.json`` pair -- what the paired-file door
writes and reads.  This one covers the **generated engine script**, which is a
separate API with a separate obligation, and the obligation is doubled:

**A frozen constraint has to be in the fdf twice, and the two are not
alternatives.**

  * in the SETUP -- ``%block Geometry.Constraints`` -- because that is what
    SIESTA reads.  Without it the atoms move, whatever any comment says.
  * in the COMMENTS -- the ``ATOM-METADATA`` block -- because that is the
    round-trippable copy of the data model.  Without it a finished run cannot be
    read back into the structure it came from: the Results tab, the bundle
    parser and any re-derivation lose which atoms were held.

One without the other is a half-written file.  The setup alone runs correctly
and cannot be reopened; the comments alone reopen cleanly and compute the wrong
physics.

NO EXTERNAL FIXTURE.  The junction is built in source (``tests/support``), so
this exercises the writer and the reader as they are today rather than trusting
a captured artefact -- see that module's docstring for why the captured ones
went stale without anyone noticing.
"""
from __future__ import annotations

import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.parse.scripts.atom_metadata import AtomMetadataTextParser
from molbuilder.sidecars import molstruct
from molbuilder.siesta.input import render_fdf

from support.junction import build_junction, frozen, regions


def _constrained(fdf: str) -> list:
    """The atoms the SETUP actually holds still, 0-based.

    Read from the ``position`` line of the real block, not by scraping digits
    out of the region around it -- the explanatory comment above the block says
    "1-based" and "0-based", and a regex over the whole slice picks those up as
    atom numbers.
    """
    for line in fdf.splitlines():
        if line.strip().startswith("position "):
            return sorted(int(n) - 1 for n in line.split()[1:])
    return []


@pytest.fixture
def fdf() -> str:
    return render_fdf(build_junction(), SiestaConfig(system_label="junction"))


# --------------------------------------------------------------------- #
#  1. The SETUP -- what the engine actually reads                       #
# --------------------------------------------------------------------- #

def test_the_frozen_atoms_are_constrained_in_the_engine_setup(fdf):
    """``Geometry.Constraints`` is the only thing that holds an atom still.

    1-BASED on the way out (model/overview.md § 2): the structure's atom 0 is
    written as 1, and the translation happens in one place rather than at each
    call site.
    """
    assert "%endblock Geometry.Constraints" in fdf, (
        "no constraints block -- the electrodes would relax")
    assert _constrained(fdf) == frozen(), (
        f"the constrained set is not the frozen set: {_constrained(fdf)}")


# --------------------------------------------------------------------- #
#  2. The COMMENTS -- the round-trippable copy of the data model        #
# --------------------------------------------------------------------- #

def test_the_whole_label_store_survives_into_the_comments(fdf):
    """Every label, read back through the project's own parser rather than by
    hand -- a test that re-implements the reader tests the test."""
    parsed = AtomMetadataTextParser.parse(fdf).atom_metadata
    assert parsed is not None, "the ATOM-METADATA block is missing entirely"
    assert set(parsed["regions"]) == set(regions()), (
        f"labels lost on the way into the script: {sorted(parsed['regions'])}")
    for name, atoms in regions().items():
        assert parsed["regions"][name] == sorted(atoms), f"{name} changed"


def test_the_frozen_atoms_come_back_through_the_designated_accessor(fdf):
    """The reserved label is an ORDINARY label in the store; what makes it
    reserved is this accessor and the interpretation applied in the setup
    block above. No caller spells the name itself."""
    parsed = AtomMetadataTextParser.parse(fdf).atom_metadata
    assert molstruct.frozen_atoms(parsed) == frozen()
    assert "frozen_atoms" not in parsed, (
        "a second top-level store came back -- the label lives in `regions`")


def test_the_comments_and_the_setup_agree(fdf):
    """The two halves describe one fact, so they cannot be allowed to drift.

    This is the assertion that would have caught the whole class of defect: a
    generator that emits the constraint and forgets the metadata, or updates one
    store and not the other, fails here rather than in somebody's run.
    """
    parsed = AtomMetadataTextParser.parse(fdf).atom_metadata
    assert _constrained(fdf) == molstruct.frozen_atoms(parsed), (
        f"the setup holds {_constrained(fdf)} still while the comments record "
        f"{molstruct.frozen_atoms(parsed)}")


# --------------------------------------------------------------------- #
#  3. The version the block claims                                      #
# --------------------------------------------------------------------- #

def test_the_metadata_block_states_the_schema_it_is_actually_written_in(fdf):
    """THE OPEN DEFECT (2026-07-31), pinned here rather than described.

    ``script_emit.emit_atom_metadata`` hardcodes ``"schema_version": 4`` and a
    ``format: molstruct-json/v4`` line, while emitting the CURRENT shape -- its
    own docstring says "regions is the whole label store, so a reserved label
    is IN it rather than beside it", which is v7's rule, not v4's.

    So the block claims a version it is not written in, and the cost is exact:
    a script generated BEFORE the label-store change and one generated after it
    both say v4 while holding different shapes. Nothing can tell them apart, so
    a reader cannot refuse the old one -- which is how a real run's fifty frozen
    electrode atoms came back as an empty list, with the file looking fine.

    The fix is one constant rather than a literal: stamp what the sidecar
    stamps, so the two can never disagree about what version means what.
    """
    result = AtomMetadataTextParser.parse(fdf)
    assert result.block_schema_versions.get("atom-metadata") == \
        molstruct.SCHEMA_VERSION, (
        "the ATOM-METADATA block claims schema_version "
        f"{result.block_schema_versions.get('atom-metadata')} while carrying "
        f"v{molstruct.SCHEMA_VERSION} content -- a version claim that cannot "
        "be trusted is worse than none, because a reader cannot refuse what it "
        "cannot recognise")
