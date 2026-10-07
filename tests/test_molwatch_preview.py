"""Initial-state ``.molwatch.log`` preview emission.

These tests cover the standalone helper used by SIESTA-path generation.

The contract these guard against is:  *the user must see the
initial molecular structure the moment they load the file in
molwatch* -- they must not have to wait for the engine to start
producing native output.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.trajectory_log import write_initial_preview
from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  trajectory_log.write_initial_preview                                 #
# --------------------------------------------------------------------- #


@pytest.fixture
def water_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0, 0, 0], [0.957, 0, 0], [-0.24, 0.927, 0]]),
        title="water",
    )


def test_preview_helper_writes_header(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="water", engine="siesta")
    text = p.read_text()
    # Format-detection marker the molwatch parser sniffs for
    assert text.startswith("# molwatch trajectory log v1")
    # Engine line drives molwatch's source_format
    assert re.search(r"^# engine:\s*siesta\s*$", text, re.MULTILINE)
    # Job line
    assert re.search(r"^# job:\s*water\s*$", text, re.MULTILINE)
    # Units declaration
    assert "energy=eV, force=eV/Ang, coords=Ang" in text


def test_preview_helper_one_block_with_all_atoms(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="water", engine="siesta")
    text = p.read_text()
    # Exactly one step block
    assert text.count("==== molwatch step 0 begin ====") == 1
    assert text.count("==== molwatch step 0 end ====") == 1
    # Coordinates section has all three atoms
    coord_block = text.split("coordinates (Ang):", 1)[1].split("energy", 1)[0]
    coord_lines = [ln for ln in coord_block.splitlines() if ln.strip()]
    assert len(coord_lines) == 3
    # Each line starts with the element symbol followed by 3 floats
    for line, el in zip(coord_lines, ["O", "H", "H"]):
        toks = line.split()
        assert toks[0] == el
        assert len(toks) >= 4


def test_preview_helper_marks_kind_and_nulls(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="w", engine="siesta")
    text = p.read_text()
    # The `kind: initial_preview` line lets a downstream consumer
    # distinguish a preview-only block from a real opt step.
    assert "kind: initial_preview" in text
    # Energy / max_force are explicitly None so the parser maps to null.
    assert "energy (eV): None" in text
    assert "max_force (eV/Ang): None" in text
    # An empty scf_history sub-block (begin immediately followed by end)
    assert re.search(r"scf_history begin\s*\n\s*scf_history end",
                     text, re.MULTILINE)


# --------------------------------------------------------------------- #
#  Cross-repo round-trip: molwatch parser reads the preview block       #
# --------------------------------------------------------------------- #


def test_molwatch_can_parse_siesta_preview(tmp_path):
    """The .molwatch.log emitted by molbuilder must be loadable by
    molwatch's MolwatchLogParser, exposing the initial geometry as
    frame 0 with null energy and empty forces.  This is the cross-repo
    contract: molbuilder writes, molwatch reads."""
    from molbuilder.cell import to_engine
    from molbuilder.parse.engines.molwatch import MolwatchLogParser

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(s, p, job="h2", engine="siesta")
    assert MolwatchLogParser.can_parse(str(p))
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    assert len(result["frames"]) == 1
    # What was written is the ENGINE's frame, like every step after it
    # (`model/structure-periodicity.md` § 6.0), at the log's 8 decimals.
    frame0 = result["frames"][0]
    assert [a[0] for a in frame0] == ["H", "H"]
    np.testing.assert_allclose([a[1:] for a in frame0],
                               to_engine(s).positions, atol=1e-8)
    assert result["energies"] == [None]
    assert result["max_forces"] == [None]
    assert result["forces"] == [[]]
    assert result["scf_history"] == [[]]
    assert result["source_format"] == "siesta"
