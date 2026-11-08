"""One rule for the format of a text with no name, asked by the build page's
load route and by the command line's stdin (`StructureCodec.format_of_text`):
an XYZ's first non-blank line is its atom count; anything else is a PDB,
whose header lines can push its first atom far into the text.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.workingcopy_structure import StructureCodec
from support.road import molbuilder

#: Each row: the format a build writes the structure in, as it writes it.
_ROWS = [
    dict(name="a PDB, its header before its first atom", fmt="pdb"),
    dict(name="an XYZ, its count first", fmt="xyz"),
]


@pytest.mark.parametrize("row", _ROWS, ids=[r["name"] for r in _ROWS])
def test_a_structure_piped_in_is_read_in_its_format(row, tmp_path):
    """A structure `molbuilder peptide` wrote, piped into `molbuilder modify -`
    as a person pipes one: read in its own format, the edit made on the
    atoms it holds.  THE SILENT FAILURE: the text taken for the other
    format -- a PDB whose header comes first read as an XYZ, an XYZ read
    as a PDB -- and the pipe refused with a parse error the text does not
    deserve."""
    built = tmp_path / f"ala.{row['fmt']}"
    made = molbuilder("peptide", "A",
                      "--pdb" if row["fmt"] == "pdb" else "--out", built)
    assert made.exit_code == 0, made.output
    out = tmp_path / "edited.xyz"
    piped = molbuilder("modify", "-", out, "--delete", "0", "--yes",
                       input=built.read_text())
    assert piped.exit_code == 0, piped.output
    was, got = StructureCodec().load(built), StructureCodec().load(out)
    assert list(got.elements) == list(was.elements)[1:]
    assert np.allclose(got.positions, was.positions[1:], atol=1e-3)
