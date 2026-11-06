"""Initial-state ``.molwatch.log`` preview emission.

This test covers the standalone helper used by SIESTA-path generation.

The contract it guards is:  *the user must see the
initial molecular structure the moment they load the file in
molwatch* -- they must not have to wait for the engine to start
producing native output.
"""

from __future__ import annotations

import numpy as np

from molbuilder.trajectory_log import write_initial_preview
from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.structure import Structure


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
