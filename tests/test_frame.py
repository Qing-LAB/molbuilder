"""The Frame dataclass takes plain lists for its arrays.

Spec: docs/design.md "Frame and Trajectory (parser output type)".  The
Trajectory tests that parsed outputs typed by hand were retired 2026-10-04
(`process/testing.md` § 6).
"""

from __future__ import annotations

import numpy as np

from molbuilder.frame import Frame
from molbuilder.structure import Structure


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 7 tests here parsed a progress log, a SIESTA output or a
# geomeTRIC trajectory typed by hand (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  Frame.scf_history: None vs []                                         #
#                                                                        #
#  The two values are NOT interchangeable:                              #
#    None  = the parser tracks no SCF data for this run at all           #
#            (e.g. PySCF .log absent, SIESTA file with no scf: lines).   #
#    []    = the parser DID track scf data for this step but there      #
#            were no cycles to record (e.g. a molwatch initial-state     #
#            preview block, which by spec carries an empty SCF section). #
#                                                                        #
#  trajectory_to_legacy_dict at the web boundary collapses an "all       #
#  None" trajectory back to a top-level [], preserving the legacy JSON   #
#  shape that the JS client uses to decide whether to hide the SCF       #
#  panel.  These tests pin both sides of the contract.                   #
# --------------------------------------------------------------------- #


def test_frame_post_init_coerces_list_inputs():
    """Frame accepts forces / lattice as plain lists; __post_init__
    upgrades them to ndarrays so downstream code sees consistent
    types."""
    s = Structure(elements=["H"],
                  positions=np.array([[0.0, 0.0, 0.0]]))
    f = Frame(
        structure  = s,
        step_index = 0,
        forces     = [[0.1, 0.2, 0.3]],
        lattice    = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert isinstance(f.forces, np.ndarray)
    assert f.forces.shape == (1, 3)
    assert isinstance(f.lattice, np.ndarray)
    assert f.lattice.shape == (3, 3)
