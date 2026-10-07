"""The Frame dataclass takes plain lists for its arrays.

Spec: docs/design.md "Frame and Trajectory (parser output type)".
"""

from __future__ import annotations

import numpy as np

from molbuilder.frame import Frame
from molbuilder.structure import Structure


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
