"""Shared fixtures for tests/validation/ — the per-submodule
validator test files.  Per docs/protocols/test-strategy.md § 3, the
tests/validation/ tree mirrors molbuilder/validation/.  The
water_struct fixture + _vacuum_cell helper are used by tests across
geometry, metadata, chemistry, sidecar, siesta, and pyscf modules.

Split from the pre-2026-06-13 flat tests/test_validation.py (1479 LoC)
on 2026-06-13 — see commit log.  The split is purely organisational;
no test body or fixture was modified.
"""


from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def water_struct():
    # A planar molecule: give it a real per-side vacuum so render_fdf's
    # derived cell isn't degenerate on the out-of-plane axis (vacuum=0 would
    # make a zero-thickness box -> hard error; structure-periodicity.md).
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        title="water",
        vacuum=(12.0, 12.0, 12.0),
    )




# --------------------------------------------------------------------- #
#  Issue dataclass invariants                                           #
# --------------------------------------------------------------------- #


