"""Shared plain-function helpers for tests/validation/.

Plain Python helpers (NOT pytest fixtures) used across the
per-submodule validator tests.  Lives at ``tests/validation/_helpers.py``
so imports are explicit (``from ._helpers import _vacuum_cell``);
pytest's conftest fixture mechanism doesn't expose plain functions.

Conventionally underscore-prefixed: they're test-internal, not part
of any public surface, and the prefix discourages downstream callers
from importing them.
"""

from __future__ import annotations

import numpy as np

from molbuilder.structure import Structure


def _vacuum_cell(size: float = 30.0) -> np.ndarray:
    """A simple cubic cell.  30 Å is a typical vacuum-padded box.

    Used by geometry / siesta tests that need a cell argument to
    ``validate()`` without exercising the cell-shape rules.
    """
    return np.eye(3) * size


def _peptide_struct(residue_names):
    """Synthetic peptide: one CA atom per residue, no actual chemistry,
    just enough metadata for the validator's residue-name check.

    Used by chemistry-rule peptide-protonation tests + the SIESTA
    aggregator's peptide hint.
    """
    n = len(residue_names)
    pos = np.column_stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)])
    return Structure(
        elements=["C"] * n,
        positions=pos,
        atom_names=["CA"] * n,
        residue_ids=list(range(1, n + 1)),
        residue_names=list(residue_names),
    )
