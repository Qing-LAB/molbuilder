"""Shared pytest fixtures for the molbuilder test suite.

Pytest auto-discovers this file and makes the fixtures it defines
available to every test module under ``tests/``.  Add a fixture here
when more than one test file needs the same setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.structure import Structure


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Absolute path to ``tests/data/`` (PDB / XYZ fixtures live here)."""
    return Path(__file__).parent / "data"


@pytest.fixture
def water_structure() -> Structure:
    """Tiny three-atom Structure (H2O) used by several modules."""
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        atom_names=["O", "H1", "H2"],
        residue_ids=[1, 1, 1],
        residue_names=["HOH", "HOH", "HOH"],
        chain_ids=["A", "A", "A"],
        title="water",
    )


@pytest.fixture
def deprotonated_diester() -> Structure:
    """Synthetic R-O-P(=O)(O-)-O-R' phosphate diester with no Hs on
    either non-bridging oxygen.  Heuristic charge -> -1.

    Used by the chemistry / siesta / pyscf charge-detection tests.
    """
    elements  = ["C", "O", "P", "O", "O", "O", "C"]
    positions = np.array([
        [-2.5, 0.0, 0.0],   # C5'
        [-1.4, 0.0, 0.0],   # O5' (bridge)
        [ 0.0, 0.0, 0.0],   # P
        [ 0.0, 1.5, 0.0],   # OP1 (non-bridging)
        [ 0.0, -0.8, 1.3],  # OP2 (non-bridging)
        [ 1.4, 0.0, 0.0],   # O3' (bridge)
        [ 2.5, 0.0, 0.0],   # C3'
    ])
    return Structure(
        elements=elements, positions=positions,
        atom_names=["C5'", "O5'", "P", "OP1", "OP2", "O3'", "C3'"],
        residue_ids=[1] * 7, residue_names=["DA"] * 7, chain_ids=["A"] * 7,
    )


@pytest.fixture
def web_client():
    """Flask test client; skips the test if Flask isn't installed."""
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    app = create_app()
    return app.test_client()
