"""Shared pytest fixtures for the molbuilder test suite.

Pytest auto-discovers this file and makes the fixtures it defines
available to every test module under ``tests/``.  Add a fixture here
when more than one test file needs the same setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder import diagnostics
from molbuilder.structure import Structure


@pytest.fixture(autouse=True)
def _reset_diagnostics_singleton():
    """Every test starts AND ends with no bound Capabilities snapshot.

    Without this, a test that calls ``cli.main()`` or otherwise
    triggers ``diagnostics.initialize()`` would leak its snapshot into
    whatever runs next, creating order-dependent failures.  Tests that
    want a specific snapshot inject it via
    :func:`molbuilder.diagnostics.set_capabilities`; that injection is
    cleaned up by this fixture afterwards.
    """
    diagnostics.reset_capabilities()
    yield
    diagnostics.reset_capabilities()


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
    """Flask test client; skips the test if Flask isn't installed.

    Passes ``config={}`` explicitly so ``create_app`` does NOT read
    the repo-root ``molbuilder.json`` (which may have auth/TLS
    enabled in a developer's working tree).  Tests against the
    no-auth/no-TLS default isolate the page-render + API surface
    from per-machine config state.  Tests that need an auth-enabled
    or otherwise non-default app build their own fixture and pass
    the matching ``config`` dict.
    """
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    # rate_limit.enabled=false in the default test client because
    # several test files fire 60+ requests within the rolling
    # window and would otherwise hit the total-burst threshold.
    # tests/test_rate_limit.py builds its own client with the
    # limiter enabled to exercise the module directly.
    app = create_app(config={"rate_limit": {"enabled": False}})
    return app.test_client()


# --------------------------------------------------------------------- #
#  Marker auto-application by file pattern (2026-06-24)                 #
#  Implements the marker discipline documented in                       #
#  docs/protocols/test-strategy.md § 2.  Tests can override at the     #
#  function level by carrying an explicit @pytest.mark.<X>.            #
# --------------------------------------------------------------------- #

def pytest_collection_modifyitems(config, items):
    """Auto-apply ``e2e`` to ``*_e2e.py`` files + ``integration`` to
    files that subprocess-run a real engine binary (siesta /
    transiesta / pyscf).

    The pyproject.toml markers list registered ``unit / module /
    interface / integration / smoke / e2e / slow`` but as of the
    2026-06-24 audit zero tests carried any of them, so
    ``pytest -m integration`` returned nothing.  This hook gives
    every existing test file the right baseline marker so the doc-
    promised selectors work, without requiring per-test annotation.
    File-name pattern is a coarse pre-classifier; finer-grained
    decisions still belong on individual tests via explicit
    decorators.
    """
    import pytest as _pt
    for item in items:
        fn = item.fspath.basename
        if fn.endswith("_e2e.py") or "_e2e_" in fn:
            item.add_marker(_pt.mark.e2e)
        if "_smoke" in fn or "_smoke_l4" in fn:
            item.add_marker(_pt.mark.integration)
            item.add_marker(_pt.mark.smoke)
