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
        vacuum=(12.0, 12.0, 12.0),
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
        vacuum=(12.0, 12.0, 12.0),
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

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's report on the item so fixtures can tell, at
    teardown, whether the test failed.  Used by the ``capture_on_fail``
    diagnostic harness (a test flagged with @pytest.mark.capture_on_fail
    dumps browser state + console to a PERSISTENT ``test-artifacts/`` dir
    when it fails -- so a rare intermittent E2E failure is inspectable
    after the fact instead of needing a re-repro)."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def _capture_on_fail(request):
    """Persistent failure diagnostics for ``@pytest.mark.capture_on_fail``
    tests (any file).  On failure of a flagged test, dump the browser state
    (viewer/store atom counts, source_file, dirty, mountRestoreTarget, the
    persisted snapshot) + the console to a PERSISTENT ``test-artifacts/``
    dir -- so a rare intermittent E2E is inspectable after the fact without
    a re-repro.  No-op (and no ``page`` dependency) for unflagged tests."""
    marked = request.node.get_closest_marker("capture_on_fail") is not None
    page = None
    msgs: list = []
    if marked:
        try:
            page = request.getfixturevalue("page")
            page.on("console", lambda m: msgs.append(m.text))
        except Exception:
            page = None
    yield
    if not marked:
        return
    rep = getattr(request.node, "rep_call", None)
    if not (rep and rep.failed):
        return
    import json as _json
    import datetime as _dt
    import pathlib as _pl
    art = _pl.Path(__file__).resolve().parent.parent / "test-artifacts"
    art.mkdir(exist_ok=True)
    snap = {"note": "no page for this test"}
    if page is not None:
        try:
            snap = page.evaluate(
                "() => { const w=window.molbuilder||{}, ws=w.workspace,"
                " t=w.__molbuilder_modify_test; let p=null;"
                " try{p=ws&&ws.readPersistedSnapshot&&ws.readPersistedSnapshot();}catch(e){}"
                " const pa=p&&p.state&&p.state.structure&&p.state.structure.atoms;"
                " return { url: location.href,"
                " viewer_n_atoms: t&&t.getNAtoms?t.getNAtoms():null,"
                " store_atoms: ws&&ws.getState?((ws.getState().atoms||[]).length):null,"
                " source_file: ws&&ws.getSourceFile?ws.getSourceFile():null,"
                " dirty: ws&&ws.isDirty?ws.isDirty():null,"
                " mount_restore_target: ws&&ws.mountRestoreTarget?ws.mountRestoreTarget():null,"
                " persisted_n_atoms: pa?pa.length:null,"
                " persisted_dirty: p&&p.state?!!p.state.dirty:null }; }")
        except Exception as e:  # noqa: BLE001
            snap = {"evaluate_error": str(e)}
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = art / f"fail-{request.node.name}-{ts}.log"
    out.write_text(
        "test:  " + request.node.nodeid + "\n\n"
        "PAGE STATE:\n" + _json.dumps(snap, indent=2) + "\n\n"
        "ERROR:\n" + str(rep.longrepr)[:3000] + "\n\n"
        "CONSOLE (last 300):\n" + "\n".join(msgs[-300:]) + "\n",
        encoding="utf-8")
    print(f"[capture_on_fail] diagnostic written -> {out}")


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
