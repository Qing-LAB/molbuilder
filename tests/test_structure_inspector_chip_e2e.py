"""End-to-end pin for the /results Structure inspector's
measurement chip (xyz / distance / angle).

Task #300 added click-pick + halo + chip on the structure
inspector but shipped without a Playwright e2e — the chip's
behaviour was only verified by the matching Modify-tab test.
Task #307 closes the gap.

Drives picks via the embed handle's ``setPickedIndices`` (the
chip's ``onPick`` consumes that the same way a real canvas click
would).  Function-scoped flask_server so each test can register
its own ``tmp_path`` as a Capabilities picker root — module-
scoped servers can't see test-time monkeypatches.
"""
from __future__ import annotations

import threading

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture
def flask_server():
    """Per-test Flask server so each test's tmp_path picker-root
    monkeypatch reaches /api/files/read."""
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Register ``tmp_path`` as a Capabilities picker root so the
    inspector's ctx.readFile can resolve files under it."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


def _open_results(page, base_url):
    """Navigate to /results and wait for the inspector registry
    to finish self-registration."""
    page.goto(f"{base_url}/results", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder "
        "&& window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4",
        timeout=8000,
    )


def _mount_structure(page, file_path):
    """Mount the structure inspector via the registry API + wait
    for the embed handle to land on the slot's test hook.  The
    test hook (``viewerSlot.__molbuilder_test_handle``) was added
    in task #307; no production reader."""
    page.evaluate(
        "(file) => {"
        "  const host = document.getElementById('inspector-host');"
        "  const reg  = window.molbuilder.inspectors;"
        "  const ctx  = reg.createDefaultContext(host);"
        "  window._handle = reg.mount(host, file, ctx);"
        "}",
        str(file_path),
    )
    # Wait for the ephemeral store hook (set in onReady, after the atoms are
    # adopted + the measurement overlay is mounted).
    page.wait_for_function(
        "() => {"
        "  const slot = document.querySelector('.structure-viewer-slot');"
        "  return slot && slot.__molbuilder_test_store;"
        "}",
        timeout=10000,
    )


def _select(page, indices):
    """Drive the ephemeral store selection (input order becomes pickOrder), which
    drives the measurement overlay (§ 6.4) -- decision A: measurement = f(selection)."""
    page.evaluate(
        "(indices) => document.querySelector('.structure-viewer-slot')"
        "  .__molbuilder_test_store.set(indices)",
        list(indices),
    )


# The measurement readout is now the module overlay, not a #structure-measurement chip.
_OVL = ".molview-measurement-overlay"


def test_measurement_overlay_renders_xyz_distance_angle_via_selection(
        page, flask_server, tmp_path, monkeypatch):
    """The measurement overlay (§ 6.4) is SELECTION-driven: 1 atom → xyz, 2 →
    distance, 3 → angle (vertex = 2nd SELECTED, via pickOrder), 0 → hidden.
    Right-triangle geometry so the geometric vertex ≠ the middle-selection vertex
    (guards the pickOrder false-positive that hid task #304)."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "right_triangle.xyz"
    xyz.write_text(
        "3\nright-triangle\n"
        "C 1.000 0.000 0.000\n"
        "N 0.000 0.000 0.000\n"
        "O 0.000 1.000 0.000\n"
    )
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))

    import re
    # 1 atom → xyz of the C atom (+x).
    _select(page, [0])
    page.wait_for_function(
        f"() => {{ const o = document.querySelector('{_OVL}'); return o && !o.hidden; }}"
    )
    text = page.locator(_OVL).inner_text()
    assert text.startswith("C #1"), text
    assert "(1.000, 0.000, 0.000)" in text, text
    assert page.evaluate(f"() => document.querySelector('{_OVL}').dataset.kind") == "xyz"

    # 2 atoms → distance C–O = sqrt(2) ≈ 1.4142.
    _select(page, [0, 2])
    page.wait_for_function(f"() => document.querySelector('{_OVL}').dataset.kind === 'distance'")
    text = page.locator(_OVL).inner_text()
    assert "C #1" in text and "O #3" in text, text
    m = re.search(r"=\s*([0-9.]+)\s*Å", text)
    assert m and 1.40 <= float(m.group(1)) <= 1.43, text

    # 3 atoms → angle.  Selection order [O, N, C] = [2, 1, 0]:
    # vertex is N (origin, the 2nd selected) → 90°.
    _select(page, [2, 1, 0])
    page.wait_for_function(f"() => document.querySelector('{_OVL}').dataset.kind === 'angle'")
    text = page.locator(_OVL).inner_text()
    m = re.match(r"∠\s*(\S+\s*#\d+)\s*–\s*(\S+\s*#\d+)\s*–\s*(\S+\s*#\d+)", text)
    assert m, text
    assert m.group(2).startswith("N "), (
        f"vertex should be N (2nd selected); got {m.group(2)!r} in {text!r}"
    )
    deg = re.search(r"=\s*([0-9.]+)°", text)
    assert deg and 89.0 <= float(deg.group(1)) <= 91.0, text

    # 0 atoms → overlay hides.
    _select(page, [])
    page.wait_for_function(f"() => document.querySelector('{_OVL}').hidden")


def test_extxyz_lattice_reaches_the_viewer(
        page, flask_server, tmp_path, monkeypatch):
    """B0 (§ 6.3): an extended-XYZ Lattice= line is parsed by the inspector and
    passed as opts.lattice, so the embed's getLattice() returns the cell — the
    prerequisite for k-grid tiling.  A plain .xyz (no Lattice=) yields null."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "periodic.xyz"
    xyz.write_text(
        '2\n'
        'Lattice="10 0 0 0 10 0 0 0 20" Properties=species:S:1:pos:R:3\n'
        'C 0 0 0\n'
        'H 1 0 0\n'
    )
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    lat = page.evaluate(
        "() => document.querySelector('.structure-viewer-slot')"
        "  .__molbuilder_test_handle.getLattice()"
    )
    assert lat is not None, "getLattice() should return the parsed cell"


def test_viewer_clicks_are_wired_to_the_store(
        page, flask_server, tmp_path, monkeypatch):
    """Under decision A the adapter wires viewer clicks to the store (single
    pick) instead of a separate triple-pick — the measurement derives from the
    selection.  Pin the embed's pick mode as 'single' (adapter-set)."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "small.xyz"
    xyz.write_text("2\nsmall\nH 0 0 0\nH 1 0 0\n")
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    # The adapter attaches asynchronously via mountPanel; wait for single-pick.
    # Null-safe: getPick() is undefined until the adapter runs setPick.
    page.wait_for_function(
        "() => {"
        "  const h = document.querySelector('.structure-viewer-slot').__molbuilder_test_handle;"
        "  const p = h && h.getPick && h.getPick();"
        "  return p && p.mode === 'single';"
        "}",
        timeout=8000,
    )
