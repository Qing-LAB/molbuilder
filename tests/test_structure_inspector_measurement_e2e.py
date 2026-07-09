"""End-to-end pins for the /results Structure inspector's molview module
(atom-annotations.md § 6.3 / § 6.4).

Covers the shipped module surfaces (the old triple-pick measurement CHIP was
retired; this file was renamed off it):
  * the measurement OVERLAY — selection-driven xyz / distance / angle (§ 6.4);
  * viewer clicks feeding the ephemeral store (decision A);
  * B0 — an extended-XYZ ``Lattice=`` line reaching the viewer (getLattice);
  * B1 — enabling k-grid tiles the supercell (atom count 2 -> 4 -> 2).

Drives the ephemeral store via the ``__molbuilder_test_store`` hook on the viewer
slot (selection + setKgrid).  Function-scoped flask_server so each test can
register its own ``tmp_path`` as a Capabilities picker root — module-scoped
servers can't see test-time monkeypatches.
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


def _write_xyz_with_cell_sidecar(tmp_path, cell):
    """A .xyz + its .molstruct.json sidecar carrying a 3x3 `cell` (the dataset
    the host reads periodicity from -- structure-periodicity.md § 8)."""
    import hashlib
    import json
    xyz = tmp_path / "periodic.xyz"
    xyz.write_text("2\nperiodic\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n")
    (tmp_path / "periodic.molstruct.json").write_text(json.dumps({
        "schema_version": 3, "n_atoms_total": 2,
        "structure_hash": hashlib.sha256(xyz.read_bytes()).hexdigest(),
        "regions": {}, "frozen_atoms": [], "cell": cell, "selection_rules": {},
    }))
    return xyz


def test_sidecar_cell_reaches_viewer_and_kgrid_tiles(
        page, flask_server, tmp_path, monkeypatch):
    """Phase 1 (structure-periodicity.md): a .xyz whose `.molstruct.json` sidecar
    carries a `cell` -> the host reads it server-side (/api/selection/atoms), hands
    it to the viewer -> `getLattice()` returns it (unit-cell box) AND k-grid tiles
    the supercell.  The viewer never parses; the cell rides the dataset."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = _write_xyz_with_cell_sidecar(tmp_path, [[10, 0, 0], [0, 10, 0], [0, 0, 20]])
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    slot = ".structure-viewer-slot"

    # the shared view-controls bar (isolate / k-grid toggles) renders in THIS card too.
    assert page.locator(f"{slot} .viewer-toggles .vc-isolate").count() == 1
    assert page.locator(f"{slot} .viewer-toggles .vc-kgrid").count() == 1

    # the cell reached the viewer (box can draw)
    lat = page.evaluate(
        f"() => document.querySelector('{slot}').__molbuilder_test_handle.getLattice()"
    )
    assert lat is not None, "sidecar cell should reach getLattice()"

    # k-grid tiles: 2 atoms -> 2x1x1 -> 4 -> disable -> 2
    _count = (f"() => document.querySelector('{slot}')"
              "  .__molbuilder_test_handle.getAtomCount()")
    assert page.evaluate(_count) == 2
    page.evaluate(
        f"() => {{ const st = document.querySelector('{slot}').__molbuilder_test_store;"
        "  st.setKgrid({ dims: [2, 1, 1] }); st.setKgrid({ enabled: true }); }"
    )
    page.wait_for_function(f"{_count} === 4", timeout=8000)
    page.evaluate(
        f"() => document.querySelector('{slot}')"
        "  .__molbuilder_test_store.setKgrid({ enabled: false })"
    )
    page.wait_for_function(f"{_count} === 2", timeout=8000)


def test_results_view_controls_bar_drives_store(
        page, flask_server, tmp_path, monkeypatch):
    """The shared view-controls bar (.vc-isolate / .vc-kgrid, molview.mountViewControls)
    drives the CARD's OWN store on Results too -- not just Modify.  Clicking the bar's
    k-grid toggle enables tiling; clicking isolate flips the isolate flag.  (The sidecar
    test above drives the same store via its API; THIS proves the bar's CLICKS reach it.)"""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = _write_xyz_with_cell_sidecar(tmp_path, [[10, 0, 0], [0, 10, 0], [0, 0, 20]])
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    slot = ".structure-viewer-slot"
    _count = (f"() => document.querySelector('{slot}')"
              "  .__molbuilder_test_handle.getAtomCount()")
    assert page.evaluate(_count) == 2
    # dims come from the card's own store here; set them, then CLICK the bar to enable.
    page.evaluate(
        f"() => document.querySelector('{slot}')"
        "  .__molbuilder_test_store.setKgrid({ dims: [2, 1, 1] })"
    )
    page.locator(f"{slot} .viewer-toggles .vc-kgrid").check()
    page.wait_for_function(
        f"() => document.querySelector('{slot}')"
        "  .__molbuilder_test_store.getState().kgrid.enabled === true")
    page.wait_for_function(f"{_count} === 4", timeout=8000)   # the bar click tiled 2 -> 4
    # isolate: select an atom, then CLICK the bar's isolate toggle -> store.isolate flips.
    page.evaluate(
        f"() => document.querySelector('{slot}').__molbuilder_test_store.set([0])")
    page.locator(f"{slot} .viewer-toggles .vc-isolate").check()
    page.wait_for_function(
        f"() => document.querySelector('{slot}')"
        "  .__molbuilder_test_store.getState().isolate === true")


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
