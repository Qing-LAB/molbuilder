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
    page.wait_for_function(
        "() => {"
        "  const slot = document.querySelector('.structure-viewer-slot');"
        "  return slot && slot.__molbuilder_test_handle !== undefined;"
        "}",
        timeout=10000,
    )


def _pick(page, indices):
    """Drive the embed's setPickedIndices AND fire the chip-refresh
    closure manually — the embed deliberately doesn't call
    ``onPick`` from ``setPickedIndices`` so the chip update has to
    be triggered separately when the test bypasses canvas clicks."""
    page.evaluate(
        "(indices) => {"
        "  const slot = document.querySelector('.structure-viewer-slot');"
        "  slot.__molbuilder_test_handle.setPickedIndices(indices);"
        "  slot.__molbuilder_test_refreshChip();"
        "}",
        list(indices),
    )


def test_chip_renders_xyz_distance_and_angle_via_pick(
        page, flask_server, tmp_path, monkeypatch):
    """1 atom → xyz, 2 atoms → distance, 3 atoms → angle with
    vertex = user's 2nd click.  Geometry is a right triangle
    where the geometric vertex (origin) ≠ middle-click vertex
    in the test below — guards against the same false-positive
    that hid the pickOrder snapshot blocker (task #304)."""
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

    # 1 atom → xyz of the C atom (+x).
    _pick(page, [0])
    page.wait_for_function(
        "() => !document.getElementById('structure-measurement').hidden"
    )
    text = page.locator("#structure-measurement").inner_text()
    assert text.startswith("C #1"), text
    assert "(1.000, 0.000, 0.000)" in text, text
    assert page.evaluate(
        "() => document.getElementById('structure-measurement').dataset.kind"
    ) == "xyz"

    # 2 atoms → distance C–O = sqrt(2) ≈ 1.4142.
    _pick(page, [0, 2])
    page.wait_for_function(
        "() => document.getElementById('structure-measurement')"
        "  .dataset.kind === 'distance'"
    )
    text = page.locator("#structure-measurement").inner_text()
    assert "C #1" in text and "O #3" in text, text
    import re
    m = re.search(r"=\s*([0-9.]+)\s*Å", text)
    assert m and 1.40 <= float(m.group(1)) <= 1.43, text

    # 3 atoms → angle.  Click order [O, N, C] = [2, 1, 0]:
    # vertex is N (origin, the 2nd click) → 90°.
    _pick(page, [2, 1, 0])
    page.wait_for_function(
        "() => document.getElementById('structure-measurement')"
        "  .dataset.kind === 'angle'"
    )
    text = page.locator("#structure-measurement").inner_text()
    m = re.match(
        r"∠\s*(\S+\s*#\d+)\s*–\s*(\S+\s*#\d+)\s*–\s*(\S+\s*#\d+)",
        text,
    )
    assert m, text
    assert m.group(2).startswith("N "), (
        f"vertex should be N (2nd click); got {m.group(2)!r} in {text!r}"
    )
    deg = re.search(r"=\s*([0-9.]+)°", text)
    assert deg and 89.0 <= float(deg.group(1)) <= 91.0, text

    # 0 atoms → chip hides.
    _pick(page, [])
    page.wait_for_function(
        "() => document.getElementById('structure-measurement').hidden"
    )


def test_chip_pick_mode_is_triple(
        page, flask_server, tmp_path, monkeypatch):
    """The structure inspector configures the embed with
    ``pick.mode = "triple"`` (task #300).  Pin the mode end-to-end
    so a regression where the inspector silently downgrades to
    ``pair`` or ``single`` surfaces here.  FIFO drop semantics
    are covered by the embed's own JS unit tests
    (tests/test_mol_viewer_embed_js.py) — they exercise
    ``_togglePick`` directly, the click code path that
    ``setPickedIndices`` cannot reach."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "small.xyz"
    xyz.write_text(
        "2\nsmall\n"
        "H 0 0 0\n"
        "H 1 0 0\n"
    )
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    mode = page.evaluate(
        "() => document.querySelector('.structure-viewer-slot')"
        "  .__molbuilder_test_handle.getPick().mode"
    )
    assert mode == "triple", (
        f"structure inspector should configure pick.mode='triple'; "
        f"got {mode!r}"
    )
