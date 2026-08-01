"""End-to-end pins for the /results Structure inspector, now mounted as the
full MolView module READ-ONLY (molview-module.md § 18; the first Results ->
MolView conversion).

Covers the shipped module surfaces (the old triple-pick measurement CHIP was
retired; this file was renamed off it):
  * the measurement OVERLAY — selection-driven xyz / distance / angle (§ 6.4);
  * viewer clicks feeding the store (decision A);
  * B0 — a sidecar ``cell`` reaching the viewer (getLattice);
  * B1 — enabling k-grid tiles the supercell (atom count 2 -> 4 -> 2).

Post-conversion the inspector hands ``molview.mount`` a THROWAWAY workspace and
opens the molecule through ``molview.data.openMolecule``; there is no per-slot
ephemeral store any more.  So the driver reads/writes the GLOBAL data singleton
``window.molbuilder.molview.data`` (``.selection`` for the selection/k-grid, the
handle exposed by molview.mount at ``.structure-viewer-slot .viewer``'s
``__molview_test_handle`` for viewer reads).  Function-scoped flask_server so each
test can register its own ``tmp_path`` as a Capabilities picker root — module-scoped
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
    """Mount the structure inspector via the registry API + wait for the mounted
    MolView module to (a) expose its viewer handle at ``.structure-viewer-slot
    .viewer``'s ``__molview_test_handle`` (set by molview.mount; no production
    reader) and (b) finish ``openMolecule`` so ``molview.data.getStructure()`` is
    non-null."""
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
        "  const v = document.querySelector('.structure-viewer-slot .viewer');"
        "  const d = window.molbuilder && window.molbuilder.molview"
        "            && window.molbuilder.molview.data;"
        "  return v && v.__molview_test_handle && d && d.getStructure();"
        "}",
        timeout=10000,
    )


# The mounted module's viewer handle (molview.mount exposes it here; the owner
# never sees it) and the GLOBAL data singleton's selection store.
_VH = "document.querySelector('.structure-viewer-slot .viewer').__molview_test_handle"
_SEL = "window.molbuilder.molview.data.selection"


def _select(page, indices):
    """Drive the module's selection (input order becomes pickOrder), which drives
    the measurement overlay (§ 6.4) -- decision A: measurement = f(selection)."""
    page.evaluate(f"(indices) => {_SEL}.set(indices)", list(indices))


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


def _write_xyz_with_cell_sidecar(tmp_path, cell, cell_origin=None):
    """A .xyz + its .molstruct.json sidecar carrying a 3x3 `cell` (and an optional
    `cell_origin`, § 3c).  This is the dataset the host reads periodicity from
    (structure-periodicity.md § 8); the viewer never parses -- the cell + origin
    ride the dataset (molview.data)."""
    import hashlib
    import json
    xyz = tmp_path / "periodic.xyz"
    xyz.write_text("2\nperiodic\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n")
    sidecar = {
        "schema_version": 3, "n_atoms_total": 2,
        "structure_hash": hashlib.sha256(xyz.read_bytes()).hexdigest(),
        "regions": {}, "frozen_atoms": [], "cell": cell, "selection_rules": {},
    }
    if cell_origin is not None:
        # § 3c: an EXPLICIT cell that wraps off-origin atoms (an electrode
        # junction) stores the low corner so the box wraps the structure.
        sidecar["cell_origin"] = cell_origin
    (tmp_path / "periodic.molstruct.json").write_text(json.dumps(sidecar))
    return xyz


def test_sidecar_cell_reaches_viewer(
        page, flask_server, tmp_path, monkeypatch):
    """Phase 1 (structure-periodicity.md): a .xyz whose `.molstruct.json` sidecar
    carries a `cell` -> the host reads it server-side (/api/selection/atoms), opens
    the molecule through molview.data.openMolecule with that periodicity ->
    `getLattice()` returns the cell (unit-cell box).  The viewer never parses; the
    cell rides the dataset (molview.data)."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = _write_xyz_with_cell_sidecar(
        tmp_path, [[10, 0, 0], [0, 10, 0], [0, 0, 20]])
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    slot = ".structure-viewer-slot"

    # the shared view-control (isolate toggle) renders on THIS card's left rail.
    assert page.locator(f"{slot} .molviewer-rail-button[data-quick=isolate]").count() == 1

    # the cell reached the viewer (box can draw)
    lat = page.evaluate(f"() => {_VH}.getLattice()")
    assert lat is not None, "sidecar cell should reach getLattice()"


def test_sidecar_cell_origin_reaches_viewer(
        page, flask_server, tmp_path, monkeypatch):
    """END-TO-END pin (structure-authority.md § 5): a sidecar carrying an EXPLICIT
    cell WITH a non-zero `cell_origin` (§ 3c, an electrode-junction cell that wraps
    off-origin atoms) must survive the whole Python->wire->JS path so the data
    model reports that origin -- NOT 0.  This is the invariant that would have
    caught the recurring `cell_origin -> 0` drift at the UI, not just in a unit
    test: it reads molview.data.getUnitCellOrigin() (the single JS key-namer for
    the origin), which surfaces the server's resolved_cell_origin
    (= struct.resolve_cell_origin()).  For an explicit cell + cell_origin the
    server resolves the origin TO that corner."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = _write_xyz_with_cell_sidecar(
        tmp_path, [[10, 0, 0], [0, 10, 0], [0, 0, 20]],
        cell_origin=[1.5, 2.5, 3.5])
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))

    origin = page.evaluate(
        "() => window.molbuilder.molview.data.getUnitCellOrigin()")
    assert origin is not None, "cell_origin dropped on the way to the viewer (was None)"
    assert abs(origin[0] - 1.5) < 1e-6, origin
    assert abs(origin[1] - 2.5) < 1e-6, origin
    assert abs(origin[2] - 3.5) < 1e-6, origin


def test_results_view_controls_bar_drives_store(
        page, flask_server, tmp_path, monkeypatch):
    """The isolate view toggle (.molviewer-rail-button[data-quick=isolate]) drives the
    module's selection store on Results too -- not just Modify.  The toggle lives on
    the viewer's always-visible left rail (.molviewer-rail), so no menu to open.
    Clicking isolate flips the isolate flag; the click reaches the store."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = _write_xyz_with_cell_sidecar(
        tmp_path, [[10, 0, 0], [0, 10, 0], [0, 0, 20]])
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    slot = ".structure-viewer-slot"
    # Precondition: the structure is loaded.  Assert the DATA MODEL (source of
    # truth) -- getAtomCount() reads the embed's 3Dmol model, which lags the
    # data render and raced this to 0 (render-timing).
    _count = "() => window.molbuilder.molview.data.getElements().length"
    assert page.evaluate(_count) == 2
    # isolate: select an atom, then CLICK the rail isolate toggle -> store.isolate flips.
    page.evaluate(f"() => {_SEL}.set([0])")
    page.locator(f"{slot} .molviewer-rail-button[data-quick=isolate]").click()
    page.wait_for_function(
        f"() => {_SEL}.getState().isolate === true")


def test_viewer_clicks_are_wired_to_the_store(
        page, flask_server, tmp_path, monkeypatch):
    """CONTRACT (molview-module.md §13.2 + §15, "decision A"): a viewer click
    forwards to ``store.toggle`` — the selection STORE is the single source of
    truth.  The adapter runs the embed in "single" pick mode with pick halos/labels
    OFF (the render engine glows the selection instead, §13.3), so the embed's pick
    buffer is NOT a second selection; it only reports which atom the click toggled.  A
    multi-atom selection is built by TOGGLING atoms into the store (there is no
    separate triple-pick chip), and the measurement overlay (§15) derives its
    1/2/3-atom readout from that store selection.

    Why this test exists / what would break it:  switching the adapter to
    "multi" pick mode + ``store.set(fullSet)`` (moving the accumulator into the
    embed's own pick buffer — a SECOND source of truth) flips the mode away from
    "single" and is exactly the regression this pins.

    Pin BOTH halves so the mode check can't pass while the wiring is dead:
      1. the embed's pick mode is "single" (adapter-set), AND
      2. invoking the pick callback actually toggles the STORE — a fresh pick
         adds the atom, and the same-atom-twice empty report (single-mode
         deselect, via the adapter's ``prevClicked`` shim) removes it again.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "small.xyz"
    xyz.write_text("2\nsmall\nH 0 0 0\nH 1 0 0\n")
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    # (1) The adapter attaches asynchronously via molview.mount; wait for the
    # embed to be in single-pick mode.  Null-safe: getPick() is undefined until
    # the adapter runs setPick.
    page.wait_for_function(
        "() => {"
        "  const v = document.querySelector('.structure-viewer-slot .viewer');"
        "  const h = v && v.__molview_test_handle;"
        "  const p = h && h.getPick && h.getPick();"
        "  return p && p.mode === 'single';"
        "}",
        timeout=8000,
    )
    # (2) Prove the click callback reaches the store (click -> store.toggle).
    # getPick().onPick IS the adapter's click handler; firing it is equivalent
    # to a real viewer click (which is otherwise unaddressable on a WebGL canvas).
    page.evaluate(f"() => {_SEL}.clear()")
    # A fresh pick of atom 1 toggles it INTO the store.
    page.evaluate(f"() => {_VH}.getPick().onPick([1])")
    assert page.evaluate(
        f"() => JSON.stringify({_SEL}.getState().indices)") == "[1]", (
        "a viewer pick did not reach the store — click wiring is broken "
        "(§13.2: a click must forward to store.toggle)."
    )
    # Clicking the SAME atom again reports an empty set (single-mode deselect);
    # the adapter's prevClicked shim toggles atom 1 back OUT of the store.
    page.evaluate(f"() => {_VH}.getPick().onPick([])")
    assert page.evaluate(
        f"() => {_SEL}.getState().indices.length") == 0, (
        "same-atom-twice deselect did not clear the store — the prevClicked "
        "shim (single-mode empty-report path) is broken."
    )


def test_real_click_selects_in_readonly_inspector(
        page, flask_server, tmp_path, monkeypatch):
    """A REAL mouse click on an atom in the READ-ONLY Results structure inspector selects it,
    and the engine draws the selection glow.  Read-only gates EDIT controls, NOT selection /
    visualization (mount.js attaches the click-to-select adapter regardless of mode; §13.2).

    Distinct from test_viewer_clicks_are_wired_to_the_store, which fires ``onPick`` directly:
    this drives the actual canvas (projecting the atom to screen coords), so it guards the whole
    real-pick chain in a read-only mount -- and that the glow renders in read-only.  It also
    guards against a highlight that RESTYLES the model (which rebuilds geometry and drops each
    atom's ``clickable`` flag, silently killing clicks); the glow is a separate SHAPE (§13.3)."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "spread.xyz"
    # atoms far apart so a projected-centre click lands cleanly on exactly one
    xyz.write_text("4\n\nC 0 0 0\nN 4 0 0\nO 8 0 0\nS 12 0 0\n")
    _open_results(page, flask_server)
    _mount_structure(page, str(xyz))
    page.wait_for_function(
        "() => { const v = document.querySelector('.structure-viewer-slot .viewer');"
        "  const h = v && v.__molview_test_handle;"
        "  const p = h && h.getPick && h.getPick(); return p && p.mode === 'single'; }",
        timeout=8000,
    )
    page.evaluate(f"() => {_SEL}.clear()")

    def click_atom(i):
        # 3dmol-ok: project atom i to screen coords to aim a REAL mouse click at it
        # (a render-FACT read -- the atom's on-screen position -- not a data value).
        scr = page.evaluate(
            "(i) => { const raw = " + _VH + "._viewer3dmol();"
            "  const a = raw.getModel().selectedAtoms({})[i];"
            "  return raw.modelToScreen({x:a.x, y:a.y, z:a.z}); }",
            i,
        )
        page.mouse.click(scr["x"], scr["y"])
        page.wait_for_timeout(150)

    click_atom(1)                       # a REAL click selects, even read-only
    assert page.evaluate(f"() => JSON.stringify({_SEL}.getState().indices)") == "[1]", (
        "a REAL click did not select in read-only mode -- read-only must gate EDIT controls, "
        "not selection (mount.js attaches the adapter regardless of mode)."
    )
    # 3dmol-ok: count the glow shape the engine drew for the selection (a render-FACT read).
    n_shapes = page.evaluate(f"() => {_VH}._viewer3dmol().shapes.length")
    assert n_shapes >= 1, "the selection glow did not render in read-only mode"
    click_atom(1)                       # click the glowing atom again -> deselect
    assert page.evaluate(f"() => {_SEL}.getState().indices.length") == 0, (
        "clicking the glowing atom again did not deselect it in read-only mode"
    )


def test_results_structure_view_uses_real_workspace_persistence(
        page, flask_server, tmp_path, monkeypatch):
    """The Results view is a session like any other consumer (molview-module.md §18):
    it uses the REAL workspace and its session state persists, so a reload restores what
    you were viewing.  "Read-only" means no EDIT controls, NOT no persistence.  Pin: the
    real persisting workspace IS loaded, and opening a molecule writes the session snapshot."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    xyz = tmp_path / "ro.xyz"
    xyz.write_text("2\nro\nH 0 0 0\nH 1 0 0\n")
    _open_results(page, flask_server)
    # The REAL persistence dispatcher IS loaded (it defines workspace.persist) -- the
    # Results tab persists its session like every other consumer; no throwaway hack.
    assert page.evaluate(
        "() => typeof (window.molbuilder.workspace"
        "  && window.molbuilder.workspace.persist) === 'function'") is True
    _mount_structure(page, str(xyz))
    # Opening the molecule anchored a session snapshot (persistence works like any tab),
    # so a reload would restore this Results view.  The mirror is NAMESPACED by owner
    # (molview-module.md §18.4): the structure inspector mounts as ``results:structure``,
    # so its snapshot lands under ``<base>::results:structure`` -- ISOLATED from the base
    # key Modify (and any other consumer) writes.  A base-key snapshot must NOT appear.
    base = page.evaluate(
        "() => (window.molbuilder.constants && window.molbuilder.constants.SS_WORKSPACE)"
        "  || 'molbuilder.workspace.v1'")
    ns_key = base + "::results:structure"
    page.wait_for_function(
        f"() => window.sessionStorage.getItem({ns_key!r}) !== null", timeout=5000)
    assert page.evaluate(f"() => window.sessionStorage.getItem({ns_key!r})") is not None
    # Isolation: this Results session did NOT leak into the shared base key.
    assert page.evaluate(f"() => window.sessionStorage.getItem({base!r})") is None
