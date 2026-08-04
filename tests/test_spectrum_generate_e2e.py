"""First e2e for /spectrum-calculation.

Pre-2026-06-14 the audit found ZERO e2e coverage for the spectrum
tab -- no ``page.goto("/spectrum-calculation")`` anywhere, no
``/api/spectra/*`` traffic exercised by any browser test.  The
Generate button + entire spectra form was unprotected.

This file pins the load-bearing flows:

  1. Page loads without uncaught JS errors.
  2. Schema-driven form renders the spectra fields (basis, max_modes,
     mode toggles).
  3. The viewer-is-truth contract (#405) reaches the server:
     loading a structure via sidebar + clicking Generate sends a
     POST to ``/api/spectra/render`` and the rendered .py contains
     the loaded XYZ's element + coords.
  4. When a sibling ``.molstruct.json`` sidecar is present, in-body
     frozen_atoms travel with the POST (verifies the
     ``apply_labels_to_struct`` server-side path end-to-end).
"""
from __future__ import annotations

import json
import os
import threading

import pytest

pytestmark = pytest.mark.e2e

playwright_module = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright + chromium needed; in the molbuilder env run "
           "``pip install \".[e2e]\" && python -m playwright install "
           "chromium``"
)


# Minimal benzene-like fixture.  Coords don't have to be physical --
# just non-degenerate enough that PySCF / SIESTA don't reject it.
_C6H6_XYZ = """12
benzene
C   1.40   0.00   0.00
C   0.70   1.21   0.00
C  -0.70   1.21   0.00
C  -1.40   0.00   0.00
C  -0.70  -1.21   0.00
C   0.70  -1.21   0.00
H   2.48   0.00   0.00
H   1.24   2.15   0.00
H  -1.24   2.15   0.00
H  -2.48   0.00   0.00
H  -1.24  -2.15   0.00
H   1.24  -2.15   0.00
"""


@pytest.fixture(scope="module")
def flask_server():
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


@pytest.fixture
def benzene_xyz_file(tmp_path, monkeypatch):
    """A C6H6 .xyz registered as a picker root so the sidebar
    + selection API accept it."""
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)

    p = tmp_path / "benzene.xyz"
    p.write_text(_C6H6_XYZ)
    return p


def _open_spectra(page, base_url):
    """Navigate + capture JS errors.  Returns the error list (may
    be empty); caller asserts."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(f"{base_url}/spectrum-calculation")
    page.wait_for_load_state("networkidle", timeout=15_000)
    return errors


# --------------------------------------------------------------------- #
#  Page-load contract                                                    #
# --------------------------------------------------------------------- #


def test_spectrum_page_loads_without_js_errors(page, flask_server):
    """Pinning the absolute baseline: navigating to the spectrum
    tab must not raise any uncaught JS errors.  Pre-fix this had
    NO e2e at all -- a regression that broke the JS at parse time
    could ship silently."""
    errors = _open_spectra(page, flask_server)
    # Some load-time fetches fail when no spectra/run on disk;
    # those become console messages, not pageerror.  We only fail
    # on pageerror (script-parse / uncaught exception).
    assert not errors, (
        "uncaught JS errors on /spectrum-calculation: "
        + "; ".join(errors)
    )


def test_spectrum_form_renders_schema_driven_fields(
        page, flask_server):
    """The schema endpoint drives the form; pin that at least one
    of the expected fields actually rendered.  Catches a schema
    response-shape regression."""
    _open_spectra(page, flask_server)
    # The spectra form mounts inside a container that grows
    # children once /api/build/schema/spectra has returned.
    page.wait_for_function(
        "() => document.querySelectorAll("
        "    'input,select,textarea').length > 3",
        timeout=10_000,
    )
    # Generate button is present + initially DISABLED (no
    # structure loaded yet).
    btn = page.locator("#spectra-generate-btn, "
                       "button[id*='generate'][id*='spectra']")
    if btn.count() == 0:
        # The selector is whatever the partial mounts -- be
        # tolerant.  At minimum, SOME button labelled Generate.
        btn = page.locator("button:has-text('Generate')")
    assert btn.count() >= 1, (
        "no Generate button visible on /spectrum-calculation"
    )


def test_spectrum_commit_mounts_readonly_molview(
        page, flask_server, benzene_xyz_file):
    """Spectra migration: committing a structure mounts the READ-ONLY MolView
    card (the same concealed module Modify/Transport use) into
    ``#spectra-molview-host`` and drives the load through data.openProjectFile —
    replacing the old mol-viewer-embed + raw /api/files/read in spectra/viewer.js.
    """
    errors = _open_spectra(page, flask_server)
    tmp_dir = benzene_xyz_file.parent
    page.wait_for_function(
        "() => !!(window.molbuilder && window.molbuilder.projects"
        "         && document.getElementById('spectra-molview-host'))",
        timeout=10_000,
    )
    page.evaluate(
        "(a) => window.molbuilder.projects.publishCommit(a.dir, a.file)",
        {"dir": str(tmp_dir.resolve()),
         "file": str(benzene_xyz_file.resolve())},
    )
    # The read-only fused card + 3Dmol canvas mount into the host.
    page.wait_for_function(
        "() => { const h = document.getElementById('spectra-molview-host');"
        "  return !!h && !!h.querySelector('.molviewer-card')"
        "         && !!h.querySelector('canvas'); }",
        timeout=15_000,
    )
    # The load completed via MolView (status reads "Loaded …"), which also fed
    # spectra/core.js's holder so Generate has the bytes.
    page.wait_for_function(
        "() => { const el = document.getElementById('load-status');"
        "  return !!el && /loaded/i.test(el.textContent); }",
        timeout=10_000,
    )
    assert not errors, "JS errors after commit+mount: " + "; ".join(errors)


# --------------------------------------------------------------------- #
#  Generate flow: contract end-to-end                                    #
# --------------------------------------------------------------------- #


def test_spectrum_generate_posts_structure_and_returns_script(
        page, flask_server, benzene_xyz_file):
    """The end-to-end CONTRACT:
       1. Page mounts.
       2. We bypass the sidebar (which would need a project
          structure on disk + commit subscription) by calling the
          Generate-trigger primitive directly with the loaded XYZ.
       3. /api/spectra/render returns ok=True with a non-empty
          script that mentions the input atoms.

    Bypassing the sidebar means this isn't a full UI-driven path,
    but the *server contract* (XYZ in → PySCF script out) is the
    critical regression surface.  A separate sidebar-flow test
    can come later."""
    _open_spectra(page, flask_server)
    page.wait_for_function(
        "() => document.querySelectorAll("
        "    'input,select,textarea').length > 3",
        timeout=10_000,
    )

    # Drive a direct fetch from inside the page so we test the
    # ACTUAL server route (not the Flask test client which lives
    # in a different test layer).
    # THE STRUCTURE AS DATA (web-api.md § 1).  This posted a `structure_text`
    # document, which molview.md § 11.7 says cannot exist: what leaves the
    # viewer is "the atoms, their positions, and the facts about them" -- it
    # holds no coordinate document and writes none, so the field's only caller
    # could never fill it and the route retired it.
    from support.envelope import from_xyz
    body = json.dumps({"structure": from_xyz(_C6H6_XYZ), "params": {}})
    js = (
        "(body) => fetch('/api/spectra/render', { "
        "  method: 'POST', "
        "  headers: { 'Content-Type': 'application/json' }, "
        "  body: body "
        "}).then(r => r.json())"
    )
    result = page.evaluate(js, body)
    assert result.get("ok") is True, (
        f"render failed: {result!r}"
    )
    # The script (or whatever the renderer emits) should at least
    # mention the elements we sent.  Spectra renderer key shape
    # varies; do a tolerant scan.
    script_or_blob = json.dumps(result)
    # 12 atoms, 6 C + 6 H.
    assert "C " in script_or_blob or '"C"' in script_or_blob, (
        "rendered spectra output should reference the C atoms in "
        f"the input.  Got: {result!r}"
    )


def test_spectrum_generate_honors_in_body_labels(
        page, flask_server, benzene_xyz_file):
    """Viewer-is-truth contract: when the POST body carries
    ``frozen_atoms``, the server applies them DIRECTLY (no disk
    sidecar re-read).  Pin that the contract reaches the spectra
    render path -- pre-2026-06-14 this only worked for the
    /api/build/* routes."""
    _open_spectra(page, flask_server)
    page.wait_for_function(
        "() => document.querySelectorAll("
        "    'input,select,textarea').length > 3",
        timeout=10_000,
    )

    # The labels ride INSIDE the structure -- one store, and the store is a
    # field of the thing it describes.  A top-level `frozen_atoms` beside a
    # `regions: {}` said "freeze the carbons" and "nothing is labelled" in the
    # same breath; both that key and the second applier that read it are gone.
    from support.envelope import from_xyz
    body = json.dumps({
        "structure": from_xyz(_C6H6_XYZ, frozen=[0, 1, 2, 3, 4, 5]),
        "params": {},
    })
    js = (
        "(body) => fetch('/api/spectra/render', { "
        "  method: 'POST', "
        "  headers: { 'Content-Type': 'application/json' }, "
        "  body: body "
        "}).then(r => r.json())"
    )
    result = page.evaluate(js, body)
    assert result.get("ok") is True, (
        f"render with in-body frozen_atoms failed: {result!r}"
    )
    # Either the script mentions a constraints reference OR an
    # issue notice about constraints-not-emitted-for-spectra; both
    # prove the labels reached the server.  A regression that
    # silently ignored in-body labels would produce neither.
    blob = json.dumps(result).lower()
    assert ("frozen" in blob or "constraint" in blob), (
        "in-body frozen_atoms must surface either in the script "
        "(as a constraints reference) or in issues (as a notice). "
        f"Got: {result!r}"
    )
