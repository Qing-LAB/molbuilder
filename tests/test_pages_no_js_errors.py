"""Every served HTML page must boot in a real browser without firing
``pageerror`` or ``console.error``.

The bug this test is here to catch
==================================

Spectra was silently broken for an entire commit window by a JS
SyntaxError in ``static/spectra/viewer.js``: a ``let r`` outer scope
collided with a ``const r`` in the same function, and the module
failed to parse.  ``init()`` therefore never ran and the parameter
form was stuck at the ``Loading from schema...`` placeholder.

No existing test caught it because:

  * the unit / HTTP layers don't load the JS at all,
  * ``test_molbuilder_e2e.py`` only loads ``/modify`` in a browser, and
  * the static-asset string-pin tests grep the JS *source* for
    expected names -- a syntactically-broken file still contains the
    names, so the pins all passed.

This file fills the gap.  It loads every served page, waits for the
module scripts to execute, and asserts that no JS error fired.  For
Spectra it additionally asserts the schema-driven form actually
populated, which proves ``init()`` ran end-to-end (a "parse OK but
init() threw" regression would be invisible to the pageerror check
alone because the unhandled rejection is async).
"""

from __future__ import annotations

import threading

import pytest


# Skip cleanly when Playwright / pytest-playwright / chromium aren't
# installed.  Mirrors the test_molbuilder_e2e.py pattern.
pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Live Flask server fixture                                            #
#  (module-scoped so all four pages share one app + port)               #
# --------------------------------------------------------------------- #


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


# --------------------------------------------------------------------- #
#  Per-page registry                                                    #
#                                                                       #
#  (route, ready_selector) -- ready_selector is an element rendered by  #
#  the server-side template, so its presence proves the HTML reached    #
#  the browser; the pageerror listener catches any JS parse / init      #
#  failure that fires while the module scripts execute on top of it.    #
# --------------------------------------------------------------------- #


_PAGE_BOOT_TIMEOUT_MS = 5000   # generous; module scripts on a cold
                               # headless Chromium take ~1-2 s in CI

_PAGES = [
    # Structure-optimization tab: SIESTA / PySCF script generator.
    # Probe for the "Load from sidebar selection" button — sole
    # structure entry after the 2026-06-08 Build-form retirement.
    ("/structure-optimization", "#load-from-sidebar-btn"),
    # Molbuilder tab: interactive build + edit workspace.  Probe
    # for the selection panel's host.
    ("/molbuilder", "#selection-host"),
    # Spectrum-calculation tab: PySCF spectra script generator.
    ("/spectrum-calculation", "#generate-btn"),
    # Transport-calculation tab: placeholder; pin its boot too so
    # silent JS errors don't hide in the placeholder.
    ("/transport-calculation", "h2"),
    # Results tab: registry-dispatched inspector shell.  Trajectory
    # inspection lives here via the inspector registry; the legacy
    # /watch tab was retired with the migration.
    # 2026-06-18: ready-selector was ``#results-current-file`` but
    # that span is now hidden by the CSS rule at results/style.css
    # line 597 ("hide the redundant top-of-page readout when the
    # picker dropdown is shown") — task #471 made the picker bar
    # always visible.  The picker bar itself is the right
    # always-visible probe.
    ("/results", "#results-file-picker-bar"),
]


def _collect_js_errors(page):
    """Wire up pageerror + console.error listeners; return the list
    that will be populated as the page boots.  The list mutates in
    place as events fire."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    page.on(
        "console",
        lambda msg: (
            errors.append(("console.error", msg.text))
            if msg.type == "error"
            else None
        ),
    )
    return errors


@pytest.mark.parametrize("path,ready_selector", _PAGES)
def test_page_boots_without_js_errors(page, flask_server, path, ready_selector):
    """Loading <path> must produce zero pageerror / console.error.

    Waiting on ``ready_selector`` proves the HTML rendered; waiting an
    extra 250 ms gives ``<script type="module">`` bundles time to
    parse + execute their top-level code (which is where a SyntaxError
    surfaces).
    """
    errors = _collect_js_errors(page)
    page.goto(f"{flask_server}{path}")
    page.wait_for_selector(ready_selector, timeout=_PAGE_BOOT_TIMEOUT_MS)
    # Module-script bundles execute after the initial parse; wait long
    # enough for their top-level code (including parse errors) to fire.
    page.wait_for_timeout(250)
    assert errors == [], (
        f"JS errors while booting {path!r}: {errors}"
    )


# --------------------------------------------------------------------- #
#  Spectra-specific: prove init() ran past the parse boundary           #
#                                                                       #
#  The original bug was a parse failure, which the pageerror check      #
#  above catches.  But a future regression could be "module parses,     #
#  but init() throws on its first await" -- an unhandled rejection      #
#  that may not surface as pageerror in every browser.  We assert       #
#  the schema-driven form populated, which only happens once init()    #
#  has fetched + rendered the schema end-to-end.                        #
# --------------------------------------------------------------------- #


def test_spectra_form_populates_after_init(page, flask_server):
    """The Spectra parameter form must lose its 'Loading from
    schema...' placeholder and render at least one input.  This is the
    end-to-end proof that ``init()`` reached its post-schema-fetch
    rendering step (i.e. nothing silently swallowed the schema fetch)."""
    errors = _collect_js_errors(page)
    page.goto(f"{flask_server}/spectrum-calculation")
    page.wait_for_selector("#generate-btn", timeout=_PAGE_BOOT_TIMEOUT_MS)
    # init() does:  fetch schema -> render fields into #spectra-form-container
    # Wait for ANY <input>/<select> to appear inside that container.
    # If init() throws before this, the wait_for_selector times out
    # below with a clear failure pointing at the broken init path.
    page.wait_for_selector(
        "#spectra-form-container input, #spectra-form-container select",
        timeout=_PAGE_BOOT_TIMEOUT_MS,
    )
    # Belt-and-braces: the placeholder text is gone.
    container_text = page.locator("#spectra-form-container").inner_text()
    assert "Loading from schema" not in container_text, (
        f"Spectra form still shows the loading placeholder after init: "
        f"{container_text!r}"
    )
    assert errors == [], (
        f"JS errors during /spectra init: {errors}"
    )
