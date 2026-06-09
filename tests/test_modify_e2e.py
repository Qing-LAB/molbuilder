"""End-to-end browser tests for the Modify tab.

Covers M2 (UI skeleton: file load, 3Dmol axis overlay) through M4
(orient + rotate, anchor-pair selection) plus Phase 1 cross-tab
persistence (the structure + selection + camera survive Build ↔
Watch ↔ Modify navigation via sessionStorage).

.. note:: 2026-05-20

   The legacy left-column ``#atom-list-body`` atom-list table +
   click-to-select handler were retired in Phase B.1.12 when the
   selection store became the canonical source of truth.  All tests
   that used to drive selection by clicking ``#atom-list-body tr``
   now drive it via the ``_set_selection`` helper, which calls
   ``window.molbuilder.selection.store.setSelection(indices)`` and
   waits for the test hook to observe the new state.  Tests that
   counted ``#atom-list-body tr`` to detect structure-size changes
   now poll ``window.__molbuilder_modify_test.getNAtoms()`` for the
   same signal.  The right-edit-panel ``#selection-info`` table was
   retired in the same pass; the tests that pinned its row contents
   were retired with it.  Phase B.1.13 = this rewrite.

What pytest can't reach
=======================

The existing ``tests/test_web.py`` smoke-tests verify that the
``/modify`` HTML carries the expected element ids and that the
``/api/modify/*`` endpoints respond correctly to JSON.  Neither layer
exercises the actual JS click-sync between the atom list and the
3Dmol viewer, the live ``|offset|`` slider readout, or the JS
runtime: a typo in ``viewer.js`` that ``node --check`` swallows
(e.g. ``state.selectd.has(idx)`` -- a method that doesn't exist on
the wrong object) only fires at click time and is invisible to a
node syntax check.

This file fills the gap: it spins up the real Flask app on a random
port, points a headless Chromium at it, and drives the UI for real.

Why these tests are split from ``test_web.py``
==============================================

* They take longer (~3-5 s each) because of browser startup.
* They depend on ``pytest-playwright`` and the chromium browser
  bundle (``python -m playwright install chromium``) being present.
  When either is missing the file is skipped cleanly via
  ``pytest.importorskip``.
* The existing ``web_client`` fixture uses Flask's ``test_client``
  which doesn't open a real port -- Playwright needs an actual TCP
  listener.  We start one in a background thread per module.

3Dmol click direction is one-way testable
=========================================

The list -> viewer direction (clicking a DOM ``<tr>`` highlights the
atom in the viewer) is testable via Playwright's normal click API.
The viewer -> list direction has to be tested by calling the click
callback via ``page.evaluate(...)`` because 3Dmol atoms live in a
WebGL canvas, not as DOM nodes -- there's no element to click in
pixel coordinates that's stable across viewer rotations.  Both
directions of the sync are still covered; the latter just goes
through ``page.evaluate`` instead of ``page.click``.
"""

from __future__ import annotations

import threading

import pytest


# Skip the entire module if either dependency is missing.  The user
# (or CI) installs them via:
#   pip install playwright pytest-playwright
#   python -m playwright install chromium
pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


_H2O_XYZ = (
    "3\nh2o\n"
    "O 0.000  0.000 0.000\n"
    "H 0.957  0.000 0.000\n"
    "H -0.240 0.927 0.000\n"
)


# --------------------------------------------------------------------- #
#  Live Flask server fixture                                            #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    """Spin up the molbuilder Flask app on a random port in a daemon
    thread.  Yields the base URL; the server stops when the module's
    tests complete."""
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
def water_xyz_file(tmp_path, monkeypatch):
    """A real on-disk water.xyz file Playwright can hand to a file
    input.  Returns the absolute path string.

    Side-effect: tmp_path is added as a Capabilities picker root so
    panel-driving tests (which call ``store.setSourceFile(path)``)
    can pass the picker-root check in /api/selection/atoms.  The
    autouse ``_reset_diagnostics_singleton`` (conftest.py) clears
    the singleton between tests; we restore the snapshot too as
    belt + braces.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "water.xyz"
    p.write_text(_H2O_XYZ)
    return str(p)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Register ``tmp_path`` as a Capabilities picker root.

    Tests that load a custom XYZ fixture into ``tmp_path`` (rather
    than going through the ``water_xyz_file`` fixture) need this so
    ``/api/selection/atoms?path=...`` accepts the file -- otherwise
    the store's atoms list never populates and any
    ``wait_for_function`` on the atom count times out.

    Originally inlined in ``water_xyz_file``; extracted so non-water
    tests can call it too.  See docs/protocols/playwright-tests.md
    § 5 "Test independence".
    """
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _open_modify(page, base_url):
    """Navigate to /modify, capture JS errors, and wait for the JS to
    finish wiring.

    Pre-2026-05-18 this waited for ``#load-btn`` (the browser-local
    file dialog's submit button); that button + the sibling
    ``#file-picker`` were deleted when the Projects sidebar took
    over file selection.  We now wait for the in-page loader API
    that ``modify/viewer.js`` installs at the END of DOMContentLoaded
    -- by then, every event handler the tests rely on is wired.
    """
    errors = []
    page.on("pageerror", lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    page.goto(f"{base_url}/molbuilder")
    page.wait_for_function(
        "() => !!window.molbuilder"
        "       && typeof window.molbuilder.loadStructureText === 'function'"
    )
    return errors


def _load_water(page, water_xyz_file):
    """Load the water.xyz fixture and wait for the structure to
    populate (3 atoms for O + 2H)."""
    _load_file(page, water_xyz_file, expected_atoms=3)


def _load_file(page, xyz_path, expected_atoms):
    """Load an XYZ fixture via the CANONICAL sidebar-Load-button
    workflow: drive ``molbuilderTab.commitFile(path)``.  That hits
    the same code path the Load button does on a real user click:

      1. ``structurePage.loadIntoCanvas`` — gates through the
         warning modal if the canvas is dirty, populates
         ``canvas-state`` (sets ``dirty=false`` for a fresh load).
      2. ``viewerLoader`` (=``loadStructureText``) — POSTs
         /api/build/load, runs ``applyStructure`` which pushes
         atoms into the selection store.
      3. ``store.adoptSession({sourceFile, selection:[]})`` —
         records the sourceFile + resets selection.

    Pre-2026-06-07 audit this called ``store.setSourceFile``
    directly, which bypassed step 1 — canvas-state stayed empty,
    so a subsequent Delete op's ``cs.replaceContent`` bailed (the
    "empty canvas" branch), leaving the dirty flag false.  That
    silently broke the dirty-gated atoms persistence on revisit
    and let the test pass while the selection store went out of
    sync with the viewer.
    """
    from pathlib import Path
    p = Path(str(xyz_path)).resolve()
    page.evaluate(
        "(path) => {"
        " const t = window.molbuilder.molbuilderTab;"
        " if (t && typeof t.commitFile === 'function') {"
        "   return t.commitFile(path);"
        " }"
        " return window.molbuilder.selection.store.setSourceFile(path);"
        "}",
        str(p),
    )
    page.wait_for_function(
        f"() => window.__molbuilder_modify_test "
        f"      && window.__molbuilder_modify_test.getNAtoms() "
        f"         === {int(expected_atoms)}"
    )
    # Also wait for the selection store to land its atoms so panel
    # tests see populated rows.  commitFile calls adoptSession
    # which fetches atoms; settle before tests read state.
    page.wait_for_function(
        f"() => window.molbuilder.selection.store.getState()"
        f"             .atoms.length === {int(expected_atoms)}"
    )


def _set_selection(page, indices):
    """Drive the canonical selection state via the store.  Replaces
    the old "click row N in the atom list" pattern.  Indices are
    0-based atom indices (same as everywhere else in the codebase).
    Awaits the microtask so subsequent reads see the new state.
    """
    page.evaluate(
        "(indices) => window.molbuilder.selection.store.setSelection(indices)",
        list(indices),
    )
    page.wait_for_function(
        f"(want) => JSON.stringify("
        f"  window.__molbuilder_modify_test.getSelected()"
        f") === JSON.stringify(want)",
        arg=sorted(set(indices)),
    )


def _get_selection(page):
    """Read the current selection from the test hook (which itself
    reads live from the store).  Returns a sorted list of indices."""
    return page.evaluate(
        "() => window.__molbuilder_modify_test.getSelected()"
    )


def _set_checkbox(page, selector, value):
    """Set the checked state on a checkbox + fire the ``change`` event
    its JS listener depends on.

    Why not ``page.locator(sel).check()``: form-schema's CSS lays out
    the checkbox inside a flex row container that, on the build form,
    collapses the native input element to width=0 -- Playwright
    rejects ``check()`` even with ``force=True`` because the element
    has no on-screen position to scroll into view.  The label wrapping
    the checkbox is the visible clickable target a real user uses.

    Setting ``checked`` + dispatching ``change`` mirrors what a real
    click would do, minus the Playwright actionability check.  See
    docs/protocols/playwright-tests.md § A1.
    """
    page.evaluate("""(args) => {
        const el = document.querySelector(args.sel);
        if (!el) throw new Error('no checkbox at ' + args.sel);
        el.checked = !!args.value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }""", {"sel": selector, "value": value})


def _set_selection_mode(page, mode):
    """Flip the selection-panel mode without simulating a UI click on
    the underlying radio input.

    The radios at ``#selection-mode-click`` / ``#selection-mode-filter``
    are CSS-hidden behind ``<label class="selection-mode-option">``
    (``opacity: 0; width: 0; height: 0; pointer-events: none``) -- a
    standard "click the label, not the input" pattern.  Playwright's
    ``click(force=True)`` and ``check(force=True)`` both fail with
    ``Element is outside of the viewport`` because the input itself
    has no on-screen area to scroll into view.

    The right way to drive the underlying state in a test is to set
    ``radio.checked = true`` and dispatch the ``change`` event the
    panel's JS listens to -- which is exactly what a click would
    end up doing, minus the Playwright click constraints.  See
    docs/protocols/playwright-tests.md § A1.
    """
    assert mode in ("click", "filter"), f"unknown mode {mode!r}"
    page.evaluate("""(mode) => {
        const id = mode === 'click'
            ? 'selection-mode-click'
            : 'selection-mode-filter';
        const el = document.getElementById(id);
        if (!el) throw new Error(id + ' is not in the DOM');
        el.checked = true;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }""", mode)


def _clear_selection(page):
    """Empty the store's selection."""
    page.evaluate(
        "() => window.molbuilder.selection.store.clearSelection()"
    )
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getSelected().length === 0"
    )


def _open_op_tab(page, op_tab):
    """Click the Modify edit-panel sub-tab (one of "atom", "pose",
    "geom", "junction") so its op-block fieldsets become visible.
    Tests that interact with Pose-/Geom-/Junction-tab controls have
    to activate the right sub-tab first; the Atom tab is the
    default."""
    assert op_tab in ("atom", "pose", "geom", "junction")
    page.locator(f'.optab[data-op-tab="{op_tab}"]').click()
    page.wait_for_function(
        "(t) => document.querySelector("
        "    `.optab-panel[data-op-panel=\"${t}\"]`"
        ").classList.contains('is-active')",
        arg=op_tab,
    )


# --------------------------------------------------------------------- #
#  M2: page loads, atom list populates                                  #
# --------------------------------------------------------------------- #


def test_modify_page_loads_without_js_errors(page, flask_server):
    """The Modify page must boot in a real browser without a single
    pageerror / console.error.  Catches the class of typo
    ``node --check`` can't see (undefined object access fires only
    when the handler runs)."""
    errors = _open_modify(page, flask_server)
    # Wait a brief moment so any deferred init fires.
    page.wait_for_timeout(200)
    assert errors == [], f"JS errors during /molbuilder boot: {errors}"
    # Active-tab marker matches the tab label owned by /molbuilder
    # (see docs/tabs/architecture.md § 2.1 for the tab inventory).
    assert page.locator("a.app-tab.is-active").inner_text() == "Molbuilder"


def test_runtime_modules_registered_on_modify(page, flask_server):
    """Pin the module-init contract (design.md "Module init contract"):
    every module-with-a-global on /modify MUST call
    ``runtime.register(name, api)`` so consumers can ``whenReady``
    instead of polling for ``window.molbuilder.foo``.

    Regression target: a future module-with-a-global that forgets
    to register would still work for same-tick consumers (because of
    the backward-compat global) but break any consumer that
    legitimately uses ``whenReady``.  This test catches that drift."""
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.runtime "
        "      && window.molbuilder.runtime.listRegistered()"
        "                 .includes('selection.store')",
        timeout=10000,
    )
    registered = page.evaluate(
        "() => window.molbuilder.runtime.listRegistered()"
    )
    # The expected set is the module-name registry from design.md;
    # adding a module requires updating BOTH the table there and
    # this assert (intentional friction so the rename gets caught).
    #
    # Review cleanup 2026-06-08 dropped ``modify.loadStructureText``
    # from the required set — its registration had no whenReady
    # consumer in the codebase, so emitting it was dead-code that
    # the workspace-state migration audit caught.  Consumers that
    # need the loader read the global
    # ``window.molbuilder.loadStructureText`` directly (it's still
    # mounted by modify/viewer.js; only the runtime-registry alias
    # was retired).
    for required in (
        "projects",
        "selection.store",
        "selection.panel",
        "selection.viewerAdapter",
        "modify.handle",
    ):
        assert required in registered, (
            f"runtime is missing the {required!r} registration; "
            f"see design.md \"Module init contract\".  "
            f"Got: {registered}"
        )
    # listPending should be empty -- a non-empty list means a
    # consumer is hung forever waiting for a producer that never
    # called register().
    pending = page.evaluate(
        "() => window.molbuilder.runtime.listPending()"
    )
    assert pending == [], (
        f"runtime has dangling whenReady() waiters; producers never "
        f"registered: {pending}"
    )


def test_load_water_populates_structure(page, flask_server, water_xyz_file):
    """Loading water.xyz -> viewer state has 3 atoms; status text
    flips to ok.  The structure is checked via the in-page test hook
    instead of the legacy left-column atom list."""
    errors = _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 3
    # Status flipped to ok.
    status_class = page.locator("#status").get_attribute("class") or ""
    assert "status" in status_class.split() and "ok" in status_class.split(), (
        f"#status class should be 'status ok', got {status_class!r}"
    )
    assert errors == [], f"JS errors during load: {errors}"


def test_sidebar_pick_is_candidate_only_not_auto_load(
        page, flask_server, water_xyz_file):
    """Per docs/tabs/architecture.md § 5.2: sidebar selection sets a
    CANDIDATE; only an explicit Load button click commits the
    viewer load.  Pinning the candidate-only contract: after a
    sidebar pick the viewer must STILL be empty until the user
    clicks Load.

    This is the contract that makes the dirty-canvas warning useful:
    if sidebar clicks auto-loaded, a stray browse-click would
    silently discard unsaved canvas modifications.
    """
    import os as _os
    _open_modify(page, flask_server)
    # Simulate a sidebar pick of water.xyz via setShared (the same
    # API the sidebar uses to publish its current selection).
    water_dir = _os.path.dirname(water_xyz_file)
    page.evaluate(
        """(c) => window.molbuilder.projects.setShared(c.dir, c.file)""",
        {"dir": water_dir, "file": water_xyz_file},
    )
    # Candidate is captured + Load button enabled + readout shows
    # the basename — but the viewer state is STILL empty.
    page.wait_for_function(
        "() => document.getElementById('load-candidate-btn')"
        "        && !document.getElementById('load-candidate-btn').disabled"
    )
    assert page.locator("#load-candidate-readout").inner_text() == (
        "Picked: water.xyz"
    )
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 0, "sidebar pick must NOT auto-load the structure"

    # Click Load.  Now the viewer commits.
    page.locator("#load-candidate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 3"
    )


def test_sidebar_filter_hides_non_matching_files(
        page, flask_server, tmp_path, monkeypatch):
    """B.5.5: the filter input at #ps-filter-input hides files whose
    names don't match.  Substring match by default; a leading-dot
    is the extension shortcut.  Folders always stay visible (they're
    navigation, not data)."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    # Build a small project dir with a mix of file types.
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "water.xyz").write_text(_H2O_XYZ)
    (proj / "benzene.xyz").write_text(_H2O_XYZ)  # contents irrelevant
    (proj / "notes.md").write_text("# notes\n")
    (proj / "structure").mkdir()  # folder
    _open_modify(page, flask_server)
    # Navigate into the project dir via the sidebar API.
    page.evaluate(
        """(d) => window.molbuilder.projects.navigateTo(d)""",
        str(proj),
    )
    page.wait_for_function(
        "() => document.querySelectorAll('#ps-list .ps-entry').length >= 4"
    )
    # Filter to .xyz only.
    page.locator("#ps-filter-input").fill(".xyz")
    # Verify: 2 xyz files visible + 1 folder (always visible) = 3.
    # notes.md is hidden.
    def _visible_names():
        return page.evaluate("""() =>
            Array.from(
                document.querySelectorAll('#ps-list .ps-entry')
            ).filter(li => !li.classList.contains('is-hidden'))
             .map(li => li.querySelector('.ps-entry-name').textContent)
             .sort()""")
    visible = _visible_names()
    assert "water.xyz" in visible
    assert "benzene.xyz" in visible
    assert "structure" in visible, "folders must stay visible"
    assert "notes.md" not in visible, (
        f"filter '.xyz' must hide notes.md; visible: {visible}"
    )
    # Clear via the × button → all entries visible again.
    page.locator("#ps-filter-clear").click()
    visible = _visible_names()
    assert "notes.md" in visible
    # SessionStorage flag cleared too.
    assert page.evaluate(
        "() => sessionStorage.getItem("
        "'molbuilder.projects_sidebar_filter') === null"
    )


def test_sidebar_filter_survives_directory_change(
        page, flask_server, tmp_path, monkeypatch):
    """The active filter re-applies after a directory change so the
    user doesn't have to re-type ``.xyz`` every time they navigate
    into a new project sub-folder."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "proj"
    proj.mkdir()
    sub = proj / "structure"
    sub.mkdir()
    (sub / "water.xyz").write_text(_H2O_XYZ)
    (sub / "readme.txt").write_text("hi\n")
    _open_modify(page, flask_server)
    page.evaluate(
        """(d) => window.molbuilder.projects.navigateTo(d)""",
        str(proj),
    )
    # Apply the filter at the project root.
    page.locator("#ps-filter-input").fill(".xyz")
    # Navigate into the structure/ sub-folder.
    page.evaluate(
        """(d) => window.molbuilder.projects.navigateTo(d)""",
        str(sub),
    )
    # Wait for the new listing.
    page.wait_for_function(
        "() => document.querySelectorAll('#ps-list .ps-entry').length >= 2"
    )
    # The filter is still in the input AND was re-applied: readme.txt
    # is hidden in the new directory too.
    assert page.locator("#ps-filter-input").input_value() == ".xyz"
    is_hidden = page.evaluate("""() => {
        const e = Array.from(
            document.querySelectorAll('#ps-list .ps-entry')
        ).find(li => li.querySelector('.ps-entry-name')
                           .textContent === 'readme.txt');
        return e ? e.classList.contains('is-hidden') : null;
    }""")
    assert is_hidden is True, (
        "active filter must re-apply after a directory change"
    )


def test_sidebar_collapse_toggle_hides_and_shows(page, flask_server):
    """B.5.4: clicking #ps-collapse-toggle adds
    ``is-projects-sidebar-collapsed`` to body (sidebar slides off,
    handle floats at the edge); clicking #ps-collapsed-handle
    brings it back.  State persists per origin in sessionStorage
    so a refresh keeps the user's preference."""
    _open_modify(page, flask_server)
    # Sidebar visible at start; body doesn't carry the collapsed class.
    assert not page.evaluate(
        "() => document.body.classList.contains("
        "'is-projects-sidebar-collapsed')"
    )
    # Collapse.
    page.locator("#ps-collapse-toggle").click()
    page.wait_for_function(
        "() => document.body.classList.contains("
        "'is-projects-sidebar-collapsed')"
    )
    # Sessionstorage persisted the preference.
    assert page.evaluate(
        "() => sessionStorage.getItem("
        "'molbuilder.projects_sidebar_collapsed') === '1'"
    )
    # Reopen via the floating handle.
    page.locator("#ps-collapsed-handle").click()
    page.wait_for_function(
        "() => !document.body.classList.contains("
        "'is-projects-sidebar-collapsed')"
    )
    # SessionStorage cleared on reopen.
    assert page.evaluate(
        "() => sessionStorage.getItem("
        "'molbuilder.projects_sidebar_collapsed') === null"
    )


def test_sidebar_collapse_state_survives_reload(page, flask_server):
    """The collapsed state is restored from sessionStorage on the
    next page mount — before any layout-sensitive widget (3Dmol,
    Plotly, CSS-grid auto-fit) measures geometry."""
    _open_modify(page, flask_server)
    page.locator("#ps-collapse-toggle").click()
    page.wait_for_function(
        "() => document.body.classList.contains("
        "'is-projects-sidebar-collapsed')"
    )
    # Same-tab navigation — sessionStorage survives, the new mount
    # picks up the persisted state.
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_selector("#projects-sidebar", state="attached")
    assert page.evaluate(
        "() => document.body.classList.contains("
        "'is-projects-sidebar-collapsed')"
    ), "collapsed state should restore from sessionStorage on reload"


def test_dblclick_in_sidebar_commits_the_file(
        page, flask_server, water_xyz_file):
    """Per the universal sidebar interaction model: a single click
    sets the candidate (preview only); a DOUBLE click is the
    commit, equivalent to clicking the Load button.  Both routes
    land on _commitFile which gates through the canvas-state
    warning modal."""
    _open_modify(page, flask_server)
    # Drive a commit via the projects.publishCommit API directly —
    # the sidebar's dblclick handler is what calls publishCommit;
    # this test pins the *response* (Molbuilder tab loads the
    # committed file) regardless of which UI affordance fires it.
    import os as _os
    page.evaluate(
        """(c) => window.molbuilder.projects.publishCommit
                ? window.molbuilder.projects.publishCommit(c.dir, c.file)
                : null""",
        {"dir": _os.path.dirname(water_xyz_file), "file": water_xyz_file},
    )
    # Viewer commits the structure — _commitFile ran through the
    # gate + viewer loader + adoptSession.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 3"
    )


def test_dblclick_commit_with_dirty_canvas_fires_warning(
        page, flask_server, water_xyz_file, tmp_path, monkeypatch):
    """A sidebar dblclick (publishCommit) must hit the SAME warning
    modal gate as the Load button when the canvas is dirty.  Pins
    the load-bearing safety: editing then dblclicking a different
    sidebar entry doesn't silently discard work."""
    import os as _os
    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    # Make the canvas dirty.
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # Now simulate a dblclick on a DIFFERENT file.  Even though the
    # file is the SAME water.xyz here (fixture only ships one), the
    # commit-event-fires-warning contract holds: a dirty canvas +
    # commit = modal.
    page.evaluate(
        """(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)""",
        {"dir": _os.path.dirname(water_xyz_file), "file": water_xyz_file},
    )
    page.wait_for_selector("dialog.molbuilder-warning-modal",
                           state="attached", timeout=2000)
    # Cancel — viewer state stays at 2 (the post-delete edit).
    page.locator(
        'dialog.molbuilder-warning-modal [data-action="cancel"]'
    ).click()
    page.wait_for_selector("dialog.molbuilder-warning-modal",
                           state="detached", timeout=2000)
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 2


def test_load_button_disabled_when_non_structure_file_picked(
        page, flask_server, tmp_path, monkeypatch):
    """A pick of a non-loadable file (.log, .fdf, README.md, ...)
    must clear the candidate so the Load button disables — a
    user browsing the project tree doesn't want a stale water.xyz
    to commit when they click on a README.md afterwards.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    structure = tmp_path / "water.xyz"
    structure.write_text(_H2O_XYZ)
    readme = tmp_path / "README.md"
    readme.write_text("# notes\n")

    _open_modify(page, flask_server)
    # Pick the loadable file first → button enables.
    page.evaluate(
        """(c) => window.molbuilder.projects.setShared(c.dir, c.file)""",
        {"dir": str(tmp_path), "file": str(structure)},
    )
    page.wait_for_function(
        "() => !document.getElementById('load-candidate-btn').disabled"
    )
    # Now pick the non-structure file → button disables again.
    page.evaluate(
        """(c) => window.molbuilder.projects.setShared(c.dir, c.file)""",
        {"dir": str(tmp_path), "file": str(readme)},
    )
    page.wait_for_function(
        "() => document.getElementById('load-candidate-btn').disabled"
    )
    assert page.locator("#load-candidate-readout").inner_text() == ""


def test_smiles_generator_renders_structure_in_viewer(page, flask_server):
    """End-to-end SMILES flow: type ``CCO`` into the SMILES input,
    click Generate.  The RDKit backend builds ethanol; the panel
    routes the XYZ through ``structurePage.loadIntoCanvas`` (canvas
    is empty → no warning fires) and the 3Dmol viewer renders the
    9 atoms (3 heavy + 6 H).  Status text reports the atom count.

    Skips if rdkit isn't installed — the build endpoint surfaces
    a clean error in that case but this test wants the happy path.
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; cannot exercise SMILES build")

    _open_modify(page, flask_server)
    # Expand the SMILES panel + type a SMILES + click Generate.
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("CCO")
    page.locator("#smiles-generate-btn").click()
    # Wait for the viewer to populate.  Ethanol with H = 9 atoms
    # via RDKit's default geometry generation.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 9",
        timeout=10_000,
    )
    # Status text reports the result with the SMILES echo.
    status_text = page.locator("#smiles-status").inner_text()
    assert "9" in status_text and "CCO" in status_text


def _load_water_via_button(page, water_xyz_file):
    """Load water.xyz the way a real user does: drive the
    candidate-only pick + click the Load button.  This routes
    through structurePage.loadIntoCanvas so canvas-state actually
    sees the load — the legacy ``_load_water`` helper calls
    ``store.setSourceFile`` directly and bypasses the canvas state
    machine, fine for viewer-only tests but not for canvas-aware
    ones.
    """
    import os as _os
    water_dir = _os.path.dirname(water_xyz_file)
    page.evaluate(
        """(c) => window.molbuilder.projects.setShared(c.dir, c.file)""",
        {"dir": water_dir, "file": water_xyz_file},
    )
    page.wait_for_function(
        "() => !document.getElementById('load-candidate-btn').disabled"
    )
    page.locator("#load-candidate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 3"
    )


def test_modifier_op_marks_canvas_dirty(
        page, flask_server, water_xyz_file):
    """A modifier op (here: Delete) flips canvas-state's dirty flag
    so a subsequent Load / Generate would fire the unsaved-
    modifications warning.  Without this wire-up, the warning
    primitive lives but never fires for non-SMILES workflows."""
    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    # Pre-op: canvas exists but isn't dirty (just loaded).
    state_before = page.evaluate("""() => ({
        empty: window.molbuilder.structureCanvas.isEmpty(),
        dirty: window.molbuilder.structureCanvas.isDirty(),
    })""")
    assert state_before == {"empty": False, "dirty": False}, (
        f"pre-op state should be loaded+clean, got {state_before!r}"
    )
    # Apply a Delete op (pick the O atom and delete).
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # Post-op: canvas-state is dirty.
    is_dirty = page.evaluate(
        "() => window.molbuilder.structureCanvas.isDirty()"
    )
    assert is_dirty is True, (
        "modifier op should flip canvas-state.dirty so a subsequent "
        "Load / Generate fires the warning modal"
    )


def test_smiles_with_dirty_canvas_fires_warning_modal(
        page, flask_server, water_xyz_file):
    """End-to-end gate behavior: a dirty canvas + a SMILES Generate
    click MUST fire the warning modal before discarding edits.
    Click Discard → the new structure lands.  This is what makes
    the canvas-state primitive load-bearing for user-facing safety
    (vs being a passive bit nothing reads)."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; cannot exercise SMILES build")

    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    # Make the canvas dirty by deleting an atom.
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # Now click Generate-from-SMILES.  Warning modal must appear.
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("C")
    page.locator("#smiles-generate-btn").click()
    # Wait for the modal to appear in the DOM.
    page.wait_for_selector("dialog.molbuilder-warning-modal",
                           state="attached", timeout=2000)
    # Click "Discard and continue" — the new structure lands.
    page.locator(
        'dialog.molbuilder-warning-modal [data-action="discard"]'
    ).click()
    # Methane has 5 atoms (1 C + 4 H) via RDKit.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 5",
        timeout=10_000,
    )


def test_smiles_with_dirty_canvas_cancel_keeps_edit(
        page, flask_server, water_xyz_file):
    """Same warning flow, but the user picks Cancel — the
    in-progress structure stays put.  Pins the "no surprise
    overwrite" promise."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; cannot exercise SMILES build")

    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    n_before = page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    )
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("C")
    page.locator("#smiles-generate-btn").click()
    page.wait_for_selector("dialog.molbuilder-warning-modal",
                           state="attached", timeout=2000)
    # Cancel.  The viewer keeps the edited 2-atom structure.
    page.locator(
        'dialog.molbuilder-warning-modal [data-action="cancel"]'
    ).click()
    # Modal closes; viewer state unchanged.
    page.wait_for_selector("dialog.molbuilder-warning-modal",
                           state="detached", timeout=2000)
    n_after = page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    )
    assert n_after == n_before, (
        f"cancel kept edits intact: was {n_before} atoms, "
        f"now {n_after}"
    )


def test_save_writes_to_source_and_clears_dirty(
        page, flask_server, tmp_path, monkeypatch):
    """Full Save round-trip: load, edit, click Save → file on disk
    is updated AND canvas.dirty clears.  The orchestrator's
    markSavedTo runs so any subsequent Load / Generate WILL NOT
    fire the warning modal (until the user re-edits).

    Uses a subdirectory under tmp_path because /api/files/upload
    requires target_dir depth ≥ 1 inside the picker root (no
    uploads directly under the root).
    """
    from pathlib import Path as _P
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    water_xyz_file = str(project_dir / "water.xyz")
    _P(water_xyz_file).write_text(_H2O_XYZ)
    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    # Edit: delete the O atom — file content will diverge from
    # the original 3-atom water.
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    assert page.evaluate(
        "() => window.molbuilder.structureCanvas.isDirty()"
    ) is True
    # Expand the Save panel + click Save.
    page.locator("#save-to-source-btn").click()
    # Wait until the inflight save resolves (status leaves "Saving…").
    page.wait_for_function(
        "() => !document.getElementById('save-status').textContent"
        "        .toLowerCase().startsWith('saving')"
    )
    status_text = page.locator("#save-status").inner_text()
    assert "saved water.xyz" in status_text.lower(), (
        f"save-status should report success; got {status_text!r}"
    )
    # Dirty bit cleared via markSavedTo.
    assert page.evaluate(
        "() => window.molbuilder.structureCanvas.isDirty()"
    ) is False
    assert page.evaluate(
        "() => window.molbuilder.structureCanvas.getLastSavedTo()"
    ) == water_xyz_file
    # The file on disk now reflects the 2-atom post-delete structure.
    new_text = _P(water_xyz_file).read_text()
    n_atoms_line = new_text.strip().splitlines()[0]
    assert n_atoms_line.strip() == "2", (
        f"file on disk should reflect post-delete 2 atoms; first "
        f"line is {n_atoms_line!r}"
    )


def test_save_button_disabled_for_smiles_without_prior_save(
        page, flask_server):
    """A SMILES-generated structure has no source.file and no
    last_save_to — Save would have nowhere to write.  The button
    must disable + the readout explains why (Save-as comes later)."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; cannot exercise SMILES build")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("C")
    page.locator("#smiles-generate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 5",
        timeout=10_000,
    )
    # Button stays disabled even with a SMILES-generated structure
    # in the workspace — there's no target to write to.
    assert page.locator("#save-to-source-btn").is_disabled()
    # Readout explains why.
    readout = page.locator("#save-readout").inner_text()
    assert "Save as" in readout or "No source" in readout, (
        f"readout should explain why Save is disabled; got {readout!r}"
    )


def test_dna_generator_renders_structure_in_viewer(page, flask_server):
    """End-to-end DNA flow: type "ACGT" into the DNA input, click
    Generate.  Backend picks 3DNA → AmberTools → RDKit in that
    preference order; the panel routes the XYZ through the canvas
    gate and the viewer renders the helix.

    Skips if no DNA backend is installed (the 3-backend cascade
    needs at least one of 3DNA / AmberTools / RDKit).
    """
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    avail = available_backends()
    if not (avail.get("threedna", False)
            or avail.get("amber",    False)
            or avail.get("rdkit",    False)):
        pytest.skip("no DNA backend (3DNA / AmberTools / RDKit) installed")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='dna']").click()
    page.locator("#dna-input").fill("ACGT")
    page.locator("#dna-generate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() >= 50",
        timeout=20_000,
    )
    status_text = page.locator("#dna-status").inner_text()
    assert "ACGT" in status_text
    assert "B-form" in status_text


def test_dna_generator_populates_selection_store(
        page, flask_server):
    """Generator → selection panel sync (2026-06-07 BOMB-0 finale).

    The user-reported regression: generating DNA on the Molbuilder
    tab put the structure in the viewer but the selection panel's
    atom list stayed empty.  Root cause: /api/build/load (the
    endpoint loadStructureText calls after every generator) was
    not including the canonical ``atoms`` payload, so the
    front-end's ``applyStructure(r) -> store.adoptAtoms(r.atoms)``
    silently no-op'd (Array.isArray(undefined) === false).

    This test pins that the selection store ACTUALLY has atoms
    after a DNA generate — not just that the viewer has them.
    The 3 prior reviews missed this because the existing
    generator tests only checked viewer atom count, not the
    selection store, and the BOMB-0 test only exercised
    /api/modify/* which already used the atoms_list helper."""
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    avail = available_backends()
    if not (avail.get("threedna", False)
            or avail.get("amber",    False)
            or avail.get("rdkit",    False)):
        pytest.skip("no DNA backend installed")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='dna']").click()
    page.locator("#dna-input").fill("ACGT")
    page.locator("#dna-generate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() >= 50",
        timeout=20_000,
    )
    # The selection store MUST have the same atoms.  Pre-fix this
    # was 0 — viewer populated, store empty, panel blank.
    store_atom_count = page.evaluate(
        "() => window.molbuilder.selection.store.getState().atoms.length"
    )
    viewer_atom_count = page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    )
    assert store_atom_count == viewer_atom_count, (
        f"selection store has {store_atom_count} atoms but viewer "
        f"shows {viewer_atom_count} — generator path failed to "
        f"push atoms through applyStructure -> adoptAtoms"
    )


def test_generator_resets_stale_selection(
        page, flask_server, water_xyz_file):
    """Pin the resetSelection=true semantic for loadStructureText.

    Without it: user loads water (3 atoms) → selects atom [1] →
    generates SMILES water → selection [1] PRESERVED (in-range
    in the new structure) → silently pointing at the wrong atom.
    With the new opts.resetSelection in applyStructure, every
    structure swap via loadStructureText clears selection.

    Counterexample is the modifier-op path (postOp), which passes
    no opts and preserves selection (filtered to in-range).  That
    contract is pinned by ``test_modify_state_after_op_survives_
    navigation``."""
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    if not available_backends().get("rdkit", False):
        pytest.skip("RDKit not installed")

    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _set_selection(page, [1])
    assert _get_selection(page) == [1]

    # Now generate methane via SMILES.  Pick a structure with a
    # DIFFERENT atom count from water so the wait_for_function
    # actually waits for the generator to fire (waiting for the
    # same atom count would succeed immediately and miss the
    # store-sync window we're trying to assert on).  Selection
    # MUST clear — index 1 from the previous water means something
    # different in methane even though it's in-range.
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("C")   # methane (5 atoms)
    page.locator("#smiles-generate-btn").click()
    page.wait_for_function(
        "() => window.molbuilder.selection.store.getState()"
        ".atoms.length === 5",
        timeout=10_000,
    )
    assert _get_selection(page) == [], (
        f"selection must clear after a generator-driven structure "
        f"swap; still got {_get_selection(page)}"
    )


def test_smiles_generator_populates_selection_store(
        page, flask_server):
    """Same contract as DNA but for SMILES — exercises the OTHER
    generator path (RDKit instead of 3DNA/Amber) so a regression
    that hits one but not the other is caught."""
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    if not available_backends().get("rdkit", False):
        pytest.skip("RDKit not installed")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("O")  # water
    page.locator("#smiles-generate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 3",
        timeout=10_000,
    )
    store_atom_count = page.evaluate(
        "() => window.molbuilder.selection.store.getState().atoms.length"
    )
    assert store_atom_count == 3, (
        f"selection store should have 3 atoms after SMILES generate; "
        f"got {store_atom_count}"
    )


def test_sidebar_load_populates_selection_store(
        page, flask_server, water_xyz_file):
    """The OTHER path that hits /api/build/load: sidebar pick →
    Load button → loadStructureText → applyStructure.  Pin that
    this also propagates atoms into the store."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    store_atom_count = page.evaluate(
        "() => window.molbuilder.selection.store.getState().atoms.length"
    )
    assert store_atom_count == 3


def test_rna_generator_renders_structure_in_viewer(page, flask_server):
    """End-to-end RNA flow: type "ACGU" → click Generate.  Backend
    picks 3DNA → AmberTools → RDKit; viewer renders the A-form
    helix.  Skips if no nucleic backend installed."""
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    avail = available_backends()
    if not (avail.get("threedna", False)
            or avail.get("amber",    False)
            or avail.get("rdkit",    False)):
        pytest.skip("no RNA backend (3DNA / AmberTools / RDKit) installed")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='rna']").click()
    page.locator("#rna-input").fill("ACGU")
    page.locator("#rna-generate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() >= 50",
        timeout=20_000,
    )
    status_text = page.locator("#rna-status").inner_text()
    assert "ACGU" in status_text
    assert "A-form" in status_text


def test_peptide_generator_renders_structure_in_viewer(page, flask_server):
    """End-to-end peptide flow: type "AC" (alanine-cysteine) into
    the Peptide input, click Generate.  Backend dispatches to
    tleap; the panel routes the XYZ through
    ``structurePage.loadIntoCanvas`` and the viewer renders the
    dipeptide.

    Skips if AmberTools isn't available — that's the peptide
    backend; without it the test would hit a 4xx and pass false.
    """
    try:
        from molbuilder.backends import available_backends
    except ImportError:
        pytest.skip("molbuilder.backends import failed")
    if not available_backends().get("amber", False):
        pytest.skip("AmberTools not available; cannot exercise peptide build")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='peptide']").click()
    page.locator("#peptide-input").fill("AC")
    page.locator("#peptide-generate-btn").click()
    # A dipeptide has ~30 atoms (alanine ~13 + cysteine ~14 +
    # caps).  Assert a reasonable floor rather than an exact
    # count so a future tleap force-field change with different
    # cap atoms still passes.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() >= 20",
        timeout=20_000,
    )
    status_text = page.locator("#peptide-status").inner_text()
    assert "AC" in status_text


def test_file_upload_panel_loads_local_xyz(
        page, flask_server, water_xyz_file):
    """End-to-end file-upload flow: click the file input, pick a
    local .xyz, watch the viewer render the structure.  Mirrors
    the SMILES + name happy-path tests; uses the existing
    water_xyz_file fixture to avoid creating a new disk file."""
    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='upload']").click()
    # Playwright's set_input_files attaches the disk path as the
    # selected file — equivalent to the user picking it via the
    # browser's file dialog.
    page.locator("#file-upload-input").set_input_files(water_xyz_file)
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 3",
        timeout=10_000,
    )
    status_text = page.locator("#file-upload-status").inner_text()
    assert "water.xyz" in status_text


def test_name_generator_renders_structure_in_viewer(page, flask_server):
    """End-to-end name flow: type "water" into the Name input,
    click Generate.  The backend hits PubChem (cached or live);
    the panel routes the XYZ through ``structurePage.loadIntoCanvas``
    and the viewer renders the molecule.

    Skips if the PubChem lookup fails (no rdkit available, no
    network) — the test wants the happy path.
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; cannot exercise name build")

    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='name']").click()
    page.locator("#name-input").fill("water")
    page.locator("#name-generate-btn").click()
    # water has 3 atoms (O + 2H).  PubChem 3D record may add no
    # extras; assert "loaded a structure" rather than an exact count
    # so a future lookup that resolves the same molecule via a
    # different conformer still passes.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() >= 3",
        timeout=15_000,
    )
    status_text = page.locator("#name-status").inner_text()
    assert "water" in status_text.lower()


def test_smiles_generator_empty_input_surfaces_inline_error(
        page, flask_server):
    """Click Generate with an empty SMILES input → inline error,
    NO network call (the module rejects empty input client-side)."""
    _open_modify(page, flask_server)
    page.locator(".init-tab[data-init-tab='smiles']").click()
    page.locator("#smiles-input").fill("")
    page.locator("#smiles-generate-btn").click()
    # Inline error appears WITHOUT a roundtrip — the message is
    # the client-side validation string.
    page.wait_for_function(
        "() => document.getElementById('smiles-status').textContent"
        "        .toLowerCase().includes('enter a smiles')"
    )
    # Viewer state is unchanged (still empty).
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 0


# --------------------------------------------------------------------- #
#  Selection-store contracts                                            #
#                                                                       #
#  The legacy "plain click = single-select, shift-click = multi-select" #
#  + #selection-readout + #selection-info-body tests were retired       #
#  2026-05-20 along with the UI that backed them.  The selection store  #
#  (window.molbuilder.selection.store) is now the canonical state,      #
#  edited via ``toggleAtom`` / ``setSelection`` / ``applyFilter``.  The #
#  tests below pin the contracts a /modify user actually relies on:     #
#  the store updates state.selection, viewer.js re-renders button       #
#  enablement, the test hook reads live from the store.                 #
# --------------------------------------------------------------------- #


def test_3dmol_atom_serial_matches_zero_based_index(
        page, flask_server, water_xyz_file):
    """Probe what 3Dmol stores as ``atom.serial`` for XYZ-loaded atoms.

    The viewer-adapter forwards ``atom.serial`` to
    ``store.toggleAtom`` -- if 3Dmol used 1-based serials, every
    viewer click would land on the wrong atom.  This test pins the
    0-based contract by reading atom records via the test hook."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    atoms = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map((a) => ({
            serial: a.serial,
            elem:   a.elem,
            clickable: a.clickable,
        }));
    }""")
    serials = [a["serial"] for a in atoms]
    assert serials == [0, 1, 2], (
        f"3Dmol XYZ atom serials are not 0-based: {atoms!r}"
    )
    # setClickable runs every render now (the viewer-adapter re-arms
    # to survive model swaps); atoms should carry clickable=true.
    assert all(a["clickable"] for a in atoms), (
        f"clickable flag didn't propagate to atoms: {atoms!r}"
    )


def test_store_set_selection_persists_through_test_hook(
        page, flask_server, water_xyz_file):
    """``store.setSelection`` writes state.selection; the test hook
    reads it live.  Pins the round-trip the rest of the suite uses
    to drive selection state."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _set_selection(page, [0, 2])
    assert _get_selection(page) == [0, 2]
    _set_selection(page, [1])
    assert _get_selection(page) == [1]
    _clear_selection(page)
    assert _get_selection(page) == []


def test_store_toggle_atom_flips_membership(
        page, flask_server, water_xyz_file):
    """Each ``store.toggleAtom`` call flips the atom's membership in
    state.selection (client-side; no HTTP).  Replaces the old
    plain-click / shift-click semantics with the new
    "every click toggles" model."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    toggle = ("(i) => "
              "window.molbuilder.selection.store.toggleAtom(i)")
    page.evaluate(toggle, 0)
    page.evaluate(toggle, 2)
    assert _get_selection(page) == [0, 2]
    page.evaluate(toggle, 0)   # toggle off
    assert _get_selection(page) == [2]


_PANEL_URL_FOR_FIXTURE = "/partials/selection-panel"


def _wait_panel_ready(page):
    """Wait until the selection-panel partial is mounted into
    #selection-host.  Selection-panel.js fetches the partial async
    on DOMContentLoaded; tests that interact with panel DOM must
    wait for that mount to complete or the locators race the
    fetch.

    ``state="attached"`` (not the default "visible") because the
    mode-radio inputs are intentionally hidden via
    ``opacity:0;width:0;height:0`` -- the user clicks the
    surrounding label/span pill, not the radio dot itself
    (segmented-control styling).  A naive ``wait_for_selector``
    with the default visibility requirement times out forever
    against a perfectly-mounted panel."""
    page.wait_for_selector("#selection-mode-click", state="attached")


# --------------------------------------------------------------------- #
#  Selection panel -- DOM-driven UI tests                               #
#                                                                       #
#  Every other test in this file drives selection via _set_selection    #
#  (which goes straight to store.setSelection); that's correct for      #
#  exercising the ops + state, but it bypasses the panel completely.    #
#  The tests below click panel DOM directly so the panel -> store ->    #
#  view round-trip is actually exercised end-to-end.                    #
# --------------------------------------------------------------------- #


def test_panel_partial_mounts_under_modify(
        page, flask_server, water_xyz_file):
    """The selection panel partial is fetched + mounted by
    selection-bootstrap.js on DOMContentLoaded, and the panel
    subscribes to the store.  Pinning BOTH is the contract -- a
    panel whose DOM mounted but whose subscribe() never fired
    would render a frozen count display, which is exactly the
    kind of silent regression this test exists to catch.
    """
    _open_modify(page, flask_server)
    _wait_panel_ready(page)
    # Required IDs come from the partial; confirm the structural
    # ones that other tests depend on.
    for required in (
        "selection-mode-click", "selection-mode-filter",
        "selection-click-section", "selection-filter-section",
        "selection-apply-filter", "selection-select-all",
        "selection-assign-target", "selection-assign-new-label",
        "selection-count",
    ):
        assert page.locator("#" + required).count() == 1, (
            f"selection panel missing required id #{required}"
        )
    # Mounting the DOM is half the contract; subscribe-wired is the
    # other half.  Drive a store change and watch the count cell
    # react -- if subscribe() never fired, the cell stays at its
    # initial "no structure" / "0 atoms" text.
    _load_water(page, water_xyz_file)
    _set_selection(page, [1])
    page.wait_for_function(
        '() => document.getElementById("selection-count")'
        '      .textContent.trim() === "1 / 3 atoms"'
    )


def test_panel_mode_swap_preserves_selection(
        page, flask_server, water_xyz_file):
    """The spec (atom-selection.md §3) promises state.selection is
    shared across modes; switching modes is a pure UI swap.  Set a
    selection in click mode, flip to filter, flip back, and confirm:

      * the store still carries the same indices,
      * the re-rendered atom-list checkboxes reflect the preserved
        selection -- the render path is the actual regression risk;
        the store is the boring part.
    """
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    _set_selection(page, [0, 2])
    # Flip to filter mode -- the click section hides, but selection
    # must persist in the store.
    _set_selection_mode(page, "filter")
    page.wait_for_function(
        "() => document.getElementById('selection-filter-section')"
        "      .hidden === false"
    )
    assert _get_selection(page) == [0, 2], (
        "selection was cleared when switching modes"
    )
    _set_selection_mode(page, "click")
    page.wait_for_function(
        "() => document.getElementById('selection-click-section')"
        "      .hidden === false"
    )
    assert _get_selection(page) == [0, 2], (
        "selection was cleared when switching back to click"
    )
    # And the atom-list checkboxes mirror the preserved selection.
    checked = page.evaluate("""() => {
        const rows = document.querySelectorAll(
            '#selection-atom-list tr');
        const out = [];
        rows.forEach((tr) => {
            const cb = tr.querySelector('input[type="checkbox"]');
            if (cb && cb.checked) {
                out.push(parseInt(tr.dataset.atomIndex, 10));
            }
        });
        return out.sort((a, b) => a - b);
    }""")
    assert checked == [0, 2], (
        f"atom-list checkboxes did not reflect preserved selection "
        f"after mode round-trip; got {checked}"
    )


def test_panel_add_filter_button_flips_mode_to_filter(
        page, flask_server, water_xyz_file):
    """The + Add filter button must flip mode to filter even if the
    user was in click mode -- otherwise the newly-added filter row
    would be hidden and the user wouldn't see it.  (panel.js:
    addFilterBtn handler.)"""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    # Confirm click mode is the default.
    assert page.locator("#selection-mode-click").is_checked()
    page.locator("#selection-add-filter").click()
    page.wait_for_function(
        "() => document.getElementById('selection-mode-filter').checked"
    )
    # The filter editor section is visible now.
    assert page.locator("#selection-filter-section").is_visible()
    # And exactly one filter row was added.
    assert page.locator(".selection-filter-row").count() == 1


def test_panel_apply_filter_with_no_filters_clears_selection(
        page, flask_server, water_xyz_file):
    """store.applyFilter() with an empty filter list treats it as
    'select nothing' and replaces state.selection with [] WITHOUT
    making a server round-trip (store.js _filtersToRule -> null
    branch returns before posting to /api/selection/eval).

    Tested two ways:
      * via the panel's Apply button so the panel -> store wiring
        is in the test;
      * via a page.route interceptor that fails if /api/selection/eval
        is ever called -- a future regression that posts ``rule: null``
        to the server would fail loudly here instead of being masked
        by a server tolerant of the bad rule.
    """
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    _set_selection(page, [0, 1, 2])
    # Intercept eval requests; any call here is a contract violation.
    eval_calls = []
    page.route("**/api/selection/eval", lambda route: (
        eval_calls.append(route.request.url),
        route.fulfill(status=599, body="forbidden by test")
    )[1])
    # Flip to filter mode; the filter list is empty.
    _set_selection_mode(page, "filter")
    page.wait_for_function(
        "() => document.getElementById('selection-mode-filter').checked"
    )
    page.locator("#selection-apply-filter").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getSelected().length === 0"
    )
    assert _get_selection(page) == []
    page.unroute("**/api/selection/eval")
    assert eval_calls == [], (
        f"empty-filter Apply must NOT hit /api/selection/eval; "
        f"got {len(eval_calls)} call(s): {eval_calls}"
    )


def test_panel_apply_filter_with_empty_row_skips_that_row(
        page, flask_server, water_xyz_file):
    """If the user adds a filter row but leaves its value empty
    (e.g. just clicked + Add filter then Apply), the empty row
    must be SKIPPED -- not sent to the server as
    ``{op: by_index_range, expression: ""}`` which evaluates to
    [] and under AND combinator would silently wipe the selection.

    Distinct from test_panel_apply_filter_with_no_filters_clears_selection
    which exercises the state.filters.length === 0 branch.  Here
    state.filters has one entry, but the entry has no value -- a
    half-typed filter that must not poison the rule.
    """
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    _set_selection(page, [0, 1, 2])
    # Seed a filter row that pins an O atom, then add an empty
    # by_index row.  Combinator OR.  The expected result is just
    # the O atom (atom 0); a buggy _filterToRule that sent the
    # empty row as expression:"" would intersect (under AND) or
    # contribute nothing (under OR) -- under OR the test would
    # still pass with the wrong rule, so we test AND explicitly
    # which is the failure mode.
    page.evaluate("""() => {
        const s = window.molbuilder.selection.store;
        s.setFilters([
            {kind: "by_element", value: "O"},
            {kind: "by_index",   value: ""}
        ]);
        s.setCombinator("and");
    }""")
    _set_selection_mode(page, "filter")
    page.wait_for_function(
        "() => document.getElementById('selection-mode-filter').checked"
    )
    page.locator("#selection-apply-filter").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getSelected().length === 1"
    )
    # Just atom 0 (the O) -- the empty by_index row was skipped
    # rather than intersected as an empty operand.
    assert _get_selection(page) == [0], (
        "empty by_index filter was not skipped when building the "
        f"rule; got selection {_get_selection(page)} (expected [0])"
    )


def test_panel_select_all_checkbox_tri_state(
        page, flask_server, water_xyz_file):
    """The #selection-select-all checkbox is tri-state:
      * unchecked when nothing is selected,
      * indeterminate when some atoms are selected,
      * checked when every atom is selected.
    (panel.js renderAtomList -- the all/none/partial branches.)"""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)

    state = page.evaluate("""() => {
        const c = document.getElementById('selection-select-all');
        return {checked: c.checked, indeterminate: c.indeterminate};
    }""")
    assert state == {"checked": False, "indeterminate": False}

    _set_selection(page, [0])
    state = page.evaluate("""() => {
        const c = document.getElementById('selection-select-all');
        return {checked: c.checked, indeterminate: c.indeterminate};
    }""")
    assert state == {"checked": False, "indeterminate": True}

    _set_selection(page, [0, 1, 2])
    state = page.evaluate("""() => {
        const c = document.getElementById('selection-select-all');
        return {checked: c.checked, indeterminate: c.indeterminate};
    }""")
    assert state == {"checked": True, "indeterminate": False}


def test_panel_select_all_click_selects_then_clears(
        page, flask_server, water_xyz_file):
    """Clicking the select-all checkbox toggles between select-all
    and clear-selection per the on-change handler in
    selection-panel.js."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    page.locator("#selection-select-all").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getSelected().length === 3"
    )
    assert _get_selection(page) == [0, 1, 2]
    page.locator("#selection-select-all").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getSelected().length === 0"
    )


def test_panel_assign_target_new_option_reveals_label_input(
        page, flask_server, water_xyz_file):
    """Picking the '+ new region label…' option un-hides the
    free-text input so the user can type a label.  (panel.js
    renderAssignVisibility.)"""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    new_input = page.locator("#selection-assign-new-label")
    assert new_input.is_hidden(), (
        "new-label input should start hidden when target is a built-in"
    )
    page.locator("#selection-assign-target").select_option("__new__")
    page.wait_for_function(
        "() => document.getElementById('selection-assign-new-label')"
        "      .hidden === false"
    )
    assert new_input.is_visible()


def test_panel_assign_writes_label_to_atoms(
        page, flask_server, water_xyz_file):
    """End-to-end Assign flow: pick atoms via the store, pick a
    built-in target in the dropdown, click Assign, and confirm
    the label appears on those atoms' tag column.  This is the
    single most error-prone code path in the panel (set vs union
    vs difference semantics across onAssign / onAddToTarget /
    onRemoveFromTarget) and had no panel-level coverage.
    """
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    _set_selection(page, [0, 1])
    # The dropdown is seeded with BUILTIN_TARGETS at mount; pick
    # the canonical L-electrode region.
    page.locator("#selection-assign-target").select_option("L-electrode")
    page.locator("#selection-assign-btn").click()
    # writeLabel re-fetches atoms on success (store.js).  The
    # atom-list rows will re-render with the new label tag; wait
    # for that DOM signal so we don't read mid-render.
    page.wait_for_function(
        '() => {'
        '  const rows = document.querySelectorAll('
        '    \'#selection-atom-list tr[data-atom-index]\');'
        '  if (rows.length !== 3) return false;'
        '  const r0 = rows[0].querySelector(".col-labels");'
        '  const r1 = rows[1].querySelector(".col-labels");'
        '  return r0 && r1'
        '         && r0.textContent.includes("L-electrode")'
        '         && r1.textContent.includes("L-electrode");'
        '}'
    )
    # And the third atom is NOT labelled (Assign is REPLACE
    # semantics on the target, scoped to the current selection;
    # the unselected atom shouldn't have suddenly gained the tag).
    third_text = page.evaluate(
        '() => document.querySelector('
        '  \'#selection-atom-list tr[data-atom-index="2"] .col-labels\''
        ').textContent'
    )
    assert "L-electrode" not in third_text, (
        f"unselected atom 2 should not have been labelled; "
        f"got tag text {third_text!r}"
    )


def test_panel_filter_drafts_persist_through_file_switch(
        page, flask_server, water_xyz_file, tmp_path):
    """Spec contract: filter drafts (state.filters) persist when
    setSourceFile swaps the structure, even though state.selection
    clears.  Future "defensively wipe filters on file switch"
    refactors should fail this test loudly.

    Verifies the comment block at lines 280-282 of
    selection-bootstrap.js ('filter drafts persist; the user must
    explicitly applyFilter() against the new structure').
    """
    other = tmp_path / "other.xyz"
    other.write_text("3\nh2o-2\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n")
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    # Add a filter draft from the store side -- equivalent to the
    # user clicking + Add filter and typing.  No applyFilter() is
    # called; the draft lives in state.filters.
    page.evaluate(
        "() => window.molbuilder.selection.store.addFilter("
        "  {kind: 'by_element', value: 'O'})"
    )
    page.wait_for_function(
        "() => window.molbuilder.selection.store"
        "      .getState().filters.length === 1"
    )
    # Switch source files through the store.
    _load_file(page, str(other), expected_atoms=3)
    drafts = page.evaluate(
        "() => window.molbuilder.selection.store.getState().filters"
    )
    assert len(drafts) == 1 and drafts[0]["kind"] == "by_element", (
        f"filter drafts were wiped on file switch; got {drafts}"
    )


def test_panel_atom_row_checkbox_toggles_store_selection(
        page, flask_server, water_xyz_file):
    """The per-row checkbox in #selection-atom-list calls
    store.toggleAtom so the store -- not panel-local state --
    is the source of truth.  Drive selection by clicking the
    checkbox; assert the store updated."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _wait_panel_ready(page)
    page.wait_for_function(
        "() => document.querySelectorAll("
        "  '#selection-atom-list tr').length === 3"
    )
    page.locator(
        '#selection-atom-list tr[data-atom-index="1"] input[type="checkbox"]'
    ).click()
    page.wait_for_function(
        "() => JSON.stringify("
        "  window.__molbuilder_modify_test.getSelected()) === '[1]'"
    )
    assert _get_selection(page) == [1]
    # And the row carries the .is-selected class so the visual state
    # follows the store.
    page.wait_for_function(
        '() => document.querySelector('
        '  \'#selection-atom-list tr[data-atom-index="1"]\''
        ').classList.contains("is-selected")'
    )


def test_clickable_survives_repeated_apply_style(
        page, flask_server, water_xyz_file):
    """``viewer.setStyle({}, ...)`` historically reset the clickable
    flag in older 3Dmol builds.  The viewer-adapter re-arms
    setClickable on every render so this regression doesn't bite,
    but pin it explicitly: cycling the rep dropdown must leave atoms
    clickable."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    # Force three style cycles to make the regression mode likely to
    # surface (each setStyle re-applies the highlight overlay path).
    # Post-Phase-6 the rep picker is a button group inside the View
    # menu's closed <details>.  Playwright auto-waits for visibility
    # on .click(); dispatch the event programmatically to bypass
    # the popover-open animation.
    for rep in ("ball-and-stick", "sphere", "stick"):
        page.locator(
            f'.mol-viewer-rep-btn[data-rep="{rep}"]'
        ).dispatch_event("click")
        page.wait_for_timeout(50)
    clickable_after = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).every((a) => a.clickable);
    }""")
    assert clickable_after, (
        "atom.clickable was reset by setStyle -- viewer->list breaks"
    )


# --------------------------------------------------------------------- #
#  M3: live |offset| readout updates client-side                        #
# --------------------------------------------------------------------- #


def test_slider_drag_updates_distance_readout(
        page, flask_server, water_xyz_file):
    """Moving a single slider updates its own value readout AND the
    composite |offset| readout, all client-side (no server roundtrip).
    This is the spec D5 contract: ''Live distance during add-atom is
    computed client-side from slider values; only commits on Apply.''"""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    # Set dx=1.5, dy=0.0, dz=0.0 via the input element directly.
    page.locator("#add-dx").evaluate(
        "(el) => { el.value = '1.5'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#add-dy").evaluate(
        "(el) => { el.value = '0'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#add-dz").evaluate(
        "(el) => { el.value = '0'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )

    assert page.locator("#add-dx-val").inner_text() == "1.50"
    assert page.locator("#add-distance").inner_text() == "1.50 Å"

    # Compose a 3-component offset; sqrt(1.0^2 + 0.5^2 + 0^2) ~= 1.118.
    page.locator("#add-dx").evaluate(
        "(el) => { el.value = '1.0'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#add-dy").evaluate(
        "(el) => { el.value = '0.5'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    readout = page.locator("#add-distance").inner_text()
    assert readout.startswith("1.12") or readout.startswith("1.11"), readout


# --------------------------------------------------------------------- #
#  M3: Apply Delete + Apply Add round-trip                              #
# --------------------------------------------------------------------- #


def test_delete_button_disabled_without_selection(
        page, flask_server, water_xyz_file):
    """The Delete button is disabled whenever the selection is empty.
    Re-enables when the store has a selection; re-disables when the
    store is cleared.  Routes through the selection store, which
    fires the subscriber that toggles button enablement."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    delete_btn = page.locator("#delete-apply")
    assert delete_btn.is_disabled()
    _set_selection(page, [1])
    assert delete_btn.is_enabled()
    _clear_selection(page)
    assert delete_btn.is_disabled()


def test_apply_delete_drops_selected_row(
        page, flask_server, water_xyz_file):
    """Select index 0 (the O), click Apply Delete -> structure shrinks
    from 3 to 2 atoms.  The two H atoms survive (verified by reading
    elements from the live viewer state)."""
    errors = _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    _set_selection(page, [0])  # O
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    elements = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map((a) => a.elem);
    }""")
    assert elements == ["H", "H"]
    # The legacy ``#atom-count`` readout was removed -- the canonical
    # "how many atoms now?" check goes through the test hook above
    # (``getNAtoms() === 2``), which already covered this test's
    # invariant.
    assert errors == [], f"JS errors during delete: {errors}"


def test_selection_store_atoms_sync_with_in_memory_edits(
        page, flask_server, water_xyz_file):
    """BOMB-0 fix (2026-06-07): after a modifier op the selection
    store's ``state.atoms`` MUST reflect the post-op structure.
    Pre-fix, modifier responses lacked the atoms list, so the
    panel re-fetched from disk via /api/selection/atoms — which
    returned the pre-op atom list because the disk hadn't been
    written yet.  Result: viewer shows 2 atoms (correct), panel
    shows 3 (stale)."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Pre-delete: 3 atoms in both the viewer AND the store.
    pre = page.evaluate("""() => ({
        viewer_n: window.__molbuilder_modify_test.getNAtoms(),
        store_n:  window.molbuilder.selection.store
                    .getState().atoms.length,
    })""")
    assert pre == {"viewer_n": 3, "store_n": 3}
    # Delete the O.
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # Post-delete: BOTH viewer AND store report 2 atoms.  Pre-fix
    # the store stayed at 3 indefinitely (no disk write).
    post = page.evaluate("""() => ({
        viewer_n:    window.__molbuilder_modify_test.getNAtoms(),
        store_n:     window.molbuilder.selection.store
                       .getState().atoms.length,
        store_elements: window.molbuilder.selection.store
                       .getState().atoms.map(a => a.element),
    })""")
    assert post["viewer_n"] == 2
    assert post["store_n"] == 2, (
        f"store atoms list out of sync; expected 2 rows after "
        f"deleting O, got {post['store_n']}"
    )
    # The post-delete atoms are the two H.
    assert post["store_elements"] == ["H", "H"]


def test_add_button_disabled_without_single_selection(
        page, flask_server, water_xyz_file):
    """The Add button is enabled only when EXACTLY ONE atom is the
    anchor (single-select).  Multi-select disables it; empty selection
    disables it.  Selection set via the store."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    add_btn = page.locator("#add-apply")
    assert add_btn.is_disabled()
    _set_selection(page, [0])
    assert add_btn.is_enabled()
    _set_selection(page, [0, 1])
    # Now two atoms selected -> Add disabled, anchor readout updated.
    assert add_btn.is_disabled()
    anchor = page.locator("#add-anchor-readout").inner_text()
    assert "exactly one" in anchor.lower()


def test_apply_button_disables_during_fetch(
        page, flask_server, water_xyz_file):
    """Visible UI lock: while an op is in flight, the Apply Delete
    button must be ``disabled`` so a double-click can't fire a
    parallel fetch.

    The op completes in ~10 ms in tests, so we slow the server
    response via ``page.route`` to keep the in-flight window open
    long enough to observe the disabled state.  ``time.sleep`` in
    the route handler delays the ``route.continue_()`` ack, which
    is exactly the delay the browser sees."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    def _slow_route(route):
        import time
        time.sleep(0.4)
        route.continue_()

    page.route("**/api/modify/delete", _slow_route)

    _set_selection(page, [0])  # select O
    btn = page.locator("#delete-apply")
    assert btn.is_enabled(), "Apply should be enabled before click"

    # Dispatch click via JS so we don't await Playwright's own
    # auto-wait-for-stable-state (which can wait for in-flight
    # network).  This way we sample the disabled state mid-flight.
    page.evaluate(
        "() => document.getElementById('delete-apply').click()"
    )
    # Read the disabled property RIGHT AFTER -- the JS click handler
    # ran synchronously up to the first ``await fetch``, so by now
    # state.inFlight is true and refreshSelectionUI has disabled the
    # button.
    disabled = page.evaluate(
        "() => document.getElementById('delete-apply').disabled"
    )
    assert disabled is True, "Apply button was not disabled mid-fetch"
    # Wait for the response to land + UI to settle.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    page.unroute("**/api/modify/delete")


def test_apply_add_atom_appends_h_at_offset(
        page, flask_server, water_xyz_file):
    """End-to-end: anchor=O, element=H, dz=1.0 -> structure grows to
    4 atoms, last atom is H.

    NB on residue tagging: the server-side ``Structure.add_atom``
    tags the new atom with ``residue_name='MOD'`` (see
    ``molbuilder/modify.py``), but the viewer-side path goes through
    an XYZ round-trip which drops residue names entirely.  The
    "MOD" tag is therefore not observable on ``atom.resn`` via 3Dmol;
    asserting on it was a stale carry-over from the PDB-flow
    prototype (~2026-04).  The element + atom-count assertions
    cover the user-observable invariant.  Per
    docs/protocols/playwright-tests.md § A4: assert on user-visible
    state, not implementation details that don't reach the user.
    """
    errors = _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    # Anchor = the O.
    _set_selection(page, [0])
    # Set dx/dy=0, dz=1.0.  The default slider value for dz is 1.0
    # so this is mostly a sanity-check that the default sticks.
    page.locator("#add-dx").evaluate(
        "(el) => { el.value = '0'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#add-dy").evaluate(
        "(el) => { el.value = '0'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    # Element field defaults to "H".  Apply.
    page.locator("#add-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 4"
    )
    last_elem = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        const atoms = v.selectedAtoms({});
        return atoms[atoms.length - 1].elem;
    }""")
    assert last_elem == "H"
    assert errors == [], f"JS errors during add_atom: {errors}"


# --------------------------------------------------------------------- #
#  Static-asset sanity                                                  #
# --------------------------------------------------------------------- #


# The legacy per-atom ``#selection-info`` table on the right-edit
# panel (and its body ``#selection-info-body``) were removed
# 2026-05-20 along with the rest of the right-column Selection
# fieldset; the new selection panel above the modify-grid carries
# the count + an actions block.  The test that pinned the table's
# row content was retired with it.


def test_axes_have_fixed_length_at_origin(
        page, flask_server, water_xyz_file):
    """The xyz axis triad must be a fixed-length compass anchored
    at the world origin.  Specifically:

      * Three arrows are drawn when ``Show xyz axes`` is checked.
      * Each arrow's geometry encodes a length of AXIS_LEN (1.5 Å).
      * The shapes count drops to zero when the box is unchecked
        and returns to 3 on re-check.

    Stronger than counting shapes alone -- that wouldn't catch a
    regression where axes scale with the molecule extent (the prior
    bug)."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Post-Phase-6: axes toggle lives inside View → Axes as a
    # button with data-action="axes".  Click handler is wired even
    # when the View menu is closed; the menu is purely cosmetic.
    axes_btn = page.locator(
        '.mol-viewer-toggle[data-action="axes"]')
    assert axes_btn.get_attribute("aria-pressed") == "true"
    # Probe the axis arrows' encoded vertex distance (3Dmol caches
    # the start/end vectors on the underlying CylinderShape; we read
    # them via the viewer's shapes array).  All three axis arrows
    # must encode a length close to 1.5 Å (AXIS_LEN); the molecule
    # bounding box is < 2 Å so a regression that re-introduced the
    # old "axes scale with molecule" bug would still pass a
    # >=3-arrows count but FAIL this length check on a larger
    # structure.
    lengths = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        const out = [];
        for (const s of (v.shapes || [])) {
            const cyl = s && s.intersectionShape && s.intersectionShape.cylinder;
            if (!cyl || !cyl.length) continue;
            for (const c of cyl) {
                const dx = c.c2.x - c.c1.x,
                      dy = c.c2.y - c.c1.y,
                      dz = c.c2.z - c.c1.z;
                out.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
        return out;
    }""")
    # 3Dmol's addArrow with ``mid: 0.85`` and total length 1.5 Å
    # builds a shaft cylinder of length 0.85 × 1.5 = 1.275 Å plus a
    # cone for the arrowhead.  We probe the shaft cylinders: three
    # axes -> three shaft cylinders all at 1.275 Å.  This guards
    # against the prior regression where axis length scaled with
    # the structure's bounding box -- a larger molecule would push
    # the lengths up far beyond 1.275.
    expected_shaft = 1.5 * 0.85
    n_axis_length = sum(1 for L in lengths if abs(L - expected_shaft) < 1e-3)
    assert n_axis_length >= 3, (
        f"expected >= 3 axis-length cylinders at {expected_shaft:.3f} Å; "
        f"got lengths {lengths}"
    )
    # Toggle off via the knob, expect axis shapes gone (the
    # selection adapter may have added its own shapes, so filter
    # to axis-length cylinders rather than asserting total == 0).
    # dispatch_event bypasses the popover-closed visibility wait.
    axes_btn.dispatch_event("click")
    page.wait_for_timeout(200)
    after_off = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        let n = 0;
        for (const s of (v.shapes || [])) {
            const cyl = s && s.intersectionShape && s.intersectionShape.cylinder;
            if (!cyl) continue;
            for (const c of cyl) {
                const dx = c.c2.x - c.c1.x,
                      dy = c.c2.y - c.c1.y,
                      dz = c.c2.z - c.c1.z;
                const L = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (Math.abs(L - 1.275) < 1e-3) n++;
            }
        }
        return n;
    }""")
    assert after_off == 0, (
        f"axis cylinders survived knob-off: {after_off}"
    )
    # Toggle back on.
    axes_btn.dispatch_event("click")
    page.wait_for_timeout(200)
    after_on = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        let n = 0;
        for (const s of (v.shapes || [])) {
            const cyl = s && s.intersectionShape && s.intersectionShape.cylinder;
            if (!cyl) continue;
            for (const c of cyl) {
                const dx = c.c2.x - c.c1.x,
                      dy = c.c2.y - c.c1.y,
                      dz = c.c2.z - c.c1.z;
                const L = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (Math.abs(L - 1.275) < 1e-3) n++;
            }
        }
        return n;
    }""")
    assert after_on >= 3, (
        f"axis cylinders did not return after knob-on: {after_on}"
    )


def test_selected_atom_adds_halo_marker_shape(
        page, flask_server, water_xyz_file):
    """Selecting an atom adds a halo overlay shape (added via the
    viewer-adapter's _addSphereOverlay path); clearing the selection
    removes it.  Behavioural contract -- counts the viewer's shape
    array before and after.  We drive selection via the store
    (toggleAtom / clearSelection) since the row-click UI is gone."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    n_before = page.evaluate(
        "() => window.__molbuilder_modify_test.getViewer().shapes.length"
    )
    _set_selection(page, [0])
    page.wait_for_function(
        f"() => window.__molbuilder_modify_test.getViewer().shapes.length"
        f" > {n_before}"
    )
    n_with_halo = page.evaluate(
        "() => window.__molbuilder_modify_test.getViewer().shapes.length"
    )
    # Selecting one atom must add at least one shape (the halo).  3Dmol
    # may decompose addSphere into multiple internal sub-shapes (e.g.
    # a wireframe sphere is built from many cylinder line segments
    # under the hood); we accept any positive delta.
    assert n_with_halo > n_before, (n_before, n_with_halo)
    _clear_selection(page)
    page.wait_for_function(
        f"() => window.__molbuilder_modify_test.getViewer().shapes.length"
        f" === {n_before}"
    )


def test_focus_molecule_button_recentres_camera(
        page, flask_server, water_xyz_file):
    """The Focus-molecule button calls viewer.zoomTo on the non-ELC
    selection and re-fits the camera.  Verify by capturing the camera
    state, panning off-axis programmatically, clicking the button,
    and asserting the camera position changes."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    initial = page.evaluate(
        "() => window.__molbuilder_modify_test.getViewer().getView()"
    )
    page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        const view = v.getView();
        view[0] += 7; view[1] -= 5;
        v.setView(view);
        v.render();
    }""")
    page.locator("#focus-molecule").click()
    after = page.evaluate(
        "() => window.__molbuilder_modify_test.getViewer().getView()"
    )
    # The pan we injected was (+7, -5) on a structure with extent ~1 Å,
    # so the focus click must move pan offsets meaningfully back
    # toward the initial values (within 0.5 Å).
    assert abs(after[0] - initial[0]) < 0.5, (after[0], initial[0])
    assert abs(after[1] - initial[1]) < 0.5, (after[1], initial[1])


def test_focus_molecule_no_op_without_structure(page, flask_server):
    """Clicking Focus-molecule with no structure loaded must not
    raise a JS error.  The handler returns early."""
    errors = _open_modify(page, flask_server)
    page.locator("#focus-molecule").click()
    page.wait_for_timeout(100)
    assert errors == [], f"JS error on focus-molecule no-op: {errors}"


def test_geom_buttons_disabled_without_structure(page, flask_server):
    """Center-at-origin and Translate buttons start disabled before a
    structure is loaded.  Mirrors the rotate / send-to-build pattern."""
    _open_modify(page, flask_server)
    _open_op_tab(page, "geom")
    assert page.locator("#center-apply").is_disabled()
    assert page.locator("#translate-apply").is_disabled()


# --------------------------------------------------------------------- #
#  M4: anchor-pair selection + Apply Orient                             #
# --------------------------------------------------------------------- #


def test_orient_button_enabled_only_with_two_anchors(
        page, flask_server, water_xyz_file):
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "pose")
    btn = page.locator("#orient-apply")
    assert btn.is_disabled(), "no selection -> Orient disabled"
    _set_selection(page, [0])
    assert btn.is_disabled(), "one selection -> still disabled"
    _set_selection(page, [0, 2])
    assert btn.is_enabled(), "two selections -> Orient enabled"
    # Anchor readout reflects both atoms.  Displayed labels are 1-based;
    # 0-based atoms 0 and 2 -> "#1" and "#3".
    readout = page.locator("#orient-anchor-readout").inner_text()
    assert "#1" in readout and "#3" in readout
    # Adding a third disables again.
    _set_selection(page, [0, 1, 2])
    assert btn.is_disabled(), "three selections -> Orient disabled"


def test_apply_orient_lays_anchor_pair_along_z(
        page, flask_server, tmp_path, monkeypatch):
    """Load a 4-atom diagonal chain, pick atoms 0 and 3, click Apply
    Orient -> the resulting xyz has atoms 0 and 3 along the z axis."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    diag_xyz = tmp_path / "diag.xyz"
    diag_xyz.write_text(
        "4\ndiag\nC 0 0 0\nC 1 1 0\nC 2 2 0\nC 3 3 0\n"
    )
    errors = _open_modify(page, flask_server)
    _load_file(page, str(diag_xyz), expected_atoms=4)

    _set_selection(page, [0, 3])
    _open_op_tab(page, "pose")
    page.locator("#orient-apply").click()
    # Wait for the response to land + UI to settle.
    page.wait_for_function(
        "() => document.querySelector('#edit-status') &&"
        " /Oriented/.test(document.querySelector('#edit-status').textContent)"
    )
    # After orient, atoms 0 and 3 must lie on the z axis (x ~ 0, y ~ 0).
    coords = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map(a => [a.x, a.y, a.z]);
    }""")
    assert abs(coords[0][0]) < 1e-3 and abs(coords[0][1]) < 1e-3, coords[0]
    assert abs(coords[3][0]) < 1e-3 and abs(coords[3][1]) < 1e-3, coords[3]
    assert errors == [], f"JS errors during orient: {errors}"


# --------------------------------------------------------------------- #
#  M4: Rotate around axis                                               #
# --------------------------------------------------------------------- #


def test_rotate_button_enabled_when_structure_loaded(
        page, flask_server, water_xyz_file):
    """Rotate doesn't need a selection; just a structure and a
    non-zero angle."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "pose")
    assert page.locator("#rotate-apply").is_enabled()


def test_apply_rotate_z_90_centroid_default(
        page, flask_server, tmp_path, monkeypatch):
    """Default pivot = centroid: +90° z-rotation about the
    centroid (1.5, 1.5, 0) maps atom 1 from (1, 1, 0) to
    (2, 1, 0).  This is the "rotate in place" default that matches
    user intent for the typical "spin this molecule N degrees" op."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    diag_xyz = tmp_path / "diag.xyz"
    diag_xyz.write_text("4\ndiag\nC 0 0 0\nC 1 1 0\nC 2 2 0\nC 3 3 0\n")
    _open_modify(page, flask_server)
    _load_file(page, str(diag_xyz), expected_atoms=4)
    _open_op_tab(page, "pose")

    page.locator("#rotate-angle").evaluate(
        "(el) => { el.value = '90'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#rotate-apply").click()
    page.wait_for_function(
        "() => /Rotated/.test(document.querySelector('#edit-status').textContent)"
    )
    coords = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map(a => [a.x, a.y, a.z]);
    }""")
    # Centroid = (1.5, 1.5, 0).  Atom 1 at (1, 1, 0) -> centroid frame
    # (-0.5, -0.5, 0) -> rotate 90° -> (0.5, -0.5, 0) -> world (2, 1, 0).
    assert abs(coords[1][0] - 2.0) < 1e-3, coords[1]
    assert abs(coords[1][1] - 1.0) < 1e-3, coords[1]
    assert abs(coords[1][2] - 0.0) < 1e-3, coords[1]


def test_apply_rotate_z_90_origin_pivot(
        page, flask_server, tmp_path, monkeypatch):
    """Pivot = origin: +90° z-rotation about the world origin maps
    atom 1 from (1, 1, 0) to (-1, 1, 0).  Pre-2026-05-10 behaviour;
    still available via the Pivot dropdown."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    diag_xyz = tmp_path / "diag.xyz"
    diag_xyz.write_text("4\ndiag\nC 0 0 0\nC 1 1 0\nC 2 2 0\nC 3 3 0\n")
    _open_modify(page, flask_server)
    _load_file(page, str(diag_xyz), expected_atoms=4)
    _open_op_tab(page, "pose")

    page.locator("#rotate-center").select_option("origin")
    page.locator("#rotate-angle").evaluate(
        "(el) => { el.value = '90'; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#rotate-apply").click()
    page.wait_for_function(
        "() => /Rotated/.test(document.querySelector('#edit-status').textContent)"
    )
    coords = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map(a => [a.x, a.y, a.z]);
    }""")
    assert abs(coords[1][0] - (-1.0)) < 1e-3, coords[1]
    assert abs(coords[1][1] - 1.0)    < 1e-3, coords[1]
    assert abs(coords[1][2] - 0.0)    < 1e-3, coords[1]


def test_undo_disabled_after_non_slab_ops(
        page, flask_server, water_xyz_file):
    """Undo is scoped to electrode-slab ops only.  Loading water
    and then running a delete (a non-slab op) must NOT enable the
    Undo button -- it lives in the Junction subtab and only the
    Add Electrode flow pushes history."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "junction")
    assert page.locator("#undo-op").is_disabled()
    # Delete the oxygen on the Atom subtab.  Even after a successful
    # mutation, undo stays disabled because deletes don't push
    # history under the slab-only policy.
    _open_op_tab(page, "atom")
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    _open_op_tab(page, "junction")
    assert page.locator("#undo-op").is_disabled()


def test_undo_after_electrode_op_restores_pre_slab_structure(
        page, flask_server, water_xyz_file):
    """End-to-end positive Undo path.  Apply an anchorless pair-mode
    electrode op (no atom selection); atom count grows.  Click Undo;
    atom count returns to 3.  Undo button disables again at depth 0."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "junction")
    # Set m=n=2, layers=1 -> 4 Au atoms per side -> +8 atoms.
    for input_id, val in [("elc-m", "2"), ("elc-n", "2"), ("elc-layers", "1")]:
        page.evaluate(
            "(args) => {"
            "  const el = document.getElementById(args.id);"
            "  el.value = args.val;"
            "  el.dispatchEvent(new Event('input', {bubbles: true}));"
            "}",
            {"id": input_id, "val": val},
        )
    # Anchorless pair mode: no selection needed.
    page.locator("#elc-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 11"
    )
    undo = page.locator("#undo-op")
    assert undo.is_enabled()
    undo.click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 3"
    )
    assert undo.is_disabled()


def test_geom_translate_then_center_returns_centroid_to_origin(
        page, flask_server, water_xyz_file):
    """End-to-end Geom subtab: translate the structure by (5, 0, 0),
    then click Center-at-origin and confirm the centroid lands on
    (0, 0, 0).  We don't assume the water fixture starts centred --
    only that Center is the inverse of any prior translate."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "geom")
    # Capture the O-atom x-coordinate before the translate so we can
    # detect when the response has been applied (xyz changes ->
    # state.positions[0] is rebuilt).
    x0 = page.evaluate(
        "() => window.__molbuilder_modify_test.getState().positions[0][0]"
    )
    page.locator("#translate-dx").fill("5")
    page.locator("#translate-apply").click()
    page.wait_for_function(
        f"() => {{"
        f"  const s = window.__molbuilder_modify_test.getState();"
        f"  if (!s.positions.length) return false;"
        f"  return Math.abs(s.positions[0][0] - ({x0} + 5)) < 1e-3;"
        f"}}"
    )
    # Now click Center: centroid must come back to (0, 0, 0).
    page.locator("#center-apply").click()
    page.wait_for_function(
        "() => {"
        "  const s = window.__molbuilder_modify_test.getState();"
        "  const N = s.positions.length;"
        "  if (!N) return false;"
        "  const cx = s.positions.reduce((a,p)=>a+p[0], 0) / N;"
        "  const cy = s.positions.reduce((a,p)=>a+p[1], 0) / N;"
        "  const cz = s.positions.reduce((a,p)=>a+p[2], 0) / N;"
        "  return Math.abs(cx) < 1e-6"
        "      && Math.abs(cy) < 1e-6"
        "      && Math.abs(cz) < 1e-6;"
        "}"
    )


def test_geom_translate_zero_offset_is_rejected(
        page, flask_server, water_xyz_file):
    """Apply Translate with all-zero Δ's must surface a friendly
    error rather than firing a no-op POST."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _open_op_tab(page, "geom")
    # All defaults are 0; click Apply without changing anything.
    page.locator("#translate-apply").click()
    page.wait_for_function(
        "() => document.querySelector('#edit-status').textContent"
        ".indexOf('Nothing to translate') !== -1"
    )


def test_modify_page_resources_load_with_200(page, flask_server):
    """Every static asset and CDN script the page references should
    return HTTP 200.  Catches a moved CDN URL or a missing local file
    that the existing pytest test (which only checks the HTML body)
    doesn't notice."""
    failures = []
    page.on("response", lambda r: (
        failures.append((r.status, r.url))
        if r.status >= 400 else None
    ))
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_load_state("networkidle")
    # Filter out the CDN host (network-flake-prone in CI); only assert
    # on locally-served paths.
    local_failures = [
        (s, u) for (s, u) in failures
        if "127.0.0.1" in u
    ]
    assert not local_failures, (
        f"local-asset 4xx/5xx responses: {local_failures}"
    )


# --------------------------------------------------------------------- #
#  Phase 1 state persistence: structure survives tab navigation.        #
#                                                                       #
#  Build, Watch, and Modify are separate Flask routes, so clicking      #
#  between them is a full page reload.  JS closure state dies on each   #
#  reload; sessionStorage doesn't.  The Phase 1 implementation in       #
#  modify/viewer.js + viewer.js writes the built/loaded structure to    #
#  sessionStorage on ``pagehide`` and restores it on the next page      #
#  load.  These tests exercise the round-trip in a real browser.        #
# --------------------------------------------------------------------- #


def test_modify_structure_survives_navigation_to_watch_and_back(
        page, flask_server, water_xyz_file):
    """Load a structure on Modify, click off to /results, then click
    the Modify tab again.  The structure must still be present in
    the viewer state -- without Phase 1 it would be empty (closure
    state was destroyed on the navigation).  /watch was retired
    2026-05-19; /results is the canonical "other tab" for
    session-storage round-trip tests."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Sanity: 3 atoms are present before we leave.
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 3
    page.locator('a.app-tab[href="/results"]').click()
    page.wait_for_url(f"{flask_server}/results")
    # And back.
    page.locator('a.app-tab[href="/molbuilder"]').click()
    page.wait_for_url(f"{flask_server}/molbuilder")
    # Structure restored by Phase 1 sessionStorage round-trip.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 3"
    )
    # First atom is still O (not e.g. "" if restore mangled metadata).
    first_elem = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({})[0].elem;
    }""")
    assert first_elem == "O"


def test_load_button_readout_switches_picked_to_loaded(
        page, flask_server, water_xyz_file):
    """BOMB-7 fix (2026-06-07): after a successful Load, the readout
    MUST switch from "Picked: X" to "Loaded: X" and the button
    MUST disable so the user can't replay the entire load pipeline
    by accident.

    Pre-fix the readout always said "Picked: water.xyz" — same
    text before and after Load, button stayed enabled, and a stray
    click would re-fetch + re-render the SAME file.
    """
    import os as _os
    _open_modify(page, flask_server)
    # Sidebar pick → candidate set, readout says "Picked", button
    # enabled.
    water_dir = _os.path.dirname(water_xyz_file)
    page.evaluate(
        """(c) => window.molbuilder.projects.setShared(c.dir, c.file)""",
        {"dir": water_dir, "file": water_xyz_file},
    )
    page.wait_for_function(
        "() => !document.getElementById('load-candidate-btn').disabled"
    )
    pre = page.locator("#load-candidate-readout").inner_text()
    assert pre.startswith("Picked:"), (
        f"pre-load readout should say 'Picked:'; got {pre!r}"
    )
    # Click Load → commit + viewer renders.
    page.locator("#load-candidate-btn").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 3"
    )
    # Wait for the post-load _refreshLoadUI fanout to reach the
    # readout (subscriber call after store.adoptSession).
    page.wait_for_function(
        "() => document.getElementById('load-candidate-readout')"
        "        .textContent.startsWith('Loaded:')"
    )
    # Button now disabled — the loaded file IS the candidate; no
    # point in re-firing the pipeline.
    assert page.locator("#load-candidate-btn").is_disabled()


def test_canvas_dirty_bit_survives_navigation_to_other_tab_and_back(
        page, flask_server, water_xyz_file):
    """BOMB-2 fix (2026-06-07): the canvas-state ``dirty`` flag MUST
    survive a navigation away and back.

    Pre-fix, restoreModifyState always called ``cs.setStructure(...)``
    on bfcache restore — which RESETS the dirty bit to false.  A
    user who had unsaved edits would silently see them marked clean
    after a round trip, and a subsequent Load wouldn't fire the
    warning modal.  Fix: skip the setStructure call when canvas-
    state's own sessionStorage mirror already holds the same bytes."""
    _open_modify(page, flask_server)
    _load_water_via_button(page, water_xyz_file)
    # Make a modification — Delete the O.  postOp marks the canvas
    # dirty (this is the bit we're going to verify survives the
    # round trip).
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    pre_nav = page.evaluate("""() => ({
        dirty:     window.molbuilder.structureCanvas.isDirty(),
        n_atoms:   window.__molbuilder_modify_test.getNAtoms(),
    })""")
    assert pre_nav == {"dirty": True, "n_atoms": 2}, (
        f"pre-nav state should be dirty + 2 atoms; got {pre_nav!r}"
    )
    # Navigate to /results.
    page.locator('a.app-tab[href="/results"]').click()
    page.wait_for_url(f"{flask_server}/results")
    # And back.  restoreModifyState fires on mount; the BOMB-2 fix
    # gates the canvas-state hydrate on bytes-match.
    page.locator('a.app-tab[href="/molbuilder"]').click()
    page.wait_for_url(f"{flask_server}/molbuilder")
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    post_nav = page.evaluate("""() => ({
        dirty:     window.molbuilder.structureCanvas.isDirty(),
        n_atoms:   window.__molbuilder_modify_test.getNAtoms(),
    })""")
    assert post_nav == {"dirty": True, "n_atoms": 2}, (
        f"post-nav state should preserve dirty=true; got {post_nav!r}.  "
        f"Pre-BOMB-2-fix this returned dirty=false because "
        f"restoreModifyState unconditionally called cs.setStructure "
        f"which resets the dirty bit."
    )


def test_modify_selection_survives_navigation(
        page, flask_server, water_xyz_file):
    """Select an atom on Modify, navigate away, come back.  The
    selection survives -- saveModifyState() snapshots state.selected
    via selectedIndices(), and restoreModifyState() pushes it back
    into the store with setSelection().  Verified via the test hook
    which reads the store live."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _set_selection(page, [1])
    page.locator('a.app-tab[href="/results"]').click()
    page.wait_for_url(f"{flask_server}/results")
    page.locator('a.app-tab[href="/molbuilder"]').click()
    page.wait_for_url(f"{flask_server}/molbuilder")
    # Structure restored.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 3"
    )
    # And the selection restored into the store.
    page.wait_for_function(
        "() => JSON.stringify("
        "  window.__molbuilder_modify_test.getSelected()) === '[1]'"
    )
    assert _get_selection(page) == [1]


def test_delete_remaps_high_index_selection_via_selection_remap(
        page, flask_server, water_xyz_file):
    """End-to-end pin for the **selection-drift bug** the whole
    workspace-state migration was driving toward.

    Pre-fix sequence (the bug):
      * User selects atom 2 (one of the H's) in a 3-atom water.
      * User deletes atom 0 (the O).
      * Server returns a 2-atom structure; client's naive
        ``selection.filter(i < 2)`` drops the old index 2 because
        it's out of range in the new 2-atom structure.
      * Selection ends up EMPTY even though the user-picked atom
        (old #2 = surviving H) is still in the workspace as new
        index 1.

    Post-fix (this test):
      * Server emits ``extra.selection_remap = [null, 0, 1]`` per
        Phase 3 of the migration.
      * Client's ``_applyWorkspacePayload`` captures the PRE-op
        selection BEFORE adoptAtoms' destructive filter, then
        applies the remap: 2 → 1.
      * Selection is correctly ``[1]`` in the new structure.

    Without the Phase 3+ correctness fix in
    ``_applyWorkspacePayload`` (the pre-capture of selection
    before the hook runs), this test asserts ``[1]`` but the
    client would produce ``[]``."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Select atom 2 (the second H — the HIGH index that the naive
    # filter would silently drop after deleting atom 0).
    _set_selection(page, [2])
    assert _get_selection(page) == [2]

    # Delete atom 0 (the O).  Surviving atoms: old [1, 2] → new [0, 1].
    page.evaluate(
        "() => window.molbuilder.workspace.applyOp('delete', "
        "       {indices: [0]})"
    )
    # Wait for the structure to shrink AND the selection to remap.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # The fix: surviving atom (old index 2) is now at new index 1.
    selection = _get_selection(page)
    assert selection == [1], (
        f"selection-drift bug: expected [1] after remap (old atom 2 "
        f"→ new atom 1), got {selection}.  This is the exact bug "
        f"the workspace-state migration's selection_remap was "
        f"supposed to fix end-to-end."
    )


def test_modify_state_after_op_survives_navigation(
        page, flask_server, water_xyz_file):
    """Apply Delete on Modify (state.xyz is now the post-delete
    structure), navigate away, come back.  The 2-atom post-delete
    state must be what restores -- NOT the 3-atom pre-load."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    _set_selection(page, [0])  # O
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    page.locator('a.app-tab[href="/results"]').click()
    page.wait_for_url(f"{flask_server}/results")
    page.locator('a.app-tab[href="/molbuilder"]').click()
    page.wait_for_url(f"{flask_server}/molbuilder")
    # Post-delete state is what restores.
    page.wait_for_function(
        "() => window.__molbuilder_modify_test"
        "      && window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    elements = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map((a) => a.elem);
    }""")
    assert elements == ["H", "H"]


def test_modify_handles_storage_quota_exceeded_gracefully(
        page, flask_server, water_xyz_file):
    """If sessionStorage is full or disabled (private mode in some
    browsers throws on setItem), the save path must catch the error
    and not crash the page.  Mocked by stubbing setItem to throw.

    Post-Phase-8 collapse (2026-06-08): the canonical key is
    ``molbuilder.workspace.v1`` owned by the workspace dispatcher.
    Pin that THE dispatcher's _persistToSession catches the throw
    and warns to console without leaking a pageerror."""
    errors = _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    # Wrap setItem so any save attempt to the workspace key throws.
    page.evaluate("""() => {
        const orig = sessionStorage.setItem.bind(sessionStorage);
        sessionStorage.setItem = (k, v) => {
            if (k === "molbuilder.workspace.v1") {
                throw new DOMException("Quota exceeded", "QuotaExceededError");
            }
            return orig(k, v);
        };
    }""")
    # Trigger the save by dispatching pagehide manually -- normally
    # fired by the browser on navigation.  Catch any thrown error.
    page.evaluate("""() => {
        window.dispatchEvent(new Event("pagehide"));
    }""")
    # No JS errors should reach the page-error handler; the save
    # function catches QuotaExceededError and warns to console
    # (which is filtered to error-only, so a console.warn doesn't
    # count).  Page is still alive.
    page.wait_for_timeout(150)
    page_errors = [e for e in errors if e[0] == "pageerror"]
    assert page_errors == [], f"unexpected pageerror: {page_errors}"
    # Structure still rendered; the page didn't break.
    assert page.evaluate(
        "() => window.__molbuilder_modify_test.getNAtoms()"
    ) == 3


# --------------------------------------------------------------------- #
#  Phase 1 (Build side): structure survives Build <-> Watch <-> Build  #
# --------------------------------------------------------------------- #


def test_build_structure_survives_navigation(
        page, flask_server, water_xyz_file):
    """Load a structure on the Optimization tab, navigate away,
    come back.  The 3D viewer must still show the molecule, the
    info panel must still show n_atoms, and the Generate buttons
    must still be enabled — without the sessionStorage handoff
    the user lost everything and had to re-pick the file.

    Post-task-295 (2026-06-08): the Build form is gone; the
    handoff is the shared Projects-sidebar pointer
    (sessionStorage.molbuilder.current_file).  Auto-load on
    mount picks it up on re-entry, restoring viewer + info +
    Generate-enabled."""
    # Prime the cross-tab handoff via add_init_script so the
    # sidebar pointer is in sessionStorage BEFORE any page script
    # runs; mount-time auto-load then commits the structure.
    import os as _os
    page.add_init_script(
        "sessionStorage.setItem('molbuilder.current_file',"
        f" {repr(water_xyz_file)});"
        "sessionStorage.setItem('molbuilder.current_dir',"
        f" {repr(_os.path.dirname(water_xyz_file))});"
    )
    page.goto(f"{flask_server}/structure-optimization")
    page.wait_for_selector("#load-from-sidebar-btn")
    page.wait_for_function(
        "() => !document.getElementById('generate-fdf').disabled",
        timeout=8000,
    )
    n_atoms_before = page.locator("#info-atoms").inner_text()
    assert n_atoms_before and n_atoms_before != "—"
    # Navigate away to /results then back.  The handoff pointer
    # plus auto-load is what restores state across the round-trip.
    page.locator('a.app-tab[href="/results"]').click()
    page.wait_for_url(f"{flask_server}/results")
    page.locator('a.app-tab[href="/structure-optimization"]').click()
    page.wait_for_url(f"{flask_server}/structure-optimization")
    page.wait_for_function(
        "() => !document.getElementById('generate-fdf').disabled",
        timeout=5000,
    )
    n_atoms_after = page.locator("#info-atoms").inner_text()
    assert n_atoms_after == n_atoms_before, (
        f"atom count not restored: was {n_atoms_before!r}, "
        f"now {n_atoms_after!r}"
    )
    assert page.locator("#generate-fdf").is_enabled()
    assert page.locator("#generate-pyscf").is_enabled()


def test_build_structure_state_round_trips_modify(
        page, flask_server, water_xyz_file):
    """Modify's pagehide save persists into the workspace
    dispatcher's own key (molbuilder.workspace.v1).  Pin the key
    name so a future rename / accidental collapse trips here.

    Post-task-295 (2026-06-08): the Build form is gone; the
    Optimization tab consumes the shared Projects-sidebar pointer
    (molbuilder.current_file) as its sole structure entrance.
    That pointer is owned by the sidebar — any file selection,
    on either tab, writes it.  So this test no longer asserts
    cross-tab key isolation (the sidebar pointer is intentionally
    shared); it only pins Modify's own workspace key."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    page.evaluate("() => window.dispatchEvent(new Event('pagehide'))")
    has_workspace_key = page.evaluate(
        "() => sessionStorage.getItem('molbuilder.workspace.v1') !== null"
    )
    assert has_workspace_key is True, (
        "Modify's pagehide save should land in molbuilder.workspace.v1"
    )


def test_measurement_readout_shows_xyz_distance_angle(
        page, flask_server, water_xyz_file):
    """Selection-driven measurement readout (task #298): the
    Selection panel renders a one-line readout that follows the
    current selection — xyz for 1 atom, distance for 2, angle for
    3 (with the middle atom as vertex), hidden for 0 or 4+ atoms.

    Pin the contract end-to-end on water (3 atoms, geometry
    chosen by the fixture so the math has nice round numbers)."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    meas_visible = lambda: page.evaluate(
        "() => !document.getElementById('selection-measurement-overlay').hidden"
    )
    meas_text = lambda: page.locator("#selection-measurement-overlay").inner_text()

    # 0 atoms — hidden.
    assert meas_visible() is False

    # 1 atom — xyz of the O.
    _set_selection(page, [0])
    page.wait_for_function(
        "() => !document.getElementById('selection-measurement-overlay').hidden"
    )
    text = meas_text()
    # Atom label uses 1-based indexing per the design.
    assert "O #1" in text, f"expected 'O #1' in {text!r}"
    assert "(0.000, 0.000, 0.000)" in text, (
        f"expected xyz coords in {text!r}"
    )

    # 2 atoms — O–H distance.  Fixture is the standard water with
    # 0.957 Å O–H bond; round-trips through the math to 4 decimals.
    _set_selection(page, [0, 1])
    page.wait_for_function(
        "() => /distance/.test("
        "  document.getElementById('selection-measurement-overlay').dataset.kind"
        ")"
    )
    text = meas_text()
    assert "|O #1" in text and "H #2|" in text, (
        f"expected '|O #1 – H #2|' in {text!r}"
    )
    assert "0.9570" in text, f"expected 0.9570 Å in {text!r}"

    # 3 atoms — H–O–H angle.  The store carries a click-order
    # shadow (``pickOrder``) so the measurement module knows which
    # atom was the SECOND click — that's the vertex.  Set click
    # order [1, 0, 2] = "H, then O (vertex), then H".
    _set_selection(page, [1, 0, 2])
    page.wait_for_function(
        "() => document.getElementById('selection-measurement-overlay')"
        "      .dataset.kind === 'angle'"
    )
    text = meas_text()
    # Vertex (O) is the middle of the display; the H's bracket it.
    assert "O #1" in text and text.count("H #") == 2, (
        f"expected 'H – O – H' shape in {text!r}"
    )
    # And the vertex really is the middle click.
    vertex = page.evaluate(
        "() => window.molbuilder.selection.measurements.compute("
        "  window.molbuilder.selection.store.getState().selection,"
        "  window.molbuilder.selection.store.getState().atoms,"
        "  window.molbuilder.selection.positionsProvider(),"
        "  window.molbuilder.selection.store.getState().pickOrder"
        ").vertexIndex"
    )
    assert vertex == 0, (
        f"vertex should be O (index 0, middle of pickOrder), got {vertex}"
    )
    # Allow some slack on the exact angle — the fixture's coords
    # give ~104.5°, which the math reproduces.
    import re
    m = re.search(r"=\s*([0-9.]+)°", text)
    assert m, f"expected 'X.X°' in {text!r}"
    assert 100.0 <= float(m.group(1)) <= 110.0, (
        f"H–O–H angle should be ~104.5°, got {m.group(1)}° in {text!r}"
    )


def test_modify_chain_ids_round_trip_through_op(
        page, flask_server, water_xyz_file):
    """Regression: ``state.chain_ids`` was previously never declared
    in the JS state initializer, so every body sent ``chain_ids:
    undefined`` and the server fell back to defaults silently.
    Verify the JS-side state actually carries chain_ids, and that
    they survive an op round-trip without resetting.

    Probed via the test hook's ``getState()`` (the legacy atom-list
    DOM that this test used to read was retired 2026-05-20)."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # The /api/build/load response sets chain_ids = ["A", "A", "A"];
    # the JS must carry them through.
    chain_ids = page.evaluate(
        "() => window.__molbuilder_modify_test.getState().chain_ids"
    )
    assert isinstance(chain_ids, list) and len(chain_ids) == 3
    # Run a delete op (which round-trips chain_ids through the body).
    _set_selection(page, [2])  # last H
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    # If chain_ids had been undefined, the server would have applied
    # default ["A"]*n; that's still the value here so we can't catch
    # the regression by comparing values directly -- but we can
    # assert chain_ids is still a real array of length n_atoms in
    # the post-op state.
    post = page.evaluate(
        "() => window.__molbuilder_modify_test.getState().chain_ids"
    )
    assert isinstance(post, list) and len(post) == 2


def test_modify_uses_sessionstorage_not_localstorage(
        page, flask_server, water_xyz_file):
    """Document the persistence boundary: ``sessionStorage`` (clears
    on browser close) NOT ``localStorage`` (persists across browser
    restarts).  This is the spec-recorded design choice -- molecular
    structures aren't sensitive but a "session ends -> fresh start"
    default fits a scientific-tool feel.

    Post-Phase-8 collapse (2026-06-08): the canonical key is
    ``molbuilder.workspace.v1`` owned by the workspace dispatcher.
    Pin that THAT key lives in sessionStorage and NOT in
    localStorage."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    page.evaluate("() => window.dispatchEvent(new Event('pagehide'))")
    in_session = page.evaluate(
        "() => sessionStorage.getItem('molbuilder.workspace.v1') !== null"
    )
    in_local   = page.evaluate(
        "() => localStorage.getItem('molbuilder.workspace.v1') !== null"
    )
    assert in_session is True, "save target should be sessionStorage"
    assert in_local   is False, "save MUST NOT leak into localStorage"


# --------------------------------------------------------------------- #
#  M5: electrode panel + Send-to-Build handoff                          #
# --------------------------------------------------------------------- #


_SS_XYZ_FOR_E2E = (
    "2\nss-pair\n"
    "S 0 0 -2\n"
    "S 0 0  2\n"
)


@pytest.fixture
def ss_pair_xyz_file(tmp_path, monkeypatch):
    """A 2-atom S pair on the z axis -- the canonical test fixture for
    the symmetric-electrode workflow (stands in for a relaxed BDT
    where the user has already deleted the thiol H caps).

    Registers ``tmp_path`` as a picker root so
    ``store.setSourceFile(path)`` -> ``/api/selection/atoms`` accepts
    the file -- without this, the panel atoms list never populates
    and any ``wait_for_function`` on the load times out.  Mirrors
    the ``water_xyz_file`` fixture's setup.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "ss_pair.xyz"
    p.write_text(_SS_XYZ_FOR_E2E)
    return str(p)


def test_electrode_apply_button_state_matches_selection_and_mode(
        page, flask_server, ss_pair_xyz_file):
    """Pair mode: 0 atoms selected = enabled (canonical origin-
    centred placement), 1 = disabled (ambiguous), 2 = enabled
    (legacy anchor-pair midpoint).  Single mode needs exactly 1."""
    _open_modify(page, flask_server)
    _load_file(page, ss_pair_xyz_file, expected_atoms=2)
    _open_op_tab(page, "junction")
    btn = page.locator("#elc-apply")
    # No selection in pair mode -> enabled (canonical origin-centred).
    assert btn.is_enabled()
    # One atom selected -> disabled (ambiguous: not 0, not 2).
    _set_selection(page, [0])
    assert btn.is_disabled()
    # Two atoms selected -> enabled (legacy anchor-pair midpoint mode).
    _set_selection(page, [0, 1])
    assert btn.is_enabled()
    # Switch to single mode -> two anchors is wrong; disabled.
    page.locator("#elc-mode").select_option("single")
    assert btn.is_disabled()
    # One anchor in single mode -> enabled.
    _set_selection(page, [0])
    assert btn.is_enabled()


def test_electrode_gap_label_tracks_mode(
        page, flask_server, ss_pair_xyz_file):
    """The gap-slider label reads "gap" in pair mode (the canonical
    electrode-to-electrode distance) and "contact" in single mode
    (anchor-to-closest-layer)."""
    _open_modify(page, flask_server)
    _load_file(page, ss_pair_xyz_file, expected_atoms=2)
    _open_op_tab(page, "junction")
    label = page.locator("#elc-gap-label")
    assert label.inner_text() == "gap"
    page.locator("#elc-mode").select_option("single")
    assert label.inner_text() == "contact"
    page.locator("#elc-mode").select_option("symmetric")
    assert label.inner_text() == "gap"


def test_electrode_side_picker_only_visible_in_single_mode(
        page, flask_server, ss_pair_xyz_file):
    """Pair mode hides the +z/-z side picker (both slabs are placed
    automatically); single mode shows it."""
    _open_modify(page, flask_server)
    _load_file(page, ss_pair_xyz_file, expected_atoms=2)
    _open_op_tab(page, "junction")
    side_row = page.locator("#elc-side-row")
    assert side_row.is_hidden()
    page.locator("#elc-mode").select_option("single")
    assert side_row.is_visible()


def test_apply_electrode_pair_mode_builds_au_junction(
        page, flask_server, ss_pair_xyz_file):
    """End-to-end: 2-atom S pair -> select both -> Apply pair-mode
    Au(111) 2x2x1 -> structure grows to 10 atoms (2 S + 8 Au).
    Elements are read from the live viewer state."""
    errors = _open_modify(page, flask_server)
    _load_file(page, ss_pair_xyz_file, expected_atoms=2)
    _set_selection(page, [0, 1])
    _open_op_tab(page, "junction")
    # Set 2x2x1 size for a tractable junction.
    for input_id, val in [("elc-m", "2"), ("elc-n", "2"), ("elc-layers", "1")]:
        page.evaluate(
            "(args) => {"
            "  const el = document.getElementById(args.id);"
            "  el.value = args.val;"
            "  el.dispatchEvent(new Event('input', {bubbles: true}));"
            "}",
            {"id": input_id, "val": val},
        )
    page.locator("#elc-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 10"
    )
    elements = page.evaluate("""() => {
        const v = window.__molbuilder_modify_test.getViewer();
        return v.selectedAtoms({}).map((a) => a.elem);
    }""")
    assert sum(1 for e in elements if e == "Au") == 8
    assert sum(1 for e in elements if e == "S")  == 2
    assert errors == [], f"JS errors during electrode apply: {errors}"


def test_send_to_build_writes_sidebar_pointer(
        page, flask_server, ss_pair_xyz_file):
    """Task #294 (2026-06-08): the Send-to-Build button is now a
    "save-first" handoff.  Clicking it sets
    ``sessionStorage["molbuilder.current_file"]`` to the project-
    saved path + navigates to /structure-optimization.  No more
    ``builder-structure`` payload — the optimization tab loads
    the structure from the project file on disk via its sidebar-
    driven Load button (task #295).

    The button is only enabled when the workspace is clean +
    backed by a file path.  ``_load_file`` populates canvas-state
    via the sidebar Load workflow, so ``source.file`` is set and
    ``isDirty()=false`` immediately after a fresh load — the
    Send button is enabled without an explicit Save click."""
    _open_modify(page, flask_server)
    _load_file(page, ss_pair_xyz_file, expected_atoms=2)
    # Send button should be enabled — fresh load = clean + has source.file.
    page.wait_for_function(
        "() => !document.getElementById('send-to-build').disabled",
        timeout=5000,
    )
    with page.expect_navigation():
        page.locator("#send-to-build").click()
    expected = flask_server.rstrip("/") + "/structure-optimization"
    assert page.url.rstrip("/") == expected, (
        f"expected redirect to {expected}, got {page.url}"
    )
    # Sidebar pointer is set; the legacy builder-structure payload is NOT.
    current_file = page.evaluate(
        "() => sessionStorage.getItem('molbuilder.current_file')"
    )
    assert current_file == ss_pair_xyz_file, (
        f"send-to-build did not point the sidebar at the saved file; "
        f"got {current_file!r}"
    )
    legacy = page.evaluate(
        "() => sessionStorage.getItem('builder-structure')"
    )
    assert legacy is None, (
        f"legacy builder-structure key should not be written by the "
        f"save-first handoff; got {legacy!r}"
    )


def test_send_to_build_disabled_when_workspace_is_dirty(
        page, flask_server, water_xyz_file):
    """The save-first rule (task #294, 2026-06-08): Send is disabled
    while the canvas is dirty (modifier ops since the last
    save/load) — even though the workspace IS associated with a
    file.  User must Save first."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Fresh load: Send enabled.
    page.wait_for_function(
        "() => !document.getElementById('send-to-build').disabled",
        timeout=5000,
    )
    # Run a modifier op → canvas dirty → Send re-disables.
    _set_selection(page, [0])
    page.locator("#delete-apply").click()
    page.wait_for_function(
        "() => window.__molbuilder_modify_test.getNAtoms() === 2"
    )
    page.wait_for_function(
        "() => document.getElementById('send-to-build').disabled === true",
        timeout=5000,
    )


def test_op_subtabs_default_to_atom_and_swap_on_click(
        page, flask_server, water_xyz_file):
    """The edit panel splits the ops into Atom / Pose / Geom /
    Junction sub-tabs.  Default-active is Atom (the most common
    starting op).  Clicking Pose shows orient + rotate; clicking
    Junction shows the electrode panel.

    The Send-to-Build handoff is OUTSIDE the op-tabs (2026-05-21)
    so it's reachable from any sub-tab.  Pinned by
    test_send_to_build_visible_across_all_op_subtabs below."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    # Default visible op-panel is the atom one.
    atom_panel = page.locator('.optab-panel[data-op-panel="atom"]')
    pose_panel = page.locator('.optab-panel[data-op-panel="pose"]')
    junction_panel = page.locator('.optab-panel[data-op-panel="junction"]')
    assert atom_panel.is_visible()
    assert pose_panel.is_hidden()
    assert junction_panel.is_hidden()

    # Click Pose -> only that panel becomes visible.
    page.locator('.optab[data-op-tab="pose"]').click()
    assert pose_panel.is_visible()
    assert atom_panel.is_hidden()
    assert junction_panel.is_hidden()
    # Apply Orient is still in the DOM (just hidden was Junction tab).
    assert page.locator("#orient-apply").is_visible()

    # Click Junction.
    page.locator('.optab[data-op-tab="junction"]').click()
    assert junction_panel.is_visible()
    assert page.locator("#elc-apply").is_visible()


def test_send_to_build_visible_across_all_op_subtabs(
        page, flask_server, water_xyz_file):
    """The Send-to-Build button is the tab-level handoff action;
    it lives in ``.modify-footer .modify-handoff`` at the bottom
    of the Modify section so the user can ship the current
    structure to /structure-optimization regardless of which
    edit-panel sub-tab (Atom / Pose / Geom / Junction) is active.

    Post-2026-06-08 restructure: the button moved from the
    viewer-card to the Modify section's footer, alongside the
    Save button.  Both are structure-level workflow actions and
    sit next to each other at the bottom of the workflow."""
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)

    send_btn = page.locator("#send-to-build")
    # Hosted in the Modify section's footer, not the viewer card.
    assert page.locator(
        ".modify-footer .modify-handoff #send-to-build"
    ).count() == 1, (
        "Send to Build button is NOT inside .modify-footer .modify-handoff"
    )

    # Atom is the default-open op-tab; the handoff button must be
    # visible while the edit-panel is on Atom.
    assert page.locator('.optab-panel[data-op-panel="atom"]').is_visible()
    assert send_btn.is_visible(), "Send to Build hidden on Atom sub-tab"

    # Pose sub-tab.
    page.locator('.optab[data-op-tab="pose"]').click()
    assert send_btn.is_visible(), "Send to Build hidden on Pose sub-tab"

    # Geom sub-tab.
    page.locator('.optab[data-op-tab="geom"]').click()
    assert send_btn.is_visible(), "Send to Build hidden on Geom sub-tab"

    # Junction sub-tab (where it used to live).
    page.locator('.optab[data-op-tab="junction"]').click()
    assert send_btn.is_visible(), "Send to Build hidden on Junction sub-tab"


# The legacy in-edit-panel ``#selection-info`` block was retired
# 2026-05-20: the new selection panel lives in its own column (left
# of the viewer in the 3-col grid) and is naturally visible across
# every op sub-tab without needing a per-sub-tab pinning test.  The
# replacement test would just assert that ``#selection-host`` is
# visible, which is already implied by the layout tests below.


def test_modify_layout_stacks_on_narrow_viewport(
        page, flask_server, water_xyz_file):
    """Post-2026-06-08 restructure: the Modify section's body
    (selection + edit side-by-side) collapses to a 1-column
    stack at viewport width <= 960px.  Use Playwright's
    set_viewport_size to drive the responsive media query and
    assert ``.modify-body``'s grid-template-columns flips to a
    single track.  Width 800px is the "tablet portrait" range
    we want to support."""
    page.set_viewport_size({"width": 800, "height": 900})
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    cols = page.evaluate(
        "() => getComputedStyle(document.querySelector('.modify-body'))"
        ".gridTemplateColumns"
    )
    # 1-column stack has exactly ONE track; side-by-side has TWO.
    n_tracks = len(cols.split())
    assert n_tracks == 1, (
        f"narrow viewport should give one track; got {n_tracks} ({cols!r})"
    )


def test_modify_layout_phone_width_no_horizontal_overflow(
        page, flask_server, water_xyz_file):
    """Phone-width viewport (360 px): the page must not produce a
    horizontal scrollbar, and the 5-child viewer toolbar must wrap to
    multiple rows.  Document body scrollWidth <= clientWidth is the
    canonical "did mobile layout break?" check -- if a child element
    overflows, scrollWidth exceeds the viewport width by exactly that
    amount.
    """
    page.set_viewport_size({"width": 360, "height": 720})
    _open_modify(page, flask_server)
    _load_water(page, water_xyz_file)
    # Page body must fit the viewport horizontally.
    overflow_px = page.evaluate(
        "() => document.documentElement.scrollWidth"
        " - document.documentElement.clientWidth"
    )
    if overflow_px > 0:
        # Diagnostic: list elements wider than the viewport so we can
        # pin the offender in the failure message instead of guessing.
        offenders = page.evaluate(
            "(vw) => Array.from(document.querySelectorAll('*'))"
            ".filter(e => e.scrollWidth > vw + 1)"
            ".slice(0, 8)"
            ".map(e => e.tagName + (e.id ? '#' + e.id : '')"
            " + (e.className && typeof e.className === 'string'"
            " ? '.' + e.className.split(' ').slice(0, 2).join('.') : '')"
            " + ' (sw=' + e.scrollWidth + ')')",
            360
        )
        raise AssertionError(
            f"phone-width page overflows horizontally by {overflow_px} px; "
            f"offenders: {offenders}"
        )
    # Viewer-controls children must wrap -- assert the toolbar height
    # exceeds a single-row height (a single row is ~32-44 px including
    # padding; multi-row should be >= 60 px on a 360 px viewport).
    toolbar_h = page.evaluate(
        "() => document.querySelector('.viewer-controls')"
        ".getBoundingClientRect().height"
    )
    assert toolbar_h >= 60, (
        f"viewer-controls did not wrap on 360 px viewport "
        f"(height = {toolbar_h:.0f} px, expected >= 60 for multi-row)"
    )


def test_send_to_build_disabled_without_structure(page, flask_server):
    """Send-to-Build button must be disabled when no structure is
    loaded -- nothing to hand off."""
    _open_modify(page, flask_server)
    assert page.locator("#send-to-build").is_disabled()


# --------------------------------------------------------------------- #
#  Watch Inspect tab: atom-pick + live distance display                 #
# --------------------------------------------------------------------- #


_MOLWATCH_ONE_STEP = (
    "# molwatch trajectory log v1\n"
    "# engine: pyscf\n"
    "==== molwatch step 0 begin ====\n"
    "step_index: 0\n"
    "n_atoms: 3\n"
    "coordinates (Ang):\n"
    "   O   0.00000000   0.00000000   0.00000000\n"
    "   H   0.95700000   0.00000000   0.00000000\n"
    "   H  -0.23900000   0.92700000   0.00000000\n"
    "energy (eV): -76.40000000\n"
    "==== molwatch step 0 end ====\n"
)

# Same one-step trajectory + the writer's clean-exit marker; the
# parser flips run_state to "finished" on this line.
_MOLWATCH_ONE_STEP_FINISHED = _MOLWATCH_ONE_STEP + "# concluded: ok\n"


# Molwatch fixture with realistic non-zero forces for the
# show-force-vectors toggle test.  Two-atom H2 with one atom pulled
# off-equilibrium so the per-atom force vector has a clear direction.
_MOLWATCH_WITH_FORCES = (
    "# molwatch trajectory log v1\n"
    "# engine: pyscf\n"
    "==== molwatch step 0 begin ====\n"
    "step_index: 0\n"
    "n_atoms: 2\n"
    "coordinates (Ang):\n"
    "   H   0.00000000   0.00000000   0.00000000\n"
    "   H   1.20000000   0.00000000   0.00000000\n"
    "energy (eV): -32.0\n"
    "forces (eV/Ang):\n"
    "   H   1.50000000   0.00000000   0.00000000\n"
    "   H  -1.50000000   0.00000000   0.00000000\n"
    "max_force (eV/Ang): 1.5\n"
    "==== molwatch step 0 end ====\n"
)


@pytest.fixture
def watch_log_file(tmp_path, monkeypatch):
    """Single-step molwatch log + tmp_path picker-root registration so
    the /results file-picker scans the test dir (not the projects
    root) and doesn't auto-replace this fixture's mounted inspector
    with one for an unrelated file."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "demo.molwatch.log"
    p.write_text(_MOLWATCH_ONE_STEP)
    return str(p)


@pytest.fixture
def watch_log_file_finished(tmp_path, monkeypatch):
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "demo-done.molwatch.log"
    p.write_text(_MOLWATCH_ONE_STEP_FINISHED)
    return str(p)


@pytest.fixture
def watch_log_file_with_forces(tmp_path, monkeypatch):
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "h2-forces.molwatch.log"
    p.write_text(_MOLWATCH_WITH_FORCES)
    return str(p)


def test_watch_show_forces_renders_arrows(
        page, flask_server, watch_log_file_with_forces):
    """Toggling 'Show force vectors' on a trajectory that carries
    per-atom forces must add arrow shapes to the 3Dmol viewer.
    User-reported regression: the toggle did nothing -- forces
    were parsed but no arrows appeared.

    Post-#234 follow-up: force vectors flow through the embed's
    arrowsPerFrame contract per § 3.9.  The user-visible
    diagnostic readout (#forces-status) carries the count;
    we assert on that string instead of the legacy
    drawForces() console.info "drew N" log.
    """
    _load_watch_log(page, flask_server, watch_log_file_with_forces)
    # The Show-force-vectors checkbox lives inside the "Overlays"
    # sub-tab panel of the Watch viewer controls (default-active
    # tab is "Style", so we click into Overlays first).
    page.locator('.ctab[data-tab="overlays"]').click()
    chk = page.locator("#show-forces")
    assert not chk.is_checked()
    chk.check()
    # Give the click a moment to fire applyForces synchronously
    # (rebuild arrowsPerFrame + handle.setAnimation partial update
    # + refreshForcesStatus).
    page.wait_for_timeout(300)
    # The new status readout next to the toggle must reflect the
    # render outcome ("Showing N arrows ..." / "Hidden ..." etc.)
    # so a user who has all-tiny forces (everything below fmin)
    # sees a clear "0 forces above the threshold" instead of an
    # invisible failure.
    status = page.locator("#forces-status").text_content() or ""
    assert "2" in status and "arrow" in status.lower(), (
        f"#forces-status should report N=2 arrows drawn; got {status!r}"
    )
    # Toggle OFF and confirm the renderer ran again + the status
    # flips to a "hidden" message.
    chk.uncheck()
    page.wait_for_timeout(150)
    assert not chk.is_checked()
    status_off = page.locator("#forces-status").text_content() or ""
    assert "off" in status_off.lower() or "hidden" in status_off.lower(), (
        f"#forces-status should report toggle off; got {status_off!r}"
    )


def _load_watch_log(page, base_url, log_path):
    """Mount the trajectory inspector with ``log_path`` on /results.

    Drives the file via ``projects.setShared(dir, file)`` -- the
    canonical user-flow: this is exactly what the projects sidebar
    publishes on a file click, and ``results/viewer.js`` subscribes
    to ``onChange`` to dispose+mount the matching inspector.

    History: pre-2026-06-01 this helper called ``reg.mount(host,
    file, ctx)`` directly, bypassing the sidebar publish.  That
    worked because nothing else was mounting inspectors.  After
    the /results tab-level file picker landed (commit 6633c4e),
    the picker subscribes to the same ``onChange`` and auto-mounts
    the most recent ``isResult: true`` file in the current
    directory; if the sidebar had auto-resolved to a default
    projects root with real result files, the picker would dispose
    this helper's reg.mount + remount for the projects-root file
    instead -- racing the test.  Routing through ``setShared``
    instead means the test's file IS what the sidebar publishes,
    so the picker and viewer agree.  See
    docs/protocols/playwright-tests.md § 9.4.

    Replaces the legacy "go to /watch, type into #path-input, click
    Load" flow (2026-05-19 /watch removal).
    """
    page.goto(f"{base_url}/results", wait_until="domcontentloaded")
    # Registry + inspector modules self-register at script load;
    # wait until the dispatch chain is ready.
    page.wait_for_function(
        "() => window.molbuilder "
        "&& window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4",
        timeout=5000,
    )
    # Publish the file via the canonical sidebar API.  viewer.js's
    # onChange subscriber then disposes any previously-mounted
    # inspector and mounts the trajectory inspector for ``log_path``.
    import os as _os
    log_dir = _os.path.dirname(log_path)
    page.evaluate(
        """(args) => window.molbuilder.projects.setShared(
            args.dir, args.file)""",
        {"dir": log_dir, "file": log_path},
    )
    # The trajectory adapter fetches /partials/trajectory-inspector
    # async, injects it, then calls the core's mount(host,{file}).
    # Phase 5e A1 (#246) retired the partial's #frame-tot/#frame-idx
    # in favor of the embed's auto-mounted frame strip; we wait
    # instead on the atom-list body populating, which is the
    # deterministic "data loaded" signal that still lives in the
    # partial.  state="attached" — the rows live inside an
    # initially-hidden table (the Inspect tab isn't active by
    # default) so we don't wait for visibility, only for
    # presence in the DOM.
    page.wait_for_selector(
        "#inspect-atom-list-body tr",
        state="attached", timeout=8000)
    # Phase 5f B-4: the atom-list populates from rebuildModel
    # AFTER setAnimation runs (trajectory/core.js:760) — but
    # rebuildModel calls setAnimation synchronously, so the atom
    # list landing IS evidence that setAnimation didn't throw.
    # Belt-and-braces: also assert the embed's frame strip is
    # showing a non-zero frame count, which catches a regression
    # where setAnimation silently rejects with invalid_input
    # (e.g. atom-count mismatch) while atom-list still renders.
    page.wait_for_function(
        "() => {"
        "  const el = document.querySelector("
        "    '.mol-viewer-frame-strip .frame-counter');"
        "  return el && /\\d+/.test(el.textContent || '');"
        "}",
        timeout=8000,
    )


def test_watch_inspect_atom_list_populates(page, flask_server, watch_log_file):
    """Loading a trajectory populates the Inspect-tab atom list with
    one row per atom; the displayed ``#`` column is 1-based."""
    _load_watch_log(page, flask_server, watch_log_file)
    page.locator(".ctab[data-tab='inspect']").click()
    page.wait_for_function(
        "() => document.querySelectorAll('#inspect-atom-list-body tr').length === 3"
    )
    rows = page.locator("#inspect-atom-list-body tr")
    # First row: 1-based index "1", element "O".
    cells = rows.nth(0).locator("td")
    assert cells.nth(0).inner_text() == "1"
    assert cells.nth(1).inner_text() == "O"


def test_watch_inspect_row_click_picks_atom(page, flask_server, watch_log_file):
    """Clicking a row in the Inspect atom list adds the atom to the
    pick list (visible via the row's .is-selected class) and the
    distance row shows '—' until a second atom is picked."""
    _load_watch_log(page, flask_server, watch_log_file)
    page.locator(".ctab[data-tab='inspect']").click()
    page.locator("#inspect-atom-list-body tr").nth(0).click()
    page.wait_for_selector("#inspect-atom-list-body tr.is-selected")
    # The hint goes away; the table becomes visible.
    page.wait_for_function(
        "() => !document.getElementById('inspect-table').hidden"
    )
    a_cell = page.locator("#inspect-a").inner_text()
    assert a_cell.startswith("#1 O"), a_cell
    # No distance yet (only one pick).
    assert page.locator("#inspect-d").inner_text() == "—"


def test_watch_inspect_distance_renders_with_two_picks(
        page, flask_server, watch_log_file):
    """Picking two atoms computes |A-B| and renders it in the
    distance row.  Water's O and first H sit (0,0,0) and (0.957,0,0)
    so the distance is 0.957 A."""
    _load_watch_log(page, flask_server, watch_log_file)
    page.locator(".ctab[data-tab='inspect']").click()
    page.locator("#inspect-atom-list-body tr").nth(0).click()  # O
    page.locator("#inspect-atom-list-body tr").nth(1).click()  # H1
    page.wait_for_function(
        "() => document.getElementById('inspect-d').textContent !== '—'"
    )
    d_text = page.locator("#inspect-d").inner_text()
    # Format: "0.9570 Å"
    assert d_text.startswith("0.957"), d_text
    assert "Å" in d_text


def test_watch_inspect_clear_button_drops_picks(
        page, flask_server, watch_log_file):
    """Clear-selection drops both picks; the hint reappears, the
    table hides, and the Clear button disables again."""
    _load_watch_log(page, flask_server, watch_log_file)
    page.locator(".ctab[data-tab='inspect']").click()
    page.locator("#inspect-atom-list-body tr").nth(0).click()
    page.locator("#inspect-atom-list-body tr").nth(1).click()
    btn = page.locator("#inspect-clear")
    assert btn.is_enabled()
    btn.click()
    page.wait_for_function(
        "() => !document.getElementById('inspect-hint').hidden"
        " && document.getElementById('inspect-table').hidden"
    )
    assert btn.is_disabled()


# --------------------------------------------------------------------- #
#  Run-state badge: must show "last result at <time>" so users know     #
#  when the simulation last produced output (distinct from the Watch    #
#  tab's poll time).  Source: wall_times[] from the parser when         #
#  available (PySCF molwatch emits per-step), else file mtime fallback  #
#  (SIESTA raw .out has no per-step wall clock).                        #
# --------------------------------------------------------------------- #


import re as _re_badge
_HHMMSS_RE = _re_badge.compile(r"\d{1,2}:\d{2}:\d{2}")


def test_watch_run_state_badge_ongoing_shows_last_result_timestamp(
        page, flask_server, watch_log_file):
    """Loading an ongoing molwatch log (no `# concluded:` marker)
    must show the Running badge with a "last result <HH:MM:SS>"
    string in the detail.  The timestamp is the per-step wall_time
    when present, else the file's mtime.

    Note: badge label text is "Running" (not "Ongoing") -- the
    server-side ``run_state`` value is still ``"ongoing"`` but the
    UI renders it as "Running" since 2026-05-22 (see
    ``lib/trajectory/core.js`` ``applyNewData``).
    """
    _load_watch_log(page, flask_server, watch_log_file)
    page.wait_for_selector("#run-state-label", state="visible")
    # CSS text-transform may uppercase the visible label; check the
    # DOM text directly (text_content() bypasses rendering).
    label  = page.locator("#run-state-label").text_content()
    detail = page.locator("#run-state-detail").text_content()
    assert (label or "").lower() == "running", (
        f"expected Running, got {label!r}"
    )
    assert "last result" in (detail or "").lower(), (
        f"ongoing badge detail should mention 'last result': {detail!r}"
    )
    assert _HHMMSS_RE.search(detail or ""), (
        f"ongoing badge detail should contain an HH:MM:SS timestamp: {detail!r}"
    )


def test_watch_run_state_badge_finished_shows_ended_timestamp(
        page, flask_server, watch_log_file_finished):
    """A molwatch log with `# concluded:` marker -> Finished badge
    with "ended <HH:MM:SS>" in the detail."""
    _load_watch_log(page, flask_server, watch_log_file_finished)
    page.wait_for_selector("#run-state-label", state="visible")
    page.wait_for_function(
        "() => document.getElementById('run-state-label')"
        ".textContent === 'Finished'",
        timeout=8000,
    )
    detail = page.locator("#run-state-detail").text_content() or ""
    assert "ended" in detail.lower(), (
        f"finished badge detail should mention 'ended': {detail!r}"
    )
    assert _HHMMSS_RE.search(detail), (
        f"finished badge detail should contain an HH:MM:SS timestamp: {detail!r}"
    )


# --------------------------------------------------------------------- #
#  Schema-driven Build form (form-schema.js renderer)                  #
#                                                                       #
#  The static SIESTA + PySCF form fields no longer ship in the index    #
#  page; they're injected by form-schema.js on page load from GET       #
#  /api/build/schema/<engine>.  These tests exercise the renderer +    #
#  collector end-to-end so the cutover stays honest in the future.      #
# --------------------------------------------------------------------- #


def _open_build(page, base_url):
    """Open the Build page and wait until both schema-driven forms
    have rendered (their containers gain children)."""
    page.goto(f"{base_url}/structure-optimization", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelector('#siesta-form-container fieldset')"
        " && document.querySelector('#pyscf-form-container fieldset')",
        timeout=8000,
    )


def test_build_page_loads_without_js_errors(page, flask_server):
    """The Build / index page must boot without any uncaught JS
    pageerror -- a single ReferenceError near the bottom of the
    IIFE halts the rest of the module's wiring (compat engine,
    preflight, stage-preset) and the symptoms are subtle.
    Previously regressed on the schema-driven cutover when an
    obsolete FORM_IDS reference survived a rename.
    """
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg:
            errors.append("console.error: " + msg.text)
            if msg.type == "error" else None)
    page.goto(f"{flask_server}/structure-optimization", wait_until="networkidle")
    # Wait for the schema-driven form to render too, so any error
    # in the async path also surfaces.
    page.wait_for_function(
        "() => document.querySelector('#siesta-form-container fieldset')"
        " && document.querySelector('#pyscf-form-container fieldset')",
        timeout=8000,
    )
    assert errors == [], (
        f"unexpected JS errors on the Build page: {errors}"
    )


def test_build_form_live_preflight_fires_on_field_edit(
        page, flask_server, water_xyz_file):
    """After loading a structure, editing a SIESTA field with a
    range-violating value must fire /api/build/preflight via the
    debounced listener and surface a warn-severity issue in the
    #fdf-issues panel WITHOUT re-clicking Generate.

    Regression test for the FORM_IDS / wirePreflightListeners
    cutover: before the fix, the JS halt on FORM_IDS prevented
    these listeners from being attached.

    Post-task-295 (2026-06-08): the Build form is gone; the sole
    structure entry on the Optimization tab is the sidebar handoff
    (sessionStorage[molbuilder.current_file] auto-load on mount).
    Prime the handoff before navigating; the auto-load triggers
    the same _commitStructure path that the retired Build button
    used to drive."""
    # Prime the cross-tab handoff so the Optimization tab's
    # mount-time auto-load picks up the water fixture.  The
    # auto-load is what triggers preflight-listener wiring.
    # Use add_init_script so the pointer is in sessionStorage
    # BEFORE any page script runs.
    import os as _os
    page.add_init_script(
        "sessionStorage.setItem('molbuilder.current_file',"
        f" {repr(water_xyz_file)});"
        "sessionStorage.setItem('molbuilder.current_dir',"
        f" {repr(_os.path.dirname(water_xyz_file))});"
    )
    _open_build(page, flask_server)
    page.wait_for_function(
        "() => !document.getElementById('generate-fdf').disabled",
        timeout=8000,
    )
    # Drop MeshCutoff below its declared range=(100, 1000) -- the
    # config-metadata validator emits a warn for out-of-range.
    page.fill("#p-mesh-cutoff", "30")
    # Debounce is 250 ms; allow a generous 2 s for the preflight
    # round-trip + render.
    #
    # The wait condition checks for the mesh_cutoff warning
    # SPECIFICALLY -- not just "any issue".  A previous preflight
    # call (e.g. before the page even loaded a structure) may have
    # left a psml_lib warning visible from when mesh_cutoff was
    # at its default; the test must wait for the NEW warning
    # produced by the mesh_cutoff edit to actually land.  Without
    # this guard, the assertion racing against the debounced
    # request could fire on the stale state and see only the
    # psml_lib warning.  See docs/protocols/playwright-tests.md
    # § A7 "Reading text from a node that's about to be replaced".
    page.wait_for_function(
        "() => {"
        "  const panel = document.getElementById('fdf-issues');"
        "  if (!panel || panel.hidden) return false;"
        "  const items = Array.from(panel.querySelectorAll('li'));"
        "  return items.some(li => "
        "    /mesh_cutoff/i.test(li.textContent));"
        "}",
        timeout=2000,
    )
    issue_texts = page.evaluate(
        "() => Array.from("
        "  document.querySelectorAll('#fdf-issues li')"
        ").map(li => li.textContent)"
    )
    # The mesh-cutoff range warning must appear among the issues.
    mesh_warn = [t for t in issue_texts if "mesh_cutoff" in t.lower()]
    assert mesh_warn, (
        f"expected a mesh_cutoff range warning after dropping below "
        f"50; got: {issue_texts}"
    )


def test_build_form_renders_siesta_sections_in_pinned_order(
        page, flask_server):
    """The rendered SIESTA <fieldset> sections must match the schema's
    section order in declaration sequence -- pins the visual layout
    to the dataclass + _form_section_order."""
    _open_build(page, flask_server)
    legends = page.evaluate(
        "() => Array.from("
        "  document.querySelectorAll('#siesta-form-container fieldset > legend')"
        ").map(l => l.textContent.trim())"
    )
    # Order matches ``SiestaConfig._form_section_order`` -- physics
    # sections first, "Parallel execution" last so the user designs
    # the calculation before sizing the machine (see the order's
    # rationale in ``molbuilder/config/siesta.py``).
    assert legends == [
        "System",
        "Basis & grid",
        "Exchange-correlation",
        "SCF",
        "Spin",
        "k-grid (Monkhorst-Pack)",
        "Relaxation",
        "Output & positioning",
        "Parallel execution",
    ], legends


def test_build_form_renders_pyscf_sections_in_pinned_order(
        page, flask_server):
    """Same pin for the PySCF panel."""
    _open_build(page, flask_server)
    legends = page.evaluate(
        "() => Array.from("
        "  document.querySelectorAll('#pyscf-form-container fieldset > legend')"
        ").map(l => l.textContent.trim())"
    )
    assert legends == [
        "System",
        "Method",
        "SCF",
        "Pre-optimization (optional)",
        "Optimization",
        "Solvent (optional)",
        "Frequencies / thermochemistry",
        "Runtime & output",
    ], legends


def test_build_form_kgrid_renders_sub_labels(page, flask_server):
    """int-triple (kgrid: Tuple[int,int,int]) must render three
    labelled cells, not three anonymous number boxes.  Each cell
    shows its sub-label (x / y / z) so the user can tell which
    input drives which dimension."""
    _open_build(page, flask_server)
    labels = page.evaluate(
        "() => Array.from("
        "  document.querySelectorAll('.schema-int-triple-label')"
        ").map(s => s.textContent.trim())"
    )
    assert labels == ["x", "y", "z"], labels
    # The three sub-inputs use the schema-derived sub-ids.
    for sub in ("x", "y", "z"):
        assert page.locator(f"#p-k-{sub}").count() == 1, sub


def test_build_form_legacy_short_ids_preserved(page, flask_server):
    """Fields with metadata["id_suffix"] override keep their legacy
    short id so the compatibility engine and sessionStorage list
    remain compatible across the cutover."""
    _open_build(page, flask_server)
    # electronic_temperature -> "p-temperature" (not "p-electronic-temperature").
    assert page.locator("#p-temperature").count() == 1
    # parallel_block_size -> "p-block-size".
    assert page.locator("#p-block-size").count() == 1
    # relax_force_tol -> "p-force-tol".
    assert page.locator("#p-force-tol").count() == 1
    # relax_max_displ -> "p-max-displ".
    assert page.locator("#p-max-displ").count() == 1
    # max_memory_mb -> "py-max-memory".
    assert page.locator("#py-max-memory").count() == 1
    # scf_init_guess -> "py-init-guess".
    assert page.locator("#py-init-guess").count() == 1


def test_build_form_collect_round_trip(page, flask_server):
    """collectForm walks the rendered DOM and reads values back as
    a dict whose keys are dataclass field names.  Round-trip a few
    edits and assert the returned shape matches what the FDF
    endpoint would receive."""
    _open_build(page, flask_server)
    # Edit a few representative fields directly.
    page.fill("#p-system-label", "ci-test")
    page.fill("#p-mesh-cutoff", "350")
    _set_checkbox(page, "#p-spin-polarized", True)
    page.fill("#p-k-x", "4")
    page.fill("#p-k-y", "4")
    page.fill("#p-k-z", "1")

    collected = page.evaluate(
        "async () => {"
        "  const fs = window.molbuilder.formSchema;"
        "  const sch = await fs.fetchSchema('siesta');"
        "  return fs.collectForm("
        "    document.getElementById('siesta-form-container'), sch"
        "  );"
        "}"
    )
    assert collected["system_label"] == "ci-test"
    assert collected["mesh_cutoff"] == 350
    assert collected["spin_polarized"] is True
    assert collected["kgrid"] == [4, 4, 1]
    # spin_total is Optional[float]; default left blank should
    # collect as null (server treats absent / null as auto).
    assert collected["spin_total"] is None
    # The dispersion field is PySCF only -- not in the SIESTA
    # schema -- so it must NOT appear in the collected SIESTA dict.
    assert "dispersion" not in collected


def test_build_form_collect_pyscf_dispersion_select(page, flask_server):
    """PySCF dispersion is an Optional[str] with explicit choices
    ("d3", "d3bj", "d4", "none").  Default is "d3bj"; the user
    selecting "none" must come through verbatim so the JS-side
    post-processing can map it to null."""
    _open_build(page, flask_server)
    # Switch to the PySCF panel -- the dispersion select lives there
    # and the default-active SIESTA panel hides it.
    page.locator('[data-tab="pyscf"]').click()
    page.locator("#py-dispersion").select_option("none")
    collected = page.evaluate(
        "async () => {"
        "  const fs = window.molbuilder.formSchema;"
        "  const sch = await fs.fetchSchema('pyscf');"
        "  return fs.collectForm("
        "    document.getElementById('pyscf-form-container'), sch"
        "  );"
        "}"
    )
    assert collected["dispersion"] == "none"


def test_build_form_tri_select_optional_bool(page, flask_server):
    """Optional[bool] (parallel_over_k) renders as a 3-option select
    auto/true/false; "auto" must collect as null."""
    _open_build(page, flask_server)
    val = page.evaluate(
        "() => document.getElementById('p-parallel-over-k').value"
    )
    assert val == "auto"
    collect_js = (
        "async () => {"
        "  const fs = window.molbuilder.formSchema;"
        "  const sch = await fs.fetchSchema('siesta');"
        "  return fs.collectForm("
        "    document.getElementById('siesta-form-container'), sch"
        "  );"
        "}"
    )
    collected = page.evaluate(collect_js)
    assert collected["parallel_over_k"] is None
    # Flip to "true" -> must collect as the JS boolean true.
    page.locator("#p-parallel-over-k").select_option("true")
    collected = page.evaluate(collect_js)
    assert collected["parallel_over_k"] is True


# --------------------------------------------------------------------- #
#  Second-visit + external-change pattern (#195, audit follow-up to    #
#  the 2026-06-02 /results stale-dropdown bug).  Per                   #
#  docs/protocols/playwright-tests.md § 9.6, every tab whose UI       #
#  is driven by a subscriber-on-state-change needs at least one       #
#  test exercising the "user navigated away, external state          #
#  changed, returned" workflow.                                       #
# --------------------------------------------------------------------- #


class TestModifySecondVisitExternalChange:
    """Audit follow-up: pin the second-visit refresh contract for
    /modify so a future regression that breaks the sidebar-pick →
    selection-panel wiring on bfcache restore / tab re-entry fails
    loudly."""

    def test_revisit_modify_with_persisted_selection_reloads_atom_list(
            self, page, flask_server, water_xyz_file):
        """User opens /modify, picks water.xyz (3 atoms land in the
        selection store), navigates to /build, comes back.  The
        atom list must still be populated -- not blank because the
        store's onChange subscriber bailed on the same source
        path."""
        _open_modify(page, flask_server)
        _load_water(page, water_xyz_file)
        # Atom list reflects 3 atoms.
        assert page.evaluate(
            "() => window.molbuilder.selection.store.getState()"
            ".atoms.length"
        ) == 3

        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#load-from-sidebar-btn", timeout=5000)

        page.goto(f"{flask_server}/molbuilder")
        # The selection store MUST repopulate.  Without the refresh
        # contract, sessionStorage has the file path but the store's
        # internal "lastSourceFile" still matches -> no re-fetch.
        page.wait_for_function(
            "() => window.molbuilder.selection.store.getState()"
            ".atoms.length === 3",
            timeout=5000,
        )

    def test_dirty_canvas_preserves_in_memory_atoms_on_revisit(
            self, page, flask_server, water_xyz_file, tmp_path,
            monkeypatch):
        """Counterpart to the external-replacement test: when the
        user did a modifier op (canvas dirty), the in-memory atoms
        are MORE authoritative than disk (disk hasn't been saved
        yet).  Restore MUST use saved.atoms, not disk-fetch.

        Pin both sides of the dirty-gate:
          * dirty=false (this is the external-replacement test):
            restore re-fetches from disk → external changes
            picked up.
          * dirty=true (this test): restore uses saved.atoms →
            user's in-memory edits survive navigation."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        xyz_path = tmp_path / "structure.xyz"
        xyz_path.write_text(
            "3\nwater\n"
            "O 0.000 0.000 0.000\n"
            "H 0.957 0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        _open_modify(page, flask_server)
        _load_file(page, str(xyz_path), expected_atoms=3)

        # Select the O and delete it — leaves 2 atoms in memory;
        # disk still has the 3-atom water file.
        _set_selection(page, [0])
        page.locator("#delete-apply").click()
        page.wait_for_function(
            "() => window.__molbuilder_modify_test.getNAtoms() === 2"
        )
        # Canvas is now dirty.
        assert page.evaluate(
            "() => window.molbuilder.structureCanvas.isDirty()"
        ) is True

        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#load-from-sidebar-btn", timeout=5000)

        page.goto(f"{flask_server}/molbuilder")
        # Selection store MUST reflect 2 atoms (in-memory post-
        # delete), NOT 3 atoms (disk).  Disk would override
        # in-memory if the dirty-gate is broken.
        page.wait_for_function(
            "() => window.molbuilder.selection.store.getState()"
            ".atoms.length === 2",
            timeout=5000,
        )

    def test_external_xyz_replacement_reloads_atom_list_on_revisit(
            self, page, flask_server, tmp_path, monkeypatch):
        """Stronger version: replace the source file with a
        different-atom-count structure between visits.  The
        selection panel MUST reflect the new atom count."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        xyz_path = tmp_path / "structure.xyz"
        xyz_path.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        _open_modify(page, flask_server)
        _load_file(page, str(xyz_path), expected_atoms=3)

        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_selector("#load-from-sidebar-btn", timeout=5000)

        # Replace the file with a different structure (5 atoms,
        # methane).
        import time
        xyz_path.write_text(
            "5\nmethane\n"
            "C 0.000  0.000 0.000\n"
            "H 0.629  0.629 0.629\n"
            "H -0.629 -0.629 0.629\n"
            "H -0.629  0.629 -0.629\n"
            "H 0.629 -0.629 -0.629\n"
        )
        time.sleep(0.5)

        page.goto(f"{flask_server}/molbuilder")
        # The atom list MUST reflect the new structure on re-entry.
        # If the JS subscriber bails on "same source file path",
        # the user sees stale 3 atoms.  This is the same bug shape
        # as #192.
        page.wait_for_function(
            "() => window.molbuilder.selection.store.getState()"
            ".atoms.length === 5",
            timeout=5000,
        )
