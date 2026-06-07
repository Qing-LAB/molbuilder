"""End-to-end Playwright tests for the /build (root) tab.

Audit task #190 (2026-06-02).  The /build tab is the primary entry
point for "build a molecule from sequence/SMILES + emit a SIESTA or
PySCF script", yet until this commit it had NO browser-level
coverage.  ``test_pages_no_js_errors.py`` exercised the page-boot
smoke path; ``test_web.py`` exercised every HTTP endpoint via
``test_client``; but the actual UI loop -- schema fetch -> form
render -> structure build -> Generate-button toggle -- only ran in
production.

This file closes that gap.  Scope:

  * The two parameter forms (SIESTA + PySCF) actually populate from
    their schema endpoints (a silent ``schema.fields`` regression
    would leave them stuck on the "Loading from schema..." placeholder
    -- same failure mode the 2026-05 Spectra parse bug demonstrated).
  * ``engine_key`` badges render alongside form labels (the
    2026-05-26 source-of-truth UI contract).
  * Tab switching between SIESTA + PySCF panels hides + shows the
    right one (covered as a wiring smoke).
  * A peptide round-trip: type "ARNDC" -> click Build -> the viewer
    mounts the structure + Generate-{fdf,pyscf} buttons enable.
    Pre-#190 a regression in viewer.js's post-build hook (e.g. a
    stale querySelector after a DOM rename) was invisible to the
    test_web.py tests because they never load any JS.
  * Sidebar-driven load: ``projects.setShared(dir, file)`` for a
    real XYZ on disk renders into the viewer + populates the info
    line.  Matches the workflow Build expects from the unified
    projects sidebar (the same flow modify_e2e and spectra use).

NOT covered here (intentional scope split):
  * Generate.fdf / Generate.py output rendering -- the HTTP layer
    already pins that in ``test_web.py``; we exercise the button
    enable/disable only.
  * Save / install-pseudos / install-wrapper -- those are out of
    Build's primary user loop (they're post-Generate actions); they
    have their own coverage in ``test_pseudos.py`` +
    ``test_runwrap.py``.
"""
from __future__ import annotations

import threading

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
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


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Pin ``tmp_path`` as the only Capabilities picker root so the
    files blueprint accepts files written under it.  Mirrors the
    helper in ``test_modify_e2e.py`` / ``test_source_inspector_e2e.py``.
    """
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
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
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


@pytest.fixture
def water_xyz_file(tmp_path, monkeypatch):
    """Minimal real XYZ on disk for the sidebar-driven load test."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "water.xyz"
    p.write_text(
        "3\nwater\n"
        "O 0.000  0.000 0.000\n"
        "H 0.957  0.000 0.000\n"
        "H -0.239 0.927 0.000\n"
    )
    return str(p)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


_BOOT_TIMEOUT_MS = 5000


def _open_build(page, base_url):
    """Navigate to / and wait for the viewer.js wiring to land.

    Capture JS errors so a regression that throws during init() (the
    same failure mode as the 2026-05 Spectra parse bug) surfaces
    here with a clear page-name attribution.
    """
    errors = []
    page.on("pageerror",
            lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    page.goto(f"{base_url}/structure-optimization")
    # #build-btn is rendered by the server-side template; its presence
    # proves the HTML reached the browser.  The form-container fields
    # are added by JS after the schema fetch lands.
    page.wait_for_selector("#build-btn", timeout=_BOOT_TIMEOUT_MS)
    return errors


# --------------------------------------------------------------------- #
#  Form schemas populate from the server                                #
# --------------------------------------------------------------------- #


class TestFormSchemasRender:
    """Both engine forms fetch their schema and render fields.  A
    regression in dataclass_to_form_schema or in form-schema.js'
    renderer would leave the containers empty -- silent UI break."""

    def test_siesta_form_renders_fields_after_init(
            self, page, flask_server):
        errors = _open_build(page, flask_server)
        # The form renderer mounts inputs/selects into
        # #siesta-form-container after the schema fetch resolves.
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # The Loading-from-schema placeholder must be gone.
        container_text = page.locator(
            "#siesta-form-container"
        ).inner_text()
        assert "Loading from schema" not in container_text
        assert errors == [], f"JS errors during /build init: {errors}"

    def test_pyscf_form_renders_fields_after_init(
            self, page, flask_server):
        _open_build(page, flask_server)
        # The PySCF tab panel starts ``hidden`` (SIESTA is the
        # default-active tab), so its rendered inputs are NOT visible
        # by default.  Use ``state="attached"`` so Playwright waits
        # for the elements to exist in the DOM rather than waiting
        # for them to become visible -- form-schema.js builds both
        # panels at init() time regardless of tab visibility, and
        # that's the property we want to pin.
        page.wait_for_selector(
            "#pyscf-form-container input, "
            "#pyscf-form-container select",
            state="attached",
            timeout=_BOOT_TIMEOUT_MS,
        )
        container_text = page.locator(
            "#pyscf-form-container"
        ).inner_text()
        assert "Loading from schema" not in container_text

    def test_engine_key_badges_present_after_schema_render(
            self, page, flask_server):
        """Per the 2026-05-26 source-of-truth contract every form
        field carries an ``engine_key`` metadata entry which the
        renderer surfaces as a ``<code class="schema-engine-key">``
        badge.  Absence here means the schema -> form binding lost
        the metadata, OR a future field was added without an
        engine_key (which would also fail the Python-side coverage
        test in ``test_web.py::test_engine_key_present_on_every_*``,
        so this test triangulates the contract from the JS side).
        """
        _open_build(page, flask_server)
        page.wait_for_selector(
            "#siesta-form-container .schema-engine-key",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # At least one badge in each container.  Existence proves
        # the renderer reached the badge branch; the text contents
        # are validated by the Python-side coverage test.
        siesta_badges = page.locator(
            "#siesta-form-container .schema-engine-key"
        ).count()
        pyscf_badges = page.locator(
            "#pyscf-form-container .schema-engine-key"
        ).count()
        assert siesta_badges > 0, \
            "SIESTA form rendered without engine_key badges"
        assert pyscf_badges > 0, \
            "PySCF form rendered without engine_key badges"


# --------------------------------------------------------------------- #
#  Tab switching                                                        #
# --------------------------------------------------------------------- #


class TestTabSwitching:
    """Clicking the SIESTA / PySCF tabs reveals the matching panel."""

    def test_clicking_pyscf_tab_shows_pyscf_panel(
            self, page, flask_server):
        _open_build(page, flask_server)
        # Wait for the SIESTA panel (default active, visible) so we
        # know the schema render landed before driving the tab click.
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # The PySCF tab activator: matched by role + name.  Tab
        # button text is "PySCF script"; ``has_text`` does substring
        # matching so this still works if the wording shifts slightly.
        page.locator("[role='tab']").filter(has_text="PySCF").click()
        # After the click the PySCF panel must be visible (not
        # ``hidden`` attribute).
        page.wait_for_selector(
            "#tab-pyscf:not([hidden])", timeout=_BOOT_TIMEOUT_MS,
        )

    def test_clicking_siesta_tab_returns_to_siesta_panel(
            self, page, flask_server):
        _open_build(page, flask_server)
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        page.locator("[role='tab']").filter(has_text="PySCF").click()
        page.locator("[role='tab']").filter(has_text="SIESTA").click()
        page.wait_for_selector(
            "#tab-siesta:not([hidden])", timeout=_BOOT_TIMEOUT_MS,
        )


# --------------------------------------------------------------------- #
#  Build a peptide and check the post-build viewer / button state       #
# --------------------------------------------------------------------- #


class TestPeptideBuild:
    """Type a sequence, click Build, verify the structure lands +
    Generate buttons enable.

    Peptide is the cheapest reliable build path (PeptideBuilder is
    a pure-Python dep, no RDKit / OpenBabel / external binaries
    needed).  Pre-#190 a regression in viewer.js's post-build hook
    (e.g. a stale querySelector after a DOM rename, or a fetch URL
    typo) shipped silently because no test loaded the JS.
    """

    def test_build_peptide_updates_info_atoms(
            self, page, flask_server):
        """After Build the #info-atoms span goes from the em-dash
        placeholder to a numeric atom count -- proves the
        /api/build/molecule round-trip + the viewer-mount hook
        ran to completion."""
        _open_build(page, flask_server)
        # The "kind" select defaults to peptide on first render,
        # but pin it explicitly so a future default change doesn't
        # silently route this test through a different build path.
        page.locator("#kind").select_option("peptide")
        page.locator("#input-text").fill("ARNDC")
        page.locator("#build-btn").click()
        # The info bar updates synchronously after the structure
        # mounts in the viewer.  Wait for the placeholder em-dash
        # to disappear from #info-atoms.
        page.wait_for_function(
            "() => /^\\d+$/.test("
            "document.querySelector('#info-atoms').textContent.trim())",
            timeout=_BOOT_TIMEOUT_MS,
        )
        atom_count = int(
            page.locator("#info-atoms").inner_text().strip()
        )
        # ARNDC has 5 residues = ~80-90 atoms with default
        # protonation.  Pin lower bound at the test_web.py
        # threshold (38) so a default-change to heavy-only doesn't
        # silently regress the e2e.
        assert atom_count >= 38

    def test_generate_buttons_enable_after_build(
            self, page, flask_server):
        """Pre-build, Generate.fdf + Generate.py are disabled
        (no structure to emit).  Post-build they must enable so
        the user can hand off to /api/build/{fdf,pyscf}."""
        _open_build(page, flask_server)
        # Pre-build sanity: both buttons disabled.
        assert page.locator("#generate-fdf").is_disabled()
        assert page.locator("#generate-pyscf").is_disabled()

        page.locator("#kind").select_option("peptide")
        page.locator("#input-text").fill("ARNDC")
        page.locator("#build-btn").click()
        # Wait for the structure to mount.
        page.wait_for_function(
            "() => /^\\d+$/.test("
            "document.querySelector('#info-atoms').textContent.trim())",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # Generate buttons should be live now.
        page.wait_for_function(
            "() => !document.querySelector('#generate-fdf').disabled",
            timeout=_BOOT_TIMEOUT_MS,
        )
        assert not page.locator("#generate-fdf").is_disabled()
        assert not page.locator("#generate-pyscf").is_disabled()


# --------------------------------------------------------------------- #
#  Sidebar-driven load: the canonical "user clicks a project file"     #
#  workflow renders into the Build viewer.                              #
# --------------------------------------------------------------------- #


class TestSidebarPickLoad:
    """Universal sidebar interaction model (B.5.3): sidebar
    single-click (``setShared``) = preview only; a dblclick
    commit (``publishCommit``) is what actually rebuilds Build's
    structure section.  The form-dirty gate then warns if the
    user has typed parameter edits since the last commit."""

    def test_setShared_alone_does_NOT_load_structure(
            self, page, flask_server, water_xyz_file):
        """Pins the candidate-only contract for Build: a single
        sidebar click sets the global pick but MUST NOT auto-
        rebuild the structure section.  Browse-clicking through
        the project tree no longer silently switches the
        molecule the form is configured for."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.setShared "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            """(ctx) => window.molbuilder.projects.setShared(
                ctx.dir, ctx.file)""",
            {"dir": parent, "file": p},
        )
        # Brief settle so an erroneous auto-load would land in
        # #info-atoms before the assertion.
        page.wait_for_timeout(500)
        n = page.evaluate(
            "() => document.querySelector('#info-atoms').textContent.trim()"
        )
        assert n in ("—", "", "0"), (
            f"setShared must NOT auto-load; #info-atoms is {n!r}"
        )

    def test_publishCommit_loads_water_xyz_into_viewer(
            self, page, flask_server, water_xyz_file):
        """Commit (dblclick equivalent) rebuilds the structure
        section — the canonical "use this file in this tab"
        action."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.publishCommit "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            """(ctx) => window.molbuilder.projects.publishCommit(
                ctx.dir, ctx.file)""",
            {"dir": parent, "file": p},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

    def test_form_edits_followed_by_commit_fires_warning(
            self, page, flask_server, tmp_path, monkeypatch):
        """The form-dirty gate: a user who typed a SIESTA / PySCF
        parameter edit since the last commit must see a warning
        before the next commit silently rebuilds the structure
        section.  Cancel = preserve edits; Discard = proceed."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        # Two distinct structure files so the second commit isn't
        # short-circuited by the same-file guard.
        water = tmp_path / "water.xyz"
        water.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        methane = tmp_path / "methane.xyz"
        methane.write_text(
            "5\nmethane\n"
            "C 0.000  0.000 0.000\n"
            "H 0.629  0.629 0.629\n"
            "H -0.629 -0.629 0.629\n"
            "H -0.629  0.629 -0.629\n"
            "H 0.629 -0.629 -0.629\n"
        )
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder"
            "       && typeof window.molbuilder.warningModal"
            "          === 'object'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # Initial commit on water.
        page.evaluate(
            """(c) => window.molbuilder.projects.publishCommit(
                c.dir, c.file)""",
            {"dir": str(tmp_path), "file": str(water)},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # Make the form dirty by typing into a SIESTA param input.
        page.wait_for_selector(
            "#siesta-form-container input", timeout=_BOOT_TIMEOUT_MS)
        siesta_input = page.locator(
            "#siesta-form-container input").first
        siesta_input.focus()
        siesta_input.press("a")
        # Commit a DIFFERENT structure (methane).  Form-dirty gate
        # fires the warning before discarding the parameter edits.
        page.evaluate(
            """(c) => window.molbuilder.projects.publishCommit(
                c.dir, c.file)""",
            {"dir": str(tmp_path), "file": str(methane)},
        )
        page.wait_for_selector(
            "dialog.molbuilder-warning-modal",
            state="attached", timeout=3000)
        # Cancel → structure stays at 3 atoms (water); the rebuild
        # didn't run.
        page.locator(
            'dialog.molbuilder-warning-modal [data-action="cancel"]'
        ).click()
        page.wait_for_selector(
            "dialog.molbuilder-warning-modal",
            state="detached", timeout=2000)
        n = page.evaluate(
            "() => document.querySelector('#info-atoms').textContent.trim()"
        )
        assert n == "3", (
            f"Cancel must preserve the current structure; "
            f"#info-atoms is {n!r} (expected 3 for water)"
        )


# --------------------------------------------------------------------- #
#  Second-visit + external-change pattern (#195, audit follow-up to    #
#  the 2026-06-02 /results stale-dropdown bug).  Per                   #
#  docs/protocols/playwright-tests.md § 9.6, every tab whose UI       #
#  is driven by a subscriber-on-state-change needs at least one       #
#  test exercising the "user navigated away, external state          #
#  changed, returned" workflow.                                       #
# --------------------------------------------------------------------- #


class TestBuildSecondVisitExternalChange:
    """Audit follow-up: pin the second-visit refresh contract for
    /build so a future regression that breaks the sidebar-pick →
    viewer-update wiring on bfcache restore / tab re-entry fails
    loudly.  The /results file-picker shipped a bug of this exact
    shape (#192) that no single-page-load test could catch."""

    def test_revisiting_build_with_existing_selection_reloads_viewer(
            self, page, flask_server, water_xyz_file):
        """User opens /build, picks water.xyz, navigates to /modify
        (the canonical "go look at the structure" flow), comes back
        to /build.  The viewer MUST still show the structure +
        atom-count line, even though the page just bootstrapped
        fresh.  Pre-fix the bug class: sessionStorage holds the
        file, the sidebar onChange subscriber fires with the same
        value as last time, picker-style "bails on same key"
        suppresses the load -> viewer is empty."""
        _open_build(page, flask_server)
        # Drive the sidebar to a real file (same path as
        # TestSidebarPickLoad).
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": parent, "file": p},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

        # Navigate to /modify -- sessionStorage carries dir + file
        # over (cross-tab handoff).
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#projects-sidebar", timeout=_BOOT_TIMEOUT_MS)

        # Come back to /build.  The viewer MUST reload from the
        # persisted selection without a sidebar click.
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

    def test_external_xyz_replacement_reloads_on_revisit(
            self, page, flask_server, tmp_path, monkeypatch):
        """Stronger version: user picks water.xyz on /build, leaves
        the tab, the file content is REPLACED on disk (different
        atom count), user comes back to /build.  Viewer MUST show
        the new atom count -- not the stale 3-atom water.

        Catches: a hypothetical sidebar-pick loader that caches the
        last-loaded XYZ text and skips the re-fetch on identical
        path."""
        # Pin tmp_path as picker root + write the initial 3-atom XYZ.
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        xyz_path = tmp_path / "structure.xyz"
        xyz_path.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        _open_build(page, flask_server)
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": str(tmp_path), "file": str(xyz_path)},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#projects-sidebar", timeout=_BOOT_TIMEOUT_MS)

        # Replace the file on disk with a different structure
        # (5 atoms, methane-ish).
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

        page.goto(f"{flask_server}/structure-optimization")
        # The new structure must appear; the OLD cached "3" must
        # NOT be what the viewer shows.
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '5'",
            timeout=_BOOT_TIMEOUT_MS,
        )
