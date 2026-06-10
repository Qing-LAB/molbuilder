"""Spectra-blueprint route tests.

Exercises the four endpoints from spec § 10 with the Flask test
client:

  GET  /spectra                       -- page renders, has the tab nav
  GET  /api/build/schema/spectra      -- schema endpoint mirrors siesta/pyscf
  POST /api/spectra/render            -- happy path + error shapes
  POST /api/spectra/load              -- multipart / path / inline-JSON modes,
                                         plus exception-class -> HTTP code

Cheap to run: no real PySCF SCF (the engine's preflight + render
happen, but the script template is just text emission); no live
disk watch.
"""

from __future__ import annotations

import io
import json

import numpy as np

from molbuilder.parsers.spectra_json import dump_spectra_json
from molbuilder.spectra import (
    ModeData,
    SpectraResults,
)
from molbuilder.spectra.results import (
    PHASE_COMPLETE,
    PHASE_EMPTY,
    SCHEMA_VERSION,
)


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


_WATER_XYZ = """3
water
O    0.00000000    0.00000000    0.00000000
H    0.96000000    0.00000000    0.00000000
H   -0.24000000    0.93000000    0.00000000
"""


def _make_minimal_results() -> SpectraResults:
    mode = ModeData(
        index_1based          = 1,
        frequency_cm1         = 412.3,
        raman_activity_a4_amu = 12.5,
        ir_intensity_km_mol   = None,
        eigenvector_canonical = np.array([[0.7, 0., 0.],
                                          [-0.7, 0., 0.]]),
        eigenvector_display   = np.array([[0.7, 0., 0.],
                                          [-0.7, 0., 0.]]),
        has_imag              = False,
    )
    return SpectraResults(
        schema_version             = SCHEMA_VERSION,
        engine                     = "pyscf",
        engine_version             = "2.6.0",
        molbuilder_version         = "1.2.0",
        timestamp                  = "2026-05-11T12:00:00Z",
        structure_hash             = "sha256:abc",
        n_atoms_total              = 2,
        free_atom_idxs             = [0, 1],
        frozen_atom_idxs           = [],
        equilibrium_scf_eh         = -76.4,
        equilibrium_mo_energies_eh = np.array([-1.0, -0.5, -0.2, 0.1, 0.3]),
        equilibrium_homo_idx       = 2,
        modes                      = [mode],
        selected_mode_idxs_1based  = [],
        config                     = {},
        methods_text               = "",
        bibliography_keys          = [],
        phase_frequencies          = PHASE_COMPLETE,
        phase_raman                = PHASE_COMPLETE,
        phase_es                   = PHASE_EMPTY,
    )


# --------------------------------------------------------------------- #
#  Page route                                                           #
# --------------------------------------------------------------------- #


class TestSpectraPage:

    def test_page_loads(self, web_client):
        r = web_client.get("/spectrum-calculation")
        assert r.status_code == 200
        body = r.data.decode()
        # Tab nav present + Spectrum-calculation tab marked active.
        assert "Spectrum calculation" in body
        assert "app-tabs" in body
        # Generate-side ids (the only surface the page carries; the
        # inspect-side surface lives in _spectra_inspector.html and
        # is served only to /results).
        assert 'id="spectra-form-container"' in body
        # The Inspect-structure card (task #296) is the sole
        # entry point for the structure; the hidden
        # ``<textarea id="structure-text">`` that pre-#309 backed
        # the schema's structure_text was retired in favour of
        # spectraInspector.setStructureText() + an in-memory
        # holder.  Pin the new entry-point ids instead.
        assert 'id="viewer"'                    in body
        assert 'id="load-from-sidebar-btn"'     in body
        assert 'id="generate-btn"'              in body
        # Methods-preview modal present (dialog element + handles).
        assert 'id="methods-modal"'           in body
        # Static assets pinned in the template.
        assert 'spectra/style.css'            in body
        assert 'spectra/viewer.js'            in body
        # Shared inspector core loaded BEFORE the per-page bootstrap
        # (the bootstrap calls window.molbuilder.spectraInspector.mount
        # which the core defines).
        assert body.index("lib/spectra/core.js") \
               < body.index("spectra/viewer.js")
        # Shared form-schema helper loaded BEFORE the inspector core
        # (the core's initSchemaForm() calls into formSchema.fetchSchema).
        assert body.index("lib/form-schema.js") \
               < body.index("lib/spectra/core.js")

    def test_page_is_generate_only(self, web_client):
        """The Spectrum-calculation page is generate-only -- the
        inspect-side surface (load controls, modes table, mode viewer,
        ES panel) lives in _spectra_inspector.html and is served only
        to /results via GET /partials/spectra-inspector.

        Pinning the EXCLUSION here gives a clear failure mode if a
        future commit accidentally re-includes the partial on the
        generator page (which would re-introduce the UX-debt the
        generator/inspector split was made to clean up).
        """
        r = web_client.get("/spectrum-calculation")
        assert r.status_code == 200, (
            f"/spectrum-calculation returned {r.status_code}; this "
            f"test's negative-body assertions would pass trivially "
            f"against an error page"
        )
        body = r.data.decode()
        for inspect_only_id in (
            "watch-path",              # load-by-path input
            "load-path-btn",           # one-shot load button
            "watch-btn",               # live-watch toggle
            "watch-stop-btn",          # live-watch stop
            "load-from-selection-btn", # sidebar handoff
            "phase-indicator",         # phase-progress chips
            "results-summary",         # results meta + chart container
            "spectrum-chart",          # plotly chart
            "broadening-fwhm",         # FWHM input
            "modes-tbody",             # modes table body
            "modes-filter",            # modes filter input
            "modes-csv-btn",           # CSV export
            "mode-viewer-wrap",        # 3D mode viewer card
            "mode-viewer",             # 3Dmol canvas
            "anim-amplitude",          # animation controls
            "anim-speed",
            "anim-toggle",
            "es-panel",                # ES bar diagram panel
            "es-bar-diagram",
            "workspace-indicator",     # workspace readout (was in partial)
        ):
            needle = f'id="{inspect_only_id}"'
            assert needle not in body, (
                f"/spectrum-calculation body unexpectedly carries "
                f"{needle!r} -- the page is generate-only; inspect-side "
                f"ids belong in the _spectra_inspector.html partial "
                f"served by GET /partials/spectra-inspector"
            )

    def test_app_header_includes_spectrum_tab(self, web_client):
        """The shared header lists Spectrum calculation among the
        canonical 5 tabs -- regression check against a future header
        refactor dropping the entry."""
        # /molbuilder is one of the canonical landing pages; reuse
        # it to fetch a header-rendered body.
        r = web_client.get("/molbuilder")
        body = r.data.decode()
        assert 'href="/spectrum-calculation"' in body

    def test_viewer_js_served_as_bootstrap_stub(self, web_client):
        """Post-task-#296 (2026-06-09) spectra/viewer.js carries
        TWO responsibilities, both /spectra-specific:

          1. Mount the shared spectra-inspector core against
             ``document`` so the schema form + Generate / Methods
             / Issues / script-preview / Save handlers wire up.
             (Unchanged pre-#296 behaviour.)

          2. Bootstrap the Inspect-structure card: mount a 3Dmol
             embed in ``#viewer-wrap`` + wire the
             ``#load-from-sidebar-btn`` so the user can pick a
             structure in the Projects sidebar and load it into
             the viewer.  Mirrors the Optimization tab pattern.

        The file is still small (no per-feature business logic;
        that lives in lib/spectra/core.js), but it's no longer
        the ~1.5 KB stub it was post-step-2.2.  Cap at 16 KB so
        the legacy 1700-line controller can't slip back in.
        """
        r = web_client.get("/static/spectra/viewer.js")
        assert r.status_code == 200
        js = r.data.decode()
        # The bootstrap calls into the shared core.
        assert "spectraInspector" in js
        assert "mount(document)"  in js
        # And it wires the Inspect-structure card (task #296).
        assert "load-from-sidebar-btn" in js, (
            "spectra/viewer.js must wire the Load-from-sidebar "
            "button — that's the sole structure entry point post-#296"
        )
        assert "viewer-wrap" in js or "#viewer" in js, (
            "spectra/viewer.js must mount the 3Dmol embed in the "
            "Inspect-structure card's #viewer / #viewer-wrap slot"
        )
        # Cap to catch the legacy 1700-line controller creeping
        # back; well above the bootstrap's natural size.
        assert len(js) < 16_000, (
            f"spectra/viewer.js grew to {len(js)} bytes -- it should "
            f"stay as bootstrap-only wiring; per-feature logic "
            f"belongs in lib/spectra/core.js"
        )

    def test_core_js_served(self, web_client):
        """The shared inspector core module is reachable as a static
        asset and contains the three Spectra API endpoint URLs +
        the generate-side hooks.
        """
        r = web_client.get("/static/lib/spectra/core.js")
        assert r.status_code == 200
        js = r.data.decode()
        # Three /api/spectra/* endpoints + the form-schema fetch.
        assert "/api/build/schema/spectra" in js
        assert "/api/spectra/render"        in js
        assert "/api/spectra/load"          in js
        # Selector / compatibility logic present (locks unused
        # ES value fields when the selector changes).
        assert "applyCompatibility" in js
        # Methods-preview modal handler.
        assert "openMethodsModal"   in js
        # Mount API export -- the contract both consumers depend on.
        assert "spectraInspector" in js
        assert "mount: mountInspector" in js

    def test_style_css_served(self, web_client):
        """The CSS imports the shared tokens so theming stays in
        lock-step with Build / Modify / Watch.

        Pre-task-#296 ``.spectra-grid`` was the bespoke two-column
        layout; that class is gone and the page now inherits the
        shared ``.app-grid`` vertical workflow from style.css.
        We still pin two load-bearing rules so a regression that
        empties the file fails here: ``.issues-panel`` (renders
        the engine validation Issues) and ``.script-preview``
        (the Generated-script block).  Both are
        spectra-specific — the shared sheet doesn't own them. """
        r = web_client.get("/static/spectra/style.css")
        assert r.status_code == 200
        css = r.data.decode()
        assert "tokens.css" in css
        assert ".issues-panel" in css, (
            "spectra/style.css is missing .issues-panel — the "
            "Issues block on /spectra would render unstyled"
        )
        assert ".script-preview" in css, (
            "spectra/style.css is missing .script-preview — the "
            "Generated-script preview on /spectra would render "
            "unstyled"
        )

    def test_shared_form_schema_css_served(self, web_client):
        """The schema-driven form's layout (fieldset / label /
        .param-grid) lives in static/lib/form-schema.css so any
        page that renders a schema form picks up the same look.
        The Spectra template includes it; without it the form
        fields fall back to browser defaults and look scattered.
        """
        r = web_client.get("/static/lib/form-schema.css")
        assert r.status_code == 200
        css = r.data.decode()
        # Three load-bearing rules.
        for needle in (
            "fieldset {",
            "fieldset label",
            ".param-grid {",
            ".schema-int-triple",
        ):
            assert needle in css, f"missing {needle!r} in form-schema.css"

    def test_spectra_page_includes_shared_form_schema_css(self, web_client):
        body = web_client.get("/spectrum-calculation").data.decode()
        # The template imports the shared form-schema CSS so the
        # rendered <fieldset>s pick up the consistent layout.
        assert "lib/form-schema.css" in body
        # And the form container has the .param-grid class so its
        # fieldsets lay out in the responsive auto-fit grid.
        assert 'class="param-grid"' in body

    def test_inspector_partial_has_path_load_and_watch_controls(self, web_client):
        """The spectra-inspector partial (served to /results, and the
        single source of truth post-step-2.5) exposes the server-side
        path-load input + Watch toggle.  /spectra no longer carries
        these markup chunks; they live exclusively in the partial.
        """
        body = web_client.get("/partials/spectra-inspector").data.decode()
        # Primary path input + three action buttons.
        assert 'id="watch-path"'     in body
        assert 'id="load-path-btn"'  in body
        assert 'id="watch-btn"'      in body
        assert 'id="watch-stop-btn"' in body
        # Three phase indicator dots, one per phase.
        for ph in ("frequencies", "raman", "es"):
            assert f'data-phase="{ph}"' in body

    def test_core_js_has_path_load_and_watch_loop(self, web_client):
        """The shared inspector core exposes loadByPath (one-shot
        path load), startWatch / stopWatch / watchTick (the live
        poller), and the allPhasesComplete / updatePhaseIndicator
        helpers.  These moved out of viewer.js into lib/spectra/
        core.js in step 2.2 of the tab-merge lift."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        for sym in (
            "loadByPath",
            "startWatch",
            "stopWatch",
            "watchTick",
            "allPhasesComplete",
            "updatePhaseIndicator",
            "WATCH_INTERVAL_MS",
        ):
            assert sym in js, f"missing {sym!r} in lib/spectra/core.js"

    def test_path_load_endpoint_works(self, web_client, tmp_path):
        """End-to-end sanity: POST /api/spectra/load with {path}
        succeeds for a real on-disk spectra.json (mirrors what the
        new 'Load once' button does).  Regression check against a
        future blueprint change accidentally breaking the path-input
        mode of the load endpoint."""
        from molbuilder.parsers.spectra_json import dump_spectra_json
        results = _make_minimal_results()
        p = tmp_path / "live.spectra.json"
        dump_spectra_json(results, p)
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(p)}),
            content_type="application/json",
        )
        body = r.get_json()
        assert r.status_code == 200, body
        assert body["ok"]
        assert body["results"]["engine"] == "pyscf"

    def test_plotly_not_loaded_on_spectra_generator(self, web_client):
        """/spectrum-calculation is the GENERATOR tab (configure +
        emit script).  Plotly is a results-viewing library used only
        by the spectra inspector chart; loading it on the generator
        page was dead weight (>1 MB of unused JS per page view).  Pin
        that the generator template does NOT pull Plotly, and that
        the chart id lives ONLY in the inspector partial (consumed
        by /results)."""
        r = web_client.get("/spectrum-calculation")
        assert r.status_code == 200
        body = r.data.decode()
        assert "/vendor/plotly.min.js" not in body, (
            "the spectrum-calculation generator tab must NOT load "
            "Plotly; it's a results-viewing library that belongs in "
            "the inspector partial only."
        )
        # ``spectrum-chart`` is an inspect-side id; verify it's served
        # by the partial endpoint rather than by /spectra.
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        assert 'id="spectrum-chart"' in partial

    def test_vendor_plotly_route_serves_js(self, web_client):
        """The /vendor/plotly.min.js route serves the file from the
        installed plotly package.  Returns JS bytes when plotly is
        importable; 404 otherwise (the spectra page degrades to the
        existing 'Plotly not loaded' fallback)."""
        r = web_client.get("/vendor/plotly.min.js")
        try:
            import plotly  # noqa: F401
            assert r.status_code == 200, r.status_code
            assert "application/javascript" in r.content_type
            # First few bytes of plotly.min.js are an IIFE / use-strict
            # banner.  Just confirm we got a sizeable JS payload.
            assert len(r.data) > 100_000, len(r.data)
        except ImportError:
            assert r.status_code == 404

    def test_core_js_has_chart_renderer(self, web_client):
        """The chart-renderer function is in the inspector core and
        builds traces for both display modes (activity / density)."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "renderSpectrumChart" in js
        # Trace buckets present.
        assert '"Real"'              in js
        assert '"Imaginary"'         in js
        # Density-mode names (used when no mode has a Raman activity
        # -- partial L2-done / L3-not-yet runs).
        assert '"Real (freq only)"'      in js
        assert '"Imaginary (freq only)"' in js
        # Partial-L3 marker.
        assert '"Raman pending"'     in js
        # Detection of density mode.
        assert "densityMode"         in js
        # Plotly entry point.
        assert "Plotly.react" in js

    def test_core_js_has_lorentzian_envelope(self, web_client):
        """The Lorentzian broadening helper, the trace name, and the
        FWHM input wiring are all in the inspector core."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "_lorentzianEnvelope" in js
        # The trace name includes the FWHM value at render time.
        assert "Lorentzian (FWHM" in js
        # Input change handler.
        assert "onBroadeningChange" in js
        # State holds the current FWHM.
        assert "broadeningFWHM" in js

    def test_inspector_partial_has_mode_viewer(self, web_client):
        """3Dmol viewer + controls live below the modes table in the
        inspector partial.  Pins the IDs the core JS depends on.

        Historical note: a prior assertion pinned "3Dmol must NOT be
        on /spectra (the generator)".  That guard was correct
        pre-task #296 (2026-06-09) when the spectra generator page
        had no viewer surface.  Task #296 reorganised
        /spectrum-calculation into the same vertical-workflow shape
        as /structure-optimization, mounting the structure-inspector
        embed in the "Inspect structure" card so users can preview
        what's about to be calculated.  3Dmol is now legitimately
        loaded on /spectrum-calculation for that purpose; the
        assertion was retired with #296.

        The CDN-fallback guard (no cdnjs.cloudflare.com) is kept
        as a separate ``test_no_cdn_fallback_in_spectrum_page``
        below — the rationale (vendored 3Dmol, no CDN) survives the
        workflow change."""
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        # Viewer wrap + canvas div.
        assert 'id="mode-viewer-wrap"'   in partial
        assert 'id="mode-viewer"'        in partial
        # Controls.
        assert 'id="anim-amplitude"'     in partial
        assert 'id="anim-speed"'         in partial
        assert 'id="anim-toggle"'        in partial

    def test_no_cdn_fallback_in_spectrum_page(self, web_client):
        """The 3Dmol script tag on /spectrum-calculation must point
        at the vendored copy (``/static/vendor/3Dmol-min.js``) —
        never a CDN.  Pre-task #296 the generator carried no 3Dmol
        at all; #296 added the inspect-structure card and pulled in
        the same vendored stack as /structure-optimization.  Pin so
        a future refactor that "just adds a CDN fallback" silently
        introducing a third-party load surfaces."""
        r = web_client.get("/spectrum-calculation")
        assert r.status_code == 200
        body = r.data.decode()
        assert "cdnjs.cloudflare.com" not in body, (
            "/spectrum-calculation must not load 3Dmol from a CDN; "
            "use the vendored copy at /static/vendor/3Dmol-min.js"
        )
        # And the vendored path IS present (positive assertion so
        # the test fails clearly if the script tag is removed
        # entirely rather than swapped for a CDN).
        assert "/static/vendor/3Dmol-min.js" in body, (
            "/spectrum-calculation should ship the vendored 3Dmol "
            "for the inspect-structure card (task #296)"
        )

    def test_core_js_has_mode_animation(self, web_client):
        """The inspector core exposes the viewer setup + animation
        entry points.

        Note: ``_parseXyz`` / ``_geomToXyz`` moved to the shared
        ``static/lib/xyz-io.js`` (window.molbuilder.xyz.{parse,toText})
        on 2026-05-16 — the symbol-name pin updates accordingly."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        for sym in (
            "renderModeViewer",
            "_ensureViewer",
            "_startAnimation",
            "_stopAnimation",
            "_equilibriumGeometry",
            "molbuilder.xyz.parse",
            "$3Dmol",
        ):
            assert sym in js, f"missing {sym!r} in lib/spectra/core.js"

    def test_inspector_partial_has_broadening_control(self, web_client):
        """The Spectrum subsection in the inspector partial has a
        FWHM number input above the chart, with a default of 20 cm⁻¹."""
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        assert 'id="broadening-fwhm"' in partial
        assert 'value="20"'           in partial
        # Hint text mentions Lorentzian.
        assert "Lorentzian" in partial

    def test_inspector_partial_has_es_panel_and_table_controls(self, web_client):
        """The inspector partial exposes the mode-table interaction
        controls (filter + CSV) and the ES-panel scaffolding the
        JS targets."""
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        # Table-control row.
        assert 'id="modes-filter"'        in partial
        assert 'id="modes-csv-btn"'       in partial
        assert 'id="modes-filter-count"'  in partial
        # Sortable headers carry the data-col attribute the JS
        # reads to decide what to sort by.
        assert 'data-col="frequency_cm1"'         in partial
        assert 'data-col="raman_activity_a4_amu"' in partial
        # ES-panel scaffolding.
        assert 'id="es-panel"'         in partial
        assert 'id="es-bar-diagram"'   in partial
        assert 'id="es-summary"'       in partial
        # Caption for screen readers.
        assert "Vibrational modes table"          in partial

    def test_core_js_has_selection_sync_and_es_renderer(self, web_client):
        """The inspector core exposes selectMode (click-row /
        click-stick synchronisation) and renderESPanel (per-mode MO
        diagram).  Pins the contract the inspector-partial markup
        depends on."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        # Public-ish API of the interaction layer.
        for sym in (
            "selectMode",            # called from row click + chart click
            "renderESPanel",         # MO bar diagram + summary
            "renderModesTable",      # sort + filter render
            "exportCSV",             # export button
            "_onChartClick",         # plotly_click handler
            "EH_TO_EV",              # Hartree -> eV conversion constant
        ):
            # JS supports both function decls and assignments; the
            # symbol just has to APPEAR in the source.
            assert sym in js, f"missing {sym!r} in lib/spectra/core.js"


# --------------------------------------------------------------------- #
#  Dispose contract                                                     #
# --------------------------------------------------------------------- #


class TestSpectraDisposeContract:
    """Pins for the spectra-inspector dispose() contract.

    The 2026-05-18 review surfaced a latent leak: element listeners
    attached inside mountInspector were never explicitly torn down.
    On /spectra the page lives forever and never disposes, so the
    bug is dormant; on /results the host's innerHTML clearing GCs the
    nodes (and with them the listeners) — also dormant.  Dormant
    bugs are the kind that wake up when the architecture grows; the
    fix routed every addEventListener through an ``_on()`` helper
    that captures a teardown closure, and ``dispose()`` walks them
    in reverse.

    These tests pin the structural invariant of that fix: the
    cleanups array exists, every element-listener registration goes
    through ``_on``, and ``dispose()`` actually runs the
    teardowns + per-resource cleanups.  Pin SHAPE, not specific
    sites — so the test survives reordering of the wiring block but
    fails loudly if someone reintroduces a direct
    ``els.foo.addEventListener(...)`` that leaks past dispose.
    """

    def test_cleanups_array_and_on_helper_exist(self, web_client):
        """The closure-local listener bookkeeping is the contract
        backbone.  If a future refactor drops _cleanups or _on, the
        rest of this test class becomes moot — pin the existence
        first so a regression here surfaces with a clear failure."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        # The cleanups array (closure-local list of teardown closures).
        assert "const _cleanups = []" in js, (
            "lib/spectra/core.js no longer declares the _cleanups "
            "array; dispose() can't tear down listeners without it"
        )
        # The _on() registration helper that captures teardowns.
        assert "function _on(target, event, handler" in js, (
            "lib/spectra/core.js no longer defines the _on() helper "
            "that wraps addEventListener + pushes a teardown into "
            "_cleanups"
        )

    def test_dispose_walks_cleanups_before_per_resource_cleanups(
            self, web_client):
        """dispose() must drain _cleanups FIRST so timer/raf
        callbacks (which themselves may dispatch events to listeners)
        don't fire against torn-down DOM.  Ordering matters."""
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        # Locate dispose() body.
        m = re.search(
            r"dispose\(\)\s*\{(.+?)\n        \},", js, re.DOTALL)
        assert m, "could not locate dispose() body in lib/spectra/core.js"
        body = m.group(1)
        # Order check: _cleanups.pop()() before clearInterval.
        cleanups_idx = body.find("_cleanups.pop()")
        cleartimer_idx = body.find("clearInterval(state.watchTimer)")
        assert cleanups_idx > -1, (
            "dispose() does not walk _cleanups — listeners leak"
        )
        assert cleartimer_idx > -1, (
            "dispose() does not clear state.watchTimer — poll leaks"
        )
        assert cleanups_idx < cleartimer_idx, (
            "dispose() must walk _cleanups BEFORE clearing timers "
            "so timer callbacks don't fire against torn-down listeners"
        )

    def test_dispose_clears_every_long_lived_resource(self, web_client):
        """dispose() must tear down every long-lived resource the
        mount allocated: watch poller, the mode-viewer embed handle
        (which owns its own vibration rAF + 3Dmol viewer per #231
        Part B), and the Plotly chart.  Pinning each cleanup site
        rather than just host.innerHTML="" so a future refactor that
        drops, say, the Plotly.purge call lands with a clear failure
        (and not silently with `host cleared therefore dispose
        worked`).
        """
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        m = re.search(
            r"dispose\(\)\s*\{(.+?)\n        \},", js, re.DOTALL)
        assert m
        body = m.group(1)
        for needle, what in (
            ("clearInterval(state.watchTimer)",
                "live-watch poller (state.watchTimer)"),
            ("state.handle.dispose()",
                "mode-viewer embed handle (state.handle — owns the "
                "vibration rAF + the 3Dmol viewer)"),
            ("Plotly.purge(els.spectrumChart)",
                "Plotly spectrum chart"),
        ):
            assert needle in body, (
                f"dispose() does not tear down {what} — searched for "
                f"{needle!r} in the dispose body"
            )

    def test_all_element_listeners_route_through_on_helper(
            self, web_client):
        """Catches the class of regression where someone adds a
        direct ``els.foo.addEventListener("click", handler)`` inside
        mountInspector.  Such a direct call escapes _cleanups and
        leaks past dispose — the same class of bug the
        2026-05-18 review surfaced.

        ``_on()`` itself contains the only legitimate
        ``addEventListener`` call site; anything else is a leak.
        """
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        # Count BARE .addEventListener( calls.
        adds = re.findall(r"\.addEventListener\(", js)
        # The _on() helper makes exactly one such call.  Anything
        # more means a direct registration that won't be torn down.
        assert len(adds) == 1, (
            f"expected exactly 1 .addEventListener call in lib/"
            f"spectra/core.js (the one inside _on()), found "
            f"{len(adds)} — a recent change has added a direct "
            f"event-listener registration that escapes the "
            f"_cleanups array; route it through _on() so dispose() "
            f"tears it down"
        )
        # Sanity: the _on() helper actually gets used a lot.  Today
        # the wiring block calls _on() ~17 times across the generate-
        # and inspect-side gates.  Pinning a floor (rather than the
        # exact count) so reordering the wiring doesn't break the
        # test, but a wholesale revert to direct addEventListener
        # does.
        on_calls = re.findall(r"\b_on\(", js)
        assert len(on_calls) >= 10, (
            f"only {len(on_calls)} _on() call sites in lib/spectra/"
            f"core.js — the listener-tracking refactor may have been "
            f"partially reverted"
        )

    def test_dispose_is_idempotent(self, web_client):
        """A second dispose() call after the first must not throw.
        Defensive against the registry calling dispose twice (which
        it shouldn't, but a future refactor might).  The closure
        guards each cleanup with a null-check + try/catch, so a
        twice-drained _cleanups array + already-nulled state should
        be a no-op rather than an error."""
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        m = re.search(
            r"dispose\(\)\s*\{(.+?)\n        \},", js, re.DOTALL)
        assert m
        body = m.group(1)
        # Each per-resource cleanup is guarded by an `if (state.X)`
        # check or a `typeof Plotly !== "undefined"` check that
        # makes a second call a no-op.  Pin the guards explicitly.
        # The vibration animation + 3Dmol viewer are owned by the
        # mode-viewer embed handle (#231 Part B); their guard is
        # ``if (state.handle)`` against the handle reference, not
        # the legacy ``state.animTimer`` / ``state.viewer`` slots.
        for guard in (
            "if (state.watchTimer)",
            "if (state.handle)",
            'typeof Plotly !== "undefined"',
        ):
            assert guard in body, (
                f"dispose() missing idempotency guard {guard!r} -- "
                f"a second dispose() call may throw"
            )


# --------------------------------------------------------------------- #
#  Inspector core parity: spectra core MUST honor opts.file like        #
#  trajectory core does.                                                #
#                                                                       #
#  Today's bug class (2026-05-18 user report): the spectra lift copied  #
#  the inspector body from spectra/viewer.js but skipped the            #
#  ``if (opts.file) loadByPath(opts.file);`` block trajectory's lift    #
#  established as the contract.  Static review didn't catch it because  #
#  every individual file LOOKED right.  Behavioural parity catches it.  #
# --------------------------------------------------------------------- #


class TestSpectraCoreHonorsOptsFile:
    """Pin the cross-inspector parity contract: a mount() invocation
    that passes ``opts.file`` must auto-load that file.

    This is the test that would have caught today's regression --
    static markup + symbol pins all passed because the parts existed
    in isolation; the cross-module contract (adapter passes file,
    core honors file) had no test.  Today's user-visible failure --
    pick .spectra.json in sidebar, /results mounts empty inspector
    -- was the consequence.
    """

    def test_spectra_core_branches_on_opts_file(self, web_client):
        """spectra/core.js must contain the ``if (opts.file)`` branch
        + a call into the file-loading path.  Pin both pieces so the
        regression mode is unambiguous when this test fails: either
        the branch is missing (the gate dropped) or the loader call
        is missing (gate kept but does nothing).

        Pin SHAPE not exact text -- e.g., trajectory uses
        ``loadByPath(opts.file)`` directly while spectra pre-fills
        watch-path and calls ``loadByPath()`` with no args; both
        satisfy the contract.
        """
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        # The gate.  Either form is acceptable: a strict
        # ``if (opts.file)`` block OR an ``opts.file && ...`` guard.
        gate_patterns = (
            r"if\s*\(\s*opts\.file",
            r"opts\.file\s*&&",
        )
        gate_hit = any(re.search(p, js) for p in gate_patterns)
        assert gate_hit, (
            "lib/spectra/core.js has no ``opts.file``-gated branch; "
            "the /results-side mount passes the sidebar selection as "
            "opts.file and the core has to honour it or the inspector "
            "mounts empty.  See trajectory/core.js for the reference "
            "shape (around line 1505)."
        )
        # The actual load call inside the gate.  We can't easily
        # locate "inside the gate" without parsing JS, so just pin
        # that loadByPath is called somewhere AFTER the init() call
        # at the bottom of mountInspector (the gate's location).  Use
        # the rfind anchor: the bottom-of-mount loadByPath() invocation
        # is distinct from the listener wiring's ``_on(..., loadByPath)``
        # which doesn't actually call it.
        # A simpler proxy: look for ``loadByPath()`` (with parens) which
        # is a CALL, not a reference -- the wiring uses ``_on(...,
        # loadByPath)`` (no parens) so a positive match here is
        # definitely a call site.
        call_sites = re.findall(r"loadByPath\(\)", js)
        assert call_sites, (
            "lib/spectra/core.js has the opts.file gate but never "
            "calls loadByPath() -- the gate is a no-op.  The "
            "auto-load contract requires calling the load function "
            "from inside the gate."
        )

    def test_trajectory_core_has_same_pattern(self, web_client):
        """Anchor for the parity test above: the trajectory core
        MUST have the same shape, otherwise the spectra test's
        notion of "parity" is meaningless.  Pin trajectory's gate +
        call site so a future trajectory refactor that drops the
        contract also fails here (and we'd notice the divergence
        instead of silently making the parity test trivially true).
        """
        import re
        js = web_client.get("/static/lib/trajectory/core.js").data.decode()
        assert re.search(r"if\s*\(\s*opts\.file", js), (
            "lib/trajectory/core.js no longer has the opts.file gate "
            "-- did the auto-load contract change?  If yes, update "
            "TestSpectraCoreHonorsOptsFile to match the new contract."
        )
        assert "loadByPath(opts.file)" in js or "loadByPath()" in js, (
            "lib/trajectory/core.js no longer calls loadByPath inside "
            "the opts.file branch -- contract drift; the parity test "
            "for spectra is now meaningless"
        )


# --------------------------------------------------------------------- #
#  Dead-controls test: every interactive element in the partial must    #
#  have a handler bound somewhere in loaded JS.                         #
#                                                                       #
#  Today's button-stays-gray bug: ``#load-from-selection-btn`` lived in #
#  the partial but its wiring lived in static/spectra/page.js, which    #
#  was deleted in step 2.5 of the lift.  On /results the button         #
#  rendered disabled forever.  This test catches that class of bug.     #
# --------------------------------------------------------------------- #


class TestSpectraPartialHasNoDeadControls:
    """Every interactive element in the spectra inspector partial
    must have a handler binding in code that's loaded on the
    consumer page.  Catches "orphan control" bugs where a button
    survives a refactor but its event listener got deleted with
    the page-specific bootstrap that wired it.
    """

    def test_every_interactive_id_has_a_js_reference(self, web_client):
        """Walk every ``id="X"`` on interactive elements (button /
        input / select / textarea / a-with-href / etc.) in the
        rendered spectra inspector partial.  For each, assert at
        least one of the loaded core/adapter JS files references X
        in a way that suggests wiring: ``$("X")``, ``$#X``,
        ``getElementById("X")``, or ``querySelector("#X")``.
        """
        import re
        partial = web_client.get(
            "/partials/spectra-inspector").data.decode()
        core_js = web_client.get(
            "/static/lib/spectra/core.js").data.decode()
        adapter_js = web_client.get(
            "/static/lib/inspectors/spectra.js").data.decode()
        # Concat all the JS that loads on /results for the spectra
        # inspector -- a handler binding in any of them counts as
        # "wired".  If a future inspector grows another JS file,
        # add it here.
        wired_sources = core_js + "\n" + adapter_js
        # Find interactive ids in the partial.  Restrict to tags
        # the user clicks/types into: <button>, <input>, <select>,
        # <textarea>, <a> with href.  Static <p> / <span> ids
        # (e.g., status banners) are output sinks, not interactive,
        # so they don't need handlers.
        interactive_tags = re.findall(
            r'<(button|input|select|textarea)\b[^>]*\bid="([^"]+)"',
            partial,
            re.IGNORECASE,
        )
        # De-dup while preserving order for stable failure messages.
        seen = set()
        ids = []
        for tag, ident in interactive_tags:
            if ident in seen:
                continue
            seen.add(ident)
            ids.append((tag, ident))
        assert ids, (
            "no interactive ids found in the spectra inspector "
            "partial -- markup regex may be broken or the partial "
            "lost all controls (a regression bigger than this test)"
        )
        orphans = []
        for tag, ident in ids:
            patterns = (
                rf'\$\("{re.escape(ident)}"\)',
                rf'getElementById\("{re.escape(ident)}"\)',
                rf'querySelector\("#{re.escape(ident)}"\)',
                rf'\bbuttonId:\s*"{re.escape(ident)}"',
            )
            if not any(re.search(p, wired_sources) for p in patterns):
                orphans.append(f"<{tag} id={ident!r}>")
        assert not orphans, (
            "Interactive element(s) in the spectra inspector "
            "partial have NO handler binding in any loaded JS "
            "(core.js, adapter.js).  The user sees a control "
            "with no behaviour:\n  "
            + "\n  ".join(orphans)
            + "\n\nIf the control is intentionally inert (pure "
            "visual marker), drop the id.  If it's supposed to be "
            "wired, add the handler -- this is the class of bug "
            "where the wiring file was deleted but the markup "
            "survived (today's #load-from-selection-btn case)."
        )


# --------------------------------------------------------------------- #
#  Schema endpoint                                                      #
# --------------------------------------------------------------------- #


class TestSchemaEndpoint:

    def test_schema_returns_ok_with_sections(self, web_client):
        r = web_client.get("/api/build/schema/spectra")
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert "schema" in body
        # Top-level shape mirrors siesta/pyscf: dict with "sections".
        assert "sections" in body["schema"]
        section_names = [s["name"] for s in body["schema"]["sections"]]
        # Workflow-order from SpectraConfig._form_section_order.
        assert section_names[:4] == [
            "System", "Method", "Frozen atoms", "Spectrum",
        ]
        assert "Electronic structure" in section_names

    def test_schema_field_ids_are_prefixed(self, web_client):
        """Schema endpoint stamps each field with id_prefix-derived id;
        the JS form renderer uses these ids verbatim."""
        body = web_client.get("/api/build/schema/spectra").get_json()
        by_name = {f["name"]: f
                   for s in body["schema"]["sections"]
                   for f in s["fields"]}
        # Spectra prefix is "s" per the blueprint.
        assert by_name["job_name"]["id"].startswith("s-")
        # id_suffix override applies (see config metadata).
        assert by_name["job_name"]["id"] == "s-job-name"
        assert by_name["es_mode_selection"]["id"] == "s-es-selection"

    def test_schema_includes_es_fields(self, web_client):
        body = web_client.get("/api/build/schema/spectra").get_json()
        by_name = {f["name"]: f
                   for s in body["schema"]["sections"]
                   for f in s["fields"]}
        assert "es_mode_selection" in by_name
        assert "es_top_n"           in by_name
        assert "es_threshold"       in by_name
        assert "es_explicit_indices" in by_name
        assert "freq_min_cm1"       in by_name
        assert "freq_max_cm1"       in by_name

    def test_schema_sections_carry_descriptions(self, web_client):
        """Each form section ships a one-paragraph description so
        the renderer can surface "what is this group for?" right
        below the legend.  Pin the contract -- adding a new section
        without a description should fail this test (and remind the
        author to write one)."""
        body = web_client.get("/api/build/schema/spectra").get_json()
        for sect in body["schema"]["sections"]:
            assert "description" in sect, (
                f"section {sect['name']!r} is missing its "
                f"`description` field"
            )
            assert len(sect["description"]) > 60, (
                f"section {sect['name']!r} description is too short "
                f"to be useful: {sect['description']!r}"
            )


class TestSchemaEndpointFrozenSeed:
    """Stage 1 of the three-stage contract (design.md "Sidecar-driven
    boundary conditions"): when the schema endpoint is called with
    ``?structure_path=<xyz>`` and the sidecar carries non-empty
    ``frozen_atoms``, the ``frozen_indices`` field's default in the
    returned schema is overridden with the comma-separated form of
    those indices.  This is what makes the boundary condition
    visible in the form before Generate.  Tests pin both the happy
    path AND the recoverable-failure paths (no sidecar / mismatch /
    corrupt -- the endpoint returns the static schema plus a
    ``notice`` rather than failing)."""

    def _setup_root(self, monkeypatch, tmp_path):
        """Reuse the file_picker_roots redirection trick from
        test_selection_blueprint.py so the schema endpoint accepts
        the tmp_path-rooted structure_path."""
        from molbuilder import diagnostics
        caps = diagnostics.Capabilities(
            runtime_config={},
            conda_binary=None,
            conda_envs=frozenset(),
        )
        cls = type(caps)
        old = cls.file_picker_roots

        def _only_tmp_roots(self):
            return ((tmp_path.resolve(), "projects"),)
        monkeypatch.setattr(cls, "file_picker_roots", _only_tmp_roots)
        diagnostics.set_capabilities(caps)
        # cleanup: the autouse _reset_diagnostics_singleton handles
        # the singleton reset; monkeypatch handles the class attr.

    def _xyz_path(self, tmp_path):
        p = tmp_path / "junction.xyz"
        p.write_text(
            "4\nfour-Au\n"
            "Au 0 0 0\n" "Au 1 0 0\n"
            "Au 2 0 0\n" "Au 3 0 0\n"
        )
        return p

    def _write_sidecar(self, xyz_path, *, frozen_atoms, n_atoms_total=None):
        import hashlib
        sha = hashlib.sha256(xyz_path.read_bytes()).hexdigest()
        sidecar = xyz_path.with_name(xyz_path.stem + ".molstruct.json")
        sidecar.write_text(json.dumps({
            "schema_version":  3,
            "n_atoms_total":   n_atoms_total if n_atoms_total is not None
                                else xyz_path.read_text().split()[0].__int__(),
            "structure_hash":  sha,
            "regions":         {},
            "frozen_atoms":    list(frozen_atoms),
            "selection_rules": {},
        }))
        return sidecar

    def _find_frozen_indices_default(self, body):
        for sect in body["schema"]["sections"]:
            for f in sect["fields"]:
                if f["name"] == "frozen_indices":
                    return f.get("default")
        return None

    def test_no_structure_path_no_override(
        self, web_client, tmp_path, monkeypatch,
    ):
        """No ``?structure_path`` arg -> static schema (regression
        against existing callers; previous default behaviour
        preserved)."""
        self._setup_root(monkeypatch, tmp_path)
        r = web_client.get("/api/build/schema/spectra")
        assert r.status_code == 200
        body = r.get_json()
        # The static default for frozen_indices is whatever
        # _serialize_default returns for a default_factory=list
        # field (currently None); the key thing is "no notice".
        assert "notice" not in body
        # And it's not a populated string from any sidecar.
        default = self._find_frozen_indices_default(body)
        assert default in (None, "", []), default

    def test_structure_path_with_no_sidecar_no_override(
        self, web_client, tmp_path, monkeypatch,
    ):
        """structure_path supplied, but no sidecar next to the
        XYZ -> static schema (no notice, no override)."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._xyz_path(tmp_path)
        r = web_client.get(
            "/api/build/schema/spectra?structure_path=" + str(xyz),
        )
        assert r.status_code == 200
        body = r.get_json()
        assert "notice" not in body
        default = self._find_frozen_indices_default(body)
        assert default in (None, "", []), default

    def test_sidecar_frozen_atoms_prefills_field_default(
        self, web_client, tmp_path, monkeypatch,
    ):
        """Happy path: sidecar with ``frozen_atoms=[1,3]`` -> the
        schema's ``frozen_indices`` field carries default = "1, 3"
        so the form renders pre-filled."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._xyz_path(tmp_path)
        self._write_sidecar(xyz, frozen_atoms=[1, 3], n_atoms_total=4)
        r = web_client.get(
            "/api/build/schema/spectra?structure_path=" + str(xyz),
        )
        assert r.status_code == 200
        body = r.get_json()
        # No error notice on the happy path.
        assert "notice" not in body, body.get("notice")
        # The frozen_indices field's default reflects the sidecar.
        default = self._find_frozen_indices_default(body)
        assert default == "1, 3", default

    def test_sidecar_empty_frozen_atoms_no_override(
        self, web_client, tmp_path, monkeypatch,
    ):
        """Sidecar exists but its ``frozen_atoms`` list is empty
        -> no override (no point pre-filling an empty value)."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._xyz_path(tmp_path)
        self._write_sidecar(xyz, frozen_atoms=[], n_atoms_total=4)
        r = web_client.get(
            "/api/build/schema/spectra?structure_path=" + str(xyz),
        )
        assert r.status_code == 200
        body = r.get_json()
        assert "notice" not in body
        default = self._find_frozen_indices_default(body)
        assert default in (None, "", []), default

    def test_sidecar_atom_count_mismatch_surfaces_notice(
        self, web_client, tmp_path, monkeypatch,
    ):
        """A stale sidecar (n_atoms_total != actual XYZ atoms)
        means its indices may point at the wrong atoms; refuse to
        pre-fill and surface a notice telling the user to
        re-export from /modify.  No silent absorption -- the user
        must be told.  Pinned per the three-stage contract."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._xyz_path(tmp_path)
        # n_atoms_total=99 but the actual XYZ has 4 atoms.
        self._write_sidecar(xyz, frozen_atoms=[1, 3], n_atoms_total=99)
        r = web_client.get(
            "/api/build/schema/spectra?structure_path=" + str(xyz),
        )
        body = r.get_json()
        assert "notice" in body, body
        assert "atom count" in body["notice"].lower()
        # And no override (the form stays empty so the user can't
        # be misled by stale indices).
        default = self._find_frozen_indices_default(body)
        assert default in (None, "", []), default

    def test_structure_path_outside_root_surfaces_notice(
        self, web_client, tmp_path, monkeypatch,
    ):
        """A structure_path that resolves outside the file-picker
        roots -> notice (path rejected) and no override.  Same
        allow-list as /api/files/* (defense in depth -- this
        endpoint shouldn't be a back door)."""
        self._setup_root(monkeypatch, tmp_path)
        outside = "/etc/passwd"
        r = web_client.get(
            "/api/build/schema/spectra?structure_path=" + outside,
        )
        body = r.get_json()
        assert "notice" in body, body
        default = self._find_frozen_indices_default(body)
        assert default in (None, "", []), default


# --------------------------------------------------------------------- #
#  Render endpoint                                                      #
# --------------------------------------------------------------------- #


class TestRenderEndpoint:

    def _render(self, web_client, **body):
        return web_client.post(
            "/api/spectra/render",
            data=json.dumps(body),
            content_type="application/json",
        )

    def test_happy_path_returns_script(self, web_client):
        r = self._render(web_client, structure_text=_WATER_XYZ, params={})
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        # The script is a non-empty Python string starting with the
        # docstring header.
        assert body["script"].startswith('"""PySCF Spectra')
        # Methods text + bibliography keys delivered alongside so the
        # UI doesn't need a separate round-trip for the Methods modal.
        assert "## Methods" in body["methods_md"]
        assert "Sun2020" in body["bibliography_keys"]
        assert body["job_name"] == "spectra"

    def test_render_with_compute_raman_off(self, web_client):
        r = self._render(web_client, structure_text=_WATER_XYZ,
                         params={"compute_raman": False})
        assert r.status_code == 200
        body = r.get_json()
        # Raman block absent in the script when off.
        assert "Phase 3: Raman" not in body["script"]
        # Komornicki1979 only fires for Raman; should NOT be in bib.
        assert "Komornicki1979" not in body["bibliography_keys"]

    def test_render_compiles_as_python(self, web_client):
        """The script delivered over the wire must compile -- this is
        the cheapest catchall for syntax bugs in the template that
        only manifest with certain configs."""
        r = self._render(web_client, structure_text=_WATER_XYZ, params={
            "compute_raman":     True,
            "es_mode_selection": "explicit",
            "es_explicit_indices": "1,2,3",
        })
        body = r.get_json()
        compile(body["script"], "<wire-render>", "exec")

    def test_missing_structure_text_400(self, web_client):
        """Empty body (no ``structure_text``) -> 400 with a
        structured error.  The body field was renamed 2026-05-22
        from the misleading ``"xyz"`` (which also accepted PDB)
        to ``"structure_text"``."""
        r = self._render(web_client, params={})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False
        assert "structure_text" in r.get_json()["error"]

    def test_render_accepts_pdb_text(self, web_client):
        """The render endpoint accepts PDB text (auto-detected by
        content, falls through to Structure.from_pdb).  Broadened
        2026-05-22 so the user can drive /spectra against PDB
        structures, not just XYZ.  See design.md "Module init
        contract" + selection.py PDB suffix dispatch."""
        pdb_water = (
            "HEADER    TEST\n"
            "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
            "ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H\n"
            "ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H\n"
            "END\n"
        )
        r = self._render(web_client, structure_text=pdb_water, params={})
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["script"].startswith('"""PySCF Spectra')

    def test_unparseable_structure_text_400(self, web_client):
        """Unparseable structure text (neither XYZ nor PDB) -> 400
        with a "could not parse structure text" error.  The
        ``xyz`` body field name is historical -- as of 2026-05-22
        the render endpoint sniffs the content and dispatches to
        from_xyz / from_pdb."""
        r = self._render(web_client, structure_text="not an xyz or pdb", params={})
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "could not parse structure text" in body["error"].lower()

    def test_unsupported_method_blocks_render(self, web_client):
        """Preflight catches an unknown method -- response is 400
        with the error issue surfaced."""
        r = self._render(web_client, structure_text=_WATER_XYZ, params={
            "method": "BOGUS_METHOD",
        })
        # Bad enum may either fail coercion (HTTP 400 from param parse)
        # OR pass through and trip preflight (HTTP 400 with issues).
        # Either way, response is 400 + ok=False.
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False

    def test_preflight_error_returns_issues(self, web_client):
        """top_n selector without prior L3 produces an error-severity
        issue from selection.validate_selection.  Response is 400."""
        r = self._render(web_client, structure_text=_WATER_XYZ, params={
            "es_mode_selection": "top_n",
            "es_top_n":          5,
        })
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "issues" in body
        error_issues = [i for i in body["issues"] if i["severity"] == "error"]
        assert any(i["where"] == "config.es_mode_selection"
                   for i in error_issues)

    def test_preflight_warns_pass_through_to_response(self, web_client):
        """A warn-severity issue (e.g. displacement amplitude outside
        the defensible window) doesn't block render -- it's
        delivered alongside the script."""
        r = self._render(web_client, structure_text=_WATER_XYZ, params={
            "displacement_amplitude_ang": 0.25,  # above the 0.20 cap
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        warns = [i for i in body["issues"] if i["severity"] == "warn"]
        assert any(i["where"] == "config.displacement_amplitude_ang"
                   for i in warns)

    def test_prior_path_missing_passes_through_as_warn(self, web_client, tmp_path):
        """A bogus prior_path is non-fatal -- preflight just doesn't
        get the L3-completed signal; the user sees a warn so they
        know the resume context was ignored."""
        r = self._render(web_client, structure_text=_WATER_XYZ,
                         params={},
                         prior_path=str(tmp_path / "nonexistent.json"))
        assert r.status_code == 200
        body = r.get_json()
        warns = [i for i in body["issues"] if i["severity"] == "warn"]
        assert any("prior" in i["message"].lower() for i in warns)

    def test_prior_path_valid_enables_top_n(self, web_client, tmp_path):
        """A real prior with phase_raman=complete unblocks the
        top_n selector preflight."""
        prior = _make_minimal_results()  # phase_raman=COMPLETE in fixture
        prior_path = tmp_path / "prior.spectra.json"
        dump_spectra_json(prior, prior_path)
        r = self._render(web_client, structure_text=_WATER_XYZ,
                         params={"es_mode_selection": "top_n",
                                 "es_top_n":          3},
                         prior_path=str(prior_path))
        # Top-n now passes the soft-dep check; render goes through.
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True


class TestRenderHonorsSidecar:
    """End-to-end check for the three-stage contract (design.md
    "Sidecar-driven boundary conditions"): the render endpoint
    must apply the sidecar to the structure BEFORE preflight runs,
    so preflight's stage-3 guards (Pattern A divergence, Pattern B
    unrecognized labels) actually see ``struct.frozen_atoms`` +
    ``struct.regions`` populated.  Without this, the rest of the
    contract is pinned only against synthetic structs in unit
    tests but silently inert in the live web flow."""

    def _setup_root(self, monkeypatch, tmp_path):
        from molbuilder import diagnostics
        caps = diagnostics.Capabilities(
            runtime_config={},
            conda_binary=None,
            conda_envs=frozenset(),
        )
        cls = type(caps)

        def _only_tmp_roots(self):
            return ((tmp_path.resolve(), "projects"),)
        monkeypatch.setattr(cls, "file_picker_roots", _only_tmp_roots)
        diagnostics.set_capabilities(caps)

    def _write_water_with_sidecar(self, tmp_path, frozen_atoms, regions=None):
        import hashlib
        xyz = tmp_path / "water.xyz"
        xyz.write_text(_WATER_XYZ)
        sha = hashlib.sha256(xyz.read_bytes()).hexdigest()
        sidecar = xyz.with_name(xyz.stem + ".molstruct.json")
        sidecar.write_text(json.dumps({
            "schema_version":  3,
            "n_atoms_total":   3,
            "structure_hash":  sha,
            "regions":         regions or {},
            "frozen_atoms":    list(frozen_atoms),
            "selection_rules": {},
        }))
        return xyz

    def test_sidecar_frozen_divergence_warns_in_render(
        self, web_client, tmp_path, monkeypatch,
    ):
        """Live render call: sidecar says freeze atom 0, form says
        freeze nothing -> preflight Pattern A WARN fires in the
        response.  Pin the end-to-end stage-3 contract."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._write_water_with_sidecar(tmp_path, frozen_atoms=[0])
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure_text": _WATER_XYZ,
                "structure_path": str(xyz),
                "params":         {},  # empty form -> no frozen_indices
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        warns = [i for i in body.get("issues", [])
                 if i["severity"] == "warn"
                 and i["where"] == "config.frozen_indices"
                 and "sidecar" in i["message"]]
        assert warns, [i for i in body["issues"]]

    def test_sidecar_regions_unrecognized_warns_in_render(
        self, web_client, tmp_path, monkeypatch,
    ):
        """Live render call: sidecar has region labels, /spectra
        doesn't consume them -> preflight Pattern B WARN fires.
        Pin the end-to-end stage-3 contract for the labels-the-
        engine-doesn't-understand case."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._write_water_with_sidecar(
            tmp_path,
            frozen_atoms=[],
            regions={"L-electrode": [0], "bridge": [1, 2]},
        )
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure_text": _WATER_XYZ,
                "structure_path": str(xyz),
                "params":         {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        warns = [i for i in body.get("issues", [])
                 if i["severity"] == "warn"
                 and i["where"] == "structure.regions"]
        assert warns
        # The notice should name the actual labels so the user knows
        # WHICH ones are being ignored.
        assert "L-electrode" in warns[0]["message"]
        assert "bridge" in warns[0]["message"]

    def test_render_without_structure_path_keeps_working(
        self, web_client,
    ):
        """No structure_path supplied (the pasted-XYZ-only flow) ->
        no sidecar load attempted, no stage-3 warn, render still
        succeeds.  Regression test for the backward-compat path:
        existing callers that don't know about structure_path keep
        working."""
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure_text": _WATER_XYZ,
                "params": {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        # No stage-3 warns possible (no sidecar applied), but other
        # preflight signals may fire (small-system, etc.).  Just
        # confirm the sidecar-specific warns are absent.
        sidecar_warns = [i for i in body.get("issues", [])
                         if "sidecar" in i.get("message", "")
                         or i.get("where") == "structure.regions"]
        assert sidecar_warns == []

    def test_render_with_bad_structure_path_surfaces_notice(
        self, web_client,
    ):
        """A structure_path that resolves outside the roots: render
        still succeeds (the form's freeze rules are authoritative),
        but the response carries a warn-severity Issue at
        ``structure_path`` so the user knows their sidecar is NOT a
        factor in this run.  Per "no silent absorption" rule
        (design.md three-stage contract)."""
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure_text": _WATER_XYZ,
                "structure_path": "/etc/passwd",
                "params":         {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        warns = [i for i in body.get("issues", [])
                 if i["severity"] == "warn"
                 and i["where"] == "structure_path"]
        assert warns, [i for i in body.get("issues", [])]
        # The notice tells the user the form is the sole truth so
        # they aren't surprised by "I thought atoms X were frozen".
        assert "form" in warns[0]["message"].lower()

    def test_render_with_corrupt_sidecar_surfaces_notice(
        self, web_client, tmp_path, monkeypatch,
    ):
        """A sidecar that exists but fails to load (atom-count
        mismatch with the pasted XYZ, malformed, etc.): render
        still succeeds (form authoritative), and the response
        carries a warn-severity Issue naming the failure -- per
        "no silent absorption"."""
        self._setup_root(monkeypatch, tmp_path)
        # Sidecar promises 99 atoms; water has 3.
        xyz = self._write_water_with_sidecar(
            tmp_path, frozen_atoms=[0],
        )
        # Now corrupt the sidecar to have n_atoms_total=99 (apply
        # raises MolstructJsonError on the mismatch).
        sidecar = xyz.with_name(xyz.stem + ".molstruct.json")
        import hashlib
        sha = hashlib.sha256(xyz.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps({
            "schema_version":  3,
            "n_atoms_total":   99,
            "structure_hash":  sha,
            "regions":         {},
            "frozen_atoms":    [],
            "selection_rules": {},
        }))
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure_text": _WATER_XYZ,
                "structure_path": str(xyz),
                "params":         {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        warns = [i for i in body.get("issues", [])
                 if i["severity"] == "warn"
                 and i["where"] == "structure_path"]
        assert warns, body["issues"]
        # The notice should be specific (mention the sidecar) and
        # tell the user the form is now authoritative.
        assert "sidecar" in warns[0]["message"].lower()
        assert "form" in warns[0]["message"].lower()


# --------------------------------------------------------------------- #
#  Load endpoint                                                        #
# --------------------------------------------------------------------- #


class TestLoadEndpoint:

    def test_multipart_upload_round_trip(self, web_client, tmp_path):
        original = _make_minimal_results()
        p = tmp_path / "x.spectra.json"
        dump_spectra_json(original, p)
        with open(p, "rb") as fh:
            r = web_client.post(
                "/api/spectra/load",
                data={"file": (io.BytesIO(fh.read()), "x.spectra.json")},
                content_type="multipart/form-data",
            )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["results"]["engine"] == "pyscf"
        assert len(body["results"]["modes"]) == 1

    def test_path_input(self, web_client, tmp_path):
        original = _make_minimal_results()
        p = tmp_path / "y.spectra.json"
        dump_spectra_json(original, p)
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(p)}),
            content_type="application/json",
        )
        body = r.get_json()
        assert r.status_code == 200, body
        assert body["ok"] is True
        assert body["results"]["n_atoms_total"] == 2

    def test_inline_json_input(self, web_client):
        original = _make_minimal_results()
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"json": original.to_dict()}),
            content_type="application/json",
        )
        body = r.get_json()
        assert r.status_code == 200, body
        assert body["ok"] is True

    def test_no_input_400(self, web_client):
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "no input" in body["error"].lower()

    def test_inline_non_dict_rejected(self, web_client):
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"json": [1, 2, 3]}),
            content_type="application/json",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert body["kind"] == "malformed"

    def test_missing_file_path_404(self, web_client, tmp_path):
        bad = tmp_path / "does_not_exist.json"
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(bad)}),
            content_type="application/json",
        )
        assert r.status_code == 404
        body = r.get_json()
        assert body["kind"] == "not_found"

    def test_schema_mismatch_422_with_versions(self, web_client, tmp_path):
        """An older / newer schema_version surfaces as 422 with the
        expected + actual versions so the UI can render an "update
        molbuilder" hint without parsing strings."""
        original = _make_minimal_results()
        d = original.to_dict()
        d["schema_version"] = SCHEMA_VERSION + 1
        p = tmp_path / "future.spectra.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(p)}),
            content_type="application/json",
        )
        assert r.status_code == 422
        body = r.get_json()
        assert body["kind"]             == "schema_mismatch"
        assert body["expected_version"] == SCHEMA_VERSION
        assert body["actual_version"]   == SCHEMA_VERSION + 1

    def test_malformed_json_400(self, web_client, tmp_path):
        p = tmp_path / "bad.spectra.json"
        p.write_text("{not valid json", encoding="utf-8")
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(p)}),
            content_type="application/json",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["kind"] == "malformed"

    def test_field_error_400(self, web_client, tmp_path):
        original = _make_minimal_results()
        d = original.to_dict()
        del d["engine"]
        p = tmp_path / "fielderr.spectra.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"path": str(p)}),
            content_type="application/json",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["kind"] == "field"

    def test_upload_non_utf8_malformed(self, web_client):
        r = web_client.post(
            "/api/spectra/load",
            data={"file": (io.BytesIO(b"\xff\xfeinvalid utf-8"),
                            "bad.spectra.json")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["kind"] == "malformed"
