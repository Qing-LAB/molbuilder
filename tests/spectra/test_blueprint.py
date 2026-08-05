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

from molbuilder.sidecars.spectra import dump_spectra_json
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
        #
        # task #62: the inspect card migrated to the concealed MolView
        # module -- an EMPTY ``#spectra-molview-host`` that molview.mount
        # builds the read-only fused card into (the old ``#viewer`` id-based
        # 3Dmol mount is gone; the module builds a ``.viewer`` CLASS inside).
        assert 'id="spectra-molview-host"'      in body
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

        The file is still small (no per-feature business logic
        for the shared spectra inspector; that lives in
        lib/spectra/core.js + is reused on /results), but it
        carries the page-specific bootstrap for the
        /spectrum-calculation workflow: 3Dmol embed mount, Load-
        from-sidebar wiring, and (post-2026-06-10) the
        Auto-detect chemistry handler that pre-fills the form
        from /api/structure/analyze.

        Cap at 22 KB so the legacy 1700-line controller can't
        slip back in; the cap was 16 KB before the Auto-detect
        handler landed and is bumped here to keep room for the
        ~3 KB it adds plus a little future headroom.  The Auto-
        detect handler is intentionally NOT in core.js — it's
        /spectrum-calculation-specific page wiring (the same
        handler would not make sense for the inspector partial
        consumed on /results, where there's no parameter form
        to pre-fill).
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
        # Auto-detect handler — page-specific wiring post-2026-06-10.
        assert "auto-detect-btn" in js, (
            "spectra/viewer.js must wire the Auto-detect button "
            "(the scientific-guard step from "
            "scientific-validation.md § 2.5).  Same chemistry that "
            "drove the hemeC-dithiol incident lives here too — "
            "auto-detect surfaces it before Generate."
        )
        # Cap to catch the legacy 1700-line controller creeping back.
        #
        # MEASURED ON CODE, NOT ON THE FILE (2026-08-03).  The cap was
        # `len(js) < 22_000` on the raw bytes, and the file crossed it at
        # 22,602 -- of which 8,839 bytes, 39%, are explanatory comments.  The
        # code was 13.7 KB and had not grown.  So the test failed for writing
        # down WHY the wiring is the way it is, which is the house style
        # everywhere else in this repo; it was penalising documentation and
        # would have been "fixed" by deleting it.  What the invariant means is
        # that no per-feature LOGIC lives here, so that is what is measured.
        import re as _re
        code = _re.sub(r"/\*.*?\*/", "", js, flags=_re.S)
        code = _re.sub(r"^\s*//.*$", "", code, flags=_re.M)
        code = _re.sub(r"\n\s*\n+", "\n", code)
        assert len(code) < 18_000, (
            f"spectra/viewer.js has {len(code)} bytes of CODE (file is "
            f"{len(js)}) -- it should stay as bootstrap-only wiring; "
            f"per-feature logic belongs in lib/spectra/core.js"
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
        from molbuilder.sidecars.spectra import dump_spectra_json
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

    def test_the_tab_reaches_the_mode_viewer_by_injection(self, web_client):
        """The Spectrum tab is HANDED the mode viewer; it does not go looking.

        REPLACES a test that listed six private function names in core.js.  Four
        of them changed when the viewer was rebuilt, and the test failed for a
        rename that broke nothing — which is what a transcription does instead of
        guarding a contract (vibrationview.md § 14, § 13.1 of molview.md).

        What is actually load-bearing is the DIRECTION of the dependency.  The
        core is a classic script and cannot import a module; the module publishes
        nothing to a global for it to find, deliberately — reaching for a global
        is precisely how the previous viewer came to be unmountable on every page
        while its own tests stayed green.  So the registry adapter, which can do
        both, imports `mount` and hands it over.
        """
        core    = web_client.get("/static/lib/spectra/core.js").data.decode()
        adapter = web_client.get("/static/lib/inspectors/spectra.js").data.decode()

        # The adapter is a module, and it hands the capability in.
        assert "import " in adapter, "the adapter must import the viewer it provides"
        assert "/static/lib/vibrationview/index.js" in adapter
        assert "mountVibrationView" in adapter

        # The core takes it as an option and never looks one up.
        assert "mountVibrationView" in core
        assert "molbuilder.vibrationview" not in core, (
            "the core must not look the viewer up in a global; it is handed one"
        )

    def test_the_tab_does_not_name_the_drawing_library(self, web_client):
        """A 3-D viewer is a module now, so the tab has no business knowing what
        draws it (vibrationview.md § 5.4).

        The core used to check for the library itself and render its own "failed
        to load" message, which meant two places knew the module was built on it.
        """
        core = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "$3Dmol" not in core

    def test_the_tab_hands_over_one_partition_not_two(self, web_client):
        """§ 6.3: which atoms move is ONE fact, as far as the VIEWER is concerned.

        The core used to hand `showMode` both a free set and a frozen set — two
        lists that must partition the atoms, with nothing checking they did, so an
        atom named in both would be greyed as frozen while being moved as free.
        The viewer derives the held-still set from the basis it is given.

        Narrowly about the handoff: the page still READS the frozen list, to show
        "Free / frozen" in the run-metadata table, and that is not the viewer's
        business either way.
        """
        core = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "basis:" in core, "the mode handed over must name its basis"
        assert "frozenAtomIdx" not in core, (
            "the viewer is given one partition and derives the other"
        )

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
        don't fire against torn-down DOM.  Ordering matters.

        Per results-state-contract.md § 2 ("All state changes go
        through one function") the canonical site for poll-timer
        cleanup is ``transition("IDLE")`` — its IDLE branch in
        lib/spectra/core.js clears state.lifecycle.watchTimer
        (guarded with an ``if`` check, so a second call is a
        no-op).  This test asserts the ordering of the canonical
        chain: _cleanups drained, THEN transition("IDLE").
        """
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        m = re.search(
            r"dispose\(\)\s*\{(.+?)\n        \},", js, re.DOTALL)
        assert m, "could not locate dispose() body in lib/spectra/core.js"
        body = m.group(1)
        cleanups_idx = body.find("_cleanups.pop()")
        idle_idx     = body.find('transition("IDLE")')
        assert cleanups_idx > -1, (
            "dispose() does not walk _cleanups — listeners leak"
        )
        assert idle_idx > -1, (
            "dispose() does not call transition(\"IDLE\") — "
            "the canonical state-machine cleanup site per "
            "results-state-contract.md § 2"
        )
        assert cleanups_idx < idle_idx, (
            "dispose() must walk _cleanups BEFORE calling "
            "transition(\"IDLE\") so torn-down listeners don't "
            "fire on the bucket-clear cascade"
        )

    def test_dispose_clears_every_long_lived_resource(self, web_client):
        """dispose() must tear down every long-lived resource the
        mount allocated: watch poller (cleared via the canonical
        ``transition("IDLE")`` site per results-state-contract.md
        § 2), the VibrationView mode viewer (``state.vib`` — the
        concealed normal-mode animation package, vibrationview.md,
        which owns its own vibration rAF + 3Dmol canvas; #231 Part B
        migrated the spectra mode viewer onto it, renaming the old
        ``state.handle`` embed field to ``state.vib``), and the
        Plotly chart.  Pinning each cleanup site rather than just
        host.innerHTML="" so a future refactor that drops, say,
        the Plotly.purge call lands with a clear failure (and not
        silently with `host cleared therefore dispose worked`).
        """
        import re
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        m = re.search(
            r"dispose\(\)\s*\{(.+?)\n        \},", js, re.DOTALL)
        assert m
        body = m.group(1)
        for needle, what in (
            ('transition("IDLE")',
                "live-watch poller (cleared via the canonical "
                "transition('IDLE') site per "
                "results-state-contract.md § 2)"),
            ("state.vib.dispose()",
                "VibrationView mode viewer (state.vib — the concealed "
                "normal-mode animation package owns the vibration rAF "
                "+ the 3Dmol canvas; vibrationview.md)"),
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
        # Each per-resource cleanup is guarded so a second dispose()
        # call is a safe no-op:
        #
        # * The watch-poller cleanup lives one level deep, inside
        #   transition("IDLE")'s IDLE branch (see lib/spectra/core.js
        #   IDLE-branch ``if (state.lifecycle.watchTimer)`` guard).
        #   dispose() invokes ``transition("IDLE")`` — that's the
        #   canonical state-machine site per
        #   results-state-contract.md § 2 ("All state changes go
        #   through one function").  A second dispose() call hits
        #   the same guard against a now-null timer and is a no-op.
        # * The VibrationView mode viewer (#231 Part B renamed the old
        #   ``state.handle`` embed field to ``state.vib``) owns the
        #   vibration rAF + 3Dmol canvas; guarded by ``if (state.vib)``
        #   against the handle reference (vibrationview.md).
        # * Plotly purge is guarded by ``typeof Plotly !== "undefined"``.
        for guard in (
            'transition("IDLE")',
            "if (state.vib)",
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
            "schema_version":  7,
            "n_atoms_total":   n_atoms_total if n_atoms_total is not None
                                else xyz_path.read_text().split()[0].__int__(),
            "structure_hash":  sha,
            # THE ONE STORE (v7): the reserved label sits in `regions` with every
            # other label. What makes it reserved is the interpretation applied
            # where it means something -- Geometry.Constraints -- and the one
            # accessor that pulls the group out, not a field of its own.
            "regions":         {"frozen_atoms": list(frozen_atoms)}
                               if frozen_atoms else {},
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


def _envelope(xyz: str, regions: dict = None) -> dict:
    """The molecule as data -- built by the ONE builder
    (`tests/support/envelope.py`), which goes through `Structure.to_dict()`.

    This split the XYZ into elements and positions by hand, and five other
    test files each had their own copy of that split.  Six hand-rolled XYZ
    parsers in the suite, for a shape the codec already produces."""
    from support.envelope import from_xyz
    return from_xyz(xyz, regions=regions)



class TestRenderEndpoint:

    def _render(self, web_client, **body):
        return web_client.post(
            "/api/spectra/render",
            data=json.dumps(body),
            content_type="application/json",
        )

    def test_happy_path_returns_script(self, web_client):
        r = self._render(web_client, structure=_envelope(_WATER_XYZ), params={})
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
        r = self._render(web_client, structure=_envelope(_WATER_XYZ),
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
        r = self._render(web_client, structure=_envelope(_WATER_XYZ), params={
            "compute_raman":     True,
            "es_mode_selection": "explicit",
            "es_explicit_indices": "1,2,3",
        })
        body = r.get_json()
        compile(body["script"], "<wire-render>", "exec")

    def test_a_body_with_no_structure_is_a_400(self, web_client):
        """No molecule in the body -> 400 with a structured error.

        Replaced `test_missing_structure_text_400`, which named the field that
        used to carry an XYZ document. The route takes the molecule as data now
        (web-api.md § 1), so the missing thing is the structure itself.
        """
        r = self._render(web_client, params={})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False
        assert "structure" in r.get_json()["error"].lower()

    def test_a_malformed_structure_is_a_400(self, web_client):
        """A molecule that cannot be built -> 400, refused at the boundary
        rather than becoming a half-built structure downstream.

        This replaced two tests about unparseable and PDB *text*. Neither has a
        subject any more: the route parses no documents. A person who wants to
        open a PDB does it through the load door, which reads the file and hands
        back a structure — one place that turns bytes into a molecule, instead of
        every route that needs one growing its own parser.
        """
        r = self._render(web_client,
                         structure={"elements": ["O"], "positions": []},
                         params={})
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert body["error"]

    def test_unsupported_method_blocks_render(self, web_client):
        """Preflight catches an unknown method -- response carries
        the error issue surfaced.

        Per web-api.md § 1.6: bad enum has two possible paths --
        (a) coercion fails in _spectra_config_from_params -> HTTP
        400 + ok:false (protocol error, class (c)); (b) coercion
        accepts the string and preflight catches it -> HTTP 200 +
        ok:false (scientific advisory, class (b)).  Either status
        is contract-compliant; the test pins the ok:false outcome.
        """
        r = self._render(web_client, structure=_envelope(_WATER_XYZ), params={
            "method": "BOGUS_METHOD",
        })
        assert r.status_code in (200, 400), r.status_code
        body = r.get_json()
        assert body["ok"] is False

    def test_preflight_error_returns_issues(self, web_client):
        """top_n selector without prior L3 produces an error-severity
        issue from selection.validate_selection.

        Per web-api.md § 1.6 (b) this is the scientific-advisory
        bucket — HTTP 200 + ok:false; the form's workflow cards
        render the findings inline.
        """
        r = self._render(web_client, structure=_envelope(_WATER_XYZ), params={
            "es_mode_selection": "top_n",
            "es_top_n":          5,
        })
        assert r.status_code == 200, r.status_code
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
        r = self._render(web_client, structure=_envelope(_WATER_XYZ), params={
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
        r = self._render(web_client, structure=_envelope(_WATER_XYZ),
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
        r = self._render(web_client, structure=_envelope(_WATER_XYZ),
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
            "schema_version":  7,
            "n_atoms_total":   3,
            "structure_hash":  sha,
            # THE ONE STORE (v7): the reserved label goes in with the rest.
            "regions":         dict(regions or {},
                                    **({"frozen_atoms": list(frozen_atoms)}
                                       if frozen_atoms else {})),
            "selection_rules": {},
        }))
        return xyz

    def test_sidecar_frozen_divergence_warns_in_render(
        self, web_client, tmp_path, monkeypatch,
    ):
        """Live render call: the MODEL says freeze atom 0, the form says
        freeze nothing -> preflight Pattern A WARN fires in the response.  Pin
        the end-to-end stage-3 contract.

        The labels arrive in the BODY, the way the tab sends them
        (molview.data.getStructure); before F2 this test delivered them by
        writing a sidecar next to the .xyz and letting the server read it,
        which is the second source the contract removed
        (science/validation.md 4.1)."""
        self._setup_root(monkeypatch, tmp_path)
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                # The reserved label rides in the ONE store, and the store
                # rides INSIDE the structure -- a top-level `regions` beside
                # the envelope was the second source, retired 2026-08-03.
                "structure": _envelope(_WATER_XYZ,
                                       regions={"frozen_atoms": [0]}),
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
        """Live render call: the model carries region labels, /spectra doesn't
        consume them -> preflight Pattern B WARN fires.  Pin the end-to-end
        stage-3 contract for the labels-the-engine-doesn't-understand case.
        Labels ride in the body (F2), not a sidecar on disk."""
        self._setup_root(monkeypatch, tmp_path)
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure": _envelope(
                    _WATER_XYZ,
                    regions={"L-electrode": [0], "bridge": [1, 2]}),
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
                "structure": _envelope(_WATER_XYZ),
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

    def test_a_sidecar_on_disk_is_never_absorbed(
        self, web_client, tmp_path, monkeypatch,
    ):
        """"No silent absorption" (design.md three-stage contract) under F2
        (science/validation.md 4.1).

        This used to be a pair of WARNINGS: the server tried to read a sidecar
        from ``structure_path``, and said so when the path was outside the roots
        or the sidecar failed to load.  The server no longer reads it at all, so
        there is nothing to absorb and nothing to warn about -- what the sidecar
        says cannot reach the run, whether it is valid, stale or corrupt.
        """
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._write_water_with_sidecar(tmp_path, frozen_atoms=[0])
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure": _envelope(_WATER_XYZ),
                "structure_path": str(xyz),   # names the file; not a label source
                "params":         {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        # The sidecar froze atom 0; the run must show no sign of it.
        assert not [i for i in body.get("issues", [])
                    if i["where"] == "config.frozen_indices"
                    and "sidecar" in i["message"]], body.get("issues")

    def test_a_path_WITH_label_facts_ignores_disk_entirely(
        self, web_client, tmp_path, monkeypatch,
    ):
        """The other half: when the body carries the facts, a stale or corrupt
        sidecar next to the .xyz is irrelevant -- it is never read, so it can
        neither help nor break the run."""
        self._setup_root(monkeypatch, tmp_path)
        xyz = self._write_water_with_sidecar(tmp_path, frozen_atoms=[0])
        sidecar = xyz.with_name(xyz.stem + ".molstruct.json")
        sidecar.write_text("{ this is not valid json")      # would have raised
        r = web_client.post(
            "/api/spectra/render",
            data=json.dumps({
                "structure": _envelope(_WATER_XYZ),
                "structure_path": str(xyz),
                "params":         {},
            }),
            content_type="application/json",
        )
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert not [i for i in body.get("issues", [])
                    if i["where"] == "structure_path"], (
            "the sidecar must not be consulted when the body carries facts")


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


class TestModeAnimationControls:
    """The two controls the mode viewer grew when it was rebuilt.

    Both are the TAB's, not the module's: VibrationView draws no controls at all
    (vibrationview.md § 5.4), holds no frequency and no physical constant
    (§ 12.2), and produces bytes without deciding where they go (§ 12).
    """

    def test_the_partial_offers_both_ways_of_asking_how_big(self, web_client):
        """§ 12.2: display is a drawing choice, physical is a quantity.

        Two pairings, and the control has to make them exclusive — there is no
        slider for a physical amplitude because there is nothing to slide: the
        size follows from the mode's own frequency.
        """
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        assert 'id="anim-amplitude-mode"' in partial
        assert 'value="zero-point"' in partial
        assert 'value="thermal"' in partial
        # thermal is the only one that takes a temperature, so its box starts hidden
        assert 'id="anim-temperature"' in partial
        assert 'id="anim-temperature-row"' in partial
        assert "hidden" in partial.split('id="anim-temperature-row"')[1][:80]

    def test_the_partial_offers_the_three_export_formats(self, web_client):
        """§ 12.1: they are not interchangeable, so the user picks."""
        partial = web_client.get("/partials/spectra-inspector").data.decode()
        for value in ('value="png-zip"', 'value="webm"', 'value="gif"'):
            assert value in partial, value
        assert 'id="anim-export-width"' in partial
        assert 'id="anim-export-background"' in partial
        assert 'value="transparent"' in partial
        assert 'id="anim-export-cycles"' in partial
        # a long export must be stoppable (§ 12)
        assert 'id="anim-export-cancel"' in partial

    def test_the_physics_lives_in_the_tab_not_the_viewer(self, web_client):
        """§ 12.2: "both are computed by the TAB… the physics of how big a
        vibration is belongs with the spectrum, not with the viewer".

        The module contains no frequency, no temperature and no physical
        constant; this is the file that does.
        """
        core = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "ZERO_POINT_Q" in core
        assert "CM1_IN_KELVIN" in core
        assert "Math.tanh" in core, "the thermal form needs coth"

        for name in ("_maths.js", "index.js"):
            js = web_client.get("/static/lib/vibrationview/" + name).data.decode()
            for forbidden in ("frequency", "kelvin", "Kelvin", "tanh"):
                assert forbidden not in js, (
                    f"{name} names {forbidden!r}: the viewer holds no physics"
                )

    def test_the_export_records_the_amplitude_with_its_normalization(self, web_client):
        """§ 12.2: the amplitude is meaningless without the normalization beside
        it — the two pairings do not share a unit, so either number alone invites
        the wrong caption."""
        core = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "out.meta.normalization" in core, (
            "what is reported after an export must name the normalization"
        )
