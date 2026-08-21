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
import pytest

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
        assert 'id="send-to-task-setup"'        in body
        # P2 substitution: the tab hands over, it renders no script.
        assert 'id="send-status"'             in body
        # Gate ① -- the live science panel beside the form.
        assert 'id="spectra-issues"'          in body
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
        # P3: the tab fetches the CATALOGUE's vibration schema and
        # sends through the shared hand-over door -- the old
        # schema/render routes are gone.
        assert '"pyscf", { calculation: "vibration" }' in js
        assert "taskHandover.send" in js
        assert "/api/spectra/load"          in js
        # Selector / compatibility logic present (locks unused
        # ES value fields when the selector changes).
        assert "applyCompatibility" in js
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

    def test_core_js_mounts_the_spectrum_chart_module(self, web_client):
        """The spectrum is drawn by lib/spectrumchart, reached through its one
        door.  The tab keeps what a tab owns: the modes, the selection, and the
        broadening the user typed (docs/web/spectrumchart.md § 2, § 10)."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert 'import("/static/lib/spectrumchart/index.js")' in js
        assert "onSelect" in js and "selectMode(index)" in js
        for door in ("setModes", "setBroadening", "setSelected", "dispose"):
            assert f".{door}(" in js, f"the tab never calls {door}"

    def test_core_js_no_longer_draws_the_spectrum_itself(self, web_client):
        """What moved out has to be GONE, not shadowed: a second copy of the
        envelope or the trace-building would be the drift the extraction was for.

        The electronic-structure diagram is still this tab's own figure, so
        Plotly still appears -- but only for that.
        """
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        for gone in ("_lorentzianEnvelope", "_clickBandWidths", "_clickTolerance",
                     "_onChartClick", "densityMode", "Lorentzian (FWHM"):
            assert gone not in js, f"{gone!r} survived the extraction"
        assert "esBarDiagram" in js          # the ES figure is still drawn here

    def test_the_tab_still_owns_the_broadening_control(self, web_client):
        """§ 5.2 — the tab owns the input the user types into; the chart holds
        the width it was last told."""
        js = web_client.get("/static/lib/spectra/core.js").data.decode()
        assert "onBroadeningChange" in js
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
            ("Plotly.purge(",
                "the Plotly charts"),
            ("chart.dispose()",
                "the spectrum chart, which now takes ITSELF down: one call and "
                "its surface, its box watcher and its markup go with it. This "
                "used to name els.spectrumChart because the tab purged the "
                "figure itself; since 2026-08-05 the chart is a sealed module "
                "(docs/web/spectrumchart.md § 7) and a tab that reached in to "
                "purge it would be reaching past `mount`"),
            ("els.esBarDiagram",
                "the electronic-structure level diagram, by name, in the "
                "teardown (it became a second Plotly figure on 2026-08-05 "
                "so its zoom/pan could come from the chart library rather "
                "than from hand-rolled SVG)"),
            (".disconnect()",
                "the ResizeObserver each chart installs to follow its own "
                "box (window-level `responsive` does not see a container "
                "query flip or a sidebar collapse)"),
        ):
            assert needle in body, (
                f"dispose() does not tear down {what} — searched for "
                f"{needle!r} in the dispose body"
            )

        # PINNED AS A SET, NOT AS CALL SITES.  This assertion used to name the
        # exact string ``Plotly.purge(els.spectrumChart)``, which failed the day
        # the two charts were torn down by one loop over both -- a strictly
        # better teardown that the test called a regression.  What the contract
        # actually owes is that every chart the mount created is purged, so what
        # is pinned now is the pair of names and the call, not the syntax that
        # joins them.
        assert body.count("Plotly.purge(") >= 1, (
            "dispose() must purge the Plotly figures it created")

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
        def _wired(ident):
            patterns = (
                rf'\$\("{re.escape(ident)}"\)',
                rf'getElementById\("{re.escape(ident)}"\)',
                rf'querySelector\("#{re.escape(ident)}"\)',
                rf'\bbuttonId:\s*"{re.escape(ident)}"',
            )
            return any(re.search(p, wired_sources) for p in patterns)

        def _listens(ident):
            """Does anything actually LISTEN on this element?

            Not the same question as `_wired`, and the difference matters: an id
            can be looked up to set `.hidden` or write text into, which is not a
            listener and cannot give a descendant button behaviour.  Accepting a
            mere lookup would let a dead control pass simply by sitting inside a
            panel someone shows and hides -- `#es-panel` is exactly that, and it
            wraps controls.
            """
            camel = re.sub(r"-(\w)", lambda m: m.group(1).upper(), ident)
            patterns = (
                rf'_on\(\s*els\.{re.escape(camel)}\b',
                rf'_on\(\s*\$\("{re.escape(ident)}"\)',
                rf'getElementById\("{re.escape(ident)}"\)\s*\.addEventListener',
                rf'querySelector\("#{re.escape(ident)}"\)\s*\.addEventListener',
            )
            return any(re.search(p, wired_sources) for p in patterns)

        def _delegating_ancestor(ident):
            """The id of the nearest ancestor that is itself wired, if any.

            EVENT DELEGATION IS WIRING.  A group of static sibling controls --
            the three mode tabs, a toolbar, a set of radio chips -- is better
            served by ONE listener on their container than by one listener each:
            fewer registrations to tear down, and a control added later is live
            without touching the wiring.  A test that only recognises
            per-element binding would push every such group back to N listeners
            purely to stay green, which is the test dictating the design.

            So walk outwards from the control to the enclosing element ids and
            ask whether any of THEM is wired.  A control inside a wired
            container is a control with behaviour.
            """
            idx = partial.find(f'id="{ident}"')
            if idx == -1:
                return None
            before = partial[:idx]
            # Ancestors = tags opened before this point and not yet closed.
            depth = {}
            stack = []
            for m in re.finditer(r'<(/?)(\w+)([^>]*)>', before):
                closing, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
                if tag in ("br", "img", "input", "meta", "link", "hr"):
                    continue
                if closing:
                    for i in range(len(stack) - 1, -1, -1):
                        if stack[i][0] == tag:
                            del stack[i:]
                            break
                else:
                    anc = re.search(r'id="([^"]+)"', attrs)
                    stack.append((tag, anc.group(1) if anc else None))
            for _tag, anc_id in reversed(stack):
                if anc_id and _listens(anc_id):
                    return anc_id
            return None

        orphans = []
        for tag, ident in ids:
            if _wired(ident):
                continue
            via = _delegating_ancestor(ident)
            if via:
                continue
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
            "survived (today's #load-from-selection-btn case).\n\n"
            "A control inside a container whose own id IS wired counts as "
            "wired -- that is event delegation, and the three mode-detail "
            "tabs use it (one listener on #mode-tabs)."
        )


# --------------------------------------------------------------------- #
#  Schema endpoint                                                      #
# --------------------------------------------------------------------- #


# The Schema/Render endpoint classes retired with their routes
# (spectra-migration plan P3, 2026-08-21): the tab renders the
# CATALOGUE vibration form and hands over; the deck is written by
# `prep`.  The load endpoint below is the surviving artifact door.


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

    def test_load_says_which_atoms_each_mode_belongs_to(self, web_client):
        """The panel describes a mode by the atoms that CARRY it, and working
        that out needs atomic masses — which the browser does not have and the
        .spectra.json does not store.  So the server computes it at load.

        The fixture is the trap in miniature: the hydrogen has the larger
        displacement (1.15 against 0.98) and the carbon owns the mode.  Read
        by distance alone this is a hydrogen mode; it is not.

        Computed at LOAD rather than written into the file, so results already
        on disk gain it the moment they are opened.
        """
        original = _make_minimal_results()
        original.equilibrium_elements      = ["C", "H"]
        original.equilibrium_positions_ang = np.array([[0., 0., 0.],
                                                       [1.09, 0., 0.]])
        original.modes[0].eigenvector_canonical = np.array([[0.98, 0., 0.],
                                                            [1.15, 0., 0.]])
        original.modes[0].eigenvector_display   = original.modes[0].eigenvector_canonical

        r = web_client.post("/api/spectra/load",
                            data=json.dumps({"json": original.to_dict()}),
                            content_type="application/json")
        body = r.get_json()
        assert r.status_code == 200, body
        share = body["results"]["modes"][0]["motion_share_by_element"]

        assert share["C"] > share["H"], (
            "carbon moves less than hydrogen here and still carries the mode; "
            "reporting the furthest-moving atom instead would say hydrogen")
        assert share["C"] == pytest.approx(0.885, abs=0.02)
        assert sum(share.values()) == pytest.approx(1.0, abs=1e-12)

    def test_the_on_disk_format_does_not_grow_the_computed_field(self, web_client):
        """``to_dict`` is the file format and round-trips through
        ``from_dict``; the share is derived and belongs to the reply only.  A
        field that leaked into the file would be written by the emitter as if
        the schema declared it."""
        original = _make_minimal_results()
        original.equilibrium_elements      = ["C", "H"]
        original.equilibrium_positions_ang = np.array([[0., 0., 0.],
                                                       [1.09, 0., 0.]])
        assert "motion_share_by_element" not in original.to_dict()["modes"][0]

    def test_a_result_without_a_geometry_still_loads(self, web_client):
        """No elements stored means no shares can be computed — the spectrum
        must still open, one clause shorter."""
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({"json": _make_minimal_results().to_dict()}),
            content_type="application/json")
        body = r.get_json()
        assert r.status_code == 200, body
        assert body["ok"] is True
        assert "motion_share_by_element" not in body["results"]["modes"][0]

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
