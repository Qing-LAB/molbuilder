"""Spectra blueprint tests: the page, and what `POST /api/spectra/load` and
`POST /api/spectra/frameset` refuse before they read any result.

What the load door serves of a result is asked of a result a run wrote, in
`tests/test_spectrum_results_e2e.py`.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _allowed_root(tmp_path, monkeypatch):
    """Say where this test's tree IS.

    Since web-api.md § 2.1 every route fences a browser-supplied path to the
    allowed roots BEFORE calling the module behind it, so a test driving those
    routes at its own ``tmp_path`` has to declare that path the way a
    deployment declares its projects root.  Without this the routes correctly
    refuse — which is the point of the fence, not a test-harness quirk.
    """
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(runtime_config={}, conda_binary=None,
                                    conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)


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
        # The Inspect-structure card is the sole entry point for the
        # structure: an EMPTY ``#spectra-molview-host`` that molview.mount
        # builds the read-only fused card into.
        assert 'id="spectra-molview-host"'      in body
        assert 'id="load-from-sidebar-btn"'     in body
        assert 'id="send-to-task-setup"'        in body
        # The tab hands over, it renders no script.
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
            "watch-status",            # the run's progress line
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
            "workspace-indicator",     # workspace readout
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
        canonical tabs -- regression check against a future header
        refactor dropping the entry."""
        # /molbuilder is one of the canonical landing pages; reuse
        # it to fetch a header-rendered body.
        r = web_client.get("/molbuilder")
        body = r.data.decode()
        assert 'href="/spectrum-calculation"' in body

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
            assert r.status_code == 200, r.status_code
            assert "application/javascript" in r.content_type
            # First few bytes of plotly.min.js are an IIFE / use-strict
            # banner.  Just confirm we got a sizeable JS payload.
            assert len(r.data) > 100_000, len(r.data)
        except ImportError:
            assert r.status_code == 404

    def test_no_cdn_fallback_in_spectrum_page(self, web_client):
        """The 3Dmol script tag on /spectrum-calculation must point
        at the vendored copy (``/static/vendor/3Dmol-min.js``) —
        never a CDN.  Pin so
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


class TestLoadEndpoint:

    @pytest.mark.parametrize("ask, refused", [
        ({}, "temperature"),
        ({"temperature_k": -5}, "temperature"),
        ({"temperature_k": "warm"}, "temperature"),
        ({"temperature_k": 300, "nodes": 9}, "3, 5 or 7"),
    ], ids=["none", "below zero", "not a number", "nine frames"])
    def test_a_frame_set_asked_without_its_temperature_or_rule_is_refused(
            self, web_client, tmp_path, ask, refused):
        """`engines/vibration.md` § 5.10 ③: the write door refuses, in the
        generator's words and before it reads any run, a temperature stated
        nowhere -- never defaulted -- one that is not a number of kelvin, 0
        or more, and another rule than three, five or seven frames.  The
        silent failure: a set written at a temperature the person never
        stated, or at a rule the page does not offer."""
        body = {"path": str(tmp_path / "x.spectra.json"), "mode": 1,
                "nodes": 3,
                "out": str(tmp_path / "frames.xyz"), **ask}
        r = web_client.post("/api/spectra/frameset", data=json.dumps(body),
                            content_type="application/json")
        said = r.get_json()
        assert r.status_code == 400 and refused in said["error"], said
        assert not (tmp_path / "frames.xyz").exists()

    @pytest.mark.parametrize("ask", [{"temperature_k": "warm"},
                                     {"frameset_temperature_k": -1}],
                             ids=["not a number", "below zero"])
    def test_a_temperature_that_is_not_one_is_refused_by_name(self, web_client,
                                                              ask):
        """`web/spectra.md` § 4.1, § 4.3: a temperature that is not a number of
        kelvin, 0 or more, is refused by name, before any result is read.  The
        silent failure: a spread or a frame set at a temperature nobody said."""
        r = web_client.post("/api/spectra/load", json={"path": "x", **ask})
        assert r.status_code == 400 and "temperature_k" in r.get_json()["error"]

    def test_no_input_400(self, web_client):
        r = web_client.post(
            "/api/spectra/load",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "path" in body["error"].lower(), body

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
        # ...naming it: the poller watches one path and the person picked
        # another, and the sentence is what there is to compare.
        assert str(bad) in body["error"], body
