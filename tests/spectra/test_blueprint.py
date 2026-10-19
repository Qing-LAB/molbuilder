"""Spectra blueprint tests -- the two routes that remain.

    GET  /spectrum-calculation   -- page renders, has the tab nav
    POST /api/spectra/load       -- parse an uploaded .spectra.json into
                                    typed results; typed error shapes
                                    (404 missing / 422 wrong schema /
                                    400 malformed)
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

    def test_path_load_endpoint_works(self, web_client, tmp_path):
        """End-to-end sanity: POST /api/spectra/load with {path}
        succeeds for a real on-disk spectra.json (what the viewer's
        loadByPath sends for the dropdown's pick).  Regression check
        against a future blueprint change accidentally breaking the
        path mode of the load endpoint."""
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


# --------------------------------------------------------------------- #
#  Dead-controls test: every interactive element in the partial must    #
#  have a handler bound somewhere in loaded JS.                         #
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

    def test_the_thermal_spread_is_served_at_the_temperature_asked(
            self, web_client, tmp_path):
        """`web/spectra.md` § 4.1: the load door serves each mode's thermal
        spread at the request's `temperature_k`; the tab computes nothing.
        The contract's own table: at 298 K a 100 cm-1 mode swings 2.1x its
        zero-point, a 3000 cm-1 mode 1.00x; asked for no temperature, no
        mode carries the key.  Silent failure otherwise: a page drawing a
        thermal size the server never derived."""
        import dataclasses
        base = _make_minimal_results()
        soft = dataclasses.replace(base.modes[0], index_1based=1,
                                   frequency_cm1=100.0)
        stiff = dataclasses.replace(base.modes[0], index_1based=2,
                                    frequency_cm1=3000.0)
        results = dataclasses.replace(base, modes=[soft, stiff])
        p = tmp_path / "t.spectra.json"
        dump_spectra_json(results, p)

        def load(**extra):
            r = web_client.post("/api/spectra/load",
                                data=json.dumps({"path": str(p), **extra}),
                                content_type="application/json")
            return r.status_code, r.get_json()

        code, body = load(temperature_k=298)
        assert code == 200, body
        modes = body["results"]["modes"]
        ratios = [m["thermal_spread_amu12_ang"]
                  / m["zero_point_amplitude_amu12_ang"] for m in modes]
        assert abs(ratios[0] - 2.1) < 0.05, ratios
        assert abs(ratios[1] - 1.00) < 0.005, ratios

        code, body = load()
        assert code == 200, body
        assert all("thermal_spread_amu12_ang" not in m
                   for m in body["results"]["modes"])

        code, body = load(temperature_k="warm")
        assert code == 400 and "temperature_k" in body["error"], body

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
        that out needs atomic masses — which the browser does not have; a
        schema-7 result states the masses its modes were weighted by
        (`engines/vibration.md` § 6.2).  So the server computes it at load,
        from those.

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
        original.equilibrium_masses_amu    = np.array([12.011, 1.008])
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


# ===================================================================== #
#  The fence (web-api.md § 2.1)                                         #
# ===================================================================== #

def test_load_still_reads_a_spectra_file_inside_the_roots(
        web_client, tmp_path):
    """...and the fence does not break the ordinary case."""
    good = tmp_path / "job.spectra.json"
    dump_spectra_json(_make_minimal_results(), good)
    r = web_client.post("/api/spectra/load", json={"path": str(good)})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["ok"] is True
