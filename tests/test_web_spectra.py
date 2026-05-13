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
        eigenvector_free      = np.array([[0.7, 0., 0.],
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
        fixed_atom_idxs            = [],
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
        r = web_client.get("/spectra")
        assert r.status_code == 200
        body = r.data.decode()
        # Tab nav present + Spectra tab marked active.
        assert "Spectra" in body
        assert "app-tabs" in body
        # Form container + key controls present (rendered into by JS).
        assert 'id="spectra-form-container"' in body
        assert 'id="xyz-text"'                in body
        assert 'id="generate-btn"'            in body
        assert 'id="load-results-btn"'        in body
        # Methods-preview modal present (dialog element + handles).
        assert 'id="methods-modal"'           in body
        # Static assets pinned in the template.
        assert 'spectra/style.css'            in body
        assert 'spectra/viewer.js'            in body
        # Shared form-schema helper loaded BEFORE the per-page viewer.
        assert body.index("lib/form-schema.js") \
               < body.index("spectra/viewer.js")

    def test_app_header_includes_spectra_tab(self, web_client):
        """The shared header now lists Spectra alongside Build / Modify /
        Watch -- regression check against a future header refactor
        dropping the entry."""
        r = web_client.get("/")
        body = r.data.decode()
        assert 'href="/spectra"' in body

    def test_viewer_js_served(self, web_client):
        """The Spectra-tab JS should be reachable as a static asset
        and contain the four endpoint URLs it talks to."""
        r = web_client.get("/static/spectra/viewer.js")
        assert r.status_code == 200
        js = r.data.decode()
        assert "/api/build/schema/spectra" in js
        assert "/api/spectra/render"        in js
        assert "/api/spectra/load"          in js
        # Selector / compatibility logic present (locks unused
        # ES value fields when the selector changes).
        assert "applyCompatibility" in js
        # Methods-preview modal handler.
        assert "openMethodsModal"   in js

    def test_style_css_served(self, web_client):
        """The CSS imports the shared tokens so theming stays in
        lock-step with Build / Modify / Watch."""
        r = web_client.get("/static/spectra/style.css")
        assert r.status_code == 200
        css = r.data.decode()
        assert "tokens.css" in css
        assert ".spectra-grid" in css

    def test_plotly_loaded_from_cdn(self, web_client):
        """The template pulls Plotly from a pinned cdnjs build so the
        spectrum chart works without a local copy."""
        body = web_client.get("/spectra").data.decode()
        assert "cdnjs.cloudflare.com/ajax/libs/plotly.js" in body
        # And a div for the chart is present.
        assert 'id="spectrum-chart"' in body

    def test_viewer_js_has_chart_renderer(self, web_client):
        """The chart-renderer function is in the JS module and
        builds three traces (real / imaginary / no-Raman)."""
        js = web_client.get("/static/spectra/viewer.js").data.decode()
        assert "renderSpectrumChart" in js
        # Three trace buckets.
        assert '"Real"'      in js
        assert '"Imaginary"' in js
        assert '"No Raman"'  in js
        # Plotly entry point.
        assert "Plotly.react" in js

    def test_page_has_es_panel_and_table_controls(self, web_client):
        """The Spectra page exposes the mode-table interaction
        controls (filter + CSV) and the ES-panel scaffolding the
        JS targets."""
        body = web_client.get("/spectra").data.decode()
        # Table-control row.
        assert 'id="modes-filter"'        in body
        assert 'id="modes-csv-btn"'       in body
        assert 'id="modes-filter-count"'  in body
        # Sortable headers carry the data-col attribute the JS
        # reads to decide what to sort by.
        assert 'data-col="frequency_cm1"'         in body
        assert 'data-col="raman_activity_a4_amu"' in body
        # ES-panel scaffolding.
        assert 'id="es-panel"'         in body
        assert 'id="es-bar-diagram"'   in body
        assert 'id="es-summary"'       in body
        # Caption for screen readers.
        assert "Vibrational modes table"          in body

    def test_viewer_js_has_selection_sync_and_es_renderer(self, web_client):
        """The JS module exposes selectMode (click-row / click-stick
        synchronisation) and renderESPanel (per-mode MO diagram).
        Pins the contract the page markup depends on."""
        js = web_client.get("/static/spectra/viewer.js").data.decode()
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
            assert sym in js, f"missing {sym!r} in spectra/viewer.js"


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
        r = self._render(web_client, xyz=_WATER_XYZ, params={})
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
        r = self._render(web_client, xyz=_WATER_XYZ,
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
        r = self._render(web_client, xyz=_WATER_XYZ, params={
            "compute_raman":     True,
            "es_mode_selection": "explicit",
            "es_explicit_indices": "1,2,3",
        })
        body = r.get_json()
        compile(body["script"], "<wire-render>", "exec")

    def test_missing_xyz_400(self, web_client):
        r = self._render(web_client, params={})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False
        assert "xyz" in r.get_json()["error"].lower()

    def test_unparseable_xyz_400(self, web_client):
        r = self._render(web_client, xyz="not an xyz", params={})
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "xyz" in body["error"].lower()

    def test_unsupported_method_blocks_render(self, web_client):
        """Preflight catches an unknown method -- response is 400
        with the error issue surfaced."""
        r = self._render(web_client, xyz=_WATER_XYZ, params={
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
        r = self._render(web_client, xyz=_WATER_XYZ, params={
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
        r = self._render(web_client, xyz=_WATER_XYZ, params={
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
        r = self._render(web_client, xyz=_WATER_XYZ,
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
        r = self._render(web_client, xyz=_WATER_XYZ,
                         params={"es_mode_selection": "top_n",
                                 "es_top_n":          3},
                         prior_path=str(prior_path))
        # Top-n now passes the soft-dep check; render goes through.
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True


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
