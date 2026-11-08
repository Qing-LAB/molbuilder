"""Spectra blueprint tests: the page, and what `POST /api/spectra/load` and
`POST /api/spectra/frameset` refuse before they read any result.

What the load door serves of a result is asked of a result a run wrote, in
`tests/test_spectrum_results_e2e.py`.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------- #
#  Page route                                                           #
# --------------------------------------------------------------------- #


class TestSpectraPage:

    def test_vendor_plotly_route_serves_js(self, web_client):
        """The /vendor/plotly.min.js route serves the JS bundle the
        installed plotly package ships -- plotly is one of molbuilder's
        dependencies (`pyproject.toml`), so the browser needs no CDN."""
        r = web_client.get("/vendor/plotly.min.js")
        assert r.status_code == 200, r.status_code
        assert "application/javascript" in r.content_type
        # First few bytes of plotly.min.js are an IIFE / use-strict
        # banner.  Just confirm we got a sizeable JS payload.
        assert len(r.data) > 100_000, len(r.data)

    def test_the_spectrum_page_names_the_vendored_3Dmol(self, web_client):
        """The Inspect-structure card draws with the vendored 3Dmol
        (``/static/vendor/3Dmol-min.js``), so the page must name it.  That
        no page names an asset off molbuilder's own origin -- a CDN -- is
        `test_wheel_ships_the_front_end.py`'s, over every page served."""
        r = web_client.get("/spectrum-calculation")
        assert r.status_code == 200
        body = r.data.decode()
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
            self, web_client, isolated_projects_root, ask, refused):
        """`engines/vibration.md` § 5.10 ③: the write door refuses, in the
        generator's words and before it reads any run, a temperature stated
        nowhere -- never defaulted -- one that is not a number of kelvin, 0
        or more, and another rule than three, five or seven frames.  The
        silent failure: a set written at a temperature the person never
        stated, or at a rule the page does not offer."""
        tree = isolated_projects_root
        body = {"path": str(tree / "x.spectra.json"), "mode": 1,
                "nodes": 3,
                "out": str(tree / "frames.xyz"), **ask}
        r = web_client.post("/api/spectra/frameset", data=json.dumps(body),
                            content_type="application/json")
        said = r.get_json()
        assert r.status_code == 400 and refused in said["error"], said
        assert not (tree / "frames.xyz").exists()

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

    def test_missing_file_path_404(self, web_client,
                                   isolated_projects_root):
        bad = isolated_projects_root / "does_not_exist.json"
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
