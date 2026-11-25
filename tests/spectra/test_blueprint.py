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
        ({}, "asked every time"),
        ({"temperature_k": -5}, "temperature"),
        ({"temperature_k": "warm"}, "temperature"),
        ({"temperature_k": 300, "nodes": 9}, "3, 5 or 7"),
        ({"swing_quanta": 1, "temprature_k": 300},
         "did you mean 'temperature_k'"),
        ({"quanta": 1}, "no run of ours"),
        ({"swing_quanta": 2}, "no run of ours"),
        ({"mean_quanta": 0.5}, "no run of ours"),
    ], ids=["no state", "below zero", "not a number", "nine frames",
            "a misspelled key", "n quanta, no temperature",
            "a driven mode, no temperature", "mean quanta"])
    def test_a_frame_set_asked_without_its_state_or_rule_is_refused(
            self, web_client, isolated_projects_root, ask, refused):
        """`engines/vibration-frames.md` § 2, § 6: the write door asks the
        state every time -- refused in the toolkit's words (its rows,
        `test_vibration_frames.py`) and before it reads any run: no state
        named (never the record's temperature assumed), a temperature that
        is not a number of kelvin, 0 or more, another frame count than
        three, five or seven, a key the door does not take (with the one it
        is close to); a state that needs no temperature -- n quanta, a
        driven mode, mean quanta -- goes on to the run (here no run at
        all).  The silent failures: a set written at a state the person
        never stated -- a misspelled temperature left out, the cold driven
        state written -- at a rule the page does not offer, or a valid state
        refused for a temperature it does not need."""
        from molbuilder.projects import populate_project_skeleton
        tree = isolated_projects_root
        # The pair asked where a file may be written: a project's folder,
        # the project made as New project makes it.
        populate_project_skeleton(tree / "P", project_name="P")
        out = tree / "P" / "structure" / "frames.xyz"
        body = {"path": str(tree / "x.spectra.json"), "mode": 1,
                "nodes": 3, "out": str(out), **ask}
        r = web_client.post("/api/spectra/frameset", data=json.dumps(body),
                            content_type="application/json")
        said = r.get_json()
        assert r.status_code == 400 and refused in said["error"], said
        assert not out.exists()

    def test_a_pair_already_there_is_refused_before_the_run_is_read(
            self, web_client, isolated_projects_root):
        """`engines/vibration-frames.md` § 5: the pair's name is asked before
        anything is read -- a pair already at ``out`` is refused in the
        writer's words, its run never read (here no run at all).  The silent
        failure: a refusal naming the run when the name was the fault, its
        way out (another run) refused again for the name."""
        from molbuilder.projects import populate_project_skeleton
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        tree = isolated_projects_root
        populate_project_skeleton(tree / "P", project_name="P")
        out = tree / "P" / "structure" / "frames.xyz"
        # A PAIR ALREADY THERE, written by the codec as any structure is.
        StructureCodec().write(Structure.from_xyz(
            "2\nH2\nH 0 0 0\nH 0 0 0.74\n"), out)
        body = {"path": str(tree / "x.spectra.json"), "mode": 1, "nodes": 3,
                "temperature_k": 300, "out": str(out)}
        r = web_client.post("/api/spectra/frameset", data=json.dumps(body),
                            content_type="application/json")
        said = r.get_json()
        assert r.status_code == 400 and "already exist" in said["error"], \
            said

    @pytest.mark.parametrize("ask, refused", [
        ({"temperature_k": "warm"}, "temperature_k"),
        ({"frameset_state": {"temperature_k": -1}}, "temperature"),
        ({"frameset_state": {}}, "asked every time"),
        ({"frameset_state": {"temprature_k": 300}},
         "did you mean 'temperature_k'"),
    ], ids=["the animation's, not a number", "the frame set's, below zero",
            "the frame set's, no state", "a misspelled key"])
    def test_a_temperature_or_state_that_is_not_one_is_refused_by_name(
            self, web_client, ask, refused):
        """`web/spectra.md` § 4.1, § 4.3: the animation's temperature that is
        not a number of kelvin, 0 or more, and a frame set's state that names
        none, a value the table refuses (its rows,
        `test_vibration_frames.py`), or a key it does not take (with the one
        it is close to), are refused by name, before any result is read.
        The silent failure: a spread or a frame set's readout at a state
        nobody said."""
        r = web_client.post("/api/spectra/load", json={"path": "x", **ask})
        assert r.status_code == 400 and refused in r.get_json()["error"], \
            r.get_json()

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
