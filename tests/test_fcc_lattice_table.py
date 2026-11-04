"""The packaged fcc lattice table (`molbuilder/data/fcc_lattice.json`,
`molbuilder/data/README.md`): its two literature references per metal, as the
loader every consumer reads answers them, and the spacing door built on it.

A lattice constant measured in the user's own SIESTA+PSML setup belongs to
ONE optimization run, not to a table every project shares, so it is read from
that run's result instead (``POST /api/modify/lattice-from-run``,
tests/test_lattice_from_run.py).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    load_fcc_lattice_full,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ROWS = REPO_ROOT / "tests" / "data" / "fcc_lattice.toml"


def test_each_metal_answers_its_two_literature_references():
    """`data/README.md`'s table, through the loader every consumer reads: the
    six metals the slab builder makes, no more and no fewer, each with
    Wyckoff's room-temperature lattice constant and Zhang et al.'s PBE one
    (rows of `tests/data/fcc_lattice.toml`).

    THE SILENT FAILURE: a slab built on a lattice constant a digit off -- a
    lead and a device mismatched by the percent or two a transport
    calculation exists to avoid, with nothing to say so; or a metal offered
    with no reference behind it.

    MUTATIONS THIS MUST FAIL AGAINST: a PBE value a digit off; an
    experimental value a digit off; a metal gone from the table."""
    rows = tomllib.loads(ROWS.read_text())["metal"]
    table = load_fcc_lattice_full()
    assert sorted(table) == sorted(r["symbol"] for r in rows), sorted(table)
    for r in rows:
        got = table[r["symbol"]]
        assert (got["a_experimental"], got["a_pbe"]) == pytest.approx(
            (r["a_experimental"], r["a_pbe"]), abs=1e-9), (r, got)


def test_meta_endpoint_exposes_lattice_table(web_client):
    """The /api/modify/meta endpoint returns lattice_table for the UI."""
    r = web_client.get("/api/modify/meta")
    assert r.status_code == 200
    j = r.get_json()
    assert j.get("ok") is True
    assert "lattice_table" in j
    table = j["lattice_table"]
    for sym in SUPPORTED_FCC_ELEMENTS:
        assert sym in table
        e = table[sym]
        assert "a_experimental" in e
        assert "a_pbe" in e


def test_meta_endpoint_lattice_error_is_none_on_happy_path(web_client):
    """When the JSON loads cleanly, lattice_error is null in the response."""
    r = web_client.get("/api/modify/meta")
    j = r.get_json()
    assert j.get("lattice_error") in (None, "")


class TestTheSpacingsDoorAnswersTheSurface:
    """PINS: `/api/modify/spacings` — the crystallography is the backend's.

    The Slab panel computed the layer spacing itself, as three literals in
    JavaScript, and was missing `d(110)` entirely; both spacings it did show
    were printed whatever surface was selected. So a person building fcc(110)
    got two numbers, neither of them the one the Cell page asks them to type.
    """

    def test_it_answers_the_plane_that_was_asked_for(self, web_client):
        got = {}
        for plane in ("100", "110", "111"):
            r = web_client.get(f"/api/modify/spacings?element=Au&plane={plane}"
                               f"&reference=experimental")
            assert r.status_code == 200, r.get_json()
            j = r.get_json()
            assert j["plane"] == plane
            got[plane] = j["d_interlayer"]
        # The row the browser never had, and the two it printed regardless.
        assert got["111"] == pytest.approx(2.354550, abs=1e-6)
        assert got["100"] == pytest.approx(2.039100, abs=1e-6)
        assert got["110"] == pytest.approx(1.441861, abs=1e-6)

    def test_the_reference_is_required_and_changes_the_answer(self, web_client):
        """*(user, 2026-09-22)* — no implicit default. Experimental and PBE
        differ by ~2% for gold, which is the whole reason it cannot be
        guessed on the caller's behalf."""
        bare = web_client.get("/api/modify/spacings?element=Au&plane=111")
        assert bare.status_code == 400
        assert "reference" in bare.get_json()["error"]

        exp = web_client.get("/api/modify/spacings?element=Au&plane=111"
                             "&reference=experimental").get_json()
        pbe = web_client.get("/api/modify/spacings?element=Au&plane=111"
                             "&reference=pbe").get_json()
        num = web_client.get("/api/modify/spacings?element=Au&plane=111"
                             "&reference=4.20").get_json()
        assert exp["a"] == pytest.approx(4.0782)
        assert pbe["a"] == pytest.approx(4.157)
        assert num["a"] == pytest.approx(4.20)
        assert exp["d_interlayer"] < pbe["d_interlayer"] < num["d_interlayer"]

    def test_an_element_the_table_does_not_carry_is_named_not_crashed(
            self, web_client):
        """The table IS the list of what we know, so an unknown name is
        answered here rather than as a `KeyError` two layers down."""
        r = web_client.get("/api/modify/spacings?element=Xx&plane=111"
                           "&reference=pbe")
        assert r.status_code == 400
        assert "Xx" in r.get_json()["error"]

    def test_a_surface_the_builder_cannot_make_is_refused(self, web_client):
        """`interplanar_spacing` answers (999) correctly — it is general
        crystallography. This door serves a panel that offers three planes,
        and `/api/modify/slab` refuses the rest, so quoting a spacing for one
        would be a number with nothing to use it on."""
        r = web_client.get("/api/modify/spacings?element=Au&plane=999"
                           "&reference=experimental")
        assert r.status_code == 400
        assert "999" in r.get_json()["error"]
