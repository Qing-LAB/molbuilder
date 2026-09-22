"""L2 tests for the packaged fcc_lattice.json (v2 added 2026-06-18,
v3 2026-08-30).

Pins:
  * JSON declares format v2 or v3.
  * Every metal carries a_experimental and a_pbe (numeric), and NOT
    a_pbe_siesta_psml -- v3 dropped it (see below).
  * PBE values land within +- 3% of experimental for each metal
    (catches a typo that would silently re-introduce ~6+% mismatch).
  * Ni a_pbe is spin-polarized (>= 3.50 A); the non-spin-polarized
    value would be ~3.43 A and is wrong physics for ferromagnetic Ni.
  * Loader returns the v1-shape ({sym: experimental_float}) for
    back-compat callers; full loader returns the v2 per-XC dict.
  * /api/modify/meta exposes lattice_table so the UI can render
    the reference radios without duplicating the numbers in JS.
  * A v2 file still loads, so a user's overriding data dir keeps working.

WHY v3 DROPPED A COLUMN.  ``a_pbe_siesta_psml`` was null for every metal
and nothing in the codebase could write it: its only homes were this
packaged file and a machine-wide ``MOLBUILDER_DATA_DIR`` override, so the
"Your bulk run" control it fed greyed itself out -- correctly -- from the
day it shipped.  A lattice constant measured in the user's own
SIESTA+PSML setup belongs to ONE optimization run, not to a table every
project shares, so it is read from that run's result instead
(``POST /api/modify/lattice-from-run``, tests/test_lattice_from_run.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    _load_fcc_lattice,
    load_fcc_lattice_full,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = REPO_ROOT / "molbuilder" / "data" / "fcc_lattice.json"


def _data():
    return json.loads(JSON_PATH.read_text())


def test_format_is_v2():
    """JSON declares v2.  Catches accidental downgrade to v1."""
    fmt = _data().get("_format", "")
    assert "v2" in fmt or "v3" in fmt, (
        f"fcc_lattice.json must be format v2 or v3; got {fmt!r}.  "
        f"v1 'a'-only schema is no longer supported."
    )


def test_every_metal_has_the_two_literature_references():
    """All 6 supported metals carry a_experimental and a_pbe."""
    metals = _data()["metals"]
    for sym in SUPPORTED_FCC_ELEMENTS:
        assert sym in metals, f"metal {sym!r} missing from fcc_lattice.json"
        entry = metals[sym]
        assert isinstance(entry.get("a_experimental"), (int, float)), (
            f"{sym}.a_experimental must be numeric")
        assert isinstance(entry.get("a_pbe"), (int, float)), (
            f"{sym}.a_pbe must be numeric")
        # AND NOT the column v3 dropped.  Asserted as an absence, because
        # re-adding it would quietly restore a control that can never be
        # reachable: nothing in the codebase writes this file, so the value
        # would be null again and the radio would grey itself out again.
        #
        # THE ONLY PLACE THIS CAN BE SEEN.  It reads the raw JSON.  Two other
        # tests carried the same line against `load_fcc_lattice_full` output,
        # where it could not fail: the loader builds a fresh dict of four
        # fixed keys, so a column re-added to the file never reaches it.
        # Removed there 2026-09-09; this is the guard.
        assert "a_pbe_siesta_psml" not in entry, (
            f"{sym} carries a_pbe_siesta_psml. A per-run measurement does not "
            f"belong in a table every project shares -- it comes from "
            f"/api/modify/lattice-from-run.")


def test_pbe_within_3pct_of_experimental():
    """Sanity gate: PBE values are within +-3% of experimental.

    Catches a typo (decimal-place shift, swapped elements, etc.) that
    would silently re-introduce a 5-10% lattice mismatch in the
    transport workflow.  PBE actually runs about +0.5 to +2% larger
    than experimental for these metals; -3% to +3% is a roomy gate.
    """
    metals = _data()["metals"]
    for sym, entry in metals.items():
        a_exp = entry["a_experimental"]
        a_pbe = entry["a_pbe"]
        ratio = (a_pbe - a_exp) / a_exp
        assert abs(ratio) < 0.03, (
            f"{sym}: PBE ({a_pbe}) is {ratio*100:+.1f}% off experimental "
            f"({a_exp}); likely a typo")


def test_ni_pbe_is_spin_treatment():
    """Ni a_pbe must reflect the ferromagnetic ground state (>= 3.50 A).

    The non-spin-polarized PBE value is ~3.43 A.  Using it for a
    transport calc on Ni electrodes silently gives wrong physics
    (the lead is in the wrong magnetic state).
    """
    a_pbe_ni = _data()["metals"]["Ni"]["a_pbe"]
    assert a_pbe_ni >= 3.50, (
        f"Ni a_pbe = {a_pbe_ni} is suspiciously small; the "
        f"non-spin-polarized value is ~3.43 A.  Use the spin-"
        f"polarized number (~3.52 A) per Janthon 2013 / Csonka 2009")


def test_sources_block_present():
    """The _sources block exists and carries citations for each XC.

    A missing citation block means the UI tooltip would render
    "(no source)" — guard against silent drift away from the canonical
    reference.
    """
    sources = _data().get("_sources", {})
    for key in ("experimental", "pbe"):
        assert key in sources, f"_sources missing entry for {key!r}"
        assert sources[key].get("citation") or sources[key].get("notes"), (
            f"_sources.{key} carries neither citation nor notes")
    # The third source block went with the column it described (v3): it cited
    # "user-measured via a bulk-cell relax in their specific SIESTA + PSML
    # setup", which is a per-run fact and never was a shared reference.
    assert "pbe_siesta_psml" not in sources


def test_a_v2_file_still_loads(tmp_path, monkeypatch):
    """A user's overriding data dir must not stop working because a column
    they never filled went away.

    Pinned rather than assumed: the loader's format check is a substring
    test, so "accepts v2 as well as v3" is one easily-broken character.
    """
    import json as _json
    from molbuilder import modify as _mod

    doc = _json.loads(JSON_PATH.read_text())
    doc["_format"] = "molbuilder.data.fcc_lattice v2"
    for entry in doc["metals"].values():
        entry["a_pbe_siesta_psml"] = None          # as a v2 file carries it
    (tmp_path / "fcc_lattice.json").write_text(_json.dumps(doc))
    monkeypatch.setattr(_mod, "_data_dir_candidates", lambda: (tmp_path,))

    table = _mod.load_fcc_lattice_full()
    assert abs(table["Au"]["a_experimental"] - 4.0782) < 1e-9
    assert "a_pbe_siesta_psml" not in table["Au"], (
        "a v2 file's dropped column must not come back through the loader")


def test_a_v1_file_is_still_refused(tmp_path, monkeypatch):
    """The 'a'-only schema carries one number per metal with no XC attached,
    which is the ambiguity v2 existed to end.  Loosening the version check to
    admit v3 must not have loosened it to admit v1."""
    import json as _json
    import pytest as _pytest
    from molbuilder import modify as _mod

    (tmp_path / "fcc_lattice.json").write_text(_json.dumps({
        "_format": "molbuilder.data.fcc_lattice v1",
        "metals": {"Au": {"a": 4.0782}},
    }))
    monkeypatch.setattr(_mod, "_data_dir_candidates", lambda: (tmp_path,))
    with _pytest.raises(RuntimeError, match="neither v2 nor v3"):
        _mod.load_fcc_lattice_full()


def test_v1_loader_returns_experimental_floats():
    """Back-compat: ``_load_fcc_lattice`` returns ``{sym: a_experimental}``.

    This is the v1 shape; callers that don't know about per-XC values
    keep working without changes.
    """
    table = _load_fcc_lattice()
    metals_json = _data()["metals"]
    for sym in SUPPORTED_FCC_ELEMENTS:
        assert sym in table
        assert isinstance(table[sym], float)
        assert table[sym] == pytest.approx(metals_json[sym]["a_experimental"])


def test_v2_loader_returns_full_dict():
    """``load_fcc_lattice_full`` returns the per-XC dict per metal."""
    full = load_fcc_lattice_full()
    for sym in SUPPORTED_FCC_ELEMENTS:
        assert sym in full
        e = full[sym]
        assert isinstance(e["a_experimental"], float)
        assert isinstance(e["a_pbe"], float)
        assert isinstance(e["name"], str)
        assert e["system"] == "fcc"


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


class TestAnOverrideIsHonouredOrRefused:
    """PINS: a data file the user PUT there is a statement of intent.

    `data/README.md` tells people to copy the JSON to
    ``$MOLBUILDER_DATA_DIR`` and edit it — for a strained lattice, a
    low-temperature value, a constant measured on their own bulk run. So the
    only two honest answers to such a file are *use it* and *refuse by name*.

    PREVENTS: the third answer, measured 2026-09-22. An override with one
    trailing comma returned Au = 4.0782 to a person who had typed 4.20 — no
    warning, no log — and the Modify panel then displayed 4.0782 under a radio
    labelled "Experimental". Consistent-looking, and not the number they set.
    Two of the four malformations already raised (a bad ``_format``, a
    malformed metal entry); a JSON syntax error and a missing ``metals`` key
    fell through to the packaged table instead.
    """

    @staticmethod
    def _override(tmp_path, text: str) -> Path:
        (tmp_path / "fcc_lattice.json").write_text(text)
        return tmp_path

    def _load_with(self, monkeypatch, data_dir):
        import molbuilder.modify as mod
        monkeypatch.setenv("MOLBUILDER_DATA_DIR", str(data_dir))
        monkeypatch.setattr(mod, "_FCC_LATTICE_A_CACHE", None)
        return mod.load_fcc_lattice_full()

    def test_a_valid_override_is_used(self, monkeypatch, tmp_path):
        good = json.loads(
            (Path(__file__).resolve().parents[1]
             / "molbuilder/data/fcc_lattice.json").read_text())
        good["metals"]["Au"]["a_experimental"] = 4.20
        d = self._override(tmp_path, json.dumps(good, indent=2))
        assert self._load_with(monkeypatch, d)["Au"]["a_experimental"] == 4.20

    def test_a_syntax_error_refuses_instead_of_falling_back(
            self, monkeypatch, tmp_path):
        good = json.loads(
            (Path(__file__).resolve().parents[1]
             / "molbuilder/data/fcc_lattice.json").read_text())
        good["metals"]["Au"]["a_experimental"] = 4.20
        broken = json.dumps(good, indent=2).replace(
            '"a_pbe": 4.158', '"a_pbe": 4.158,', 1)     # one trailing comma
        d = self._override(tmp_path, broken)
        with pytest.raises(RuntimeError, match="failed to read"):
            self._load_with(monkeypatch, d)

    def test_a_missing_metals_key_refuses_too(self, monkeypatch, tmp_path):
        d = self._override(tmp_path, json.dumps(
            {"_format": "molbuilder.data.fcc_lattice v3"}, indent=2))
        with pytest.raises(RuntimeError, match="missing required 'metals'"):
            self._load_with(monkeypatch, d)

    def test_an_ABSENT_override_still_falls_back(self, monkeypatch, tmp_path):
        """Absence is not malformation. A data dir that simply does not hold
        this file is a legitimate state, and the packaged table is the
        answer — that path must not become a refusal."""
        table = self._load_with(monkeypatch, tmp_path / "not-there")
        assert table["Au"]["a_experimental"] == pytest.approx(4.0782)


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
        assert pbe["a"] == pytest.approx(4.158)
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

    def test_a_metal_the_builder_cannot_make_is_refused(self, monkeypatch,
                                                        tmp_path, web_client):
        """The element list is CLOSED; the lattice table is user-overridable.
        Two different questions, and only the plane was being asked.

        `data/README.md` invites people to copy the JSON to
        `$MOLBUILDER_DATA_DIR` and edit it. Adding an `Fe` entry with
        `"system": "bcc"` made this door answer `d(110) = 2.0269` from the
        bcc rule — genuinely correct crystallography — for a metal
        `/api/modify/slab` refuses outright. A spacing for a slab that
        cannot be built is a plausible number with nothing to use it on.
        """
        import molbuilder.modify as mod
        good = json.loads(
            (Path(__file__).resolve().parents[1]
             / "molbuilder/data/fcc_lattice.json").read_text())
        good["metals"]["Fe"] = {"name": "Iron", "system": "bcc",
                                "a_experimental": 2.8665, "a_pbe": 2.83}
        (tmp_path / "fcc_lattice.json").write_text(json.dumps(good, indent=2))
        monkeypatch.setenv("MOLBUILDER_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(mod, "_FCC_LATTICE_A_CACHE", None)

        # The table carries it...
        assert "Fe" in mod.load_fcc_lattice_full()
        # ...and the door still refuses it, the way the builder does.
        r = web_client.get("/api/modify/spacings?element=Fe&plane=110"
                           "&reference=experimental")
        assert r.status_code == 400, r.get_json()
        assert "Fe" in r.get_json()["error"]

    def test_a_surface_the_builder_cannot_make_is_refused(self, web_client):
        """`interplanar_spacing` answers (999) correctly — it is general
        crystallography. This door serves a panel that offers three planes,
        and `/api/modify/slab` refuses the rest, so quoting a spacing for one
        would be a number with nothing to use it on."""
        r = web_client.get("/api/modify/spacings?element=Au&plane=999"
                           "&reference=experimental")
        assert r.status_code == 400
        assert "999" in r.get_json()["error"]
