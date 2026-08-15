"""L2 tests for the fcc_lattice.json v2 schema (added 2026-06-18).

Pins:
  * JSON declares format v2.
  * Every metal carries a_experimental, a_pbe (numeric) and
    a_pbe_siesta_psml (numeric-or-null).
  * PBE values land within +- 3% of experimental for each metal
    (catches a typo that would silently re-introduce ~6+% mismatch).
  * Ni a_pbe is spin-polarized (>= 3.50 A); the non-spin-polarized
    value would be ~3.43 A and is wrong physics for ferromagnetic Ni.
  * Loader returns the v1-shape ({sym: experimental_float}) for
    back-compat callers; full loader returns the v2 per-XC dict.
  * /api/modify/meta exposes lattice_table so the UI can render
    the 3-way radio without duplicating the numbers in JS.
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
    assert "v2" in fmt, (
        f"fcc_lattice.json must be format v2; got {fmt!r}.  "
        f"v1 'a'-only schema is no longer supported."
    )


def test_every_metal_has_v2_fields():
    """All 6 supported metals carry a_experimental, a_pbe, a_pbe_siesta_psml."""
    metals = _data()["metals"]
    for sym in SUPPORTED_FCC_ELEMENTS:
        assert sym in metals, f"metal {sym!r} missing from fcc_lattice.json"
        entry = metals[sym]
        assert isinstance(entry.get("a_experimental"), (int, float)), (
            f"{sym}.a_experimental must be numeric")
        assert isinstance(entry.get("a_pbe"), (int, float)), (
            f"{sym}.a_pbe must be numeric")
        # a_pbe_siesta_psml is either a number OR explicit null.
        assert ("a_pbe_siesta_psml" in entry), (
            f"{sym} missing a_pbe_siesta_psml key (use null when unmeasured)")
        psml = entry["a_pbe_siesta_psml"]
        assert psml is None or isinstance(psml, (int, float)), (
            f"{sym}.a_pbe_siesta_psml must be a number or null; got {psml!r}")


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
    for key in ("experimental", "pbe", "pbe_siesta_psml"):
        assert key in sources, f"_sources missing entry for {key!r}"
        assert sources[key].get("citation") or sources[key].get("notes"), (
            f"_sources.{key} carries neither citation nor notes")


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
        assert e["a_pbe_siesta_psml"] is None or isinstance(
            e["a_pbe_siesta_psml"], float)
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
        assert "a_pbe_siesta_psml" in e


def test_meta_endpoint_lattice_error_is_none_on_happy_path(web_client):
    """When the JSON loads cleanly, lattice_error is null in the response."""
    r = web_client.get("/api/modify/meta")
    j = r.get_json()
    assert j.get("lattice_error") in (None, "")
