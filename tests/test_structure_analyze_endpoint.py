"""Endpoint-shape tests for ``/api/structure/analyze`` post Phase 1c.

The endpoint is now a thin (~25 LoC) iterator over the adapter
registry; the chemistry analysis happens in
``molbuilder.chemistry.analyze_structure``, the per-engine
translation in each ``molbuilder.<engine>.auto_defaults`` module.

These tests pin:

* The response shape documented in
  ``docs/protocols/web-api.md`` § 10 (every top-level key + each
  ``suggested.<engine>`` sub-shape).
* The new-engine on-ramp — a freshly-registered adapter appears
  in ``suggested`` without any endpoint code change.
* The engine-agnostic rationale (no PySCF / SIESTA keyword in
  the analyzer's rationale text).
* Error paths — bad inputs surface as HTTP 400 with the canonical
  ``{ok: false, error: ...}`` envelope, NOT 500 with a stack trace.

Existing flavour-detection tests in ``test_pseudos.py`` cover the
SIESTA / PySCF adapter outputs against real fixtures; this file
covers the endpoint contract specifically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pytest


pytest.importorskip("flask")


@pytest.fixture
def web():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _post_analyze(web, body):
    r = web.post("/api/structure/analyze", json=body)
    return r, r.get_json()


# --------------------------------------------------------------------- #
#  Top-level envelope                                                   #
# --------------------------------------------------------------------- #


def test_response_shape_carries_every_documented_key(web):
    """Every key documented in web-api.md § 10 must appear in the
    response on the happy path.  Pin against a refactor that
    silently drops a field."""
    xyz = "1\nFe\nFe 0 0 0\n"
    r, body = _post_analyze(web, {"structure_text": xyz})
    assert r.status_code == 200
    assert body["ok"] is True
    for key in ("n_atoms", "elements", "n_electrons_neutral",
                 "metals", "metal_hints", "suggested", "warnings"):
        assert key in body, f"response missing documented key {key!r}"


def test_response_suggested_includes_both_built_in_engines(web):
    """The built-in adapter registry covers SIESTA and PySCF.  Pin
    that both appear in the response."""
    xyz = "1\nFe\nFe 0 0 0\n"
    _, body = _post_analyze(web, {"structure_text": xyz})
    sug = body["suggested"]
    assert set(sug.keys()) >= {"siesta", "pyscf"}, (
        f"suggested missing built-in engines: {set(sug.keys())}"
    )


def test_suggested_siesta_shape(web):
    """SIESTA adapter's dataclass field names appear verbatim in
    ``suggested.siesta``.  Pin so a SiestaSuggestedParams rename
    breaks this test before it breaks the UI form-fill path."""
    xyz = "1\nFe\nFe 0 0 0\n"
    _, body = _post_analyze(web, {"structure_text": xyz})
    si = body["suggested"]["siesta"]
    assert set(si.keys()) == {
        "net_charge", "spin_polarized", "spin_total", "rationale"
    }
    assert isinstance(si["spin_polarized"], bool)
    assert isinstance(si["spin_total"], float)


def test_suggested_pyscf_shape(web):
    """Same for the PySCF adapter."""
    xyz = "1\nFe\nFe 0 0 0\n"
    _, body = _post_analyze(web, {"structure_text": xyz})
    py = body["suggested"]["pyscf"]
    assert set(py.keys()) == {"charge", "spin", "method", "rationale"}
    assert isinstance(py["spin"], int)
    assert py["method"] in {"UKS", "RKS"}


def test_metal_hints_are_dicts_not_dataclasses(web):
    """The endpoint serialises via asdict at the HTTP boundary.
    The wire shape MUST be plain dicts so the JS consumer doesn't
    need to know about Python dataclass internals."""
    xyz = "1\nFe\nFe 0 0 0\n"
    _, body = _post_analyze(web, {"structure_text": xyz})
    hint = body["metal_hints"][0]
    assert isinstance(hint, dict)
    assert hint["element"] == "Fe"
    assert isinstance(hint["common_spins"], list)
    assert isinstance(hint["common_spins"][0], dict)


# --------------------------------------------------------------------- #
#  Engine-agnostic rationale                                            #
# --------------------------------------------------------------------- #


def test_analyzer_rationale_does_not_leak_engine_strings(web):
    """The analyzer's rationale is engine-agnostic by design
    (scientific-validation.md § 3.1).  Engine-specific keywords
    ("UKS", "RKS", "SpinPolarized") belong in the per-engine
    ``suggested.<engine>`` block, not in the analyzer's rationale.

    Pin so a future refactor that adds engine names to the
    rationale (re-fragmenting cross-engine consistency) surfaces.
    """
    xyz = "1\nFe\nFe 0 0 0\n"
    _, body = _post_analyze(web, {"structure_text": xyz})
    # The analyzer's rationale is echoed into each adapter's
    # ``rationale`` field.  Either adapter's value would do here;
    # use PySCF.
    rationale = body["suggested"]["pyscf"]["rationale"]
    for engine_keyword in ("UKS", "RKS", "RHF", "UHF",
                            "SpinPolarized", "Spin.Total"):
        assert engine_keyword not in rationale, (
            f"engine-specific keyword {engine_keyword!r} leaked into "
            f"the engine-agnostic analyzer rationale: {rationale}"
        )


# --------------------------------------------------------------------- #
#  Error paths                                                          #
# --------------------------------------------------------------------- #


def test_missing_body_returns_400(web):
    r, body = _post_analyze(web, {})
    assert r.status_code == 400
    assert body["ok"] is False
    assert "required" in body["error"].lower()


def test_unparseable_structure_returns_400(web):
    r, body = _post_analyze(web, {"structure_text": "garbage"})
    assert r.status_code == 400
    assert body["ok"] is False
    assert "could not parse" in body["error"].lower()


def test_unknown_element_returns_400_not_500(web):
    """An unknown element symbol must surface as a clean 400 (it
    propagates from ``total_electrons`` via the analyzer).  Pre-Phase-1c
    this leaked a 500 with a stack trace."""
    # 'Xx' is not a real element; total_electrons KeyErrors on it
    xyz = "1\nXx\nXx 0 0 0\n"
    r, body = _post_analyze(web, {"structure_text": xyz})
    assert r.status_code == 400
    assert body["ok"] is False
    assert "Xx" in body["error"] or "unknown" in body["error"].lower()


# --------------------------------------------------------------------- #
#  New-engine on-ramp (registered_adapters round-trip)                  #
# --------------------------------------------------------------------- #


def test_freshly_registered_adapter_appears_in_endpoint_response(web):
    """The endpoint iterates ``registered_adapters()`` — registering
    a new adapter at runtime must surface it in ``suggested.<name>``
    on the next request.  Catches a regression where the endpoint
    hardcodes the engine list (which was the pre-Phase-1c state).
    """
    from molbuilder.chemistry import (
        register_adapter, registered_adapters, _clear_adapters_for_test,
    )
    # Save + restore the registry around this test so the synthetic
    # adapter doesn't leak into sibling tests (per the
    # ba96288 lesson about test-fixture isolation).
    saved = registered_adapters()

    @dataclass(frozen=True)
    class StubParams:
        stub_charge: int
        rationale: str

    @register_adapter("stub_engine")
    class StubAdapter:
        name = "stub_engine"
        @classmethod
        def to_params(cls, analysis):
            return StubParams(
                stub_charge=analysis.suggested_charge,
                rationale=analysis.rationale,
            )

    try:
        xyz = "1\nFe\nFe 0 0 0\n"
        r, body = _post_analyze(web, {"structure_text": xyz})
        assert r.status_code == 200
        assert "stub_engine" in body["suggested"]
        stub = body["suggested"]["stub_engine"]
        assert stub == {"stub_charge": 0, "rationale": stub["rationale"]}
    finally:
        _clear_adapters_for_test()
        for name, cls in saved.items():
            register_adapter(name)(cls)
