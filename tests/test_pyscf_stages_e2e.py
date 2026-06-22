"""L3 wire-path tests for PySCFConfig.stages (task #534 commit 3c —
pre-cutover groundwork).

Three-stage contract: the Build tab POSTs the stage-table payload
to ``/api/build/pyscf``; the server runs ``_pyscf_config_from_params``
which dispatches through ``coerce_to_field_type`` (List[StageSpec]
branch, _shared.py) to rebuild each row as a typed StageSpec.

This file pins the wire path BEFORE commit 4 (the generator cutover)
lands -- so the stages payload round-trips through the API without
500s today, and the post-cutover commit 4 can EXTEND these tests
with the ``STAGES = [...]`` script-body assertion.

What's pinned:

* /api/build/pyscf accepts a stages payload and returns ok=True
  (no crash on coerce, validator, or render).
* Direct ``_pyscf_config_from_params`` produces a PySCFConfig whose
  ``.stages`` list matches the input payload field-for-field --
  bool / int / float / str all coerce correctly even when the
  payload sends them as strings (mirrors a non-browser HTTP client).
* Per-row sparse update: a payload that touches only ``enabled``
  preserves the StageSpec defaults for the other knobs (this is the
  three-stage contract's "stage may be toggled off without re-
  declaring its knobs" guarantee).
* Coerce REJECTS structural errors (non-list, non-dict items) at
  the boundary with a typed exception, surfacing as HTTP 400 from
  the API rather than a silent drop.

What's NOT pinned here (deferred to commit 4):

* The rendered script's ``STAGES = [...]`` literal -- the generator
  doesn't read ``cfg.stages`` yet.  When commit 4 wires
  ``_emit_stages_loop()`` in, add the body assertion to
  ``test_api_build_pyscf_accepts_stages_payload``.
"""
from __future__ import annotations

import pytest


pytest.importorskip("flask")


from molbuilder.config.pyscf import PySCFConfig, StageSpec, _default_stages
from molbuilder.web.blueprints.build import _pyscf_config_from_params


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


_H2O_XYZ = """3
water
O   0.000   0.000   0.000
H   0.757   0.586   0.000
H  -0.757   0.586   0.000
"""


def _stage_payload(name, enabled, conv_tol, gmax, max_steps,
                   grms=3.0e-4, dmax=1.8e-3, drms=1.2e-3, etol=1.0e-6):
    """One stage row as it arrives from the JS stage-table form."""
    return {
        "name":      name,
        "enabled":   enabled,
        "conv_tol":  conv_tol,
        "gmax":      gmax,
        "grms":      grms,
        "dmax":      dmax,
        "drms":      drms,
        "etol":      etol,
        "max_steps": max_steps,
    }


_PUBLISHABLE_STAGES = [
    _stage_payload("stage1", True,  1.0e-7, 2.0e-3,  50),
    _stage_payload("stage2", True,  1.0e-9, 4.5e-4, 200),
    _stage_payload("stage3", False, 1.0e-10, 1.5e-5, 100,
                   grms=1.0e-5, dmax=6.0e-5, drms=4.0e-5, etol=1.0e-7),
]


# --------------------------------------------------------------------- #
#  /api/build/pyscf accepts the stages payload                          #
# --------------------------------------------------------------------- #


class TestApiBuildPyscfAcceptsStages:

    def test_stages_payload_yields_ok_true(self, web_client):
        """Generator (post-#534 commit 4) emits a ``STAGES = [...]``
        literal followed by a ``for STAGE in STAGES:`` driver loop.
        Pin the literal exists and carries the wire values verbatim
        so a schema regression that silently drops a stage shows up
        at the wire tier.
        """
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "xyz": _H2O_XYZ,
                "params": {"stages": _PUBLISHABLE_STAGES},
                "frozen_atoms": [],
                "regions": {},
            },
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is True, f"render failed: {body!r}"
        script = body.get("script") or ""
        assert script, "rendered script is empty"
        # Generator emits STAGES literal + per-stage driver.
        assert "STAGES = [" in script, "STAGES literal missing"
        assert "for STAGE in STAGES:" in script, "stages loop missing"
        assert "mol_eq = optimize(" in script
        assert "convergence_grms      = STAGE['grms']" in script
        # Only the two ENABLED rows from _PUBLISHABLE_STAGES land
        # in the literal (stage3 has enabled=False).
        assert "'name':      'stage1'" in script
        assert "'name':      'stage2'" in script
        assert "'name':      'stage3'" not in script, (
            "disabled stage leaked into STAGES literal"
        )
        # Per-stage conv_tol literals: stage1 = 1e-07, stage2 = 1e-09.
        assert "'conv_tol':  1e-07" in script
        assert "'conv_tol':  1e-09" in script

    def test_stages_payload_passes_through_validator(self, web_client):
        """A stages payload doesn't trip the existing PySCF validator.
        Guards against the validator gaining a ``stages`` check in
        commit 4 that accidentally rejects the publication-guide
        defaults.
        """
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "xyz": _H2O_XYZ,
                "params": {"stages": _PUBLISHABLE_STAGES},
                "frozen_atoms": [],
                "regions": {},
            },
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is True
        # Issues list is allowed to carry warnings (e.g. basis-tier
        # advisories) but must not surface a stages-related error.
        for issue in body.get("issues", []) or []:
            where = (issue.get("where") or "").lower()
            assert "stages" not in where, (
                f"unexpected stages-related issue: {issue!r}"
            )

    def test_omitting_stages_still_works(self, web_client):
        """Backwards compatibility: a payload without ``stages`` (the
        legacy shape) must still render -- guarantees the
        dataclass-default factory kicks in.
        """
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "xyz": _H2O_XYZ,
                "params": {},
                "frozen_atoms": [],
                "regions": {},
            },
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json().get("ok") is True


# --------------------------------------------------------------------- #
#  Coerce path: form payload -> PySCFConfig.stages                      #
# --------------------------------------------------------------------- #


class TestCoerceStagesFromParams:

    def test_coerce_round_trips_publishable_default(self):
        """The publication-guide default payload coerces into 3
        StageSpec instances with the exact wire values."""
        cfg = _pyscf_config_from_params({"stages": _PUBLISHABLE_STAGES})
        assert isinstance(cfg, PySCFConfig)
        assert len(cfg.stages) == 3
        for row, ref in zip(cfg.stages, _PUBLISHABLE_STAGES):
            assert isinstance(row, StageSpec)
            assert row.name      == ref["name"]
            assert row.enabled   is ref["enabled"]
            assert row.conv_tol  == pytest.approx(ref["conv_tol"])
            assert row.gmax      == pytest.approx(ref["gmax"])
            assert row.max_steps == ref["max_steps"]

    def test_coerce_accepts_string_typed_values(self):
        """Mirrors a non-browser HTTP client that sends number /
        bool fields as JSON strings ("1e-9", "true").  The
        ``_coerce_scalar`` branches in _shared.py must convert each
        per its declared dataclass type.
        """
        stringy = [
            {
                "name": "stage1",
                "enabled": "true",
                "conv_tol": "1e-7",
                "gmax": "2e-3",
                "max_steps": "50",
            },
            {
                "name": "stage2",
                "enabled": "false",
                "conv_tol": "1e-9",
                "gmax": "4.5e-4",
                "max_steps": "200",
            },
        ]
        cfg = _pyscf_config_from_params({"stages": stringy})
        assert cfg.stages[0].enabled is True
        assert cfg.stages[1].enabled is False
        assert cfg.stages[0].conv_tol  == pytest.approx(1.0e-7)
        assert cfg.stages[1].conv_tol  == pytest.approx(1.0e-9)
        assert cfg.stages[0].max_steps == 50
        assert isinstance(cfg.stages[0].max_steps, int)

    def test_coerce_sparse_row_falls_back_to_defaults(self):
        """A row that only ships ``name`` + ``enabled`` keeps the
        StageSpec defaults for the other knobs.  This protects the
        three-stage contract guarantee that toggling a stage off
        doesn't blank its tier-specific knobs.
        """
        sparse = [{"name": "only_toggle", "enabled": False}]
        cfg = _pyscf_config_from_params({"stages": sparse})
        assert len(cfg.stages) == 1
        row = cfg.stages[0]
        assert row.name    == "only_toggle"
        assert row.enabled is False
        # Defaults preserved.
        ref = StageSpec()
        assert row.conv_tol  == pytest.approx(ref.conv_tol)
        assert row.gmax      == pytest.approx(ref.gmax)
        assert row.max_steps == ref.max_steps

    def test_coerce_rejects_non_dict_row(self):
        """Structural error: a stages entry that isn't a dict must
        raise (the API converts the exception into a 400)."""
        with pytest.raises(TypeError, match="cannot coerce"):
            _pyscf_config_from_params({"stages": ["not a dict"]})

    def test_coerce_passes_non_list_payload_through(self):
        """The List[<dataclass>] branch only kicks in on list/tuple
        input -- a non-list value passes through unchanged and ends
        up on cfg.stages as-is.  This documents the boundary: the
        coerce path doesn't itself reject non-list payloads, so the
        upstream consumer (commit 4's generator + a future validator
        check) is responsible for surfacing a typed error.
        """
        # A bare dict passes through.  The dataclass stores it as-is.
        cfg = _pyscf_config_from_params({"stages": {"not": "a list"}})
        assert cfg.stages == {"not": "a list"}


# --------------------------------------------------------------------- #
#  API surface: bad payload -> HTTP 400                                  #
# --------------------------------------------------------------------- #


class TestApiBuildPyscfRejectsBadStages:

    def test_non_dict_stages_row_returns_400(self, web_client):
        """The wire surface translates a coerce TypeError into a
        clean 400 -- the user sees a typed error rather than a
        500 / silently dropped row."""
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "xyz": _H2O_XYZ,
                "params": {"stages": ["bogus"]},
                "frozen_atoms": [],
                "regions": {},
            },
        )
        assert r.status_code == 400, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is False
        assert "bad parameters" in (body.get("error") or "").lower()


# --------------------------------------------------------------------- #
#  PySCFConfig default-factory regression                                #
# --------------------------------------------------------------------- #


def test_default_factory_matches_wire_default_payload():
    """The dataclass's ``_default_stages()`` factory and the wire-
    payload's ``_PUBLISHABLE_STAGES`` fixture above are two views
    of the same publication-guide defaults; they must agree
    field-for-field so a wire round-trip is a no-op for the
    default UI state.
    """
    defaults = _default_stages()
    assert len(defaults) == len(_PUBLISHABLE_STAGES)
    for code_default, wire in zip(defaults, _PUBLISHABLE_STAGES):
        assert code_default.name      == wire["name"]
        assert code_default.enabled   is wire["enabled"]
        assert code_default.conv_tol  == pytest.approx(wire["conv_tol"])
        assert code_default.gmax      == pytest.approx(wire["gmax"])
        assert code_default.max_steps == wire["max_steps"]
