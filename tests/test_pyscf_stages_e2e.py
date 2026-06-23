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
        # 6c: optimize() now lives inside the _mb_run_stage_opt
        # helper (factored out of the loop body so the 3-policy
        # dispatch reads cleanly).
        assert "def _mb_run_stage_opt(STAGE, _hard_fail):" in script
        assert "return optimize(" in script
        # Per-stage kwarg keyed on the STAGE dict (not a hardcoded
        # value that would defeat per-stage control).
        assert "convergence_grms      = STAGE['grms']" in script
        assert "convergence_gmax      = STAGE['gmax']" in script
        assert "mf.conv_tol = STAGE['conv_tol']" in script

        # The STAGES literal must round-trip via ast.literal_eval to
        # the EXACT enabled-stage payload (not a substring check that
        # could pass with hardcoded defaults inside the loop body).
        import ast
        stages_text = script.split("STAGES = ")[1].split("\n]")[0] + "\n]"
        parsed = ast.literal_eval(stages_text)
        assert isinstance(parsed, list) and len(parsed) == 2, (
            f"expected 2 enabled stages in STAGES literal; got "
            f"{len(parsed) if isinstance(parsed, list) else type(parsed)}"
        )
        # Disabled stage3 must NOT leak through.
        names = [row["name"] for row in parsed]
        assert names == ["stage1", "stage2"], names
        # Values match _PUBLISHABLE_STAGES exactly.
        assert parsed[0]["conv_tol"] == 1.0e-7
        assert parsed[0]["max_steps"] == 50
        assert parsed[1]["conv_tol"] == 1.0e-9
        assert parsed[1]["max_steps"] == 200
        # 6c: per-stage non-convergence policy fields.  The user
        # payload didn't pass on_nonconvergence so defaults from
        # StageSpec apply: ``halt`` for un-customised rows.  The
        # is_final flag is derived from position at generation time
        # (last enabled = True).  These replace 5b's assert_convergence.
        assert "on_nonconvergence" in parsed[0]
        assert parsed[0]["is_final"] is False
        assert parsed[1]["is_final"] is True
        # continue_retries: integer in [1, 5], default 1.
        assert parsed[0]["continue_retries"] == 1

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

    def test_coerce_rejects_non_list_payload(self):
        """The List[<dataclass>] branch REJECTS non-list values at
        the boundary (#534 commit 5c).  Previously a dict-on-stages
        silently passed through to cfg.stages, where validate_stages
        would crash with AttributeError on ``s.enabled``.  Now the
        coerce raises TypeError -- the API path translates this to
        a clean 400 via the existing "bad parameters" branch in
        web/blueprints/build.py."""
        with pytest.raises(TypeError, match="expected list"):
            _pyscf_config_from_params({"stages": {"not": "a list"}})


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
#  Validator wires validate_stages: structural invariants that the      #
#  generator can't recover from must surface as a typed config.stages   #
#  error, NOT a runtime NameError on `mol_eq` from an empty STAGES      #
#  literal.  Validation errors travel via render_script's               #
#  ValidationError path, which returns HTTP 200 + ok=false + issues     #
#  (per web-api.md § 1.6).                                              #
# --------------------------------------------------------------------- #


class TestApiBuildPyscfRejectsBrokenStages:

    @staticmethod
    def _stages_issue(body):
        """Pull the first issue whose ``where`` is ``config.stages``."""
        for i in body.get("issues") or []:
            if (i.get("where") or "") == "config.stages":
                return i
        return None

    def test_empty_stages_list_surfaces_stages_error(self, web_client):
        """``cfg.stages = []`` -> generator would emit ``STAGES = []``
        + an empty for-loop -> ``mol_eq`` never bound -> downstream
        ``_save_xyz(mol_eq, ...)`` raises NameError at runtime.
        The validator must catch this at the API tier."""
        r = web_client.post(
            "/api/build/pyscf",
            json={"xyz": _H2O_XYZ, "params": {"stages": []},
                  "frozen_atoms": [], "regions": {}},
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is False, f"expected ok=false; got {body!r}"
        issue = self._stages_issue(body)
        assert issue is not None, (
            f"missing config.stages issue.  Body: {body!r}"
        )
        assert issue["severity"] == "error"
        assert "empty" in issue["message"].lower() \
            or "at least one" in issue["message"].lower()

    def test_all_disabled_stages_surfaces_stages_error(self, web_client):
        """Same trap as empty list: every stage disabled means the
        enabled-filter produces ``[]`` at generation time."""
        all_off = [
            {"name": "stage1", "enabled": False},
            {"name": "stage2", "enabled": False},
        ]
        r = web_client.post(
            "/api/build/pyscf",
            json={"xyz": _H2O_XYZ, "params": {"stages": all_off},
                  "frozen_atoms": [], "regions": {}},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is False
        issue = self._stages_issue(body)
        assert issue is not None and issue["severity"] == "error"

    def test_duplicate_stage_names_surfaces_stages_error(self, web_client):
        """Per-stage geomeTRIC trajectory files are named
        ``<JOB>_geom_<stage>``; duplicate names would collide on disk
        AND duplicate-key the nested convergence_targets header."""
        dups = [
            {"name": "stage1", "enabled": True},
            {"name": "stage1", "enabled": True},
        ]
        r = web_client.post(
            "/api/build/pyscf",
            json={"xyz": _H2O_XYZ, "params": {"stages": dups},
                  "frozen_atoms": [], "regions": {}},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is False
        issue = self._stages_issue(body)
        assert issue is not None and "duplicate" in issue["message"].lower()

    def test_bad_stage_name_charset_surfaces_stages_error(self, web_client):
        """Stage names land in filesystem paths (``_geom_<stage>``);
        bogus chars would either break filename sanitisation or
        produce surprising path traversal."""
        bad = [{"name": "stage 1!", "enabled": True}]
        r = web_client.post(
            "/api/build/pyscf",
            json={"xyz": _H2O_XYZ, "params": {"stages": bad},
                  "frozen_atoms": [], "regions": {}},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is False
        issue = self._stages_issue(body)
        assert issue is not None

    def test_optimize_false_skips_stages_validation(self, web_client):
        """Single-point runs (``cfg.optimize = False``) don't emit a
        stages loop -- the structural invariants don't apply.  Pin
        that the validator doesn't surface a stages issue on an
        otherwise-empty stages list when optimize is off."""
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "xyz": _H2O_XYZ,
                "params": {"optimize": False, "stages": []},
                "frozen_atoms": [], "regions": {},
            },
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("ok") is True, (
            f"single-point with empty stages should NOT trip "
            f"validate_stages; got: {body!r}"
        )


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


# --------------------------------------------------------------------- #
#  7a (#534 closure): per-stage geomeTRIC criteria reach molwatch       #
#  convergence_targets header end-to-end.                                #
#                                                                       #
#  Previously only max-grad + scf-energy + 2 iter caps were emitted     #
#  per stage; user-set grms / dmax / drms / etol were silently lost.    #
#  Now the Results-tab inspector can draw threshold lines for ANY of    #
#  the 5 geomeTRIC convergence checks.                                  #
# --------------------------------------------------------------------- #


class TestMolwatchConvergenceTargetsCarriesAllSixKnobs:

    HA_BOHR_TO_EV_ANG = 51.42208619
    HARTREE_TO_EV     = 27.211386245988

    def _render(self, web_client):
        r = web_client.post(
            "/api/build/pyscf",
            json={"xyz": _H2O_XYZ, "params": {},
                  "frozen_atoms": [], "regions": {}},
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is True
        return body["script"]

    def _parse_conv_targets(self, script):
        import ast, re
        m = re.search(r"_CONVERGENCE_TARGETS = (\{.*?\n\})",
                      script, re.DOTALL)
        assert m, "missing _CONVERGENCE_TARGETS literal in script"
        return ast.literal_eval(m.group(1))

    def test_each_enabled_stage_emits_all_six_leaves(self, web_client):
        """Every enabled stage's dict must carry all 6 geomeTRIC
        criteria converted to the molwatch unit convention (eV/Å for
        forces, Å for displacements, eV for energy step) PLUS the
        SCF-energy + iter caps.  8 leaves total per stage."""
        targets = self._parse_conv_targets(self._render(web_client))
        # Two stages enabled by default (stage1 + stage2).
        assert set(targets.keys()) == {"stage1", "stage2"}
        REQUIRED = {
            "max_force_tol_eV_per_A",
            "rms_force_tol_eV_per_A",
            "max_displ_ang",
            "rms_displ_ang",
            "energy_step_tol_eV",
            "scf_energy_tol",
            "max_scf_iter",
            "max_geom_iter",
        }
        for stage_name, leaves in targets.items():
            missing = REQUIRED - set(leaves.keys())
            assert not missing, (
                f"{stage_name} missing leaves: {missing}.  "
                f"Got: {sorted(leaves.keys())}"
            )

    def test_unit_conversion_matches_emitter_constants(self, web_client):
        """The generator's Ha/Bohr -> eV/Å and Ha -> eV conversions
        must round-trip to the same constants MolwatchEmitter uses
        for the per-step force / energy emission, so threshold lines
        and plotted values are on the same axis."""
        targets = self._parse_conv_targets(self._render(web_client))
        # stage2 = publishable, the Gaussian-OPT defaults.
        s2 = targets["stage2"]
        # gmax = 4.5e-4 Ha/Bohr  -> 4.5e-4 * 51.42208619 eV/Å
        assert s2["max_force_tol_eV_per_A"] == pytest.approx(
            4.5e-4 * self.HA_BOHR_TO_EV_ANG)
        # grms = 3.0e-4 Ha/Bohr  -> 3.0e-4 * 51.42208619 eV/Å
        assert s2["rms_force_tol_eV_per_A"] == pytest.approx(
            3.0e-4 * self.HA_BOHR_TO_EV_ANG)
        # dmax / drms already in Å -- carried verbatim
        assert s2["max_displ_ang"] == pytest.approx(1.8e-3)
        assert s2["rms_displ_ang"] == pytest.approx(1.2e-3)
        # etol = 1e-6 Ha -> 1e-6 * 27.211386245988 eV
        assert s2["energy_step_tol_eV"] == pytest.approx(
            1.0e-6 * self.HARTREE_TO_EV)
        # SCF energy tol stays in Hartree (parser tags the unit)
        assert s2["scf_energy_tol"] == pytest.approx(1.0e-9)

    def test_per_stage_values_differ_between_tiers(self, web_client):
        """A regression that silently uses stage1's values for stage2
        (or vice versa) would slip through if we only checked
        existence.  Pin that the per-tier numbers ACTUALLY differ
        per stage -- the whole point of the staged ladder."""
        targets = self._parse_conv_targets(self._render(web_client))
        s1, s2 = targets["stage1"], targets["stage2"]
        # stage1 is loose preopt (gmax 2e-3), stage2 publishable
        # (gmax 4.5e-4).  Their force-tol-eV-per-A values must
        # differ by the same ~4.4x factor.
        assert s1["max_force_tol_eV_per_A"] > s2["max_force_tol_eV_per_A"]
        assert s1["rms_force_tol_eV_per_A"] > s2["rms_force_tol_eV_per_A"]
        # SCF energy tol: stage1 1e-7, stage2 1e-9 (100x tighter)
        assert s1["scf_energy_tol"] > s2["scf_energy_tol"]
        # Geom-iter cap: stage1 50, stage2 200
        assert s1["max_geom_iter"] == 50
        assert s2["max_geom_iter"] == 200
