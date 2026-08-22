"""The spectra config surface -- defaults and validation metadata.

The config a spectra calculation is described by is `PySCFConfig`,
seen through the vibration deck's view (`_spectra_cfg`).  A separate
`SpectraConfig` class carried this surface until 2026-08-22.

What remains after P3: the dataclass's defaults (the values the
registry-path validator and the parity reference read) and its
validation-facing metadata.  The form-schema half of this file retired
with the /api/build/schema/spectra route -- nothing renders a form from
this dataclass any more (see the tombstones below), and the class
itself is a recorded retirement candidate deferred to transport's
round (it shares the four-engine validator registry).
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.spectra._helpers import _spectra_cfg



# --------------------------------------------------------------------- #
#  the spectra config surface -- defaults, metadata, form-schema shape #
# --------------------------------------------------------------------- #


class TestSpectraDefaults:
    """The dataclass instantiates with all-defaults and the values
    match the v1 spec defaults so a user's first-pass run is
    cheap (es_mode_selection=none) and uses production-defensible
    method/basis choices (B3LYP/def2-SVP/D3BJ, grid 4)."""

    def test_a_spectra_calculations_science_defaults(self):
        """What a vibration run gets when the user chooses nothing:
        B3LYP / def2-SVP / D3BJ, restricted, density-fitted.

        Pinned HERE and nowhere else.  The catalogue-agreement gate
        mirrors help / range / unit / choices / label / engine_key
        between the class and the catalogue -- **not defaults** -- so
        without this a silent change to the functional or the basis
        would reach a user's spectrum with nothing failing.
        """
        cfg = _spectra_cfg()
        assert cfg.job_name   == "pyscf_relax"
        assert cfg.method     == "RKS"
        assert cfg.functional == "B3LYP"
        assert cfg.basis      == "def2-SVP"
        assert cfg.dispersion == "d3bj"
        assert cfg.density_fit is True

    def test_atom_freeze_list_is_empty_by_default(self):
        """Frozen atoms are INDICES and come from the structure's region
        store -- empty until the user freezes something.  (Two further
        lists, by element and by residue name, went with `SpectraConfig`
        on 2026-08-22: nothing populated them and one unreachable branch
        read them.)"""
        cfg = _spectra_cfg()
        assert cfg.frozen_indices == []

    def test_es_off_by_default(self):
        """First-pass run should be cheap -- spectrum only, no
        displaced SCFs.  User opts in to top_n / explicit after
        seeing the spectrum."""
        cfg = _spectra_cfg()
        assert cfg.es_mode_selection == "skip"

    def test_ir_off_v1_reserved(self):
        """compute_ir is reserved for 1c and ignored in v1."""
        cfg = _spectra_cfg()
        assert cfg.compute_ir is False

    def test_displacement_amplitude_production_default(self):
        """0.02 Å keeps ES probes inside the linear-response regime
        (ΔE_orbital ∝ displacement) and well below the threshold
        where Mills 1972 §2.4 anharmonic mixing becomes meaningful.
        Lowered from 0.10 → 0.02 on 2026-05-19; see docstring on
        ``SpectraConfig.displacement_amplitude_ang`` for the
        trade-off rationale."""
        cfg = _spectra_cfg()
        assert cfg.displacement_amplitude_ang == pytest.approx(0.02)


class TestSpectraFieldMetadata:
    """The metadata that SURVIVED P3 is the validation vocabulary
    (range / validate / choices / pattern) -- the form keys moved to
    the catalogue with the vibration items."""

    # test_every_sectioned_field_has_label_and_help retired at P3:
    # the form keys (section/label/help/...) left the dataclass
    # when the catalogue became the one form source -- only the
    # VALIDATION keys (range/validate/choices/pattern) remain,
    # and the two tests below pin exactly those.

    def test_basename_validator_attached(self):
        """job_name carries the shared _validate_basename callable
        in its metadata so the validation pass refuses paths-with-
        dots / slashes / whitespace (per docs/execution/job-contracts.md).

        Also pinned only here: `validate` is not in the catalogue gate's
        mirrored set, so nothing else notices if the callable is dropped.
        """
        from molbuilder.config.pyscf import PySCFConfig
        jn = next(f for f in dataclasses.fields(PySCFConfig)
                  if f.name == "job_name")
        assert callable(jn.metadata.get("validate"))

    def test_choices_are_enforced_where_a_user_supplies_one(self, tmp_path):
        """A value outside a field's declared ``choices`` is refused --
        checked at the DESCRIPTION layer, which is where a user actually
        supplies one.

        The retired `SpectraConfig` enforced this in `__post_init__`,
        which only ever fired for a caller constructing the class by
        hand.  The guard that protects a real user reads `task.json`'s
        stage overrides against the catalogue
        (`validation/task.py::preflight`), and it refuses at ERROR --
        verified here rather than assumed, because retiring a class that
        carried a check is exactly when a check goes missing.
        """
        import json
        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.identity import run_id
        from molbuilder.task import read_task
        from molbuilder.validation.task import preflight

        (tmp_path / "task.json").write_text(json.dumps({
            "schema": "molbuilder/task@1", "engine": {"name": "pyscf"},
            "shape": "flat", "calculation": "vibration",
            "run": {"name": "v", "id": run_id("v", "H2O"),
                    "created": "2026-08-22T00:00:00-07:00"},
            "structure": {"source": "w.xyz", "formula": "H2O", "atoms": 3},
            "varies": ["dispersion"],
            "stages": [{"name": "coarse", "enabled": True,
                        "overrides": {"dispersion": "d5"}}]}))
        issues = preflight(read_task(tmp_path / "task.json"), PySCFConfig)
        bad = [i for i in issues
               if i.severity == "error" and "dispersion" in (i.message or "")]
        assert bad, [i.message for i in issues]
        assert "not one of" in bad[0].message
        # None on Optional[str] field stays valid.
        _spectra_cfg(dispersion=None)
        # All declared choices remain accepted.
        for v in ("skip", "all", "top_n", "threshold", "explicit"):
            _spectra_cfg(es_mode_selection=v)


# TestSpectraConfigSchema retired at P3 with the
# /api/build/schema/spectra route: nothing renders a form from
# this dataclass any more -- the catalogue vibration schema is
# pinned by test_catalogue_form_schema.py per kind.


class TestFreqRangeFilter:
    """archived-spec (docs/archive/old_docs/tabs/spectra/spec.md) § 8.1: the freq_min_cm1 / freq_max_cm1 fields appear
    in the Electronic-structure section of the form schema and
    are typed Optional[float].  The Model-2 selector logic that
    APPLIES the filter lives in selection.py (next commit); this
    class just pins the dataclass + schema surface."""

    def test_default_no_filter(self):
        cfg = _spectra_cfg()
        assert cfg.freq_min_cm1 is None
        assert cfg.freq_max_cm1 is None

    # test_freq_fields_in_es_section + test_freq_fields_render_as_
    # optional_number retired at P3: form placement and widget kind
    # are the CATALOGUE's facts now (the freq_min/max_cm1 items
    # carry calculations = ["vibration"] and optional float typing
    # there, pinned by test_catalogue_form_schema.py).
