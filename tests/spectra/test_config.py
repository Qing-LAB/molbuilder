"""SpectraConfig surface tests -- defaults and validation metadata.

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

from molbuilder.spectra import SpectraConfig


# --------------------------------------------------------------------- #
#  SpectraConfig -- L1 dataclass, field metadata, form-schema shape     #
# --------------------------------------------------------------------- #


class TestSpectraConfigDefaults:
    """The dataclass instantiates with all-defaults and the values
    match the v1 spec defaults so a user's first-pass run is
    cheap (es_mode_selection=none) and uses production-defensible
    method/basis choices (B3LYP/def2-SVP/D3BJ, grid 4)."""

    def test_default_construction(self):
        cfg = SpectraConfig()
        # Spec-pinned defaults.
        assert cfg.engine     == "pyscf"
        assert cfg.job_name   == "spectra"
        assert cfg.method     == "RKS"
        assert cfg.functional == "B3LYP"
        assert cfg.basis      == "def2-SVP"
        assert cfg.dispersion == "d3bj"
        assert cfg.density_fit is True

    def test_atom_freeze_lists_empty_by_default(self):
        cfg = SpectraConfig()
        assert cfg.frozen_elements      == []
        assert cfg.frozen_residue_names == []
        assert cfg.frozen_indices       == []

    def test_es_off_by_default(self):
        """First-pass run should be cheap -- spectrum only, no
        displaced SCFs.  User opts in to top_n / explicit after
        seeing the spectrum."""
        cfg = SpectraConfig()
        assert cfg.es_mode_selection == "skip"

    def test_ir_off_v1_reserved(self):
        """compute_ir is reserved for 1c and ignored in v1."""
        cfg = SpectraConfig()
        assert cfg.compute_ir is False

    def test_displacement_amplitude_production_default(self):
        """0.02 Å keeps ES probes inside the linear-response regime
        (ΔE_orbital ∝ displacement) and well below the threshold
        where Mills 1972 §2.4 anharmonic mixing becomes meaningful.
        Lowered from 0.10 → 0.02 on 2026-05-19; see docstring on
        ``SpectraConfig.displacement_amplitude_ang`` for the
        trade-off rationale."""
        cfg = SpectraConfig()
        assert cfg.displacement_amplitude_ang == pytest.approx(0.02)


class TestSpectraConfigFieldMetadata:
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
        dots / slashes / whitespace (per docs/execution/job-contracts.md)."""
        jn = next(f for f in dataclasses.fields(SpectraConfig)
                  if f.name == "job_name")
        assert callable(jn.metadata.get("validate"))

    def test_choices_metadata_is_enforced(self):
        """Any field carrying a ``choices`` tuple rejects values
        outside that tuple at construction time -- catches both UI
        typos and old-value drift after a rename (e.g. the pre-skip
        ``es_mode_selection="none"`` smoke-test value, which used to
        be silently accepted and then no-op'd downstream)."""
        with pytest.raises(ValueError, match="es_mode_selection"):
            SpectraConfig(es_mode_selection="none")  # renamed -> "skip"
        with pytest.raises(ValueError, match="dispersion"):
            SpectraConfig(dispersion="d5")
        # None on Optional[str] field stays valid.
        SpectraConfig(dispersion=None)
        # All declared choices remain accepted.
        for v in ("skip", "all", "top_n", "threshold", "explicit"):
            SpectraConfig(es_mode_selection=v)


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
        cfg = SpectraConfig()
        assert cfg.freq_min_cm1 is None
        assert cfg.freq_max_cm1 is None

    # test_freq_fields_in_es_section + test_freq_fields_render_as_
    # optional_number retired at P3: form placement and widget kind
    # are the CATALOGUE's facts now (the freq_min/max_cm1 items
    # carry calculations = ["vibration"] and optional float typing
    # there, pinned by test_catalogue_form_schema.py).
