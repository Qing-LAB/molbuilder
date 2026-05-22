"""SpectraConfig surface tests: defaults, field metadata, form-schema shape.

Pins the dataclass shape documented in ``docs/tabs/spectra/spec.md``
§ 4 + § 8.1.  These tests are runtime-cheap (no PySCF) and live here
because the data they protect is the form/script handoff contract:

  * default values match the v1 spec (cheap first-pass + production-
    defensible method/basis choices);
  * every form-exposed field carries the ``section`` + ``label`` +
    ``help`` metadata the UI consumes;
  * the form schema's section order + per-section field counts pin
    the rendered form against accidental reorders.

Selector behaviour (which modes a config selects) lives in
``test_selection.py``.  Engine wiring lives in ``test_engine.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from molbuilder.spectra import SpectraConfig
from molbuilder.web.blueprints._shared import dataclass_to_form_schema


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
    """Every form-exposed field carries the metadata the
    schema-driven UI + validator + CLI bridge consume."""

    def test_every_sectioned_field_has_label_and_help(self):
        """Spec-required: any field with a ``section`` key must
        also carry ``label`` + ``help`` so the rendered form has
        a name and a tooltip and the Methods generator has a
        human-facing string to compose with."""
        missing_label = []
        missing_help  = []
        for f in dataclasses.fields(SpectraConfig):
            if "section" not in f.metadata:
                continue
            if not f.metadata.get("label"):
                missing_label.append(f.name)
            if not f.metadata.get("help"):
                missing_help.append(f.name)
        assert missing_label == [], f"missing label on: {missing_label}"
        assert missing_help  == [], f"missing help on:  {missing_help}"

    def test_basename_validator_attached(self):
        """job_name carries the shared _validate_basename callable
        in its metadata so the validation pass refuses paths-with-
        dots / slashes / whitespace (per docs/protocols/job-layout.md)."""
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


class TestSpectraConfigSchema:
    """Form-schema shape pin: section names + per-section field
    counts.  A stray field-reorder or a forgotten metadata addition
    would silently rearrange the UI; this test catches it."""

    def test_schema_section_layout(self):
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        assert sch["config"]    == "SpectraConfig"
        assert sch["id_prefix"] == "sp"

        expected = [
            ("System",               2),   # engine, job_name
            ("Method",               5),   # method, functional, basis,
                                           # dispersion, density_fit
            ("Frozen atoms",         3),   # elements, residue_names, indices
            ("Spectrum",             3),   # compute_raman, compute_ir,
                                           # displacement_amplitude_ang
            ("Electronic structure", 8),   # selection, top_n, threshold,
                                           # explicit_indices, freq_min_cm1,
                                           # freq_max_cm1, n_homo_below,
                                           # n_lumo_above
            ("SCF",                  3),   # conv_tol, max_cycle, grid_level
            ("Runtime",              5),   # max_memory_mb, threads,
                                           # use_gpu, verbose, verbose_comments
        ]
        got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
        assert got == expected, got

    def test_es_selection_choices_match_spec(self):
        """The Model 2 selector (spec § 8) must offer exactly the
        five documented options -- none / all / top_n / threshold /
        explicit -- in that order so the form ordering is stable."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        es_section = next(s for s in sch["sections"]
                          if s["name"] == "Electronic structure")
        sel_field = next(f for f in es_section["fields"]
                         if f["name"] == "es_mode_selection")
        assert sel_field["kind"]    == "select"
        assert sel_field["choices"] == ["skip", "all", "top_n",
                                        "threshold", "explicit"]

    def test_engine_field_carries_only_pyscf_in_v1(self):
        """v1 ships PySCF only; the SIESTA slot is reserved but
        not yet a valid choice.  The schema test pins this so
        adding SIESTA later is an explicit one-line change to
        SpectraConfig.engine's choices tuple AND to this test."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        system_section = next(s for s in sch["sections"]
                              if s["name"] == "System")
        engine_field = next(f for f in system_section["fields"]
                            if f["name"] == "engine")
        assert engine_field["choices"] == ["pyscf"]

    def test_legacy_id_overrides_preserved(self):
        """A handful of fields carry id_suffix overrides so the
        rendered HTML id is shorter / matches the UI affordances.
        Pinning the mapping so a future rename of the field name
        doesn't accidentally break the form selectors."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        fmap = {f["name"]: f
                for s in sch["sections"]
                for f in s["fields"]}
        # All these fields opt out of the default underscore->hyphen
        # ID transform via id_suffix metadata.
        assert fmap["job_name"]["id"]          == "sp-job-name"
        assert fmap["max_memory_mb"]["id"]     == "sp-max-memory"
        assert fmap["es_mode_selection"]["id"] == "sp-es-selection"
        assert fmap["es_n_homo_below"]["id"]   == "sp-es-n-homo-below"
        assert fmap["es_n_lumo_above"]["id"]   == "sp-es-n-lumo-above"


class TestFreqRangeFilter:
    """Spec § 8.1: the freq_min_cm1 / freq_max_cm1 fields appear
    in the Electronic-structure section of the form schema and
    are typed Optional[float].  The Model-2 selector logic that
    APPLIES the filter lives in selection.py (next commit); this
    class just pins the dataclass + schema surface."""

    def test_default_no_filter(self):
        cfg = SpectraConfig()
        assert cfg.freq_min_cm1 is None
        assert cfg.freq_max_cm1 is None

    def test_freq_fields_in_es_section(self):
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        es_section = next(s for s in sch["sections"]
                          if s["name"] == "Electronic structure")
        names = [f["name"] for f in es_section["fields"]]
        assert "freq_min_cm1" in names
        assert "freq_max_cm1" in names

    def test_freq_fields_render_as_optional_number(self):
        """Both freq fields are Optional[float] -> the schema
        emits kind='number' with null_option=True so the form
        widget can offer "no bound" empty-string mode."""
        sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
        fmap = {f["name"]: f
                for s in sch["sections"]
                for f in s["fields"]}
        for fname in ("freq_min_cm1", "freq_max_cm1"):
            f = fmap[fname]
            assert f["kind"] == "number"
            assert f.get("null_option") is True
            assert f["unit"] == "cm⁻¹"

