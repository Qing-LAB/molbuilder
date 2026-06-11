"""Tests for :class:`TransportConfig` and its field metadata.

The transport config is engine-agnostic -- TranSIESTA and PySCF-NEGF
both consume it -- so the tests pin:

  1.  Defaults instantiate without arguments (zero-friction config
      object that the form/CLI/engines can read straight away).
  2.  All fields carry the metadata keys the form-schema pipeline
      depends on (``section``, ``label``, ``help``).
  3.  The form-section order + descriptions are coherent (no
      ``section`` value appears in fields that isn't in
      ``_form_section_order``).
  4.  Engine choices include both backends.
  5.  Region-label constants are stable strings (downstream engines
      import them by name, not by value).
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from molbuilder.config.transport import (
    EXPECTED_REGIONS_2T,
    REGION_BRIDGE,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    TransportConfig,
)


class TestDefaults:
    def test_instantiates_with_no_args(self):
        cfg = TransportConfig()
        assert cfg.engine == "transiesta"
        assert cfg.job_name == "transport"
        assert cfg.bias_voltages_v == [0.0]
        assert cfg.k_mesh_transverse == (1, 1, 1)

    def test_transmission_window_default_is_symmetric(self):
        cfg = TransportConfig()
        assert cfg.transmission_emin_ev == -2.0
        assert cfg.transmission_emax_ev == 2.0
        assert cfg.transmission_n_points == 401

    def test_default_mutables_are_per_instance(self):
        """The ``default_factory`` machinery prevents the classic
        Python footgun where two instances share the same list."""
        a = TransportConfig()
        b = TransportConfig()
        a.bias_voltages_v.append(0.5)
        assert b.bias_voltages_v == [0.0]


class TestFieldMetadata:
    def test_every_field_has_label_and_section(self):
        for f in fields(TransportConfig):
            assert "section" in f.metadata, f"{f.name}: missing section"
            assert "label"   in f.metadata, f"{f.name}: missing label"

    def test_every_field_has_help(self):
        """Help text drives both the UI tooltip and the Methods-text
        generator; a missing help means a knob ships undocumented."""
        for f in fields(TransportConfig):
            assert "help" in f.metadata, f"{f.name}: missing help"
            assert len(f.metadata["help"]) > 10, \
                f"{f.name}: help is too short to be useful"

    def test_sections_match_declared_order(self):
        declared = set(TransportConfig._form_section_order)
        for f in fields(TransportConfig):
            sect = f.metadata["section"]
            assert sect in declared, \
                f"{f.name}: section {sect!r} not in _form_section_order"

    def test_every_section_has_a_description(self):
        for sect in TransportConfig._form_section_order:
            assert sect in TransportConfig._form_section_descriptions, \
                f"section {sect!r}: missing description"


class TestEngineChoices:
    def test_both_engines_present(self):
        engine_field = next(f for f in fields(TransportConfig)
                            if f.name == "engine")
        assert "transiesta" in engine_field.metadata["choices"]
        assert "pyscf-negf" in engine_field.metadata["choices"]


class TestRegionLabels:
    def test_canonical_strings(self):
        assert REGION_LEFT_ELECTRODE  == "L-electrode"
        assert REGION_RIGHT_ELECTRODE == "R-electrode"
        assert REGION_BRIDGE          == "bridge"

    def test_expected_set_is_complete(self):
        assert set(EXPECTED_REGIONS_2T) == {
            REGION_LEFT_ELECTRODE,
            REGION_RIGHT_ELECTRODE,
            REGION_BRIDGE,
        }


class TestJobNameValidation:
    def test_default_is_valid(self):
        """The validate callback should accept the default value
        (otherwise the form would error before the user types anything)."""
        job_field = next(f for f in fields(TransportConfig)
                         if f.name == "job_name")
        validate = job_field.metadata["validate"]
        # Returns None / empty list on success; truthy on failure.
        # Both conventions are used across the validate callbacks
        # in the rest of the codebase; the relevant thing is "no
        # error for the default".
        result = validate("transport")
        assert not result, f"default job_name rejected: {result!r}"

    def test_rejects_spaces(self):
        job_field = next(f for f in fields(TransportConfig)
                         if f.name == "job_name")
        validate = job_field.metadata["validate"]
        assert validate("has spaces"), "should reject names with spaces"
