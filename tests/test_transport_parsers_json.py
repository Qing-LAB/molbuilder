"""Tests for ``molbuilder.parsers.transport_json``.

Pin the schema_version gate, malformed-input handling, field-error
wrapping, and missing-file case so the future live-watch poller +
the ``/api/transport/load`` endpoint have stable exception
semantics.

Pure JSON I/O + dataclass round trip — no engine work.  Mirrors
``tests/spectra/test_parsers_json.py`` so the two parsers stay in
lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molbuilder.transport.results import (
    SCHEMA_VERSION,
    TransportResults,
)
from molbuilder.parsers.transport_json import (
    dump_transport_json,
    parse_transport_json,
    parse_transport_json_dict,
    TransportJsonError,
    TransportJsonFieldError,
    TransportJsonMalformedError,
    TransportJsonNotFoundError,
    TransportJsonSchemaError,
)


def _make_minimal_results() -> TransportResults:
    """Minimal valid TransportResults — single-energy-point T(E) +
    metadata, no PDOS, no I(V).  Smallest payload that round-trips."""
    return TransportResults(
        metadata={
            "engine":          "transiesta",
            "engine_version":  "5.0.0",
            "molbuilder_version": "1.2.0",
            "job_name":        "test_junction",
            "timestamp":       "2026-06-09T12:00:00Z",
            "structure_hash":  "sha256:abc123",
        },
        energy_grid_eV=np.array([-1.0, 0.0, 1.0]),
        transmission=np.array([0.001, 0.012, 0.003]),
        fermi_energy_eV=0.0,
        conductance_G0=0.012,
        pdos={},
        bias_grid_V=None,
        current_uA=None,
        methods_text="Transport calculations were performed with TranSIESTA.",
        bibliography_keys=["transiesta_brandbyge_2002"],
        complete=True,
    )


# --------------------------------------------------------------------- #
#  Round trip                                                           #
# --------------------------------------------------------------------- #


class TestRoundTrip:

    def test_dump_then_parse_recovers_results(self, tmp_path: Path):
        results = _make_minimal_results()
        path = tmp_path / "test.transport.json"
        dump_transport_json(results, path)
        recovered = parse_transport_json(path)
        # Equality is forbidden on TransportResults; compare fields.
        assert recovered.metadata == results.metadata
        np.testing.assert_allclose(recovered.energy_grid_eV,
                                   results.energy_grid_eV)
        np.testing.assert_allclose(recovered.transmission,
                                   results.transmission)
        assert recovered.fermi_energy_eV == results.fermi_energy_eV
        assert recovered.conductance_G0 == results.conductance_G0
        assert recovered.bias_grid_V is None
        assert recovered.current_uA is None
        assert recovered.complete is True
        assert recovered.methods_text == results.methods_text
        assert recovered.bibliography_keys == results.bibliography_keys

    def test_dump_then_parse_with_iv_curve(self, tmp_path: Path):
        results = _make_minimal_results()
        results.bias_grid_V = np.array([0.0, 0.1, 0.2])
        results.current_uA = np.array([0.0, 1.5, 3.2])
        path = tmp_path / "iv.transport.json"
        dump_transport_json(results, path)
        recovered = parse_transport_json(path)
        np.testing.assert_allclose(recovered.bias_grid_V,
                                   results.bias_grid_V)
        np.testing.assert_allclose(recovered.current_uA,
                                   results.current_uA)

    def test_parse_dict_skips_filesystem(self):
        results = _make_minimal_results()
        d = results.to_dict()
        recovered = parse_transport_json_dict(d)
        np.testing.assert_allclose(recovered.transmission,
                                   results.transmission)


# --------------------------------------------------------------------- #
#  Schema-version gate                                                  #
# --------------------------------------------------------------------- #


class TestSchemaVersion:

    def test_missing_schema_version_raises(self, tmp_path: Path):
        path = tmp_path / "noversion.transport.json"
        path.write_text(json.dumps({
            "metadata": {},
            "energy_grid_eV": [],
            "transmission": [],
            "fermi_energy_eV": 0.0,
            "conductance_G0": 0.0,
            "pdos": {},
            "bias_grid_V": None,
            "current_uA": None,
            "methods_text": "",
            "bibliography_keys": [],
            "complete": False,
        }))
        with pytest.raises(TransportJsonSchemaError) as ei:
            parse_transport_json(path)
        assert ei.value.expected == SCHEMA_VERSION
        assert ei.value.actual is None

    def test_wrong_schema_version_raises(self, tmp_path: Path):
        path = tmp_path / "wrongversion.transport.json"
        path.write_text(json.dumps({
            "schema_version": "999",
            "metadata": {},
            "energy_grid_eV": [],
            "transmission": [],
            "fermi_energy_eV": 0.0,
            "conductance_G0": 0.0,
            "pdos": {},
            "bias_grid_V": None,
            "current_uA": None,
            "methods_text": "",
            "bibliography_keys": [],
            "complete": False,
        }))
        with pytest.raises(TransportJsonSchemaError) as ei:
            parse_transport_json(path)
        assert ei.value.actual == "999"

    def test_int_schema_version_rejected(self):
        """``schema_version`` is a str in the contract; int (even ``1``)
        is the wrong type."""
        d = _make_minimal_results().to_dict()
        d["schema_version"] = 1
        with pytest.raises(TransportJsonSchemaError):
            parse_transport_json_dict(d)

    def test_bool_schema_version_rejected(self):
        """Python's ``True == 1`` would slip past a naive str-check —
        the validator must explicitly reject bool."""
        d = _make_minimal_results().to_dict()
        d["schema_version"] = True
        with pytest.raises(TransportJsonSchemaError):
            parse_transport_json_dict(d)


# --------------------------------------------------------------------- #
#  Missing file / malformed JSON                                        #
# --------------------------------------------------------------------- #


class TestMissingAndMalformed:

    def test_missing_file_raises_not_found(self, tmp_path: Path):
        path = tmp_path / "does-not-exist.transport.json"
        with pytest.raises(TransportJsonNotFoundError):
            parse_transport_json(path)

    def test_not_found_inherits_filenotfound(self, tmp_path: Path):
        """Existing ``except FileNotFoundError`` handlers must
        continue to work — the dual-base inheritance preserves that."""
        path = tmp_path / "absent.transport.json"
        with pytest.raises(FileNotFoundError):
            parse_transport_json(path)

    def test_invalid_json_raises_malformed(self, tmp_path: Path):
        path = tmp_path / "garbage.transport.json"
        path.write_text("this is not json {{{")
        with pytest.raises(TransportJsonMalformedError):
            parse_transport_json(path)

    def test_non_object_top_level_raises_malformed(self, tmp_path: Path):
        path = tmp_path / "array.transport.json"
        path.write_text('[1, 2, 3]')
        with pytest.raises(TransportJsonMalformedError):
            parse_transport_json(path)

    def test_nan_token_rejected_on_read(self, tmp_path: Path):
        """Symbolic ``NaN`` is non-standard JSON; must be rejected
        even though Python's json module accepts it by default."""
        path = tmp_path / "nan.transport.json"
        path.write_text(json.dumps(_make_minimal_results().to_dict())
                         .replace('0.012', 'NaN'))
        with pytest.raises(TransportJsonMalformedError):
            parse_transport_json(path)

    def test_overflow_to_inf_rejected_on_read(self, tmp_path: Path):
        """Valid-syntax numeric literal that overflows to ±Inf must
        also be rejected — parse_float catches what parse_constant
        does not."""
        path = tmp_path / "overflow.transport.json"
        d = _make_minimal_results().to_dict()
        # Inject a numeric literal that decodes to inf.
        raw = json.dumps(d).replace('0.012', '1e500')
        path.write_text(raw)
        with pytest.raises(TransportJsonMalformedError):
            parse_transport_json(path)


# --------------------------------------------------------------------- #
#  Field-error wrapping                                                 #
# --------------------------------------------------------------------- #


class TestFieldErrors:

    def test_missing_required_field_raises(self, tmp_path: Path):
        d = _make_minimal_results().to_dict()
        del d["transmission"]
        path = tmp_path / "missing.transport.json"
        path.write_text(json.dumps(d))
        # Reconstitution sees a 0-length transmission default and
        # mismatched shape vs energy_grid_eV (length 3).  Should
        # surface as a FieldError.
        with pytest.raises(TransportJsonFieldError):
            parse_transport_json(path)

    def test_mismatched_array_shapes_raise(self, tmp_path: Path):
        d = _make_minimal_results().to_dict()
        d["transmission"] = [0.001, 0.012]  # length 2 vs energy length 3
        path = tmp_path / "mismatch.transport.json"
        path.write_text(json.dumps(d))
        with pytest.raises(TransportJsonFieldError):
            parse_transport_json(path)


# --------------------------------------------------------------------- #
#  Write path — atomic + NaN-safe                                       #
# --------------------------------------------------------------------- #


class TestWritePath:

    def test_dump_rejects_nan_in_payload(self, tmp_path: Path):
        results = _make_minimal_results()
        results.transmission[1] = float("nan")
        path = tmp_path / "nan-write.transport.json"
        # allow_nan=False in dump_transport_json raises ValueError
        # BEFORE bytes hit disk.
        with pytest.raises(ValueError):
            dump_transport_json(results, path)
        assert not path.exists(), (
            "dump_transport_json must not leave a partial file "
            "when validation fails"
        )

    def test_dump_atomic_no_temp_left_behind(self, tmp_path: Path):
        """The temp file used for atomic replace must not survive a
        successful write."""
        results = _make_minimal_results()
        path = tmp_path / "atomic.transport.json"
        dump_transport_json(results, path)
        # No sibling temp files should remain.
        tmp_siblings = list(tmp_path.glob("atomic.transport.json.*.tmp"))
        assert tmp_siblings == []
