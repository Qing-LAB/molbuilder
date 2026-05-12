"""Tests for ``molbuilder.parsers.spectra_json``.

Pin the schema_version gate, the malformed-input handling, the
field-error wrapping, and the missing-file case so the live-watch
poller + the ``/api/spectra/load`` endpoint have stable
exception semantics.

No PySCF, no engine work -- pure JSON I/O + dataclass round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molbuilder.spectra import (
    ModeData,
    SpectraConfig,  # noqa: F401  (kept for symmetry with other tests)
    SpectraResults,
)
from molbuilder.spectra.results import (
    SCHEMA_VERSION,
    PHASE_COMPLETE,
    PHASE_EMPTY,
)
from molbuilder.parsers.spectra_json import (
    dump_spectra_json,
    parse_spectra_json,
    parse_spectra_json_dict,
    SpectraJsonError,
    SpectraJsonFieldError,
    SpectraJsonMalformedError,
    SpectraJsonNotFoundError,
    SpectraJsonSchemaError,
)


# --------------------------------------------------------------------- #
#  Fixture                                                              #
# --------------------------------------------------------------------- #


def _make_minimal_results(complete: bool = True) -> SpectraResults:
    """Single-mode result -- the smallest valid SpectraResults so
    each test can write a tiny JSON file and round-trip it without
    masking failures with fixture noise."""
    phases = ((PHASE_COMPLETE, PHASE_COMPLETE, PHASE_EMPTY) if complete
              else (PHASE_COMPLETE, PHASE_EMPTY, PHASE_EMPTY))
    return SpectraResults(
        schema_version             = SCHEMA_VERSION,
        engine                     = "pyscf",
        engine_version             = "2.6.0",
        molbuilder_version         = "1.2.0",
        timestamp                  = "2026-05-11T12:00:00Z",
        structure_hash             = "sha256:abc123",
        n_atoms_total              = 2,
        free_atom_idxs             = [0, 1],
        fixed_atom_idxs            = [],
        equilibrium_scf_eh         = -76.4123,
        equilibrium_mo_energies_eh = np.array([-1.0, -0.5, -0.2, 0.1, 0.3]),
        equilibrium_homo_idx       = 2,
        modes                      = [
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 412.3,
                raman_activity_a4_amu = 12.5,
                ir_intensity_km_mol   = None,
                eigenvector_free      = np.array([[0.7, 0.0, 0.0],
                                                  [-0.7, 0.0, 0.0]]),
                has_imag              = False,
            ),
        ],
        selected_mode_idxs_1based  = [],
        config                     = {"engine": "pyscf"},
        methods_text               = "",
        bibliography_keys          = [],
        phase_frequencies          = phases[0],
        phase_raman                = phases[1],
        phase_es                   = phases[2],
    )


def _write_json(tmp_path: Path, payload: dict, name: str = "spectra.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
#  Happy path                                                           #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonHappyPath:

    def test_round_trip_minimal_results(self, tmp_path):
        original = _make_minimal_results()
        p = _write_json(tmp_path, original.to_dict())
        loaded = parse_spectra_json(p)
        assert loaded.engine == "pyscf"
        assert loaded.n_atoms_total == 2
        assert len(loaded.modes) == 1
        assert loaded.modes[0].frequency_cm1 == pytest.approx(412.3)
        # MO energies are numpy arrays -- compare element-wise.
        np.testing.assert_allclose(
            loaded.equilibrium_mo_energies_eh,
            original.equilibrium_mo_energies_eh,
        )

    def test_accepts_pathlike_input(self, tmp_path):
        """os.PathLike (e.g. pathlib.Path) should work directly --
        the live-watch poller hands us Paths, not strings."""
        original = _make_minimal_results()
        p = _write_json(tmp_path, original.to_dict())
        # Passing the Path object directly:
        loaded = parse_spectra_json(p)
        assert loaded.engine == "pyscf"

    def test_accepts_str_path(self, tmp_path):
        original = _make_minimal_results()
        p = _write_json(tmp_path, original.to_dict())
        loaded = parse_spectra_json(str(p))
        assert loaded.engine == "pyscf"

    def test_intermediate_phase_state_round_trips(self, tmp_path):
        """A partially-complete file (L2 done, L3+L4 empty) round-
        trips cleanly -- the parser doesn't reject incomplete runs,
        only malformed ones."""
        partial = _make_minimal_results(complete=False)
        p = _write_json(tmp_path, partial.to_dict())
        loaded = parse_spectra_json(p)
        assert loaded.phase_frequencies == PHASE_COMPLETE
        assert loaded.phase_raman       == PHASE_EMPTY
        assert loaded.phase_es          == PHASE_EMPTY


# --------------------------------------------------------------------- #
#  Missing file                                                         #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonMissing:

    def test_missing_file_raises_not_found_error(self, tmp_path):
        bad = tmp_path / "does_not_exist.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError) as exc_info:
            parse_spectra_json(bad)
        # Inherits FileNotFoundError -- legacy callers using the
        # OSError-shaped except keep working.
        assert isinstance(exc_info.value, FileNotFoundError)
        # And the SpectraJsonError base so generic catches also work.
        assert isinstance(exc_info.value, SpectraJsonError)

    def test_missing_file_message_names_path(self, tmp_path):
        bad = tmp_path / "missing.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError) as exc_info:
            parse_spectra_json(bad)
        assert str(bad) in str(exc_info.value)


# --------------------------------------------------------------------- #
#  Malformed JSON                                                       #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonMalformed:

    def test_invalid_json_raises_malformed(self, tmp_path):
        p = tmp_path / "bad.spectra.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_top_level_not_object_raises_malformed(self, tmp_path):
        """A bare JSON list / number is valid JSON but the wrong
        top-level shape."""
        p = tmp_path / "bad.spectra.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError) as exc_info:
            parse_spectra_json(p)
        assert "object" in str(exc_info.value).lower()

    def test_empty_file_raises_malformed(self, tmp_path):
        p = tmp_path / "empty.spectra.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Schema-version mismatch                                              #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonSchemaVersion:

    def test_missing_schema_version_is_schema_error(self, tmp_path):
        """No ``schema_version`` key -> SchemaError, not FieldError,
        because we check it BEFORE reconstitution."""
        payload = _make_minimal_results().to_dict()
        del payload["schema_version"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError) as exc_info:
            parse_spectra_json(p)
        assert exc_info.value.expected == SCHEMA_VERSION
        assert exc_info.value.actual is None

    def test_future_schema_version_rejected(self, tmp_path):
        """A schema_version=2 file (future engine) is rejected here;
        users who hit this get an "update molbuilder" message."""
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = SCHEMA_VERSION + 1
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError) as exc_info:
            parse_spectra_json(p)
        assert exc_info.value.actual == SCHEMA_VERSION + 1

    def test_legacy_schema_version_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = 0
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError):
            parse_spectra_json(p)

    def test_schema_error_message_names_both_versions(self, tmp_path):
        """The error message must contain BOTH the expected and the
        actual version so the user can decide if they need to
        update or downgrade."""
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = 99
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError) as exc_info:
            parse_spectra_json(p)
        msg = str(exc_info.value)
        assert str(SCHEMA_VERSION) in msg
        assert "99" in msg


# --------------------------------------------------------------------- #
#  Field-level errors                                                   #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonFieldErrors:

    def test_missing_required_field_wrapped_with_path(self, tmp_path):
        """A required top-level field that's missing raises
        FieldError naming the field, not a raw KeyError."""
        payload = _make_minimal_results().to_dict()
        del payload["engine"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError) as exc_info:
            parse_spectra_json(p)
        assert "engine" in str(exc_info.value)

    def test_modes_with_bad_shape_raises_field_error(self, tmp_path):
        """ModeData.__post_init__ raises TypeError on wrong eigvec
        shape; the parser wraps it as FieldError."""
        payload = _make_minimal_results().to_dict()
        # Corrupt the eigenvector to wrong shape.
        payload["modes"][0]["eigenvector_free"] = [[0.7], [-0.7]]  # 2x1, not 2x3
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Forward-compatibility                                                #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonForwardCompat:
    """SpectraResults.from_dict ignores unknown keys by design
    (spec § 5 forward-compat rule).  Test that the parser inherits
    this -- new engines can add ``engine_metadata.foo`` keys
    without breaking older readers."""

    def test_extra_top_level_keys_ignored(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["future_field_added_in_v2"] = {"some": "data"}
        p = _write_json(tmp_path, payload)
        loaded = parse_spectra_json(p)
        # Parse succeeded, the extra key didn't surface as a field on
        # the typed dataclass (no auto-attach).
        assert loaded.engine == "pyscf"
        assert not hasattr(loaded, "future_field_added_in_v2")

    def test_extra_engine_metadata_keys_round_trip(self, tmp_path):
        """``engine_metadata`` is a free-form dict -- engines can
        stuff anything in there and it round-trips intact."""
        original = _make_minimal_results()
        # Stuff some engine-specific metadata in.
        d = original.to_dict()
        d["engine_metadata"] = {
            "pyscf_xc_grid_radial": 75,
            "custom_engine_flag":   True,
            "list_of_things":       [1, 2, 3],
        }
        p = _write_json(tmp_path, d)
        loaded = parse_spectra_json(p)
        assert loaded.engine_metadata["pyscf_xc_grid_radial"] == 75
        assert loaded.engine_metadata["custom_engine_flag"]   is True
        assert loaded.engine_metadata["list_of_things"]       == [1, 2, 3]


# --------------------------------------------------------------------- #
#  In-memory variant                                                    #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonDict:
    """parse_spectra_json_dict is the in-memory cousin used by the
    web /api/spectra/load endpoint when the JSON arrived over the
    wire as a Python dict (already decoded by Flask)."""

    def test_round_trip_dict(self):
        original = _make_minimal_results()
        loaded = parse_spectra_json_dict(original.to_dict())
        assert loaded.engine == "pyscf"
        assert len(loaded.modes) == 1

    def test_non_dict_rejected(self):
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_missing_schema_version_rejected(self):
        original = _make_minimal_results()
        d = original.to_dict()
        del d["schema_version"]
        with pytest.raises(SpectraJsonSchemaError):
            parse_spectra_json_dict(d)

    def test_field_error_wrapped(self):
        original = _make_minimal_results()
        d = original.to_dict()
        del d["engine"]
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json_dict(d)


# --------------------------------------------------------------------- #
#  Exception hierarchy                                                  #
# --------------------------------------------------------------------- #


class TestExceptionHierarchy:
    """The exception hierarchy is part of the public contract --
    callers (live-watch poller, web endpoint) need to be able to
    distinguish failure modes by type.  Pin the inheritance."""

    def test_specific_errors_inherit_base(self):
        for cls in (SpectraJsonNotFoundError,
                    SpectraJsonMalformedError,
                    SpectraJsonSchemaError,
                    SpectraJsonFieldError):
            assert issubclass(cls, SpectraJsonError)

    def test_not_found_also_inherits_filenotfound(self):
        """Legacy ``except FileNotFoundError`` blocks must keep
        catching missing-file errors."""
        assert issubclass(SpectraJsonNotFoundError, FileNotFoundError)

    def test_schema_error_carries_expected_and_actual_attrs(self):
        """The SchemaError carries the two version numbers as
        attributes so the web layer can render a structured "update
        molbuilder" response without parsing the message string."""
        err = SpectraJsonSchemaError(1, 2)
        assert err.expected == 1
        assert err.actual == 2


# --------------------------------------------------------------------- #
#  Type-strict schema version                                            #
# --------------------------------------------------------------------- #


class TestSchemaVersionTypeSafety:
    """``True == 1`` in Python because ``bool`` subclasses ``int``.
    A naive ``d['schema_version'] != 1`` check passes for ``True``,
    which is a quiet correctness hole (some other format using
    JSON could put a boolean there).  The parser uses isinstance
    to reject bool explicitly."""

    def test_bool_true_rejected_as_schema_version(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = True
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError) as exc_info:
            parse_spectra_json(p)
        assert exc_info.value.actual is True

    def test_string_schema_version_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = "1"  # string "1", not int
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError):
            parse_spectra_json(p)

    def test_float_schema_version_rejected(self, tmp_path):
        """``1.0`` matches ``1`` numerically but isn't an int -- the
        wire contract is integer schema versions only."""
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = 1.0
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Non-finite floats (NaN / Infinity)                                    #
# --------------------------------------------------------------------- #


class TestParseRejectsNonFinite:
    """Python's json.loads silently accepts the non-standard
    ``NaN``, ``Infinity``, ``-Infinity`` tokens.  Other consumers
    (browsers' JSON.parse, jq, RFC-8259 parsers) reject them.  The
    parser uses ``parse_constant`` to catch these at decode time
    so a divergent SCF surfaces as a MalformedError with a
    pointed message, not silent NaN propagation."""

    def test_nan_token_rejected(self, tmp_path):
        """Python json writes NaN as the literal token ``NaN``;
        we reject it on read."""
        p = tmp_path / "with_nan.spectra.json"
        # Hand-craft the JSON so we get a NaN token without using
        # Python's json.dumps (which we configure to reject NaN
        # in the writer path anyway).
        raw = (
            '{"schema_version": 1, "engine": "pyscf", '
            '"equilibrium_scf_eh": NaN}'
        )
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError) as exc_info:
            parse_spectra_json(p)
        assert "non-finite" in str(exc_info.value).lower() or \
               "nan" in str(exc_info.value).lower()

    def test_infinity_token_rejected(self, tmp_path):
        p = tmp_path / "with_inf.spectra.json"
        raw = (
            '{"schema_version": 1, "engine": "pyscf", '
            '"equilibrium_scf_eh": Infinity}'
        )
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_negative_infinity_token_rejected(self, tmp_path):
        p = tmp_path / "with_neginf.spectra.json"
        raw = (
            '{"schema_version": 1, "engine": "pyscf", '
            '"equilibrium_scf_eh": -Infinity}'
        )
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Encoding edge cases                                                  #
# --------------------------------------------------------------------- #


class TestParseEncodingTolerance:
    """The reader uses utf-8-sig so a BOM-prefixed file (some
    Windows editors insert one) is transparently stripped instead
    of poisoning the first byte of the JSON document."""

    def test_utf8_bom_tolerated(self, tmp_path):
        p = tmp_path / "bom.spectra.json"
        payload = _make_minimal_results().to_dict()
        # Write the file with an explicit BOM.
        body = json.dumps(payload)
        p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
        loaded = parse_spectra_json(p)
        assert loaded.engine == "pyscf"

    def test_utf8_special_chars_round_trip(self, tmp_path):
        """cm⁻¹ / Å characters in `methods_text` survive the
        utf-8 round-trip; ensure_ascii=False in the writer keeps
        them readable in the file (no \\uXXXX escapes)."""
        original = _make_minimal_results()
        original.methods_text = "Displacement = 0.10 Å; ω in cm⁻¹"
        p = tmp_path / "unicode.spectra.json"
        dump_spectra_json(original, p)
        # File contents should contain the literal Å / cm⁻¹ chars,
        # not \\uXXXX escapes, because ensure_ascii=False is set.
        raw = p.read_text(encoding="utf-8")
        assert "Å" in raw
        assert "cm⁻¹" in raw
        # Round-trip preserves the chars.
        loaded = parse_spectra_json(p)
        assert "Å" in loaded.methods_text
        assert "cm⁻¹" in loaded.methods_text

    def test_non_utf8_file_raises_malformed(self, tmp_path):
        """A file that isn't UTF-8 (e.g. Latin-1 with a high byte)
        is content-malformed, not a filesystem error -- the parser
        raises MalformedError so the caller can handle it
        uniformly with other content-corruption cases."""
        p = tmp_path / "latin1.spectra.json"
        # 0xFF is not valid as a UTF-8 starter byte; encodes fine
        # in Latin-1 but UTF-8 will reject.
        p.write_bytes(b"\xff\xfeinvalid utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  dump_spectra_json (writer)                                            #
# --------------------------------------------------------------------- #


class TestDumpSpectraJson:
    """The writer is the second half of the wire-format contract.
    It enforces the safety rules every Spectra engine has to
    follow when emitting JSON checkpoints."""

    def test_round_trip(self, tmp_path):
        original = _make_minimal_results()
        p = tmp_path / "out.spectra.json"
        dump_spectra_json(original, p)
        loaded = parse_spectra_json(p)
        assert loaded.engine == original.engine
        assert len(loaded.modes) == len(original.modes)

    def test_nan_in_scalar_field_rejected(self, tmp_path):
        """allow_nan=False means a NaN energy raises ValueError
        BEFORE the file is touched -- the engine has to filter
        non-finite values explicitly rather than silently emit
        junk JSON."""
        original = _make_minimal_results()
        original.equilibrium_scf_eh = float("nan")
        p = tmp_path / "out.spectra.json"
        with pytest.raises(ValueError):
            dump_spectra_json(original, p)
        # File should not have been created.
        assert not p.exists()

    def test_inf_in_array_rejected(self, tmp_path):
        """A non-finite value buried in an MO-energy array also
        trips the writer."""
        original = _make_minimal_results()
        original.equilibrium_mo_energies_eh[0] = np.inf
        p = tmp_path / "out.spectra.json"
        with pytest.raises(ValueError):
            dump_spectra_json(original, p)
        assert not p.exists()

    def test_atomic_replace_no_torn_file_on_failure(self, tmp_path):
        """If the writer raises mid-write, the destination path is
        either absent (fresh write) or still holds the OLD content
        (overwrite case) -- never a half-written temp file
        masquerading as the real one."""
        # Seed an existing file with known content.
        p = tmp_path / "existing.spectra.json"
        good = _make_minimal_results()
        dump_spectra_json(good, p)
        original_bytes = p.read_bytes()

        # Now attempt to overwrite with a non-finite payload -- it
        # must fail BEFORE touching the destination.
        bad = _make_minimal_results()
        bad.equilibrium_scf_eh = float("inf")
        with pytest.raises(ValueError):
            dump_spectra_json(bad, p)

        # Old content is intact.
        assert p.read_bytes() == original_bytes
        # No temp file dangling next to it.
        siblings = list(tmp_path.iterdir())
        assert siblings == [p], f"unexpected temp files: {siblings}"

    def test_no_bom_in_output(self, tmp_path):
        """The writer never emits a BOM, even though the reader
        tolerates one on input.  Symmetric tolerance + strict
        emission is the convention."""
        original = _make_minimal_results()
        p = tmp_path / "no_bom.spectra.json"
        dump_spectra_json(original, p)
        first_three = p.read_bytes()[:3]
        assert first_three != b"\xef\xbb\xbf"
        # And the first char is actually the JSON opening brace.
        assert p.read_bytes()[:1] == b"{"

    def test_indent_zero_compact_form(self, tmp_path):
        """indent=0 gives the compact wire form -- useful when the
        file is large and human-readability is less important."""
        original = _make_minimal_results()
        p = tmp_path / "compact.spectra.json"
        dump_spectra_json(original, p, indent=0)
        # Compact form has no two-space indentation on field names.
        raw = p.read_text(encoding="utf-8")
        assert '\n  "engine"' not in raw

    def test_pathlike_accepted(self, tmp_path):
        original = _make_minimal_results()
        # pathlib.Path is os.PathLike.
        dump_spectra_json(original, tmp_path / "x.json")
        assert (tmp_path / "x.json").exists()
