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
    ModeElectronicStructure,
    SpectraResults,
)
from molbuilder.spectra.results import (
    SCHEMA_VERSION,
    PHASE_COMPLETE,
    PHASE_EMPTY,
)
from molbuilder.sidecars.spectra import (
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
        frozen_atom_idxs            = [],
        equilibrium_scf_eh         = -76.4123,
        equilibrium_mo_energies_eh = np.array([-1.0, -0.5, -0.2, 0.1, 0.3]),
        equilibrium_homo_idx       = 2,
        modes                      = [
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 412.3,
                raman_activity_a4_amu = 12.5,
                ir_intensity_km_mol   = None,
                eigenvector_canonical = np.array([[0.7, 0.0, 0.0],
                                                  [-0.7, 0.0, 0.0]]),
                eigenvector_display   = np.array([[0.7, 0.0, 0.0],
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
        """A future schema_version (e.g. v4, written by a newer
        molbuilder) is rejected here; users who hit this get an
        "update molbuilder" message."""
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
        """ModeData.__post_init__ raises ValueError on wrong eigvec
        shape; the parser wraps it as FieldError.  Corrupt all three
        eigenvector fields (canonical + display + legacy) since the
        dataclass validates each independently."""
        payload = _make_minimal_results().to_dict()
        bad = [[0.7], [-0.7]]  # 2x1, not 2x3
        payload["modes"][0]["eigenvector_canonical"] = bad
        payload["modes"][0]["eigenvector_display"]         = bad
        payload["modes"][0]["eigenvector_free"]                          = bad
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
            '{"schema_version": 4, "engine": "pyscf", '
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
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium_scf_eh": Infinity}'
        )
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_negative_infinity_token_rejected(self, tmp_path):
        p = tmp_path / "with_neginf.spectra.json"
        raw = (
            '{"schema_version": 4, "engine": "pyscf", '
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


# --------------------------------------------------------------------- #
#  Optional / null fields                                               #
#                                                                       #
#  Robustness against the wire shapes the engine ACTUALLY writes:       #
#  compute_raman=False -> raman_activity_a4_amu=null on every mode;     #
#  selector=none -> every mode has electronic_structure=null;           #
#  in-progress writes -> modes=[] until L2 finishes (spec § 6.1).       #
# --------------------------------------------------------------------- #


class TestOptionalNullFields:

    def _build_with_modes(self, modes, **overrides) -> SpectraResults:
        """Helper: build a SpectraResults with custom modes."""
        results = _make_minimal_results()
        results.modes = list(modes)
        for k, v in overrides.items():
            setattr(results, k, v)
        return results

    def test_raman_activity_null_round_trips(self, tmp_path):
        """compute_raman=False produces modes with
        raman_activity_a4_amu=None; the wire form is JSON null."""
        mode = ModeData(
            index_1based          = 1,
            frequency_cm1         = 500.0,
            raman_activity_a4_amu = None,   # compute_raman=False path
            ir_intensity_km_mol   = None,
            eigenvector_canonical = np.array([[0.7, 0.0, 0.0],
                                              [-0.7, 0.0, 0.0]]),
            eigenvector_display   = np.array([[0.7, 0.0, 0.0],
                                              [-0.7, 0.0, 0.0]]),
            has_imag              = False,
        )
        results = self._build_with_modes([mode])
        p = tmp_path / "no_raman.spectra.json"
        dump_spectra_json(results, p)
        # The JSON file actually has the literal null token.
        raw = p.read_text(encoding="utf-8")
        assert '"raman_activity_a4_amu": null' in raw
        loaded = parse_spectra_json(p)
        assert loaded.modes[0].raman_activity_a4_amu is None

    def test_ir_intensity_null_round_trips(self, tmp_path):
        """ir_intensity_km_mol is always None in v1 (reserved for
        v1.2 IR add-on)."""
        results = _make_minimal_results()
        p = tmp_path / "ir_null.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        assert all(m.ir_intensity_km_mol is None for m in loaded.modes)

    def test_electronic_structure_null_round_trips(self, tmp_path):
        """When selector=none (or a mode wasn't picked) the mode
        has electronic_structure=None.  Wire form is the literal
        null at that key."""
        results = _make_minimal_results()
        p = tmp_path / "no_es.spectra.json"
        dump_spectra_json(results, p)
        raw = p.read_text(encoding="utf-8")
        assert '"electronic_structure": null' in raw
        loaded = parse_spectra_json(p)
        assert loaded.modes[0].electronic_structure is None


class TestImaginaryModeRoundTrip:
    """Saddle-point / spurious modes show up as negative
    frequencies with has_imag=True (spec § 5).  The wire shape
    must preserve sign + flag faithfully."""

    def test_negative_frequency_with_has_imag(self, tmp_path):
        from molbuilder.spectra.results import PHASE_COMPLETE, SCHEMA_VERSION
        imag_mode = ModeData(
            index_1based          = 1,
            frequency_cm1         = -150.5,
            raman_activity_a4_amu = 0.0,
            ir_intensity_km_mol   = None,
            eigenvector_canonical = np.array([[0.5, 0.5, 0.0],
                                              [-0.5, -0.5, 0.0]]),
            eigenvector_display   = np.array([[0.5, 0.5, 0.0],
                                              [-0.5, -0.5, 0.0]]),
            has_imag              = True,
        )
        # Mix one imaginary + one real mode so the parser handles
        # both in the same file.
        real_mode = ModeData(
            index_1based          = 2,
            frequency_cm1         = 800.3,
            raman_activity_a4_amu = 5.0,
            ir_intensity_km_mol   = None,
            eigenvector_canonical = np.array([[0.0, 1.0, 0.0],
                                              [0.0, -1.0, 0.0]]),
            eigenvector_display   = np.array([[0.0, 1.0, 0.0],
                                              [0.0, -1.0, 0.0]]),
            has_imag              = False,
        )
        results = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 2,
            free_atom_idxs             = [0, 1],
            frozen_atom_idxs            = [],
            equilibrium_scf_eh         = -76.0,
            equilibrium_mo_energies_eh = np.array([-1.0, 0.0, 1.0]),
            equilibrium_homo_idx       = 1,
            modes                      = [imag_mode, real_mode],
            selected_mode_idxs_1based  = [],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
            phase_frequencies          = PHASE_COMPLETE,
            phase_raman                = PHASE_COMPLETE,
            phase_es                   = PHASE_EMPTY,
        )
        p = tmp_path / "imag.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        # Imaginary mode preserved with NEGATIVE frequency + flag.
        m1 = loaded.modes[0]
        assert m1.frequency_cm1 == pytest.approx(-150.5)
        assert m1.has_imag is True
        # Real mode unchanged.
        m2 = loaded.modes[1]
        assert m2.frequency_cm1 == pytest.approx(800.3)
        assert m2.has_imag is False


class TestEmptyModesList:
    """In-progress wire state per spec § 6.1: between phase
    Setup-complete and L2-complete the file can carry an empty
    modes list with phase_frequencies=running.  Parser must
    accept this without barfing."""

    def test_empty_modes_in_progress_state(self, tmp_path):
        from molbuilder.spectra.results import PHASE_RUNNING, SCHEMA_VERSION
        results = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 2,
            free_atom_idxs             = [0, 1],
            frozen_atom_idxs            = [],
            equilibrium_scf_eh         = -76.0,
            equilibrium_mo_energies_eh = np.array([-1.0, 0.0]),
            equilibrium_homo_idx       = 0,
            modes                      = [],   # <-- pre-L2
            selected_mode_idxs_1based  = [],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
            phase_frequencies          = PHASE_RUNNING,
            phase_raman                = PHASE_EMPTY,
            phase_es                   = PHASE_EMPTY,
        )
        p = tmp_path / "in_progress.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        assert loaded.modes == []
        assert loaded.phase_frequencies == PHASE_RUNNING


# --------------------------------------------------------------------- #
#  Cross-mode invariants (dataclass __post_init__ via parser)           #
# --------------------------------------------------------------------- #


class TestCrossModeInvariants:
    """SpectraResults.__post_init__ enforces several cross-field
    invariants that the parser must surface as FieldError when the
    on-disk file violates them.  These are NOT just construction-
    time bugs -- a hand-edited or version-skewed file could carry
    inconsistent shapes that the parser layer is the last line of
    defence against."""

    def test_free_fixed_overlap_surfaces_as_field_error(self, tmp_path):
        """free_atom_idxs ∩ frozen_atom_idxs must be empty."""
        payload = _make_minimal_results().to_dict()
        payload["free_atom_idxs"]  = [0, 1]
        payload["frozen_atom_idxs"] = [1]   # overlap on atom 1
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError) as exc_info:
            parse_spectra_json(p)
        assert "overlap" in str(exc_info.value).lower()

    def test_free_plus_fixed_mismatched_count(self, tmp_path):
        """len(free) + len(fixed) must == n_atoms_total."""
        payload = _make_minimal_results().to_dict()
        payload["n_atoms_total"]   = 5
        payload["free_atom_idxs"]  = [0, 1]   # only 2
        payload["frozen_atom_idxs"] = []       # plus 0 -> != 5
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_homo_idx_out_of_range_field_error(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        # 5 MO energies in the fixture array -> valid range [0, 5)
        payload["equilibrium"]["homo_idx"] = 99
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError) as exc_info:
            parse_spectra_json(p)
        assert "homo_idx" in str(exc_info.value)

    def test_cross_mode_es_window_mismatch(self, tmp_path):
        """Per spec § 5, every mode's ES window must have the same
        orbital count -- the engine writes the same n_window for
        all selected modes.  A file with mismatched windows is
        corrupted and the parser refuses it."""
        # Build a results with TWO modes, both with ES, but
        # different window sizes -- the dataclass should reject.
        original = _make_minimal_results()
        es_3 = ModeElectronicStructure(
            amplitude_ang        = 0.1,
            mo_energies_eq_eh    = np.array([-1.0, -0.5, 0.5]),    # 3
            mo_energies_minus_eh = np.array([-1.01, -0.51, 0.49]),
            mo_energies_plus_eh  = np.array([-0.99, -0.49, 0.51]),
            homo_index_in_window = 1,
            scf_energy_eq_eh     = -76.0,
            scf_energy_minus_eh  = -76.0,
            scf_energy_plus_eh   = -76.0,
        )
        original.modes[0].electronic_structure = es_3
        # Add a second mode whose ES has DIFFERENT window size:
        mode_2 = ModeData(
            index_1based          = 2,
            frequency_cm1         = 1500.0,
            raman_activity_a4_amu = 7.0,
            ir_intensity_km_mol   = None,
            eigenvector_canonical = np.array([[0.5, 0., 0.],
                                              [-0.5, 0., 0.]]),
            eigenvector_display   = np.array([[0.5, 0., 0.],
                                              [-0.5, 0., 0.]]),
            has_imag              = False,
            electronic_structure  = ModeElectronicStructure(
                amplitude_ang        = 0.1,
                mo_energies_eq_eh    = np.array([-1., -0.5, 0., 0.5, 1.]),  # 5
                mo_energies_minus_eh = np.array([-1.01, -0.51, -0.01, 0.49, 0.99]),
                mo_energies_plus_eh  = np.array([-0.99, -0.49, 0.01, 0.51, 1.01]),
                homo_index_in_window = 2,
                scf_energy_eq_eh     = -76.0,
                scf_energy_minus_eh  = -76.0,
                scf_energy_plus_eh   = -76.0,
            ),
        )
        # We need to write this to JSON BYPASSING the dataclass
        # post-init (because that's the rule we're trying to test
        # at the parser level).  Build the dict by hand.
        d = original.to_dict()
        d["modes"].append(mode_2.to_dict())
        p = _write_json(tmp_path, d)
        with pytest.raises(SpectraJsonFieldError) as exc_info:
            parse_spectra_json(p)
        assert "window" in str(exc_info.value).lower() or \
               "size" in str(exc_info.value).lower()

    def test_eigenvector_shape_mismatch_against_n_free(self, tmp_path):
        """An eigenvector with the wrong (n_free, 3) shape relative
        to the global free_atom_idxs is rejected by the parser."""
        payload = _make_minimal_results().to_dict()
        # fixture has n_free=2; corrupt one eigvec to have 3 rows.
        # Corrupt the canonical field (the science-authoritative one);
        # the parser's shape check is what we're pinning here.
        bad = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0]]
        payload["modes"][0]["eigenvector_canonical"] = bad
        payload["modes"][0]["eigenvector_display"]         = bad
        payload["modes"][0]["eigenvector_free"]                          = bad
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Required-but-null fields (different from "optional null")            #
# --------------------------------------------------------------------- #


class TestNullForRequiredFields:
    """JSON null in a place where the wire shape REQUIRES a number
    (e.g. equilibrium.scf_energy_eh) must surface as FieldError --
    we don't want a NaN-shaped failure 100 lines downstream when
    the typed dataclass tries to math with None."""

    def test_null_scf_energy_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["equilibrium"]["scf_energy_eh"] = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_null_n_atoms_total_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["n_atoms_total"] = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_null_mode_frequency_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["modes"][0]["frequency_cm1"] = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_null_eigenvector_rejected(self, tmp_path):
        """The eigenvector fields are mandatory -- not Optional like
        raman_activity_a4_amu.  Test all three to pin that nulling
        ANY of them is rejected (canonical, display, or legacy alias)."""
        payload = _make_minimal_results().to_dict()
        payload["modes"][0]["eigenvector_canonical"] = None
        payload["modes"][0]["eigenvector_display"]         = None
        payload["modes"][0]["eigenvector_free"]                          = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Nested-field errors                                                  #
# --------------------------------------------------------------------- #


class TestNestedFieldErrors:
    """Errors arising from missing or wrong-shape fields inside
    nested sub-dicts (``equilibrium``, individual mode dicts,
    individual ES dicts) must surface as FieldError so the user
    can locate them."""

    def test_missing_equilibrium_block(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_nested_scf_energy(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]["scf_energy_eh"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_nested_mo_energies(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]["mo_energies_eh"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_mode_required_field(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        del payload["modes"][0]["frequency_cm1"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_es_sub_field(self, tmp_path):
        """An ES block missing one of its required arrays should
        also surface as FieldError, not a raw KeyError."""
        # Build a SpectraResults with an ES block, then drop a
        # required ES field from the wire-form before re-loading.
        results = _make_minimal_results()
        results.modes[0].electronic_structure = ModeElectronicStructure(
            amplitude_ang        = 0.1,
            mo_energies_eq_eh    = np.array([-1.0, -0.5, 0.5]),
            mo_energies_minus_eh = np.array([-1.01, -0.51, 0.49]),
            mo_energies_plus_eh  = np.array([-0.99, -0.49, 0.51]),
            homo_index_in_window = 1,
            scf_energy_eq_eh     = -76.0,
            scf_energy_minus_eh  = -76.0,
            scf_energy_plus_eh   = -76.0,
        )
        payload = results.to_dict()
        del payload["modes"][0]["electronic_structure"]["scf_energy_eq_eh"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Phase status                                                         #
# --------------------------------------------------------------------- #


class TestPhaseStatusValidation:
    """Phase status strings are constrained to {empty, running,
    complete} (spec § 5).  An invalid value comes through the
    parser as a FieldError because the dataclass __post_init__
    rejects it."""

    def test_invalid_phase_string_rejected(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        payload["phase_frequencies"] = "halfway"  # not a valid state
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_phase_defaults_to_empty(self, tmp_path):
        """SpectraResults.from_dict uses .get(..., PHASE_EMPTY) so a
        legacy file that doesn't have the per-phase fields loads as
        all-empty rather than failing -- backward compat."""
        payload = _make_minimal_results().to_dict()
        del payload["phase_frequencies"]
        del payload["phase_raman"]
        del payload["phase_es"]
        p = _write_json(tmp_path, payload)
        loaded = parse_spectra_json(p)
        assert loaded.phase_frequencies == PHASE_EMPTY
        assert loaded.phase_raman       == PHASE_EMPTY
        assert loaded.phase_es          == PHASE_EMPTY


# --------------------------------------------------------------------- #
#  Numeric precision                                                    #
# --------------------------------------------------------------------- #


class TestNumericPrecisionRoundTrip:
    """JSON's repr() encoding of IEEE 754 doubles gives full
    round-trip precision (Python uses the shortest repr that
    uniquely identifies the float).  Pin that we don't lose
    significant figures in the round trip."""

    def test_high_precision_scf_energy_round_trip(self, tmp_path):
        """SCF energies converge to 1e-9 Hartree; the wire format
        must preserve that."""
        results = _make_minimal_results()
        results.equilibrium_scf_eh = -76.41234567890123
        p = tmp_path / "precision.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        # Bit-exact float round-trip (not pytest.approx -- we want
        # to detect any precision loss).
        assert loaded.equilibrium_scf_eh == results.equilibrium_scf_eh

    def test_very_small_and_very_large_floats(self, tmp_path):
        results = _make_minimal_results()
        # Subnormal-ish + a huge value, both finite.
        results.equilibrium_mo_energies_eh = np.array([
            -1.234567890123456e-10,
            0.0,
            +9.876543210987654e+5,
            -1.0,
            +1.0,
        ])
        results.equilibrium_homo_idx = 2
        p = tmp_path / "extremes.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        np.testing.assert_array_equal(
            loaded.equilibrium_mo_energies_eh,
            results.equilibrium_mo_energies_eh,
        )

    def test_negative_zero_round_trip(self, tmp_path):
        """``-0.0 == 0.0`` is True but they're distinct floats; we
        don't claim to preserve sign of zero, but the value should
        at least round-trip equal."""
        results = _make_minimal_results()
        results.equilibrium_scf_eh = -0.0
        p = tmp_path / "negzero.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_scf_eh == 0.0


# --------------------------------------------------------------------- #
#  Filesystem edge cases                                                #
# --------------------------------------------------------------------- #


class TestFilesystemEdgeCases:

    def test_directory_path_raises_oserror_branch(self, tmp_path):
        """Passing a directory where a file is expected should
        raise -- some flavor of OSError-shaped SpectraJsonError
        (the existence check passes, the open() inside fails)."""
        # tmp_path is a directory.  os.path.exists(dir) -> True so
        # we skip the NotFoundError branch and hit the read step.
        with pytest.raises((SpectraJsonError, IsADirectoryError)):
            parse_spectra_json(tmp_path)

    def test_json_with_javascript_comments_rejected(self, tmp_path):
        """Standard JSON doesn't permit // or /* */ comments.  An
        editor that inserts them produces an invalid file -- the
        parser must reject it with MalformedError, not silently
        accept."""
        p = tmp_path / "commented.spectra.json"
        p.write_text(
            '{ // a comment\n'
            '  "schema_version": 1\n'
            '}',
            encoding="utf-8",
        )
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_modes_not_a_list_rejected(self, tmp_path):
        """If `modes` is an object instead of a list, the
        reconstitution iterates over its keys and produces noise.
        Surface as FieldError."""
        payload = _make_minimal_results().to_dict()
        payload["modes"] = {"weird": "shape"}
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)


# --------------------------------------------------------------------- #
#  Round-trip stress: an "everything" result                            #
# --------------------------------------------------------------------- #


class TestGeometryRoundTrip:
    """The optional ``equilibrium.elements`` + ``equilibrium.
    positions_ang`` fields (added late in the schema) round-trip
    cleanly and remain backward-compatible: older JSON without
    these keys still loads."""

    def test_geometry_round_trips(self, tmp_path):
        results = _make_minimal_results()
        results.equilibrium_elements = ["O", "H"]
        results.equilibrium_positions_ang = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
        ])
        # Re-run __post_init__ via the constructor since we
        # mutated fields directly.
        results.__post_init__()
        p = tmp_path / "geom.spectra.json"
        dump_spectra_json(results, p)
        # Wire form has the new keys under equilibrium.
        raw = json.loads(p.read_text())
        assert "elements"      in raw["equilibrium"]
        assert "positions_ang" in raw["equilibrium"]
        # Round-trip.
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_elements == ["O", "H"]
        np.testing.assert_allclose(
            loaded.equilibrium_positions_ang,
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        )

    def test_geometry_omitted_back_compat(self, tmp_path):
        """A spectra.json without the geometry keys still parses --
        the optional fields fall back to None on the typed side."""
        results = _make_minimal_results()
        # Explicitly leave equilibrium_elements / positions_ang as
        # None (default).
        assert results.equilibrium_elements      is None
        assert results.equilibrium_positions_ang is None
        p = tmp_path / "no-geom.spectra.json"
        dump_spectra_json(results, p)
        raw = json.loads(p.read_text())
        # Keys not in the wire form when None.
        assert "elements"      not in raw["equilibrium"]
        assert "positions_ang" not in raw["equilibrium"]
        # And the parser handles missing keys cleanly.
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_elements      is None
        assert loaded.equilibrium_positions_ang is None

    def test_geometry_partial_rejected(self):
        """Elements without positions (or vice versa) is incoherent
        -- reject at __post_init__."""
        with pytest.raises(ValueError, match="must be supplied together"):
            SpectraResults(
                schema_version=SCHEMA_VERSION,
                engine="pyscf", engine_version="x",
                molbuilder_version="y", timestamp="t",
                structure_hash="h", n_atoms_total=1,
                free_atom_idxs=[0], frozen_atom_idxs=[],
                equilibrium_scf_eh=-1.0,
                equilibrium_mo_energies_eh=np.zeros(3),
                equilibrium_homo_idx=0,
                modes=[], selected_mode_idxs_1based=[],
                config={}, methods_text="", bibliography_keys=[],
                equilibrium_elements=["O"],     # but no positions
            )


class TestComprehensiveRoundTrip:
    """A single file with the full feature set: imaginary modes,
    selected + unselected modes, populated config + engine_metadata,
    long methods_text, unicode chars, scientific-notation energies.
    The whole shebang round-trips bit-exact (within float repr)."""

    def test_full_feature_set_round_trip(self, tmp_path):
        from molbuilder.spectra.results import PHASE_COMPLETE, SCHEMA_VERSION

        es = ModeElectronicStructure(
            amplitude_ang        = 0.10,
            mo_energies_eq_eh    = np.array([-1.234e-2, -5.678e-3, 0.0,
                                             1.111e-3, 2.222e-3]),
            mo_energies_minus_eh = np.array([-1.235e-2, -5.679e-3, -1e-9,
                                             1.110e-3, 2.221e-3]),
            mo_energies_plus_eh  = np.array([-1.233e-2, -5.677e-3, 1e-9,
                                             1.112e-3, 2.223e-3]),
            homo_index_in_window = 2,
            scf_energy_eq_eh     = -76.41234567890123,
            scf_energy_minus_eh  = -76.41234567880123,
            scf_energy_plus_eh   = -76.41234567900123,
        )
        modes = [
            ModeData(  # imaginary
                index_1based=1, frequency_cm1=-120.5,
                raman_activity_a4_amu=0.0, ir_intensity_km_mol=None,
                eigenvector_canonical = np.array([[0.7, 0., 0.], [-0.7, 0., 0.]]),
                eigenvector_display   = np.array([[0.7, 0., 0.], [-0.7, 0., 0.]]),
                has_imag=True,
            ),
            ModeData(  # selected for ES
                index_1based=2, frequency_cm1=1023.4,
                raman_activity_a4_amu=87.2, ir_intensity_km_mol=None,
                eigenvector_canonical = np.array([[0., 0.7, 0.], [0., -0.7, 0.]]),
                eigenvector_display   = np.array([[0., 0.7, 0.], [0., -0.7, 0.]]),
                has_imag=False,
                electronic_structure=es,
            ),
            ModeData(  # not selected (no ES) + no Raman activity
                index_1based=3, frequency_cm1=3656.0,
                raman_activity_a4_amu=None, ir_intensity_km_mol=None,
                eigenvector_canonical = np.array([[0., 0., 0.7], [0., 0., -0.7]]),
                eigenvector_display   = np.array([[0., 0., 0.7], [0., 0., -0.7]]),
                has_imag=False,
            ),
        ]
        original = SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc123",
            n_atoms_total              = 2,
            free_atom_idxs             = [0, 1],
            frozen_atom_idxs            = [],
            equilibrium_scf_eh         = -76.41234567890123,
            equilibrium_mo_energies_eh = np.array([
                -1.234567e-1, -2.345678e-2, 0.0, 1.0e-3, 2.0e-3,
            ]),
            equilibrium_homo_idx       = 2,
            modes                      = modes,
            selected_mode_idxs_1based  = [2],
            config                     = {
                "engine":      "pyscf",
                "functional":  "B3LYP",
                "basis":       "def2-SVP",
                "dispersion":  "d3bj",
                "nested":      {"foo": "bar", "list": [1, 2, 3]},
            },
            methods_text               = (
                "Vibrational analysis at the B3LYP/def2-SVP level "
                "with D3BJ dispersion (displacement amplitude 0.10 Å; "
                "frequencies reported in cm⁻¹)."
            ),
            bibliography_keys          = [
                "Sun2020", "Sun2018", "Becke1993", "Grimme2011",
                "Mills1972", "Galperin2007",
            ],
            phase_frequencies          = PHASE_COMPLETE,
            phase_raman                = PHASE_COMPLETE,
            phase_es                   = PHASE_COMPLETE,
            engine_metadata            = {
                "pyscf_grid_radial": 75,
                "pyscf_grid_angular": 302,
                "wall_time_seconds": 1234.567,
            },
        )

        p = tmp_path / "full.spectra.json"
        dump_spectra_json(original, p)
        loaded = parse_spectra_json(p)

        # Field-by-field comparison (loud equality on the dataclass
        # itself is intentional -- we have to assert per-field).
        assert loaded.engine                  == original.engine
        assert loaded.engine_version          == original.engine_version
        assert loaded.molbuilder_version      == original.molbuilder_version
        assert loaded.timestamp               == original.timestamp
        assert loaded.structure_hash          == original.structure_hash
        assert loaded.n_atoms_total           == original.n_atoms_total
        assert loaded.free_atom_idxs          == original.free_atom_idxs
        assert loaded.frozen_atom_idxs         == original.frozen_atom_idxs
        # Floats: exact round-trip.
        assert loaded.equilibrium_scf_eh      == original.equilibrium_scf_eh
        np.testing.assert_array_equal(
            loaded.equilibrium_mo_energies_eh,
            original.equilibrium_mo_energies_eh,
        )
        assert loaded.equilibrium_homo_idx    == original.equilibrium_homo_idx
        # Modes: count + per-mode key fields.
        assert len(loaded.modes) == 3
        m1, m2, m3 = loaded.modes
        assert m1.frequency_cm1 == -120.5 and m1.has_imag is True
        assert m2.electronic_structure is not None
        assert m2.electronic_structure.amplitude_ang == pytest.approx(0.10)
        np.testing.assert_array_equal(
            m2.electronic_structure.mo_energies_eq_eh,
            original.modes[1].electronic_structure.mo_energies_eq_eh,
        )
        assert m3.raman_activity_a4_amu is None
        assert m3.electronic_structure is None
        # Selected modes list.
        assert loaded.selected_mode_idxs_1based == [2]
        # Config (nested dict round-trip).
        assert loaded.config["nested"]["list"] == [1, 2, 3]
        # Methods text + bibliography keys.
        assert "B3LYP" in loaded.methods_text
        assert "cm⁻¹" in loaded.methods_text
        assert loaded.bibliography_keys == original.bibliography_keys
        # Phase flags.
        assert loaded.phase_frequencies == PHASE_COMPLETE
        assert loaded.phase_raman       == PHASE_COMPLETE
        assert loaded.phase_es          == PHASE_COMPLETE
        # Engine metadata (mixed types).
        assert loaded.engine_metadata["pyscf_grid_angular"] == 302
        assert loaded.engine_metadata["wall_time_seconds"] == pytest.approx(1234.567)


# --------------------------------------------------------------------- #
#  Numeric format edge cases                                            #
#                                                                       #
#  JSON accepts decimal and scientific-notation numbers; it rejects     #
#  Fortran D-exponent, hex floats, and symbolic NaN/Inf tokens.  In     #
#  addition, valid-syntax numbers like "1e500" silently overflow to     #
#  ``float('inf')`` in stock json -- the parser must catch that too.    #
# --------------------------------------------------------------------- #


class TestNumericFormats:
    """Robustness against the numeric-literal flavors that engines
    or hand-edited files might produce."""

    def test_scientific_notation_lowercase_e(self, tmp_path):
        payload = _make_minimal_results().to_dict()
        # Replace SCF energy with a scientific-notation literal.
        # We can't easily inject the textual literal via to_dict
        # (Python json picks the form on emit), so write the JSON
        # by hand for this case.
        payload["equilibrium"]["scf_energy_eh"] = -1.5e-10
        p = _write_json(tmp_path, payload)
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_scf_eh == pytest.approx(-1.5e-10)

    def test_scientific_notation_uppercase_E(self, tmp_path):
        """JSON allows both ``1e10`` and ``1E10`` -- our reader
        accepts both indifferently."""
        # Hand-craft the JSON with an uppercase-E literal.
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"engine_version": "x", "molbuilder_version": "y", '
            '"timestamp": "t", "structure_hash": "h", '
            '"n_atoms_total": 2, "free_atom_idxs": [0, 1], '
            '"frozen_atom_idxs": [], '
            '"equilibrium": {"scf_energy_eh": -1.5E-10, '
            '  "mo_energies_eh": [-1.0, 0.5], "homo_idx": 0}, '
            '"modes": [], '
            '"selected_mode_idxs_1based": [], "config": {}, '
            '"methods_text": "", "bibliography_keys": [], '
            '"phase_frequencies": "empty", "phase_raman": "empty", '
            '"phase_es": "empty", "engine_metadata": {}}'
        )
        p = tmp_path / "uppercase_e.spectra.json"
        p.write_text(body, encoding="utf-8")
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_scf_eh == pytest.approx(-1.5e-10)

    def test_integer_for_float_field_accepted(self, tmp_path):
        """A bare integer (``-76``) where a float is expected works
        -- ``float(-76)`` is clean.  JSON only has one numeric type
        and the engine may emit either form."""
        payload = _make_minimal_results().to_dict()
        payload["equilibrium"]["scf_energy_eh"] = -76    # int on the wire
        p = _write_json(tmp_path, payload)
        loaded = parse_spectra_json(p)
        assert loaded.equilibrium_scf_eh == -76.0

    def test_float_for_int_field_accepted_when_whole(self, tmp_path):
        """``42.0`` where an int field is expected loads cleanly --
        Python's ``int(42.0)`` succeeds.  JSON doesn't distinguish
        int from float, so we have to be lenient on this direction."""
        payload = _make_minimal_results().to_dict()
        payload["n_atoms_total"]   = 2.0   # float on the wire
        p = _write_json(tmp_path, payload)
        loaded = parse_spectra_json(p)
        assert loaded.n_atoms_total == 2

    def test_overflow_literal_rejected(self, tmp_path):
        """``1e500`` decodes to ``float('inf')`` in stock json
        silently.  parse_float catches this and surfaces as
        MalformedError."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": 1e500}}'
        )
        p = tmp_path / "overflow.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError) as exc_info:
            parse_spectra_json(p)
        msg = str(exc_info.value).lower()
        assert "overflow" in msg or "non-finite" in msg

    def test_negative_overflow_literal_rejected(self, tmp_path):
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": -1e500}}'
        )
        p = tmp_path / "negoverflow.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_overflow_inside_array_rejected(self, tmp_path):
        """Overflow inside an MO-energies array is also caught --
        the parse_float hook runs for every float literal in the
        document, not just top-level scalars."""
        # We need to inject the overflow into the JSON text since
        # Python can't represent it as a finite literal in source.
        # Take a valid payload, dump it, then patch in 1e500 in
        # the mo_energies_eh array.
        results = _make_minimal_results()
        d = results.to_dict()
        # Stomp in the textual literal directly.
        raw = json.dumps(d)
        raw = raw.replace(
            '"mo_energies_eh": [-1.0, -0.5, -0.2, 0.1, 0.3]',
            '"mo_energies_eh": [-1.0, 1e500, -0.2, 0.1, 0.3]',
        )
        p = tmp_path / "overflow_in_array.spectra.json"
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_underflow_to_zero_accepted(self, tmp_path):
        """``1e-500`` underflows to ``0.0`` -- mathematically zero,
        a valid finite IEEE 754 value.  The parser accepts it."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"engine_version": "x", "molbuilder_version": "y", '
            '"timestamp": "t", "structure_hash": "h", '
            '"n_atoms_total": 2, "free_atom_idxs": [0, 1], '
            '"frozen_atom_idxs": [], '
            '"equilibrium": {"scf_energy_eh": 1e-500, '
            '  "mo_energies_eh": [-1.0, 0.5], "homo_idx": 0}, '
            '"modes": [], '
            '"selected_mode_idxs_1based": [], "config": {}, '
            '"methods_text": "", "bibliography_keys": [], '
            '"phase_frequencies": "empty", "phase_raman": "empty", '
            '"phase_es": "empty", "engine_metadata": {}}'
        )
        p = tmp_path / "underflow.spectra.json"
        p.write_text(body, encoding="utf-8")
        loaded = parse_spectra_json(p)
        # Underflow snapped to 0; the field is still finite.
        assert loaded.equilibrium_scf_eh == 0.0

    def test_fortran_d_exponent_rejected(self, tmp_path):
        """Fortran double-precision literal style ``1.5d10`` is not
        valid JSON; the JSON lexer raises JSONDecodeError, which
        we surface as MalformedError.

        Pinning this so we don't accidentally loosen the parser
        later (some users hand-edit values from SIESTA / Fortran
        output and would expect them to load -- they shouldn't:
        the wire format is JSON, not free-form scientific text)."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": -76.4d-1}}'
        )
        p = tmp_path / "fortran.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_hex_float_rejected(self, tmp_path):
        """C99-style hex floats (``0x1.fp10``) aren't JSON either."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": 0x1.fp10}}'
        )
        p = tmp_path / "hexfloat.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_leading_plus_sign_rejected(self, tmp_path):
        """JSON forbids a leading ``+`` on numbers (``+1.5`` is
        invalid).  The lexer will catch this."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": +1.5}}'
        )
        p = tmp_path / "leading_plus.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_leading_decimal_point_rejected(self, tmp_path):
        """``.5`` (no leading zero) isn't valid JSON either."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": .5}}'
        )
        p = tmp_path / "leading_dot.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_trailing_decimal_point_rejected(self, tmp_path):
        """``5.`` (trailing dot, no fractional digits) isn't
        valid JSON."""
        body = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium": {"scf_energy_eh": 5.}}'
        )
        p = tmp_path / "trailing_dot.spectra.json"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)


class TestComplexNumbersNotInWireFormat:
    """JSON has no native complex-number type.  Our v1 result
    surface is all real-valued (MO energies, SCF energies, Raman
    activities, frequencies, eigenvectors).  If a future engine
    needs complex (e.g. resonance-Raman polarizability), we'd
    encode as ``[re, im]`` or ``{"re":..., "im":...}`` -- but in
    v1 a complex value reaching any typed field is a bug.

    These tests pin the v1 contract: complex doesn't show up on
    the wire, and if it did (via hand-edited JSON with a string
    like ``"1+2j"``), it would be rejected.
    """

    def test_complex_dtype_on_input_array_rejected(self):
        """Attempting to build a ModeData with a complex
        eigenvector fails at dataclass post_init -- numpy can't
        cast complex to float without explicit .real."""
        with pytest.raises((TypeError, ValueError)):
            ModeData(
                index_1based          = 1,
                frequency_cm1         = 100.0,
                raman_activity_a4_amu = 1.0,
                ir_intensity_km_mol   = None,
                eigenvector_canonical = np.array([[1+2j, 0, 0],
                                                           [-1-2j, 0, 0]]),
                eigenvector_display   = np.array([[1+2j, 0, 0],
                                                           [-1-2j, 0, 0]]),
                has_imag              = False,
            )

    def test_complex_string_in_required_field_rejected(self, tmp_path):
        """A hand-edited file with a complex-looking string where
        a float is expected fails at the typed reconstitution."""
        payload = _make_minimal_results().to_dict()
        payload["equilibrium"]["scf_energy_eh"] = "1+2j"
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)
