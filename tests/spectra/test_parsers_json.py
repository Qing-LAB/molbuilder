"""Tests for the ``.spectra.json`` sidecar door
(``molbuilder.sidecars.spectra``: the write side, plus the read side it
re-exports from ``molbuilder.parse.sidecars.spectra``).

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
        """PLUMBING. The smallest valid result survives write -> read with its numbers
        intact, including the MO-energy ndarray.

        Catches the round trip losing the array fields. `to_dict` turns ndarrays
        into lists and `from_dict` turns them back; a half-done conversion leaves
        a Python list where the viewer expects an array, and the failure surfaces
        far downstream as a shape error in the level diagram. This is the file's
        baseline -- if it is red, nothing else here means anything.

        Contract: `web/spectra.md` § 6 (`POST /api/spectra/load` parses an
        existing `.spectra.json`); the wire shape is `spectra/results.py`.
        """
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
        """PLUMBING. A missing file raises `SpectraJsonNotFoundError`, which is BOTH a
        `FileNotFoundError` and a `SpectraJsonError`.

        Catches the route's single `except SpectraJsonError` losing this case: the
        live-watch poller reads the checkpoint before the engine has written it, so
        "not there yet" is the NORMAL state, not an error. If the class stopped
        inheriting `SpectraJsonError`, that ordinary poll would escape the handler
        and answer 500 instead of the typed 404 the UI branches on.

        Contract: `web/spectra.md` § 6 (missing -> 404 with `kind: not_found`).
        """
        bad = tmp_path / "does_not_exist.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError) as exc_info:
            parse_spectra_json(bad)
        # Inherits FileNotFoundError -- legacy callers using the
        # OSError-shaped except keep working.
        assert isinstance(exc_info.value, FileNotFoundError)
        # And the SpectraJsonError base so generic catches also work.
        assert isinstance(exc_info.value, SpectraJsonError)

    def test_missing_file_message_names_path(self, tmp_path):
        """PLUMBING. The not-found message contains the path that was not found.

        Catches a message that says only "spectra.json not found" -- the poller
        watches one path and the user picked another, and without the path in the
        sentence there is nothing to compare.

        Contract: `web/spectra.md` § 6 (the error text is what the UI shows).
        """
        bad = tmp_path / "missing.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError) as exc_info:
            parse_spectra_json(bad)
        assert str(bad) in str(exc_info.value)


# --------------------------------------------------------------------- #
#  Malformed JSON                                                       #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonMalformed:

    def test_invalid_json_raises_malformed(self, tmp_path):
        """PLUMBING. Syntactically broken JSON is a `MalformedError`, not a raw
        `json.JSONDecodeError`.

        Catches the decode error escaping unwrapped. The live-watch poller catches
        `SpectraJsonError`; a bare `JSONDecodeError` goes past it and kills the
        poll loop on a file the engine is halfway through replacing.

        Contract: `web/spectra.md` § 6 (malformed -> 400 with `kind`).

        NOTE: this and seven other tests in the file land on the same
        `except json.JSONDecodeError` branch; see the audit note on that cluster.
        """
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
        """PLUMBING. A zero-byte file is Malformed, not an unhandled exception.

        Catches the create-then-write race: the poller stats a file that exists
        and has no content yet. `json.loads("")` raises, and if it raised
        unwrapped the watch loop would die on a file that is about to be perfectly
        valid a millisecond later.

        Contract: `web/spectra.md` § 7 (live updating) + § 6 (typed errors).
        """
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
        """PLUMBING. `schema_version: 0` is a `SchemaError`.

        Contract: `spectra/results.py` -- `READABLE_SCHEMA_VERSIONS` is the one
        home for what this parser can read.

        CUT CANDIDATE, and the name is wrong. Version 4 IS legacy and IS readable
        (`READABLE_SCHEMA_VERSIONS == {4, 5}`), so "legacy rejected" states the
        opposite of the contract. What it actually exercises -- `actual not in
        READABLE_SCHEMA_VERSIONS` -- is the same line
        `test_future_schema_version_rejected` and
        `test_schema_error_message_names_both_versions` already run.
        """
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
    (archived-spec (docs/archive/old_docs/tabs/spectra/spec.md) § 5 forward-compat rule).  Test that the parser inherits
    this -- new engines can add ``engine_metadata.foo`` keys
    without breaking older readers."""

    def test_extra_top_level_keys_ignored(self, tmp_path):
        """PLUMBING. An unknown top-level key does not fail the parse, and does not get
        attached to the typed object either.

        Catches forward-compat breaking in both directions: a strict reader makes
        a file written by a newer molbuilder unreadable by an older one, and an
        auto-attaching reader would let a stray key shadow a real field name on
        the dataclass. `from_dict` names the fields it reads, which is what keeps
        both from happening.

        Contract: the forward-compat rule in
        `archive/old_docs/tabs/spectra/spec.md` § 5, inherited by this door.
        """
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
        """PLUMBING. The in-memory door (`parse_spectra_json_dict`) reconstitutes a
        result from an already-decoded dict.

        Catches the two doors drifting: `/api/spectra/load` accepts either a
        `path` or an inline `json` object, and the inline arm never touches the
        filesystem or `json.loads`. It is a SEPARATE function with its own copy of
        the schema check, so nothing in the file-path tests above covers it.

        Contract: `web/spectra.md` § 6 (the door takes `{'json': {...}}` too).
        """
        original = _make_minimal_results()
        loaded = parse_spectra_json_dict(original.to_dict())
        assert loaded.engine == "pyscf"
        assert len(loaded.modes) == 1

    def test_non_dict_rejected(self):
        """PLUMBING. A JSON array handed to the in-memory door is Malformed.

        Catches the dict door indexing into a list and raising `TypeError` from
        inside `from_dict` -- which would surface as a FieldError ("malformed
        field") for what is actually a wrong top-level shape, sending the user to
        look for a bad field in a document that has none.

        Contract: `web/spectra.md` § 6.
        """
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_missing_schema_version_rejected(self):
        """PLUMBING. The in-memory door runs the schema-version gate too.

        Catches the inline arm of `/api/spectra/load` skipping the version check.
        The dict path duplicates that check in `_parse_spectra_json_dict` rather
        than sharing it with the file path, so it can be deleted from one and not
        the other, and a v6 payload posted inline would then be reconstituted by a
        v5 reader -- silently, with whatever fields happened to still line up.

        Contract: `web/spectra.md` § 6 (wrong schema -> 422) + `spectra/results.py`.
        """
        original = _make_minimal_results()
        d = original.to_dict()
        del d["schema_version"]
        with pytest.raises(SpectraJsonSchemaError):
            parse_spectra_json_dict(d)

    def test_field_error_wrapped(self):
        """PLUMBING. A missing required field on the in-memory path is a FieldError,
        not a bare `KeyError`.

        Catches the dict door's error wrapping being dropped: `KeyError('engine')`
        escaping `except SpectraJsonError` at the route means a 500 and an HTML
        body where the UI expects `{ok, error, kind}`.

        Contract: `web/spectra.md` § 6.
        """
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
        """PLUMBING, and load-bearing. All four parser exceptions inherit
        `SpectraJsonError`.

        Catches the silent hole this file's whole exception design exists to
        close: `/api/spectra/load` catches `SpectraJsonError` ONCE
        (`web/blueprints/spectra.py`) and maps by type inside the handler. Re-parent
        any one of the four to `Exception` and that route stops catching it -- the
        failure becomes a 500 HTML page, the browser's `r.json()` throws, and the
        user sees "network error" instead of "update molbuilder". Nothing else
        would go red: every other test in this file catches the specific class.

        Contract: `web/spectra.md` § 6 (each `kind` -> its own status code).
        """
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
        """PLUMBING. `schema_version: true` is a SchemaError, and the error carries
        `True` as the actual version.

        Catches the check being written as `d["schema_version"] not in READABLE`:
        `True == 1` in Python because `bool` subclasses `int`, so a boolean would
        compare equal to a version number and walk straight past a naive gate.
        The parser has to say `isinstance(x, int) and not isinstance(x, bool)`,
        and this is the only test that distinguishes the two spellings.

        Contract: `spectra/results.py` (`READABLE_SCHEMA_VERSIONS`) +
        `web/spectra.md` § 6.
        """
        payload = _make_minimal_results().to_dict()
        payload["schema_version"] = True
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonSchemaError) as exc_info:
            parse_spectra_json(p)
        assert exc_info.value.actual is True


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
        """SCIENCE-ADJACENT. The bare `Infinity` token is rejected at decode time.

        Catches a diverged SCF loading as a real energy. Python's `json` accepts
        `NaN` / `Infinity` / `-Infinity` although RFC 8259 does not, so without
        the `parse_constant` hook an energy that blew up would reconstitute as
        `float('inf')` and propagate into every derived quantity -- the level
        diagram, the gap, the Methods numbers -- with no complaint anywhere.

        Contract: `web/spectra.md` § 6; the hook and its message live at
        `parse/sidecars/spectra.py::_reject_nonfinite_constant`.

        REDUNDANT WITH ITS SIBLINGS: `parse_constant` fires identically for all
        three tokens, and `test_nan_token_rejected` additionally asserts the
        message. Recorded, not cut -- the class is protected.
        """
        p = tmp_path / "with_inf.spectra.json"
        raw = (
            '{"schema_version": 4, "engine": "pyscf", '
            '"equilibrium_scf_eh": Infinity}'
        )
        p.write_text(raw, encoding="utf-8")
        with pytest.raises(SpectraJsonMalformedError):
            parse_spectra_json(p)

    def test_negative_infinity_token_rejected(self, tmp_path):
        """SCIENCE-ADJACENT. The `-Infinity` token is rejected at decode time.

        Same failure as the sibling above -- a diverged SCF energy entering the
        result as a finite-looking number -- for the sign an SCF energy actually
        runs away in.

        Contract: `web/spectra.md` § 6;
        `parse/sidecars/spectra.py::_reject_nonfinite_constant`.

        REDUNDANT WITH ITS SIBLINGS (one `parse_constant` hook, three tokens).
        Recorded, not cut.
        """
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
        """PLUMBING. A UTF-8 BOM at the head of the file is stripped, not treated as
        the first character of the document.

        Catches the reader dropping `encoding="utf-8-sig"`. A BOM makes
        `json.loads` fail on character 0 with an unhelpful message, and the user's
        only clue is an invisible byte -- so the failure looks like file
        corruption rather than an editor's default.

        Contract: this is the reader's own stated tolerance
        (`parse/sidecars/spectra.py`, `utf-8-sig` on open); the writer's strict
        half is `test_no_bom_in_output`.
        """
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
        """PLUMBING. The writer's product is readable by the reader.

        Catches the two halves of the wire format drifting apart. Every other test
        in this class checks one property of the written file; this is the one
        that checks the pair still closes -- and it is the only place a writer
        change is required to survive an actual parse.

        Contract: `web/spectra.md` § 6; the format is `spectra/results.py`.
        """
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
        """PLUMBING. The writer takes an `os.PathLike` destination.

        Catches the writer being narrowed to `str` -- every caller in the package
        holds a `Path`, so a str-only writer would break the engine's checkpoint
        write while every string-based test kept passing.

        Contract: `web/spectra.md` § 6 (the engine writes the checkpoint the door
        later reads).
        """
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
#  in-progress writes -> modes=[] until L2 finishes (archived-spec § 6.1).       #
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
    frequencies with has_imag=True (archived-spec § 5).  The wire shape
    must preserve sign + flag faithfully."""

    def test_negative_frequency_with_has_imag(self, tmp_path):
        """SCIENCE. An imaginary mode round-trips with its NEGATIVE frequency AND its
        `has_imag` flag, alongside a real mode in the same file.

        Catches the sign being lost. An imaginary frequency is how a saddle point
        announces itself -- the structure is not a minimum, and the reported
        "frequency" is the magnitude of an imaginary number written negative by
        convention. Drop the sign (an `abs()`, a `max(0, ...)`, an unsigned wire
        type) and a transition state loads as a perfectly ordinary vibration, with
        nothing on screen to say the geometry is wrong. The flag alone is not
        enough and the sign alone is not enough: both travel, and the mixed-mode
        file is what makes a per-mode rather than per-file handling visible.

        Contract: the wire format's imaginary-mode rule,
        `archive/old_docs/tabs/spectra/spec.md` § 5, enforced in
        `spectra/results.py::ModeData`.
        """
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
    """In-progress wire state per archived-spec § 6.1: between phase
    Setup-complete and L2-complete the file can carry an empty
    modes list with phase_frequencies=running.  Parser must
    accept this without barfing."""

    def test_empty_modes_in_progress_state(self, tmp_path):
        """PLUMBING. A checkpoint with `modes: []` and `phase_frequencies: running`
        parses.

        Catches the reader treating "no modes yet" as corruption. This is the
        normal state for the whole of L2 -- the live-watch poller reads it on
        every tick between setup and the first Hessian -- so a parser that
        demanded at least one mode would make the progress display fail exactly
        while there is progress to display.

        Contract: `web/spectra.md` § 7 (live updating) + the phase model in
        `archive/old_docs/tabs/spectra/spec.md` § 6.1.
        """
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
        """SCIENCE -- one of the four physics-carrying tests in this file. A
        `homo_idx` outside the MO-energy array is refused, naming the field.

        Catches a result whose HOMO points at nothing. Every electronic-structure
        number the tab shows is read RELATIVE to this index: the HOMO level, the
        LUMO above it, the gap between them, and the gap SHIFT that is the whole
        point of the ES phase (`web/spectra.md` § 3.1 -- the shifts are ~0.018 meV,
        so a silently wrong index does not look wrong, it looks like a different
        answer). Out of range means the reader either raises deep in the viewer or
        wraps around and reports a different orbital's energy as the HOMO.

        Contract: `web/spectra.md` § 3.1 (the level diagram is indexed from
        `homo_idx`); enforced at `spectra/results.py::SpectraResults.__post_init__`.

        DESIGN NOTE: this pins the OUT-OF-RANGE case only. Nothing here or
        elsewhere checks that `homo_idx` is the index of the highest OCCUPIED
        orbital -- an index that is in range but off by one passes, and an off-by-
        one HOMO is the realistic failure, not 99.
        """
        payload = _make_minimal_results().to_dict()
        # 5 MO energies in the fixture array -> valid range [0, 5)
        payload["equilibrium"]["homo_idx"] = 99
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError) as exc_info:
            parse_spectra_json(p)
        assert "homo_idx" in str(exc_info.value)

    def test_cross_mode_es_window_mismatch(self, tmp_path):
        """Per archived-spec § 5, every mode's ES window must have the same
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
        """SCIENCE-ADJACENT. A JSON `null` where the equilibrium SCF energy belongs is
        a FieldError at the door.

        Catches `None` entering the typed object in a slot everything downstream
        does arithmetic on. The energy is the reference every ES displacement is
        measured against, so a null does not fail here -- it fails a hundred lines
        later, as `TypeError: unsupported operand` inside a subtraction, with no
        hint that a file field was empty.

        Contract: `web/spectra.md` § 6 (bad field -> 400); the required-vs-optional
        split is `spectra/results.py`.
        """
        payload = _make_minimal_results().to_dict()
        payload["equilibrium"]["scf_energy_eh"] = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_null_n_atoms_total_rejected(self, tmp_path):
        """SCIENCE-ADJACENT. A null `n_atoms_total` is a FieldError.

        Catches the atom count going missing. It is the right-hand side of the
        free/frozen partition check (`web/spectra.md` § 8) -- with `None` there,
        the partition arithmetic cannot run, so the check that decides WHICH atoms
        the frequencies belong to is skipped rather than failed.

        Contract: `web/spectra.md` § 8 (the two lists must partition
        `range(n_atoms_total)`).
        """
        payload = _make_minimal_results().to_dict()
        payload["n_atoms_total"] = None
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_null_mode_frequency_rejected(self, tmp_path):
        """SCIENCE-ADJACENT. A null `frequency_cm1` on a mode is a FieldError.

        Catches a mode with no frequency reaching the spectrum. The frequency is
        the mode's x-coordinate on the chart and its argument to every
        thermal/zero-point amplitude formula (`web/spectra.md` § 4.1); a `None`
        there either crashes the chart or -- worse, if something coerces it -- puts
        a peak at 0 cm-1 that no calculation produced.

        Contract: `web/spectra.md` § 6; the required fields are
        `spectra/results.py::ModeData`.
        """
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
        """PLUMBING. A file with no `equilibrium` block at all is a FieldError.

        Catches the whole nested block going missing being reported as something
        other than a field problem -- `from_dict` would raise `KeyError` and, if
        unwrapped, escape the route's `except SpectraJsonError` as a 500.

        Contract: `web/spectra.md` § 6.
        """
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_nested_scf_energy(self, tmp_path):
        """PLUMBING. A field missing from INSIDE `equilibrium` is a FieldError, not a
        raw KeyError.

        Catches the wrapping being applied only at the top level. Nested lookups
        happen inside `SpectraResults.from_dict`, one frame further in, so a
        `try` placed around the outer dict only would let the inner `KeyError`
        through -- and the nested fields are exactly where a hand-edited or
        version-skewed file goes wrong.

        Contract: `web/spectra.md` § 6.
        """
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]["scf_energy_eh"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_nested_mo_energies(self, tmp_path):
        """PLUMBING. A missing `equilibrium.mo_energies_eh` is a FieldError.

        Catches the same wrapping gap as its sibling above, on the array field --
        the one whose absence would otherwise surface as a numpy error rather than
        a named missing field.

        Contract: `web/spectra.md` § 6.

        NEAR-DUPLICATE of `test_missing_nested_scf_energy`: both delete one key
        from the same sub-dict and reach the same `except KeyError` line.
        """
        payload = _make_minimal_results().to_dict()
        del payload["equilibrium"]["mo_energies_eh"]
        p = _write_json(tmp_path, payload)
        with pytest.raises(SpectraJsonFieldError):
            parse_spectra_json(p)

    def test_missing_mode_required_field(self, tmp_path):
        """PLUMBING. A required field missing from a MODE dict is a FieldError.

        Catches the wrapping gap one level deeper still: modes are reconstituted
        in a loop inside `ModeData.from_dict`, so this is the third distinct frame
        a `KeyError` can be raised from, and each has to be caught by the same
        outer handler.

        Contract: `web/spectra.md` § 6.
        """
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
    complete} (archived-spec § 5).  An invalid value comes through the
    parser as a FieldError because the dataclass __post_init__
    rejects it."""

    def test_invalid_phase_string_rejected(self, tmp_path):
        """PLUMBING. A phase status outside {empty, running, complete} is a FieldError.

        Catches an unknown phase word reaching the UI, which branches on these
        three strings to decide what to draw. An unrecognised value is not an
        error there -- it just matches none of the branches, so the tab renders a
        finished run as if nothing had started.

        Contract: the phase vocabulary in
        `archive/old_docs/tabs/spectra/spec.md` § 5, enforced in
        `spectra/results.py::SpectraResults.__post_init__`; `web/spectra.md` § 7
        for what reads it.
        """
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
        """SCIENCE-ADJACENT. MO energies spanning 1e-10 to 1e+5 Hartree round-trip
        EXACTLY (`assert_array_equal`, not `approx`).

        Catches precision loss in the array path. MO energies are compared to each
        other, not to zero: the ES gap shifts the tab reports are ~1e-5 eV
        differences between numbers of order 1e+1, so any rounding on write --
        a `%.6f` format, a float32 cast, a `round()` -- destroys the quantity
        while leaving every value looking plausible. Exact equality is the
        assertion that can see that; `approx` cannot.

        Contract: `web/spectra.md` § 3.1 (the shifts are small and are the point).
        """
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
        """SCIENCE-ADJACENT. The optional `equilibrium.elements` + `positions_ang`
        survive the round trip, and appear under `equilibrium` on the wire.

        Catches the geometry the modes belong TO being dropped. An eigenvector is
        a displacement of specific atoms at specific places; without the elements
        and positions, the viewer animates a mode against whatever structure
        happens to be loaded, which is how a displacement gets drawn on the wrong
        atom. The wire-shape half matters because the reader looks under
        `equilibrium` -- written at the top level, they round-trip as None and the
        loss is silent.

        Contract: `model/overview.md`'s atom-index invariant + `web/spectra.md`
        § 4 (clicking a mode animates it on this geometry).
        """
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
        """SCIENCE-ADJACENT, and the file's integration test. One file carrying every
        feature at once -- an imaginary mode, a mode with an ES block, a mode with
        neither Raman nor ES, nested config, unicode methods text, scientific-
        notation energies -- round-trips field by field.

        Catches the interactions the single-feature tests structurally cannot see:
        a mode's ES block being attached to the WRONG mode (here mode 2 has one
        and modes 1 and 3 must not), `selected_mode_idxs_1based` drifting off the
        1-based convention it is named for, and an Optional field written as
        `null` for one mode being read back onto another. Every other test in this
        file uses a one-mode fixture where none of those can happen.

        Contract: `web/spectra.md` § 6 + the wire format in
        `spectra/results.py`.

        DESIGN NOTE: it asserts field equality, not the physics -- the mode
        ORDER (frequencies ascending, imaginary first) is a convention the viewer
        depends on and nothing here checks it.
        """
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
        """PLUMBING. A small energy written in scientific notation loads as the number
        it spells.

        Catches `parse_float` -- the hook the non-finite check is installed
        through -- mangling ordinary exponent notation. The hook replaces
        CPython's float conversion for EVERY float literal in the document, so a
        mistake in it does not fail loudly; it changes values.

        Contract: `parse/sidecars/spectra.py::_strict_finite_float`.

        NOTE: this writes the value through `json.dumps`, so the literal actually
        on disk is whatever Python chose to emit -- the notation the name promises
        is only guaranteed by the uppercase-E sibling, which hand-writes its JSON.
        """
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
        """SCIENCE-ADJACENT. `-1e500` -- valid JSON syntax, overflows to `-inf` -- is
        rejected.

        Catches the negative half of the runaway-energy case, which is the half
        that actually happens: an SCF that fails to converge runs the total energy
        down, not up. `parse_constant` never sees this literal (it is syntactically
        a normal number), so `_strict_finite_float` is the only thing standing
        between it and a `-inf` equilibrium energy.

        Contract: `parse/sidecars/spectra.py::_strict_finite_float`.

        REDUNDANT WITH `test_overflow_literal_rejected`: `math.isfinite` is
        sign-blind. Recorded, not cut -- the class is protected.
        """
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
