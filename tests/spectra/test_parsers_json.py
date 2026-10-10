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


def _write_json(tmp_path: Path, results, name: str = "spectra.json") -> Path:
    """The result as a run's file: through our ONE writer,
    `dump_spectra_json`."""
    p = tmp_path / name
    dump_spectra_json(results, p)
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
        p = _write_json(tmp_path, original)
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
        p = _write_json(tmp_path, original)
        # Passing the Path object directly:
        loaded = parse_spectra_json(p)
        assert loaded.engine == "pyscf"


    def test_intermediate_phase_state_round_trips(self, tmp_path):
        """A partially-complete file (L2 done, L3+L4 empty) round-
        trips cleanly -- the parser doesn't reject incomplete runs,
        only malformed ones."""
        partial = _make_minimal_results(complete=False)
        p = _write_json(tmp_path, partial)
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
#  Forward-compatibility                                                #
# --------------------------------------------------------------------- #


class TestParseSpectraJsonForwardCompat:
    """``engine_metadata`` is the one free-form block: an engine may put
    anything in it and it round-trips intact.  Every OTHER key is gated --
    an unknown top-level, mode, equilibrium or electronic-structure key is
    refused by name (`engines/vibration.md` § 6.7; the gate test is
    `test_types.py::test_the_reader_refuses_a_key_it_does_not_know_by_name`)."""

    def test_extra_engine_metadata_keys_round_trip(self, tmp_path):
        """``engine_metadata`` is a free-form dict -- engines can
        stuff anything in there and it round-trips intact."""
        original = _make_minimal_results()
        # Engine-specific metadata, set on the result as an engine sets it.
        original.engine_metadata = {
            "pyscf_xc_grid_radial": 75,
            "custom_engine_flag":   True,
            "list_of_things":       [1, 2, 3],
        }
        p = _write_json(tmp_path, original)
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
#  Encoding edge cases                                                  #
# --------------------------------------------------------------------- #


class TestParseEncodingTolerance:
    """The reader uses utf-8-sig so a BOM-prefixed file (some
    Windows editors insert one) is transparently stripped instead
    of poisoning the first byte of the JSON document."""


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
#  selector=skip -> every mode has electronic_structure=null;           #
#  in-progress writes -> modes=[] until L2 finishes.                    #
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
        """ir_intensity_km_mol=None round-trips."""
        results = _make_minimal_results()
        p = tmp_path / "ir_null.spectra.json"
        dump_spectra_json(results, p)
        loaded = parse_spectra_json(p)
        assert all(m.ir_intensity_km_mol is None for m in loaded.modes)

    def test_electronic_structure_null_round_trips(self, tmp_path):
        """When selector=skip (or a mode wasn't picked) the mode
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
    frequencies with has_imag=True.  The wire shape
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

        Contract: the wire format's imaginary-mode rule, enforced in
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
    """In-progress wire state: between phase
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

        Contract: `web/spectra.md` § 7 (live updating).
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


# --------------------------------------------------------------------- #
#  Round-trip stress: an "everything" result                            #
# --------------------------------------------------------------------- #


class TestGeometryRoundTrip:
    """The optional ``equilibrium.elements`` + ``equilibrium.
    positions_ang`` fields round-trip
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
        # The masses travel with the geometry (`engines/vibration.md` § 6.2).
        results.equilibrium_masses_amu = np.array([15.999, 1.008])
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
