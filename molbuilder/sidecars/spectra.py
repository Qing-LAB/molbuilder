"""``.spectra.json`` sidecar — its writer, its reader and exception classes.

The ONE DOOR for this sidecar: the writer (:func:`write_spectra_payload`)
and the reader (:func:`parse_spectra_json`, the gates of
``engines/vibration.md`` § 6.7) both live here, so a caller needs one import
for both; the registry's FileParser in
:mod:`molbuilder.parse.sidecars.spectra` asks the reader.

This module is the home of the spectra-JSON reader and its exception
classes (``SpectraJsonError`` etc.); the FileParser's module asks the reader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

# THIS MODULE TRAVELS, and imports the standard library, numpy and netCDF4
# (through its data file's writer) and the members its `TRAVELS_WITH` names at
# load: the
# SIESTA vibration's finish writes the file beside the job through it
# (`runwrap.members`), and the PySCF vibration script does
# (`runwrap.members`) -- the one writer every engine's result goes
# through, where the package is not installed.  The results class is an
# annotation here, nothing more.
if TYPE_CHECKING:
    from ..spectra.results import SpectraResults


# --------------------------------------------------------------------- #
#  Exceptions                                                            #
# --------------------------------------------------------------------- #


# TWO WAYS, because this module travels beside the job in both bundles.
try:                                        # inside molbuilder
    from ..persist import json_text as _json_text
    from ..persist import object_of as _object_of
    from ..persist import sha256_of_bytes as _sha256_of_bytes
    from ..persist import write_bytes as _write_bytes
    from ..runfiles import (ROLE_SPECTRA, ROLE_SPECTRA_DATA,
                            compose as _compose, role_of as _role_of,
                            stem_before as _stem_before)
    from ..spectra.datafile import write_spectra_data as _write_spectra_data
    from ..spectra.derived import with_derived as _with_derived
except ImportError:                         # beside a job, in a bundle
    from persist import json_text as _json_text  # type: ignore[no-redef]
    from persist import object_of as _object_of  # type: ignore[no-redef]
    from persist import sha256_of_bytes as _sha256_of_bytes  # type: ignore[no-redef]
    from persist import write_bytes as _write_bytes
    from runfiles import (ROLE_SPECTRA, ROLE_SPECTRA_DATA,  # type: ignore[no-redef]
                          compose as _compose, role_of as _role_of,
                          stem_before as _stem_before)
    from spectra_datafile import write_spectra_data as _write_spectra_data
    from derived import with_derived as _with_derived


class SpectraJsonError(Exception):
    """Base class for spectra-JSON parser failures.  Catch this when
    the caller wants "any parse problem"; catch the specific
    subclasses below when the failure mode matters (live-watch
    poller distinguishes "file not yet written" from "file is wrong
    shape", for instance)."""


class SpectraJsonNotFoundError(SpectraJsonError, FileNotFoundError):
    """The file does not exist (yet).  Inherits
    :class:`FileNotFoundError` so existing ``except
    FileNotFoundError`` blocks keep working; the dual base lets
    callers also catch via :class:`SpectraJsonError` when they're
    handling all parse problems generically."""


class SpectraJsonMalformedError(SpectraJsonError):
    """The file exists but isn't valid JSON, isn't a JSON object at
    the top level, contains a non-standard token (``NaN`` /
    ``Infinity``), or can't be decoded as UTF-8."""


class SpectraJsonSchemaError(SpectraJsonError):
    """``schema_version`` is missing, the wrong type, or doesn't
    match :data:`molbuilder.spectra.results.SCHEMA_VERSION`."""

    def __init__(self, expected: int, actual: Any):
        super().__init__(
            f"spectra.json schema_version mismatch: expected "
            f"{expected}, got {actual!r}.  Either the file was "
            f"written by a different molbuilder version, or it "
            f"isn't a Spectra-tab result file."
        )
        self.expected = expected
        self.actual   = actual


class SpectraJsonFieldError(SpectraJsonError):
    """A required field was missing / had the wrong type at the
    :meth:`SpectraResults.from_dict` reconstitution step.  Wraps
    the underlying ``KeyError`` / ``TypeError`` / ``ValueError``
    with a message that names the field path."""


# --------------------------------------------------------------------- #
#  Write entry-point                                                    #
# --------------------------------------------------------------------- #


def write_spectra_payload(payload: dict,
                          path: Union[str, "os.PathLike[str]"]) -> None:
    """Write the result to ``path`` -- ``<label>.spectra.json`` -- and, from
    the same payload, its data file ``<label>.spectra.nc`` beside it
    (`spectra.datafile.write_spectra_data`; `engines/vibration.md`
    § 6.10.5): one source, two files, both through the one atomic writer
    (`persist.write_bytes`).  ``path`` is named as the catalogue names the
    result (`runfiles`: the `.spectra.json` role on the run's label), and
    the data file's name follows from it; another name is refused.

    The wire-format contract every writer follows: :func:`dump_spectra_json`
    with a results object, the SIESTA finish through that, and the PySCF
    vibration script -- importing it from ``mb_pyscf.pyz`` -- with the
    payload it builds phase by phase.

    Behaviour:

      * every mode's derived values -- the activity classes, the zero-point
        amplitude and displacement -- are stated beside its own keys by the
        one derivation (`spectra.derived`), whichever engine built the
        payload;
      * a non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk (``engines/vibration.md``
        § 6.8: *no NaN*): what may stand as null is written so by the record
        (the orbital energies, `spectra.results.finite_or_none`), and anything
        else stops the writer;
      * the bytes are the one shape of every JSON file molbuilder writes
        (`persist.json_text`): UTF-8 without a BOM, indented, cm⁻¹ / Å
        verbatim, a trailing newline;
      * atomic, both files: a reader opening either path mid-write sees the
        prior version or the new one, never a half-written file; the data
        file states the sha256 of the JSON bytes written with it.
    """
    p = Path(os.fspath(path))
    role = _role_of(p.name)
    if role != ROLE_SPECTRA:
        raise ValueError(
            f"{p.name}: a vibration result is named <label>{ROLE_SPECTRA} "
            f"(runfiles.WRITTEN), and its data file follows from that name")
    label = _stem_before(p.name, ROLE_SPECTRA)

    # EVERY MODE'S DERIVED VALUES, stated beside what they come from by the
    # one derivation (`spectra.derived`, `engines/vibration.md` § 6.6) -- in
    # this writer, so every engine's file states them alike at every write.
    if payload.get("modes"):
        payload = {**payload, "modes": _with_derived(payload["modes"])}

    # THE CONTRACT'S REFUSAL (§ 6.8) of a number JSON cannot hold, where
    # every other file writes it null.
    data = _json_text(payload, refuse_non_finite=True).encode("utf-8")
    _write_bytes(p, data)
    _write_spectra_data(payload, p.with_name(_compose(label, ROLE_SPECTRA_DATA)),
                        json_sha256=_sha256_of_bytes(data),
                        label=label)


def dump_spectra_json(results: "SpectraResults",
                      path: Union[str, "os.PathLike[str]"]) -> None:
    """Write ``results`` to ``path`` -- :func:`write_spectra_payload` of its
    ``to_dict()``, so a results object and a payload built field by field
    (the PySCF script's, phase by phase) are written by the one writer."""
    write_spectra_payload(results.to_dict(), path)


# --------------------------------------------------------------------- #
#  Read entry-point                                                      #
#                                                                       #
#  The results class is imported inside each reader, from the package:  #
#  this module travels (what it loads is said at its head), and no      #
#  module of either bundle asks the reader.                             #
# --------------------------------------------------------------------- #


def _validate_schema_version(actual: Any) -> None:
    """The version, by the one rule (`spectra.results.readable_version`,
    which `SpectraResults.from_dict` asks too), refused in this reader's
    own error."""
    from ..spectra.results import SCHEMA_VERSION, readable_version
    try:
        readable_version(actual)
    except ValueError:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, actual) from None


def _validate_dict_to_results(d: Dict[str, Any],
                              where: str) -> "SpectraResults":
    """Shared reconstitution + error-wrapping logic.  ``where`` is a
    short label (path or ``"input dict"``) interpolated into the
    FieldError message so callers can tell which input failed."""
    from ..spectra.results import SpectraResults
    try:
        return SpectraResults.from_dict(d)
    except KeyError as e:
        raise SpectraJsonFieldError(
            f"{where} is missing required field {e.args[0]!r}"
        ) from e
    except (TypeError, ValueError, AttributeError) as e:
        raise SpectraJsonFieldError(
            f"{where} has a malformed field: {e}"
        ) from e


def parse_spectra_json(
        path: Union[str, "os.PathLike[str]"]) -> "SpectraResults":
    """Read a ``<job>.spectra.json`` file and return a typed
    :class:`SpectraResults` -- the file's one reader.  Raises
    :class:`SpectraJsonNotFoundError` / :class:`SpectraJsonMalformedError`
    / :class:`SpectraJsonSchemaError` / :class:`SpectraJsonFieldError`
    on the corresponding failure modes."""
    from ..spectra.results import SCHEMA_VERSION
    p = os.fspath(path)

    if not os.path.exists(p):
        raise SpectraJsonNotFoundError(
            f"spectra.json not found at {p!r} (engine hasn't "
            f"written its first phase checkpoint yet, or the path "
            f"is wrong)"
        )

    try:
        with open(p, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise SpectraJsonError(f"failed to read {p!r}: {e}") from e

    # THE ONE RECORD READER (`persist.object_of`): one JSON object, its
    # text UTF-8 -- and no number JSON cannot hold: the writer refuses one
    # before the file is written (`engines/vibration.md` § 6.8), and a file
    # that holds one anyway is refused here by name.
    try:
        d = _object_of(raw, refuse_non_finite=True)
    except ValueError as e:
        raise SpectraJsonMalformedError(f"{p!r}: {e}") from e

    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])

    return _validate_dict_to_results(d, repr(p))


def read_vibration_result(path: Union[str, "os.PathLike[str]"]
                          ) -> "SpectraResults":
    """A vibration result from either of its two files, by the role its name
    has (`runfiles.WRITTEN`): ``<label>.spectra.json`` through
    :func:`parse_spectra_json`, its data file ``<label>.spectra.nc`` through
    the data file's reader (`spectra.datafile.read_spectra_data`) into the
    same typed result -- one answer whichever a person points at
    (`engines/vibration.md` § 6.11 § 6).  Another name is refused."""
    p = Path(os.fspath(path))
    role = _role_of(p.name)
    if role == ROLE_SPECTRA:
        return parse_spectra_json(p)
    if role == ROLE_SPECTRA_DATA:
        from ..spectra.datafile import read_spectra_data
        if not p.is_file():
            raise SpectraJsonNotFoundError(f"no data file at {str(p)!r}")
        return _validate_dict_to_results(read_spectra_data(p), repr(str(p)))
    raise SpectraJsonError(
        f"{p.name}: a vibration result is <label>{ROLE_SPECTRA} or its data "
        f"file <label>{ROLE_SPECTRA_DATA}")


__all__ = [
    "write_spectra_payload",
    "dump_spectra_json",
    "parse_spectra_json",
    "read_vibration_result",
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.persist",
                "molbuilder.runfiles",
                "molbuilder.spectra.datafile",
                "molbuilder.spectra.derived")


#: ITS NAME BESIDE A JOB (`runwrap.bundle_members`): its last name,
#: ``spectra``, is several modules' (validation's, the parsers', the page's
#: route), so it travels under the one it has always had.
TRAVELS_AS = "spectra_sidecar"
