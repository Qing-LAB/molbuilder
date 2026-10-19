"""``.spectra.json`` sidecar — write-side + exception classes.

The ONE DOOR for this sidecar: the write side lives here, and the read
side is re-exported from :mod:`molbuilder.parse.sidecars.spectra`, so a
caller needs one import for both.  The split is what
``model/parse.md`` § 4 requires.

This module is the canonical home for the spectra-JSON exception
classes (``SpectraJsonError`` etc.); the read-side re-imports them
so callers can ``except`` on either side without caring which
module raised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

# THIS MODULE TRAVELS, and imports only the standard library at load: the
# SIESTA vibration's finish writes the file beside the job through it
# (`runwrap.VIBRATION_COMPANIONS`), and the PySCF vibration script does
# (`runwrap.PYSCF_COMPANIONS`) -- the one writer every engine's result goes
# through, where the package is not installed.  The results class is an
# annotation here, nothing more.
if TYPE_CHECKING:
    from ..spectra.results import SpectraResults


# --------------------------------------------------------------------- #
#  Exceptions (canonical home; read-side re-imports)                     #
# --------------------------------------------------------------------- #


# TWO WAYS, because this module travels beside the job in both bundles.
try:                                        # inside molbuilder
    from ..persist import write_bytes as _write_bytes
    from ..runfiles import compose as _compose, role_of as _role_of
    from ..spectra.datafile import write_spectra_data as _write_spectra_data
    from ..spectra.derived import with_derived as _with_derived
except ImportError:                         # beside a job, in a bundle
    from persist import write_bytes as _write_bytes
    from runfiles import compose as _compose, role_of as _role_of
    from spectra_datafile import write_spectra_data as _write_spectra_data
    from derived import with_derived as _with_derived

#: The result's role and its data file's, the catalogue's (`runfiles.WRITTEN`).
SPECTRA_ROLE = ".spectra.json"
DATA_ROLE = ".spectra.nc"


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
                          path: Union[str, "os.PathLike[str]"],
                          *,
                          indent: int = 2) -> None:
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
      * ``payload`` is encoded with ``allow_nan=False`` -- a non-finite
        scalar anywhere in the payload raises :class:`ValueError` BEFORE any
        bytes hit disk, so the engine is forced to filter or null out
        NaN/Inf SCF energies explicitly (:func:`finite_or_none`);
      * UTF-8 without a BOM (cm⁻¹ / Å survive verbatim, ``ensure_ascii=False``);
      * atomic, both files: a reader opening either path mid-write sees the
        prior version or the new one, never a half-written file; the data
        file states the sha256 of the JSON bytes written with it.
    """
    import hashlib
    p = Path(os.fspath(path))
    role = _role_of(p.name)
    if role != SPECTRA_ROLE:
        raise ValueError(
            f"{p.name}: a vibration result is named <label>{SPECTRA_ROLE} "
            f"(runfiles.WRITTEN), and its data file follows from that name")
    label = p.name[:-len(SPECTRA_ROLE)]

    # EVERY MODE'S DERIVED VALUES, stated beside what they come from by the
    # one derivation (`spectra.derived`, `engines/vibration.md` § 6.6) -- in
    # this writer, so every engine's file states them alike at every write.
    if payload.get("modes"):
        payload = {**payload, "modes": _with_derived(payload["modes"])}

    text = json.dumps(payload,
                      indent=indent,
                      ensure_ascii=False,
                      allow_nan=False,
                      sort_keys=False)
    data = text.encode("utf-8")
    _write_bytes(p, data)
    _write_spectra_data(payload, p.with_name(_compose(label, DATA_ROLE)),
                        json_sha256=hashlib.sha256(data).hexdigest(),
                        label=label)


def dump_spectra_json(results: "SpectraResults",
                      path: Union[str, "os.PathLike[str]"],
                      *,
                      indent: int = 2) -> None:
    """Write ``results`` to ``path`` -- :func:`write_spectra_payload` of its
    ``to_dict()``, so a results object and a payload built field by field
    (the PySCF script's, phase by phase) are written by the one writer."""
    write_spectra_payload(results.to_dict(), path, indent=indent)


def finite_or_none(values):
    """``values`` -- a number array of any shape -- as nested lists, each
    NaN or infinity as ``None``: what an optional field of the payload
    holds where a number is not finite, since :func:`write_spectra_payload`
    refuses one.  The PySCF vibration script scrubs every array it records
    with it (imported from ``mb_pyscf.pyz``)."""
    import math

    import numpy as np
    a = np.asarray(values, dtype=float)
    if np.isfinite(a).all():
        return a.tolist()
    flat = [float(x) if math.isfinite(x) else None for x in a.flat]
    if a.ndim == 1:
        return flat
    return np.asarray(flat, dtype=object).reshape(a.shape).tolist()


def parse_spectra_json(path):
    """Read-side convenience re-export — delegates to
    :func:`molbuilder.parse.sidecars.spectra._parse_spectra_json`
    so callers have a single ``molbuilder.sidecars.spectra``
    namespace for both read + write.  Local import avoids module-
    load-time cycle (parse module imports exceptions from here)."""
    from molbuilder.parse.sidecars.spectra import _parse_spectra_json
    return _parse_spectra_json(path)


def parse_spectra_json_dict(d):
    """In-memory variant of :func:`parse_spectra_json`.  Re-exports
    :func:`molbuilder.parse.sidecars.spectra._parse_spectra_json_dict`."""
    from molbuilder.parse.sidecars.spectra import _parse_spectra_json_dict
    return _parse_spectra_json_dict(d)


__all__ = [
    "write_spectra_payload",
    "dump_spectra_json",
    "finite_or_none",
    "parse_spectra_json",
    "parse_spectra_json_dict",
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
]
