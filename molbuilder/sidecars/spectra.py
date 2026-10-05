"""``.spectra.json`` sidecar — write-side + exception classes.

The ONE DOOR for this sidecar: the write side lives here, and the read
side is re-exported from :mod:`molbuilder.parse.sidecars.spectra`, so a
caller needs one import for both. Absorbed from the legacy
``molbuilder.parsers.spectra_json`` (deleted 2026-06-21).  The split is
what ``model/parse.md`` § 4 requires (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

This module is the canonical home for the spectra-JSON exception
classes (``SpectraJsonError`` etc.); the read-side re-imports them
so callers can ``except`` on either side without caring which
module raised.
"""

from __future__ import annotations

import json
import os
import tempfile
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


def structure_hash_text(n_atoms, label, elements, positions_ang):
    """The ``structure_hash`` an artifact carries: ``sha256:`` over the
    lines ``n_atoms``, ``label`` and one ``'<el:<3s> <x:14.8f> <y> <z>'``
    per atom, joined by newlines, UTF-8.  The PySCF vibration script imports
    it from ``mb_pyscf.pyz``; the SIESTA derivation calls it here.
    """
    import hashlib as _hashlib
    lines = [f"{int(n_atoms)}", f"{label}"]
    for el, (x, y, z) in zip(elements, positions_ang):
        lines.append(f"{str(el):<3s} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f}")
    return "sha256:" + _hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_spectra_payload(payload: dict,
                          path: Union[str, "os.PathLike[str]"],
                          *,
                          indent: int = 2) -> None:
    """Write a ``.spectra.json`` payload to ``path`` via atomic rename.

    The wire-format contract that every Spectra-tab writer follows
    — providing it as a helper here keeps engines from diverging on
    the details (NaN handling, indent, BOM, atomicity).  Every writer
    calls it: :func:`dump_spectra_json` with a results object, the SIESTA
    finish through that, and the PySCF vibration script -- importing it from
    ``mb_pyscf.pyz`` -- with the payload it builds phase by phase.  (The
    script carried a copy of this until 2026-10-05.)

    Behaviour:

      * ``payload`` is encoded with ``allow_nan=False`` —
        a non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk, so the engine
        is forced to filter or null out NaN/Inf SCF energies
        explicitly instead of producing JSON that downstream
        consumers can't read.
      * UTF-8 without a BOM (cm⁻¹ / Å survive verbatim thanks to
        ``ensure_ascii=False``).
      * Atomic: write to a temp file beside ``path`` (``tempfile.mkstemp``,
        named ``<name>.<random>.tmp``) first, then :func:`os.replace` it on
        top of ``path``.  A reader opening the path mid-write sees either
        the prior version (intact) or the new version (intact) — never a
        half-written file.

    Parameters
    ----------
    payload
        The ``.spectra.json`` document, as a dict.
    path
        Destination path.  Parent directory must exist.
    indent
        ``json.dumps`` indent.  Default 2 (readable + diffable);
        pass 0 / ``None`` for the compact wire form.

    Raises
    ------
    ValueError
        ``payload`` contains a non-finite float.  The engine must
        filter NaN/Inf before calling this (:func:`finite_or_none`).
    OSError
        Path can't be written (permission / no such directory /
        disk full).  The temp file is cleaned up before re-raise.
    """
    p = os.fspath(path)

    # ``allow_nan=False`` is the safety net: dataclass __post_init__
    # validates shapes but doesn't enforce finiteness on scalar
    # fields (an SCF that didn't converge can leave NaN in
    # equilibrium_scf_eh).  json.dumps would otherwise happily emit
    # the bare token `NaN`.
    text = json.dumps(payload,
                      indent=indent,
                      ensure_ascii=False,
                      allow_nan=False,
                      sort_keys=False)

    # Atomic write: temp file in the same directory (so os.replace
    # is a same-filesystem rename), fsync the data before replace
    # to survive a crash between write() and replace().
    parent  = os.path.dirname(os.path.abspath(p)) or "."
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(p) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some filesystems (tmpfs on some kernels) reject
                # fsync; the data is still in the OS write buffer
                # and will land before the replace anyway.  Don't
                # let a quirky FS block the write.
                pass
        os.replace(tmp, p)
    except BaseException:
        # Best-effort cleanup of the temp file on any failure
        # (including KeyboardInterrupt).  Swallow errors during
        # cleanup -- the original exception is what the caller
        # cares about.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    "structure_hash_text",
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
