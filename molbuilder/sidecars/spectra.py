"""``.spectra.json`` sidecar — write-side + exception classes.

H2 of parse-module.md migration: absorbed from the legacy
:mod:`molbuilder.parsers.spectra_json` module's write-side surface.
The read-side parser lives in
:mod:`molbuilder.parse.sidecars.spectra` per the parse-module
contract.

This module is the canonical home for the spectra-JSON exception
classes (``SpectraJsonError`` etc.); the read-side re-imports them
so callers can ``except`` on either side without caring which
module raised.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Union

from molbuilder.spectra.results import SpectraResults


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


def dump_spectra_json(results: SpectraResults,
                      path: Union[str, "os.PathLike[str]"],
                      *,
                      indent: int = 2) -> None:
    """Write ``results`` to ``path`` via atomic rename.

    The wire-format contract that every Spectra-tab writer follows
    — providing it as a helper here keeps engines from diverging on
    the details (NaN handling, indent, BOM, atomicity).  The emitted
    script template in ``spectra/<engine>_engine.py::render_script``
    either imports this directly or inlines the equivalent.

    Behaviour:

      * ``results.to_dict()`` is encoded with ``allow_nan=False`` —
        a non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk, so the engine
        is forced to filter or null out NaN/Inf SCF energies
        explicitly instead of producing JSON that downstream
        consumers can't read.
      * UTF-8 without a BOM (cm⁻¹ / Å survive verbatim thanks to
        ``ensure_ascii=False``).
      * Atomic: write to ``<path>.tmp.<pid>`` first, then
        :func:`os.replace` it on top of ``path``.  A reader opening
        the path mid-write sees either the prior version (intact) or
        the new version (intact) — never a half-written file.

    Parameters
    ----------
    results
        The :class:`SpectraResults` to serialise.
    path
        Destination path.  Parent directory must exist.
    indent
        ``json.dumps`` indent.  Default 2 (readable + diffable);
        pass 0 / ``None`` for the compact wire form.

    Raises
    ------
    ValueError
        ``results.to_dict()`` contains a non-finite float.  The
        engine must filter NaN/Inf before calling this.
    OSError
        Path can't be written (permission / no such directory /
        disk full).  The temp file is cleaned up before re-raise.
    """
    p = os.fspath(path)
    payload = results.to_dict()

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


__all__ = [
    "dump_spectra_json",
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
]
