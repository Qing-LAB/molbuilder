"""``.transport.json`` sidecar — write-side + exception classes.

The ONE DOOR for this sidecar: the write side lives here, and the read
side is re-exported from :mod:`molbuilder.parse.sidecars.transport`, so
a caller needs one import for both. Absorbed from the legacy
``molbuilder.parsers.transport_json`` (deleted 2026-06-21).  The split
is what ``model/parse.md`` § 4 requires (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

This module is the canonical home for the transport-JSON exception
classes (``TransportJsonError`` etc.); the read-side re-imports
them so callers can ``except`` on either side without caring which
module raised.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Union

from molbuilder.transport.results import TransportResults


# --------------------------------------------------------------------- #
#  Exceptions (canonical home; read-side re-imports)                     #
# --------------------------------------------------------------------- #


class TransportJsonError(Exception):
    """Base class for transport-JSON parser failures."""


class TransportJsonNotFoundError(TransportJsonError, FileNotFoundError):
    """The file does not exist (yet).  Dual base lets existing
    ``except FileNotFoundError`` blocks keep working."""


class TransportJsonMalformedError(TransportJsonError):
    """The file exists but isn't valid JSON, isn't a JSON object at
    the top level, contains a non-standard token (``NaN`` /
    ``Infinity``), or can't be decoded as UTF-8."""


class TransportJsonSchemaError(TransportJsonError):
    """``schema_version`` is missing, the wrong type, or doesn't
    match :data:`molbuilder.transport.results.SCHEMA_VERSION`."""

    def __init__(self, expected: str, actual: Any):
        super().__init__(
            f"transport.json schema_version mismatch: expected "
            f"{expected}, got {actual!r}.  Either the file was "
            f"written by a different molbuilder version, or it "
            f"isn't a Transport-tab result file."
        )
        self.expected = expected
        self.actual   = actual


class TransportJsonFieldError(TransportJsonError):
    """A required field was missing / had the wrong type at the
    :meth:`TransportResults.from_dict` reconstitution step."""


# --------------------------------------------------------------------- #
#  Write entry-point                                                    #
# --------------------------------------------------------------------- #


def dump_transport_json(results: TransportResults,
                        path: Union[str, "os.PathLike[str]"],
                        *,
                        indent: int = 2) -> None:
    """Write ``results`` to ``path`` via atomic rename.

    The wire-format contract that every Transport-tab writer
    follows — providing it as a helper here keeps engines from
    diverging on the details (NaN handling, indent, BOM, atomicity).

    Behaviour:

      * ``results.to_dict()`` is encoded with ``allow_nan=False`` —
        a non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk, so the
        engine is forced to filter or null out NaN/Inf transmission
        values explicitly.
      * UTF-8 without a BOM (cm⁻¹ / Å / G_0 survive verbatim thanks
        to ``ensure_ascii=False``).
      * Atomic: write to ``<path>.tmp.<pid>`` first, then
        :func:`os.replace` it on top of ``path``.  A reader opening
        the path mid-write sees either the prior version (intact)
        or the new version (intact) — never a half-written file.

    Parameters
    ----------
    results
        The :class:`TransportResults` to serialise.
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

    text = json.dumps(payload,
                      indent=indent,
                      ensure_ascii=False,
                      allow_nan=False,
                      sort_keys=False)

    parent = os.path.dirname(os.path.abspath(p)) or "."
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
        # Best-effort cleanup on any failure (including
        # KeyboardInterrupt).
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def parse_transport_json(path):
    """Read-side convenience re-export — delegates to
    :func:`molbuilder.parse.sidecars.transport._parse_transport_json`
    so callers have a single ``molbuilder.sidecars.transport``
    namespace for both read + write."""
    from molbuilder.parse.sidecars.transport import _parse_transport_json
    return _parse_transport_json(path)


def parse_transport_json_dict(d):
    """In-memory variant of :func:`parse_transport_json`.  Re-exports
    :func:`molbuilder.parse.sidecars.transport._parse_transport_json_dict`."""
    from molbuilder.parse.sidecars.transport import _parse_transport_json_dict
    return _parse_transport_json_dict(d)


__all__ = [
    "dump_transport_json",
    "parse_transport_json",
    "parse_transport_json_dict",
    "TransportJsonError",
    "TransportJsonNotFoundError",
    "TransportJsonMalformedError",
    "TransportJsonSchemaError",
    "TransportJsonFieldError",
]
