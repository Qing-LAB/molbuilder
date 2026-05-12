"""Engine-independent ``<job>.spectra.json`` parser.

Reads the structured JSON-result file written by a Spectra-tab
engine (PySCF in v1; SIESTA reserved) and returns a typed
:class:`molbuilder.spectra.results.SpectraResults`.  The wire
shape is engine-agnostic by design (spec § 6); engine-specific
extras live under ``engine_metadata`` and round-trip through the
dataclass as a free-form ``Dict[str, Any]``.

The on-disk file is updated by the engine via atomic-replace at
each phase boundary (spec § 6.1), so a reader will never see a
torn JSON document -- but the file may be MISSING (engine hasn't
written the first checkpoint yet), or carry a different
``schema_version`` (older / newer engine), or be malformed JSON
(disk corruption, manual edit gone wrong).  Each failure mode
gets its own exception so the live-watch poller and the web
``/api/spectra/load`` endpoint can render targeted messages.

This is NOT a :class:`TrajectoryParser` subclass -- those handle
per-step trajectories (SIESTA .out, geomeTRIC .xyz, molwatch logs).
The spectra JSON is a single-shot structured result, so it has
its own module + function-level API.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Union

from ..spectra.results import SCHEMA_VERSION, SpectraResults


class SpectraJsonError(Exception):
    """Base class for spectra-JSON parser failures.

    Catch this when the caller wants "any parse problem"; catch
    the specific subclasses below when the failure mode matters
    (live-watch poller distinguishes "file not yet written" from
    "file is wrong shape", for instance).
    """


class SpectraJsonNotFoundError(SpectraJsonError, FileNotFoundError):
    """The file does not exist (yet).

    Inherits :class:`FileNotFoundError` so existing ``except
    FileNotFoundError`` blocks keep working; the dual base lets
    callers also catch via :class:`SpectraJsonError` when they're
    handling all parse problems generically.
    """


class SpectraJsonMalformedError(SpectraJsonError):
    """The file exists but isn't valid JSON, or the top level is
    not a JSON object.  Typically disk corruption, an editor
    sneaking in a stray newline, or the file was hand-rolled."""


class SpectraJsonSchemaError(SpectraJsonError):
    """``schema_version`` is missing or doesn't match
    :data:`SCHEMA_VERSION`.

    A future newer engine writing schema_version=2 will trip this
    on a v1 reader; the message names both the expected and the
    actual version so an "update molbuilder" hint is straightforward
    to render.
    """

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
    :meth:`SpectraResults.from_dict` reconstitution step.

    Wraps the underlying ``KeyError`` / ``TypeError`` / ``ValueError``
    with a message that names the field path so the user knows
    where to look.  The original exception is on ``__cause__``.
    """


def parse_spectra_json(path: Union[str, "os.PathLike[str]"]) -> SpectraResults:
    """Read a ``<job>.spectra.json`` file and return a typed
    :class:`SpectraResults`.

    Parameters
    ----------
    path
        Filesystem path (str or :class:`os.PathLike`) to the JSON
        file.  Relative paths are resolved against the current
        working directory.

    Returns
    -------
    SpectraResults
        Fully validated; ready to feed into the web blueprint's
        response shape or the Methods composer.

    Raises
    ------
    SpectraJsonNotFoundError
        Path doesn't exist.  Common during live-watch before the
        engine has written the first phase checkpoint.
    SpectraJsonMalformedError
        File exists but isn't a JSON object.
    SpectraJsonSchemaError
        ``schema_version`` missing or != :data:`SCHEMA_VERSION`.
    SpectraJsonFieldError
        Required field missing / wrong type / wrong shape.  Often
        means the file is from an older molbuilder where the L1
        result shape was different.
    """
    p = os.fspath(path)

    # 1. File-existence check -- distinct exception so the live-
    #    watch poller can quietly retry instead of treating it as
    #    a load failure.
    if not os.path.exists(p):
        raise SpectraJsonNotFoundError(
            f"spectra.json not found at {p!r} (engine hasn't "
            f"written its first phase checkpoint yet, or the path "
            f"is wrong)"
        )

    # 2. Read + JSON-decode.  Atomic-replace at write time means
    #    we never see a torn file -- but we may see hand-edited
    #    junk or non-JSON content.
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as e:
        raise SpectraJsonError(f"failed to read {p!r}: {e}") from e

    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SpectraJsonMalformedError(
            f"{p!r} is not valid JSON ({e.msg} at line {e.lineno} "
            f"col {e.colno})"
        ) from e

    if not isinstance(d, dict):
        raise SpectraJsonMalformedError(
            f"{p!r} top-level JSON value must be an object, got "
            f"{type(d).__name__}"
        )

    # 3. schema_version gate.  We check BEFORE reconstitution so
    #    a v2 file doesn't trip a misleading "missing field" error
    #    in from_dict.
    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    if d["schema_version"] != SCHEMA_VERSION:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, d["schema_version"])

    # 4. Typed reconstitution.  SpectraResults.from_dict is the
    #    authority on field types + shapes; we catch its errors and
    #    re-raise with the field-name hint included so the user
    #    doesn't see a bare KeyError out of dataclass internals.
    try:
        return SpectraResults.from_dict(d)
    except KeyError as e:
        raise SpectraJsonFieldError(
            f"{p!r} is missing required field {e.args[0]!r}"
        ) from e
    except (TypeError, ValueError) as e:
        raise SpectraJsonFieldError(
            f"{p!r} has a malformed field: {e}"
        ) from e


def parse_spectra_json_dict(d: Dict[str, Any]) -> SpectraResults:
    """In-memory variant of :func:`parse_spectra_json`.

    Same validation pipeline (schema_version check + typed
    reconstitution) but skips the filesystem layer.  Useful when
    the JSON arrived over the wire (e.g. a multipart POST to
    ``/api/spectra/load``) and the caller already has the dict.

    Raises the same exception hierarchy minus
    :class:`SpectraJsonNotFoundError` (we have the dict; it's
    necessarily present).
    """
    if not isinstance(d, dict):
        raise SpectraJsonMalformedError(
            f"expected a JSON object, got {type(d).__name__}"
        )
    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    if d["schema_version"] != SCHEMA_VERSION:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, d["schema_version"])
    try:
        return SpectraResults.from_dict(d)
    except KeyError as e:
        raise SpectraJsonFieldError(
            f"missing required field {e.args[0]!r}"
        ) from e
    except (TypeError, ValueError) as e:
        raise SpectraJsonFieldError(f"malformed field: {e}") from e


__all__ = [
    "parse_spectra_json",
    "parse_spectra_json_dict",
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
]
