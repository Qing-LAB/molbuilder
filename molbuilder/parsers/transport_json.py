"""Engine-independent ``<job>.transport.json`` parser + writer.

Reads (and writes) the structured JSON-result file produced by a
Transport-tab engine (TranSIESTA in v1; pyscf-negf reserved) and
returns a typed
:class:`molbuilder.transport.results.TransportResults`.  The wire
shape is engine-agnostic by design (workspace-contract for
Transport — see ``transport/results.py``); engine-specific extras
live under ``metadata`` and round-trip through the dataclass as a
free-form ``Dict[str, Any]``.

The on-disk file is written by the engine via atomic-replace
(:func:`os.replace`), so a reader will never see a torn JSON
document — but the file may be MISSING (engine hasn't finished
yet), or carry a different ``schema_version`` (older / newer
engine), or be malformed JSON (disk corruption, manual edit).
Each failure mode gets its own exception so the live-watch poller
and the web ``/api/transport/load`` endpoint can render targeted
messages.

JSON safety contract (applies to BOTH read and write):

  * non-finite floats (NaN, ±Infinity) are forbidden — they are
    valid Python ``float`` values but not valid in standard JSON
    (RFC 8259), and downstream consumers (browsers, jq, strict
    parsers) reject them.  The reader raises
    :class:`TransportJsonMalformedError` on encounter; the writer
    raises ``ValueError`` BEFORE touching disk so the engine has
    to filter or null out NaN/Inf transmission points explicitly.
  * UTF-8 with an optional BOM is accepted; output is UTF-8 with
    no BOM and ``ensure_ascii=False`` so cm⁻¹ / Å / e²/h survive
    verbatim.
  * ``schema_version`` is compared with a strict type check
    (``bool`` rejected even though Python treats ``True == 1``).
  * Atomic write goes through ``os.replace`` of a sibling temp
    file so a reader holding the path mid-write either sees the
    old file or the new one, never a torn document.

Pattern: this file mirrors :mod:`molbuilder.parsers.spectra_json`
exactly — same exception hierarchy, same finiteness gates, same
atomic-write contract.  The two parsers stay in lockstep so engine
authors only need to learn one shape.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, Union

from ..transport.results import SCHEMA_VERSION, TransportResults


# --------------------------------------------------------------------- #
#  Exception hierarchy                                                  #
# --------------------------------------------------------------------- #


class TransportJsonError(Exception):
    """Base class for transport-JSON parser failures.

    Catch this when the caller wants "any parse problem"; catch the
    specific subclasses below when the failure mode matters
    (live-watch poller distinguishes "file not yet written" from
    "file is wrong shape", for instance).
    """


class TransportJsonNotFoundError(TransportJsonError, FileNotFoundError):
    """The file does not exist (yet).

    Inherits :class:`FileNotFoundError` so existing ``except
    FileNotFoundError`` blocks keep working; the dual base lets
    callers also catch via :class:`TransportJsonError` when handling
    parse problems generically.
    """


class TransportJsonMalformedError(TransportJsonError):
    """The file exists but isn't valid JSON, isn't a JSON object at
    the top level, contains a non-standard token (``NaN`` /
    ``Infinity``), or can't be decoded as UTF-8.

    Typically disk corruption, an editor sneaking in a stray newline
    / BOM mismatch, an engine that wrote a non-finite float without
    filtering (a divergent NEGF self-consistency iteration would
    produce one), or the file was hand-rolled.
    """


class TransportJsonSchemaError(TransportJsonError):
    """``schema_version`` is missing, the wrong type, or doesn't
    match :data:`SCHEMA_VERSION`.

    A future newer engine writing schema_version="2" will trip this
    on a v1 reader; the message names both expected and actual so
    an "update molbuilder" hint is straightforward to render.
    """

    def __init__(self, expected: str, actual: Any):
        super().__init__(
            f"transport.json schema_version mismatch: expected "
            f"{expected!r}, got {actual!r}.  Either the file was "
            f"written by a different molbuilder version, or it "
            f"isn't a Transport-tab result file."
        )
        self.expected = expected
        self.actual = actual


class TransportJsonFieldError(TransportJsonError):
    """A required field was missing / had the wrong type at the
    :meth:`TransportResults.from_dict` reconstitution step.

    Wraps the underlying ``KeyError`` / ``TypeError`` / ``ValueError``
    with a message that names the field path so the user knows where
    to look.  The original exception is on ``__cause__``.
    """


# --------------------------------------------------------------------- #
#  Internals (shared with spectra_json — kept parallel by design)       #
# --------------------------------------------------------------------- #


def _reject_nonfinite_constant(token: str):
    """``json.loads(..., parse_constant=...)`` hook.

    ``parse_constant`` fires for the SYMBOLIC tokens ``NaN``,
    ``Infinity``, ``-Infinity`` — non-standard tokens that Python's
    json module accepts on input by default but that downstream
    consumers (browsers, jq, RFC-8259 parsers) reject.  Companion to
    :func:`_strict_finite_float` which catches the related case of a
    syntactically-valid numeric literal that overflows to infinity
    (``1e500``).
    """
    raise TransportJsonMalformedError(
        f"non-finite numeric literal {token!r} is not valid JSON; "
        f"the engine must filter or null out NaN/Inf before writing "
        f"(a divergent NEGF iteration or unconverged bias point is "
        f"the common cause for transport data)"
    )


def _strict_finite_float(s: str) -> float:
    """``json.loads(..., parse_float=...)`` hook.

    Defensive wrapper around ``float()`` that rejects any numeric
    literal whose decoded value isn't IEEE 754 finite.  Catches the
    overflow-to-infinity case (``1e500``) that ``parse_constant``
    does not — ``parse_constant`` only fires for symbolic
    ``Infinity`` / ``NaN`` tokens, not valid-syntax numeric literals
    that happen to overflow.

    Underflow to 0.0 (``1e-500``) is INTENTIONALLY accepted — it's
    physically equivalent to zero and is a valid IEEE 754 finite
    value (a vanishing transmission tail at deep negative energy
    is common).
    """
    v = float(s)
    if not math.isfinite(v):
        raise TransportJsonMalformedError(
            f"numeric literal {s!r} overflows to non-finite {v!r}; "
            f"the wire format requires finite IEEE 754 doubles"
        )
    return v


# Schema versions this parser can read.  The current writer always
# emits ``SCHEMA_VERSION`` from :mod:`molbuilder.transport.results`;
# older versions are accepted on read so previously-saved JSONs still
# load.  When dropping support for an old version, remove it here.
_READABLE_SCHEMA_VERSIONS = {"1"}


def _validate_schema_version(actual: Any) -> None:
    """Version check against :data:`_READABLE_SCHEMA_VERSIONS`.

    Rejects bool (would otherwise sneak past as a str check), non-str
    types, and unsupported version numbers.  Centralised so the
    dict-input and file-input entry points apply the same rule.
    """
    if not isinstance(actual, str) or isinstance(actual, bool):
        raise TransportJsonSchemaError(SCHEMA_VERSION, actual)
    if actual not in _READABLE_SCHEMA_VERSIONS:
        raise TransportJsonSchemaError(SCHEMA_VERSION, actual)


def _validate_dict_to_results(d: Dict[str, Any],
                              where: str) -> TransportResults:
    """Shared reconstitution + error-wrapping logic.  ``where`` is a
    short label (path or ``"input dict"``) interpolated into the
    FieldError message so callers can tell at a glance which input
    failed.

    The catch list covers every exception class
    :meth:`TransportResults.from_dict` can raise on a malformed
    payload — ``KeyError`` (missing field), ``TypeError`` (wrong
    scalar type), ``ValueError`` (dataclass ``__post_init__``
    rejection: shape mismatch, paired-array mismatch, etc.).
    """
    try:
        return TransportResults.from_dict(d)
    except KeyError as e:
        raise TransportJsonFieldError(
            f"{where} is missing required field {e.args[0]!r}"
        ) from e
    except (TypeError, ValueError, AttributeError) as e:
        raise TransportJsonFieldError(
            f"{where} has a malformed field: {e}"
        ) from e


# --------------------------------------------------------------------- #
#  Public API: read                                                     #
# --------------------------------------------------------------------- #


def parse_transport_json(
        path: Union[str, "os.PathLike[str]"]) -> TransportResults:
    """Read a ``<job>.transport.json`` file and return a typed
    :class:`TransportResults`.

    Parameters
    ----------
    path
        Filesystem path (str or :class:`os.PathLike`) to the JSON
        file.  Relative paths are resolved against the current
        working directory.

    Returns
    -------
    TransportResults
        Fully validated; ready to feed into the web blueprint's
        response shape or the Methods composer.

    Raises
    ------
    TransportJsonNotFoundError
        Path doesn't exist.  Common during live-watch before the
        engine has written the result file.
    TransportJsonMalformedError
        File exists but isn't valid JSON, isn't a JSON object,
        contains a NaN/Inf token, or isn't UTF-8 decodable.
    TransportJsonSchemaError
        ``schema_version`` missing, wrong type, or
        != :data:`SCHEMA_VERSION`.
    TransportJsonFieldError
        Required field missing / wrong type / wrong shape.  Often
        means the file is from an older molbuilder where the L1
        result shape was different.
    """
    p = os.fspath(path)

    if not os.path.exists(p):
        raise TransportJsonNotFoundError(
            f"transport.json not found at {p!r} (engine hasn't "
            f"written its result yet, or the path is wrong)"
        )

    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            raw = fh.read()
    except UnicodeDecodeError as e:
        raise TransportJsonMalformedError(
            f"{p!r} is not valid UTF-8 ({e.reason} at byte {e.start})"
        ) from e
    except OSError as e:
        raise TransportJsonError(f"failed to read {p!r}: {e}") from e

    try:
        d = json.loads(
            raw,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_strict_finite_float,
        )
    except json.JSONDecodeError as e:
        raise TransportJsonMalformedError(
            f"{p!r} is not valid JSON ({e.msg} at line {e.lineno} "
            f"col {e.colno})"
        ) from e

    if not isinstance(d, dict):
        raise TransportJsonMalformedError(
            f"{p!r} top-level JSON value must be an object, got "
            f"{type(d).__name__}"
        )

    if "schema_version" not in d:
        raise TransportJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])

    return _validate_dict_to_results(d, repr(p))


def parse_transport_json_dict(d: Dict[str, Any]) -> TransportResults:
    """In-memory variant of :func:`parse_transport_json`.

    Same validation pipeline (schema_version + typed reconstitution)
    but skips the filesystem and the JSON decode layers.  Useful when
    the JSON arrived over the wire (e.g. a multipart POST to
    ``/api/transport/load``) and the caller already has the dict.

    Note: NaN-rejection only fires at the json.loads layer, so this
    in-memory path does NOT re-validate finiteness — if the caller
    constructed the dict in Python they're responsible for not
    putting non-finite floats in it.  The dataclass's
    ``__post_init__`` is the second line of defence.

    Raises the same exception hierarchy minus
    :class:`TransportJsonNotFoundError`.
    """
    if not isinstance(d, dict):
        raise TransportJsonMalformedError(
            f"expected a JSON object, got {type(d).__name__}"
        )
    if "schema_version" not in d:
        raise TransportJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])
    return _validate_dict_to_results(d, "input dict")


# --------------------------------------------------------------------- #
#  Public API: write                                                    #
# --------------------------------------------------------------------- #


def dump_transport_json(results: TransportResults,
                        path: Union[str, "os.PathLike[str]"],
                        *,
                        indent: int = 2) -> None:
    """Write ``results`` to ``path`` via atomic rename.

    The wire-format contract that every Transport-tab writer follows
    — providing it as a helper here keeps engines from diverging on
    the details (NaN handling, indent, BOM, atomicity).

    Behaviour:

      * ``results.to_dict()`` is encoded with ``allow_nan=False`` — a
        non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk, so the engine
        is forced to filter or null out NaN/Inf transmission values
        explicitly.
      * UTF-8 without a BOM (cm⁻¹ / Å / G_0 survive verbatim thanks
        to ``ensure_ascii=False``).
      * Atomic: write to ``<path>.tmp.<pid>`` first, then
        :func:`os.replace` it on top of ``path``.  A reader opening
        the path mid-write sees either the prior version (intact) or
        the new version (intact) — never a half-written file.

    Parameters
    ----------
    results
        The :class:`TransportResults` to serialise.
    path
        Destination path.  Parent directory must exist.
    indent
        ``json.dumps`` indent.  Default 2 (readable + diffable); pass
        0 / ``None`` for the compact wire form.

    Raises
    ------
    ValueError
        ``results.to_dict()`` contains a non-finite float.  The
        engine must filter NaN/Inf before calling this.
    OSError
        Path can't be written (permission / no such directory / disk
        full).  The temp file is cleaned up before re-raise.
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
                # fsync; the data is still in the OS write buffer and
                # will land before the replace anyway.  Don't let a
                # quirky FS block the write.
                pass
        os.replace(tmp, p)
    except BaseException:
        # Best-effort cleanup on any failure (including KeyboardInterrupt).
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = [
    # Read
    "parse_transport_json",
    "parse_transport_json_dict",
    # Write
    "dump_transport_json",
    # Exceptions
    "TransportJsonError",
    "TransportJsonNotFoundError",
    "TransportJsonMalformedError",
    "TransportJsonSchemaError",
    "TransportJsonFieldError",
]
