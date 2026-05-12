"""Engine-independent ``<job>.spectra.json`` parser + writer.

Reads (and writes) the structured JSON-result file produced by a
Spectra-tab engine (PySCF in v1; SIESTA reserved) and returns a
typed :class:`molbuilder.spectra.results.SpectraResults`.  The
wire shape is engine-agnostic by design (spec § 6); engine-
specific extras live under ``engine_metadata`` and round-trip
through the dataclass as a free-form ``Dict[str, Any]``.

The on-disk file is updated by the engine via atomic-replace at
each phase boundary (spec § 6.1), so a reader will never see a
torn JSON document -- but the file may be MISSING (engine hasn't
written the first checkpoint yet), or carry a different
``schema_version`` (older / newer engine), or be malformed JSON
(disk corruption, manual edit gone wrong).  Each failure mode
gets its own exception so the live-watch poller and the web
``/api/spectra/load`` endpoint can render targeted messages.

JSON safety contract (applies to BOTH read and write):

  * non-finite floats (NaN, ±Infinity) are forbidden -- they are
    valid Python ``float`` values but not valid in standard JSON,
    and downstream consumers (browsers, jq, strict parsers)
    reject them.  The reader raises
    :class:`SpectraJsonMalformedError` on encounter; the writer
    raises a ``ValueError`` BEFORE touching disk so the engine
    has to filter or null out NaN/Inf SCF energies explicitly.
  * UTF-8 with an optional BOM is accepted; output is UTF-8 with
    no BOM and ``ensure_ascii=False`` so cm⁻¹ / Å survive
    verbatim.
  * ``schema_version`` is compared with a strict type check
    (``bool`` rejected even though Python treats ``True == 1``).
  * Atomic write goes through ``os.replace`` of a sibling temp
    file so a reader holding the path mid-write either sees the
    old file or the new one, never a torn document.

This is NOT a :class:`TrajectoryParser` subclass -- those handle
per-step trajectories (SIESTA .out, geomeTRIC .xyz, molwatch logs).
The spectra JSON is a single-shot structured result, so it has
its own module + function-level API.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Union

from ..spectra.results import SCHEMA_VERSION, SpectraResults


# --------------------------------------------------------------------- #
#  Exception hierarchy                                                  #
# --------------------------------------------------------------------- #


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
    """The file exists but isn't valid JSON, isn't a JSON object
    at the top level, contains a non-standard token (``NaN`` /
    ``Infinity``), or can't be decoded as UTF-8.

    Typically disk corruption, an editor sneaking in a stray
    newline / BOM mismatch, an engine that wrote a non-finite
    float without filtering, or the file was hand-rolled.
    """


class SpectraJsonSchemaError(SpectraJsonError):
    """``schema_version`` is missing, the wrong type, or doesn't
    match :data:`SCHEMA_VERSION`.

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


# --------------------------------------------------------------------- #
#  Internals                                                            #
# --------------------------------------------------------------------- #


def _reject_nonfinite_constant(token: str):
    """``json.loads(..., parse_constant=...)`` hook.

    ``parse_constant`` fires for ``NaN``, ``Infinity``,
    ``-Infinity`` -- non-standard tokens that Python's json
    module accepts on input by default but that downstream
    consumers (browsers, jq, RFC-8259 parsers) reject.  We turn
    each into a parse failure so the user gets a clear "the
    engine wrote a non-finite value" diagnosis instead of NaN
    silently propagating into SpectraResults.
    """
    raise SpectraJsonMalformedError(
        f"non-finite numeric literal {token!r} is not valid JSON; "
        f"the engine must filter or null out NaN/Inf before "
        f"writing (likely an SCF that failed to converge produced "
        f"a divergent energy)"
    )


def _validate_schema_version(actual: Any) -> None:
    """Strict version check.  Rejects bool (``True == 1`` would
    otherwise sneak past), non-int types, and wrong-version ints.
    Centralised so the dict-input and file-input entry points
    apply the same rule."""
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise SpectraJsonSchemaError(SCHEMA_VERSION, actual)
    if actual != SCHEMA_VERSION:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, actual)


def _validate_dict_to_results(d: Dict[str, Any], where: str) -> SpectraResults:
    """Shared reconstitution + error-wrapping logic.  ``where``
    is a short label (path or ``"input dict"``) interpolated into
    the FieldError message so callers can tell at a glance which
    input failed."""
    try:
        return SpectraResults.from_dict(d)
    except KeyError as e:
        raise SpectraJsonFieldError(
            f"{where} is missing required field {e.args[0]!r}"
        ) from e
    except (TypeError, ValueError) as e:
        raise SpectraJsonFieldError(
            f"{where} has a malformed field: {e}"
        ) from e


# --------------------------------------------------------------------- #
#  Public API: read                                                     #
# --------------------------------------------------------------------- #


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
        File exists but isn't valid JSON, isn't a JSON object,
        contains a NaN/Inf token, or isn't UTF-8 decodable.
    SpectraJsonSchemaError
        ``schema_version`` missing, wrong type, or
        != :data:`SCHEMA_VERSION`.
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

    # 2. Read.  Use utf-8-sig so a BOM (some Windows editors
    #    insert one) is transparently stripped instead of poisoning
    #    the first byte of the JSON document.  UnicodeDecodeError
    #    is a ValueError, not an OSError, so we catch it separately
    #    and re-raise as MalformedError -- it means the file isn't
    #    UTF-8 at all, which is a content problem (malformed), not
    #    a filesystem problem.
    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            raw = fh.read()
    except UnicodeDecodeError as e:
        raise SpectraJsonMalformedError(
            f"{p!r} is not valid UTF-8 ({e.reason} at byte {e.start})"
        ) from e
    except OSError as e:
        raise SpectraJsonError(f"failed to read {p!r}: {e}") from e

    # 3. JSON-decode.  parse_constant rejects NaN / Infinity /
    #    -Infinity (non-standard tokens that Python's json module
    #    silently accepts by default).
    try:
        d = json.loads(raw, parse_constant=_reject_nonfinite_constant)
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

    # 4. schema_version gate.  We check BEFORE reconstitution so
    #    a v2 file doesn't trip a misleading "missing field" error
    #    in from_dict.
    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])

    # 5. Typed reconstitution.
    return _validate_dict_to_results(d, repr(p))


def parse_spectra_json_dict(d: Dict[str, Any]) -> SpectraResults:
    """In-memory variant of :func:`parse_spectra_json`.

    Same validation pipeline (schema_version + typed reconstitution)
    but skips the filesystem and the JSON decode layers.  Useful
    when the JSON arrived over the wire (e.g. a multipart POST to
    ``/api/spectra/load``) and the caller already has the dict.

    Note: NaN-rejection only fires at the json.loads layer, so this
    in-memory path does NOT re-validate finiteness -- if the caller
    constructed the dict in Python they're responsible for not
    putting non-finite floats in it.  The dataclass's
    ``__post_init__`` is the second line of defence.

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
    _validate_schema_version(d["schema_version"])
    return _validate_dict_to_results(d, "input dict")


# --------------------------------------------------------------------- #
#  Public API: write                                                    #
# --------------------------------------------------------------------- #


def dump_spectra_json(results: SpectraResults,
                      path: Union[str, "os.PathLike[str]"],
                      *,
                      indent: int = 2) -> None:
    """Write ``results`` to ``path`` via atomic rename.

    The wire-format contract that every Spectra-tab writer
    follows -- providing it as a helper here keeps engines from
    diverging on the details (NaN handling, indent, BOM, atomicity).
    The emitted script template in
    ``spectra/<engine>_engine.py::render_script`` either imports
    this directly or inlines the equivalent (see spec § 6.1).

    Behaviour:

      * ``results.to_dict()`` is encoded with ``allow_nan=False``
        -- a non-finite scalar anywhere in the payload raises
        :class:`ValueError` BEFORE any bytes hit disk, so the
        engine is forced to filter or null out NaN/Inf SCF energies
        explicitly instead of producing JSON that downstream
        consumers can't read.
      * UTF-8 without a BOM (cm⁻¹ / Å survive verbatim thanks to
        ``ensure_ascii=False``).
      * Atomic: write to ``<path>.tmp.<pid>`` first, then
        :func:`os.replace` it on top of ``path``.  A reader
        opening the path mid-write sees either the prior version
        (intact) or the new version (intact) -- never a half-
        written file.

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
        # (including KeyboardInterrupt).  We swallow errors during
        # cleanup -- the original exception is what the caller
        # cares about.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = [
    # Read
    "parse_spectra_json",
    "parse_spectra_json_dict",
    # Write
    "dump_spectra_json",
    # Exceptions
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
]
