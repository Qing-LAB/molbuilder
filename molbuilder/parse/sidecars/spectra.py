"""``.spectra.json`` sidecar FileParser.

H1 of parse-module.md migration (was Phase D wrapper around
``molbuilder.parsers.spectra_json.parse_spectra_json``): absorbed
the read-side ``parse_spectra_json`` + supporting validators +
exception classes directly so this module no longer imports from
``molbuilder.parsers``.  The legacy module stays in place until H4;
consumers (web blueprints) still use it for the write-side helper
``dump_spectra_json`` which H2 rehomes.

The legacy ``parse_spectra_json`` returns a typed
:class:`molbuilder.spectra.results.SpectraResults` dataclass.  The
sidecar wrapper unwraps it back to a dict for the SidecarResult
payload via ``results.to_dict()`` (NOT :func:`dataclasses.asdict`
— that preserves ndarrays which then break ``json.dumps`` at
webhook delivery + cache pickle).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.spectra.results import SCHEMA_VERSION, SpectraResults
# Canonical home for the exception classes is the write-side
# module so read and write surfaces stay in lock-step.
from molbuilder.sidecars.spectra import (
    SpectraJsonError,
    SpectraJsonNotFoundError,
    SpectraJsonMalformedError,
    SpectraJsonSchemaError,
    SpectraJsonFieldError,
)

from ._helpers import build_sidecar_result


# --------------------------------------------------------------------- #
#  Internals                                                             #
# --------------------------------------------------------------------- #


def _reject_nonfinite_constant(token: str):
    """``json.loads(..., parse_constant=...)`` hook.  Fires for symbolic
    ``NaN`` / ``Infinity`` / ``-Infinity`` -- non-standard tokens that
    Python accepts on input but downstream consumers (browsers, jq,
    RFC-8259 parsers) reject."""
    raise SpectraJsonMalformedError(
        f"non-finite numeric literal {token!r} is not valid JSON; "
        f"the engine must filter or null out NaN/Inf before "
        f"writing (likely an SCF that failed to converge produced "
        f"a divergent energy)"
    )


def _strict_finite_float(s: str) -> float:
    """``json.loads(..., parse_float=...)`` hook.  Rejects overflow-
    to-infinity from valid-syntax literals (``1e500``) that
    ``parse_constant`` does not see.  Underflow to 0.0 is accepted."""
    v = float(s)
    if not math.isfinite(v):
        raise SpectraJsonMalformedError(
            f"numeric literal {s!r} overflows to non-finite {v!r}; "
            f"the wire format requires finite IEEE 754 doubles "
            f"(a divergent SCF or unit-mismatched energy is the "
            f"common cause)"
        )
    return v


# Schema versions this parser can read.  Older versions are accepted on
# read so previously-saved JSONs still load; remove from this set when
# dropping support.  Per-version read-time handling lives in the typed-
# class ``from_dict`` methods (see ``ModeData.from_dict`` for the
# v1 -> v2 eigenvector remap).
_READABLE_SCHEMA_VERSIONS = {4}


def _validate_schema_version(actual: Any) -> None:
    """Version check against :data:`_READABLE_SCHEMA_VERSIONS`.  Rejects
    bool (``True == 1`` would otherwise sneak past), non-int types, and
    unsupported version numbers."""
    if not isinstance(actual, int) or isinstance(actual, bool):
        raise SpectraJsonSchemaError(SCHEMA_VERSION, actual)
    if actual not in _READABLE_SCHEMA_VERSIONS:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, actual)


def _validate_dict_to_results(d: Dict[str, Any],
                              where: str) -> SpectraResults:
    """Shared reconstitution + error-wrapping logic.  ``where`` is a
    short label (path or ``"input dict"``) interpolated into the
    FieldError message so callers can tell which input failed."""
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


# --------------------------------------------------------------------- #
#  Read entry-point                                                      #
# --------------------------------------------------------------------- #


def _parse_spectra_json(
        path: Union[str, "os.PathLike[str]"]) -> SpectraResults:
    """Read a ``<job>.spectra.json`` file and return a typed
    :class:`SpectraResults`.  Raises :class:`SpectraJsonNotFoundError`
    / :class:`SpectraJsonMalformedError` /
    :class:`SpectraJsonSchemaError` / :class:`SpectraJsonFieldError`
    on the corresponding failure modes."""
    p = os.fspath(path)

    if not os.path.exists(p):
        raise SpectraJsonNotFoundError(
            f"spectra.json not found at {p!r} (engine hasn't "
            f"written its first phase checkpoint yet, or the path "
            f"is wrong)"
        )

    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            raw = fh.read()
    except UnicodeDecodeError as e:
        raise SpectraJsonMalformedError(
            f"{p!r} is not valid UTF-8 ({e.reason} at byte {e.start})"
        ) from e
    except OSError as e:
        raise SpectraJsonError(f"failed to read {p!r}: {e}") from e

    try:
        d = json.loads(
            raw,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_strict_finite_float,
        )
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

    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])

    return _validate_dict_to_results(d, repr(p))


def _parse_spectra_json_dict(d: Dict[str, Any]) -> SpectraResults:
    """In-memory variant of :func:`_parse_spectra_json`.

    Used when JSON arrived over the wire (e.g. a multipart POST to
    ``/api/spectra/load``) and the caller already has the dict.
    Same validation pipeline (schema_version + typed reconstitution)
    minus the filesystem + JSON-decode layers.

    NaN-rejection only fires at the json.loads layer, so this
    in-memory path does NOT re-validate finiteness — if the caller
    constructed the dict in Python they're responsible for not
    putting non-finite floats in it.  The dataclass's
    ``__post_init__`` is the second line of defence.

    Raises the same exception hierarchy minus
    :class:`SpectraJsonNotFoundError` (we have the dict; it's
    necessarily present)."""
    if not isinstance(d, dict):
        raise SpectraJsonMalformedError(
            f"expected a JSON object, got {type(d).__name__}"
        )
    if "schema_version" not in d:
        raise SpectraJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])
    return _validate_dict_to_results(d, "input dict")


# --------------------------------------------------------------------- #
#  FileParser wrapper                                                    #
# --------------------------------------------------------------------- #


class SpectraSidecarFileParser(FileParser):
    """Parse a ``<stem>.spectra.json`` sidecar — molbuilder
    spectrum-results envelope (peaks / dipole strengths / etc.).
    Returns a :class:`SidecarResult` with
    ``schema = "spectra/v<N>"``."""

    name   = "spectra-json"
    label  = "molbuilder .spectra.json sidecar"
    hint   = "files ending in .spectra.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".spectra.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        results = _parse_spectra_json(path)
        # Use .to_dict() (not dataclasses.asdict) so numpy ndarrays
        # convert to JSON-trivially-serialisable lists -- ndarrays
        # leaked through asdict would break json.dumps at webhook
        # delivery + cache pickle.
        payload = results.to_dict()
        sv = payload.get("schema_version", 1)
        return build_sidecar_result(
            payload=payload,
            schema=f"spectra/v{sv}",
            parser_name=cls.name,
            source=path,
        )


__all__ = [
    "SpectraJsonError",
    "SpectraJsonNotFoundError",
    "SpectraJsonMalformedError",
    "SpectraJsonSchemaError",
    "SpectraJsonFieldError",
    "SpectraSidecarFileParser",
]
