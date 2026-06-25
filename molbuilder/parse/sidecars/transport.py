"""``.transport.json`` sidecar FileParser.

H1 of parse-module.md migration (was Phase D wrapper around
``molbuilder.parsers.transport_json.parse_transport_json``): absorbed
the read-side ``parse_transport_json`` + supporting validators +
exception classes directly so this module no longer imports from
``molbuilder.parsers``.  The legacy module stays in place until H4;
consumers (web blueprints) still use it for the write-side helper
``dump_transport_json`` which H2 rehomes.

The legacy ``parse_transport_json`` returns a typed
:class:`molbuilder.transport.results.TransportResults` dataclass.
The sidecar wrapper unwraps it back to a dict for the SidecarResult
payload via ``results.to_dict()`` (NOT :func:`dataclasses.asdict`).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.transport.results import SCHEMA_VERSION, TransportResults
# Canonical home for the exception classes is the write-side
# module so read and write surfaces stay in lock-step.
from molbuilder.sidecars.transport import (
    TransportJsonError,
    TransportJsonNotFoundError,
    TransportJsonMalformedError,
    TransportJsonSchemaError,
    TransportJsonFieldError,
)

from ._helpers import build_sidecar_result


# --------------------------------------------------------------------- #
#  Internals                                                             #
# --------------------------------------------------------------------- #


def _reject_nonfinite_constant(token: str):
    """``json.loads(..., parse_constant=...)`` hook.  Fires for
    symbolic ``NaN`` / ``Infinity`` / ``-Infinity`` -- non-standard
    tokens that downstream consumers reject."""
    raise TransportJsonMalformedError(
        f"non-finite numeric literal {token!r} is not valid JSON; "
        f"the engine must filter or null out NaN/Inf before writing "
        f"(a divergent NEGF iteration or unconverged bias point is "
        f"the common cause for transport data)"
    )


def _strict_finite_float(s: str) -> float:
    """``json.loads(..., parse_float=...)`` hook.  Rejects overflow-
    to-infinity from valid-syntax literals (``1e500``).  Underflow to
    0.0 is INTENTIONALLY accepted -- a vanishing transmission tail at
    deep negative energy is common."""
    v = float(s)
    if not math.isfinite(v):
        raise TransportJsonMalformedError(
            f"numeric literal {s!r} overflows to non-finite {v!r}; "
            f"the wire format requires finite IEEE 754 doubles"
        )
    return v


# Schema versions this parser can read.  v2 ONLY (2026-06-25 user
# directive): v1 sidecars omit regions + frozen_atoms, which are
# load-bearing for a transport calculation -- the device geometry has
# no meaning without the L/M/R region assignment, and the electrode
# atoms must be marked frozen for the .TSHS self-energy match.  A
# v1 sidecar is a sidecar with no boundary conditions; that's not
# something Transport can work with, so we refuse to decode it
# rather than silently producing an empty-boundary record that
# downstream code mistakes for a "free-electron sanity check."
_READABLE_SCHEMA_VERSIONS = {"2"}


def _validate_schema_version(actual: Any) -> None:
    """Version check against :data:`_READABLE_SCHEMA_VERSIONS`.  Rejects
    bool (would otherwise sneak past as a str check), non-str types,
    and unsupported version numbers."""
    if not isinstance(actual, str) or isinstance(actual, bool):
        raise TransportJsonSchemaError(SCHEMA_VERSION, actual)
    if actual not in _READABLE_SCHEMA_VERSIONS:
        raise TransportJsonSchemaError(SCHEMA_VERSION, actual)


def _validate_dict_to_results(d: Dict[str, Any],
                              where: str) -> TransportResults:
    """Shared reconstitution + error-wrapping logic."""
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
#  Read entry-point                                                      #
# --------------------------------------------------------------------- #


def _parse_transport_json(
        path: Union[str, "os.PathLike[str]"]) -> TransportResults:
    """Read a ``<job>.transport.json`` file and return a typed
    :class:`TransportResults`."""
    from molbuilder.parse._log import ParseLogger
    p = os.fspath(path)
    with ParseLogger(p, parser_name="transport-sidecar") as _scan_log:
        if not os.path.exists(p):
            _scan_log.error(f"transport.json not found at {p}")
            raise TransportJsonNotFoundError(
                f"transport.json not found at {p!r} (engine hasn't "
                f"written its result yet, or the path is wrong)"
            )

        try:
            with open(p, "r", encoding="utf-8-sig") as fh:
                raw = fh.read()
        except UnicodeDecodeError as e:
            _scan_log.error(f"not valid UTF-8: {e.reason} at byte {e.start}")
            raise TransportJsonMalformedError(
                f"{p!r} is not valid UTF-8 ({e.reason} at byte {e.start})"
            ) from e
        except OSError as e:
            _scan_log.error(f"read failed: {e}")
            raise TransportJsonError(f"failed to read {p!r}: {e}") from e

        try:
            d = json.loads(
                raw,
                parse_constant=_reject_nonfinite_constant,
                parse_float=_strict_finite_float,
            )
        except json.JSONDecodeError as e:
            _scan_log.error(
                f"not valid JSON: {e.msg} at line {e.lineno} col {e.colno}")
            raise TransportJsonMalformedError(
                f"{p!r} is not valid JSON ({e.msg} at line {e.lineno} "
                f"col {e.colno})"
            ) from e

        if not isinstance(d, dict):
            _scan_log.error(
                f"top-level JSON value is {type(d).__name__}, expected object")
            raise TransportJsonMalformedError(
                f"{p!r} top-level JSON value must be an object, got "
                f"{type(d).__name__}"
            )

        if "schema_version" not in d:
            _scan_log.error("schema_version field missing")
            raise TransportJsonSchemaError(SCHEMA_VERSION, None)
        version = d["schema_version"]
        _scan_log.info(f"schema_version={version!r}")
        _validate_schema_version(version)

        result = _validate_dict_to_results(d, repr(p))
        _scan_log.info(
            f"parsed: {len(result.energy_grid_eV)} energy points, "
            f"{len(result.regions)} regions, "
            f"{len(result.frozen_atoms)} frozen atoms, "
            f"complete={result.complete}")
        return result


def _parse_transport_json_dict(d: Dict[str, Any]) -> TransportResults:
    """In-memory variant of :func:`_parse_transport_json`.  Used when
    JSON arrived over the wire (e.g. a multipart POST to
    ``/api/transport/load``) and the caller already has the dict.

    Same validation pipeline minus the filesystem + JSON-decode
    layers.  NaN-rejection only fires at json.loads; the in-memory
    path doesn't re-validate finiteness — caller's responsibility."""
    if not isinstance(d, dict):
        raise TransportJsonMalformedError(
            f"expected a JSON object, got {type(d).__name__}"
        )
    if "schema_version" not in d:
        raise TransportJsonSchemaError(SCHEMA_VERSION, None)
    _validate_schema_version(d["schema_version"])
    return _validate_dict_to_results(d, "input dict")


# --------------------------------------------------------------------- #
#  FileParser wrapper                                                    #
# --------------------------------------------------------------------- #


class TransportSidecarFileParser(FileParser):
    """Parse a ``<stem>.transport.json`` sidecar — molbuilder
    transport-results envelope (T(E) curve / I-V / scattering region
    geometry).  Returns a :class:`SidecarResult` with
    ``schema = "transport/v<N>"``."""

    name   = "transport-json"
    label  = "molbuilder .transport.json sidecar"
    hint   = "files ending in .transport.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".transport.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        results = _parse_transport_json(path)
        # Use .to_dict() (not dataclasses.asdict) so numpy ndarrays
        # convert to JSON-trivially-serialisable lists -- ndarrays
        # leaked through asdict would break json.dumps at webhook
        # delivery + cache pickle.
        payload = results.to_dict()
        sv = payload.get("schema_version", "1")
        return build_sidecar_result(
            payload=payload,
            schema=f"transport/v{sv}",
            parser_name=cls.name,
            source=path,
        )


__all__ = [
    "TransportJsonError",
    "TransportJsonNotFoundError",
    "TransportJsonMalformedError",
    "TransportJsonSchemaError",
    "TransportJsonFieldError",
    "TransportSidecarFileParser",
]
