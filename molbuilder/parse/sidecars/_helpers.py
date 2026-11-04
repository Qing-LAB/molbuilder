"""Helpers shared across sidecar FileParser wrappers.

Each wrapper reads its record (:func:`read_record`) and builds a
:class:`SidecarResult`, its envelope and its tag (:func:`build_sidecar_result`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.types import ParseResult, SidecarResult


def read_record(path, *, schema: str, refused, what: str) -> Dict[str, Any]:
    """A record of ours, read for its parser: through `persist.read_json`, a
    JSON object, its schema checked (`persist.check_schema`) -- else
    ``refused`` (the parser's error class) naming the file; ``what`` is the
    record in words (*a transport record*)."""
    from molbuilder.persist import check_schema, read_json
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise refused(f"{Path(path).name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise refused(f"{Path(path).name}: {what} is a JSON object, got "
                      f"{type(payload).__name__}")
    try:
        check_schema(str(payload.get("schema", "")), schema,
                     label=Path(path).name)
    except ValueError as exc:
        raise refused(str(exc)) from exc
    return payload


def build_sidecar_result(payload: Dict[str, Any], kind: str,
                         parser_name: str, source: Path) -> SidecarResult:
    """Wrap a loaded sidecar dict in the typed SidecarResult, its tag
    ``<kind>/v<version>`` (``"molstruct/v17"``, ``"transport/v16"``) -- the
    discriminator consumers type-narrow by -- the version the record's own:
    its ``molbuilder/<name>@<major>`` schema's major, or the integer
    ``schema_version`` the structure and spectra sidecars state
    (`model/parse.md` § 6).  Source path is resolved to an absolute path.
    """
    from molbuilder.persist import schema_major
    stated = payload.get("schema")
    version = (schema_major(stated) if stated
               else str(payload["schema_version"]))
    return SidecarResult(
        **ParseResult.envelope(parser_name, source),
        payload=payload,
        schema=f"{kind}/v{version}",
    )
