"""``.molstruct.json`` sidecar FileParser.

Phase D of parse-module.md migration: thin wrapper over the
legacy :mod:`molbuilder.parsers.molstruct_json` module.  Returns
the validated payload in a :class:`SidecarResult` with
``schema = "molstruct/v3"``.

The legacy module is kept intact for the emit / save path
(``to_dict`` / ``save`` / ``sidecar_path_for``); Phase H absorbs
or reorganises those alongside the deletion of ``parsers/``.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.parsers.molstruct_json import (
    SCHEMA_VERSION as _MOLSTRUCT_SCHEMA_VERSION,
    load as _legacy_load,
)

from ._helpers import build_sidecar_result


class MolstructSidecarFileParser(FileParser):
    """Parse a ``<stem>.molstruct.json`` sidecar — the per-atom
    region + frozen_atom metadata that rides next to a structure
    file.  Returns a :class:`SidecarResult` whose ``payload`` is
    the raw schema-validated dict and ``schema`` is the version-
    qualified discriminator (``"molstruct/v3"``)."""

    name   = "molstruct-json"
    label  = "molbuilder .molstruct.json sidecar"
    hint   = "files ending in .molstruct.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".molstruct.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        payload = _legacy_load(path)
        sv = payload.get("schema_version", _MOLSTRUCT_SCHEMA_VERSION)
        return build_sidecar_result(
            payload=payload,
            schema=f"molstruct/v{sv}",
            parser_name=cls.name,
            source=path,
        )
