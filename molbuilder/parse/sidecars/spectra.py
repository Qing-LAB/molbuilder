"""``.spectra.json`` sidecar FileParser.

The registry's entry for the file: it asks the file's one reader, the
owner's (:func:`molbuilder.sidecars.spectra.parse_spectra_json`, the gates
of ``engines/vibration.md`` § 6.7), which returns a typed
:class:`molbuilder.spectra.results.SpectraResults`.  The
sidecar wrapper unwraps it back to a dict for the SidecarResult
payload via ``results.to_dict()`` (NOT :func:`dataclasses.asdict`
— that preserves ndarrays which then break ``json.dumps``).
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
from molbuilder.runfiles import ROLE_SPECTRA, role_of
# The file's reader is its owner's (`sidecars.spectra`).
from molbuilder.sidecars.spectra import parse_spectra_json

from ._helpers import build_sidecar_result


# --------------------------------------------------------------------- #
#  FileParser wrapper                                                    #
# --------------------------------------------------------------------- #


class SpectraSidecarFileParser(FileParser):
    """Parse a ``<stem>.spectra.json`` sidecar — molbuilder
    spectrum-results envelope (peaks / dipole strengths / etc.).
    Returns a :class:`SidecarResult` with
    ``schema = "spectra/v<N>"``, ``N`` the version the file states."""

    name   = "spectra-json"
    label  = f"molbuilder {ROLE_SPECTRA} sidecar"
    hint   = f"files ending in {ROLE_SPECTRA}"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return role_of(path) == ROLE_SPECTRA and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        results = parse_spectra_json(path)
        # Use .to_dict() (not dataclasses.asdict) so numpy ndarrays
        # convert to JSON-trivially-serialisable lists -- ndarrays
        # leaked through asdict would break json.dumps.
        payload = results.to_dict()
        return build_sidecar_result(
            payload=payload,
            kind="spectra",
            parser_name=cls.name,
            source=path,
        )


__all__ = ["SpectraSidecarFileParser"]
