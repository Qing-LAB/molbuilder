"""Sidecar FileParsers — molbuilder JSON sidecars.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`SidecarResult`.  Adding a new sidecar kind is
two steps: new module here, import + ``register`` below.
"""

from molbuilder.parse.registry import register
from .molstruct import MolstructSidecarFileParser
from .spectra import SpectraSidecarFileParser


register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)

# `TransportSidecarFileParser` DELETED 2026-09-17.  It claimed a
# `*.transport.json` only when the payload carried a top-level
# `schema_version` -- and no molbuilder version has ever written that shape:
# `sidecars.transport.dump_transport_json` had zero production callers in
# every revision since 2026-06-11.  The composite's record
# (`transport/record.py`, `schema: molbuilder/transport-result@1`) is what
# lands on disk, and this parser was written to decline exactly that.

__all__ = [
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
]
