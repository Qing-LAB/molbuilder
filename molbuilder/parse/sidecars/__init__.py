"""Sidecar FileParsers — molbuilder JSON sidecars.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`SidecarResult`.  Adding a new sidecar kind is
two steps: new module here, import + ``register`` below.
"""

from molbuilder.parse.registry import register
from .molstruct import MolstructSidecarFileParser
from .spectra import SpectraSidecarFileParser
from .transport import TransportSidecarFileParser


register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)
register(TransportSidecarFileParser)

__all__ = [
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
    "TransportSidecarFileParser",
]
