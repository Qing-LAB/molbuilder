"""Engine FileParsers — wrap one engine's `.out` / `.log` format.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`TrajectoryResult`.  Adding a new engine is two
steps: new module here, import + ``register`` below.

Order matters: the registry tries parsers in insertion order, so
more-specific parsers go first.  MolwatchLogParser leads because
its header marker is unambiguous and never false-matches an
engine-native format.
"""

from molbuilder.parse.registry import register
from .molwatch import MolwatchLogFileParser
from .siesta_mdnc import SiestaMdNcFileParser
from .siesta import SiestaOutFileParser
from .pyscf import PySCFOutFileParser


register(MolwatchLogFileParser)
# Ahead of the text parsers: it claims a file by EXTENSION plus a netCDF
# magic number, so it decides in a few bytes and can never false-match a
# text .out.  Putting it after would cost every .MD.nc a content scan by
# parsers that were always going to decline it.
register(SiestaMdNcFileParser)
register(SiestaOutFileParser)
register(PySCFOutFileParser)

__all__ = [
    "MolwatchLogFileParser",
    "SiestaMdNcFileParser",
    "SiestaOutFileParser",
    "PySCFOutFileParser",
]
