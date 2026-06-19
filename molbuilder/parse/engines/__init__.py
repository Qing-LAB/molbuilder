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
from .siesta import SiestaOutFileParser
from .pyscf import PySCFOutFileParser


register(MolwatchLogFileParser)
register(SiestaOutFileParser)
register(PySCFOutFileParser)

__all__ = [
    "MolwatchLogFileParser",
    "SiestaOutFileParser",
    "PySCFOutFileParser",
]
