"""Engine FileParsers — wrap one engine's `.out` / `.log` format.

Each PARSER module here defines exactly one :class:`FileParser` subclass: an
engine's trajectory (a :class:`TrajectoryResult`), or -- ``siesta_fdflog`` --
the engine's own record of the settings it used (an
:class:`EngineParamsResult`, `model/parse.md` § 5d.3).  Adding a new engine is
two steps: new module here, import + ``register`` below.  The rest define no
parser and are imported by the ones that read with them: ``siesta_grammar``,
the SIESTA family's one table of output lines (§ 5d.5); ``tbtrans``, the
transmission rung's readers, which the transport record calls rather than the
registry; ``_run_ending``, ``_diag``, ``_sidecar`` and ``_helpers``.

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
from .siesta_fdflog import SiestaFdfLogParser


register(MolwatchLogFileParser)
# Ahead of the text parsers: it claims a file by EXTENSION plus a netCDF
# magic number, so it decides in a few bytes and can never false-match a
# text .out.  Putting it after would cost every .MD.nc a content scan by
# parsers that were always going to decline it.
register(SiestaMdNcFileParser)
register(SiestaOutFileParser)
register(PySCFOutFileParser)
register(SiestaFdfLogParser)

__all__ = [
    "MolwatchLogFileParser",
    "SiestaMdNcFileParser",
    "SiestaOutFileParser",
    "PySCFOutFileParser",
    "SiestaFdfLogParser",
]
