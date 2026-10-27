"""Engine FileParsers — wrap one engine's `.out` / `.log` format.

Each PARSER module here defines exactly one :class:`FileParser` subclass: an
engine's trajectory (a :class:`TrajectoryResult`), or -- ``siesta_fdflog`` --
the engine's own record of the settings it used (an
:class:`EngineParamsResult`, `model/parse.md` § 5d.3), or -- ``siesta_epsimg``
-- an optical calculation's spectrum (a :class:`DielectricResult`, § 5f).  Adding a new engine is
two steps: new module here, import + ``register`` below.  The rest define no
parser and are imported by the ones that read with them: ``siesta_grammar``,
the SIESTA family's one table of output lines (§ 5d.5); ``tbtrans``, the
transmission rung's loop and its k-averaged text files (§ 5e);
``siesta_epsimg_reader``, the ``.EPSIMG``'s one reader, which travels (§ 5f);
``_run_ending`` and ``_helpers``.

Order matters: the registry tries parsers in insertion order, so
more-specific parsers go first.  MolwatchLogFileParser leads because
its header marker is unambiguous and never false-matches an
engine-native format.
"""

from molbuilder.parse.registry import register
from .molwatch import MolwatchLogFileParser
from .siesta_mdnc import SiestaMdNcFileParser
from .siesta import SiestaOutFileParser
from .pyscf import PySCFOutFileParser
from .siesta_fdflog import SiestaFdfLogParser
from .tbtrans_nc import TBTransNcFileParser
from .siesta_epsimg import SiestaEpsimgFileParser


register(MolwatchLogFileParser)
# Ahead of the text parsers: it claims a file by EXTENSION plus a netCDF
# magic number, so it decides in a few bytes and can never false-match a
# text .out.  Putting it after would cost every .MD.nc a content scan by
# parsers that were always going to decline it.
register(SiestaMdNcFileParser)
# By name and the HDF5 magic, as the history is: TBtrans's data file
# (`model/parse.md` § 5e).
register(TBTransNcFileParser)
# By its first line, its name shared with SIESTA's scratch beside it
# (`model/parse.md` § 5f): an optical call's spectrum.
register(SiestaEpsimgFileParser)
register(SiestaOutFileParser)
register(PySCFOutFileParser)
register(SiestaFdfLogParser)

__all__ = [
    "MolwatchLogFileParser",
    "SiestaMdNcFileParser",
    "TBTransNcFileParser",
    "SiestaEpsimgFileParser",
    "SiestaOutFileParser",
    "PySCFOutFileParser",
    "SiestaFdfLogParser",
]
