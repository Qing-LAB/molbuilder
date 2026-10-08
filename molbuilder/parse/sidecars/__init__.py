"""Sidecar FileParsers — molbuilder JSON sidecars.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`SidecarResult`.  Adding a new sidecar kind is
two steps: new module here, import + ``register`` below.
"""

from molbuilder.parse.registry import register
from .fc_sweep import FcSweepRecordFileParser
from .job_set import JobSetSweepFileParser
from .molstruct import MolstructSidecarFileParser
from .spectra import SpectraSidecarFileParser
from .task import TransportTaskFileParser
from .transport import TransportRecordFileParser


register(JobSetSweepFileParser)
register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)
register(TransportRecordFileParser)
# A transport calculation's description -- the handle of its report,
# composed on read at the root (`engines/transport.md` § 2a.12).
register(TransportTaskFileParser)
# A SIESTA vibration's displacement sweep (`engines/vibration.md` § 5.9) --
# the transport record's twin, offered at the calculation root.
register(FcSweepRecordFileParser)

__all__ = [
    "FcSweepRecordFileParser",
    "JobSetSweepFileParser",
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
    "TransportRecordFileParser",
    "TransportTaskFileParser",
]
