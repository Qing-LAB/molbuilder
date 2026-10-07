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
from .transport import TransportRecordFileParser


register(JobSetSweepFileParser)
register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)
register(TransportRecordFileParser)
# A SIESTA vibration's displacement sweep (`engines/vibration.md` § 5.9) --
# the transport record's twin, offered at the calculation root.
register(FcSweepRecordFileParser)

__all__ = [
    "FcSweepRecordFileParser",
    "JobSetSweepFileParser",
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
    "TransportRecordFileParser",
]
