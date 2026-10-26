"""Sidecar FileParsers — molbuilder JSON sidecars.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`SidecarResult`.  Adding a new sidecar kind is
two steps: new module here, import + ``register`` below.

The readers of a calculation's own records -- its description and its
plan, read through their owners' doors -- sit on those records' floors
(`task_reader`, `jobset.sweep_reader`), and the surfaces register them at
start (`calculation.register_readers`, `execution/architecture.md` § 2.1).
"""

from molbuilder.parse.registry import register
from .fc_sweep import FcSweepRecordFileParser
from .molstruct import MolstructSidecarFileParser
from .spectra import SpectraSidecarFileParser
from .transport import TransportRecordFileParser
from .vibration_phases import VibrationPhasesFileParser
from .notified import NotifiedFileParser


register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)
register(TransportRecordFileParser)
# A SIESTA vibration's displacement sweep (`engines/vibration.md` § 5.9) --
# the transport record's twin, offered at the calculation root.
register(FcSweepRecordFileParser)
# A run in phases' index -- every SIESTA call of a vibration run
# (`engines/vibration.md` § 5.11 § 7), offered in the run.
register(VibrationPhasesFileParser)
register(NotifiedFileParser)

__all__ = [
    "FcSweepRecordFileParser",
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
    "TransportRecordFileParser",
    "VibrationPhasesFileParser",
]
