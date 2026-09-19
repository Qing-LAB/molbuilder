"""Sidecar FileParsers — molbuilder JSON sidecars.

Each module here defines exactly one :class:`FileParser` subclass
returning a :class:`SidecarResult`.  Adding a new sidecar kind is
two steps: new module here, import + ``register`` below.
"""

from molbuilder.parse.registry import register
from .molstruct import MolstructSidecarFileParser
from .spectra import SpectraSidecarFileParser
from .transport import TransportRecordFileParser


register(MolstructSidecarFileParser)
register(SpectraSidecarFileParser)
register(TransportRecordFileParser)

# `TransportSidecarFileParser` was DELETED 2026-09-17 and `TransportRecord-
# FileParser` above is NOT its return.  The deleted one claimed a
# `*.transport.json` only when the payload carried a top-level
# `schema_version` -- and no molbuilder version has ever written that shape:
# `sidecars.transport.dump_transport_json` had zero production callers in
# every revision since 2026-06-11.  The composite's record
# (`transport/record.py`, `schema: molbuilder/transport-result@1`) is what
# lands on disk, and that parser was written to decline exactly that.
#
# The new one reads the LIVE shape, checks that discriminator in `can_parse`,
# and exists because the reader now has a consumer: `/api/results/dir` asks
# the registry what reads each file, and the answer for a transport record
# was *nothing* -- so the Results tab parsed it in the BROWSER, the one
# result kind whose format was understood only in JavaScript.

__all__ = [
    "MolstructSidecarFileParser",
    "SpectraSidecarFileParser",
    "TransportRecordFileParser",
]
