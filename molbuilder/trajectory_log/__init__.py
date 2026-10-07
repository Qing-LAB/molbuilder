"""molbuilder.trajectory_log -- writer for ``.molwatch.log v1``.

Submodules:
    format  -- write_initial_preview (one-block preview-only writer
               used by prep's seed)
    emitter -- MolwatchEmitter (streaming class for runs with SCF
               + opt-step hooks; the PySCF script imports it from
               mb_pyscf.pyz beside the job, where molbuilder is not
               installed -- engines/pyscf.md § 3)

Both submodules emit the same v1 spec.  The reader for the format
lives at :mod:`molbuilder.parse.engines.molwatch`.
"""

from .emitter import MolwatchEmitter
from .format import write_initial_preview

__all__ = ["MolwatchEmitter",
           "write_initial_preview"]
