"""molbuilder.trajectory_log -- writer for ``.molwatch.log v1``.

Submodules:
    format  -- write_initial_preview (one-block preview-only writer
               used by the SIESTA path)
    emitter -- MolwatchEmitter (streaming class for runs with SCF
               + opt-step hooks; inlined into generated PySCF scripts
               via inspect.getsource so the user-runnable script
               stays self-contained -- no molbuilder runtime
               dependency)

Both submodules emit the same v1 spec.  The reader for the format
lives at :mod:`molbuilder.parsers.molwatch_log`.
"""

from .emitter import MolwatchEmitter
from .format import molwatch_log_basename, write_initial_preview

__all__ = ["MolwatchEmitter", "molwatch_log_basename",
           "write_initial_preview"]
