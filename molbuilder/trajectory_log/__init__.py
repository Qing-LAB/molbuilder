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

Back-compat: ``molbuilder.molwatch_log`` is a shim package that
re-exports from here, so existing
``from molbuilder.molwatch_log import write_initial_preview`` imports
keep working.
"""

from .emitter import MolwatchEmitter
from .format import write_initial_preview

__all__ = ["MolwatchEmitter", "write_initial_preview"]
