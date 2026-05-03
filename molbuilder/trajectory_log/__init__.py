"""molbuilder.trajectory_log -- writer for ``.molwatch.log v1``.

Submodules:
    format -- write_initial_preview + shared block writers (single
              source of truth for the file format spec)

The on-disk file extension is unchanged (``.molwatch.log v1``); only
the Python module name moves.  The reader for the same format lives
at ``molbuilder.parsers.molwatch_log``, sibling to the SIESTA / PySCF
trajectory parsers.

Back-compat: ``molbuilder.molwatch_log`` is a shim package that
re-exports from here, so existing
``from molbuilder.molwatch_log import write_initial_preview`` imports
keep working.
"""

from .format import write_initial_preview

__all__ = ["write_initial_preview"]
