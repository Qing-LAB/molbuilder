"""molbuilder.molwatch_log -- shared writer/reader for ``.molwatch.log v1``.

Submodules:
    format -- write_initial_preview + shared block writers (single source
              of truth for the file format spec)
    parser -- MolwatchLogParser for reading the same format (added in
              Phase 2.1 of the molwatch merge; not yet present)

Re-exports the public writer so existing imports
``from molbuilder.molwatch_log import write_initial_preview`` work after
the ``_molwatch_log`` module became a ``molwatch_log`` package.
"""

from .format import write_initial_preview

__all__ = ["write_initial_preview"]
