"""DirParsers — directory-level composers.

DirParsers register themselves at import so the dispatch registry can
route ``parse_dir(path)`` to them automatically.  ``JobDirParser`` is
the one production parser here: it decodes a run directory whole.

(A second parser, ``BundleDirParser``, lived beside it until
2026-08-29 — the run-dir → next-calculation handoff fuse.  It retired
with calculation-to-calculation passing: a calculation that builds on
a finished result CITES it, and prep composes —
``transport/compose.py``.)
"""

from molbuilder.parse.registry import register
from .atom_metadata import atom_metadata_json_for_run_dir   # noqa: F401  -- re-export
from .job import JobDirParser, decode_run_dir   # noqa: F401  -- re-export

register(JobDirParser)

__all__ = [
    "JobDirParser",
    "atom_metadata_json_for_run_dir",
    "decode_run_dir",
]
