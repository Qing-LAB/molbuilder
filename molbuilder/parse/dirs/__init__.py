"""DirParsers — directory-level composers.

Most DirParsers register themselves at import so the dispatch
registry can route ``parse_dir(path)`` to them automatically.
The exception is :class:`BundleDirParser`: it shares the
"directory with .fdf" claim with :class:`JobDirParser` but
expresses a DIFFERENT user intent (bundle handoff vs job
decoding) and would deadlock dispatch on ambiguity if both
auto-registered.  Bundle callers invoke ``BundleDirParser.parse``
directly.
"""

from molbuilder.parse.registry import register
from .atom_metadata import atom_metadata_json_for_run_dir   # noqa: F401  -- re-export
from .bundle import BundleDirParser   # noqa: F401  -- re-export
from .job import JobDirParser, decode_run_dir   # noqa: F401  -- re-export

# Only JobDirParser auto-registers.  BundleDirParser is explicit-
# dispatch (see its module docstring for the rationale).
register(JobDirParser)

__all__ = [
    "BundleDirParser",
    "JobDirParser",
    "atom_metadata_json_for_run_dir",
    "decode_run_dir",
]
