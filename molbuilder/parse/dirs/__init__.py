"""DirParsers — directory-level composers.

Each module here defines one :class:`DirParser` subclass and
registers it at import time.  Adding a new DirParser is two steps:
new module here, import + ``register`` below.
"""

from molbuilder.parse.registry import register
from .job import JobDirParser, decode_run_dir   # noqa: F401  -- re-export

register(JobDirParser)

__all__ = ["JobDirParser", "decode_run_dir"]
