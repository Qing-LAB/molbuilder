"""Directory-level readers — what a whole run directory says.

**No DirParser ships today** *(2026-09-04)*.  ``JobDirParser`` stood here
and decoded a run directory into an eleven-field ``JobResult``.  Ten of
those fields had no reader anywhere in the tree, and the eleventh --
``status`` -- was reached by parsing every result file to build PLOTS and
then discarding them.  Its one caller wanted the status and nothing else,
so the status is now its own answer (``job.run_status``) and the summary
is gone.

The ``DirParser`` ABC and ``registry.parse_dir`` remain: the composer
pattern (`model/parse.md` § 5) is how a directory-level answer is built
when one is needed, and this module is where it would land.

(``BundleDirParser`` lived here until 2026-08-29 -- the run-dir ->
next-calculation handoff fuse.  It retired with
calculation-to-calculation passing: a calculation that builds on a
finished result CITES it, and prep composes -- ``transport/compose.py``.)
"""

from .atom_metadata import atom_metadata_json_for_run_dir   # noqa: F401
from .job import run_status   # noqa: F401  -- re-export

__all__ = [
    "atom_metadata_json_for_run_dir",
    "run_status",
]
