"""Directory-level readers — what a whole run directory says.

**One DirParser ships, since 2026-09-18**: :class:`~.rundir.JobDirParser`,
step 1 of `plans/plan.md` § 5c.  It answers the four questions
`model/parse.md` § 5.0 names, each with a reader named before the field was
written, and it COMPOSES the readers that already exist here rather than
re-parsing: ``job.run_status``, ``job._enumerate_files``,
``contract.engine_of``.  The one thing it absorbed bodily is the
openable-discovery chain, which lived in ``web/blueprints/watch.py`` —
the web layer, which nothing below it may import.

**Proved before any caller moved**, which is the migration's own gate:
141/141 run directories in the checkout answer identically through the new
door and through the six functions it absorbs.

*What follows is the PREDECESSOR, deleted 2026-09-04, and is not what
shipped.*  ``JobDirParser`` stood here
and decoded a run directory into an eleven-field ``JobResult``.  Ten of
those fields had no reader anywhere in the tree, and the eleventh --
``status`` -- was reached by parsing every result file to build PLOTS and
then discarding them.  Its one caller wanted the status and nothing else,
so the status is now its own answer (``job.run_status``) and the summary
is gone.

The ``DirParser`` ABC and ``registry.parse_dir`` outlived it by two weeks
with nothing registered behind them, which is what made every consumer
reach past the registry and call a reader by name -- and what left the one
consumer that cannot import Python guessing from filenames.  That is the
gap the file above closes.

(``BundleDirParser`` lived here until 2026-08-29 -- the run-dir ->
next-calculation handoff fuse.  It retired with
calculation-to-calculation passing: a calculation that builds on a
finished result CITES it, and prep composes -- ``transport/compose.py``.)
"""

from ..registry import register
from .atom_metadata import atom_metadata_json_for_run_dir   # noqa: F401
from .job import run_status   # noqa: F401  -- re-export
from .rundir import (JobDirParser, labels_in, openable_in,  # noqa: F401
                     read_back)

# REGISTERED HERE, not in the module: "per-package ``__init__.py`` files own
# the registration order for their parsers" (`registry.register`).  With it
# registered, `parse_dir(<a run directory>)` and `detect(<a run directory>)`
# answer -- `model/parse.md` § 5's banner said they "can only raise" for want
# of a registered DirParser, which is the gap step 1 closes.
register(JobDirParser)

__all__ = [
    "atom_metadata_json_for_run_dir",
    "run_status",
    "JobDirParser",
    "openable_in",
    "labels_in",
    "read_back",
]
