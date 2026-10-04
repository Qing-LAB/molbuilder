"""Directory-level readers — what a run folder's files say, read on floor 1
(`execution/architecture.md` § 2.1): how a run is doing (``job.run_status``,
asked with its launch record by ``rundir.run_state_of``), what a viewer opens
in a folder no calculation claims (``rundir.openable_in``), a run's record
(``record.run_record``) and the per-atom metadata its deck carries
(``atom_metadata``).

**This package knows no calculation.**  What a folder IS, the description it
belongs to, the label its files are named on and the run it speaks for are the
run door's, `molbuilder.runs` (floor 2), which asks these with what it read --
`runs.folder_answer` is the directory door the Results tab serves.  *(It was
``rundir.JobDirParser``, registered here as the one DirParser, until
2026-10-04: it read the description raw and guessed labels from the decks,
plan B11, B14.)*
"""

from .atom_metadata import atom_metadata_json_for_run_dir   # noqa: F401
from .job import run_status   # noqa: F401  -- re-export
from .rundir import openable_in, run_state_of   # noqa: F401

__all__ = [
    "atom_metadata_json_for_run_dir",
    "run_status",
    "openable_in",
    "run_state_of",
]
