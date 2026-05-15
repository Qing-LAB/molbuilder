"""Project directory layout: ``projects/<project>/<topic>/<structure>/``.

The hierarchy organises work scientifically; the innermost
``structure`` directory is a **job-layout-v1** directory (one job per
directory, all files at the same level — see
``docs/protocols/job-layout.md``).  No subdirectories inside it; tools
like SIESTA write their output files at the cwd level and reference
them by basename.

Categories of work in this module:

  * **Name validation** — enforce ``[A-Za-z0-9_-]+`` so SIESTA's
    basename-discipline isn't broken by stray dots or spaces.
  * **Topic vocabulary** — the six canonical topics
    (:data:`CANONICAL_TOPICS`).  Hard-coded; ad-hoc topic names
    aren't accepted.  If a workflow needs a label outside this set,
    pick the closest one or open an issue.
  * **Path resolution** — :func:`project_dir` / :func:`topic_dir` /
    :func:`structure_dir`.  Pure functions; no I/O.
  * **Directory creation** — :func:`ensure_structure_dir` is the
    only function here that touches the filesystem to *write*.
  * **Discovery** — :func:`list_projects` / :func:`list_topics` /
    :func:`list_structures`; read-only.
  * **Output detection by convention + mtime** —
    :func:`find_geom_candidates` scans the tree for files matching
    known output-name patterns, sorted by recency.

This module does not know about scripts, conda envs, run wrappers, or
the web UI.  Those concerns live in their own modules
(``molbuilder.pyscf.input``, ``molbuilder.siesta.input``,
``molbuilder.runwrap``, ``molbuilder.web``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

_log = logging.getLogger(__name__)


PROJECTS_ROOT_NAME = "projects"

# Six canonical topics — the "kind of analysis" axis.  Hard-coded so
# the tree shape is consistent across users.  Ad-hoc topics would
# fragment the workflow vocabulary; for anything outside this set,
# pick the closest match or extend this tuple in a real release.
CANONICAL_TOPICS: Tuple[str, ...] = (
    "optimization",
    "frequency",
    "spectrum",
    "transport",
    "single-point",
    "scan",
)


# Name-validation regex.  Same character set the job-layout v1 spec
# requires for the basename (no dots, no slashes, no spaces).
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidName(ValueError):
    """A project / topic / structure name failed validation."""


def validate_name(name: str, *, kind: str = "name") -> str:
    """Reject names that would break basename-based file discovery.

    Returns the name unchanged on success; raises :class:`InvalidName`
    with a helpful message on failure.  ``kind`` is interpolated into
    the message so users can tell which field is at fault.
    """
    if not isinstance(name, str) or not name:
        raise InvalidName(f"{kind} must be a non-empty string")
    if not _NAME_PATTERN.match(name):
        raise InvalidName(
            f"{kind} {name!r} contains characters outside [A-Za-z0-9_-]; "
            f"these would clash with SIESTA's basename-based file "
            f"discovery (see docs/protocols/job-layout.md)."
        )
    return name


def validate_topic(topic: str) -> str:
    """Topic must match the canonical vocabulary (:data:`CANONICAL_TOPICS`)."""
    validate_name(topic, kind="topic")
    if topic not in CANONICAL_TOPICS:
        raise InvalidName(
            f"topic {topic!r} is not one of the canonical six: "
            f"{', '.join(CANONICAL_TOPICS)}.  Pick the closest match "
            f"or extend molbuilder.projects.CANONICAL_TOPICS."
        )
    return topic


# ---------------------------------------------------------------- #
#  Path resolution (pure functions, no I/O)                        #
# ---------------------------------------------------------------- #


def projects_root(base: Optional[Path] = None) -> Path:
    """Return the ``projects/`` directory under ``base`` (default cwd).

    Does NOT create the directory.  Read-only path resolution.
    """
    return (base if base is not None else Path.cwd()) / PROJECTS_ROOT_NAME


def project_dir(project: str, *, base: Optional[Path] = None) -> Path:
    return projects_root(base) / validate_name(project, kind="project")


def topic_dir(project: str, topic: str, *,
              base: Optional[Path] = None) -> Path:
    return project_dir(project, base=base) / validate_topic(topic)


def structure_dir(project: str, topic: str, structure: str, *,
                  base: Optional[Path] = None) -> Path:
    return topic_dir(project, topic, base=base) / \
           validate_name(structure, kind="structure")


def ensure_structure_dir(project: str, topic: str, structure: str, *,
                          base: Optional[Path] = None) -> Path:
    """Create ``<base>/projects/<project>/<topic>/<structure>/`` and
    return its path.  Idempotent (existing dirs OK)."""
    d = structure_dir(project, topic, structure, base=base)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- #
#  Discovery (read-only filesystem traversal)                       #
# ---------------------------------------------------------------- #


def list_projects(*, base: Optional[Path] = None) -> List[str]:
    """Sorted list of project names, or ``[]`` if no ``projects/`` dir.

    Directories whose names fail :func:`validate_name` (would clash
    with the SIESTA basename rule -- spaces, dots, etc.) are skipped
    with a warning logged at WARNING level so the user can tell why
    something they created by hand isn't appearing.
    """
    root = projects_root(base)
    if not root.is_dir():
        return []
    valid: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if _NAME_PATTERN.match(p.name):
            valid.append(p.name)
        else:
            _log.warning(
                "skipping %s: name fails the [A-Za-z0-9_-]+ rule "
                "and won't work with SIESTA's basename discipline",
                p,
            )
    return sorted(valid)


def list_topics(project: str, *,
                 base: Optional[Path] = None) -> List[str]:
    """Topics present under ``project``, in canonical order.

    Returns the canonical six intersected with what's on disk -- so
    the result is always a prefix of :data:`CANONICAL_TOPICS` ordering.
    Non-canonical directories under the project root are ignored.
    """
    pd = project_dir(project, base=base)
    if not pd.is_dir():
        return []
    on_disk = {p.name for p in pd.iterdir() if p.is_dir()}
    return [t for t in CANONICAL_TOPICS if t in on_disk]


def list_structures(project: str, topic: str, *,
                     base: Optional[Path] = None) -> List[str]:
    """Sorted structure names under ``project/topic/``, or ``[]``."""
    td = topic_dir(project, topic, base=base)
    if not td.is_dir():
        return []
    return sorted(p.name for p in td.iterdir()
                   if p.is_dir() and _NAME_PATTERN.match(p.name))


# ---------------------------------------------------------------- #
#  Output detection by name convention + mtime                      #
# ---------------------------------------------------------------- #


# File patterns that look like "a converged geometry output".  These
# are deliberately specific -- generic *.xyz / *.pdb would catch user
# inputs, intermediate frames, and other noise the picker shouldn't
# surface.  Add new patterns here if a new engine ships its own output
# naming convention.
_GEOM_OUTPUT_PATTERNS: Tuple[str, ...] = (
    "*_optimized.xyz",     # PySCF geomopt final-frame export
    "*.STRUCT_OUT",        # SIESTA final relaxed coords
    "*_geom_optim.xyz",    # PySCF geomeTRIC trajectory (last frame is opt)
)


def find_geom_candidates(*, base: Optional[Path] = None,
                          project: Optional[str] = None,
                          newest_first: bool = True) -> List[Path]:
    """Return paths matching known "starting geometry" name conventions.

    Scans either the whole ``projects/`` tree (``project=None``) or
    just one project subtree.  Returns paths matching one of
    :data:`_GEOM_OUTPUT_PATTERNS`, sorted by mtime descending (newest
    first) when ``newest_first``, alphabetically otherwise.

    Pure read; returns ``[]`` if the scanned root doesn't exist.

    Design intent (see ``docs/design.md`` decisions-log entry
    2026-05-14, "Four-env model" + "Projects hierarchy"): no metadata
    DB, no lineage tracker -- the picker shows files matching
    output-name conventions, sorted by recency.  A user picking a
    starting geometry for a new spectrum job sees the most recently
    optimised structures first.
    """
    if project is None:
        root = projects_root(base)
    else:
        root = project_dir(project, base=base)
    if not root.is_dir():
        return []

    # Patterns in _GEOM_OUTPUT_PATTERNS are deliberately non-overlapping
    # (no bare *.xyz / *.pdb fallback) -- one file matches at most one
    # pattern, so we don't need dedup.
    found: List[Path] = []
    for pattern in _GEOM_OUTPUT_PATTERNS:
        found.extend(root.rglob(pattern))

    if newest_first:
        # Race-safe key: if a file is deleted between rglob enumeration
        # and stat (e.g. by a SLURM scratch-cleaner), treat it as
        # "infinitely old" instead of crashing the whole sort.  Such
        # files end up at the end of the list; the caller can ignore
        # them (they'll fail again when read).
        def _mtime_or_neg_inf(p):
            try:
                return p.stat().st_mtime
            except OSError:
                return float("-inf")
        found.sort(key=_mtime_or_neg_inf, reverse=True)
    else:
        found.sort()
    return found


__all__ = [
    "PROJECTS_ROOT_NAME",
    "CANONICAL_TOPICS",
    "InvalidName",
    "validate_name",
    "validate_topic",
    "projects_root",
    "project_dir",
    "topic_dir",
    "structure_dir",
    "ensure_structure_dir",
    "list_projects",
    "list_topics",
    "list_structures",
    "find_geom_candidates",
]
