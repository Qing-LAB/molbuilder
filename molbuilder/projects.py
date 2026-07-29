"""Project directory layout: ``projects/<project>/<topic>/<structure>/``.

The hierarchy organises work scientifically; the innermost
``structure`` directory is a **job-layout-v1** directory (one job per
directory, all files at the same level — see
``docs/execution/job-contracts.md``).  No subdirectories inside it; tools
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

# Canonical direct children of ``projects/<project>/``.  Three flavours
# of subdir live at this level:
#
#   * **Run-topic dirs** (run-job parents).  Each ``<topic>/<structure>/``
#     is one flat job-layout-v1 run directory.  Six canonical run-topics
#     -- the "kind of analysis" axis.
#
#   * **Storage dirs** (flat file dirs).  ``structure/`` holds raw
#     structure files (``.xyz`` / ``.pdb`` / ``.cif``) the user
#     curates per project.  ``pseudopotential/`` holds the SIESTA
#     pseudos the project uses (project-local cache; the SIESTA
#     generator can populate this on demand from pseudo-dojo).
#
#   * **Free-form workspace** (``user/``).  No molbuilder rules apply
#     inside it; the user is the decider.  Use it for notes, scratch
#     files, organising experiments that don't fit the other topics.
#     ``validate_topic`` still gates *its name* at depth 1 (must be
#     in this tuple), but everything below ``user/`` is free-form.
#
# The flavour split is in docs + comments only; ``validate_topic``
# treats every entry in this tuple identically.  Ad-hoc names at depth
# 1 are rejected to keep the workflow vocabulary consistent across
# projects; if a real new analysis category emerges, extend this tuple.
CANONICAL_TOPICS: Tuple[str, ...] = (
    "structure",        # flat storage:  .xyz / .pdb / .cif
    "pseudopotential",  # flat storage:  SIESTA pseudos (project-local cache)
    "optimization",     # run topic
    "frequency",        # run topic
    "spectrum",         # run topic
    "transport",        # run topic
    "single-point",     # run topic
    "scan",             # run topic
    "user",             # free-form workspace; no rules inside it
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
            f"discovery (see docs/execution/job-contracts.md)."
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


class ProjectExists(InvalidName):
    """A project with this name already exists.  Strict-create only."""


# Per-canonical-topic README content.  Written into the bootstrap'd
# project as `<project>/<topic>/README.md` so a user navigating the
# tree (or a colleague handed a project tarball) sees what each subdir
# is for without reading docs.  Kept short -- ~5 lines max; deep dives
# live in docs/execution/job-contracts.md.
_TOPIC_READMES: Mapping[str, str] = {
    "structure": (
        "# structure/\n\n"
        "Curated input structure files for this project.\n\n"
        "Drop `.xyz`, `.pdb`, `.cif` files here.  Flat directory --\n"
        "no subdirs by convention.  Use these as starting points for\n"
        "runs in `optimization/`, `single-point/`, etc.\n"
    ),
    "pseudopotential": (
        "# pseudopotential/\n\n"
        "SIESTA pseudopotentials for this project.  Drop `.psml` /\n"
        "`.psf` files here, one per element (`H.psml`, `O.psml`, ...).\n"
        "When the SIESTA generator emits a run script for this project\n"
        "it can populate this directory from a local pseudo-dojo install\n"
        "or download as needed (see `molbuilder.pseudos` -- planned).\n"
    ),
    "optimization": (
        "# optimization/\n\n"
        "Geometry optimization runs.\n\n"
        "Each subdirectory is one job-layout-v1 flat run dir, e.g.\n"
        "`water_relax/water_relax.fdf` + `water_relax.molwatch.log` +\n"
        "`water_relax_optimized.xyz`.\n"
    ),
    "frequency": (
        "# frequency/\n\n"
        "Vibrational frequency calculations (Hessian + harmonic\n"
        "thermochemistry).  Used to confirm minima (no imaginary\n"
        "modes) or characterise transition states, and to compute\n"
        "ZPE / U / H / G / S at temperature.  Each subdirectory is\n"
        "one job-layout-v1 flat run dir.\n"
    ),
    "spectrum": (
        "# spectrum/\n\n"
        "Vibrational spectroscopy runs (Raman, IR).  Each subdirectory\n"
        "is one job-layout-v1 flat run dir, with `<job>.spectra.json`\n"
        "carrying the typed SpectraResults (see `docs/web/spectra.md`).\n"
    ),
    "transport": (
        "# transport/\n\n"
        "Electron transport calculations (TBTrans + NEGF, when wired).\n"
        "Each subdirectory is one job-layout-v1 flat run dir.\n"
    ),
    "single-point": (
        "# single-point/\n\n"
        "Single-point energy / property calculations at a fixed geometry\n"
        "(no optimization).  Common uses: higher-level energy after a\n"
        "cheaper relax (e.g. PBE -> B3LYP); HOMO/LUMO, dipole, NMR\n"
        "shieldings, partial charges on a relaxed structure.  Each\n"
        "subdirectory is one job-layout-v1 flat run dir.\n"
    ),
    "scan": (
        "# scan/\n\n"
        "Potential-energy-surface scans -- a series of single-points\n"
        "along a coordinate (bond length, angle, dihedral).  Used for\n"
        "conformational analysis (torsional barriers), reaction\n"
        "coordinates, binding curves, PES sampling.\n"
    ),
    "user": (
        "# user/\n\n"
        "Free-form workspace -- no molbuilder rules apply inside this\n"
        "directory.  Use it for notes, scratch files, organising\n"
        "experiments that don't fit the other canonical topics, or\n"
        "any non-canonical directory structure you need.  The sidebar's\n"
        "`+ New subdir` button lets you create arbitrary names here at\n"
        "any depth.\n"
    ),
}


def _project_root_readme(project: str) -> str:
    """Project-level README content; reminds the user what the layout is."""
    canonical_lines = "\n".join(
        f"  {t}/" + " " * max(1, 18 - len(t)) + "-- see " + t + "/README.md"
        for t in CANONICAL_TOPICS
    )
    return (
        f"# {project}\n\n"
        f"[Project description placeholder -- edit this file with the\n"
        f" purpose, references, and any project-specific conventions.]\n\n"
        f"## Layout\n\n"
        f"This project follows the molbuilder canonical hierarchy.  Each\n"
        f"subdirectory has a brief `README.md` explaining its purpose:\n\n"
        f"```\n{canonical_lines}\n```\n\n"
        f"See `docs/execution/job-contracts.md` for the full job-layout v1\n"
        f"convention that the run-topic subdirectories follow.\n"
    )


def populate_project_skeleton(proj: Path, *, project_name: str) -> None:
    """Populate an empty ``proj`` directory with the canonical skeleton.

    Writes the project-level ``README.md`` then creates every
    :data:`CANONICAL_TOPICS` subdir with its own ``README.md``.

    Caller-supplied ``proj`` MUST already exist and be empty.  Callers
    are responsible for the existence check (whether they got there via
    :func:`create_project_skeleton` or by some other path -- e.g., the
    web blueprint resolves the path against the picker root and creates
    the empty dir first, then calls this).

    Atomic-ish: this function does NOT rollback on partial failure --
    that's the caller's responsibility (it knows whether the dir
    pre-existed).  See :func:`create_project_skeleton` for the
    rollback-on-failure wrapper.
    """
    (proj / "README.md").write_text(_project_root_readme(project_name))
    for topic in CANONICAL_TOPICS:
        sub = proj / topic
        sub.mkdir(parents=False, exist_ok=False)
        readme = _TOPIC_READMES.get(topic)
        if readme:
            (sub / "README.md").write_text(readme)


def create_project_skeleton(project: str, *,
                            base: Optional[Path] = None) -> Path:
    """Bootstrap a new project under ``<base>/projects/`` with the full skeleton.

    Combines :func:`project_dir` path resolution + the existence check +
    :func:`populate_project_skeleton` + rollback-on-failure.

    Strict: the project dir must NOT already exist.  Raises
    :class:`ProjectExists` when it does -- callers translate to
    HTTP 409 / a user-facing message.

    For the web blueprint's path-resolution flow (which uses the
    picker root, not ``base/projects/``), call
    :func:`populate_project_skeleton` directly with a pre-created
    empty dir instead.

    Returns the absolute path of the new project root.
    """
    validate_name(project, kind="project")
    proj = project_dir(project, base=base)
    if proj.exists():
        raise ProjectExists(
            f"project {project!r} already exists at {proj!s}.  "
            f"Pick a different name; use ``+ New subdir`` inside the "
            f"existing project to add to it."
        )
    proj.mkdir(parents=True, exist_ok=False)
    try:
        populate_project_skeleton(proj, project_name=project)
    except OSError:
        import shutil
        shutil.rmtree(proj, ignore_errors=True)
        raise
    return proj


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
    "ProjectExists",
    "validate_name",
    "validate_topic",
    "projects_root",
    "project_dir",
    "topic_dir",
    "structure_dir",
    "ensure_structure_dir",
    "create_project_skeleton",
    "populate_project_skeleton",
    "list_projects",
    "list_topics",
    "list_structures",
    "find_geom_candidates",
]
