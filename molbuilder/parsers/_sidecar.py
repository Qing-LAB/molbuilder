"""Shared helpers: read ``frozen_atoms`` indices from sources
adjacent to a trajectory file.

Two sources are supported.  Both return a set of 0-based int indices;
the empty set means "no frozen atoms known from this source."  A
caller (typically ``parsers/siesta.py``) consults them in order and
uses the first non-empty result.

  1. ``read_frozen_atoms(traj_path)`` — read from the
     ``.molstruct.json`` sidecar next to the trajectory.  Conventions
     used by ``parsers/pyscf.py::_read_sidecar_frozen_atoms`` (2026-
     06-11).  Sidecar contract lives in
     ``docs/protocols/sidecar-contract.md``.

  2. ``read_frozen_atoms_from_siesta_fdf(traj_path)`` — read from a
     sibling SIESTA ``.fdf`` input file's ``%block Geometry
     .Constraints`` block.  Added 2026-06-14 so SIESTA trajectories
     without a sidecar but with a sibling ``.fdf`` still surface
     frozen-atom indices to the trajectory inspector (otherwise the
     "Hide frozen atoms" toggle stays hidden even though the parser
     correctly emits ``max_force_constrained`` from the ``.out``
     file).  See ``docs/protocols/code-audit.md § 3.3``
     (cross-source-of-truth gap) for the audit precedent.

Both functions return the empty set on any failure — missing file,
parse error, missing block.  Frozen-atom data is optional UI metadata;
a failure here must not break trajectory loading.
"""

from __future__ import annotations

import json as _json
import os
import re
from typing import Set


def read_frozen_atoms(traj_path: str) -> Set[int]:
    """Return frozen-atom 0-based indices from the sidecar next to
    ``traj_path``.  Looks for several naming conventions used across
    engines (``<stem>.molstruct.json``, with ``_optim`` and ``_geom``
    suffix-strip fallbacks for PySCF/geomeTRIC outputs)."""
    base, fname = os.path.split(traj_path)
    stem = fname
    # PySCF / geomeTRIC convention: foo_optim.xyz pairs with foo.molstruct.json.
    if stem.endswith("_optim.xyz"):
        stem = stem[: -len("_optim.xyz")]
    elif stem.endswith(".xyz"):
        stem = stem[: -len(".xyz")]
    elif stem.endswith(".out"):
        stem = stem[: -len(".out")]
    elif stem.endswith(".molwatch.log"):
        stem = stem[: -len(".molwatch.log")]
    # Try a couple of conventions: the geomeTRIC pair often has a
    # stripped ``_geom`` suffix on the sidecar too.
    candidates = [
        os.path.join(base, f"{stem}.molstruct.json"),
    ]
    if stem.endswith("_geom"):
        candidates.append(
            os.path.join(base, stem[:-5] + ".molstruct.json"))
    sidecar_path = next(
        (p for p in candidates if os.path.isfile(p)), None)
    if sidecar_path is None:
        return set()
    try:
        with open(sidecar_path, "r", errors="replace") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return set()
    frozen = data.get("frozen_atoms")
    if not isinstance(frozen, list):
        return set()
    return {int(i) for i in frozen if isinstance(i, int)}


# Match the SIESTA ``%block`` / ``%endblock`` markers case-
# insensitively.  The block name is a single dotted token; SIESTA
# treats ``.`` and underscore the same in some places but the
# Geometry.Constraints name is canonical.
_FDF_BLOCK_START_RE = re.compile(
    r"^\s*%block\s+Geometry\.?Constraints\b", re.IGNORECASE,
)
_FDF_BLOCK_END_RE = re.compile(
    r"^\s*%endblock\s+Geometry\.?Constraints\b", re.IGNORECASE,
)

# Match the two ``position`` line forms inside a Constraints block:
#
#   position from N to M [step S]
#       a range of 1-based atom indices (SIESTA convention).
#   position N1 N2 N3 ...
#       a whitespace-separated list of 1-based atom indices.
#
# A leading ``position`` keyword is required; SIESTA also accepts
# ``stress`` (lattice constraint) and ``cellangle`` which we ignore —
# the UI only cares about per-atom position locks.
_FDF_POSITION_KEYWORD_RE = re.compile(
    r"^\s*position\b\s*(.*)$", re.IGNORECASE,
)
_FDF_POSITION_RANGE_RE = re.compile(
    r"^\s*from\s+(\d+)\s+to\s+(\d+)"
    r"(?:\s+step\s+(\d+))?\s*$",
    re.IGNORECASE,
)


def _siesta_fdf_path_for(traj_path: str) -> str | None:
    """Return the path of the SIESTA ``.fdf`` file most likely
    paired with ``traj_path``, or ``None`` if no candidate exists.

    The ``.out`` is SIESTA's stdout redirect; ``run.out`` typically
    pairs with ``run.fdf``.  The ``.molwatch.log`` is the molbuilder
    unified format; ``run.molwatch.log`` typically pairs with
    ``run.fdf`` too.  When the stem strip leaves nothing identifiable,
    fall back to any single ``*.fdf`` in the same directory (common
    for one-run-per-folder layouts).
    """
    base, fname = os.path.split(traj_path)
    if not base:
        base = "."
    stem = fname
    for suffix in (".out", ".molwatch.log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Same-stem candidate first.
    same_stem = os.path.join(base, f"{stem}.fdf")
    if os.path.isfile(same_stem):
        return same_stem
    # Fall back to any *.fdf in the directory — one-run folders are
    # the common case.  Skip when there's more than one to avoid
    # picking the wrong constraint set.
    try:
        fdfs = [
            os.path.join(base, f) for f in os.listdir(base)
            if f.lower().endswith(".fdf")
        ]
    except OSError:
        return None
    if len(fdfs) == 1:
        return fdfs[0]
    return None


def read_frozen_atoms_from_siesta_fdf(traj_path: str) -> Set[int]:
    """Return frozen-atom 0-based indices parsed from the SIESTA
    ``.fdf`` paired with ``traj_path``'s ``Geometry.Constraints``
    block.

    SIESTA's ``.fdf`` uses 1-based atom indices; this function
    converts to 0-based for parity with the sidecar contract.

    Returns the empty set on any failure (no ``.fdf`` paired, block
    absent, parse error).  The frontend treats an empty set as "no
    frozen atoms known" and hides the corresponding UI affordance.
    """
    fdf_path = _siesta_fdf_path_for(traj_path)
    if fdf_path is None:
        return set()
    try:
        with open(fdf_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return set()

    frozen_one_based: Set[int] = set()
    in_block = False
    for raw in lines:
        # Strip inline comments (``#`` starts a comment per SIESTA
        # fdf syntax) and trailing whitespace; do this BEFORE the
        # block boundary checks so a commented marker doesn't toggle
        # state.
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not in_block:
            if _FDF_BLOCK_START_RE.match(line):
                in_block = True
            continue
        if _FDF_BLOCK_END_RE.match(line):
            in_block = False
            continue
        # Inside the block: only ``position`` lines contribute atom
        # indices.  ``stress``, ``cellangle``, and similar block
        # entries are silently ignored — they don't translate to
        # per-atom freezes.
        m_kw = _FDF_POSITION_KEYWORD_RE.match(line)
        if m_kw is None:
            continue
        rest = m_kw.group(1).strip()
        if not rest:
            continue
        m_range = _FDF_POSITION_RANGE_RE.match(rest)
        if m_range is not None:
            start = int(m_range.group(1))
            stop = int(m_range.group(2))
            step = int(m_range.group(3) or 1)
            if step <= 0 or stop < start:
                continue
            for i in range(start, stop + 1, step):
                frozen_one_based.add(i)
            continue
        # Enumeration: ``position 3 7 12``.  Bail on any malformed
        # token rather than silently mis-freezing the wrong atom.
        try:
            ints = [int(tok) for tok in rest.split()]
        except ValueError:
            continue
        for i in ints:
            if i >= 1:
                frozen_one_based.add(i)

    # Convert 1-based fdf indices to 0-based.
    return {i - 1 for i in frozen_one_based}
