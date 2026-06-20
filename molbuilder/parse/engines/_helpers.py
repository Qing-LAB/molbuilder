"""Helpers shared across engine wrappers.

The wrappers delegate to the legacy ``molbuilder.parsers.*``
TrajectoryParser classes; the only conversion here is from
``Trajectory`` (legacy) to :class:`TrajectoryResult` (new typed
ParseResult subclass).  Once Phase H drops the legacy parsers/
package, the wrappers absorb the parse logic directly and this
helper goes away.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from molbuilder.frame import Trajectory
from molbuilder.parse.types import ParseWarning, TrajectoryResult


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def wrap_trajectory(traj: Trajectory, parser_name: str,
                    source: Path) -> TrajectoryResult:
    """Convert a legacy ``Trajectory`` into a typed
    :class:`TrajectoryResult`.

    The frames/lattice/run_state/etc. fields pass through
    unchanged.  parse_warnings get normalised from the legacy
    ``Warning`` dataclass into :class:`ParseWarning`.

    Source path is resolved to an absolute path for envelope
    consistency across phases B/C/D (post-2026-06-19 round-2 fix).
    """
    try:
        source_str = str(Path(source).resolve())
    except OSError:
        source_str = str(source)
    warnings = [
        ParseWarning(
            source=source_str,
            line_no=getattr(w, "line_no", None),
            snippet=getattr(w, "snippet", None),
            error=getattr(w, "error", ""),
            category=getattr(w, "category", ""),
        )
        for w in (getattr(traj, "parse_warnings", None) or [])
    ]
    # Round-3 BLOCKER fix: legacy Trajectory is NOT frozen; the
    # caller can mutate its frames/lattice after parse.  Copy both
    # so the returned TrajectoryResult's frozen contract holds end-
    # to-end, not just on top-level attribute reassignment.
    # ``frames`` is a list of Frame dataclasses — shallow list-copy
    # is enough because each Frame is itself constructed once per
    # parse and not mutated by downstream code.  ``lattice`` is a
    # numpy ndarray; .copy() gives a fresh buffer the caller can
    # later modify without surprising the consumer.
    frames_copy = list(traj.frames)
    lattice_copy = traj.lattice.copy() if traj.lattice is not None else None
    return TrajectoryResult(
        schema_version=1,
        parsed_at=_iso_z(),
        parser_name=parser_name,
        source=source_str,
        frames=frames_copy,
        lattice=lattice_copy,
        source_format=traj.source_format,
        run_state=traj.run_state or "unknown",
        error_message=traj.error_message,
        runtime_info=dict(getattr(traj, "runtime_info", None) or {}),
        parse_warnings=warnings,
    )
