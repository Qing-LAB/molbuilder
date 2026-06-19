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
    """
    warnings = [
        ParseWarning(
            source=str(source),
            line_no=getattr(w, "line_no", None),
            snippet=getattr(w, "snippet", None),
            error=getattr(w, "error", ""),
            category=getattr(w, "category", ""),
        )
        for w in (getattr(traj, "parse_warnings", None) or [])
    ]
    return TrajectoryResult(
        schema_version=1,
        parsed_at=_iso_z(),
        parser_name=parser_name,
        source=str(source),
        frames=traj.frames,
        lattice=traj.lattice,
        source_format=traj.source_format,
        run_state=traj.run_state or "unknown",
        error_message=traj.error_message,
        runtime_info=dict(getattr(traj, "runtime_info", None) or {}),
        parse_warnings=warnings,
    )
