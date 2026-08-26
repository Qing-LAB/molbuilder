"""Helpers shared across engine wrappers.

H2 of parse-module.md migration: adds
:func:`trajectory_result_to_legacy_dict` as the canonical adapter
between a :class:`TrajectoryResult` and the legacy molwatch v1 JSON
dict the 3Dmol.js frontend consumes.  The legacy
``molbuilder.parsers.trajectory_to_legacy_dict`` still exists for
consumers that haven't migrated yet; H3 wires them onto the new
function, H4 deletes the legacy adapter.

Public surface:

* :func:`wrap_trajectory` — legacy ``Trajectory`` → typed
  :class:`TrajectoryResult`.  Used by the absorbed engine bodies in
  this directory.
* :func:`trajectory_result_to_legacy_dict` — typed
  :class:`TrajectoryResult` → 3Dmol.js JSON dict.  Used by the web
  layer (watch / results blueprints) to keep returning the same
  shape the existing client code consumes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from numbers import Real as _Real
from pathlib import Path
from typing import Any, Dict, List

from molbuilder.frame import Trajectory
from molbuilder.parse.types import ParseWarning, TrajectoryResult


_POS_INF = float("inf")
_NEG_INF = float("-inf")


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def wrap_trajectory(traj: Trajectory, parser_name: str,
                    source: Path) -> TrajectoryResult:
    """Convert a legacy ``Trajectory`` into a typed
    :class:`TrajectoryResult`.

    The frames/lattice/run_state/etc. fields pass through unchanged.
    parse_warnings get normalised from the legacy ``Warning``
    dataclass into :class:`ParseWarning`.

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
        scf_converged=getattr(traj, "scf_converged", None),
        error_message=traj.error_message,
        runtime_info=dict(getattr(traj, "runtime_info", None) or {}),
        parse_warnings=warnings,
    )


# --------------------------------------------------------------------- #
#  Legacy-dict adapter (web layer + 3Dmol.js consumer)                   #
# --------------------------------------------------------------------- #


def _nan_to_none(obj: Any) -> Any:
    """Recursively replace NaN / ±inf scalars with ``None`` for JSON
    safety.

    The JSON spec doesn't allow ``NaN`` / ``Infinity`` tokens; Python's
    :func:`json.dumps` emits them anyway (``allow_nan=True`` is the
    default), but the BROWSER's strict ``r.json()`` / ``JSON.parse``
    rejects them and the fetch throws — silently blanking the
    Results-tab view.

    NaN reaches the trajectory dict primarily from
    ``_parse_fortran_float`` (Fortran fixed-width overflow on a
    divergent SCF cycle).  Replacing with ``None`` (which serialises
    as ``null``) lets the frontend treat the overflow column as
    "value missing" — Plotly renders the gap in the trace.

    2026-06-14 robustness pass: also handles ``numpy.float64`` /
    ``numpy.float32`` / 0-d numpy arrays (their isinstance(., float)
    check returns False!) and ``numpy.ndarray``.
    """
    # Python float fast-path.  ``obj != obj`` is a stdlib-free NaN
    # check; ``not math.isfinite(obj)`` also catches ±inf which would
    # otherwise serialise as ``Infinity`` (out-of-spec).
    if isinstance(obj, float):
        if obj != obj:           # NaN
            return None
        if obj == _POS_INF or obj == _NEG_INF:
            return None
        return obj
    # Numpy scalars (float16/32/64/128, complex skipped).
    # isinstance(np.float64, float) is False on some numpy / python
    # combinations, so check numbers.Real to catch the duck-typed
    # case without importing numpy here.
    if isinstance(obj, _Real) and not isinstance(obj, bool):
        try:
            f = float(obj)
        except (TypeError, ValueError, OverflowError):
            return obj
        if f != f:
            return None
        if f == _POS_INF or f == _NEG_INF:
            return None
        # Preserve int identity for normal ints.
        return obj
    # Numpy ndarray — walk the underlying tolist().  Duck-type on
    # ``.tolist()`` so we don't need a numpy import here.
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return _nan_to_none(obj.tolist())
        except (TypeError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nan_to_none(v) for v in obj)
    return obj


def trajectory_result_to_legacy_dict(
        result: TrajectoryResult) -> Dict[str, Any]:
    """Adapter: build the molwatch v1 JSON dict from a
    :class:`TrajectoryResult`.

    The watch web layer (``molbuilder/web/blueprints/watch.py``)
    calls this to keep returning the same JSON shape the existing
    3Dmol.js client consumes.  H2 of parse-module.md migration:
    canonical home for the adapter; the legacy
    ``molbuilder.parsers.trajectory_to_legacy_dict`` re-exports this
    function so callers can switch over incrementally before H4
    deletes the legacy entry point.

    Output shape::

        {
          "frames":      [ [[el, x, y, z], ...], ... ],
          "energies":    [ float | null, ... ],
          "max_forces":  [ float | null, ... ],
          "forces":      [ [[fx, fy, fz], ...] | [], ... ],
          "iterations":  [ int, ... ],
          "lattice":     [[ax, ay, az], ...] | null,
          "scf_history": [ [ {cycle, energy, delta_E, ...}, ... ], ... ],
          "wall_clock_s": [ float | null, ... ],
          "elapsed_s":    [ float | null, ... ],
          "in_progress": [ bool, ... ] | [],
          "source_format": "siesta" | "pyscf" | "molwatch" | ...,
          "run_state":     "running" | "ended" | "stopped"
                           | "out_of_memory" | "unknown",   # parse.md 2b
          "scf_converged": true | false | null,             # P-S2: a FACT
          "error_message": str | null,
          "runtime_info":  {...},
          "parse_warnings": [ {line_no, snippet, error, category}, ... ],
          "max_forces_constrained": [...] | [],
        }

    None / empty-list mapping (matches the legacy parsers' output):
      Frame.forces      = None  → forces      = []
      Frame.scf_history = None  → scf_history = []   (when ALL frames)
      Trajectory.lattice = None → lattice     = null
      Frame.energy / max_force unset stays None → JSON null.

    Top-level collapses (size + signal optimisation):
      * ``scf_history`` collapses to ``[]`` when no frame had SCF data.
      * ``max_forces_constrained`` collapses to ``[]`` when no frame
        had a constrained value (typical: no frozen atoms in run).
      * ``in_progress`` collapses to ``[]`` when no frame is mid-write.
    """
    out_frames:     List[List[List[Any]]] = []
    out_energies:   List[Any] = []
    out_max_forces: List[Any] = []
    out_max_forces_constrained: List[Any] = []
    out_forces:     List[List[List[float]]] = []
    out_iterations: List[int] = []
    out_scf:        List[List[Dict[str, Any]]] = []
    # Two clocks, never one (parse.md § 2a).  ``wall_clock_s`` is an
    # absolute epoch and is the only one a consumer may render as a
    # date; ``elapsed_s`` counts from the run's start and is the only
    # one it may render as a duration.  A parser fills whichever its
    # engine actually reports and leaves the other None.
    out_wall_clock: List[Any] = []
    out_elapsed:    List[Any] = []
    # Per-frame "this frame is still being written" flag.  Set True
    # on the synthetic EOF-flush frame the SIESTA parser appends
    # when the file ends mid-step so the JS results-state-contract.md
    # § 4 Invariant 2 filter can omit it from plots.  The frame still
    # ships — the inspector still shows it with a "computing..." badge
    # in the inspect list — but its energy/forces aren't real numbers
    # yet and don't belong in the energy/force plot.
    out_in_progress: List[bool] = []

    for f in result.frames:
        atom_rows: List[List[Any]] = []
        for el, pos in zip(f.structure.elements, f.structure.positions):
            atom_rows.append([el, float(pos[0]), float(pos[1]), float(pos[2])])
        out_frames.append(atom_rows)

        out_energies.append(f.energy)
        out_max_forces.append(f.max_force)
        out_max_forces_constrained.append(
            getattr(f, "max_force_constrained", None))
        if f.forces is not None:
            out_forces.append([[float(v) for v in row] for row in f.forces])
        else:
            out_forces.append([])
        out_iterations.append(f.step_index)
        out_scf.append(f.scf_history if f.scf_history is not None else [])
        out_wall_clock.append(f.wall_clock_s)
        out_elapsed.append(f.elapsed_s)
        out_in_progress.append(bool(getattr(f, "in_progress", False)))

    # Legacy shape quirk: when no parser tracks SCF data for ANY
    # frame (PySCF .log absent, SIESTA had no `scf:` lines), the
    # original parsers returned a top-level empty list — not
    # [[], [], ...].  Collapse only when EVERY frame's scf_history
    # is None (the parser-says-no-scf-data signal); a frame with
    # ``scf_history=[]`` is intentional (e.g. a molwatch preview
    # block carries an empty SCF section) and stays as [] in the
    # output.
    if result.frames and all(f.scf_history is None for f in result.frames):
        out_scf = []

    if result.lattice is not None:
        lattice_out: Any = [list(row) for row in result.lattice.tolist()]
    else:
        lattice_out = None

    # 2026-06-12: ``max_forces_constrained`` collapses to a single
    # top-level empty list when NO frame had a constrained value
    # (the typical case: no frozen atoms anywhere in the run).
    if all(v is None for v in out_max_forces_constrained):
        out_max_forces_constrained = []

    # Browser-side ``r.json()`` is strict-spec: ``NaN`` and ``Infinity``
    # are not legal JSON tokens.  Python's ``json.dumps`` will happily
    # emit ``NaN`` (default ``allow_nan=True``); when the response
    # reaches the browser, the fetch throws SyntaxError and the
    # Results-tab trajectory plot renders blank.  Plotly renders
    # ``null`` as a gap in the line; that gap IS the divergence
    # diagnostic (per the 2026-06-14 BDT-stage-2 report).
    out_scf = _nan_to_none(out_scf)
    out_energies = _nan_to_none(out_energies)
    out_max_forces = _nan_to_none(out_max_forces)
    out_max_forces_constrained = _nan_to_none(out_max_forces_constrained)
    out_forces = _nan_to_none(out_forces)
    out_frames = _nan_to_none(out_frames)
    lattice_out = _nan_to_none(lattice_out)

    # When NO frame is in-progress (the typical case for a finished
    # run), collapse to a top-level empty list — same shape pattern
    # as max_forces_constrained.  The JS filter falls back to "all
    # frames plottable" when this is [] (vs. expecting per-frame
    # bools).
    if not any(out_in_progress):
        out_in_progress = []

    # THE TWO CLOCKS GO THROUGH THE SAME SIEVE AS EVERY OTHER NUMBER, and
    # BEFORE the derivation below rather than after.  `_maybe_float` turns
    # a `wall_time: nan` line into a float NaN quite happily, and NaN is
    # not a legal JSON token: the browser's `r.json()` throws and the whole
    # Results tab renders blank -- the failure the comment above describes,
    # reached through a series that was not being sieved.  (`wall_times`
    # was not either, before the split; two series now, so twice the way
    # in.)  Sieving first also means the derivation can only ever see None
    # or a finite number, so it cannot manufacture a fresh NaN by
    # subtracting one.
    out_wall_clock = _nan_to_none(out_wall_clock)
    out_elapsed = _nan_to_none(out_elapsed)

    # P-T3 (parse.md § 2a): derive `elapsed_s` from the epoch series,
    # here and nowhere else.  A run's start IS knowable from the frames
    # themselves -- it is the first frame carrying a clock reading -- so
    # an engine that reports only epochs (the molwatch emitter) still
    # yields a usable duration without any parser inventing one.
    #
    # The reverse is NOT done and must never be: `wall_clock_s` cannot
    # be recovered from `elapsed_s`, because the file does not contain
    # the missing addend.  An engine with no time of day (SIESTA .out)
    # keeps a null series, and the consumer falls back to the file's
    # mtime knowingly instead of formatting a duration as a date.
    if any(v is not None for v in out_wall_clock) and \
            all(v is None for v in out_elapsed):
        origin = next(v for v in out_wall_clock if v is not None)
        out_elapsed = [None if v is None else float(v) - float(origin)
                       for v in out_wall_clock]

    return {
        "frames":        out_frames,
        "lattice":       lattice_out,
        "iterations":    out_iterations,
        "energies":      out_energies,
        "max_forces":    out_max_forces,
        "max_forces_constrained": out_max_forces_constrained,
        "forces":        out_forces,
        "scf_history":   out_scf,
        "wall_clock_s":  out_wall_clock,
        "elapsed_s":     out_elapsed,
        "in_progress":   out_in_progress,
        "source_format": result.source_format,
        # Run-state markers from end-of-run detection.
        "run_state":     result.run_state,
        "scf_converged": getattr(result, "scf_converged", None),
        "error_message": result.error_message,
        # Runtime facts (CPU / memory / GPU / host) captured by the
        # parser from the file's header.  Same keys for every engine
        # so /results' inspector renders uniformly.  Empty dict for
        # logs from before the header emission landed.
        "runtime_info":  dict(result.runtime_info or {}),
        # Level-3 fail-soft warnings (2026-05-28).  Per-line parser
        # issues that did NOT abort the parse but the user should see.
        "parse_warnings": [
            {
                "line_no":  w.line_no,
                "snippet":  w.snippet,
                "error":    w.error,
                "category": w.category,
            }
            for w in (result.parse_warnings or [])
        ],
    }


def trajectory_to_legacy_dict(traj) -> Dict[str, Any]:
    """Adapter overload for the legacy :class:`Trajectory` shape.

    Convenience wrapper that lifts a legacy ``Trajectory`` into a
    minimal :class:`TrajectoryResult` envelope and routes through
    :func:`trajectory_result_to_legacy_dict`.  Used by tests that
    build a :class:`Trajectory` directly (via
    :class:`SiestaParser`/:class:`PySCFParser`/:class:`MolwatchLogParser`
    body parsers) and need the JSON-friendly dict shape without
    materialising a full :class:`TrajectoryResult`.
    """
    return trajectory_result_to_legacy_dict(TrajectoryResult(
        schema_version=1,
        parsed_at="",
        parser_name="legacy-adapter",
        source="",
        frames=list(traj.frames),
        lattice=traj.lattice,
        source_format=traj.source_format,
        run_state=traj.run_state or "unknown",
        scf_converged=getattr(traj, "scf_converged", None),
        error_message=traj.error_message,
        runtime_info=dict(getattr(traj, "runtime_info", None) or {}),
        parse_warnings=list(getattr(traj, "parse_warnings", None) or []),
    ))


__all__ = [
    "wrap_trajectory",
    "trajectory_result_to_legacy_dict",
    "trajectory_to_legacy_dict",
    "_nan_to_none",
]
