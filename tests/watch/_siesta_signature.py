"""Compact, deterministic signature of a parsed SIESTA Trajectory.

Used by ``test_combined_dispatch.py`` to verify that the combined-regex
driver loop produces the SAME parse output as the pre-refactor per-rule
iteration.  We don't serialise the whole Trajectory (it carries
``np.ndarray`` + ``Structure`` objects that don't round-trip cleanly)
-- just the fields a researcher would notice if they drifted.

Fields snapshotted, per frame:

  * ``step_index``        -- frame index in the trajectory.
  * ``energy``            -- E_KS in eV.
  * ``max_force``         -- post-Forces "Max <value>" reading.
  * ``forces_sum``        -- sum(forces) for each frame, rounded to
                              6 decimals.  Catches missing /
                              mis-attributed Forces blocks without
                              needing to compare the full N x 3 array.
  * ``forces_n``          -- number of force rows.
  * ``coords_sum``        -- sum of all Cartesian coordinates,
                              rounded to 6 decimals.
  * ``coords_n``          -- number of atoms.
  * ``scf_history_len``   -- number of SCF iterations recorded for
                              this frame, or 0 when absent.

Trajectory-level:

  * ``source_format``
  * ``run_state``   -- how the run ENDED (§ 2b)
  * ``scf_converged`` -- reported beside it, never folded into it
  * ``error_message``
  * ``frame_count``
  * ``runtime_info``      -- sorted dict.
  * ``lattice_sum``       -- sum(lattice) rounded to 6 decimals (or
                              None for non-periodic).
  * ``parse_warning_count``
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def signature(t: Any) -> Dict[str, Any]:
    """Return a JSON-serialisable signature of trajectory ``t``."""
    frames = []
    for f in t.frames:
        forces_sum = None
        forces_n = 0
        if f.forces is not None:
            arr = np.asarray(f.forces, dtype=float)
            forces_sum = round(float(arr.sum()), 6)
            forces_n = int(arr.shape[0])

        coords_sum = None
        coords_n = 0
        if f.structure is not None and getattr(f.structure, "coords", None) is not None:
            arr = np.asarray(f.structure.coords, dtype=float)
            coords_sum = round(float(arr.sum()), 6)
            coords_n = int(arr.shape[0])

        frames.append({
            "step_index":      int(f.step_index),
            "energy":          (None if f.energy is None
                                else round(float(f.energy), 6)),
            "max_force":       (None if f.max_force is None
                                else round(float(f.max_force), 6)),
            "forces_sum":      forces_sum,
            "forces_n":        forces_n,
            "coords_sum":      coords_sum,
            "coords_n":        coords_n,
            "scf_history_len": (0 if f.scf_history is None
                                else len(f.scf_history)),
        })

    lattice_sum = None
    if t.lattice is not None:
        lattice_sum = round(float(np.asarray(t.lattice, dtype=float).sum()),
                            6)

    return {
        "source_format":       t.source_format,
        "run_state":           t.run_state,
        # `model/parse.md` § 2b, P-S2: convergence is a REPORTED FACT and
        # therefore parser-observable, so it belongs in the signature --
        # otherwise a regression that silently stopped reporting it would
        # not show up as drift.
        "scf_converged":       getattr(t, "scf_converged", None),
        "error_message":       t.error_message,
        "frame_count":         len(t.frames),
        "runtime_info":        dict(sorted((t.runtime_info or {}).items())),
        "lattice_sum":         lattice_sum,
        "parse_warning_count": len(t.parse_warnings or []),
        "frames":              frames,
    }


def signature_json(t: Any) -> str:
    """Dump a stable, sorted-key JSON for diffing."""
    return json.dumps(signature(t), indent=2, sort_keys=True)
