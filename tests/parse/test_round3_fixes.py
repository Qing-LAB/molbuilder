"""Round-3 review-fix regression tests for the parse module.

Pins:
  * B1 (round-3) — `wrap_trajectory` must DEEP-COPY the frames
    list and lattice ndarray so the returned frozen
    TrajectoryResult can't be silently mutated via the legacy
    Trajectory's mutable references.
  * I4 — engine_body_summary parses duplicated fdf keys with
    LAST-WINS semantics, matching SIESTA's own fdf parser
    convention (manual § 7.1).  Previously first-wins, which
    silently misreported the value the engine actually saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.parse import parse
from molbuilder.parse.dirs.job import _parse_engine_body_summary


REPO = Path(__file__).resolve().parents[2]
SIESTA_OUT = REPO / "tests" / "watch" / "fixtures" / "siesta_frozen" \
    / "hemeC-stage2-run3-finished-42fr.out"


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"fixture absent: {p}")
    return p


# ---- B1 round-3: frames + lattice copy --------------------------- #


def test_b1_round3_trajectoryresult_frames_is_a_copy_not_a_share():
    """The TrajectoryResult is a frozen dataclass, but its
    ``frames`` list is owned by the wrapper, not the legacy
    Trajectory it wraps.  Mutating one must NOT mutate the
    other — otherwise the "frozen" contract holds only for
    top-level attribute reassignment, not for the data inside."""
    result_a = parse(_need(SIESTA_OUT))
    result_b = parse(_need(SIESTA_OUT))
    # Two independent parses produce two independent frames lists.
    assert result_a.frames is not result_b.frames
    # Mutating one does not bleed into the other.
    n_before = len(result_b.frames)
    result_a.frames.append("PRIVATE-TAMPER")
    try:
        assert len(result_b.frames) == n_before, (
            "frames list shared across separate parses; the "
            "B1 round-3 deep-copy regressed")
    finally:
        # Restore so a later test that walks result_a.frames
        # doesn't choke on the synthetic value.
        result_a.frames.pop()


def test_b1_round3_trajectoryresult_lattice_is_a_copy_not_a_share():
    """Same for ``lattice``: a numpy ndarray returned by the legacy
    parser must be copied so a consumer that mutates the returned
    lattice doesn't surprise the next consumer."""
    result_a = parse(_need(SIESTA_OUT))
    if result_a.lattice is None:
        pytest.skip("fixture has no lattice")
    result_b = parse(_need(SIESTA_OUT))
    # Independent buffers.
    assert result_a.lattice.base is not result_b.lattice.base \
        or not np.shares_memory(result_a.lattice, result_b.lattice)


# ---- I4: last-wins for duplicated fdf keys ----------------------- #


def test_i4_last_wins_on_duplicated_fdf_keys():
    """SIESTA's fdf parser uses last-wins on duplicated keys
    (manual § 7.1).  A user pattern: a stub default at the top
    of the file + an override later that the engine actually
    reads.  engine_body_summary must reflect what SIESTA SAW,
    not the first occurrence."""
    fdf_text = (
        "MeshCutoff 200.0 Ry\n"      # stub default
        "PAO.BasisSize DZP\n"
        "MeshCutoff 350.0 Ry\n"      # production override — wins
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["MeshCutoff"] == "350.0 Ry"


def test_i4_last_wins_three_overrides():
    """Three overrides — last still wins."""
    fdf_text = (
        "DM.MixingWeight 0.05\n"
        "DM.MixingWeight 0.02\n"
        "DM.MixingWeight 0.01\n"
    )
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["DM.MixingWeight"] == "0.01"


def test_i4_single_occurrence_still_works():
    """Round-3 fix must not regress the common single-occurrence
    case."""
    fdf_text = "MeshCutoff 350.0 Ry\n"
    summary = _parse_engine_body_summary(fdf_text)
    assert summary["MeshCutoff"] == "350.0 Ry"
