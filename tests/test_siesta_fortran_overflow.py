"""Regression pin for the 2026-06-14 SCF Fortran-field-overflow bug.

A divergent SCF run can blow a value past its fixed-width Fortran
format field (typically ``f10.6``).  Fortran emits the field as all
asterisks (``**********``); the pre-fix parser tried ``float("****")``,
got ``ValueError``, and dropped the entire SCF cycle from the parse
output -- losing the very rows the user needs to see WHY their job
was diverging.

User-reported failing line (BDT optimization stage 2, run 0):

    scf:    2  -152787.998333  -593325.180671  -593325.460881280.683754  6.401317**********

Two pathologies at once:

  1. ``-593325.460881280.683754`` -- two tight-packed f10.6 columns
     (handled by the grammar's tight-pack separator).
  2. trailing ``**********`` -- Fortran field overflow (the NEW
     case this test pins).

After the fix, this row parses as: ``[-152787.998333, -593325.180671,
-593325.460881, 280.683754, 6.401317, NaN]``.  Downstream consumers
treat the NaN column as "value lost in Fortran I/O" rather than
losing the entire cycle.

Same Fortran-overflow pattern can hit any fixed-width column SIESTA
emits:

  * ``outcoor`` / ``outcell`` -- atomic coords / lattice vectors
    (coords overflow only on pathological geometry blowups but the
    parser path is identical).
  * forces section -- divergent SCF blows force magnitudes.
  * ``Max`` line + constrained variant -- same source as forces.
  * ``E_KS`` line -- total energy overflow.

The fix routes EVERY single-column ``float(...)`` in the parser
through the grammar's ``fortran_float`` so the overflow path is uniform
across SCF / forces / cell / coords / energy / max-force.
"""
from __future__ import annotations

import math

import pytest

from molbuilder.parse.engines.siesta_grammar import fortran_float


# --------------------------------------------------------------------- #
#  fortran_float -- the single-token helper                       #
# --------------------------------------------------------------------- #


class TestParseFortranFloat:
    """The drop-in replacement for ``float(tok)`` that handles
    Fortran's all-asterisks overflow indicator as NaN."""

    def test_normal_float_passes_through(self):
        assert fortran_float("-152787.998333") == -152787.998333
        assert fortran_float("1.0e-6") == 1.0e-6
        assert fortran_float("0") == 0.0

    @pytest.mark.parametrize("tok", [
        "**", "***", "**********", "*" * 20,
    ])
    def test_all_asterisks_returns_nan(self, tok):
        result = fortran_float(tok)
        assert math.isnan(result), (
            f"`{tok}` should parse as NaN (Fortran field overflow); "
            f"got {result!r}"
        )

    def test_single_asterisk_raises(self):
        """A lone ``*`` is too generic to be Fortran overflow
        (which always fills the full field width >=2).  Treat as
        a real format error so the parser warns instead of silently
        coercing to NaN."""
        with pytest.raises(ValueError):
            fortran_float("*")

    @pytest.mark.parametrize("tok", [
        "", "siesta:", "abc", "12.3.4", "1.0e", "12*",
    ])
    def test_other_garbage_raises_valueerror(self, tok):
        """Any token that's neither a real float nor the
        all-asterisks pattern must raise -- the helper isn't a
        permissive ``try: float ... except: NaN`` blanket; it has
        ONE Fortran-specific escape hatch and that's it."""
        with pytest.raises(ValueError):
            fortran_float(tok)


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 2 tests here parsed SCF rows invented as text or pasted out of
# a real output (two classes) (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  JSON-safety guard on the trajectory_to_legacy_dict adapter            #
#                                                                       #
#  The 2026-06-14 follow-up BLOCKER: parser fix emitted NaN in the      #
#  scf_history dict; Python's json.dumps writes the literal ``NaN``     #
#  token (out-of-spec for JSON); browser's r.json() rejects it,         #
#  trajectory plot renders blank.  Fix: convert NaN -> None in the     #
#  adapter so the wire payload is strict-spec.                          #
# --------------------------------------------------------------------- #


class TestTrajectoryDictNanSafe:
    """End-to-end pin on the parser -> adapter -> browser pipeline.

    The adapter is the single boundary every /api/watch/* and
    /api/results/* endpoint flows through, so sanitising NaN here
    covers both the SIESTA overflow case AND any future numeric
    path that legitimately produces NaN.
    """

    def test_nan_to_none_pure_helper(self):
        import math
        from molbuilder.parse.engines._helpers import _nan_to_none
        assert _nan_to_none(1.0) == 1.0
        assert _nan_to_none(0) == 0
        assert _nan_to_none(None) is None
        assert _nan_to_none(float("nan")) is None
        out = _nan_to_none({"a": 1.0, "b": float("nan"), "c": None})
        assert out == {"a": 1.0, "b": None, "c": None}
        out = _nan_to_none([
            {"x": float("nan")}, {"x": 5.0}, [float("nan"), 1.0]
        ])
        assert out == [{"x": None}, {"x": 5.0}, [None, 1.0]]
        # Original input untouched.
        src = {"v": float("nan")}
        _ = _nan_to_none(src)
        assert math.isnan(src["v"])

    def test_nan_to_none_handles_infinity(self):
        """+inf and -inf serialise as ``Infinity`` / ``-Infinity``
        out-of-spec too.  Sanitise to None alongside NaN so the
        browser's strict JSON parser doesn't reject."""
        import json
        from molbuilder.parse.engines._helpers import _nan_to_none
        assert _nan_to_none(float("inf")) is None
        assert _nan_to_none(float("-inf")) is None
        # Round-trip a structure with all three pathological values.
        cleaned = _nan_to_none(
            [float("nan"), float("inf"), float("-inf"), 1.0])
        assert cleaned == [None, None, None, 1.0]
        # Strict JSON must succeed.
        assert json.dumps(cleaned, allow_nan=False) == "[null, null, null, 1.0]"

    def test_nan_to_none_handles_numpy_scalars(self):
        """``isinstance(np.float64(...), float)`` is False on some
        numpy/python combos so the pre-2026-06-14 path would skip
        numpy NaN entirely -- causing strict-JSON failure
        downstream.  Catch numpy scalars via numbers.Real
        duck-typing."""
        import json
        import numpy as np
        from molbuilder.parse.engines._helpers import _nan_to_none
        assert _nan_to_none(np.float64("nan")) is None
        assert _nan_to_none(np.float32("nan")) is None
        assert _nan_to_none(np.float64(2.5)) == 2.5
        # Nested.
        cleaned = _nan_to_none({
            "a": np.float64("nan"),
            "b": [np.float32("inf"), 1.0, np.float64(-3.5)],
        })
        assert cleaned["a"] is None
        assert cleaned["b"][0] is None
        assert cleaned["b"][1] == 1.0
        assert cleaned["b"][2] == -3.5
        json.dumps(cleaned, allow_nan=False)   # must not raise

    def test_nan_to_none_handles_numpy_ndarray(self):
        """A numpy ndarray that contains NaN must serialise to a
        nested list with None in place of NaN.  Catches the
        ``trajectory_to_legacy_dict.lattice`` path if a parser
        ever forgets to call ``.tolist()`` before assigning."""
        import json
        import numpy as np
        from molbuilder.parse.engines._helpers import _nan_to_none
        arr = np.array([[1.0, float("nan")],
                        [float("inf"), 3.0]])
        cleaned = _nan_to_none(arr)
        assert cleaned == [[1.0, None], [None, 3.0]]
        json.dumps(cleaned, allow_nan=False)   # must not raise

    def test_overflow_cycle_serialises_as_strict_json(self):
        """End-to-end: build a one-frame trajectory with a NaN in
        scf_history (mirroring the user's BDT case), run through
        the adapter, dump as STRICT JSON (allow_nan=False).  Must
        not raise; the round-tripped dHmax must be ``null``."""
        import json
        import numpy as np
        from molbuilder.frame import Frame, Trajectory
        from molbuilder.structure import Structure
        from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict

        struct = Structure(
            elements=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
        )
        scf = [
            {"cycle": 1, "energy": -1.0, "dDmax": 0.5, "dHmax": 0.1},
            {"cycle": 2, "energy": -2.0, "dDmax": 999.0,
             "dHmax": float("nan")},
        ]
        f = Frame(structure=struct, step_index=0,
                  energy=-2.0, scf_history=scf)
        traj = Trajectory(source_format="siesta", frames=[f])
        d = trajectory_to_legacy_dict(traj)
        # Strict JSON: pre-fix this raised "Out of range float
        # values are not JSON compliant: nan".
        payload = json.dumps(d, allow_nan=False)
        rt = json.loads(payload)
        cycle2 = rt["scf_history"][0][1]
        assert cycle2["cycle"] == 2
        assert cycle2["dHmax"] is None, (
            f"overflowed dHmax must be null on the wire; got "
            f"{cycle2['dHmax']!r}.  If you see NaN here the "
            f"adapter regressed."
        )

    def test_adapter_does_not_mutate_input(self):
        """Adapter must produce a new dict tree, not mutate the
        Trajectory it was given (callers might reuse it)."""
        import math
        import numpy as np
        from molbuilder.frame import Frame, Trajectory
        from molbuilder.structure import Structure
        from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict

        struct = Structure(
            elements=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
        )
        scf = [{"cycle": 2, "energy": -2.0, "dDmax": 999.0,
                "dHmax": float("nan")}]
        f = Frame(structure=struct, step_index=0,
                  energy=-2.0, scf_history=scf)
        traj = Trajectory(source_format="siesta", frames=[f])
        _ = trajectory_to_legacy_dict(traj)
        # Source trajectory still holds NaN -- we copied, not mutated.
        assert math.isnan(traj.frames[0].scf_history[0]["dHmax"])
