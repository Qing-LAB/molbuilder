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
     (handled by the pre-existing ``_SCF_TIGHT_PACK_RE``).
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
through ``_parse_fortran_float`` so the overflow path is uniform
across SCF / forces / cell / coords / energy / max-force.
"""
from __future__ import annotations

import math

import pytest

from molbuilder.parsers.siesta import (
    _SCF_PREFIX_RE,
    _parse_fortran_float,
    _parse_scf_floats,
)


# --------------------------------------------------------------------- #
#  _parse_fortran_float -- the single-token helper                       #
# --------------------------------------------------------------------- #


class TestParseFortranFloat:
    """The drop-in replacement for ``float(tok)`` that handles
    Fortran's all-asterisks overflow indicator as NaN."""

    def test_normal_float_passes_through(self):
        assert _parse_fortran_float("-152787.998333") == -152787.998333
        assert _parse_fortran_float("1.0e-6") == 1.0e-6
        assert _parse_fortran_float("0") == 0.0

    @pytest.mark.parametrize("tok", [
        "**", "***", "**********", "*" * 20,
    ])
    def test_all_asterisks_returns_nan(self, tok):
        result = _parse_fortran_float(tok)
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
            _parse_fortran_float("*")

    @pytest.mark.parametrize("tok", [
        "", "siesta:", "abc", "12.3.4", "1.0e", "12*",
    ])
    def test_other_garbage_raises_valueerror(self, tok):
        """Any token that's neither a real float nor the
        all-asterisks pattern must raise -- the helper isn't a
        permissive ``try: float ... except: NaN`` blanket; it has
        ONE Fortran-specific escape hatch and that's it."""
        with pytest.raises(ValueError):
            _parse_fortran_float(tok)


# --------------------------------------------------------------------- #
#  _parse_scf_floats -- the user's actual failing line                   #
# --------------------------------------------------------------------- #


class TestParseSCFLine:
    """End-to-end check on the exact failing line the user
    reported.  This is the BLOCKER regression pin: if this test
    fails, the SCF parser has regressed on the divergent-cycle
    case."""

    USER_LINE = (
        "   scf:    2  -152787.998333  -593325.180671  "
        "-593325.460881280.683754  6.401317**********"
    )

    def test_user_reported_line_parses(self):
        m = _SCF_PREFIX_RE.match(self.USER_LINE)
        assert m is not None, (
            "SCF prefix regex must match the user's failing line"
        )
        iscf = int(m.group(1))
        rest = m.group(2)
        assert iscf == 2
        vals = _parse_scf_floats(rest)
        assert vals is not None, (
            "user's failing line must parse to a value list, not "
            "None; if this returns None the pre-2026-06-14 bug is "
            "back: a single overflowed column nuked the cycle."
        )

    def test_user_reported_line_values(self):
        m = _SCF_PREFIX_RE.match(self.USER_LINE)
        vals = _parse_scf_floats(m.group(2))
        assert vals is not None
        # 6 columns after iscf: Eharris, E_KS, FreeEng, dDmax, Ef, dHmax.
        assert len(vals) == 6, (
            f"closed-shell SCF should have 6 values after iscf; "
            f"got {len(vals)}: {vals}"
        )
        # First five are real numbers from the user's line.
        assert vals[0] == pytest.approx(-152787.998333)
        assert vals[1] == pytest.approx(-593325.180671)
        # Tight-pack split correctly separates the third and fourth.
        assert vals[2] == pytest.approx(-593325.460881)
        assert vals[3] == pytest.approx(280.683754)
        assert vals[4] == pytest.approx(6.401317)
        # Sixth column was ``**********`` -- overflow -> NaN.
        assert math.isnan(vals[5]), (
            f"sixth column (dHmax) was ``**********`` in the source; "
            f"must parse as NaN, got {vals[5]}"
        )


class TestParseSCFOverflowVariants:
    """Cover the overflow + tight-pack combinatorics so a future
    refactor doesn't accidentally drop one path."""

    def test_clean_closed_shell_unchanged(self):
        rest = "  -1234.5  -1234.5  -1234.5  0.001  0.5  0.001"
        vals = _parse_scf_floats(rest)
        assert vals == [-1234.5, -1234.5, -1234.5, 0.001, 0.5, 0.001]

    def test_overflow_alone(self):
        """Overflow with whitespace separators (no tight-pack
        adjacency).  All-asterisks -> NaN; otherwise unchanged."""
        rest = "  -1.0  -1.0  -1.0  0.001  **********  0.5"
        vals = _parse_scf_floats(rest)
        assert vals is not None
        assert vals[:4] == [-1.0, -1.0, -1.0, 0.001]
        assert math.isnan(vals[4])
        assert vals[5] == 0.5

    def test_tight_pack_alone(self):
        """Pre-existing case the 2026-05-28 patch fixed."""
        rest = "  -1.0  -1.0  -1.0  -1.929956131.029438  -1.0  -1.0"
        vals = _parse_scf_floats(rest)
        assert vals is not None
        # ``-1.929956`` and ``131.029438`` were tight-packed; expect
        # them as separate tokens after the SCF parser splits them.
        assert vals[3] == pytest.approx(-1.929956)
        assert vals[4] == pytest.approx(131.029438)

    def test_tight_pack_and_overflow_together(self):
        """The user's actual case: BOTH pathologies in the same
        line.  Verified above; this is the targeted regression
        guard distilled into a self-contained sample."""
        rest = (
            "  -152787.998333  -593325.180671  "
            "-593325.460881280.683754  6.401317**********"
        )
        vals = _parse_scf_floats(rest)
        assert vals is not None
        assert len(vals) == 6
        assert math.isnan(vals[5])

    def test_non_overflow_garbage_still_returns_none(self):
        """Real format mismatch (textual content where a float
        should be) must still surface as None so the caller emits
        a ParseWarning.  NaN-coercion is ONLY for the Fortran
        overflow signature."""
        rest = "  1.0  2.0  some_text  3.0  4.0  5.0"
        assert _parse_scf_floats(rest) is None


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
        from molbuilder.parsers import _nan_to_none
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

    def test_overflow_cycle_serialises_as_strict_json(self):
        """End-to-end: build a one-frame trajectory with a NaN in
        scf_history (mirroring the user's BDT case), run through
        the adapter, dump as STRICT JSON (allow_nan=False).  Must
        not raise; the round-tripped dHmax must be ``null``."""
        import json
        import numpy as np
        from molbuilder.frame import Frame, Trajectory
        from molbuilder.structure import Structure
        from molbuilder.parsers import trajectory_to_legacy_dict

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
        from molbuilder.parsers import trajectory_to_legacy_dict

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
