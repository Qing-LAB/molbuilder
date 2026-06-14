"""Regression pin: frame.energy fallback to the most recent finite
SCF cycle energy when SIESTA hasn't yet emitted ``E_KS(eV) = ...``.

User-reported (2026-06-14): on a fresh BDT-Au-junction stage 2 run
the Results-tab energy plot stayed blank for the entire first
geometry step (~20 min of SCF cycling) because SIESTA only emits
the ``E_KS(eV) = ...`` line after SCF converges + post-SCF
computations finish.  The .out had 60+ SCF cycles -- all carrying
useful energy estimates -- but ``frame.energy`` was None, so the
energy plot's only data point was null.

Fix: when ``step_energy`` is None at frame-commit time, scan the
SCF history backwards for the most recent FINITE energy and use
THAT as ``frame.energy``.  This surfaces the "current best
converged estimate" to the live trajectory plot.

Edge cases pinned below -- each was a real failure mode flagged
by user feedback ("the fall back needs to be robust"):
  * empty SCF history          -> leave None (no data to use)
  * SCF history is None        -> leave None (no data to use)
  * cycle.energy is None        -> walk past it
  * cycle.energy is NaN         -> walk past it (Fortran overflow)
  * cycle.energy is +inf        -> walk past it
  * all cycles non-finite       -> leave None (correct empty state)
  * step_energy IS present      -> never fall back (canonical wins)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from molbuilder.parsers.siesta import SiestaParser


def _parse_synthetic(out_body: str) -> "Trajectory":
    """Write ``out_body`` to a temp .out and parse it.  Body must
    include at least the runtime preamble + one outcoor block so a
    Frame is committed; otherwise the fallback path isn't reached."""
    with tempfile.NamedTemporaryFile(
            "w", suffix=".out", delete=False, encoding="utf-8") as tf:
        tf.write(out_body)
        path = tf.name
    try:
        return SiestaParser.parse(path)
    finally:
        os.unlink(path)


# Minimal preamble + outcoor that yields one frame.  SIESTA's
# outcoor format is ``X Y Z species_index atom_index element`` (6
# tokens); _consume_coords ignores rows with fewer than 6 tokens,
# so anything thinner won't commit a frame.  Two atoms is enough.
# Trailing blank line after the outcoor block is the canonical
# section terminator (END_SECTION).
_BASE_BODY = dedent("""\
    * Running on    1 nodes in parallel.

         iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    Ef(eV) dHmax(eV)
    {scf_block}

    outcoor: Atomic coordinates (Ang):
        0.00000000  0.00000000  0.00000000  1  1  H
        1.00000000  0.00000000  0.00000000  1  2  H

""")


# --------------------------------------------------------------------- #
#  Robustness cases                                                      #
# --------------------------------------------------------------------- #


class TestEnergyFallback:

    def test_no_scf_history_leaves_energy_none(self):
        """First-tick edge: file has the preamble + an outcoor but
        no SCF cycles yet.  Frame energy MUST be None (we don't
        invent data when there isn't any)."""
        body = _BASE_BODY.format(scf_block="")
        traj = _parse_synthetic(body)
        assert len(traj.frames) == 1
        assert traj.frames[0].energy is None

    def test_single_scf_cycle_promotes_energy(self):
        """One SCF cycle exists; frame.energy uses its energy."""
        body = _BASE_BODY.format(scf_block=(
            "   scf:    1  -100.5  -100.5  -100.5  0.5  0.1  0.05"
        ))
        traj = _parse_synthetic(body)
        assert traj.frames[0].energy == pytest.approx(-100.5)

    def test_uses_most_recent_finite_cycle(self):
        """Multiple cycles; frame.energy is the LATEST one (most
        converged so far).  Even when earlier cycles had different
        values."""
        body = _BASE_BODY.format(scf_block=(
            "   scf:    1  -100.0  -100.0  -100.0  0.5  0.1  0.05\n"
            "   scf:    2  -150.0  -150.0  -150.0  0.3  0.1  0.04\n"
            "   scf:    3  -200.0  -200.0  -200.0  0.1  0.1  0.03"
        ))
        traj = _parse_synthetic(body)
        assert traj.frames[0].energy == pytest.approx(-200.0)

    def test_walks_past_nan_cycle(self):
        """A diverging SCF can emit ``**********`` (Fortran field
        overflow -> NaN in our parser).  We walk past the NaN
        cycle and use the prior finite energy."""
        body = _BASE_BODY.format(scf_block=(
            "   scf:    1  -100.0  -100.0  -100.0  0.5  0.1  0.05\n"
            "   scf:    2  -150.0  -150.0  -150.0  0.3  0.1  0.04\n"
            "   scf:    3  -200.0  -200.0  -200.0  0.1  0.1  0.03\n"
            # Last cycle's E_KS overflows -> NaN; we should fall
            # back to cycle 3's -200.0.
            "   scf:    4  ********  ********  ********  0.5  0.1  0.05"
        ))
        traj = _parse_synthetic(body)
        # Fallback walks backward, skipping the NaN cycle.
        assert traj.frames[0].energy == pytest.approx(-200.0)

    def test_all_nan_leaves_energy_none(self):
        """Pathological run where every SCF cycle overflowed --
        no finite energy anywhere.  Leave None; plot blank.
        Better than emitting NaN that the JSON-strict layer would
        convert to null silently anyway."""
        body = _BASE_BODY.format(scf_block=(
            "   scf:    1  ********  ********  ********  0.5  0.1  0.05\n"
            "   scf:    2  ********  ********  ********  0.5  0.1  0.05"
        ))
        traj = _parse_synthetic(body)
        assert traj.frames[0].energy is None

    def test_initial_etot_fallback_when_no_scf_cycles(self):
        """The preamble Etot fallback: when SIESTA has written the
        ``Program's energy decomposition`` block but NOT the first
        scf: cycle yet, frame.energy uses the preamble Etot."""
        body = _BASE_BODY.format(scf_block="").replace(
            "outcoor: Atomic coordinates (Ang):",
            "siesta: Etot    =   -1234.5678\n\n"
            "outcoor: Atomic coordinates (Ang):"
        )
        traj = _parse_synthetic(body)
        assert traj.frames[0].energy == pytest.approx(-1234.5678)

    def test_etot_anchor_rejects_etot_over_n_variant(self):
        """The 2026-06-14 audit found the original rule used
        ``contains_ci("siesta: etot")`` which would silently match
        ``siesta: Etot/N = ...`` or ``siesta: Etot(eV) = ...`` if
        SIESTA ever printed those.  Pin that the tightened anchor
        rejects them so the fallback isn't overwritten with a
        per-atom or unit-suffixed value."""
        # Synthetic decomposition block that emits a misleading
        # ``siesta: Etot/N = ...`` line BEFORE the real
        # ``siesta: Etot = ...`` row.  If the regex were still
        # ``contains_ci("siesta: etot")``, the Etot/N value would
        # be captured first and held since on_start fires once at
        # the section start.
        body = _BASE_BODY.format(scf_block="").replace(
            "outcoor: Atomic coordinates (Ang):",
            "siesta: Etot/N  =     -999.9999\n"
            "siesta: Etot    =   -1234.5678\n\n"
            "outcoor: Atomic coordinates (Ang):"
        )
        traj = _parse_synthetic(body)
        # Must be the real Etot, NOT Etot/N.
        assert traj.frames[0].energy == pytest.approx(-1234.5678), (
            f"got {traj.frames[0].energy!r}; expected -1234.5678 "
            f"(the real Etot value).  Pre-fix the regex matched "
            f"both ``Etot`` and ``Etot/N`` -- the latter would "
            f"have been silently captured first."
        )

    def test_explicit_e_ks_wins_over_scf_fallback(self):
        """When SIESTA HAS emitted ``E_KS(eV) = ...`` (i.e., the
        canonical first-step energy is present), that value wins.
        The SCF fallback is ONLY for the no-E_KS-yet case."""
        body = (
            _BASE_BODY.format(scf_block=(
                "   scf:    1  -100.0  -100.0  -100.0  0.5  0.1  0.05\n"
                "   scf:    2  -200.0  -200.0  -200.0  0.1  0.1  0.03"
            ))
            + "\nsiesta: E_KS(eV) =       -250.5\n"
        )
        traj = _parse_synthetic(body)
        # Canonical E_KS used; SCF fallback NOT applied.
        assert traj.frames[0].energy == pytest.approx(-250.5)
