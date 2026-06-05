"""Makov-Payne image-charge correction for charged supercells.

Computes the leading-order finite-size correction for the total energy
of a charged molecule in a periodic supercell:

.. math::

    \\Delta E_{MP} = \\frac{\\alpha\\, q^2}{2\\, \\varepsilon_r\\, L}

where :math:`\\alpha` is the Madelung constant of the lattice
(``2.8373`` for simple cubic), :math:`q` is the total charge of the
unit cell in units of |e|, :math:`\\varepsilon_r` is the relative
permittivity of the medium (1 in vacuum), and :math:`L` is the
characteristic supercell length (taken here as
:math:`L = V^{1/3}` so a slightly non-cubic cell still gets a
reasonable estimate).

Convention: the **corrected** total energy is

.. math::

    E_\\text{tot,corr} = E_\\text{tot,raw} - \\Delta E_{MP}

i.e. the correction is *subtracted* from the SIESTA total — this
removes the spurious electrostatic self-image interaction that the
uniform compensating background introduces.  See Makov & Payne,
Phys. Rev. B 51, 4014 (1995).

This module gives molbuilder two related artefacts when the user
emits a SIESTA input deck with ``NetCharge != 0``:

1. :func:`compute_correction` — the bare formula, used by
   :mod:`molbuilder.validation` so the warn message can quote a
   numeric estimate alongside the qualitative caveat.
2. :func:`render_correction_script` — a small self-contained Python
   script the wrapper writes next to the FDF.  Once SIESTA has run
   the script reads the ``.out``, extracts the converged total
   energy, computes :math:`\\Delta E_{MP}` for the *actual* cell
   that SIESTA used (which may differ from what the user typed
   when ``cell_padding`` auto-bumps), and prints the corrected
   value.

Why a post-process script rather than an in-FDF tweak?  SIESTA has
no native Makov-Payne keyword.  The correction is a single-number
energy shift; it has zero effect on geometry, density, or any other
output of the SCF — it is purely a finite-size cosmetic.  Emitting
a separate script keeps the SIESTA input bit-for-bit standard while
giving the user a one-command way to extract the corrected number.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

# Cubic-cell Madelung constant (the most common molbuilder case:
# vacuum padding around a small molecule).  For tetragonal /
# hexagonal supercells the value differs slightly; ``alpha`` is
# exposed as a parameter so callers with a non-cubic cell can
# override.  See Leslie & Gillan, J. Phys. C 18, 973 (1985) and
# Schultz, PRB 60, 1551 (1999) for the cell-shape table.
MADELUNG_CUBIC: float = 2.8373

# Hartree-to-eV conversion (CODATA 2018).  The correction formula
# is naturally in atomic units (Hartree per electron²/Bohr); we
# convert L from Angstrom and the result to eV so it matches what
# SIESTA writes in the .out file.
HARTREE_EV: float = 27.211386245988
BOHR_ANGSTROM: float = 0.529177210903


def compute_correction(
    q: int,
    L_angstrom: float,
    epsilon_r: float = 1.0,
    madelung: float = MADELUNG_CUBIC,
) -> float:
    """Compute the Makov-Payne correction in eV.

    Parameters
    ----------
    q
        Net charge of the supercell in units of |e|.  May be
        negative for anions.
    L_angstrom
        Characteristic supercell length in Å.  For a cubic cell
        this is the side length; for a non-cubic cell pass
        :math:`V^{1/3}` (or set ``madelung`` to the appropriate
        shape constant).
    epsilon_r
        Relative permittivity of the surrounding medium.  ``1.0``
        for vacuum (the typical molbuilder case); use the solvent
        ε if the SIESTA calculation already includes implicit
        solvation.
    madelung
        Madelung constant for the supercell shape.  Defaults to the
        simple-cubic value 2.8373.

    Returns
    -------
    float
        The correction :math:`\\Delta E_{MP}` in eV.  *Subtract*
        this from the raw SIESTA total energy to get the
        finite-size-corrected value.
    """
    if L_angstrom <= 0:
        raise ValueError(
            f"L_angstrom must be positive; got {L_angstrom}")
    if epsilon_r <= 0:
        raise ValueError(
            f"epsilon_r must be positive; got {epsilon_r}")
    # Formula in atomic units: ΔE = α q² / (2 ε_r L) with L in Bohr,
    # ΔE in Hartree.  Convert L from Å to Bohr, then Hartree to eV.
    L_bohr = L_angstrom / BOHR_ANGSTROM
    dE_hartree = madelung * (q * q) / (2.0 * epsilon_r * L_bohr)
    return dE_hartree * HARTREE_EV


def _cell_volume_angstrom3(cell: List[List[float]]) -> float:
    """Volume of a 3×3 lattice in Å³.  Used to derive an effective
    L = V^(1/3) for the correction when the cell isn't strictly
    cubic.
    """
    a, b, c = cell
    # |a · (b × c)|
    bx = b[1] * c[2] - b[2] * c[1]
    by = b[2] * c[0] - b[0] * c[2]
    bz = b[0] * c[1] - b[1] * c[0]
    return abs(a[0] * bx + a[1] * by + a[2] * bz)


def effective_L(cell: List[List[float]]) -> float:
    """Effective supercell side length L = V^(1/3) in Å for use
    with the cubic Madelung constant.  For a strictly cubic cell
    this is the side length; for a slightly distorted cell it
    gives the best-shape-matched estimate."""
    V = _cell_volume_angstrom3(cell)
    if V <= 0:
        raise ValueError("Lattice volume must be positive")
    return V ** (1.0 / 3.0)


def render_correction_script(
    system_label: str,
    q: int,
    epsilon_r: float = 1.0,
) -> str:
    """Generate a self-contained Python post-process script that
    reads ``<system_label>.out``, extracts the converged total
    energy, computes the Makov-Payne correction for the actual
    cell SIESTA used, and prints the corrected value.

    The script takes ``--epsilon`` on the command line so a user
    re-running with implicit solvation can override the default
    vacuum dielectric without re-emitting.  It also takes
    ``--out`` so a renamed output file (rare but possible) still
    works.

    Returns
    -------
    str
        The complete script text.  The wrapper writes this to
        ``makov_payne_correction.py`` next to the ``.fdf``.
    """
    # The script is deliberately self-contained — it doesn't import
    # anything from the molbuilder package — so a user who copies
    # the SIESTA bundle to a cluster without molbuilder installed
    # can still run it.  All physical constants are inlined.
    return f'''#!/usr/bin/env python3
"""Makov-Payne post-process correction for SIESTA total energy.

Run AFTER SIESTA has finished:

    python3 makov_payne_correction.py
    python3 makov_payne_correction.py --epsilon 4.0
    python3 makov_payne_correction.py --out my-job.out

Reads the converged total energy from the SIESTA .out file and the
lattice vectors from the same file (SIESTA echoes the final
LatticeVectors block), computes the leading Makov-Payne
finite-size correction

    DeltaE_MP = alpha * q^2 / (2 * epsilon_r * L)

with L = V^(1/3), and prints the corrected total energy in eV.

================================================================
APPLICABILITY: vacuum supercell of an isolated charged molecule
================================================================

This correction is the right physics for a *charged molecule in
a finite vacuum supercell* — the standard molbuilder build path
(auto-pad / cell_padding).  Two NON-vacuum cases the script will
silently compute the WRONG number for:

  1. A charged periodic crystal (the unit cell carries net
     charge and the lattice is the physical periodic
     repetition).  Here the image-image interaction IS the
     physics, not an artefact.  Don't apply this correction.

  2. A charged slab / surface (2-D periodicity + vacuum on the
     non-periodic axis).  The correct finite-size correction is
     a 2-D dipole/quadrupole correction
     (e.g. Bengtsson PRB 59, 12301 (1999); SIESTA has built-in
     SlabDipoleCorrection support for this).  Makov-Payne is the
     3-D-periodic formula; applying it to a 2-D slab over-counts.

If your cell is anything other than a vacuum box containing an
isolated charged molecule, IGNORE the corrected value below.
molbuilder emits this script unconditionally on NetCharge != 0;
the script does not know whether the cell SIESTA used was a
vacuum box, a crystal, or a slab.

Emitted by molbuilder because the input deck has NetCharge = {q:+d}.
See Makov & Payne, Phys. Rev. B 51, 4014 (1995).
"""

import argparse
import math
import re
import sys
from pathlib import Path

SYSTEM_LABEL = "{system_label}"
NET_CHARGE   = {q}
DEFAULT_EPS  = {epsilon_r!r}
MADELUNG     = {MADELUNG_CUBIC!r}   # cubic; override if non-cubic
HARTREE_EV   = {HARTREE_EV!r}
BOHR_ANGSTROM= {BOHR_ANGSTROM!r}


def parse_total_energy(text):
    """Extract the converged total energy in eV from a SIESTA .out.

    SIESTA prints the converged total energy as a line like

        siesta: E_KS(eV) =          -12345.67

    near the end of the .out.  We take the LAST occurrence so a
    multi-stage run (geometry opt then single-point) reports the
    final value.
    """
    pat = re.compile(r"siesta:\\s*E_KS\\(eV\\)\\s*=\\s*(-?\\d+\\.?\\d*)")
    m = pat.findall(text)
    if not m:
        return None
    return float(m[-1])


def parse_lattice_vectors_angstrom(text):
    """Extract the final lattice vectors (Å) from a SIESTA .out.

    SIESTA echoes the lattice as

        outcell: Unit cell vectors (Ang):
                a1x  a1y  a1z
                a2x  a2y  a2z
                a3x  a3y  a3z

    Tolerates the case variants real SIESTA builds produce
    (``outcell`` / ``OUTCELL``) and skips blank/comment lines
    between the marker and the lattice rows — molbuilder's
    first-class SIESTA parser already handles both
    (``parsers/siesta.py``); the post-process script mirrors that
    leniency so it doesn't fail on a perfectly-good .out from a
    build that prints a blank line after the marker.
    """
    # Case-insensitive marker search via re.IGNORECASE on a plain
    # substring — cheaper than lowercasing the whole text.
    pat = re.compile(
        r"outcell:\\s*Unit cell vectors \\(Ang\\):", re.IGNORECASE)
    matches = list(pat.finditer(text))
    if not matches:
        return None
    idx_end = matches[-1].end()
    # Walk forward, skipping blanks / comments / continuation
    # whitespace until we accumulate 3 rows of 3+ floats each.
    rows = []
    for line in text[idx_end:].splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            # If we already have rows but hit a non-data line, the
            # block is malformed.  If we have no rows yet, keep
            # walking — some printers add a header row before the
            # data.
            if rows:
                return None
            continue
        try:
            row = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            if rows:
                return None
            continue
        rows.append(row)
        if len(rows) == 3:
            break
    return rows if len(rows) == 3 else None


def lattice_volume_angstrom3(cell):
    a, b, c = cell
    bx = b[1] * c[2] - b[2] * c[1]
    by = b[2] * c[0] - b[0] * c[2]
    bz = b[0] * c[1] - b[1] * c[0]
    return abs(a[0] * bx + a[1] * by + a[2] * bz)


def compute_correction(q, L_angstrom, epsilon_r, madelung):
    L_bohr = L_angstrom / BOHR_ANGSTROM
    dE_hartree = madelung * (q * q) / (2.0 * epsilon_r * L_bohr)
    return dE_hartree * HARTREE_EV


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",
        default=f"{{SYSTEM_LABEL}}.out",
        help="SIESTA .out file to read (default: <system_label>.out)")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPS,
        help=f"Relative permittivity (default: {{DEFAULT_EPS}}, vacuum)")
    p.add_argument("--madelung", type=float, default=MADELUNG,
        help=f"Madelung constant (default: {{MADELUNG}}, cubic)")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_file():
        print(f"error: {{out_path}} not found.  Run SIESTA first, or pass "
              "--out to point at the correct file.", file=sys.stderr)
        sys.exit(1)
    text = out_path.read_text(errors="replace")

    E_raw = parse_total_energy(text)
    if E_raw is None:
        print(f"error: could not find a converged 'siesta: E_KS(eV)' "
              f"line in {{out_path}} (incomplete or failed run?)",
              file=sys.stderr)
        sys.exit(1)

    cell = parse_lattice_vectors_angstrom(text)
    if cell is None:
        print(f"error: could not parse 'outcell: Unit cell vectors' "
              f"from {{out_path}}", file=sys.stderr)
        sys.exit(1)
    V = lattice_volume_angstrom3(cell)
    L = V ** (1.0 / 3.0)

    dE = compute_correction(NET_CHARGE, L, args.epsilon, args.madelung)
    E_corr = E_raw - dE

    # Cell-shape sanity: warn if the cell is markedly non-cubic
    # (the cubic Madelung loses validity).  Heuristic: max/min of
    # the three lattice norms beyond 1.5 is "non-cubic".
    norms = [math.sqrt(sum(x * x for x in row)) for row in cell]
    if max(norms) / max(min(norms), 1e-9) > 1.5:
        print("warning: cell is markedly non-cubic; the cubic Madelung "
              "constant under-estimates the correction.  Re-run with "
              "--madelung for the appropriate shape constant.",
              file=sys.stderr)

    print(f"# Makov-Payne post-process correction")
    print(f"#   SystemLabel       = {{SYSTEM_LABEL}}")
    print(f"#   NetCharge q       = {{NET_CHARGE:+d}}")
    print(f"#   epsilon_r         = {{args.epsilon}}")
    print(f"#   Madelung alpha    = {{args.madelung}}")
    print(f"#   Lattice volume V  = {{V:.4f}} A^3")
    print(f"#   Effective L=V^1/3 = {{L:.4f}} A")
    print(f"#")
    print(f"E_total (raw, eV)         = {{E_raw:.6f}}")
    print(f"DeltaE_MP (correction, eV) = {{dE:.6f}}")
    print(f"E_total (corrected, eV)    = {{E_corr:.6f}}")


if __name__ == "__main__":
    main()
'''


def emit_correction_script(
    fdf_path: Path,
    system_label: str,
    q: int,
    epsilon_r: float = 1.0,
) -> Optional[Path]:
    """Write ``makov_payne_correction.py`` next to the FDF.

    Returns the path to the emitted script, or ``None`` if the
    FDF parent directory doesn't exist (rare; caller should
    write the FDF first).
    """
    parent = Path(fdf_path).parent
    if not parent.is_dir():
        return None
    out_path = parent / "makov_payne_correction.py"
    out_path.write_text(render_correction_script(
        system_label=system_label, q=q, epsilon_r=epsilon_r))
    return out_path
