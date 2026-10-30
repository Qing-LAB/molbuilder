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

    E_\\text{tot,corr} = E_\\text{tot,raw} + \\Delta E_{MP}

i.e. the correction is *added* to the SIESTA total.  Makov & Payne
(Phys. Rev. B 51, 4014 (1995), Eq. 15) give
:math:`E_\\text{supercell} = E_\\text{isolated} - q^2\\alpha/(2\\varepsilon L)`,
so :math:`E_\\text{isolated} = E_\\text{supercell} + q^2\\alpha/(2\\varepsilon L)`.
The charged periodic cell carries a compensating uniform background;
the point-charge-in-jellium (Wigner-lattice) Madelung self-energy is
NEGATIVE (stabilizing), so the raw periodic energy is spuriously *too
low* and the correction must *raise* it.

This module gives molbuilder two related artefacts when the user
emits a SIESTA input deck with ``NetCharge != 0``:

1. :func:`compute_correction` — the bare formula, used by
   :mod:`molbuilder.validation` so the warn message can quote a
   numeric estimate alongside the qualitative caveat.
2. :func:`render_correction_script` — a small self-contained Python
   script `prep` writes next to the FDF.  Once SIESTA has run
   the script reads the ``.out``, extracts the converged total
   energy, computes :math:`\\Delta E_{MP}` for the *actual* cell
   that SIESTA used (read back from the ``.out`` so the correction
   is robust to any manual cell edit -- the emitter warns on thin
   vacuum but never auto-pads, so the emitted LatticeVectors ==
   structure bbox + 2*vacuum), and prints the corrected value.

Why a post-process script rather than an in-FDF tweak?  SIESTA has
no Makov-Payne keyword: it applies the monopole term ITSELF, but only
when it sees a molecule in a simple, face- or body-centred cubic cell
(``madelung.f``; printed as ``siesta: Emadel`` and already summed into
E_KS), and otherwise prints "Energy correction terms can not be
applied" and adds nothing.  So the script reads ``Emadel`` first and
adds only what SIESTA did not.  The
correction is a single-number energy shift with no effect on geometry
or density, so a separate script keeps the SIESTA input standard while
giving the user a one-command way to the corrected number.  It is
written only for a FINITE charged system: the point-charge formula is
the wrong one for a charged slab or crystal
(``science/chemistry-correctness.md`` § 2b).
"""

from __future__ import annotations

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
from molbuilder.constants import HARTREE_EV          # re-exported below
from molbuilder.constants import BOHR_ANGSTROM       # re-exported below


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
        The correction :math:`\\Delta E_{MP}` in eV (always positive
        for q≠0).  *Add* this to the raw SIESTA total energy to get
        the finite-size-corrected value (the raw charged-cell energy
        is spuriously too low; see the module docstring / Makov-Payne
        Eq. 15).
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
    *,
    stage: str,
) -> str:
    """Generate a self-contained Python post-process script that
    reads the run's output -- its newest try's, else a hand run's
    unnumbered one (``stage`` the run's stage token) -- extracts the
    converged total energy, computes the Makov-Payne correction for the
    actual cell SIESTA used, and prints the corrected value.

    The script takes ``--epsilon`` on the command line so a user
    re-running with implicit solvation can override the default
    vacuum dielectric without re-emitting.  It also takes
    ``--out`` so a renamed output file (rare but possible) still
    works.

    Returns
    -------
    str
        The complete script text.  `prep` writes this to
        ``makov_payne_correction.py`` next to the ``.fdf``.
    """
    # The script runs where molbuilder may not be installed -- beside a
    # finished run -- so it imports nothing from the package.  It FINDS the
    # output by the one rule every reader of a run asks
    # (`runrecord.newest_run`, the catalogue's names) and READS it with
    # SIESTA's one reading pass, both from the monitor's bundle that lies
    # beside it in every run (`runfiles.MONITOR_BUNDLE`), so the energy, the
    # cell and SIESTA's own term are read as every reader of a run reads
    # them; the physical constants are inlined.
    #
    # SO THE FORMULA IS WRITTEN TWICE, ON PURPOSE.  `compute_correction`
    # above is the one molbuilder calls; the copy below is the one that runs
    # beside a finished job.  They were MEASURED to agree (2026-09-09), to
    # the six decimals the script prints.
    #
    # WHAT IS NOT DUPLICATED: the constants.  `MADELUNG_CUBIC`, `HARTREE_EV` and
    # `BOHR_ANGSTROM` are interpolated in from their one home, so the copy
    # cannot drift on a VALUE -- only on the four lines of algebra, which have
    # no branches and which a reviewer sees whole.
    #
    # IF IT EVER GROWS A BRANCH -- a non-cubic Madelung term, a second-order
    # correction -- stop copying and make it a callable that travels as its
    # module's own file, the way the PySCF script imports its rules from
    # `mb_pyscf.pyz` (`runwrap._zip_bundle`, `engines/pyscf.md` § 3).  That is
    # the moment, not before: a bundle is more moving parts than four lines
    # of arithmetic.
    from ..runfiles import MONITOR_BUNDLE as monitor_bundle
    return f'''#!/usr/bin/env python3
"""Makov-Payne post-process correction for SIESTA total energy.

Run AFTER SIESTA has finished:

    python3 makov_payne_correction.py
    python3 makov_payne_correction.py --epsilon 4.0
    python3 makov_payne_correction.py --out my-job.out

Reads the converged total energy from the SIESTA .out file, then
SIESTA's own monopole correction, `siesta: Emadel`.  SIESTA applies it
itself for a molecule in a simple, face- or body-centred cubic cell, and
it is then already in E_KS: this script adds NOTHING and says so.
Otherwise it reads the final lattice vectors, computes the leading
Makov-Payne finite-size correction

    DeltaE_MP = alpha * q^2 / (2 * epsilon_r * L)

with L = V^(1/3), and prints the corrected total energy in eV.

================================================================
APPLICABILITY: vacuum supercell of an isolated charged molecule
================================================================

This correction is the right physics for a *charged molecule in
a finite vacuum supercell* — the standard molbuilder build path
(box = structure bounding-box + 2*vacuum).  Two NON-vacuum cases
the script will silently compute the WRONG number for:

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

molbuilder writes this script only for a finite (isolated) charged
system.  If the cell was edited into anything else by hand, IGNORE the
corrected value below: the script does not know what the cell holds.

Emitted by molbuilder because the input deck has NetCharge = {q:+d}.
See Makov & Payne, Phys. Rev. B 51, 4014 (1995).
"""

import argparse
import math
import sys
from pathlib import Path

SYSTEM_LABEL = "{system_label}"
# THE RUN'S STAGE: with the label, the names the file catalogue gives this
# run's files (`runfiles.RunNames`).
STAGE        = "{stage}"
NET_CHARGE   = {q}
DEFAULT_EPS  = {epsilon_r!r}
MADELUNG     = {MADELUNG_CUBIC!r}   # cubic; override if non-cubic
HARTREE_EV   = {HARTREE_EV!r}
BOHR_ANGSTROM= {BOHR_ANGSTROM!r}


def _bundle():
    """The monitor's bundle beside this script ({monitor_bundle}, put in
    every run by its launch) on the path -- the run's names, its newest try
    and SIESTA's one reading pass are read from it."""
    bundle = Path(__file__).resolve().with_name("{monitor_bundle}")
    if not bundle.is_file():
        print(f"error: {{bundle.name}} is not beside this script -- run it "
              f"in the run's folder, where the launch put both.",
              file=sys.stderr)
        sys.exit(1)
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))


def read_siesta_output(path):
    """``(E_KS in eV, the final cell's rows in Angstrom, SIESTA's own
    Emadel in eV or None)`` -- the output read by SIESTA's one reading pass:
    the last step's energy, the last cell the output printed, and SIESTA's
    monopole term."""
    _bundle()
    from siesta_reader import read_output
    reading = read_output(path).finish()
    steps = [st for st in reading.get("steps") or []
             if st.get("energy") is not None]
    return ((steps[-1]["energy"] if steps else None),
            reading.get("lattice"),
            (reading.get("runtime_info") or {{}}).get("emadel_ev"))


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
    p.add_argument("--out", default=None,
        help="SIESTA .out file to read (default: this run's newest try, "
             "else a hand run's)")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPS,
        help=f"Relative permittivity (default: {{DEFAULT_EPS}}, vacuum)")
    p.add_argument("--madelung", type=float, default=MADELUNG,
        help=f"Madelung constant (default: {{MADELUNG}}, cubic)")
    args = p.parse_args()

    if args.out is not None:
        out_path = Path(args.out)
    else:
        # THIS RUN'S NEWEST TRY, by the one rule every reader of a run asks
        # (`runrecord.newest_run`) -- never another stage's output, never by
        # a file's time; with no try, a hand run's unnumbered output.
        _bundle()
        from runfiles import RunNames
        from runrecord import newest_run
        names = RunNames(SYSTEM_LABEL, STAGE)
        out_path = Path(names.name(".out", newest_run(".", names)))
        if not out_path.is_file():
            print(f"error: {{out_path}} not found here.  Run SIESTA first, "
                  "or pass --out to point at the correct file.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"note: reading {{out_path}}.")
    if not out_path.is_file():
        print(f"error: {{out_path}} not found.", file=sys.stderr)
        sys.exit(1)
    E_raw, cell, emadel = read_siesta_output(out_path)
    if E_raw is None:
        print(f"error: {{out_path}} states no energy of a finished step "
              f"(incomplete or failed run?)", file=sys.stderr)
        sys.exit(1)

    # SIESTA'S OWN TERM FIRST: when it applied the correction, E_KS already
    # carries it, and adding ours would count the image energy twice.
    by_siesta = emadel is not None and abs(emadel) > 1e-6
    if by_siesta:
        dE = 0.0
    else:
        if cell is None:
            print(f"error: could not parse 'outcell: Unit cell vectors' "
                  f"from {{out_path}}", file=sys.stderr)
            sys.exit(1)
        V = lattice_volume_angstrom3(cell)
        L = V ** (1.0 / 3.0)
        dE = compute_correction(NET_CHARGE, L, args.epsilon, args.madelung)
        # Cell-shape sanity: warn if the cell is markedly non-cubic
        # (the cubic Madelung loses validity).  Heuristic: max/min of
        # the three lattice norms beyond 1.5 is "non-cubic".
        norms = [math.sqrt(sum(x * x for x in row)) for row in cell]
        if max(norms) / max(min(norms), 1e-9) > 1.5:
            print("warning: cell is markedly non-cubic; the cubic Madelung "
                  "constant under-estimates the correction.  Re-run with "
                  "--madelung for the appropriate shape constant.",
                  file=sys.stderr)
    # ADD the (positive) correction: the raw charged-periodic energy is
    # spuriously too low (Makov-Payne Eq. 15; compensating-background
    # Madelung self-energy is stabilizing).
    E_corr = E_raw + dE

    print(f"# Makov-Payne post-process correction")
    print(f"#   SystemLabel       = {{SYSTEM_LABEL}}")
    print(f"#   NetCharge q       = {{NET_CHARGE:+d}}")
    if by_siesta:
        print(f"#   SIESTA applied the term itself: Emadel = {{emadel:.6f}} "
              f"eV, already in E_KS -- nothing is added here.")
    else:
        print(f"#   SIESTA did not apply it (Emadel "
              f"{{'absent' if emadel is None else '= 0'}}): added here.")
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
    *,
    names,
    plan=None,
) -> Optional[Path]:
    """Write ``makov_payne_correction.py`` next to the FDF -- or put it in
    ``plan`` (`jobset.planned.Plan`), beside the deck the plan holds.

    Returns the path to the emitted script, or ``None`` if the
    FDF parent directory doesn't exist (rare; caller should
    write the FDF first).
    """
    parent = Path(fdf_path).parent
    if plan is None and not parent.is_dir():
        return None
    # THROUGH THE ONE WRITER (`script-preparation.md` § 3.2, W4).  This is a
    # generated script like any other -- the deck's own text tells a person to
    # run it.
    from .. import script_emit as _sc
    from ..runfiles import MAKOV_PAYNE_SCRIPT
    return _sc.write_script(parent / MAKOV_PAYNE_SCRIPT,
                            render_correction_script(
                                system_label=system_label, q=q,
                                epsilon_r=epsilon_r,
                                stage=names.stage),
                            plan=plan)
