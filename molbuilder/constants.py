"""Physical constants — one home, so a number cannot mean two things.

MODULE  constants (floor 0; imports nothing at all)
ROLE    the ONE place a physical constant is spelled
USED-BY the parsers, the emitters, the transport composer — anywhere a
        conversion between atomic units and the units a file speaks happens

WHY THIS EXISTS (2026-08-30).  The Bohr radius was written out **eight
times in three different values**::

    0.529177210903   makov_payne · pyscf/vibration_emitters · siesta_mdnc
    0.5291772108     parse/coords/siesta_xv · parse/dirs · parse/engines/pyscf
    0.529177         parse/fdf · parse/ion

The consequence was not theoretical.  Two modules read the same SIESTA
``.XV`` file — ``parse.coords.siesta_xv`` and ``transport.compose`` (through
what is now ``parse.fdf``) — using the first and third of those, so **the same
file gave coordinates 4e-7 apart depending on which reader was asked**.  That
surfaced as a test comparing the two answers and failing by 1.6e-6 Å on a gold
lattice constant.

The size of the discrepancy is not the point.  A physical constant is a fact
about the universe, not about a module, and eight copies of it are eight things
that can be edited apart.

VALUES are CODATA 2018 throughout, which is what the majority of the call sites
already used and the most recent set the project cites.
"""

from __future__ import annotations

#: 1 Bohr in Ångström.  CODATA 2018.
BOHR_ANGSTROM: float = 0.529177210903

#: 1 Hartree in electronvolt.  CODATA 2018.
HARTREE_EV: float = 27.211386245988

#: 1 Rydberg in electronvolt — half a Hartree, DERIVED rather than typed, so
#: the two cannot drift apart by a digit.  SIESTA speaks Rydberg.
RYDBERG_EV: float = HARTREE_EV / 2.0

#: Boltzmann's constant in eV per kelvin.  CODATA 2018.  SIESTA's
#: ``ElectronicTemperature`` may be written as a temperature OR as an energy,
#: so a reader of it needs this to answer in one unit.
BOLTZMANN_EV_K: float = 8.617333262e-5

#: The same constant in Hartree per kelvin — DERIVED, so the thermochemistry
#: deck's spelling and the validation gate's cannot drift apart.
BOLTZMANN_HARTREE_K: float = BOLTZMANN_EV_K / HARTREE_EV

#: 1 Hartree in wavenumbers (cm⁻¹).  CODATA 2018.  The vibrational decks
#: speak wavenumbers; the engines compute in Hartree.
HARTREE_CM1: float = 219474.6313632

#: 1 unified atomic mass unit in electron masses.  CODATA 2018.
AMU_ELECTRON_MASS: float = 1822.888486209

#: cm⁻¹ per sqrt(Hartree / (Bohr²·amu)) -- the frequency conversion for a
#: Hessian mass-weighted in AMU rather than in electron masses.  ≈ 5140.48.
#:
#: **DERIVED, not typed**, because it is exactly the two constants above and a
#: third spelling of the same physics is how they drift apart.  A Hessian in
#: Hartree/Bohr² weighted by amu gives eigenvalues in Hartree/(Bohr²·amu); the
#: electron-mass convention would give true atomic units and use `HARTREE_CM1`
#: directly.  Both are correct; mixing them silently is not, which is what the
#: vibration deck did until 2026-09-21 -- PySCF's `harmonic_analysis` weights
#: in amu and the deck's frozen-atom path weighted in electron masses, so the
#: two paths disagreed about mode normalisation by sqrt(AMU_ELECTRON_MASS) and
#: every frozen IR intensity came out ~1823x too small.
#:
#: (No `math.sqrt`: this module imports nothing at all -- see the header.)
CM1_PER_SQRT_HARTREE_BOHR2_AMU: float = HARTREE_CM1 / (AMU_ELECTRON_MASS ** 0.5)

#: 1 atomic unit of electric dipole (e·a₀) in Debye.  CODATA 2018.
AU_DIPOLE_DEBYE: float = 2.541746473

#: 1 e·Å in Debye — DERIVED from the atomic-unit value above, so the two
#: dipole spellings cannot drift.
DEBYE_E_ANGSTROM: float = AU_DIPOLE_DEBYE / BOHR_ANGSTROM

#: 1 Hartree/Bohr in eV/Å, **the ASE/NIST value**.  The name carries the
#: convention because there are two: this one, and ``HARTREE_EV /
#: BOHR_ANGSTROM`` = 51.422067476, which differs by 0.36 ppm.  Force numbers
#: and the threshold lines drawn over them must use the SAME one or the line
#: sits in the wrong place, so whichever a call site needs, it names.
HARTREE_BOHR_EV_ANGSTROM_ASE: float = 51.42208619

#: Avogadro's number, per mole.  CODATA 2018 (exact by SI definition).
AVOGADRO: float = 6.02214076e23

#: 1 joule in electronvolt — the reciprocal of the elementary charge in
#: coulombs, which SI fixes exactly.
JOULE_EV: float = 1.0 / 1.602176634e-19

#: 1 kcal/mol in electronvolt — DERIVED, since a calorie is defined as
#: exactly 4.184 J.
KCAL_MOL_EV: float = 4184.0 * JOULE_EV / AVOGADRO

#: 1 hertz in electronvolt (Planck's constant in eV·s).  CODATA 2018.
HZ_EV: float = 4.135667696e-15

#: 1 wavenumber (cm⁻¹) in electronvolt — DERIVED from `HARTREE_CM1`, so
#: the two spellings of the same physics cannot drift.
CM1_EV: float = HARTREE_EV / HARTREE_CM1
