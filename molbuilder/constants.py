"""Physical constants — one home, so a number cannot mean two things.

MODULE  constants (floor 0; imports nothing at all)
ROLE    the ONE place a physical constant is spelled
USED-BY the parsers, the emitters, the transport composer — anywhere a
        conversion between atomic units and the units a file speaks happens

WHY THIS EXISTS (2026-08-30).  The Bohr radius was written out **eight
times in three different values**::

    0.529177210903   makov_payne · pyscf/vibration_emitters · siesta_mdnc
    0.5291772108     parse/coords/siesta_xv · parse/dirs · parse/engines/pyscf
    0.529177         transport/preflight · parse/ion

The consequence was not theoretical.  Two modules read the same SIESTA
``.XV`` file — ``parse.coords.siesta_xv`` and ``transport.compose`` (through
``transport.preflight``) — using the first and third of those, so **the same
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
