"""A known Au-BDT-Au junction, built in source.

WHY THIS EXISTS.  Tests used to reach for ``tests/data/au_bdt_au.*`` and for real
directories under ``projects/`` as though they were ground truth.  They are not:
a checked-in artefact is an unversioned assumption.  It was written by whatever
the code did on the day it was captured, nothing re-checks it, and when the
format moves it goes quietly stale -- so a test built on one can pass while
describing a file nobody would produce today, or fail for reasons that have
nothing to do with the code under test.

That is not hypothetical here.  The 2026-07-31 label-store change made every
such fixture a pre-migration file, and the tests that read them turned red for a
reason ("your fixture is old") that looked exactly like a reason they must not be
used for ("your reader is broken").  Telling those apart cost a bisect.

So the junction is BUILT, every time, from the constructor the application uses.
It is current by construction, it is readable in source -- twelve lines and you
can count the atoms -- and a test written on it exercises the real
write-then-read path instead of trusting a snapshot of one.

Used by ``tests/test_junction_sidecar_roundtrip.py`` (the sidecar pair) and
``tests/parse/dirs/test_bundle.py`` (the generated-script round trip).
"""
from __future__ import annotations

import numpy as np

from molbuilder.structure import Structure

#: Two gold electrodes along z with a benzenedithiol bridge between them.
#: Small enough to read, shaped like the real thing: the OUTER gold layers are
#: what a transport calculation holds still, and the bridge is what relaxes.
ELECTRODE_LAYERS = 3       # per side
ATOMS_PER_LAYER = 4

#: 24 gold + 12 molecule (2 S, 6 C, 4 H).
N_ATOMS = 2 * ELECTRODE_LAYERS * ATOMS_PER_LAYER + 12


def build_junction() -> Structure:
    """Au(3 layers) - S-C6H4-S - Au(3 layers), stacked along z, labelled."""
    elements: list[str] = []
    positions: list[list[float]] = []

    def add(symbol, x, y, z):
        elements.append(symbol)
        positions.append([float(x), float(y), float(z)])

    # ---- left electrode: layers at z = 0, 2.4, 4.8 -------------------- #
    for layer in range(ELECTRODE_LAYERS):
        for k in range(ATOMS_PER_LAYER):
            add("Au", 2.9 * (k % 2), 2.9 * (k // 2), 2.4 * layer)

    # ---- the molecule: S, six ring carbons, four H, S ----------------- #
    add("S", 1.45, 1.45, 7.0)
    for k in range(6):                      # a flat hexagon, 1.4 A bonds
        angle = 2.0 * np.pi * k / 6.0
        add("C", 1.45 + 1.4 * np.cos(angle), 1.45 + 1.4 * np.sin(angle), 9.0)
    for k in range(4):                      # hydrogens on four of the six
        angle = 2.0 * np.pi * k / 6.0
        add("H", 1.45 + 2.5 * np.cos(angle), 1.45 + 2.5 * np.sin(angle), 9.0)
    add("S", 1.45, 1.45, 11.0)

    # ---- right electrode: layers at z = 13.4, 15.8, 18.2 -------------- #
    for layer in range(ELECTRODE_LAYERS):
        for k in range(ATOMS_PER_LAYER):
            add("Au", 2.9 * (k % 2), 2.9 * (k // 2), 13.4 + 2.4 * layer)

    struct = Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        title="Au-BDT-Au junction",
        cell=[[5.8, 0.0, 0.0], [0.0, 5.8, 0.0], [0.0, 0.0, 20.6]],
        axis_kind=("periodic", "periodic", "transport"),
    )
    struct.regions = regions()
    return struct


def regions() -> dict:
    """The label store the junction carries.

    ``frozen_atoms`` is an ORDINARY label in it -- what makes it reserved is the
    interpretation applied where it means something (SIESTA's
    ``Geometry.Constraints``) and the one accessor that pulls the group out, not
    a home of its own.
    """
    left = list(range(0, ELECTRODE_LAYERS * ATOMS_PER_LAYER))
    molecule = list(range(len(left), len(left) + 12))
    right = list(range(molecule[-1] + 1, N_ATOMS))
    return {
        "L-electrode":  left,
        "bridge":       molecule,
        "R-electrode":  right,
        # THE OUTERMOST LAYER OF EACH ELECTRODE -- the bulk contact the junction
        # is bolted to. This is the fact that has to survive every round trip.
        "frozen_atoms": left[:ATOMS_PER_LAYER] + right[-ATOMS_PER_LAYER:],
    }


def frozen() -> list:
    """The atoms a calculation must hold still, sorted."""
    return sorted(regions()["frozen_atoms"])
