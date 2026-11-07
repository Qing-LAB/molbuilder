"""A known Au-BDT-Au junction, built in source.

WHY THIS EXISTS.  A checked-in artefact or a real directory under
``projects/`` is not ground truth: it is an unversioned assumption.  It was written by whatever
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

Used by ``tests/test_script_emit.py``.

Below it, the layers' positions of a chain sandwich
(``tests/test_transport_compose.py``), and the junctions a real relaxation of
`support.real_runs` starts from.
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


def au_bdt_au() -> dict:
    """The committed Au-BDT-Au pair (`tests/data/au_bdt_au.README.md`) as a
    calculation relaxes it: its labels, its leads held, as a lead must be
    (`engines/transport.md` § 4), in the cell a transport calculation runs it
    in, which the junction states and the calculation never derives (§ 2a.9):
    the gold chain a wire, isolated across in an 8 A box; along transport the
    atoms' length and one Au-Au spacing of room, so each lead continues into
    the image."""
    from pathlib import Path

    from molbuilder.workingcopy_structure import StructureCodec
    s = StructureCodec().load(Path(__file__).parent.parent / "data"
                              / "au_bdt_au.xyz")
    z = s.positions[:, 2]
    return {"elements": list(s.elements),
            "positions": s.positions.tolist(),
            "regions": {**s.regions,
                        "frozen_atoms": (s.regions["L-electrode"]
                                         + s.regions["R-electrode"])},
            "cell": [8.0, 8.0, float(z.max() - z.min()) + abs(z[1] - z[0])],
            "axis_kind": ["isolated", "isolated", "transport"]}


def uneven_chain() -> dict:
    """Hydrogen on a square lattice, stacked along transport at the spacing
    it has across, so its leads continue through the side boundaries as they
    must (`cell.transverse_vacuum`), a buffer atom past its left lead, its
    leads of eight and nine atoms and held: what a symmetric junction cannot
    tell, in the seconds hydrogen takes -- a count hardcoded for both leads, a
    buffer's anchors, the k-points across."""
    n = 22
    regions = {"buffer": [0], "L-electrode": list(range(1, 9)),
               "bridge": list(range(9, 13)), "R-electrode": list(range(13, n))}
    regions["frozen_atoms"] = regions["L-electrode"] + regions["R-electrode"]
    return {"elements": ["H"] * n,
            "positions": [[1.0, 1.0, 2.0 * i] for i in range(n)],
            "regions": regions, "cell": [2.0, 2.0, 2.0 * n],
            "axis_kind": ["periodic", "periodic", "transport"]}


# ---- a chain sandwich's layers --------------------------------------- #

#: Six 2.5 Å gold layers a side (12.5 Å span -- over the wizard's 12 Å lead
#: floor), a four-atom bridge between.
_LAYERS_L = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
_BRIDGE = [("S", 15.0), ("C", 16.4), ("C", 17.8), ("S", 19.2)]
_LAYERS_R = [22.0, 24.5, 27.0, 29.5, 32.0, 34.5]
