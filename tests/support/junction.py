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

Used by ``tests/test_fdf_generator_roundtrip.py``, ``tests/test_script_emit.py``
and ``tests/test_no_tests_read_the_projects_tree.py`` (the structure), and its
spectrum sidecar by ``tests/parse/test_sidecars.py`` and
``tests/parse/test_round2_fixes.py`` (written by our writer, read back by our
reader).

*(It also named ``tests/parse/dirs/test_bundle.py`` until 2026-09-06.  That
file does not exist and neither does a bundle DirParser -- the pointer
outlived both, which is how ``parse/dirs/_assembler_helpers`` came to look
like it served something.)*
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


# --------------------------------------------------------------------- #
#  A spectra sidecar, WRITTEN by the real writer                        #
# --------------------------------------------------------------------- #
#
# Built here rather than read from projects/BDT/spectrum/BDT-only/, which is
# the user's scientific record: a test that consumes it asserts facts about
# data whose relevance nobody confirmed, skips silently on any machine that
# lacks it (so the suite reads green while proving nothing), and changes
# meaning the day that run is regenerated.
#
# The writer is the application's own `dump_spectra_json`, so the document is
# valid by construction -- if the format moves, this moves with it.

def spectra_sidecar(path):
    """Write a minimal, VALID ``*.spectra.json`` at ``path``; return ``path``."""
    from molbuilder.spectra import SpectraResults
    from molbuilder.sidecars.spectra import dump_spectra_json

    s = build_junction()
    from molbuilder.spectra.results import SCHEMA_VERSION as _V
    results = SpectraResults(
        schema_version=_V,
        timestamp="2026-01-01T00:00:00Z",
        structure_hash="a" * 64,
        equilibrium_mo_energies_eh=[-1.0, -0.5],
        equilibrium_homo_idx=0,
        selected_mode_idxs_1based=[],
        methods_text="constructed fixture",
        bibliography_keys=[],
        engine="pyscf",
        engine_version="test",
        molbuilder_version="test",
        n_atoms_total=s.n_atoms,
        free_atom_idxs=sorted(set(range(s.n_atoms)) - set(frozen())),
        frozen_atom_idxs=frozen(),
        equilibrium_scf_eh=-1.0,
        equilibrium_elements=list(s.elements),
        equilibrium_positions_ang=[[float(x) for x in row] for row in s.positions],
        modes=[],
        config={"method": "DFT", "spin_treatment": "unrestricted"},
    )
    dump_spectra_json(results, path)
    return path


# `run_dir` -- a deck rendered into a bare folder -- stood here until
# 2026-10-04, for the one test that read a run's labels by searching
# that folder; a run's labels are its own deck's, through the run door
# (`runs.declared`, plan B12).  `xv_file`, a SIESTA `.XV` written by a
# hand-rolled writer, went the same day: a `.XV` is SIESTA's, and a test
# reads one a real run wrote (`tests/fixtures/siesta_flat_h2/H2.XV`).
