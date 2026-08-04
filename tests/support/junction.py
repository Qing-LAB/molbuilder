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
        config={"method": "UKS"},
    )
    dump_spectra_json(results, path)
    return path


# --------------------------------------------------------------------- #
#  A SIESTA run directory, and a .XV, both BUILT                        #
# --------------------------------------------------------------------- #

def run_dir(tmp_path, label="junction"):
    """Render a real SIESTA run directory for the junction; return its path.

    Uses the application's OWN emitter, so the directory is whatever the app
    produces today -- which is the point.  A run directory copied from
    projects/ asserts against numbers captured on the day it was written, and
    goes stale silently when the format moves.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_fdf

    run = tmp_path / f"{label}-run"
    run.mkdir(parents=True, exist_ok=True)
    # ``diag_algorithm`` is set explicitly: the parse tests assert that curated
    # body keys survive as RAW STRINGS, and a key left at its default gives
    # them nothing to check.  A fixture has to exercise what is asserted.
    (run / f"{label}.fdf").write_text(
        render_fdf(build_junction(),
                   SiestaConfig(system_label=label,
                                diag_algorithm="ELPA-2STAGE")),
        encoding="utf-8")
    return run


#: SIESTA writes .XV in BOHR; the parser converts back on read.
_ANGSTROM_PER_BOHR = 0.5291772108


def xv_file(path, struct=None):
    """Write a valid SIESTA ``.XV`` for ``struct`` (default: the junction).

    The format, straight from what the parser reads: three lattice rows (vector
    then its velocity), the atom count, then one row per atom of
    ``species_index Z x y z vx vy vz`` -- all lengths in Bohr.

    Hand-written because the application has no .XV WRITER (it only ever reads
    them; SIESTA is the writer in production).  So this fixture is verified the
    only honest way -- by parsing it back and comparing to the structure it was
    built from; see the round-trip test beside its consumers.
    """
    # The one atomic-number table in the tree (pyscf/input.py owns it).
    from molbuilder.pyscf.input import _ATOMIC_NUMBER

    s = struct if struct is not None else build_junction()
    cell = s.resolve_cell() / _ANGSTROM_PER_BOHR
    pos = s.positions / _ANGSTROM_PER_BOHR

    species = {}
    for el in s.elements:
        species.setdefault(el, len(species) + 1)

    rows = []
    for i in range(3):
        rows.append("  {:16.9f}{:16.9f}{:16.9f}    {:16.9f}{:16.9f}{:16.9f}".format(
            *cell[i], 0.0, 0.0, 0.0))
    rows.append(f"{len(s.elements):8d}")
    for el, xyz in zip(s.elements, pos):
        rows.append("{:3d}{:6d}{:18.9f}{:18.9f}{:18.9f}    "
                    "{:16.9f}{:16.9f}{:16.9f}".format(
                        species[el], _ATOMIC_NUMBER.get(el.capitalize(), 0),
                        *xyz, 0.0, 0.0, 0.0))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def job_run_dir(tmp_path, out_names=("hemeC-stage2-run3-finished-42fr.out",),
                label="junction"):
    """A run directory with INPUT and OUTPUT, both under test control.

    The input is rendered from the junction; the output is one of the FROZEN
    SIESTA logs in ``tests/watch/fixtures/siesta_frozen/`` -- real engine
    output, checked in beside the tests, reviewed when it changes.

    Why not invent the ``.out``: a hand-written one would test my guess at
    SIESTA's format rather than SIESTA's format.  Why not point at
    ``projects/``: that is the user's record, its relevance is unconfirmed, and
    a missing directory turns the test into a silent skip.  A frozen fixture is
    neither.

    ``out_names`` takes more than one for the multi-stage cases (several
    ``.out`` files in one directory, each its own plot bucket).
    """
    import shutil
    from pathlib import Path as _P

    fixtures = _P(__file__).resolve().parents[1] / "watch" / "fixtures" / "siesta_frozen"
    run = run_dir(tmp_path, label=label)
    for name in out_names:
        src = fixtures / name
        if not src.is_file():                     # pragma: no cover
            raise AssertionError(
                f"frozen SIESTA fixture missing: {src}.  These live in the "
                f"test tree on purpose -- do not repoint this at projects/.")
        shutil.copyfile(src, run / name)
    return run
