"""A gold-benzenedithiol-gold junction, written and read and written again.

WHY THIS FILE EXISTS.  On 2026-07-31 the reserved ``frozen_atoms`` list stopped
being a field of its own and became an ordinary label in ``regions`` -- one
store, one designated accessor, interpreted only where it means something.  The
design is right.  What went wrong is that the change was verified entirely
against data the NEW code had produced: round-tripping new->new proves the code
is self-consistent, and cannot notice that it stopped understanding a file
already on disk.  A real project sidecar came back with its fifty frozen
electrode atoms silently gone, and the generated SIESTA input lost its
``Geometry.Constraints`` block -- so a junction whose electrodes were supposed to
be pinned would have relaxed, converged, and reported a structure nobody asked
for.

So this file does not pin a checked-in artefact.  It BUILDS the junction here,
in source, where the reader can see every atom and every label, and then puts it
through the whole API:

    build -> write -> read -> write

and checks that the second write is byte-identical to the first and that every
label survives (the reserved one among them).  A fixture nobody can read is a fixture nobody
notices has gone stale; a structure built in twelve lines is one anybody can
check by eye.

Contract: model/structure.md § 2.4 (the paired-file door), § 2.2 (raw vs
resolved cell), model/structure-molstruct.md (the sidecar), web/molview.md § 6.6
(one label store, reserved names interpreted at the end).
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import StructureCodec


# --------------------------------------------------------------------- #
#  The junction, built where it can be read                             #
# --------------------------------------------------------------------- #

#: Two gold electrodes along z with a benzenedithiol bridge between them.
#: Small enough to read, shaped like the real thing: the OUTER gold layers are
#: what a transport calculation holds still, and the bridge is what relaxes.
_ELECTRODE_LAYERS = 3       # per side
_ATOMS_PER_LAYER = 4


def _junction() -> Structure:
    """Au(3 layers) - S-C6H4-S - Au(3 layers), stacked along z."""
    elements: list[str] = []
    positions: list[list[float]] = []

    def add(symbol, x, y, z):
        elements.append(symbol)
        positions.append([float(x), float(y), float(z)])

    # ---- left electrode: layers at z = 0, 2.4, 4.8 -------------------- #
    for layer in range(_ELECTRODE_LAYERS):
        for k in range(_ATOMS_PER_LAYER):
            add("Au", 2.9 * (k % 2), 2.9 * (k // 2), 2.4 * layer)

    # ---- the molecule: S, six ring carbons, four H, S ----------------- #
    add("S", 1.45, 1.45, 7.0)
    for k in range(6):                      # a flat hexagon, 1.4 Å bonds
        angle = 2.0 * np.pi * k / 6.0
        add("C", 1.45 + 1.4 * np.cos(angle), 1.45 + 1.4 * np.sin(angle), 9.0)
    for k in range(4):                      # hydrogens on four of the six
        angle = 2.0 * np.pi * k / 6.0
        add("H", 1.45 + 2.5 * np.cos(angle), 1.45 + 2.5 * np.sin(angle), 9.0)
    add("S", 1.45, 1.45, 11.0)

    # ---- right electrode: layers at z = 13.4, 15.8, 18.2 -------------- #
    for layer in range(_ELECTRODE_LAYERS):
        for k in range(_ATOMS_PER_LAYER):
            add("Au", 2.9 * (k % 2), 2.9 * (k // 2), 13.4 + 2.4 * layer)

    struct = Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        title="Au-BDT-Au junction",
        cell=[[5.8, 0.0, 0.0], [0.0, 5.8, 0.0], [0.0, 0.0, 20.6]],
        axis_kind=("periodic", "periodic", "transport"),
    )

    left = list(range(0, _ELECTRODE_LAYERS * _ATOMS_PER_LAYER))
    molecule = list(range(len(left), len(left) + 12))
    right = list(range(molecule[-1] + 1, len(elements)))
    # THE OUTERMOST LAYER OF EACH ELECTRODE IS HELD STILL -- the bulk contact
    # the junction is bolted to.  This is the fact that has to survive.
    frozen = left[:_ATOMS_PER_LAYER] + right[-_ATOMS_PER_LAYER:]

    struct.regions = {
        "L-electrode":  left,
        "bridge":       molecule,
        "R-electrode":  right,
        "frozen_atoms": frozen,      # an ORDINARY label in the one store
    }
    return struct


@pytest.fixture
def junction() -> Structure:
    return _junction()


# --------------------------------------------------------------------- #
#  build -> write -> read -> write                                      #
# --------------------------------------------------------------------- #

def test_the_second_write_is_byte_identical_to_the_first(tmp_path, junction):
    """The round trip that would have caught the frozen-atom loss.

    Write it, read it back, write THAT, and compare the bytes. Anything the
    reader fails to recover shows up as a difference in the second pair --
    which is exactly what a dropped ``frozen_atoms`` key is, and exactly what
    reading only the writer's own fresh output can never reveal.
    """
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"

    StructureCodec().write(junction, first)
    back = StructureCodec().load(first)
    StructureCodec().write(back, second)

    assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8"), (
        "the geometry changed on the way through the reader")

    import json
    first_side = json.loads(
        molstruct.sidecar_path_for(first).read_text(encoding="utf-8"))
    second_side = json.loads(
        molstruct.sidecar_path_for(second).read_text(encoding="utf-8"))

    # ``created_at`` is PROVENANCE, not content: it records when each file was
    # written, so two writes of one structure differ there by construction and
    # comparing it would make this test fail whenever the pair straddles a
    # second boundary. Everything else must match exactly -- and the assertion
    # below states that the stamp is the ONLY difference, so a field quietly
    # joining the provenance side cannot hide here.
    assert set(first_side) == set(second_side)
    differing = sorted(k for k in first_side
                       if first_side[k] != second_side.get(k))
    assert differing in ([], ["created_at"]), (
        f"the sidecar changed on the way through the reader, in {differing}:\n"
        f"  wrote {first_side}\n  read  {second_side}")


def test_every_label_survives_the_round_trip_including_the_reserved_one(
        tmp_path, junction):
    """One store: the reserved name travels with the others, and is asked for
    through the one accessor rather than looked up by name at each call site."""
    target = tmp_path / "junction.xyz"
    StructureCodec().write(junction, target)
    back = StructureCodec().load(target)

    assert back.regions == junction.regions, (
        f"labels did not survive:\n  wrote {junction.regions}\n  read  {back.regions}")
    # THE DESIGNATED ACCESSOR, on both the Structure and the sidecar payload.
    assert list(back.frozen_atoms) == sorted(junction.regions["frozen_atoms"])
    payload = molstruct.load(molstruct.sidecar_path_for(target))
    assert molstruct.frozen_atoms(payload) == sorted(
        junction.regions["frozen_atoms"])
    assert "frozen_atoms" not in payload, (
        "a second top-level store came back -- v7 keeps it in `regions`")


def test_the_cell_and_its_axis_kinds_survive(tmp_path, junction):
    """A transport axis is not a periodic one, and the difference decides what
    the engine writes. It has to come back as it went in."""
    target = tmp_path / "junction.xyz"
    StructureCodec().write(junction, target)
    back = StructureCodec().load(target)

    assert back.axis_kind == ("periodic", "periodic", "transport")
    assert np.allclose(np.asarray(back.cell), np.asarray(junction.cell))
