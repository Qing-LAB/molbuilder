"""The vibration calculation's deck on SIESTA — the force-constant run.

Contract: `engines/vibration.md` § 5 (the deck, the sorted copy, the
job's own finish) and `science/normal-modes.md` § 3; the seam this
serves is `script-preparation.md` § 4, and the shape is the one
`transport/deck.py` and `pyscf/vibration_deck.py` share: the KIND is a
render argument, the seam stays ONE per engine, and the kind's own module
owns what is kind-specific.

WHAT THE DECK IS.  SIESTA's part of a vibration is ``MD.TypeOfRun FC``:
nudge each atom of one contiguous range, ``FC.First``..``FC.Last``, by
``FC.Displacement`` along x, y and z both ways, and write the force
constants to ``<SystemLabel>.FC``.  Everything else in the deck is the
optimisation deck's own sections -- basis, functional, SCF, spin,
k-points, output -- resolved from the same catalogue rows; what is left
OUT is the geometry section, because a force-constant run moves nothing.

THE RANGE IS THE FREE ATOMS, AND THEY ARE ONE RUN BECAUSE THE COPY WAS
SORTED.  The person's held set is scattered; SIESTA cannot take a
scattered range.  So `prep` renders this deck from a copy sorted with the
``held-first`` key (`transport/sort.py`, `model/overview.md` § 2.2):
held atoms first, free atoms last, the copy's order record stated in the
deck's ATOM-METADATA block.  This module only CHECKS that shape and names the range; it
does not sort, because a deck writer that reorders atoms is a second place
deciding an atom's number.
"""
from __future__ import annotations

from typing import List, Optional

from .. import script_emit as _sc
from ..atom_permutation import SORT_HELD_FIRST
from ..engine_atom_index import siesta_numbering
from ..structure import Structure
from .vibration_phases import PHASES, phase

#: THE PARTS A VIBRATION DECK IS, one per phase of the run
#: (`vibration_phases.PHASES`, `engines/vibration.md` § 5.11 § 4) -- what
#: `spec_for(part=)` renders; each spelled once, the phases table's.
RELAX_PART, FC_PART, OPTICAL_PART = (p.part for p in PHASES)
PARTS = tuple(p.part for p in PHASES)


def optical_facts(cfg, mesh) -> dict:
    """What the optical call derives, worked out when the SPEC is built
    (`script-preparation.md` § 4.1; `engines/vibration.md` § 5.11 § 9): the
    light's direction as the item states it, and the k-point mesh the
    optical sum runs over -- the item's, or the stage's own k-point grid
    (``mesh``, `kmesh.mesh_for`) when none is set: for a molecule Γ
    alone, which is what SIESTA takes for a cell it judges one.  The
    light's direction is written as SIESTA's ``Optical.Vector`` block, the
    mesh as ``Optical.Mesh``; each says where it came from."""
    direction = [float(x) for x in cfg.optical_direction]
    if not any(abs(x) > 0 for x in direction):
        raise ValueError("optical_direction is the zero vector: the "
                         "light's electric field needs a direction")
    own = getattr(cfg, "optical_mesh", None)
    if own is not None:
        counts, mesh_from = [int(n) for n in own], "optical_mesh"
    elif mesh is not None:
        counts, mesh_from = [int(n) for n in mesh.counts], "the stage's k-point grid (kgrid)"
    else:
        counts, mesh_from = [1, 1, 1], "Γ alone: the stage states no k-point grid"
    return {"direction": direction, "mesh": counts, "mesh_from": mesh_from}


def optical_block(struct: Structure, cfg, derived: dict) -> str:
    """The optical single point -- structural text, like the force-constant
    block: a single point (no atom moves), the optical calculation on, the
    field's polarization and direction, the mesh the sum runs over
    (`engines/vibration.md` § 5.11 § 4, § 9; SIESTA 5.4.2 manual, *Optical
    properties*).  ``Optical.Broaden``, ``Optical.Energy.Maximum`` and
    ``Optical.NumberOfBands`` are items and reach the deck through the
    syntax door in their own section, with their declaration and note."""
    o = derived["optical"]
    v = bool(getattr(cfg, "verbose_comments", True))
    out: List[str] = [""]
    if v:
        out += [
            "# --- The optical single point ---",
            "# One SCF at this call's coordinates, no atom moved (MD.Steps 0),",
            "# then SIESTA's optical calculation: eps2(w) for a field along",
            "# Optical.Vector, every occupied state coupled with every empty one",
            "# (OpticalCalculation; engines/vibration.md 5.11), written to",
            f"# {cfg.system_label}.EPSIMG.  The run's Raman phase runs one such",
            "# call at the equilibrium and at +/-dQ along each selected mode;",
            "# the polarizability and the Raman strength are derived from",
            "# eps2 when the result is presented, at the laser and the width",
            "# a person asks (science/vibrational-optical-response.md 3).",
            f"# The mesh: {o['mesh_from']}.",
        ]
    out += [
        "MD.TypeOfRun      CG",
        "MD.Steps          0",
        "OpticalCalculation .true.",
        "Optical.PolarizationType polarized",
        "%block Optical.Vector",
        "  " + " ".join(f"{x:.6f}" for x in o["direction"]),
        "%endblock Optical.Vector",
        "%block Optical.Mesh",
        "  " + " ".join(str(n) for n in o["mesh"]),
        "%endblock Optical.Mesh",
        "",
    ]
    return "\n".join(out)


def fc_facts(struct: Structure, cfg) -> dict:
    """What the force-constant run derives from the structure, worked out
    when the SPEC is built (`script-preparation.md` § 4.1): the free range
    as SIESTA numbers it, and the count of atoms held.

    Refuses a copy whose free atoms are not one trailing run -- the
    signature of a structure that did not come through the sort -- rather
    than nudging the wrong atoms.
    """
    n = struct.n_atoms
    held = sorted(int(i) for i in (struct.frozen_atoms or []))
    held_set = set(held)
    free = [i for i in range(n) if i not in held_set]
    if not free:
        raise ValueError(
            "every atom is held: a force-constant run needs at least one "
            "free atom to nudge")
    if free != list(range(n - len(free), n)):
        raise ValueError(
            "the free atoms are not one trailing run of atom numbers, so "
            "no FC.First/FC.Last can name them.  The vibration deck is "
            f"rendered from a copy sorted with the {SORT_HELD_FIRST!r} key, "
            "which records its order (model/overview.md 2.2a) -- `prep` "
            "does this; a deck rendered "
            "from the user order is a deck rendered from the wrong "
            "structure.")
    return {
        # SIESTA'S NUMBERS of the deck's own atom list, said where they are
        # stated -- the pipeline log's row among them (F5).
        "numbering": "SIESTA's, from 1, of the deck's own atom list",
        "fc_first": siesta_numbering(free[0]),
        "fc_last": siesta_numbering(free[-1]),
        "n_free": len(free),
        "n_held": len(held),
        "fc_displacement": float(cfg.fc_displacement),
    }


def fc_block(struct: Structure, cfg, derived: dict) -> str:
    """The run type, the range and the partial charges at every step --
    structural text, like the electrode blocks of a transport deck: fixed
    by the kind, not a parameter anybody answers (`engines/vibration.md`
    § 5.3).  ``FC.Displacement`` is NOT here: it is a catalogue item and
    reaches the deck through the syntax door in its own section, with its
    declaration, unit and note.
    """
    f = derived["fc"]
    v = bool(getattr(cfg, "verbose_comments", True))
    out: List[str] = [""]
    if v:
        out += [
            "# --- The force-constant run ---",
            "# MD.TypeOfRun FC nudges atoms FC.First..FC.Last by FC.Displacement",
            "# along x, y, z, both ways, and writes the force constants to",
            f"# {cfg.system_label}.FC (engines/vibration.md § 5.4).  No atom is",
            "# relaxed here: the coordinates are taken as the stationary point",
            "# -- the ones the run's relax call reached, or the input's when the",
            "# person stated them relaxed (already_relaxed) -- and the record",
            "# step after this call judges them with the forces of FC step 0,",
            f"# then writes {cfg.system_label}.spectra.json at the run's root",
            "# (engines/vibration.md 5.5, 5.11 3).",
            f"# The range is the {f['n_free']} FREE atom(s); the {f['n_held']} held",
            "# atom(s) come first in this deck's numbering because the structure",
            f"# was sorted with the {SORT_HELD_FIRST!r} key -- the order record",
            "# back to the user order is this deck's ATOM-METADATA block",
            "# (model/overview.md 2.2a).  Fixed by the kind, not a preference.",
        ]
    out += [
        "MD.TypeOfRun      FC",
        f"FC.First          {f['fc_first']}",
        f"FC.Last           {f['fc_last']}",
        "",
    ]
    return "\n".join(out)


def start_state_lines(cfg) -> List[str]:
    """What the force-constant run reads from its directory -- fixed by
    the kind, not by the description's ``restart`` (the catalogue offers
    that item to the relaxation rungs only -- an optimization's and a
    vibration's -- and a force-constant run has no optimizer history to
    resume).

    THE DENSITY IS READ WHEN PRESENT.  The reference step's SCF starts
    from it, and SIESTA saves that step's converged density and reloads it
    before EVERY displacement (SIESTA 5.4.2 `save_density_matrix.F90`,
    `m_new_dm.F90`), so each displacement starts from the undisplaced
    density.  A stale density costs iterations, never the answer.

    THE GEOMETRY IS NEVER READ.  A force-constant run writes its LAST
    DISPLACEMENT to ``<SystemLabel>.XV`` -- measured on a two-atom run
    (2026-09-23, SIESTA 5.4.2): the last free atom sits
    FC.Displacement off the input along z after the run.  A deck that
    honoured that file would take a nudged geometry as the stationary
    point and converge without complaint.  SIESTA reads a density it
    finds unless told ``.false.`` (``DM.UseSaveDM`` defaults true) and a
    geometry only when told ``.true.`` (``MD.UseSaveXV`` defaults false),
    so both answers are written, not left out.

    The keys are the restart group's own declaration (the catalogue's
    ``[item.restart].expands``); the optimizer's history has no answer
    here because nothing here optimizes.
    """
    answer = {"DM.UseSaveDM": ".true.", "MD.UseSaveXV": ".false."}
    keys = _sc.parameter("restart", "siesta").writes
    return [f"{k:<18}{answer[k]}" for k in keys if k in answer]


def stage_science(cfg, derived: dict, part: Optional[str]) -> str:
    """The one-line header for a vibration call, in the deck's own units --
    `_stage_science`'s job for each part of the run (`engines/vibration.md`
    § 5.11 § 4): the relaxation's as an optimisation's, the force-constant
    run's range, the optical call's light."""
    if part == FC_PART:
        f = derived["fc"]
        return (f"force constants over SIESTA's atoms {f['fc_first']}.."
                f"{f['fc_last']} -- its own numbering, of the deck's "
                f"held-first order ({f['n_free']} free, {f['n_held']} held), "
                f"FC.Displacement {f['fc_displacement']:g} Bohr, "
                f"MeshCutoff {cfg.mesh_cutoff} Ry")
    if part == OPTICAL_PART:
        o = derived["optical"]
        return (f"the optical single point: a field along "
                f"({', '.join(f'{x:g}' for x in o['direction'])}), the sum "
                f"over {'x'.join(map(str, o['mesh']))} k-points, "
                f"Optical.Broaden {cfg.optical_broaden:g} Ry, "
                f"MeshCutoff {cfg.mesh_cutoff} Ry")
    return (f"the relaxation of the run, {phase('relax').role}: "
            f"MeshCutoff {cfg.mesh_cutoff} Ry, force tol "
            f"{cfg.relax_force_tol} eV/Ang, {cfg.relax_type}, max "
            f"{cfg.relax_steps} steps")


__all__ = ["RELAX_PART", "FC_PART", "OPTICAL_PART", "PARTS", "optical_facts",
           "optical_block", "fc_facts", "fc_block", "start_state_lines",
           "stage_science"]
