"""The vibration calculation's deck on SIESTA — the force-constant run.

Contract: `engines/vibration.md` § 5 (the deck, the sorted copy, the
read-back) and `science/normal-modes.md` § 3; the seam this
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
held atoms first, free atoms last, the permutation recorded beside the
calculation.  This module only CHECKS that shape and names the range; it
does not sort, because a deck writer that reorders atoms is a second place
deciding an atom's number.
"""
from __future__ import annotations

from typing import List

from .. import script_emit as _sc
from ..engine_atom_index import siesta_atom_index
from ..structure import Structure


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
            "rendered from a copy sorted with the 'held-first' key and the "
            "permutation is recorded beside the calculation "
            "(model/overview.md 2.2) -- `prep` does this; a deck rendered "
            "from the input order is a deck rendered from the wrong "
            "structure.")
    return {
        "fc_first": siesta_atom_index(free[0]),
        "fc_last": siesta_atom_index(free[-1]),
        "n_free": len(free),
        "n_held": len(held),
        "fc_displacement": float(cfg.fc_displacement),
    }


def fc_block(struct: Structure, cfg, derived: dict) -> str:
    """The run type and the range -- structural text, like the electrode
    blocks of a transport deck: derived from the structure, not a
    parameter anybody answers.  ``FC.Displacement`` is NOT here: it is a
    catalogue item and reaches the deck through the syntax door in its
    own section, with its declaration, unit and note.
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
            "# relaxed here: the input geometry is taken as the stationary point.",
            f"# The range is the {f['n_free']} FREE atom(s); the {f['n_held']} held",
            "# atom(s) come first in this deck's numbering because the structure",
            "# was sorted with the 'held-first' key -- the permutation back to the",
            "# input order is atom-permutation.json beside this calculation",
            "# (model/overview.md 2.2).  Fixed by the kind, not a preference.",
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
    that item to optimizations only, and a force-constant run has no
    optimizer history to resume).

    THE DENSITY IS READ WHEN PRESENT.  The first displacement's SCF starts
    from it; every later one starts from the previous displacement's, in
    memory.  A stale density costs iterations, never the answer.

    THE GEOMETRY IS NEVER READ.  A force-constant run writes its LAST
    DISPLACEMENT to ``<SystemLabel>.XV`` -- measured on the two-atom run
    (``tests/fixtures/siesta_fc``): the last free atom sits
    FC.Displacement off the input along z after the run.  A deck that
    honoured that file would take a nudged geometry as the stationary
    point and converge without complaint.  SIESTA reads the files it
    finds unless a deck says ``.false.``, so the answer is written, not
    left out.

    The keys are the restart group's own declaration (the catalogue's
    ``[item.restart].expands``), so a member the engine gains is written
    here without a second list; the optimizer's history has no member
    because nothing here optimizes.
    """
    answer = {"DM.UseSaveDM": ".true.", "MD.UseSaveXV": ".false."}
    keys = _sc.parameter("restart", "siesta").writes
    return [f"{k:<18}{answer[k]}" for k in keys if k in answer]


def stage_science(cfg, derived: dict) -> str:
    """The one-line header for a force-constant rung, in the deck's own
    units -- `_stage_science`'s job for a run that moves nothing."""
    f = derived["fc"]
    return (f"force constants over atoms {f['fc_first']}..{f['fc_last']} "
            f"({f['n_free']} free, {f['n_held']} held), "
            f"FC.Displacement {f['fc_displacement']:g} Bohr, "
            f"MeshCutoff {cfg.mesh_cutoff} Ry")


__all__ = ["fc_facts", "fc_block", "start_state_lines", "stage_science"]
