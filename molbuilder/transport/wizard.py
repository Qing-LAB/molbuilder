"""Electrode wizard — derive a bulk-lead model from a labeled device.

What this solves (``engines/transport.md`` § 5, invariants I2/I5/I6/I10)
=======================================================

The #1 TranSIESTA footgun is that the device run and the separate
electrode run must share **one geometry + one numerical contract**, yet
the user assembles the electrode by hand and the couplings silently
drift (invariant set § 6.7).  The wizard removes the hand-assembly: it
extracts each lead's *exact* atoms (``L-electrode``, ``R-electrode``) from
the device, and the electrode rung's deck is rendered from that model
(`transport/deck.py`), so the cross-run
invariants hold **by construction**, not by the user remembering to copy
values.

Guaranteed-by-construction invariants
-------------------------------------

* **I2 (pseudos) / I10 (geometry):** the electrode atoms are the device
  electrode-region coordinates + species, copied verbatim.
* **I6 (lateral cell):** the electrode ``a, b`` are the device's own lattice
  vectors, **copied verbatim** -- character and all, so a hexagonal Au(111)
  surface stays hexagonal and the lead tiles the device cross-section.
* **I7 (transverse k):** the electrode's ``(kx, ky)`` and offset are the
  device's -- one shared ``kgrid`` and ``kgrid_displacement``, which every
  rung's k-point mesh reads (``kmesh.mesh_for``, `engines/siesta.md` § 6.1).
* **I1/I3/I4/I5 (XC/MeshCutoff/EnergyShift/basis):** the lead and the
  device render them from the **same catalogue sections**
  (``BASIS_SECTION`` / ``XC_SECTION`` + ``electronic_temperature``), so the
  electronic contract is identical by construction.
* **I9 (electrode kz dense) / I13 (writes ``.TSHS``):** set by the electrode
  rung's deck (`transport/deck.py`).

What the USER must still verify (warned, not guaranteed)
-------------------------------------------------------

* **The bulk z-period.**  A finite slab does not tell us the lead's true
  periodic repeat unambiguously.  We *derive* it as
  ``z_period = z_span + d_interlayer`` (the block's one layer spacing,
  checked rather than averaged) so the
  slab tiles seamlessly under uniform spacing, and **warn** that the user
  must confirm it matches the real bulk lattice (e.g. the layer count is a
  whole stacking period — a multiple of 3 for FCC(111) ABC).  ``z_period``
  overrides it.
* **Thickness vs the principal layer (I11):** reported here;
  `transport/compose.py` gates on it.

This is the geometry derivation only; it does NOT run SIESTA or write a
deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..cell import (LAYER_TOL_ANG, bulk_z_period, classify_seam,
                    detect_layers)
from ..structure import Structure
from .sort import ELECTRODE_LABELS
from .transiesta import (
    _compute_cell_from_extents,
    _find_electrode_regions,
)

#: A heuristic advisory floor (Å) for the wizard's note below --
#: nothing refuses on it.  The REAL principal-layer gate lives in
#: `transport/compose.py` and compares the orbital interaction range
#: (read from the citation's own ``.ion`` files, never guessed)
#: against the lead's period (engines/transport.md § 3).
MIN_ELECTRODE_THICKNESS_ANG = 12.0


@dataclass
class ElectrodeModel:
    """The derived bulk-lead geometry + the invariants it pins.

    ``positions`` are shifted so the lowest layer sits at z = 0; the
    lateral cell is the *device's* (a, b); ``z_period`` is the proposed
    bulk repeat along the transport axis.
    """

    label: str                       # device region label, e.g. "L-electrode"
    block_name: str                  # sanitized TS.Elec name, e.g. "L"
    elements: List[str]
    positions: np.ndarray            # (M, 3) Å, shifted to min-z = 0
    lat_a: np.ndarray                # device lateral vector a (3,), Å
    lat_b: np.ndarray                # device lateral vector b (3,), Å
    cell_a: float                    # |lat_a| (Å), for display/checks
    cell_b: float                    # |lat_b| (Å)
    z_period: float                  # proposed bulk repeat (Å)
    z_span: float                    # top-layer − bottom-layer z (Å)
    n_layers: int
    d_interlayer: float              # THE layer spacing (Å) -- checked equal
    n_atoms: int
    #: The DEVICE's transverse axis kinds, carried so the lead can state its
    #: own periodicity honestly.  A slab electrode tiles the plane and is
    #: `periodic, periodic`; a nanowire or chain lead is vacuum-surrounded
    #: and is `isolated, isolated` -- and `as_structure` must not assert the
    #: first about the second (user, 2026-09-23).
    transverse_kind: Tuple[str, str] = ("periodic", "periodic")
    notes: List[str] = field(default_factory=list)

    def as_structure(self) -> Structure:
        """The lead as an ordinary :class:`~molbuilder.structure.Structure`,
        carrying its own cell.

        **Not a conversion into something new — a restatement.** These atoms
        came out of the cited junction by their region label
        (:func:`extract_electrode_model`): same atoms, same relaxation, a
        subset selected rather than a geometry derived from elsewhere. That
        is the point of the one-file design — *"all the structure and facts
        involved in the calculation are consistently constructed"* (user,
        2026-09-16).

        What it buys is that the framework's seam can stay what it is. A deck
        describes a structure, `spec_for(struct, cfg, …)` takes one, and an
        electrode rung describes **this** structure while the device rung
        describes the junction. Without it the seam would have to grow an
        argument for a composite kind, and four other callers would carry a
        parameter they never use.

        The cell is the lead's own, and its two halves come from different
        places ON PURPOSE (`engines/transport.md` § 6.2):

        * ``lat_a``/``lat_b`` are the device's lattice vectors **copied** --
          I6, and a computed transverse box would sever the crystal;
        * the third vector is the bulk repeat **computed** from the layer
          spacing, because a finite slab cannot state its own period.  It is
          the one number a person is asked to verify (§ 7.1).

        So "derived" is not the suspicious word here; *fabricated from atom
        extents* is.

        WHAT THIS DELIBERATELY DOES NOT CARRY.  Only elements, positions,
        title, cell and ``axis_kind`` are stated, so a lead reaches the
        renderer with no ``regions``, no ``annotations``, no ``info`` and no
        identity columns.  The partition is right to drop -- a bulk lead has
        no lead/bridge/buffer division to state, and the electrode deck emits
        no ``%block TS.Elec``.  **The other three are an open question, not a
        decision**: `model/structure.md` § 2.2a says a strip must be explicit
        and names a silent one a defect, and the recorded contract in ``info``
        is exactly what the citation warnings read.  Recorded here rather than
        quietly fixed, because whether a lead inherits the junction's recorded
        contract is a contract question (`plans/plan.md`, 2026-09-23).
        """
        cell = np.array([self.lat_a, self.lat_b,
                         [0.0, 0.0, float(self.z_period)]], dtype=float)
        # STATED AT CONSTRUCTION, not assigned afterwards: a field write
        # skips `__post_init__`, so the cell would go unchecked and the
        # periodicity unsettled.
        #
        # THE TRANSPORT AXIS IS THE ONE A LEAD CHANGES.  The device is OPEN
        # along transport -- the leads enter as self-energies -- and the lead
        # is not: it is genuinely periodic there, and that difference is the
        # whole reason the lead is computed separately.
        #
        # ACROSS the wire it is whatever the device is, and asserting
        # `periodic` there was wrong for a real case.  A slab electrode tiles
        # the plane; a NANOWIRE OR CHAIN lead is vacuum-surrounded, so a
        # device of `isolated, isolated, transport` yields a lead of
        # `isolated, isolated, periodic` (user, 2026-09-23).
        #
        # The lead's k-point mesh reads these kinds (`kmesh.mesh_for`,
        # `engines/siesta.md` § 6.1): a transverse count above 1 on an
        # isolated axis is warned, and the transport axis is the lead's own,
        # `electrode_kz`.
        return Structure(
            elements=list(self.elements),
            positions=np.asarray(self.positions, dtype=float).copy(),
            title=f"bulk lead ({self.label})",
            cell=cell,
            axis_kind=(self.transverse_kind[0], self.transverse_kind[1],
                       "periodic"))


# --------------------------------------------------------------------- #
#  Geometry derivation                                                  #
# --------------------------------------------------------------------- #


#: How far a frozen atom may sit from where the relaxation started
#: before the constraint is judged broken (``engines/transport.md`` § 3).
#: A constrained relaxation reproduces its fixed atoms to writing
#: precision; this absorbs the .XV's Angstrom/Bohr round trip.
FROZEN_TOL_ANG = 1e-3


def _atoms_named(device: Structure, idxs, atom_ids=None,
                 limit: int = 6) -> str:
    """``3 (Au), 4 (Au) and 12 more`` -- the spelling `validation/sidecar`
    uses, in the identity the person can act on.

    `engine_atom_index`: the canonical atom identity is the index in the
    source file's order, which is what the Modify tab shows.  A device
    that has been through `categorical_sort` is in TranSIESTA's deck
    order instead, so *atom_ids* carries ``sorted_to_original``; ``None``
    means this device's indices are already canonical.
    """
    els = getattr(device, "elements", ()) or ()

    def name(i):
        shown_i = atom_ids[i] if atom_ids is not None else i
        return f"{shown_i} ({els[i]})" if i < len(els) else str(shown_i)

    shown = ", ".join(name(i) for i in idxs[:limit])
    more = f" and {len(idxs) - limit} more" if len(idxs) > limit else ""
    return shown + more


def _seam_note(pos, lat_a, lat_b, zper: float) -> str:
    """What this lead's periodic boundary does to the crystal, measured.

    A lead tiles along z, so its top layer meets its own bottom layer one
    cell up.  Whether that seam continues the crystal is a layer-COUNT
    question: (111) stacks ABC, so only a multiple of three continues;
    four layers puts the image's first layer back on the same sites
    (`eclipsed`, a head-on contact at the interlayer distance instead of
    the nearest-neighbour one) and five gives a mirror `twin`.
    `junction-cell.md` § 3.1.

    Reported, never refused -- the rule `blueprints/transport.py` states
    for the electrode orientation.  A faulted seam is wrong for a bulk
    lead and the person is the one who knows whether they meant it.
    """
    cell = np.array([lat_a, lat_b, [0.0, 0.0, float(zper)]], dtype=float)
    try:
        v = classify_seam(pos, cell)
    except Exception as exc:                       # noqa: BLE001
        return (f"the periodic seam could not be classified ({exc}); "
                f"the lead's tiling is UNCHECKED")
    n_layers = len(detect_layers(pos[:, 2], LAYER_TOL_ANG))
    if n_layers < 3:
        # TWO LAYERS CANNOT SAY.  A,B is fcc and hcp alike -- the block
        # carries only two registries, so ABAB and ABC agree on every
        # layer it has and the tiling picks one without the block having
        # chosen.  `classify_seam` answers `continues, period 2` here for
        # any plane, which is right for (100)/(110) and wrong for (111),
        # and nothing on the lead path knows which plane it is.  So the
        # limit is stated rather than a verdict asserted.
        plural = "layer" if n_layers == 1 else "layers"
        return (f"the periodic seam is UNDETERMINED: a block of "
                f"{n_layers} {plural} carries fewer stacking registries "
                f"than one period, so it cannot say which crystal it "
                f"tiles into -- ABAB and ABC agree on everything it "
                f"contains, which on fcc(111) is the difference between "
                f"gold and hcp.  A lead that states its own stacking is "
                f"at least one whole period thick (3 layers for (111); "
                f"junction-cell.md § 3.1).")
    if v.verdict == "continues":
        per = f" (stacking period {v.period} layers)" if v.period else ""
        return (f"the periodic seam CONTINUES the crystal{per}: the layer "
                f"one cell up sits where the stacking says it should, "
                f"{v.gap:.3f} Å from the top layer.")
    return (f"the periodic seam is {v.verdict.upper()} — {v.message}.  "
            f"This lead is what TranSIESTA turns into the self-energy, so "
            f"a faulted seam is a faulted bulk Hamiltonian.  Not refused: "
            f"you may mean it (junction-cell.md § 3.1).")


def _spacing_or_refuse(layer_z, label: str):
    """:func:`cell.bulk_z_period`, with the block's name on the refusal.

    The rule is `cell`'s; the label is this layer's, and `cell` has no
    way to know it.  Without it a two-lead junction of one element gives
    no way to tell which end to re-label.
    """
    try:
        return bulk_z_period(layer_z)
    except ValueError as exc:
        raise ValueError(f"the {label} block: {exc}") from exc


def _refuse_unless_frozen_bulk(
    device: Structure,
    label: str,
    idxs,
    prior_positions: Optional[np.ndarray],
    atom_ids=None,
) -> None:
    """The lead must be frozen, and must have stayed where it was.

    Split out so :func:`extract_electrode_model` reads as *check, then
    build* rather than interleaving the two; see that docstring for why
    the order of these questions is a dependency and not a preference.
    """
    frozen = set(getattr(device, "frozen_atoms", None) or ())
    loose = [i for i in idxs if i not in frozen]
    if loose:
        raise ValueError(
            f"{len(loose)} atom(s) in {label!r} are NOT FROZEN: "
            f"{_atoms_named(device, loose, atom_ids)}.  A lead is the "
            f"pristine bulk "
            f"the self-energy attaches to, so every atom carrying an "
            f"electrode label must be held still — freeze them (the Modify "
            f"tab's selection writes \"frozen_atoms\"), then relax and cite "
            f"again.  Freezing them now does not make the geometry they are "
            f"already in bulk -- nothing held them while the bridge relaxed, "
            f"so whether they moved was never constrained")

    if prior_positions is None:
        return
    prior = np.asarray(prior_positions, dtype=float)
    now = np.asarray(device.positions, dtype=float)
    if prior.shape != now.shape:
        raise ValueError(
            f"cannot check {label!r} against the geometry the relaxation "
            f"started from: {len(prior)} atoms there, {len(now)} here")
    moved = [(i, float(np.linalg.norm(now[i] - prior[i]))) for i in idxs]
    moved = [(i, d) for i, d in moved if d > FROZEN_TOL_ANG]
    if moved:
        shown = "; ".join(
            f"atom {atom_ids[i] if atom_ids is not None else i} "
            f"({device.elements[i]}) moved {d:.4f} A"
            for i, d in moved[:6])
        more = f" and {len(moved) - 6} more" if len(moved) > 6 else ""
        raise ValueError(
            f"{len(moved)} atom(s) in {label!r} MOVED during the cited "
            f"relaxation: {shown}{more}.  Frozen means unmoved "
            f"(engines/transport.md § 4): the "
            f"electrode blocks "
            f"are the seam the self-energies attach to.  Re-relax the "
            f"junction with the electrode atoms constrained, or fix the "
            f"labels")


def extract_electrode_model(
    device: Structure,
    label: str,
    *,
    prior_positions: Optional[np.ndarray] = None,
    atom_ids=None,
    z_period: Optional[float] = None,
    layer_tol_ang: float = LAYER_TOL_ANG,
    min_thickness_ang: float = MIN_ELECTRODE_THICKNESS_ANG,
) -> ElectrodeModel:
    """Build an :class:`ElectrodeModel` for one lead, ``L-electrode`` or
    ``R-electrode`` — and refuse the region outright if it is not a bulk lead.

    The lateral cell is the **device's** (a, b) — so the lead tiles the
    device cross-section (I6).  The z-period is derived from the layer
    spacing unless ``z_period`` is given.

    **This is the one gate.**  Three questions, in the order their
    answers depend on one another, each refusal naming what to do:

    1. **Declared frozen** -- every atom of the region is in
       ``frozen_atoms`` (``engines/transport.md`` § 4: the lead atoms
       are frozen bulk by construction).
    2. **Actually unmoved**, when *prior_positions* is given -- the
       geometry the relaxation started from, in this structure's index
       order.  Form B has none and passes on 1 alone, which is why 1
       exists separately.
    3. **Evenly spaced** -- :func:`cell.bulk_z_period`, which refuses a
       block whose layers do not share one spacing.

    1 before 2 because it is cheaper and its answer is the fix for both;
    1 and 2 before 3 because the spacings of a block that moved describe
    nothing.  Raises ``ValueError``; `compose` turns it into a
    ``ComposeError`` verbatim.

    *atom_ids* maps this device's indices back to the canonical ones the
    person sees; see :func:`_atoms_named`.
    """
    electrodes = {lab: (name, idxs)
                  for lab, name, idxs in _find_electrode_regions(device)}
    if label not in electrodes:
        raise ValueError(
            f"region {label!r} is not an electrode region; found "
            f"{sorted(electrodes) or 'none'} "
            f"(the leads are {' and '.join(ELECTRODE_LABELS)})")
    block_name, idxs = electrodes[label]
    idxs = sorted(idxs)

    _refuse_unless_frozen_bulk(device, label, idxs, prior_positions,
                               atom_ids)

    pos = np.asarray(device.positions, dtype=float)[idxs]
    elems = [device.elements[i] for i in idxs]

    # Lateral cell from the DEVICE (not the electrode's own extent) so
    # the lead tiles the device cross-section exactly (I6).  Prefer the
    # device's REAL lattice vectors (preserves a hexagonal Au(111)
    # surface); fall back to an orthorhombic extent box only when the
    # device carries no cell.
    if device.cell is not None:
        dc = np.asarray(device.cell, dtype=float)
        lat_a, lat_b = dc[0].copy(), dc[1].copy()
    else:
        a, b, _c = _compute_cell_from_extents(device)
        lat_a, lat_b = np.array([a, 0.0, 0.0]), np.array([0.0, b, 0.0])

    # Shift so the lowest layer is at z = 0 (clean periodic cell).
    z = pos[:, 2]
    layer_z = detect_layers(z, layer_tol_ang)
    pos = pos.copy()
    pos[:, 2] = pos[:, 2] - float(min(layer_z)) if layer_z else pos[:, 2]
    layer_z = [lz - float(min(layer_z)) for lz in layer_z] if layer_z else []

    notes: List[str] = []
    if z_period is not None:
        zper = float(z_period)
        # Still report the detected layer structure for the thickness check --
        # measured through the SAME derivation, with only the period overridden
        # (science/junction-cell.md § 5).
        if len(layer_z) >= 2:
            _derived, d_inter, n_layers = _spacing_or_refuse(layer_z, label)
        else:
            d_inter, n_layers = float("nan"), len(layer_z)
        z_span = float(layer_z[-1] - layer_z[0]) if layer_z else 0.0
        notes.append(
            f"z-period set explicitly to {zper:.3f} Å (overriding the "
            f"layer-spacing estimate).")
    else:
        zper, d_inter, n_layers = _spacing_or_refuse(layer_z, label)
        z_span = float(layer_z[-1] - layer_z[0])
        notes.append(
            f"z-period DERIVED as z_span ({z_span:.3f}) + layer spacing "
            f"({d_inter:.3f}) = {zper:.3f} Å, from {n_layers} layers.")

    notes.append(_seam_note(pos, lat_a, lat_b, zper))

    if z_span < min_thickness_ang:
        notes.append(
            f"electrode z-span {z_span:.2f} Å < ~{min_thickness_ang:.0f} Å: "
            f"may be thinner than the electronic principal layer (§ 4.1).  "
            f"The compose gate verifies the real condition against the "
            f"basis's orbital ranges when .ion files sit beside the "
            f"citation; consider more lead layers if it refuses.")

    # THE DEVICE'S OWN ANSWER, not a guess: the lead tiles the same
    # cross-section, so it is periodic across the wire exactly when the
    # device is (`engines/transport.md` § 5, I6).
    # `("isolated",) * 3` like every other such fallback in the tree;
    # unreachable -- `__post_init__` always fills `axis_kind`.
    dev_kind = tuple(getattr(device, "axis_kind", None)
                     or ("isolated",) * 3)
    return ElectrodeModel(
        label=label, block_name=block_name, elements=elems, positions=pos,
        lat_a=lat_a, lat_b=lat_b,
        transverse_kind=(dev_kind[0], dev_kind[1]),
        cell_a=float(np.linalg.norm(lat_a)), cell_b=float(np.linalg.norm(lat_b)),
        z_period=zper, z_span=z_span, n_layers=n_layers,
        d_interlayer=d_inter, n_atoms=len(elems), notes=notes)


__all__ = [
    "ElectrodeModel",
    "FROZEN_TOL_ANG",
    "extract_electrode_model",
]
