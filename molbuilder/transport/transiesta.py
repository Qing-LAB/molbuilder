"""The TranSIESTA emitters — the transport composite's deck layer.

The TranSIESTA emitters `transport/deck.py` (and `wizard.py`, for the lead's
box) reach into:

* :func:`emit_electrode_declarations` and :func:`_emit_geometry` — reused by
  `transport/deck.py`, the pipeline every one of the five rungs renders
  through.
* :func:`_compute_cell_from_extents` and
  :func:`_find_electrode_regions` — reused by `transport/wizard.py`, whose
  `extract_electrode_model` derives the lead `compose.py` hands to prep.
* :func:`electrode_hs_stem` — the ONE spelling of an electrode run's
  identity, so the device deck's ``HS`` line and the electrode deck's
  ``SystemLabel`` cannot disagree.
* :func:`pole_count` — TranSIESTA's rule for how many poles an equilibrium
  pole ENERGY gives, written once for the settings gate's refusal and the
  count the device deck states beside the energy.
This module emits NO deck of its own and holds NO gate.  Transport's science
is the KIND's — ``_KIND_VALIDATORS["transport"]`` — and its decks are written
by ``transport/deck.py`` through ``spec_for`` → ``prepare_deck``.  The record
a transmission inspector reads is `transport/record.py`'s,
``<label>.transport.json``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .sort import (
    ELECTRODE_LABELS,
    REGION_BUFFER,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
)
from ..structure import Structure


#: Transverse vacuum per side (Å) for an ISOLATED electrode — a nanowire or
#: chain lead, periodic along transport and genuinely vacuum-surrounded
#: across it — when the structure states no vacuum of its own.
#:
#: 15 Å is the SIESTA-canonical floor for isolated transverse padding
#: (electronic tails fall off by ~5–8 Å in vacuum).  Lower → exchange-
#: correlation tails leak; higher → unnecessary basis cost.  **A transport
#: lead needs more than an ordinary molecule does**: it is the object the
#: self-energy is built from, so its images must be electrostatically
#: isolated or Σ describes a wire coupled to its own copies.
#:
#: A TRANSPORT DEFAULT, NOT AN OVERRIDE.  `Structure.effective_vacuum`
#: supplies 3 Å where nobody has chosen — right for a molecule in a box and
#: five times too thin for a lead — so transport answers the same question
#: with its own default, the way the electrode's dense transport-axis k is a
#: default rather than something a person must discover
#: (`engines/transport.md` § 2a.7).  What it is NOT is a second home for the
#: value: a structure that STATES a vacuum is obeyed verbatim, because
#: `vacuum` is already the person's control (Modify → Cell, carried in the
#: sidecar).
_ISOLATED_ELECTRODE_VACUUM_ANG = 15.0


#: Each lead's TranSIESTA name, the ``<name>`` of its ``%block
#: TS.Elec.<name>``.  TranSIESTA takes any string there; these are the two
#: the decks have always carried.
_BLOCK_NAME = {REGION_LEFT_ELECTRODE: "L", REGION_RIGHT_ELECTRODE: "R"}


def electrode_hs_stem(job_name: str, label: str) -> str:
    """The ONE spelling of an electrode run's identity — its SystemLabel,
    and therefore the stem of the ``.TSHS`` the device deck references.

    Two writers need it to agree byte-for-byte: the device deck's
    ``TS.Elec.<name> HS`` line (below) and `jobset/prep.py`, which names
    the electrode rung's ``SystemLabel`` with it so SIESTA writes exactly
    the file the device will ask for (and `stages.stage_inputs` names the
    ``.TSHS`` a rung consumes).
    """
    return f"{job_name}_{label}"


#: TranSIESTA's floor: the continued-fraction branch stops a run whose
#: equilibrium contour has fewer poles than this (SIESTA 5.4.2
#: `Src/m_ts_chem_pot.F90`:324) -- after the queue wait.
MIN_EQ_POLES = 20

#: TranSIESTA's other floor: it stops a run whose electronic temperature is
#: under this, before any pole is counted -- *"TranSiesta electronic
#: temperature *must* be larger than 10 kT"* (`m_ts_options.F90`:258-262, and
#: again per chemical potential, :284-288; the message means kelvin).
MIN_TS_TEMPERATURE_K = 10.0


def pole_count(energy_ev: float, temperature_k: float) -> int:
    """How many poles TranSIESTA takes for an equilibrium pole ENERGY.

    TranSIESTA's own rule, not a fitted one.  Our device deck declares
    ``%block TS.ChemPot.<name>`` with no ``contour.eq`` inside it, which takes
    the continued-fraction branch (`m_ts_chem_pot.F90`:299), where the count
    is ``int(E_pole / (pi * kT))`` (`:319`) and ``TS.Contours.Eq.Pole.N`` is
    overwritten by it.  So the energy is the handle and the count follows the
    temperature: the same energy gives fewer poles hotter
    (`engines/transport.md` § 6.1c).

    AN ENERGY AT OR BELOW ZERO IS NOT TAKEN AS ONE: the branch reads the
    energy only when it is positive (`:318`), so the count stays at
    ``TS.Contours.Eq.Pole.N``'s default of 8 (`:25`, `:113`) and the run
    stops at the floor -- which is what this answers for it.  At 0 K the
    rule divides by zero, so a temperature at or below zero is refused here
    rather than answered (TranSIESTA stops below :data:`MIN_TS_TEMPERATURE_K`
    first).
    """
    from ..constants import BOLTZMANN_EV_K
    kT = BOLTZMANN_EV_K * float(temperature_k)
    if kT <= 0:
        raise ValueError(
            f"the pole count is int(E / (pi kT)), undefined at "
            f"{float(temperature_k):g} K")
    if float(energy_ev) <= 0:
        return _DEFAULT_EQ_POLES
    return int(float(energy_ev) / (math.pi * kT))


#: The count TranSIESTA keeps when the energy is not positive -- the
#: `TS.Contours.Eq.Pole.N` default (`m_ts_chem_pot.F90`:25, `def_poles`).
_DEFAULT_EQ_POLES = 8


def pole_energy_for(poles: int, temperature_k: float) -> float:
    """The least energy, in eV to two decimals, that :func:`pole_count` turns
    into at least *poles* at *temperature_k* -- the rule inverted, beside it,
    so a refusal can say what to type and have it accepted.  Rounded UP: the
    count truncates, so an energy rounded to the nearest hundredth can fall
    one pole short."""
    from ..constants import BOLTZMANN_EV_K
    exact = int(poles) * math.pi * BOLTZMANN_EV_K * float(temperature_k)
    energy = math.ceil(exact * 100.0) / 100.0
    while pole_count(energy, temperature_k) < poles:      # float edge
        energy += 0.01
    return round(energy, 2)


def _find_electrode_regions(
    struct: Structure,
) -> List[Tuple[str, str, List[int]]]:
    """Return (label, block_name, indices) for each of the two leads,
    ``L-electrode`` and ``R-electrode`` by exact name (`sort.ELECTRODE_LABELS`).

    Sorted by z-centroid ascending so the first entry is the LOWER
    electrode (minimum z) and the last the upper one.  This ordering
    is what the emitter uses to assign ``semi-inf-direction -A3`` /
    ``+A3`` and the ``elec-pos`` ends — the GEOMETRIC half of the
    deck.  It does NOT decide the chemical potentials: those bind to
    the region's own name (``L-electrode`` → ``Left`` → µ = +V/2), so
    the two halves name different blocks on a junction labeled the
    other way round, and the deck says so.
    """
    out: List[Tuple[str, str, List[int], float]] = []
    regions = struct.regions or {}
    for label in ELECTRODE_LABELS:
        indices = list(regions.get(label) or ())
        if not indices:
            continue
        z_centroid = float(np.mean(struct.positions[indices, 2]))
        out.append((label, _BLOCK_NAME[label], indices, z_centroid))
    out.sort(key=lambda e: e[3])
    return [(label, name, idxs) for (label, name, idxs, _z) in out]


def _compute_cell_from_extents(struct: Structure) -> Tuple[float, float, float]:
    """The box for a structure that states none — PER AXIS, by its kind.

    `Structure.resolve_cell`'s rule, with transport's own vacuum default:

      * ``isolated``  -> ``bbox + 2 * vacuum``.  A nanowire or chain lead is
        vacuum-surrounded across the wire and this is its cross-section;
      * ``transport`` -> ``bbox``.  The device length is matched, not padded
        (§ 6.2), and vacuum is meaningless there;
      * ``periodic``  -> **refused.**  § 7: *"the box is NOT recoverable from
        atom extents (padding fabricates an orthorhombic box that severs the
        periodic gold)"* — a rectangle cannot tile Au(111), so there is no
        honest number and none is invented.

    THE VACUUM IS THE PERSON'S.  Three states
    (`structure-periodicity.md` § 2): a number used verbatim, ``[0,0,0]``
    meaning no gap DELIBERATELY and also used, and unset -- the only one
    :data:`_ISOLATED_ELECTRODE_VACUUM_ANG` answers.

    **Why not just call `resolve_cell`?**  Because its default where nobody
    chose is 3 A -- right for a molecule in a box and five times too thin
    for a lead, which is the object the self-energy is built from and whose
    images must be electrostatically isolated.  The RULE is shared; only the
    default differs, the way the electrode's dense transport-axis k is a
    transport default rather than something to discover (§ 2a.7).
    """
    pos = struct.positions
    extent = pos.max(axis=0) - pos.min(axis=0)
    stated = struct.vacuum
    out = []
    for i, kind in enumerate(struct.axis_kind):
        if kind == "periodic":
            raise ValueError(
                f"axis {i} is 'periodic' but this structure states no cell; "
                f"a periodic axis needs a commensurate lattice from "
                f"construction or import, never a bounding box "
                f"(engines/transport.md 7)")
        per_side = (float(stated[i]) if stated is not None
                    else _ISOLATED_ELECTRODE_VACUUM_ANG)
        pad = 2.0 * per_side if kind == "isolated" else 0.0
        out.append(float(extent[i]) + pad)
    return (out[0], out[1], out[2])


# --------------------------------------------------------------------- #
#  .fdf emission helpers -- the geometry and electrode blocks           #
#  `transport/deck.py` lays into the deck (`_emit_geometry`,            #
#  `emit_electrode_declarations`: lists of lines) and the box and       #
#  frame they are written in.                                           #
# --------------------------------------------------------------------- #


def axis_vacuum(cell: np.ndarray,
                positions: np.ndarray) -> List[float]:
    """Per-axis vacuum gap (Å): the part of each lattice vector NOT
    spanned by the atoms.

    Computes fractional coordinates, takes each axis' atom span, and
    returns ``(1 - span_frac) * |lattice_vector|``.  For a genuinely
    periodic axis this is roughly one inter-layer spacing (the next
    atom is the periodic image); for a vacuum axis it is the real
    empty padding.  A large value on an axis the user declared
    periodic (or on the transport axis) is the diagnostic the
    boundary check warns on.
    """
    cell = np.asarray(cell, dtype=float)
    pos = np.asarray(positions, dtype=float)
    inv = np.linalg.inv(cell)
    frac = pos @ inv                                  # (N, 3)
    veclen = np.linalg.norm(cell, axis=1)             # |a|, |b|, |c|
    out: List[float] = []
    for ax in range(3):
        span = float(frac[:, ax].max() - frac[:, ax].min())
        out.append(max(0.0, (1.0 - span) * float(veclen[ax])))
    return out


# Above this much empty space (Å) on a periodic axis we flag it: a
# real bulk lattice leaves ~one interlayer spacing (a few Å); much
# more usually means the axis is actually vacuum (or the cell is
# mis-sized), which changes the physics.
_VACUUM_FLAG_ANG = 5.0


def _lattice_block(struct: Structure, cell: np.ndarray, *,
                   fabricated: bool = False) -> List[str]:
    """Emit the LatticeVectors block.

    If ``cell`` is provided (the structure's real lattice — hexagonal,
    triclinic, whatever), it is emitted VERBATIM and the per-axis
    vacuum is reported with a warning when an axis declared periodic
    leaves large empty space or the transport axis (c) has vacuum.

    ``fabricated`` says the structure stated no lattice, so ``cell`` is the
    orthorhombic vacuum box :func:`_emit_geometry` built from atom extents — a
    model of an ISOLATED cluster, flagged loudly because it is wrong for a
    periodic surface electrode (the hex Au(111) case).  It is written at the
    precision the atoms were placed against, never rounded.

    **THIS ARM IS FOR AN ISOLATED ELECTRODE, AND IT IS A REAL CASE**
    (user, 2026-09-23).  A nanowire or chain lead is periodic along
    transport and genuinely vacuum-surrounded across it, so a derived
    transverse box IS the model -- and it needs MORE vacuum than an
    ordinary molecule, because the lead is what the self-energy is built
    from and its images must be electrostatically isolated.

    It is NOT for a periodic surface electrode.  `engines/transport.md`
    § 7 is about that case: *"the box is NOT recoverable from atom extents
    (padding fabricates an orthorhombic box that severs the periodic
    gold)"* -- a rectangle cannot tile Au(111).  That case never reaches
    here: the citation door refuses a junction stating no cell, and I6 is
    held by copying the device's lateral vectors (§ 5).
    """
    lines = ["LatticeConstant        1.0 Ang"]
    cell = np.asarray(cell, dtype=float)
    if not fabricated:
        # THE KIND, NOT THE BOOLEAN: `struct.pbc` flattens `transport` and
        # `periodic` to the same True, and the per-axis line and the warning
        # below need them apart.  The fallback answers `("isolated",) * 3`,
        # as every other `axis_kind or ...` in the tree does (`__post_init__`
        # always fills the kinds).
        kinds = struct.axis_kind or ("isolated",) * 3
        vac = axis_vacuum(cell, struct.positions)
        lines += [
            "# Explicit lattice preserved from the structure (NOT",
            "# recomputed from atom extents).  Per-axis boundary:",
        ]
        names = ("a", "b", "c (transport)")
        for ax in range(3):
            kind = kinds[ax]
            lines.append(
                f"#   {names[ax]:<14} {kind:<8} | empty span "
                f"{vac[ax]:.2f} Å")
        # Transport axis (c) must be periodic / seamless for the leads.
        # A transport axis is the device length matched to the leads -- the
        # seam question is about vacuum at the boundary, not about the kind.
        # `isolated` here IS the failure this warns about.
        if vac[2] > _VACUUM_FLAG_ANG or kinds[2] == "isolated":
            lines.append(
                "# WARNING: the transport axis (c) has vacuum / is not "
                "periodic;")
            lines.append(
                "#   the electrode .TSHS cannot attach seamlessly "
                "(Brandbyge 2002 § III).")
        for ax in (0, 1):
            if kinds[ax] != "isolated" and vac[ax] > _VACUUM_FLAG_ANG:
                lines.append(
                    f"# NOTE: transverse axis {names[ax]} declared periodic "
                    f"but leaves {vac[ax]:.1f} Å empty — confirm the surface "
                    f"actually tiles (else it is an isolated cluster).")
        lines.append("%block LatticeVectors")
        for ax in range(3):
            v = cell[ax]
            lines.append(f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}")
        lines.append("%endblock LatticeVectors")
    else:
        lines += [
            "# WARNING: no lattice on the structure — an orthorhombic",
            "# VACUUM BOX was derived from the atom extents: each ISOLATED",
            "# axis gets the structure's own vacuum on both sides (15 Å per",
            "# side where none is set), and a transport axis gets the atom",
            "# span with no padding at all.  This models an",
            "# ISOLATED CLUSTER, NOT a periodic surface electrode.  For a",
            "# real Au(111) lead, supply the structure's hexagonal cell",
            "# (set Structure.cell / the molstruct sidecar's 'cell').",
            "%block LatticeVectors",
            *(f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}" for v in cell),
            "%endblock LatticeVectors",
        ]
    return lines


def engine_frame_for(struct: Structure, cell: Optional[np.ndarray] = None):
    """The frame a transport rung's atoms are placed with -- ONE box decision
    and ONE placement (`model/structure-periodicity.md` § 6.0).

    The box is ``cell`` when given, else the structure's own, else -- an
    isolated lead that states none -- the vacuum box
    :func:`_compute_cell_from_extents` sizes with transport's own default.
    ``deck.py`` computes it once per deck and hands the same frame to the
    coordinate block and to the deck's ENGINE-OFFSET record."""
    from ..cell import to_engine
    if cell is None and struct.cell is None:
        box = np.diag(_compute_cell_from_extents(struct))
    else:
        box = np.asarray(cell if cell is not None else struct.cell, dtype=float)
    return to_engine(struct, box=box)


def _emit_geometry(struct: Structure,
                   cell: Optional[np.ndarray] = None,
                   cfg=None, frame=None) -> List[str]:
    """Lattice + AtomicCoordinates blocks.

    The lattice comes from ``cell`` (or ``struct.cell``) when present —
    preserved verbatim so a hexagonal Au(111) surface keeps its real
    cell.  Only when no lattice is available does it fall back to the
    orthorhombic vacuum box (an isolated-cluster model), with a loud
    warning.

    Coordinates are emitted in the ENGINE frame by the one placement rule
    (`model/structure-periodicity.md` § 6.0): the design coordinates plus
    ``cell.engine_offset``, which centres the atoms in the box -- the rule
    every emitter takes, so a lead, the device and a relaxation cannot place
    their atoms differently.
    """
    from ..chemistry import atomic_number
    # ONE SPECIES RULE, AND ONE OVERRIDE, SHARED WITH THE SIESTA EMITTER
    # (`model/chemistry.md` § 3a, decided 2026-09-23 -- GLOBALLY, which
    # includes transport).  The index this fixes is the orbital ordering
    # inside `.DM` and `.TSHS`, a value § 2a.13 binds to every stage, so a
    # second rule here is not a variation -- it is the defect.
    from ..chemistry import species_order as _species_order
    species = _species_order(
        struct.elements,
        getattr(cfg, "species_order", None) if cfg is not None else None)
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}
    # AN OVERRIDE THAT OMITS A SPECIES REFUSES HERE: `species_order` is
    # honoured verbatim, so a list missing an element the structure contains
    # would leave `species_idx[el]` to raise a bare `KeyError` while writing
    # the coordinate block.
    missing = sorted(set(struct.elements) - set(species_idx))
    if missing:
        raise ValueError(
            f"species_order does not list {', '.join(missing)}, which this "
            f"structure contains.  The order is honoured verbatim, so every "
            f"species has to appear in it -- or leave it unset and the "
            f"default rule orders them (model/chemistry.md 3a)")

    # THE BOX, THEN ONE PLACEMENT (`engine_frame_for`).  A deck passes the
    # frame it records; a direct call gets the same answer computed here.
    fabricated = cell is None and struct.cell is None
    if frame is None:
        frame = engine_frame_for(struct, cell)

    lines: List[str] = ["# --- Geometry ---", ""]
    lines.append(f"NumberOfAtoms          {struct.n_atoms}")
    lines.append(f"NumberOfSpecies        {len(species)}")
    lines.append("")
    lines.append("%block ChemicalSpeciesLabel")
    for sp in species:
        # ``atomic_number`` raises rather than defaulting: a species with
        # no element cannot be calculated, and Z=0 in this block is a
        # ghost atom SIESTA would silently accept.
        z = atomic_number(sp)
        lines.append(f"  {species_idx[sp]:>3}  {z:>3}  {sp}")
    lines.append("%endblock ChemicalSpeciesLabel")
    lines.append("")
    lines.extend(_lattice_block(struct, frame.cell, fabricated=fabricated))
    lines.append("")
    lines.append("AtomicCoordinatesFormat        Ang")
    lines.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, frame.positions):
        lines.append(
            f"  {x:14.8f} {y:14.8f} {z:14.8f}  {species_idx[el]}"
        )
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    lines.append("")
    return lines


def emit_electrode_declarations(struct: Structure, cfg) -> List[str]:
    """The junction as TranSIESTA and tbtrans read it -- which atoms are each
    lead and where its bulk Hamiltonian is, the two reservoirs and the bias
    between them, the buffer atoms (`engines/transport.md` § 6.1b).

    **Structure, not settings**, and that is why it is a block: every line is
    derived from the region labels (`struct.regions`), and no parameter models
    which atoms a lead is.  Both programs read it -- TranSIESTA for the device,
    and `tbtrans`, which falls back to these `TS.*` blocks when it is given no
    `TBT.*` ones -- so the device and the transmission decks carry the same
    text.  The settings beside it (the voltage, the bulk treatment, the
    contours, the transmission's own) are catalogue items written with their
    notes; this writes none of them.

    Modern (SIESTA 4.1+ / 5.x) syntax, verified against the 5.4.2 binary
    (audit SCI-B1, 2026-06-18): per-electrode ``%block TS.Elec.<name>`` with
    ``HS``, ``chem-pot``, ``used-atoms``, ``elec-pos``, ``bloch`` and
    ``semi-inf-direction``; chemical potentials in ``%block TS.ChemPots`` and
    one ``%block TS.ChemPot.<name>`` each.  The leads are the regions named
    ``L-electrode`` and ``R-electrode``, ordered by z-centroid: the LOWER block
    gets ``semi-inf-direction -A3`` and the first ``elec-pos``; the ``Left`` /
    ``Right`` chemical potential binds by the region's NAME, and the deck says
    which lead ends up at mu = +V/2.

    **Not rewritten, and deliberately so.**  TranSIESTA identifies each
    electrode by a CONTIGUOUS ATOM RANGE, so an off-by-one in a position line
    computes transmission through a region that is not the molecule, and
    converges while doing it.  This text has been measured against a live
    5.4.2 run (§ 6.1b: 27 / 27 atoms at 1-27 and 94-120, as written).
    """
    # TWO LEADS, ALWAYS: no other label is one (`sort.ELECTRODE_LABELS`),
    # and the sort refuses a junction missing either before any deck is
    # rendered (`sort.categorical_sort`).
    electrodes = _find_electrode_regions(struct)
    semi_inf = {0: "-A3", len(electrodes) - 1: "+A3"}
    # The chempot binds by the region's own NAME: L-electrode -> Left
    # (mu = +V/2), R-electrode -> Right.  On the usual convention
    # (L-electrode = low z) that is the same as binding by z-centroid, so
    # the deck reads `TS.Elec.L -> chem-pot Left` and `V = V_left -
    # V_right` means what the labels say; on a junction labeled the other
    # way round the two halves name different blocks, and the deck says so
    # below.
    chempot_for = {block_name: ("Left" if label == REGION_LEFT_ELECTRODE
                                else "Right")
                   for label, block_name, _idxs in electrodes}

    lines: List[str] = [
        "# --- The junction: its electrodes and reservoirs ---",
        "#",
        "# Read by BOTH programs: TranSIESTA solves the device with these",
        "# leads attached, and tbtrans -- given no TBT.* electrode blocks --",
        "# reads these same TS.* ones (SIESTA 5.4.2, Util/TS/TBtrans/",
        "# m_tbt_options.F90).  Every line is derived from the region labels:",
        "# the regions named L-electrode and R-electrode are the leads, and",
        "# each one's atoms are the contiguous range written below.",
        "%block TS.Elecs",
    ]
    for _label, block_name, _idxs in electrodes:
        lines.append(f"  {block_name}")
    lines.append("%endblock TS.Elecs")
    lines.append("")

    # Buffer atoms (engines/transport.md § 4, the `buffer` label): padding at
    # the OUTER ends, excluded from the NEGF region via TS.Atoms.Buffer.
    # With buffers present TranSIESTA's DEFAULT electrode placement
    # (first electrode = first atoms, last = last atoms) no longer
    # holds, so each electrode's position is then stated EXPLICITLY
    # (``elec-pos``) from its region's own indices.
    buffer_idx = sorted((struct.regions or {}).get(REGION_BUFFER, []))
    n_total = struct.n_atoms

    # Per-electrode blocks.
    for i, (label, block_name, idxs) in enumerate(electrodes):
        cp = chempot_for[block_name]
        sid = semi_inf.get(i, "+A3")
        lines.append(f"%block TS.Elec.{block_name}")
        lines.append(f"  HS                 "
                     f"{electrode_hs_stem(cfg.system_label, label)}.TSHS")
        lines.append(f"  chem-pot           {cp}")
        lines.append(f"  used-atoms         {len(idxs)}")
        # ALWAYS, because the manual lists it among the four lines a
        # `%block TS.Elec.<name>` MUST carry -- HS, semi-inf-dir,
        # electrode-pos, chem-pot (SIESTA 5.4.0 manual).
        #
        # `begin` / `end` are the binary's own tokens (it accepts
        # elec-pos | start | begin | end), which is more permissive than the
        # manual documents.
        if i == 0:
            # 1-based index of the electrode's first atom.
            lines.append(f"  elec-pos begin     {min(idxs) + 1}")
        else:
            # Counted from the end: -1 is the last atom, so the
            # electrode's last atom (0-based ``max``) sits at
            # -(n_total - max).
            lines.append(f"  elec-pos end       "
                         f"{-(n_total - max(idxs))}")
        # `bloch 1 1 1`, FIXED, not a control.  molbuilder derives
        # the electrode from the junction's own labelled atoms, so the
        # electrode cell IS the device cross-section and no expansion
        # applies.
        lines.append("  bloch              1 1 1")
        lines.append(f"  semi-inf-direction {sid}")
        lines.append(f"%endblock TS.Elec.{block_name}")
        lines.append("")

    # WHICH LEAD IS BIASED POSITIVE, said in the deck itself -- read
    # off what was actually emitted above, never from an assumed
    # convention.  The semi-infinite directions came from the GEOMETRY
    # (``electrodes`` is z-sorted) and mu comes from the NAME, so on a
    # junction labeled the other way round these two lines name
    # different blocks.
    low_label = electrodes[0][0]
    plus_label = next(lab for lab, name, _i in electrodes
                      if chempot_for[name] == "Left")
    lines.append(
        f"# {low_label} is the LOW-z lead: listed first, "
        f"semi-inf-direction -A3.")
    lines.append(
        f"# {plus_label} carries mu = +V/2 (chem-pot Left), so "
        f"V = V_left - V_right.")
    if plus_label != low_label:
        lines.append(
            "# NOTE: those are DIFFERENT blocks.  This junction "
            "is labeled with")
        lines.append(
            f"#   {plus_label} on the HIGH-z end, so the HIGH-z "
            f"lead is the positively")
        lines.append(
            "#   biased one -- the reverse of the usual "
            "convention, and intentional")
        lines.append(
            "#   unless the labels were swapped by mistake "
            "(engines/transport.md § 4).")
    lines.append(
        "# The two reservoirs.  `%block TS.ChemPots` names them; each is")
    lines.append(
        "# defined in its own %block TS.ChemPot.<name>.  V is TS.Voltage,")
    lines.append(
        "# written with its note beside this block: at zero bias the ±V/2")
    lines.append(
        "# split is inert; at a finite bias it sets the two Fermi levels.")
    lines.extend([
        "%block TS.ChemPots",
        "  Left",
        "  Right",
        "%endblock TS.ChemPots",
        "",
        "%block TS.ChemPot.Left",
        "  mu  V/2",
        "%endblock TS.ChemPot.Left",
        "",
        "%block TS.ChemPot.Right",
        "  mu -V/2",
        "%endblock TS.ChemPot.Right",
        "",
    ])

    if buffer_idx:
        # The sorted layout puts buffers OUTERMOST ([buf][L][bridge]
        # [R][buf]), so the indices compress to at most two ranges;
        # emitted generically all the same.
        lines.append("# Buffer atoms: padding outside the electrode "
                     "blocks, excluded from")
        lines.append("# the NEGF region entirely "
                     "(engines/transport.md § 4, the `buffer` label).")
        lines.append("%block TS.Atoms.Buffer")
        run_start = prev = buffer_idx[0]
        for j in buffer_idx[1:] + [None]:
            if j is None or j != prev + 1:
                lines.append(f"  atom [ {run_start + 1} -- {prev + 1} ]")
                run_start = j
            prev = j if j is not None else prev
        lines.append("%endblock TS.Atoms.Buffer")
        lines.append("")
    return lines
