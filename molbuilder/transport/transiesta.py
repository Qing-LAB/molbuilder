"""The TranSIESTA emitters — the transport composite's deck layer.

**THIS MODULE NO LONGER WRITES A DECK OF ITS OWN** (2026-09-17).  It is a
set of emitters the two live writers reach into, plus the engine preflight:

* :func:`emit_electrode_declarations` and :func:`_emit_geometry` — reused by
  `transport/deck.py`, the pipeline every one of the five rungs renders
  through.  *(The first was `_emit_transiesta_block` until 2026-09-29, which
  also wrote every TranSIESTA and TBtrans VALUE as an f-string; those are
  catalogue items now, written by the section walk with their notes —
  `engines/transport.md` § 6.1b.)*
* :func:`_compute_cell_from_extents` and
  :func:`_find_electrode_regions` — reused by `transport/wizard.py`, whose
  `extract_electrode_model` derives the lead `compose.py` hands to prep.
* :func:`electrode_hs_stem` — the ONE spelling of an electrode run's
  identity, so the device deck's ``HS`` line and the electrode deck's
  ``SystemLabel`` cannot disagree.
This module emits NO deck of its own and holds NO gate.  ``TransiestaEngine``
and its ``preflight`` were deleted 2026-09-17 (the tombstone at the foot of the
file lists every check and where it lives now), as were the ``TransportEngine``
Protocol and its registry before them.  Transport's science is the KIND's —
``_KIND_VALIDATORS["transport"]`` — and its decks are written by
``transport/deck.py`` through ``spec_for`` → ``prepare_deck``.

``render_script``, ``_emit_header`` and ``_emit_k_mesh`` were a SECOND
device-deck writer and are deleted; so are ``parse_output`` (which raised
``NotImplementedError``) and ``methods_fragment`` (a placeholder paragraph),
neither of which anything called.  The tombstones below say why each went.
The record a future transmission inspector will read already exists —
`transport/record.py` writes ``<label>.transport.json`` — and needs no stub
here to wait for it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config.transport import (
    ELECTRODE_LABEL_SUFFIX,
    REGION_BUFFER,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    is_electrode_label,
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
#: sidecar) and this emitter ignored it until 2026-09-23.
_ISOLATED_ELECTRODE_VACUUM_ANG = 15.0


def _sanitize_electrode_block_name(label: str) -> str:
    """Convert a user-facing region label to a SIESTA block-safe name.

    Examples:
        "L-electrode"   → "L"
        "R-electrode"   → "R"
        "tip-electrode" → "tip"
        "tip_electrode" → "tip"
        "electrode"     → "electrode"   (no prefix — keep verbatim)

    The SIESTA fdf parser is forgiving about block names; we only
    need to strip the convention's suffix so the per-electrode
    blocks (``%block TS.Elec.<name>``) don't carry the redundant
    "-electrode" tag.  Empty or pure-suffix labels are returned
    unchanged so the user sees their input echoed in errors.
    """
    if not is_electrode_label(label):
        return label
    stem = label
    for sep in ("-", "_"):
        candidate = sep + ELECTRODE_LABEL_SUFFIX
        if stem.lower().endswith(candidate):
            return stem[: -len(candidate)] or label
    # ends with "electrode" directly (no separator) — keep verbatim
    return label


def electrode_hs_stem(job_name: str, label: str) -> str:
    """The ONE spelling of an electrode run's identity — its SystemLabel,
    and therefore the stem of the ``.TSHS`` the device deck references.

    Two writers need it to agree byte-for-byte: the device deck's
    ``TS.Elec.<name> HS`` line (below) and the transport ladder's
    electrode-stage renderer (`transport/stages.py`), which sets the
    electrode deck's ``SystemLabel`` so SIESTA writes exactly the file
    the device will ask for.
    """
    return f"{job_name}_{label}"


def _find_electrode_regions(
    struct: Structure,
) -> List[Tuple[str, str, List[int]]]:
    """Return (user_label, block_name, indices) for every region
    whose label ends with the electrode-suffix convention.

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
    for label, indices in (struct.regions or {}).items():
        if not is_electrode_label(label):
            continue
        if not indices:
            continue
        block_name = _sanitize_electrode_block_name(label)
        z_centroid = float(np.mean(struct.positions[indices, 2]))
        out.append((label, block_name, list(indices), z_centroid))
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

    THE VACUUM IS THE PERSON'S, and it was ignored until 2026-09-23.  Three
    states (`structure-periodicity.md` § 2): a number used verbatim,
    ``[0,0,0]`` meaning no gap DELIBERATELY and also used, and unset -- the
    only one :data:`_ISOLATED_ELECTRODE_VACUUM_ANG` answers.  Someone who
    typed 8 A on the Cell page got 15 A in the deck and nothing said so.

    **Why not just call `resolve_cell`?**  Because its default where nobody
    chose is 3 A -- right for a molecule in a box and five times too thin
    for a lead, which is the object the self-energy is built from and whose
    images must be electrostatically isolated.  The RULE is shared; only the
    default differs, the way the electrode's dense transport-axis k is a
    transport default rather than something to discover (§ 2a.7).

    *(This padded 15 A onto both transverse axes whatever their kind and
    answered the third with ``int(bbox_z + 2) + 1`` -- a rounding that
    honoured nothing and had no source.)*
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
#  .fdf emission helpers — kept as small private functions so each      #
#  block is independently testable.  Each returns a list of lines;      #
#  ``render_script`` concatenates them.                                 #
# --------------------------------------------------------------------- #


# `_emit_header` DELETED 2026-09-17 with `render_script`, its only caller.
# The live deck gets its banner, its SystemLabel and the `# runtime.<key>:`
# echo lines from `script_emit`, shared with the SIESTA emitter -- which is
# what made one reader (`molwatch.parse_runtime_line`) serve both engines in
# the first place.


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

    *(Two 2026-09-22 reviews read the § 7 sentence as a verdict on this
    function and concluded it was residue -- one attempted a deletion,
    which was reverted.  The sentence is about the OTHER case.)*
    """
    lines = ["LatticeConstant        1.0 Ang"]
    cell = np.asarray(cell, dtype=float)
    if not fabricated:
        # THE KIND, NOT THE BOOLEAN.  This read `struct.pbc`, which flattens
        # `transport` and `periodic` to the same True -- so the per-axis line
        # below labelled every transport axis "periodic", and the warning
        # further down ("the transport axis has vacuum / is not periodic")
        # could never fire on a transport axis at all.  `axis_kind` is the
        # field that distinguishes them, and it is what this was asking for.
        # The fallback AGREES WITH ITS TEN NEIGHBOURS.  It arrived as the
        # literal swap for the old `struct.pbc or (True,)*3` field read and
        # said `("periodic",) * 3`, while every other `axis_kind or ...` in
        # the tree -- including `Structure.pbc()` itself -- answers
        # `("isolated",) * 3`.  All eleven are unreachable (`__post_init__`
        # always fills the kinds), but this is the one whose waking would
        # relabel every axis in an emitted deck and silence the warning
        # below, so it is the one that must not disagree.
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
    #
    # This sorted ALPHABETICALLY until 2026-09-23 while `siesta/input.py`
    # sorted by atomic number, so one structure was declared `Au, C, H, S`
    # here and `H, C, S, Au` there.  And `cfg.species_order` -- a catalogue
    # row a person can fill in -- was honoured there and unreachable here,
    # because this block took no `cfg`.  **The config was already in scope
    # at the call site and simply not passed** (`deck.py`), so the "seam
    # change" that gap looked like was one argument.
    from ..chemistry import species_order as _species_order
    species = _species_order(
        struct.elements,
        getattr(cfg, "species_order", None) if cfg is not None else None)
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}
    # AN OVERRIDE THAT OMITS A SPECIES REFUSES HERE, and it used to crash.
    # `species_order` is honoured verbatim, so a list missing an element the
    # structure contains left `species_idx[el]` to raise a bare
    # `KeyError('Au')` while writing the coordinate block.  The SIESTA
    # emitter has always refused this in words; transport could not reach it
    # at all until `cfg` began arriving here on 2026-09-23, so the exposure
    # is as new as the control.
    missing = sorted(set(struct.elements) - set(species_idx))
    if missing:
        raise ValueError(
            f"species_order does not list {', '.join(missing)}, which this "
            f"structure contains.  The order is honoured verbatim, so every "
            f"species has to appear in it -- or leave it unset and the "
            f"default rule orders them (model/chemistry.md 3a)")

    # THE BOX, THEN ONE PLACEMENT (`engine_frame_for`).  A deck passes the
    # frame it records; a direct call gets the same answer computed here.  The
    # fabricated box used to take the atoms untranslated: a box at the origin
    # around coordinates that were not.
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
        # ghost atom SIESTA would silently accept (fixed 2026-09-09).
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


# `_emit_basis_and_xc` DELETED 2026-09-18 -- ZERO call sites, and it could
# never gain one.  `deck.py`'s header states why: "its six keywords are
# exactly `BASIS_SECTION` + `XC_SECTION` + `electronic_temperature`, and
# lifting it beside them would write each twice -- which `layout.check_rules`
# now catches."  The composite takes basis/XC from the catalogue sections.
#
# It outlived its caller (`render_electrode_fdf`, deleted 2026-09-17) by a
# day in code and longer in prose: `wizard.py` imported it without calling
# it, this module's header called it "reused by transport/wizard.py", the
# tombstone below listed it among symbols with "real callers", and
# `plan.md` listed it under "NOT redundant, so not deleted by association".
# Four statements, one dead function.

# `_emit_k_mesh` DELETED 2026-09-17 with `render_script`, its only caller.
# The live deck writes the transverse grid through `deck.py`'s own
# `_emit_kgrid_block`, which resolves it from the catalogue row rather than
# off `TransportConfig.k_mesh_transverse` -- `deck.py`'s header already
# recorded that the two restated each other.


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
    one ``%block TS.ChemPot.<name>`` each.  Electrodes are the regions whose
    label ends in ``-electrode``, ordered by z-centroid: the LOWER block gets
    ``semi-inf-direction -A3`` and the first ``elec-pos``; the ``Left`` /
    ``Right`` chemical potential binds by the region's NAME, and the deck says
    which lead ends up at mu = +V/2.

    **Not rewritten, and deliberately so.**  TranSIESTA identifies each
    electrode by a CONTIGUOUS ATOM RANGE, so an off-by-one in a position line
    computes transmission through a region that is not the molecule, and
    converges while doing it.  This text has been measured against a live
    5.4.2 run (§ 6.1b: 27 / 27 atoms at 1-27 and 94-120, as written).
    """
    electrodes = _find_electrode_regions(struct)
    # Canonical 2-terminal naming: the z-min electrode binds to the
    # ``Left`` chempot (mu = +V/2); z-max binds to ``Right`` (mu = -V/2).
    # For arbitrary multi-electrode runs the chempot binding is the
    # electrode's own name (and the user must set mu per chempot via
    # a future form field; today the multi-terminal path raises a
    # preflight notice).
    is_two_terminal = len(electrodes) == 2
    semi_inf = {0: "-A3", len(electrodes) - 1: "+A3"}
    chempot_for = {}
    labels = [lab for lab, _n, _i in electrodes]
    canonical = (sorted(labels) == sorted([REGION_LEFT_ELECTRODE,
                                           REGION_RIGHT_ELECTRODE]))
    if is_two_terminal and canonical:
        # Bind the chempot by the region's own NAME.  Under the one
        # convention (L-electrode = low z; sort.py refuses anything
        # else) this is identical to binding by z-centroid -- so the deck
        # reads `TS.Elec.L -> chem-pot Left` with no inversion possible,
        # and `V = V_left - V_right` means what the labels say.  Naming it
        # keeps the deck honest if the gate ever moves.
        for label, block_name, _idxs in electrodes:
            chempot_for[block_name] = (
                "Left" if label == REGION_LEFT_ELECTRODE else "Right")
    elif is_two_terminal:
        # Two leads under non-canonical labels: nothing says which
        # reservoir is which, so follow the CONVENTION -- Left is the
        # first electrode, the -A3 (low-z) end (legacy <=4.0
        # TS.NumUsedAtomsLeft: "the first N atoms").
        chempot_for[electrodes[0][1]] = "Left"
        chempot_for[electrodes[1][1]] = "Right"
    else:
        # Fallback: each electrode binds to its own chempot.  Same
        # name; the user customises bias per chempot when the
        # multi-terminal UI lands.
        for _label, block_name, _idxs in electrodes:
            chempot_for[block_name] = block_name

    lines: List[str] = [
        "# --- The junction: its electrodes and reservoirs ---",
        "#",
        "# Read by BOTH programs: TranSIESTA solves the device with these",
        "# leads attached, and tbtrans -- given no TBT.* electrode blocks --",
        "# reads these same TS.* ones (SIESTA 5.4.2, Util/TS/TBtrans/",
        "# m_tbt_options.F90).  Every line is derived from the region labels:",
        "# a region named `*-electrode` is a lead, and its atoms are the",
        "# contiguous range written below.",
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
        # This sat inside `if buffer_idx:` until 2026-09-15, so an ORDINARY
        # junction -- no buffer atoms -- got two electrode blocks without
        # it.  Probably harmless, and that is the problem: molbuilder sorts
        # the junction so the electrodes ARE the first and last atoms, which
        # is where an omitted position would land anyway, so the deck relied
        # on an undocumented default agreeing with the truth.
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
        # `bloch 1 1 1`, FIXED, not a control (the field was withdrawn
        # 2026-09-15 -- see `config/transport.py`).  molbuilder derives
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
    if is_two_terminal:
        low_label = electrodes[0][0]
        plus_label = next((lab for lab, name, _i in electrodes
                           if chempot_for[name] == "Left"), None)
        lines.append(
            f"# {low_label} is the LOW-z lead: listed first, "
            f"semi-inf-direction -A3.")
        if plus_label is not None:
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
    lines.append("%block TS.ChemPots")
    if is_two_terminal:
        lines.append("  Left")
        lines.append("  Right")
    else:
        for _label, block_name, _idxs in electrodes:
            lines.append(f"  {block_name}")
    lines.append("%endblock TS.ChemPots")
    lines.append("")

    if is_two_terminal:
        lines.extend([
            "%block TS.ChemPot.Left",
            "  mu  V/2",
            "%endblock TS.ChemPot.Left",
            "",
            "%block TS.ChemPot.Right",
            "  mu -V/2",
            "%endblock TS.ChemPot.Right",
            "",
        ])
    else:
        # Multi-terminal placeholder: equal-spaced chempots.  Users
        # must override via the form's per-chempot mu when the
        # multi-terminal scope ships.
        for _label, block_name, _idxs in electrodes:
            lines.extend([
                f"%block TS.ChemPot.{block_name}",
                "  mu  0.0  # multi-terminal placeholder — set "
                "explicitly per chempot",
                f"%endblock TS.ChemPot.{block_name}",
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


# --------------------------------------------------------------------- #
#  The engine                                                            #
# --------------------------------------------------------------------- #


# `TransiestaEngine` DELETED 2026-09-17 -- the last of the June 2026 era.
#
# The class ended as one classmethod, `preflight`, reached only through
# `_ENGINE_VALIDATORS[TransportConfig]` in `validation/__init__.py`.  **Nothing
# validates a `TransportConfig`.**  Every rung resolves a `SiestaConfig`
# (`engines/transport.md` 2a.14), and the two sites that built a
# `TransportConfig` then -- `deck.py::_legacy_view` and
# `stages.py::config_for` -- built it as a projection to feed the lifted
# NEGF block emitter and never validated it.  So the registration
# dispatched for nothing.  (`_legacy_view` went on 2026-09-29, when the
# block's values moved to the catalogue; `config_for` has no production
# caller.)
#
# It was kept after that became true because one surface still built a
# `TransportConfig` and validated it: `POST /api/transport/render`.  That route
# was deleted 2026-09-17, and with it the last reason.
#
# EVERY CHECK IT CARRIED HAS A NAMED LIVE HOLDER, verified one by one before
# this deletion (`engines/transport.md` 5 names them, and none named this):
#
#   device kz = 1 ................. `_validate_transport_kind`, error on
#                                   `config.kgrid` (I8)
#   unknown region labels ......... `validation/sidecar.py::
#                                   check_unconsumed_region_labels`, run by
#                                   `_validate_siesta` -- which every rung
#                                   reaches, because every rung IS a SiestaConfig
#   missing / empty L, bridge, R .. `sort.py::categorical_sort` REFUSES
#   unlabeled or double-labeled ... `sort.py::_partition_of` REFUSES
#   interleaved electrodes ........ `sort.py::categorical_sort` REFUSES
#   atom order [lower][bridge][upper]
#                                   held by CONSTRUCTION -- `compose` runs the
#                                   categorical sort before any deck is
#                                   rendered, and the extracted lead inherits
#                                   that order
#   |V| > 2 V advisory ............ `_validate_transport_kind`, warn on
#                                   `config.bias_voltage_v` (re-homed 2026-09-16)
#
# What the module still exports is the emission library the live deck path
# reuses.  MEASURED 2026-09-18, because this list used to assert callers that
# do not exist:
#
#   `_emit_geometry` ............. `deck.py` (all four deck shapes, so every
#                                  one of the five rungs)
#   `emit_electrode_declarations`  `deck.py`, the device and transmission
#                                  shapes (`_emit_transiesta_block` until
#                                  2026-09-29, when its values left it)
#   `electrode_hs_stem` .......... `stages.py` x2 and `jobset/prep.py` -- NOT
#                                  `deck.py`/`wizard.py`, which the old list
#                                  claimed
#   `_find_electrode_regions` .... `wizard.py`, plus internal
#   `_compute_cell_from_extents` . `wizard.py`, plus internal
#   `axis_vacuum`, `_lattice_block` . INTERNAL ONLY -- reached through
#                                  `_emit_geometry`, never imported out
#
# (`_emit_basis_and_xc` was in this list too, with zero callers anywhere.
# Deleted 2026-09-18; see the note where it stood.)
