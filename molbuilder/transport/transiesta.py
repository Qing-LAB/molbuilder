"""The TranSIESTA emitters — the transport composite's deck layer.

**THIS MODULE NO LONGER WRITES A DECK OF ITS OWN** (2026-09-17).  It is a
set of emitters the two live writers reach into, plus the engine preflight:

* :func:`_emit_transiesta_block` and :func:`_emit_geometry` — reused by
  `transport/deck.py`, the pipeline every one of the five rungs renders
  through.
* :func:`_emit_basis_and_xc`, :func:`_compute_cell_from_extents` and
  :func:`_find_electrode_regions` — reused by `transport/wizard.py`, whose
  `extract_electrode_model` derives the lead `compose.py` hands to prep.
* :func:`electrode_hs_stem` — the ONE spelling of an electrode run's
  identity, so the device deck's ``HS`` line and the electrode deck's
  ``SystemLabel`` cannot disagree.
* :meth:`TransiestaEngine.preflight` — region/order/kz gates, called
  directly by `validation/__init__.py`.  There was a `TransportEngine`
  Protocol and a registry in front of it until 2026-09-17: four declared
  members of which one survived, one registered engine, and one caller, so
  the lookup could only ever return the class below.

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
    EXPECTED_REGIONS_2T,
    REGION_BRIDGE,
    REGION_BUFFER,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    TransportConfig,
    is_electrode_label,
)
from ..issues import Issue
from ..structure import Structure


# Transverse vacuum padding per side (Å) when computing the device
# cell from atom extents.  15 Å is the SIESTA-canonical floor for
# isolated-molecule transverse padding (electronic tails fall off
# by ~5–8 Å in vacuum).  Lower → exchange-correlation tails leak;
# higher → unnecessary basis cost.
_TRANSVERSE_PAD_ANG = 15.0


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


#: `log_level`'s three words as the integer `TBT.Verbosity` takes.
#:
#: **Sourced, not invented** (TBtrans reference, checked 2026-09-15):
#: `TBT.Verbosity` is an integer in 0-10, default 5, "for smaller numbers
#: less information will be printed".  The field claimed `WriteVerbosity`
#: until then -- zero occurrences in the 5.4.2 binary -- and a mapping was
#: NOT invented for it in the same pass that retired
#: `transmission_relative_to_ef` precisely for want of a sourced one
#: (`plan.md` § 5o).  `info` is the engine's own default.
_TBT_VERBOSITY = {"warning": 2, "info": 5, "debug": 8}


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
    """Default cell (a, b, c) in Å from the atom extents.

    Transverse (a, b): atom-extent + ``2 × _TRANSVERSE_PAD_ANG``.
    Transport (c): atom-extent of the z-coordinates ROUNDED UP to
    the nearest Å.  The user is expected to OVERRIDE c so it matches
    their electrode z-periodicity (the comment in the .fdf says so);
    auto-computing it is a starting point, not a defensible final
    value.  For a 2-Å-buffer device, the rounding adds a few Å of
    slack so the auto-default doesn't accidentally clip atoms at
    the boundary.
    """
    pos = struct.positions
    extent_x = float(pos[:, 0].max() - pos[:, 0].min())
    extent_y = float(pos[:, 1].max() - pos[:, 1].min())
    extent_z = float(pos[:, 2].max() - pos[:, 2].min())
    a = extent_x + 2.0 * _TRANSVERSE_PAD_ANG
    b = extent_y + 2.0 * _TRANSVERSE_PAD_ANG
    c = float(int(extent_z + 2.0) + 1)  # round up + 2 Å buffer
    return (a, b, c)


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


def _lattice_block(struct: Structure,
                   cell: Optional[np.ndarray]) -> List[str]:
    """Emit the LatticeVectors block.

    If ``cell`` is provided (the structure's real lattice — hexagonal,
    triclinic, whatever), it is emitted VERBATIM and the per-axis
    vacuum is reported with a warning when an axis declared periodic
    leaves large empty space or the transport axis (c) has vacuum.

    If ``cell`` is None there is no lattice to preserve, so an
    orthorhombic vacuum box is fabricated from atom extents — a model
    of an ISOLATED cluster, flagged loudly because it is wrong for a
    periodic surface electrode (the hex Au(111) case).
    """
    lines = ["LatticeConstant        1.0 Ang"]
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        pbc = struct.pbc or (True, True, True)
        vac = axis_vacuum(cell, struct.positions)
        lines += [
            "# Explicit lattice preserved from the structure (NOT",
            "# recomputed from atom extents).  Per-axis boundary:",
        ]
        names = ("a", "b", "c (transport)")
        for ax in range(3):
            kind = "periodic" if pbc[ax] else "vacuum"
            lines.append(
                f"#   {names[ax]:<14} {kind:<8} | empty span "
                f"{vac[ax]:.2f} Å")
        # Transport axis (c) must be periodic / seamless for the leads.
        if vac[2] > _VACUUM_FLAG_ANG or not pbc[2]:
            lines.append(
                "# WARNING: the transport axis (c) has vacuum / is not "
                "periodic;")
            lines.append(
                "#   the electrode .TSHS cannot attach seamlessly "
                "(Brandbyge 2002 § III).")
        for ax in (0, 1):
            if pbc[ax] and vac[ax] > _VACUUM_FLAG_ANG:
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
        a, b, c = _compute_cell_from_extents(struct)
        lines += [
            "# WARNING: no lattice on the structure — an orthorhombic",
            "# VACUUM BOX was fabricated from atom extents (a,b = extent",
            "# + 30 Å padding; c = extent + 2 Å).  This models an",
            "# ISOLATED CLUSTER, NOT a periodic surface electrode.  For a",
            "# real Au(111) lead, supply the structure's hexagonal cell",
            "# (set Structure.cell / the molstruct sidecar's 'cell').",
            "%block LatticeVectors",
            f"  {a:7.3f}    0.000    0.000",
            f"    0.000  {b:7.3f}    0.000",
            f"    0.000    0.000  {c:7.3f}",
            "%endblock LatticeVectors",
        ]
    return lines


def _emit_geometry(struct: Structure,
                   cell: Optional[np.ndarray] = None) -> List[str]:
    """Lattice + AtomicCoordinates blocks.

    The lattice comes from ``cell`` (or ``struct.cell``) when present —
    preserved verbatim so a hexagonal Au(111) surface keeps its real
    cell.  Only when no lattice is available does it fall back to the
    orthorhombic vacuum box (an isolated-cluster model), with a loud
    warning.

    Coordinates are emitted in the ENGINE frame (structure-periodicity.md
    § 6.1 clause 5): SIESTA anchors the cell at (0,0,0), so atoms are
    shifted by ``-resolve_cell_origin()`` — the SAME convention
    ``render_fdf`` applies.  Emitting the cell at zero with world-frame
    coordinates mistranslated a junction by its origin (review finding
    2026-07-29: far-face atoms wrapped into the leads).
    """
    from ..chemistry import atomic_number
    species = sorted(set(struct.elements), key=lambda e: e.capitalize())
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}

    resolved_cell = cell if cell is not None else struct.cell
    origin = (struct.resolve_cell_origin()
              if resolved_cell is not None else None)
    positions = (struct.positions - np.asarray(origin, dtype=float)
                 if origin is not None else struct.positions)

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
    lines.extend(_lattice_block(struct, resolved_cell))
    lines.append("")
    lines.append("AtomicCoordinatesFormat        Ang")
    lines.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, positions):
        lines.append(
            f"  {x:14.8f} {y:14.8f} {z:14.8f}  {species_idx[el]}"
        )
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    lines.append("")
    return lines


def _emit_basis_and_xc(cfg: TransportConfig) -> List[str]:
    """Basis set + XC + mesh cutoff + electronic temperature.

    Every value comes from ``cfg`` — the ONE object the electrode,
    seed and device decks all render from, which is what makes the
    transport ladder's electronic contract identical by construction
    (transport-design.md § 3).  Basis / XC / energy shift were
    hard-coded here (DZP / PBE / 0.01 Ry) until 2026-08-28; the
    composite fills the fields from the cited junction's own .fdf.
    """
    return [
        "# --- Basis + XC ---",
        "",
        f"PAO.BasisSize          {cfg.basis_size}",
        f"PAO.EnergyShift        {cfg.energy_shift_ry:g} Ry",
        f"XC.functional          {cfg.xc_functional}",
        f"XC.authors             {cfg.xc_authors}",
        f"MeshCutoff             {cfg.siesta_mesh_cutoff_ry} Ry",
        f"ElectronicTemperature  {cfg.electronic_temperature_k:.1f} K",
        "",
    ]


# `_emit_k_mesh` DELETED 2026-09-17 with `render_script`, its only caller.
# The live deck writes the transverse grid through `deck.py`'s own
# `_emit_kgrid_block`, which resolves it from the catalogue row rather than
# off `TransportConfig.k_mesh_transverse` -- `deck.py`'s header already
# recorded that the two restated each other.


def _emit_transiesta_block(struct: Structure,
                            cfg: TransportConfig) -> List[str]:
    """The TS.* block — modern (SIESTA 4.1+ / 5.x) NEGF syntax.

    2026-06-18 modernization (audit SCI-B1, verified against
    SIESTA 5.4.2 binary):

    * Per-electrode blocks ``%block TS.Elec.<name>`` carry the
      electrode metadata (``HS``, ``chem-pot``, ``used-atoms``,
      ``bloch``, ``semi-inf-direction``) instead of the legacy
      ``TS.HSFile<Left|Right>`` + ``TS.NumUsedAtoms<Left|Right>``
      flat keys.
    * Chemical potentials are declared in ``%block TS.ChemPots``
      and configured per-name in ``%block TS.ChemPot.<name>``;
      the implicit ``±V/2`` of the legacy form is gone, the
      bias is explicit per chempot.
    * Electrodes are discovered from ``struct.regions`` by the
      ``*-electrode`` label convention (any region whose label
      ends with ``-electrode`` becomes an electrode block);
      ``L-electrode`` / ``R-electrode`` (the defaults) fit
      naturally.  Order is by z-centroid, so the LOWER block gets
      ``semi-inf-direction -A3`` and the first ``elec-pos``; the
      ``Left``/``Right`` chempot binding is by region NAME, and the
      deck states which lead ends up at µ = +V/2.

    For the canonical 2-terminal case (the only fully-validated
    scope today), the emitter produces exactly the verified
    Au-BDT-Au template.  Multi-terminal (3+ electrodes) is a
    planned follow-up; today such a structure emits a single
    notice in render_script's pre-emit pass.

    **The TBtrans half DID migrate, and this docstring said it had
    not** -- "its keyword names didn't migrate in 4.1+", which is the
    false belief that kept four SIESTA-3.x scalars in this emitter
    until 2026-09-15.  They are a `%block TBT.Contour` now; the block
    below carries the measurement (`plan.md` § 5o).
    """
    bias = cfg.bias_voltages_v[0] if cfg.bias_voltages_v else 0.0

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
        # else, and the preflight below repeats it for structures that
        # never went through prep) this is identical to binding by
        # z-centroid -- so the deck reads `TS.Elec.L -> chem-pot Left`
        # with no inversion possible, and `V = V_left - V_right` means
        # what the labels say.  Naming it keeps the deck honest if the
        # gate ever moves.
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
        "# --- TranSIESTA NEGF (modern syntax, SIESTA 4.1+ / 5.x) ---",
        "",
        "# ``SolutionMethod transiesta`` switches the SCF cycle to NEGF.",
        "# (``TS.SolutionMethod`` exists as a separate keyword for the",
        "# NEGF inversion algorithm, NOT the engine selector — emitting",
        "# ``TS.SolutionMethod transiesta`` triggers 'Unrecognized "
        "TranSiesta",
        "# solution method' in SIESTA 5.4.2.  Empirically verified "
        "2026-06-18.)",
        "SolutionMethod         transiesta",
        "",
        "# Start the NEGF SCF from a saved density when one is present:",
        "# the transport ladder's seed stage leaves <SystemLabel>.DM",
        "# beside this deck (transport-design.md 4.2; SIESTA's default",
        "# for this keyword is false, so without it the seed would sit",
        "# unread -- 'present but not honoured').  With no file, SIESTA",
        "# initialises from atomic densities as usual.  A .TSDE needs no",
        "# keyword: TranSIESTA reads it by presence.",
        "DM.UseSaveDM           true",
        "",
        "# Electrode declarations.  Each electrode is a region in the",
        "# input structure whose label ends with ``-electrode``; the",
        "# emitter discovers them from ``struct.regions`` and emits one",
        "# %block TS.Elec.<name> per side.  See "
        "docs/engines/transport.md.",
        "%block TS.Elecs",
    ]
    for _label, block_name, _idxs in electrodes:
        lines.append(f"  {block_name}")
    lines.append("%endblock TS.Elecs")
    lines.append("")

    # Buffer atoms (transport-design.md § 3, last bullet): padding at
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
                     f"{electrode_hs_stem(cfg.job_name, label)}.TSHS")
        lines.append(f"  chem-pot           {cp}")
        lines.append(f"  used-atoms         {len(idxs)}")
        # ALWAYS, because the manual lists it among the four lines a
        # `%block TS.Elec.<name>` MUST carry -- HS, semi-inf-dir,
        # electrode-pos, chem-pot (SIESTA 5.4.0 manual;
        # `engines/transport.md` 3.3).
        #
        # This sat inside `if buffer_idx:` until 2026-09-15, so an ORDINARY
        # junction -- no buffer atoms -- got two electrode blocks without
        # it.  Probably harmless, and that is the problem: molbuilder sorts
        # the junction so the electrodes ARE the first and last atoms, which
        # is where an omitted position would land anyway, so the deck relied
        # on an undocumented default agreeing with the truth.  It stops
        # agreeing the moment a junction is not sorted that way or a third
        # electrode appears -- and the indices are right here.
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
        # applies; and nothing here adjusts the electrode's transverse
        # grid to match one, which is what made any other value a
        # silently mismatched lead.
        lines.append("  bloch              1 1 1")
        lines.append(f"  semi-inf-direction {sid}")
        lines.append(f"%endblock TS.Elec.{block_name}")
        lines.append("")

    # WHICH LEAD IS BIASED POSITIVE, said in the deck itself -- read
    # off what was actually emitted above, never from an assumed
    # convention.  The semi-infinite directions came from the GEOMETRY
    # (``electrodes`` is z-sorted) and mu comes from the NAME, so on a
    # junction labeled the other way round these two lines name
    # different blocks.  That disagreement is the fact a reader most
    # needs and is the one the deck used to hide.
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
                    "(transport-design.md 4.1a).")
    lines.append(
        "# Chemical potentials.  ``%block TS.ChemPots`` lists the names; "
        "each is")
    lines.append(
        "# defined in its own %block TS.ChemPot.<name>.  At zero bias the")
    lines.append(
        "# ±V/2 split is conventional and inert; at finite bias it sets the")
    lines.append("# left- vs right-Fermi-level offset.")
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
                f"  mu  0.0  # multi-terminal placeholder — set "
                f"explicitly per chempot",
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
                     "(transport-design.md § 3).")
        lines.append("%block TS.Atoms.Buffer")
        run_start = prev = buffer_idx[0]
        for j in buffer_idx[1:] + [None]:
            if j is None or j != prev + 1:
                lines.append(f"  atom [ {run_start + 1} -- {prev + 1} ]")
                run_start = j
            prev = j if j is not None else prev
        lines.append("%endblock TS.Atoms.Buffer")
        lines.append("")

    lines.extend([
        "# Bias voltage (used by the ``V`` substitution in the chempot "
        "mu lines).",
        "# Single value per .fdf today; the bias-scan workflow emits one "
        ".fdf per bias.",
        f"TS.Voltage             {bias:.4f} eV",
        "",
        "# THE NEGF DENSITY CONTOUR, and the lead treatment.",
        "# Defaults are the SIESTA 5.4.0 manual's; a 0 above means \"leave",
        "# it to the engine\" for the two whose default is a FORMULA rather",
        "# than a number, and nothing is emitted for those",
        "# (`engines/transport.md` 3.3 -- the template shape).",
        f"TS.Elecs.Bulk          "
        f"{'true' if cfg.elecs_bulk else 'false'}",
    ] + ([f"TS.Contours.Eq.Pole    {cfg.negf_eq_pole_ev:.4f} eV"]
         if cfg.negf_eq_pole_ev > 0 else []) + (
        [f"TS.Contours.nEq.Eta    {cfg.negf_neq_eta_ev:.6f} eV"]
         if cfg.negf_neq_eta_ev > 0 else []) + [
        "",
        "# TBtrans transmission post-processing.",
        "# Brandbyge et al., Phys. Rev. B 65, 165401 (2002) § IV.",
        "#",
        "# A CONTOUR BLOCK, NOT SCALARS.  This emitted four `TS.TBT.*`",
        "# scalars until 2026-09-15 -- named in `plan.md` § 5o, and NOT",
        "# spelled here on purpose: two tests assert a keyword by plain",
        "# substring over the whole deck, so naming the dead tokens in a",
        "# COMMENT made both pass on this prose (caught 2026-09-15, hours",
        "# after writing it).  A comment must not be able to satisfy an",
        "# assertion.  They were SIESTA-3.x spellings that the 5.4.2 tbtrans",
        "# this project installs cannot read: `strings` on the binary finds",
        "# zero occurrences of their stems in any spelling or prefix.",
        "# fdf ignores a label nobody queries, so the",
        "# run completed and T(E) came out on tbtrans's DEFAULT energy grid",
        "# while the form said otherwise -- a wrong answer that looks right",
        "# (`plan.md` § 5o).",
        "#",
        "# The modern mechanism is `TBT.Contours` naming one or more blocks.",
        "# `part line` is not a choice: tbtrans refuses anything else with",
        "# \"Unrecognized contour type for tbtrans, MUST be a line part\" --",
        "# its own string, which is where this grammar was read from.",
        "%block TBT.Contours",
        "  window",
        "%endblock TBT.Contours",
        "",
        "%block TBT.Contour.window",
        "  part line",
        f"   from {cfg.transmission_emin_ev:.5f} eV "
        f"to {cfg.transmission_emax_ev:.5f} eV",
        f"    points {cfg.transmission_n_points}",
        "     method mid-rule",
        "%endblock TBT.Contour.window",
        "",
        "# THE REST OF TBTRANS'S OWN SURFACE (`engines/transport.md` 3.3).",
        "#",
        "# `TBT.k` IS THE ONE THAT MATTERS.  tbtrans inherits the SCF's",
        "# kgrid_Monkhorst_Pack, and a grid converged for a total energy is",
        "# routinely far too coarse for transmission -- T(E) is an integral",
        "# over the transverse Brillouin zone.  0 0 0 in the form means",
        "# inherit, which is the old behaviour, so nothing is written.",
        "#",
        "# Every output below defaults to false in tbtrans, so a run wrote",
        "# transmission and NOTHING else until 2026-09-15 -- which is why",
        "# the Results transmission inspector had no DOS or eigenchannel",
        "# data to read even in principle.",
        # ALWAYS WRITTEN.  This was emitted only `if any(cfg.tbt_k_grid)`,
        # so an all-zero triple meant "inherit" and a PARTIAL zero -- which
        # the range allowed -- went into the deck verbatim, asking tbtrans
        # for zero k-points along an axis.  The grid is a fact the deck
        # records, like every other number here.
        f"TBT.k                  "
        f"{cfg.tbt_k_grid[0]} {cfg.tbt_k_grid[1]} {cfg.tbt_k_grid[2]}",
    ] + (
        [f"TBT.Spin               {cfg.tbt_spin}"]
        if cfg.tbt_spin else []) + [
        f"TBT.Elecs.Eta          {cfg.tbt_elecs_eta_ev:.6f} eV",
    ] + ([f"TBT.Contours.Eta       {cfg.tbt_contours_eta_ev:.6f} eV"]
         if cfg.tbt_contours_eta_ev > 0 else []) + [
        f"TBT.DOS.Gf             "
        f"{'true' if cfg.tbt_dos_gf else 'false'}",
        f"TBT.DOS.A              "
        f"{'true' if cfg.tbt_dos_a else 'false'}",
        f"TBT.DOS.Elecs          "
        f"{'true' if cfg.tbt_dos_elecs else 'false'}",
        f"TBT.T.Eig              {cfg.tbt_t_eig}",
        f"TBT.T.Bulk             "
        f"{'true' if cfg.tbt_t_bulk else 'false'}",
        f"TBT.T.All              "
        f"{'true' if cfg.tbt_t_all else 'false'}",
        f"TBT.Verbosity          {_TBT_VERBOSITY.get(cfg.log_level, 5)}",
        "# WHERE the device Hamiltonian is: SIESTA 5.x TranSIESTA writes",
        "# the converged H as <SystemLabel>.TS.HSX (the sparse container",
        "# that replaced the 4.x device .TSHS), and tbtrans 5.x looks for",
        "# <SystemLabel>.HSX unless told -- measured live 2026-08-29 on",
        "# 5.4.2: without this line it stops with 'Could not read",
        "# CT.HSX'.  Inert for the SCF run itself (TBT.* keys are",
        "# tbtrans's own).",
        f"TBT.HS                 {cfg.job_name}.TS.HSX",
        "",
    ])
    return lines


# --------------------------------------------------------------------- #
#  The engine                                                            #
# --------------------------------------------------------------------- #


class TransiestaEngine:
    """The TranSIESTA NEGF engine (the composite's deck emitter).

    See the module docstring for the deck contract; the composite's
    stage renders drive this through ``transport/stages.py``.  The
    engine self-registers via the ``@register_engine`` decorator
    on import; the production import path is
    ``molbuilder.web.blueprints.__init__`` (mirrors the chemistry
    adapters).
    """

    name = "transiesta"
    label = "TranSIESTA (NEGF, pseudopotentials)"

    # `render_script` DELETED 2026-09-17 -- the SECOND writer of a device
    # deck.  The framework's own pipeline writes all five rungs
    # (`siesta.input.spec_for(calculation="transport")` ->
    # `transport/deck.py` -> `prepare_deck`), and this one was reached only
    # by `/api/transport/render`, a route no browser calls.
    #
    # The two homes had already disagreed, with a run-killing result: this
    # path read `TransportConfig` while the live path reads `SiestaConfig`,
    # so the pole-energy correction of 2026-09-16 reached one and not the
    # other, and a deck rendered here stopped SIESTA before the SCF loop
    # ("requires at least 20 poles") for two days.
    #
    # `_emit_geometry`, `_emit_basis_and_xc`, `_emit_transiesta_block` and
    # `_find_electrode_regions` all SURVIVE: the live deck path reuses the
    # first and third, and the electrode wizard reuses the second and
    # fourth.  What went with this method is `_emit_header` and
    # `_emit_k_mesh`, which nothing else called.

    @classmethod
    def preflight(cls, struct: Structure,
                  cfg: TransportConfig,
                  prior=None,   # accepted, never passed non-None
                  ) -> List[Issue]:
        """Basic sidecar-region validation + cross-engine chemistry.

        Errors block generation; warnings are informational and
        the user can override (the form-rendering layer surfaces
        them inline).
        """
        issues: List[Issue] = []
        regions = struct.regions or {}

        # Device transport-axis k-points MUST be 1 (SCIENTIFIC-AUDIT FIX).
        # The transport direction (A3 = index 2) is treated by NEGF as an OPEN
        # boundary (semi-infinite leads); it is NOT part of the BZ sum.  kz > 1
        # imposes a fake Bloch periodicity along the wire -> physically WRONG
        # transmission with NO runtime error.  _emit_k_mesh writes
        # cfg.k_mesh_transverse[2] straight into the device kgrid, and
        # TransportConfig has no validator, so a bad preset (e.g. (4,4,2))
        # silently shipped a Bloch-periodic 'transport' run.  The cross-run
        # `transport preflight` CLI catches this (preflight.py C2), but the web
        # Generate path dispatches to THIS engine preflight -- so the invariant
        # must live here too (`engines/transport.md` § 5, I8).
        try:
            kz = int(cfg.k_mesh_transverse[2])
        except (TypeError, ValueError, IndexError):
            kz = 1
        if kz != 1:
            issues.append(Issue(
                severity="error",
                message=(
                    f"Device transport-axis k-points = {kz} (must be 1).  The "
                    f"transport direction is handled by NEGF as an open "
                    f"boundary and is NOT Brillouin-zone sampled; kz > 1 "
                    f"imposes a fake Bloch periodicity along the wire and "
                    f"gives physically wrong transmission.  Set "
                    f"k_mesh_transverse = (Nx, Ny, 1) (only the two TRANSVERSE "
                    f"directions are sampled)."
                ),
                where="config.k_mesh_transverse",
            ))

        # No silent absorption (`engines/transport.md` § 4): TranSIESTA
        # consumes only the canonical 2-terminal region labels.  A
        # structure carrying any OTHER region label has it silently
        # ignored unless we say so -- warn (don't drop quietly) so the
        # user knows that label plays no part in this calculation.
        # Emitted before the missing-region early return so it surfaces
        # even on an otherwise-incomplete region set.
        # ``buffer`` joined the consumed set 2026-08-28: the emitter
        # writes it as TS.Atoms.Buffer (`engines/transport.md` § 4), so
        # warning that it is ignored would be false.
        consumed = set(EXPECTED_REGIONS_2T) | {REGION_BUFFER}
        unknown_regions = [r for r in regions if r not in consumed]
        if unknown_regions:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"TranSIESTA preflight: structure carries region "
                    f"label(s) {sorted(unknown_regions)} that TranSIESTA "
                    f"does not consume (it uses only "
                    f"{sorted(consumed)}).  They are ignored "
                    f"for this calculation."
                ),
                where="struct.regions",
            ))

        # Required region labels for a 2-terminal calculation.
        missing = [r for r in EXPECTED_REGIONS_2T if r not in regions]
        if missing:
            issues.append(Issue(
                severity="error",
                message=(
                    f"TranSIESTA preflight: missing required region "
                    f"labels {missing}.  Assign them on the Molbuilder "
                    f"tab (region picker) before generating; the "
                    f".molstruct.json sidecar carries them through "
                    f"to the engine."
                ),
                where="struct.regions",
            ))
            # Without region labels we cannot validate further.
            return issues

        # Non-empty electrode atom counts.
        for r in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
            n = len(regions.get(r, []))
            if n == 0:
                issues.append(Issue(
                    severity="error",
                    message=(
                        f"TranSIESTA preflight: region {r!r} is empty.  "
                        f"Each electrode region must contain at least "
                        f"one atom."
                    ),
                    where=f"struct.regions.{r}",
                ))

        # Non-empty bridge.
        n_bridge = len(regions.get(REGION_BRIDGE, []))
        if n_bridge == 0:
            issues.append(Issue(
                severity="error",
                message=(
                    f"TranSIESTA preflight: region {REGION_BRIDGE!r} is "
                    f"empty.  The device region (bridge) must contain "
                    f"at least one atom — typically the molecule between "
                    f"the two electrodes."
                ),
                where=f"struct.regions.{REGION_BRIDGE}",
            ))

        # CRITICAL — atom ordering for TS.NumUsedAtomsLeft / Right.
        #
        # TranSIESTA reads ``TS.NumUsedAtomsLeft = N`` as "the FIRST
        # N atoms in the AtomicCoordinatesAndAtomicSpecies block are
        # the left electrode."  Same for Right (last M atoms).  If
        # the user's input XYZ has atoms in any order other than
        # [L-electrode][bridge][R-electrode], the .fdf SILENTLY
        # misidentifies which atoms go into which electrode self-
        # energy — producing chemically wrong transmission curves
        # with no run-time error.
        #
        # Reference: Brandbyge et al., Phys. Rev. B 65, 165401
        # (2002) § III; TranSIESTA manual ``TS.NumUsedAtomsLeft``
        # description.
        #
        # Block emission with an error so the user re-exports a
        # contiguous-ordered structure from the Molbuilder tab
        # (a "reorder for transport" affordance is a planned
        # follow-up).
        left_idx   = sorted(regions.get(REGION_LEFT_ELECTRODE, []))
        bridge_idx = sorted(regions.get(REGION_BRIDGE, []))
        right_idx  = sorted(regions.get(REGION_RIGHT_ELECTRODE, []))
        if left_idx and bridge_idx and right_idx:
            # WHICH BLOCK MUST COME FIRST IS GEOMETRY, not the label
            # (transport-design.md 4.1a): the block TranSIESTA reads as
            # the first electrode is the one that extends to -A3, and
            # that is the LOWER one.  A junction labeled the other way
            # round still sorts and still runs (the warning below says
            # what it means); what may never happen is the upper block
            # sitting first, because its lead would then point down
            # into the bridge.
            pos_z = np.asarray(struct.positions, dtype=float)[:, 2]
            _zl = float(np.mean(pos_z[left_idx]))
            _zr = float(np.mean(pos_z[right_idx]))
            lower_lab, upper_lab = ((REGION_LEFT_ELECTRODE,
                                     REGION_RIGHT_ELECTRODE) if _zl <= _zr
                                    else (REGION_RIGHT_ELECTRODE,
                                          REGION_LEFT_ELECTRODE))
            first_idx, last_idx = ((left_idx, right_idx) if _zl <= _zr
                                   else (right_idx, left_idx))
            ordering_ok = (
                first_idx[-1] < bridge_idx[0] and
                bridge_idx[-1] < last_idx[0] and
                # Each region must be contiguous (no gaps); a non-
                # contiguous electrode would also break the
                # "first N atoms" assumption.
                first_idx  == list(range(first_idx[0],
                                          first_idx[-1] + 1)) and
                bridge_idx == list(range(bridge_idx[0],
                                          bridge_idx[-1] + 1)) and
                last_idx   == list(range(last_idx[0],
                                          last_idx[-1] + 1))
            )
            if not ordering_ok:
                issues.append(Issue(
                    severity="error",
                    message=(
                        f"TranSIESTA preflight: atoms must be ordered "
                        f"as [{lower_lab}][{REGION_BRIDGE}]"
                        f"[{upper_lab}] in the AtomicCoordinates block "
                        f"-- the LOWER electrode block first -- with "
                        f"each region contiguous (no gaps).  TranSIESTA "
                        f"identifies electrode atoms POSITIONALLY (first "
                        f"N atoms = the first electrode, the -A3 lead); "
                        f"out-of-order labels produce SILENTLY WRONG "
                        f"transmission with no run-time error.  Got: "
                        f"{REGION_LEFT_ELECTRODE}={left_idx[0]}.."
                        f"{left_idx[-1]}, "
                        f"{REGION_BRIDGE}={bridge_idx[0]}.."
                        f"{bridge_idx[-1]}, "
                        f"{REGION_RIGHT_ELECTRODE}={right_idx[0]}.."
                        f"{right_idx[-1]}.  Run transport prep (it sorts "
                        f"by geometry), or re-export the structure in "
                        f"that order."
                    ),
                    where="struct.regions",
                ))

        # THE CONVENTION, CHECKED AND REPORTED -- never enforced (user
        # ruling, 2026-08-29).  An inverted junction is a valid
        # experiment whose author biased the other end; only they can
        # say which end they meant, so this states the measurement and
        # its consequence and leaves the decision where it belongs.
        # THE SAME DOOR the sort and the tab read (sort.py), never a
        # second derivation of the rule here.
        if left_idx and right_idx:
            from .sort import (ORDER_INVERTED, electrode_orientation,
                               inverted_note)
            if electrode_orientation(struct) == ORDER_INVERTED:
                issues.append(Issue(
                    severity="warn",
                    message=("TranSIESTA preflight: "
                             + inverted_note(struct)),
                    where="struct.regions",
                ))

        # High-bias INFO: Landauer linear-response regime breaks
        # down above ~2 V for typical molecular junctions (di Ventra,
        # Electrical Transport in Nanoscale Systems, 2008).  Surface
        # so users interpret high-bias results as snapshots of a
        # nonlinear I-V, NOT linearized conductance.
        if cfg.bias_voltages_v:
            max_v = max(abs(v) for v in cfg.bias_voltages_v)
            if max_v > 2.0:
                issues.append(Issue(
                    severity="warn",
                    message=(
                        f"Bias voltage |V| = {max_v:.2f} V is above the "
                        f"~2 V linear-response limit for typical "
                        f"molecular junctions.  TranSIESTA will still "
                        f"converge but the result should be interpreted "
                        f"as a single point on a nonlinear I-V curve, "
                        f"NOT a linearized Landauer conductance.  "
                        f"Consult Reed et al. 2006 / di Ventra 2008 "
                        f"for nonlinear-regime interpretation guidance."
                    ),
                    where="config.bias_voltages_v",
                ))

        # Multi-bias note: THIS single-deck render emits point 0 only.
        # The composite renders one deck per point at prep and launches
        # the .TSDE-chained walker (jobset/submit.py) -- this surface
        # (/api/transport/render) validates a single deck.
        if len(cfg.bias_voltages_v) > 1:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"This validation render emits ONE deck, at "
                    f"V = {cfg.bias_voltages_v[0]:.4f} V.  A scan over "
                    f"{list(cfg.bias_voltages_v)} is the COMPOSITE's "
                    f"job: describe it (--bias) and prep renders one "
                    f"deck per point, launched as one chain."
                ),
                where="config.bias_voltages_v",
            ))

        # Cross-engine chemistry: shared open-shell-metal check.
        # TransportConfig doesn't carry a spin treatment today;
        # treat the run as closed-shell unless future config adds
        # spin handling.  The check returns [] when there's no
        # open-shell metal present, so it's harmless on organics.
        from ..validation import check_open_shell_metal
        issues.extend(check_open_shell_metal(
            struct,
            is_closed_shell=True,
            engine_label="TranSIESTA (this Transport calculation)",
        ))

        return issues

    # `parse_output` and `methods_fragment` DELETED 2026-09-17.  Neither had
    # a caller anywhere; the first raised `NotImplementedError` and the
    # second returned a paragraph ending "(Full Methods paragraph deferred
    # to the follow-up release...)".  Both were declared on the Protocol as
    # future work, and a placeholder that nothing calls is not a contract --
    # it is a promise stored in the wrong place.  The record the first would
    # have read does exist (`transport/record.py` writes
    # `<label>.transport.json`), so when a transmission inspector lands it
    # reads that file; it does not need a stub kept warm here.

