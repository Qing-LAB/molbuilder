"""The TranSIESTA emitters — the transport composite's deck layer.

**THIS MODULE NO LONGER WRITES A DECK OF ITS OWN** (2026-09-17).  It is a
set of emitters the two live writers reach into, plus the engine preflight:

* :func:`_emit_transiesta_block` and :func:`_emit_geometry` — reused by
  `transport/deck.py`, the pipeline every one of the five rungs renders
  through.
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
    TransportConfig,
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

    Transverse (a, b): atom-extent + twice the per-side vacuum — the
    structure's own when it states one, else
    :data:`_ISOLATED_ELECTRODE_VACUUM_ANG`.
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
    # THE PERSON'S VACUUM WINS, and it was ignored until 2026-09-23.
    # `vacuum` has three states (`structure-periodicity.md` § 2): a number,
    # used verbatim; `[0,0,0]`, meaning no gap DELIBERATELY, also verbatim;
    # and unset, which is the only one this default answers.  Someone who
    # typed 8 Å on the Cell page got 15 Å in the deck and nothing said so.
    stated = struct.vacuum
    pad_x, pad_y = ((float(stated[0]), float(stated[1])) if stated is not None
                    else (_ISOLATED_ELECTRODE_VACUUM_ANG,) * 2)
    a = extent_x + 2.0 * pad_x
    b = extent_y + 2.0 * pad_y
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
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
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
                   cell: Optional[np.ndarray] = None,
                   cfg=None) -> List[str]:
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


# `TransiestaEngine` DELETED 2026-09-17 -- the last of the June 2026 era.
#
# The class ended as one classmethod, `preflight`, reached only through
# `_ENGINE_VALIDATORS[TransportConfig]` in `validation/__init__.py`.  **Nothing
# validates a `TransportConfig`.**  Every rung resolves a `SiestaConfig`
# (`engines/transport.md` 2a.14), and the two sites that still BUILD a
# `TransportConfig` -- `deck.py::_legacy_view` and `stages.py::config_for` --
# build it as a projection to feed the lifted NEGF block emitter and never
# validate it.  So the registration dispatched for nothing.
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
#   `_emit_geometry` ............. `deck.py` (all three deck shapes, so every
#                                  one of the five rungs)
#   `_emit_transiesta_block` ..... `deck.py`, the `negf` shape -- the device
#                                  and transmission rungs
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
