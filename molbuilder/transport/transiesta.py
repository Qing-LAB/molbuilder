"""TranSIESTA engine for the Transport tab (Phase B.3, 2026-06-10).

Scope of THIS module — what's in vs deferred
============================================

**Shipped today (B.3 zero-bias scope):**

* ``TransiestaEngine`` class registered as ``"transiesta"`` in the
  :mod:`molbuilder.transport` engine registry.
* :meth:`TransiestaEngine.render_script` — emits a runnable device
  ``.fdf`` at ZERO bias (``TS.Voltage = bias_voltages_v[0]``).
  The script assumes the electrode ``.TSHS`` files already exist
  in the run directory; see "Electrode .fdf workflow" below for
  how to produce them.
* :meth:`TransiestaEngine.preflight` — basic region-label
  validation + the cross-engine ``check_open_shell_metal`` check.

**Deferred to follow-up sessions:**

* :meth:`TransiestaEngine.parse_output` — needs the
  ``<job>.transport.json`` schema designed first.  Raises
  ``NotImplementedError`` with a clear message today.
* :meth:`TransiestaEngine.methods_fragment` — full manuscript-
  ready paragraph; today returns a one-line placeholder.
* Electrode ``.fdf`` generation (see below).
* Bias-scan looping (see below).
* PySCF-NEGF engine.

Electrode .fdf workflow
=======================

TranSIESTA needs a ``.TSHS`` Hamiltonian file per electrode (left
+ right).  Each is the output of a SEPARATE SIESTA run on a lead
geometry (typically a few unit cells of the lead material — e.g.
Au(111) slab for gold electrodes — with periodic boundary
conditions along the transport direction).  The lead ``.fdf``
must include ``TS.HSFileEnable = T`` so SIESTA writes the
``.TSHS`` file on completion.

**Current path (manual):**

1. Build the lead geometry on the ``/molbuilder`` tab (Au(111)
   slab, etc.).
2. On ``/structure-optimization``, generate a SIESTA ``.fdf``;
   manually add ``TS.HSFileEnable = T`` to the emitted file
   (today's SIESTA emitter doesn't set this — a future cleanup).
3. Run SIESTA on the lead.  It emits ``<jobname>.TSHS``.
4. Rename to ``<device_jobname>_L_electrode.TSHS`` (or
   ``_R_electrode.TSHS``) and copy into the device run directory.
5. Repeat for the right electrode.
6. Generate the device ``.fdf`` here.  ``TS.HSFileLeft`` and
   ``TS.HSFileRight`` reference the files copied in step 4.

**Planned (B.3 follow-up):** Add an "electrode wizard" step on
the Transport tab that extracts the ``L-electrode`` /
``R-electrode`` regions from the labeled device structure, plus a
few periodic-image unit cells, and emits the matching SIESTA
``.fdf`` automatically.  Tracked in memory
``project_transport_electrode_bias_workflow.md``.

Bias-scan workflow
==================

TranSIESTA's ``TS.Voltage`` is a single value per run.  T(E) at
multiple bias values = multiple TranSIESTA runs (re-converge NEGF
density at each bias point — expensive).

``TransportConfig.bias_voltages_v`` is already a ``List[float]``.
Today's engine emits ONLY ``bias_voltages_v[0]``; the preflight
WARNs when ``len > 1``.

**Current path (manual):** Generate a separate ``.fdf`` for each
bias value by changing ``bias_voltages_v`` and re-clicking
Generate.  Run each ``.fdf`` separately.  Post-process the per-
run ``.TBT.AVTRANS_*`` files into a single I-V curve.

**Planned (B.3 follow-up):** For ``len(bias_voltages_v) > 1``,
emit ONE ``.fdf`` per bias point named ``<jobname>_V0.fdf``,
``<jobname>_V1.fdf``, etc., plus a ``<jobname>_run.sh`` driver
script that loops over them and post-processes the per-bias
transmission files into a single ``<job>.transport.json``.
Tracked in the same memory above.

Runnable assumptions
====================

The device ``.fdf`` emitted by :meth:`render_script` is runnable
**if and only if**:

* ``<jobname>_L_electrode.TSHS`` and ``<jobname>_R_electrode.TSHS``
  exist in the run directory (see Electrode workflow above).
* SIESTA-MPI is built with TranSIESTA enabled.
* Atomic pseudopotentials (``.psml`` files) for every species
  present are reachable via the standard SIESTA pseudo-lookup
  (project's ``pseudopotential/`` dir or ``$SIESTA_PP_PATH``).

If any of these are missing, SIESTA fails at parse time with a
clear error.  ``render_script`` doesn't try to predict the
runtime; preflight catches structural issues that would block
emission regardless.

Four-env model + the runner shell script
========================================

molbuilder's :doc:`four-env model </README_install>` separates the
host env from the per-backend envs.  TranSIESTA is part of the
SIESTA suite, so the device ``.fdf`` runs in
``molbuilder-siesta`` — the same env the standalone SIESTA
optimization flow uses.

The web blueprint pairs ``render_script``'s ``.fdf`` output with
a ``<jobname>.run.sh`` shell wrapper produced by
:func:`molbuilder.runwrap.render_run_wrapper`.  The wrapper does
NOT need to be customised per engine; runwrap routes ``.fdf``
extensions to the ``siesta`` category automatically + picks up
``molbuilder-siesta`` via ``Capabilities.env_for_category('siesta')``.
The wrapper handles the three conda-activation paths (already in
env / conda on PATH / conda missing) and surfaces a clear error
if the target env isn't installed.

So a user's full lifecycle is:

1. Generate the device ``.fdf`` from the Transport tab (this
   engine's :meth:`render_script`).
2. Web blueprint also emits ``<jobname>.run.sh`` via
   :func:`molbuilder.runwrap.write_run_wrapper`.
3. User runs ``bash <jobname>.run.sh`` from the run directory.
4. Wrapper activates ``molbuilder-siesta``, launches
   ``mpirun -np <N> siesta <jobname>.fdf``, captures output.

The conda env model is invisible to the engine's emitter; the
runner ``.run.sh`` carries it.  Pinned in memory
``project_four_env_model.md``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config.transport import (
    ELECTRODE_LABEL_SUFFIX,
    EXPECTED_REGIONS_2T,
    REGION_BRIDGE,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    TransportConfig,
    is_electrode_label,
)
from ..issues import Issue
from ..structure import Structure
from .engine_base import TransportEngine, register_engine
from .results import TransportResults


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


def _find_electrode_regions(
    struct: Structure,
) -> List[Tuple[str, str, List[int]]]:
    """Return (user_label, block_name, indices) for every region
    whose label ends with the electrode-suffix convention.

    Sorted by z-centroid ascending so the first entry is the
    "leftmost" electrode (minimum z); the last is the "rightmost"
    (maximum z).  This deterministic ordering is what the emitter
    uses to assign ``semi-inf-direction -A3`` / ``+A3`` and the
    canonical ``Left`` / ``Right`` chempot names.
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


def _emit_header(cfg: TransportConfig, struct: Structure) -> List[str]:
    """SystemLabel + a banner explaining what the file is.

    The banner is verbose-comments-style (matches the SIESTA
    emitter convention) so a user reading the ``.fdf`` cold sees
    the deferred-scope notes immediately.  Electrode .TSHS file
    names are derived from the actual region labels in the input
    structure (per the *-electrode convention) — so users with
    custom electrode names see their own file names in the banner.

    Runtime-hint echo (``# runtime.<key>: <val>``) mirrors the
    SIESTA emitter at ``siesta/input.py:522-525`` — the .out parser
    reads these via ``_SIESTA_RUNTIME_RE`` so the Results tab can
    show what the user requested even when the run is paused / the
    wrapper has already exited.  Same key names as the SIESTA side
    so a single parser handles both engines.
    """
    electrodes = _find_electrode_regions(struct)
    file_lines = [
        f"#    {cfg.job_name}_{label}.TSHS"
        for label, _block_name, _idxs in electrodes
    ] or [f"#    {cfg.job_name}_<electrode-label>.TSHS"]
    lines = [
        "# ================================================================== #",
        f"#  TranSIESTA device .fdf — {cfg.job_name}",
        "#  Generated by molbuilder (modern SIESTA 4.1+ / 5.x syntax).",
        "#",
        "#  Assumes electrode .TSHS files exist in this directory:",
        *file_lines,
        "#  See the SIESTA manual § 'TranSIESTA' for electrode-generation",
        "#  workflow; molbuilder's automated electrode wizard ships in a",
        "#  follow-up release.",
        "#",
        "#  Single bias point per .fdf (zero-bias today).  Bias-scan",
        "#  generation lands alongside the .transport.json parser.",
        "# ================================================================== #",
        "",
    ]
    # Runtime hints -- same key names + value format as SIESTA so the
    # shared parser at parse/engines/siesta.py:1407 reads both.
    if getattr(cfg, "num_threads", None):
        lines.append(
            f"# runtime.omp_threads_requested: {int(cfg.num_threads)}")
    if getattr(cfg, "max_memory_mb", None):
        lines.append(
            f"# runtime.max_memory_mb: {int(cfg.max_memory_mb)}")
    if lines[-1].startswith("# runtime."):
        lines.append("")
    lines.extend([
        f"SystemLabel            {cfg.job_name}",
        f"SystemName             Transport device for {cfg.job_name}",
        "",
    ])
    return lines


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
    warning.  Atom coordinates are always emitted verbatim.
    """
    from ase.data import atomic_numbers as _Z
    species = sorted(set(struct.elements), key=lambda e: e.capitalize())
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}

    resolved_cell = cell if cell is not None else struct.cell

    lines: List[str] = ["# --- Geometry ---", ""]
    lines.append(f"NumberOfAtoms          {struct.n_atoms}")
    lines.append(f"NumberOfSpecies        {len(species)}")
    lines.append("")
    lines.append("%block ChemicalSpeciesLabel")
    for sp in species:
        z = _Z.get(sp.capitalize(), 0)
        lines.append(f"  {species_idx[sp]:>3}  {z:>3}  {sp}")
    lines.append("%endblock ChemicalSpeciesLabel")
    lines.append("")
    lines.extend(_lattice_block(struct, resolved_cell))
    lines.append("")
    lines.append("AtomicCoordinatesFormat        Ang")
    lines.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, struct.positions):
        lines.append(
            f"  {x:14.8f} {y:14.8f} {z:14.8f}  {species_idx[el]}"
        )
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    lines.append("")
    return lines


def _emit_basis_and_xc(cfg: TransportConfig) -> List[str]:
    """Basis set + XC + mesh cutoff + electronic temperature.

    Production-defensible defaults matching the SIESTA emitter's
    Method tab.  Mesh cutoff comes from the form; everything else
    is hard-coded for the zero-bias scope (B.3 follow-up: surface
    these as form fields).
    """
    return [
        "# --- Basis + XC ---",
        "",
        "PAO.BasisSize          DZP",
        "PAO.EnergyShift        0.01 Ry",
        "XC.Functional          GGA",
        "XC.Authors             PBE",
        f"MeshCutoff             {cfg.siesta_mesh_cutoff_ry} Ry",
        f"ElectronicTemperature  {cfg.electronic_temperature_k:.1f} K",
        "",
    ]


def _emit_k_mesh(cfg: TransportConfig) -> List[str]:
    """Monkhorst-Pack block for the TRANSVERSE k-mesh.

    The transport direction is treated by NEGF and is NOT part of
    the BZ sum.  For a finite molecule between leads, (1, 1, 1) is
    correct; for 1D-periodic electrodes set Nx, Ny appropriately.
    """
    kx, ky, kz = cfg.k_mesh_transverse
    return [
        "# --- Transverse k-mesh ---",
        "# Transport direction is NOT BZ-summed (NEGF handles it).",
        "# For a finite molecule between leads: (1, 1, 1).  For",
        "# 1D-periodic electrodes set Nx, Ny to the lead",
        "# periodicities; Nz stays at 1.",
        "",
        "%block kgrid_Monkhorst_Pack",
        f"  {kx:>3}    0    0      0.0",
        f"    0  {ky:>3}    0      0.0",
        f"    0    0  {kz:>3}      0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "",
    ]


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
      naturally.  Order is by z-centroid so the leftmost gets
      ``semi-inf-direction -A3`` and the canonical ``Left``
      chempot binding.

    For the canonical 2-terminal case (the only fully-validated
    scope today), the emitter produces exactly the verified
    Au-BDT-Au template.  Multi-terminal (3+ electrodes) is a
    planned follow-up; today such a structure emits a single
    notice in render_script's pre-emit pass.

    TBtrans block (transmission post-processing) is unchanged —
    its keyword names didn't migrate in 4.1+.
    """
    bias = cfg.bias_voltages_v[0] if cfg.bias_voltages_v else 0.0
    erange_relative = "T" if cfg.transmission_relative_to_ef else "F"

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
    if is_two_terminal:
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
        "# Electrode declarations.  Each electrode is a region in the",
        "# input structure whose label ends with ``-electrode``; the",
        "# emitter discovers them from ``struct.regions`` and emits one",
        "# %block TS.Elec.<name> per side.  See "
        "docs/protocols/region-labels.md.",
        "%block TS.Elecs",
    ]
    for _label, block_name, _idxs in electrodes:
        lines.append(f"  {block_name}")
    lines.append("%endblock TS.Elecs")
    lines.append("")

    # Per-electrode blocks.
    for i, (label, block_name, idxs) in enumerate(electrodes):
        cp = chempot_for[block_name]
        sid = semi_inf.get(i, "+A3")
        lines.extend([
            f"%block TS.Elec.{block_name}",
            f"  HS                 {cfg.job_name}_{label}.TSHS",
            f"  chem-pot           {cp}",
            f"  used-atoms         {len(idxs)}",
            "  bloch              1 1 1",
            f"  semi-inf-direction {sid}",
            f"%endblock TS.Elec.{block_name}",
            "",
        ])

    # Chemical potentials.
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

    lines.extend([
        "# Bias voltage (used by the ``V`` substitution in the chempot "
        "mu lines).",
        "# Single value per .fdf today; the bias-scan workflow emits one "
        ".fdf per bias.",
        f"TS.Voltage             {bias:.4f} eV",
        "",
        "# TBtrans transmission post-processing.",
        "# Brandbyge et al., Phys. Rev. B 65, 165401 (2002) § IV.",
        f"TS.TBT.NumE            {cfg.transmission_n_points}",
        f"TS.TBT.Emin            {cfg.transmission_emin_ev:.2f} eV",
        f"TS.TBT.Emax            {cfg.transmission_emax_ev:.2f} eV",
        f"TS.TBT.Erange.RelToEF  {erange_relative}",
        "",
    ])
    return lines


# --------------------------------------------------------------------- #
#  The engine                                                            #
# --------------------------------------------------------------------- #


@register_engine
class TransiestaEngine:
    """TranSIESTA NEGF engine (Phase B.3 zero-bias scope).

    See the module docstring for what's in vs deferred, the
    electrode-generation workflow, and the bias-scan plan.  The
    engine self-registers via the ``@register_engine`` decorator
    on import; the production import path is
    ``molbuilder.web.blueprints.__init__`` (mirrors the chemistry
    adapters).
    """

    name = "transiesta"
    label = "TranSIESTA (NEGF, pseudopotentials)"

    @classmethod
    def render_script(cls, struct: Structure,
                       cfg: TransportConfig) -> str:
        """Emit a runnable device ``.fdf`` for TranSIESTA at zero bias.

        Concatenates: header / geometry / basis+XC / k-mesh / TS.*.
        Output assumes electrode ``.TSHS`` files exist at the
        standard names (see module docstring).
        """
        lines: List[str] = []
        lines.extend(_emit_header(cfg, struct))
        lines.extend(_emit_geometry(struct))
        # ATOM-METADATA sidecar v3 block.  Mirrors siesta/input.py:1330
        # so a parsed-back transport .fdf can recover the original
        # regions + frozen_atoms (electrode block names are sanitized
        # for SIESTA at line 165; the original Python labels are
        # ONLY here).  emit_atom_metadata returns None when both
        # regions and frozen_atoms are empty, in which case nothing
        # is appended.
        from molbuilder.script_emit import emit_atom_metadata
        block = emit_atom_metadata(
            regions      = struct.regions or {},
            frozen_atoms = list(struct.frozen_atoms or []),
            annotations  = dict(getattr(struct, "annotations", {}) or {}),
            n_atoms_total= struct.n_atoms,
        )
        if block:
            lines.append(block)
            lines.append("")
        lines.extend(_emit_basis_and_xc(cfg))
        lines.extend(_emit_k_mesh(cfg))
        lines.extend(_emit_transiesta_block(struct, cfg))
        return "\n".join(lines) + "\n"

    @classmethod
    def preflight(cls, struct: Structure,
                  cfg: TransportConfig,
                  prior: Optional[TransportResults] = None,
                  ) -> List[Issue]:
        """Basic sidecar-region validation + cross-engine chemistry.

        Errors block generation; warnings are informational and
        the user can override (the form-rendering layer surfaces
        them inline).
        """
        issues: List[Issue] = []
        regions = struct.regions or {}

        # No silent absorption (sidecar-contract.md § 6): TranSIESTA
        # consumes only the canonical 2-terminal region labels.  A
        # structure carrying any OTHER region label has it silently
        # ignored unless we say so -- warn (don't drop quietly) so the
        # user knows that label plays no part in this calculation.
        # Emitted before the missing-region early return so it surfaces
        # even on an otherwise-incomplete region set.
        unknown_regions = [r for r in regions
                           if r not in EXPECTED_REGIONS_2T]
        if unknown_regions:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"TranSIESTA preflight: structure carries region "
                    f"label(s) {sorted(unknown_regions)} that TranSIESTA "
                    f"does not consume (it uses only "
                    f"{list(EXPECTED_REGIONS_2T)}).  They are ignored "
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
            ordering_ok = (
                left_idx[-1]  < bridge_idx[0] and
                bridge_idx[-1] < right_idx[0] and
                # Each region must be contiguous (no gaps); a non-
                # contiguous L-electrode would also break the
                # "first N atoms" assumption.
                left_idx   == list(range(left_idx[0],
                                          left_idx[-1] + 1)) and
                bridge_idx == list(range(bridge_idx[0],
                                          bridge_idx[-1] + 1)) and
                right_idx  == list(range(right_idx[0],
                                          right_idx[-1] + 1))
            )
            if not ordering_ok:
                issues.append(Issue(
                    severity="error",
                    message=(
                        "TranSIESTA preflight: atoms must be ordered "
                        "as [L-electrode][bridge][R-electrode] in the "
                        "AtomicCoordinates block, with each region "
                        "contiguous (no gaps).  TranSIESTA reads "
                        "TS.NumUsedAtomsLeft as 'first N atoms = "
                        "left electrode'; out-of-order labels "
                        "produce SILENTLY WRONG transmission with "
                        "no run-time error.  Got: "
                        f"L-electrode={left_idx[0]}..{left_idx[-1]}, "
                        f"bridge={bridge_idx[0]}..{bridge_idx[-1]}, "
                        f"R-electrode={right_idx[0]}..{right_idx[-1]}.  "
                        "Re-export the structure from the Molbuilder "
                        "tab with atoms in contiguous L→bridge→R "
                        "order before re-running."
                    ),
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

        # Multi-bias warning (today's scope is zero-bias only).
        if len(cfg.bias_voltages_v) > 1:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"Bias-scan generation is deferred to a follow-up "
                    f"release.  Today's engine emits a single .fdf at "
                    f"V = {cfg.bias_voltages_v[0]:.4f} V; subsequent "
                    f"values {cfg.bias_voltages_v[1:]} are ignored.  "
                    f"For now, regenerate the .fdf with each bias "
                    f"value separately (see module docstring under "
                    f"'Bias-scan workflow')."
                ),
                where="config.bias_voltages_v",
            ))

        # Cross-engine chemistry: shared open-shell-metal check.
        # TransportConfig doesn't carry spin_polarized today;
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

    @classmethod
    def parse_output(cls, path: str) -> TransportResults:
        """Read a ``<job>.transport.json`` produced by this engine.

        Deferred to a follow-up release alongside the schema design
        for ``.transport.json``.  Tracked in memory
        ``project_transport_results_tab_framework.md``.
        """
        raise NotImplementedError(
            "TranSIESTA parse_output is deferred to a follow-up release "
            "alongside the <job>.transport.json schema design + the "
            "/results inspector.  Today's engine ships render_script + "
            "preflight only.  See molbuilder/transport/transiesta.py "
            "module docstring for the deferred-scope plan."
        )

    @classmethod
    def methods_fragment(cls, cfg: TransportConfig,
                          results: TransportResults) -> str:
        """Engine-specific paragraph for the Methods section.

        Today returns a one-line placeholder.  Full version lands
        alongside parse_output (the manuscript prose interpolates
        actual run parameters from ``results``, which doesn't exist
        yet for TranSIESTA).
        """
        return (
            f"Transport calculations were performed with TranSIESTA "
            f"[CITE:transiesta_brandbyge_2002] from the SIESTA suite, "
            f"using a DZP basis, the PBE functional, and a "
            f"{cfg.siesta_mesh_cutoff_ry} Ry real-space mesh cutoff.  "
            f"NEGF density integration used a complex contour with "
            f"{cfg.contour_n_circle} imaginary-axis points "
            f"(Brandbyge et al. 2002 § IV).  "
            f"Electronic temperature was {cfg.electronic_temperature_k:.0f} K.  "
            f"(Full Methods paragraph deferred to the follow-up release "
            f"that lands parse_output + the .transport.json schema.)"
        )
