"""SIESTA .fdf input generator.

Takes a Structure (or an XYZ/PDB file path) and emits a .fdf input file
ready to drop into a SIESTA run, with optional auto-copy of PSML
pseudopotentials from a flat library on disk.

Public API:
    SiestaConfig      -- dataclass holding every FDF parameter
    Config            -- backwards-compat alias for SiestaConfig
    render_fdf(...)   -- format an in-memory Structure as FDF text
    convert(...)      -- read XYZ/PDB, write FDF, optionally copy psml
    copy_pseudopotentials(...) -- standalone psml copy helper

The CLI lives in :mod:`molbuilder.cli` as the ``fdf`` subcommand.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ase.data import atomic_numbers
    from ase.io import read as _ase_read
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "molbuilder.siesta needs ASE; install with `pip install ase`"
    ) from exc

from ..structure import Structure
# SiestaConfig is the L1 dataclass; this module imports it for use by
# the generator below.  External callers can import it from either
# molbuilder.config.siesta (the canonical location) or from
# molbuilder.siesta (re-exported by siesta/__init__.py).
from ..config.siesta import SiestaConfig




# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _auto_block_size(n_atoms: int,
                     mpi_np: Optional[int] = None,
                     gpu_mode: bool = False) -> int:
    """Pick a SIESTA ``BlockSize`` for the ScaLAPACK orbital
    distribution.  Affects cache efficiency at moderate rank counts.

    HISTORICAL NOTE (2026-05-28 correction)
    ----------------------------------------
    This function previously claimed to guard against ``propor:
    ERROR: IMAX = 0`` via the formula
    ``BlockSize <= floor(n_atoms / Nrank)``.  Empirical sweep
    (probe in /tmp/siesta-mpi-probe, results captured in design
    notes) confirms that claim was wrong: SIESTA crashes IDENTICALLY
    with BlockSize = 1, 2, 4 at mpi_np = 15 on hemeC-dithiol.

    The propor crash is in ``matel_table.F90``'s MPI-deduplication
    of radial-function tables, not in any BLACS distribution.  It
    is a function of ``mpi_np`` vs the molecule's species count and
    radial-table size; predicting it from BlockSize is impossible
    because BlockSize doesn't enter the matel_table loop.  See
    ``runwrap.py``'s post-run diagnostic for the user-facing fix
    (the wrapper's ``-np`` runtime override).

    What this function STILL does
    -----------------------------
    Pick a power-of-2 BlockSize that gives ScaLAPACK good cache
    behaviour at the requested rank count.  Larger BlockSize
    reduces communication overhead per orbital block; too-large
    leaves some ranks idle on the diag step.  The ``floor(n_atoms
    / mpi_np)`` formula is a reasonable upper bound for the diag
    block; it just isn't the propor-fixing constraint it was
    advertised as.

    Strategy
    --------
    Two regimes -- CPU mode and GPU mode -- because the optimal
    BlockSize is fundamentally different on the two solvers.

    CPU mode (default)
      * mpi_np is None or 1: size-only ladder (1, 2, 4, 8 by
        n_atoms), capped at 8.
      * mpi_np >= 2: largest power of 2 satisfying
        ``BlockSize <= floor(n_atoms / mpi_np)``.

    GPU mode (``gpu_mode=True``)
      Orbital-aware formula with two caps:

        BlockSize = largest power of 2 in
                    ``[8, min(256, 1024, floor(10·n_atoms/mpi_np))]``

      The 10·n_atoms term estimates n_orbitals (rough DZP heuristic;
      underestimates for heavy elements like Au where DZP gives
      ~25 orb/atom).  The 256 cap is a defensible upper bound:
      bigger than the historical CPU number (which under-shot by
      using n_atoms not n_orbitals), within the range of values
      ELPA-GPU benchmark papers actually swept, and small enough
      that load balance stays good with ≤4 ranks.  The 1024 cap is
      the ELPA CUDA kernel hard limit (2^10).

      Honest framing: the "right" GPU BlockSize is hardware- and
      problem-dependent.  Without measurement on the target box
      no single number can claim "the optimum".  256 is a
      defensible default that's bigger than 64 (overhead-bound)
      and smaller than 512 (load-imbalance-prone), measured in
      kernel launch latency × work-per-launch ratios.  See
      ``scripts/bench-siesta-blocksize.sh`` for an in-tree sweep
      script that runs a few short SIESTA jobs with different
      BlockSize values and prints wall-time/iter for direct
      comparison.

      Concrete numbers:
        n_atoms=212, mpi_np=4 (the 2026-06-16 Au-BDT case):  256
        n_atoms=16,  mpi_np=4 (tiny test fixture):            32
        n_atoms=1000, mpi_np=4 (big metal slab):             256

    HISTORY
      The pre-2026-06-16 CPU formula was used uniformly.  For the
      typical GPU-form path (cfg.mpi_np = None, gpu_mode = True)
      it fell into the size-only ladder and returned 8 for any
      system >=16 atoms -- way below the ELPA-CUDA optimum.  A
      live 212-atom Au-BDT GPU run was using BlockSize=8 the whole
      time before this fix.

    Returns
    -------
    A positive power of 2.  Safe to use regardless of mpi_np; if
    SIESTA still crashes at startup with propor IMAX=0, the issue
    is mpi_np / molecule mismatch, not BlockSize.
    """
    if gpu_mode:
        # GPU mode wants bigger BlockSize than the historical CPU-
        # mode formula gives (which uses n_atoms instead of
        # n_orbitals and so under-estimates by ~10x).
        #
        # The orbital-aware cap is ``floor(10 * n_atoms / mpi_np)``
        # (the 10x is a rough DZP-basis heuristic; underestimates
        # for heavy elements like Au where DZP gives ~25 orb/atom).
        # Two further caps narrow the choice to a defensible range:
        #
        #   * ``256`` (load-balance ceiling).  With BlockSize > 256
        #     and mpi_np <= 4, you typically get fewer than 12 blocks
        #     per rank; tail-effect load imbalance starts to bite.
        #     Above 256 also moves outside the BlockSize range that
        #     ELPA-GPU benchmark papers actually swept, so we'd be
        #     extrapolating off the measured curve.
        #   * ``1024`` (ELPA CUDA kernel hard limit, 2^10).
        #
        # An empirical sweep on the target hardware (see
        # ``scripts/bench-siesta-blocksize.sh``) can refine this
        # default for power users.  Without measurement, 256 is a
        # defensible upper bound -- bigger than the historical CPU
        # number, smaller than the kernel limit, and within the
        # range the literature has actually measured.
        np_ = max(1, int(mpi_np)) if mpi_np else 4  # 4 = GPU+MPS default
        orbital_estimate = 10 * max(1, n_atoms)
        upper = min(256, 1024, orbital_estimate // np_)
        pow2 = 8
        while pow2 * 2 <= upper:
            pow2 *= 2
        return pow2
    if not mpi_np or int(mpi_np) <= 1:
        # CPU + no rank info -- conservative size-only baseline.
        # Cap at 8 is the historical safety choice; with mpi_np
        # known we remove this ceiling below.
        if   n_atoms >= 16:  return 8
        elif n_atoms >=  8:  return 4
        elif n_atoms >=  4:  return 2
        else:                return 1
    # CPU mode with mpi_np >= 2.  Rank constraint: every rank must
    # get >= 1 atom block, i.e. ``BlockSize <= floor(n_atoms /
    # mpi_np)``.  Take the LARGEST power of 2 that satisfies it.
    cap = max(1, n_atoms // int(mpi_np))
    pow2 = 1
    while pow2 * 2 <= cap:
        pow2 *= 2
    return pow2


def _detect_species(elements: Iterable[str]) -> List[str]:
    """Unique species, sorted by atomic number, preserving first-seen order
    only as a tiebreaker."""
    seen: List[str] = []
    for s in elements:
        if s not in seen:
            seen.append(s)
    return sorted(seen, key=lambda s: atomic_numbers[s])


def _wrap_into_cell(positions: np.ndarray, cell: np.ndarray
                    ) -> Tuple[np.ndarray, int]:
    """Fold atoms so their fractional coordinates sit in [0, 1).

    Treats ``cell`` as a (3, 3) matrix whose ROWS are the lattice
    vectors a, b, c -- i.e. a Cartesian position P satisfies
    ``P = (u, v, w) @ cell`` for some fractional triple (u, v, w).

    Returns ``(wrapped_positions, n_wrapped_atoms)`` so callers can
    print a useful note in the FDF.  Atoms whose original fractional
    coordinates lie within 1e-9 of an integer (i.e. essentially on the
    cell face -- either just inside the cell at frac ~ 0.9999... or
    just outside at frac ~ 1.0 + 1e-12) are wrapped if needed but NOT
    counted as moved.  Their motion is numerical-precision noise, not
    a meaningful position change.

    Algorithm
    ---------
    The 2026-05-28 audit caught the pre-existing
    ``floor(fractional + 1e-9)`` form doing the OPPOSITE of its
    docstring: it wrapped frac = 0.9999999999 (in the cell) to
    -1e-10 (outside the cell, counted as moved) while leaving
    frac = -1e-10 (outside the cell) at -1e-10 (still outside, NOT
    counted).  Both wrong.  The clean fix is to (a) do a standard
    ``floor`` wrap so EVERY atom lands cleanly in [0, 1), and (b)
    decide separately whether to *count* the wrap as a move.
    """
    inv = np.linalg.inv(cell)
    fractional = positions @ inv
    # Standard wrap into [0, 1).  No tolerance hack here -- every
    # atom lands cleanly inside the cell.
    wrapped = fractional - np.floor(fractional)
    # Signed fractional displacement caused by the wrap.
    delta = wrapped - fractional
    # A "real" wrap moved the atom by > 1e-6 in fractional space.
    # Anything smaller is round-off (no atom geometry actually
    # depended on the wrap).
    big_move = np.any(np.abs(delta) > 1e-6, axis=1)
    # Atoms whose ORIGINAL fractional was within 1e-9 of an integer
    # (~0, ~1, ~2, ...) were sitting essentially on a cell face.
    # The wrap may have shifted them visibly (e.g. frac = -1e-10
    # -> wrapped = 1 - 1e-10, a delta of ~1) but the motion is
    # purely numerical noise, not a meaningful translation.  Exclude
    # them from the count so the user-facing notice doesn't lie.
    on_boundary = np.any(
        np.abs(fractional - np.round(fractional)) < 1e-9, axis=1
    )
    moved_mask = big_move & ~on_boundary
    n_moved = int(moved_mask.sum())
    new_positions = wrapped @ cell
    # Round-trip preservation: when the wrap was a no-op in
    # fractional space (atom was already inside), restore the
    # ORIGINAL Cartesian so 1e-12 matrix-product drift doesn't
    # appear in the FDF.  GATED ON big_move, not moved_mask --
    # boundary atoms that genuinely DID wrap (frac ~ 1.0 + 1e-12)
    # need to keep their post-wrap Cartesian, otherwise we'd
    # silently re-place them outside the cell.
    new_positions[~big_move] = positions[~big_move]
    return new_positions, n_moved


def find_psml(element: str, lib: Path) -> Optional[Path]:
    """Locate a pseudopotential file for `element` in a flat lib folder."""
    for name in (f"{element}.psml", f"{element.lower()}.psml",
                 f"{element.upper()}.psml"):
        p = lib / name
        if p.is_file():
            return p
    matches = sorted(lib.glob(f"{element}*.psml"))
    if matches:
        if len(matches) > 1:
            print(f"  note: {len(matches)} variants of {element}.psml in "
                  f"{lib}; using {matches[0].name}", file=sys.stderr)
        return matches[0]
    return None


def copy_pseudopotentials(species: Sequence[str], lib: Path,
                          dest_dir: Path) -> List[str]:
    """Copy psml files for each species. Returns list of missing species."""
    missing: List[str] = []
    for s in species:
        src = find_psml(s, lib)
        if src is None:
            missing.append(s)
            print(f"  WARN: no psml file found for {s!r} in {lib}",
                  file=sys.stderr)
            continue
        dst = dest_dir / f"{s}.psml"
        if src.resolve() == dst.resolve():
            print(f"  ok:   {dst.name} (already present)", file=sys.stderr)
            continue
        shutil.copyfile(src, dst)
        print(f"  ok:   {src} -> {dst}", file=sys.stderr)
    return missing


# --------------------------------------------------------------------- #
#  FDF emitter                                                          #
# --------------------------------------------------------------------- #


# Recommended per-side vacuum (Angstrom) for an isolated molecule so its periodic
# images don't interact: >= 25 A for a charged system (image-image Coulomb decays
# slowly), a smaller floor for a neutral one (density/wavefunction overlap).
_VACUUM_MIN_NEUTRAL = 8.0
_VACUUM_MIN_CHARGED = 25.0


def _warn_insufficient_vacuum(struct: Structure, charge: int) -> None:
    """WARN (never mutate) when an ISOLATED axis carries too little vacuum for a
    clean SIESTA calc.  Skipped for periodic / transport axes -- a device or
    crystal cell sets the box there, not the vacuum.  Geometry is left untouched;
    the structure is the single source of truth (structure-periodicity.md)."""
    import warnings
    min_vac = _VACUUM_MIN_CHARGED if charge != 0 else _VACUUM_MIN_NEUTRAL
    kinds = struct.axis_kind or ("isolated", "isolated", "isolated")
    thin = [(i, float(struct.vacuum[i])) for i, k in enumerate(kinds)
            if k == "isolated" and float(struct.vacuum[i]) < min_vac]
    if not thin:
        return
    where = ", ".join(f"axis {i} (vacuum={v:g} A)" for i, v in thin)
    warnings.warn(
        f"Thin vacuum on an isolated system: {where}.  Recommended >= {min_vac:g} "
        f"A per side ({'charged' if charge else 'neutral'}) so the molecule's "
        f"periodic images don't interact.  Set 'vacuum' on the structure "
        f"(Modify -> Cell tab).  The FDF is emitted with the structure's geometry "
        f"unchanged.",
        RuntimeWarning, stacklevel=3)


def render_fdf(struct: Structure, config: Optional["SiestaConfig"] = None,
               *, cell: Optional[np.ndarray] = None) -> str:
    """Format a Structure as SIESTA .fdf text.

    If ``cell`` is None (default), the vacuum cell is derived from the STRUCTURE
    -- ``struct.resolve_cell()`` (isolated axes = ``bbox + 2*vacuum``) -- and the
    atoms are translated by ``-struct.resolve_cell_origin()`` so the molecule sits
    centred with ``vacuum`` of clearance on every isolated face.  Vacuum comes
    with the structure (``Structure.vacuum``, per-side gap), the single source of
    truth for lattice/vacuum (structure-periodicity.md); there is no cell_padding.
    A thin-vacuum isolated system is WARNED about (never mutated).

    Pass an explicit ``(3, 3) cell`` (Angstrom, row vectors) to override -- in that
    case atom coordinates are passed through unchanged, since a user-supplied cell
    typically goes with a known atom frame (e.g. crystallographic positions).
    """
    cfg = config or SiestaConfig()
    species = (list(cfg.species_order) if cfg.species_order
               else _detect_species(struct.elements))
    species_index = {s: i + 1 for i, s in enumerate(species)}

    # ---------- net charge: explicit override or auto-detect ----------
    # The rule itself lives in molbuilder.chemistry.resolve_net_charge
    # (shared with the PySCF generator); here we only also track a
    # user-facing comment label so the emitted FDF says WHY the
    # NetCharge value was picked.
    from ..chemistry import resolve_net_charge
    auto_charge = resolve_net_charge(struct, cfg.net_charge)
    charge_source = ("user-specified" if cfg.net_charge is not None
                     else "auto (phosphate protonation)")

    # Validate every element has a species index
    for el in struct.elements:
        if el not in species_index:
            raise ValueError(
                f"Atom element {el!r} not in --species-order "
                f"{list(species_index)!r}"
            )

    # Cell + atom positioning.
    #
    # Two cases:
    #
    #   (1) No cell provided -> derive the vacuum box from the STRUCTURE.
    #       struct.resolve_cell() sizes isolated axes as (extent + 2*vacuum);
    #       translate atoms by -struct.resolve_cell_origin() so the molecule sits
    #       centred with `vacuum` clearance on every isolated face -- atoms fall
    #       in [vacuum, size - vacuum], always inside the cell, no wrapping.
    #       Vacuum is the structure's, not a config knob (structure-periodicity.md).
    #
    #   (2) Cell provided -> periodic system (slab, crystal, junction).
    #       Trust the cell, but check whether atoms fall inside.  If
    #       any are outside [0, 1) in fractional coordinates and
    #       `wrap_into_cell` is True (default), fold them back via
    #       fractional arithmetic.  This is what every PBC-aware
    #       structure tool does (ASE's `wrap`, VASP's POSCAR Direct,
    #       3DNA's fiber output, etc.) and avoids surprises in the
    #       SIESTA mesh and in the post-relaxation visualisation.
    positions = np.asarray(struct.positions, dtype=float)
    if cell is None:
        # Derive the vacuum box from the STRUCTURE -- the single source of truth
        # for lattice/vacuum (structure-periodicity.md).  resolve_cell() sizes
        # isolated axes as bbox + 2*vacuum; resolve_cell_origin() is the box's low
        # corner (bbox_min - vacuum), so translating atoms by -origin centres the
        # molecule with `vacuum` of clearance on every isolated face (exactly what
        # the Modify-tab display shows).  There is no more cfg.cell_padding /
        # center_in_vacuum -- vacuum comes with the structure.
        cell = struct.resolve_cell()
        if cell is None:
            raise ValueError(
                "cannot derive a vacuum cell: the structure is empty")
        cell = np.asarray(cell, dtype=float).reshape(3, 3)
        # A flat / linear molecule (water, benzene, a DNA base) has a thin axis
        # with zero extent; with vacuum = 0 the box has no thickness there -> a
        # zero-volume cell SIESTA cannot run.  Fail with a clear, actionable
        # message (we never invent a vacuum -- the structure is the source of
        # truth) rather than the generic "degenerate cell" validation error.
        if abs(float(np.linalg.det(cell))) < 1e-6:
            raise ValueError(
                "the vacuum cell derived from the structure is degenerate (zero "
                "volume): a flat or linear molecule has a thin axis where "
                "vacuum = 0, so the box has no thickness there.  Set 'vacuum' > 0 "
                "on the structure (Modify -> Cell tab) so SIESTA has a real box "
                "-- the geometry is not changed for you.")
        origin = struct.resolve_cell_origin()
        if origin is not None:
            positions = positions - np.asarray(origin, dtype=float)
        # WARN (never mutate): too little vacuum on an isolated axis lets the
        # molecule's periodic images interact.  The user sets vacuum in the
        # structure (Modify -> Cell); we only flag it.
        _warn_insufficient_vacuum(struct, auto_charge)
        sizes = np.diag(cell)
        cell_note = (
            f"# (vacuum cell derived from the structure: "
            f"{sizes[0]:.2f} x {sizes[1]:.2f} x {sizes[2]:.2f} A; "
            f"vacuum = {tuple(round(float(v), 2) for v in struct.vacuum)} A/side; "
            f"atoms centred)"
        )
    else:
        cell = np.asarray(cell, dtype=float).reshape(3, 3)
        # Sanity: cell must be physically large enough to contain the
        # atoms.  The check is ``vol < n_atoms * 1.0 A^3`` -- 1 A^3
        # per atom is well below any real atom's van der Waals volume
        # (H ~ 7 A^3, C ~ 20 A^3), so it only fires on degenerate
        # cells: coplanar lattice vectors (vol = 0), a typo zeroing
        # one vector, or a unit confusion that produces sub-Angstrom
        # vectors (e.g., user wrote ``0.5`` meaning 0.5 nm = 5 A, but
        # the call passed it as A: vol = 0.5^3 = 0.125 A^3, fails for
        # any non-empty molecule).  Pre-2026-05-28 the threshold was
        # a flat ``vol < 1.0 A^3`` which only caught the most extreme
        # cases; the per-atom floor scales with the molecule.  The
        # check does NOT false-positive on legitimate dense cells:
        # compressed Fe at 100 GPa is ~10 A^3 per atom, 10x above the
        # threshold; even hard-sphere close-packed C is ~6 A^3 per
        # atom.
        vol = abs(float(np.linalg.det(cell)))
        n = max(1, len(positions))
        if vol < n * 1.0:
            raise ValueError(
                f"Provided cell has volume {vol:.3f} A^3, which is "
                f"below the minimum physical volume for {n} atom(s) "
                f"({float(n):.1f} A^3, = 1 A^3 per atom).  Likely "
                f"causes: lattice vectors in the wrong unit (nm vs "
                f"A), coplanar vectors, or a typo.  Inspect the "
                f"lattice vectors."
            )
        if cfg.wrap_into_cell:
            positions, n_wrapped = _wrap_into_cell(positions, cell)
            cell_note = (
                "# (using user-supplied lattice;"
                + (f" {n_wrapped} atom(s) wrapped into the unit cell)"
                   if n_wrapped else " all atoms already inside the cell)")
            )
        else:
            cell_note = "# (using user-supplied lattice; wrap_into_cell=False)"

    # ---------- pre-emission validation ----------
    # By now `cell` and `positions` are final; run the validation pass
    # before any FDF text is generated so error-severity issues block
    # emission cleanly.  Warnings print to stderr but the run proceeds.
    # See molbuilder.validation and docs/design.md for the check list.
    from ..validation import validate, report
    # The validation_struct mirrors the input struct but uses the FINAL
    # post-positioning ``positions`` array (so geometry-based checks
    # see what SIESTA will actually read).  CRITICAL: must carry the
    # transport metadata (``frozen_atoms`` + ``regions``) through, or
    # the validator's ``_check_frozen_atoms_consumed`` sees an empty
    # frozen list and never fires its "N atoms held fixed" / "won't
    # honor" issues -- the contract carrier silently dropped between
    # the Build endpoint that loaded the sidecar and the validator that
    # was supposed to consume it.  Caught by the 2026-05-26 review.
    validation_struct = Structure(
        elements      = list(struct.elements),
        positions     = positions,
        atom_names    = list(struct.atom_names),
        residue_ids   = list(struct.residue_ids),
        residue_names = list(struct.residue_names),
        chain_ids     = list(struct.chain_ids),
        title         = struct.title,
        frozen_atoms  = list(getattr(struct, "frozen_atoms", []) or []),
        # struct.regions is Dict[str, List[int]] per Structure's
        # declaration; the previous list-comprehension assumed an
        # iterable of lists and crashed at __post_init__ when the
        # dict was non-empty (caught by task #303's Pattern-B test).
        regions       = {
            k: list(v)
            for k, v in (getattr(struct, "regions", {}) or {}).items()
        },
    )
    report(validate(validation_struct, cfg, cell=cell))

    out: List[str] = []

    # job-layout v1 hint -- the basename ``cfg.system_label`` is the
    # protocol's "job name"; every output / restart file shares it.
    # Suggest the canonical ``mpirun`` invocation so the user redirects
    # stdout to ``<basename>.out`` (the Watch tab's discovery chain
    # also looks for that filename).  See docs/protocols/job-layout.md.
    #
    # Stage-aware filenames: when ``cfg.stage`` is set (1/2/3), the
    # FDF + stdout-redirect filenames pick up the ``-stage<N>`` suffix
    # so a coarse->medium->tight workflow produces
    # ``<basename>-stage1.fdf`` / ``…stage2…`` / ``…stage3…`` in one
    # directory.  The SystemLabel itself stays unsuffixed (so SIESTA's
    # .XV / .DM / .CG restart files transfer cleanly between stages).
    from ..trajectory_log.format import molwatch_log_basename
    if cfg.stage is not None:
        _stage_suffix = f"-stage{int(cfg.stage)}"
    else:
        _stage_suffix = ""
    _fdf_name  = f"{cfg.system_label}{_stage_suffix}.fdf"
    _out_name  = f"{cfg.system_label}{_stage_suffix}.out"
    _mw_name   = molwatch_log_basename(cfg.system_label, cfg.stage)
    if cfg.verbose_comments:
        out.append("# === Run with (job-layout v1) ===")
        out.append(
            "# Run from this directory -- all outputs share the "
            "SystemLabel basename below.")
        out.append(f"#     mpirun -np 4 siesta < {_fdf_name} > {_out_name}")
        if cfg.stage is not None:
            out.append(
                f"# Stage {int(cfg.stage)} of a staged relaxation; "
                "SIESTA reads .XV / .DM from the previous stage")
            out.append(
                "# (same SystemLabel, same directory).  See the Watch "
                "tab's 'Staged relaxation workflow' panel.")
        out.append(
            "# Watch the run live: open the Watch tab and point it "
            "at this directory")
        out.append(f"# (the loader resolves it to {_mw_name}).")
        # 2026-06-12: SIESTA 5.x emits an unconditional WARNING about
        # ``BASIS_ENTHALPY`` / ``BASIS_HARRIS_ENTHALPY`` being
        # deprecated.  The warning is INFORMATIONAL — the data is
        # also written to ``<SystemLabel>.BASIS_ENTHALPY`` (the new,
        # supported filename), which post-processing scripts should
        # consume.  There's no fdf flag to suppress the warning in
        # SIESTA 5.4.2; future versions will simply drop the legacy
        # unprefixed file.  Recording the note here so users who see
        # the warning don't worry their run is broken.
        out.append(
            "# Note: SIESTA's 'BASIS_ENTHALPY ... deprecated' WARNING "
            "in the output is harmless")
        out.append(
            f"# — read {cfg.system_label}{_stage_suffix}.BASIS_ENTHALPY "
            "in any post-processing.")
        out.append("")

    # Runtime-hint header.  Same shape as molwatch logs use
    # (``# runtime.<key>: <value>``) so the SIESTA parser can read
    # the user's configured caps back out of the .fdf at /results
    # load time.  These are HINTS the wrapper turns into env vars +
    # ulimits; SIESTA itself ignores comment lines.  Per the
    # cross-cutting "every script declares what it wanted" rule.
    if cfg.omp_threads is not None:
        out.append(f"# runtime.omp_threads_requested: {int(cfg.omp_threads)}")
    if cfg.max_memory_mb is not None:
        out.append(f"# runtime.max_memory_mb: {int(cfg.max_memory_mb)}")
    if (cfg.omp_threads is not None) or (cfg.max_memory_mb is not None):
        out.append("")

    # 2026-05-27: SystemName + SystemLabel both driven by
    # ``system_label`` -- the dataclass dropped ``system_name`` after
    # the web UI's one-field design proved the dup field only ever
    # caused divergence bugs (output filenames keyed on SystemLabel,
    # so the SystemName header always had to mirror it anyway).
    out.append(f"SystemName        {cfg.system_label}")
    out.append(f"SystemLabel       {cfg.system_label}")
    out.append("")
    out.append(f"NumberOfAtoms     {struct.n_atoms}")
    out.append(f"NumberOfSpecies   {len(species)}")
    out.append("")

    # Lattice
    out.append("# --- Lattice ---")
    if cell_note:
        out.append(cell_note.rstrip())
    out.append("LatticeConstant 1.0 Ang")
    out.append("%block LatticeVectors")
    for v in cell:
        out.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    out.append("%endblock LatticeVectors")
    out.append("")

    # Species
    out.append("# --- Species ---")
    out.append("%block ChemicalSpeciesLabel")
    for i, s in enumerate(species):
        out.append(f"{i + 1} {atomic_numbers[s]} {s}")
    out.append("%endblock ChemicalSpeciesLabel")
    out.append("")

    # Coordinates
    out.append("# --- Atomic coordinates ---")
    out.append("AtomicCoordinatesFormat Ang")
    out.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, positions):
        out.append(f"{x:.10f} {y:.10f} {z:.10f} {species_index[el]}")
    out.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    out.append("")

    v = cfg.verbose_comments

    # --- Frozen atoms (geometry constraints) ---
    # Three-stage contract carrier: Structure.frozen_atoms is populated
    # from the /modify sidecar (0-based indices) and flows through to
    # SIESTA's %block Geometry.Constraints (1-based indices, native
    # keyword as of v5.4.2).  Without this block SIESTA's relaxer
    # moves every atom -- the user's "frozen" backbone silently drifts.
    #
    # Syntax (verified against SIESTA 5.4.2 binary strings + the
    # TransSIESTA "buffer atoms *MUST* be fixed" error path):
    #   %block Geometry.Constraints
    #   position N1 N2 ... NK     # individual 1-based indices
    #   %endblock Geometry.Constraints
    # Range form (``position from N1 to N2``) is also supported; we
    # emit the explicit-list form so the user can grep / edit by
    # index without having to mentally expand a range.
    frozen = list(getattr(struct, "frozen_atoms", []) or [])
    if frozen:
        if v: out += [
            "# %block Geometry.Constraints holds atom indices SIESTA's",
            "# relaxer must NOT move.  1-based indices (SIESTA convention)",
            "# converted from molbuilder's 0-based Structure.frozen_atoms.",
            "# Source: /modify sidecar (or Python API: struct.frozen_atoms).",
            "# Without this block SIESTA relaxes every atom.",
        ]
        out.append("%block Geometry.Constraints")
        # Emit one ``position`` line per chunk of up to 20 indices
        # (~80 chars) for readability.  SIESTA accepts arbitrarily
        # many ``position`` lines inside the block; chunking makes
        # the .fdf easy to grep + edit.
        from ..engine_atom_index import siesta_atom_index
        ids_1based = [siesta_atom_index(i) for i in frozen]
        chunk = 20
        for i in range(0, len(ids_1based), chunk):
            segment = ids_1based[i:i + chunk]
            out.append("position " + " ".join(str(x) for x in segment))
        out.append("%endblock Geometry.Constraints")
        out.append("")

    # Extensible annotation channels (atom-annotations.md § 4): emit fdf for
    # any Structure.annotations channel that carries a REGISTERED fdf
    # strategy.  No registered strategies / no annotations -> no-op (the
    # frozen/region built-ins above are untouched).
    from ..annotations_fdf import emit_channels as _emit_channels
    _channel_lines = _emit_channels(struct)
    if _channel_lines:
        out += _channel_lines
        out.append("")

    # Basis & grid
    out.append("# --- Basis & grid ---")
    if v: out += [
        "# MeshCutoff: real-space integration grid (Ry).  Sets the",
        "# spacing of the 3D mesh SIESTA uses for Hartree + XC",
        "# potentials, via the plane-wave-equivalent kinetic-energy",
        "# cutoff.  Per-tier:",
        "#   150     screening (sanity-check only)",
        "#   200-250 loose preopt",
        "#   350     publishable (forces stable to < 0.01 eV/Ang on",
        "#           organic + Au systems)",
        "#   500+    tight / vibrational (egg-box noise below 0.001",
        "#           eV/Ang; 600 for first-row elements)",
        "# Below 150 Ry the forces / energies are noticeably wrong on",
        "# organic + biomolecule systems.  Test by varying +-50 Ry.",
        "# See docs/engines/tuning.md sect. 2.6.",
    ]
    out.append(f"MeshCutoff {cfg.mesh_cutoff} Ry")
    if v: out += [
        "",
        "# PAO.BasisSize: orbital basis quality (cheap -> expensive)",
        "#   SZ    minimal -- screening only",
        "#   SZP   single-zeta + polarization",
        "#   DZ    double-zeta",
        "#   DZP   double-zeta + polarization  (recommended for production)",
        "#   TZP   triple-zeta + polarization  (accurate, ~2x slower)",
    ]
    out.append(f"PAO.BasisSize {cfg.basis_size}")
    if v: out += [
        "",
        "# PAO.EnergyShift: how diffuse the PAO orbitals are (Ry).",
        "# Range 0.001 - 0.05.  Smaller = more diffuse + accurate + slower.",
        "#   0.02 Ry      typical production value",
        "#   0.001 Ry     accuracy-critical (band gaps, weak interactions)",
        "#   0.05 Ry      fast screening only",
    ]
    out.append(f"PAO.EnergyShift {cfg.pao_energy_shift} Ry")
    out.append("")

    # XC
    out.append("# --- Exchange-correlation ---")
    if v: out += [
        "# XC.functional: GGA (recommended for most systems),",
        "#                LDA (faster, underestimates band gaps),",
        "#                VDW (dispersion-dominated systems),",
        "#                HYB (hybrid -- much more expensive).",
        "# XC.authors:    PBE (standard GGA),  BLYP, PW92 (LDA),",
        "#                DRSLL / KBM (vdW),  HSE06 (hybrid).",
    ]
    out.append(f"XC.functional {cfg.xc_functional}")
    out.append(f"XC.authors    {cfg.xc_authors}")
    out.append("")

    # Dispersion-correction template for non-vdW XC (gap #3).
    #
    # Non-dispersive XC (PBE / BLYP / LDA / hybrids without explicit
    # dispersion) systematically under-binds vdW-dominated systems --
    # DNA stacking by 5-10 kcal/mol per pair, peptide folding, molecular
    # crystals' lattice constants too long by 0.1-0.3 A, surface
    # adsorption energies off by an order of magnitude.  PBE on a
    # biomolecule looks converged but the chemistry is wrong.
    #
    # We emit a COMMENTED template (don't auto-impose chemistry) so
    # the user sees the option exists and can uncomment when it
    # matters for their system.  Skipped when XC.functional is
    # already a vdW-aware functional (XC.functional VDW + DRSLL /
    # KBM / LMKLL): the non-local correlation lives in the functional
    # itself, and an additional MM.Potentials block would double-count.
    # Strip + upper so leading/trailing whitespace doesn't make a
    # vdW-aware functional miss the gate (SP2).
    if cfg.xc_functional.strip().upper() != "VDW":
        out += _emit_dispersion_template(cfg.xc_authors, v)
        out.append("")

    # SCF
    out.append("# --- SCF ---")
    if v: out += [
        "# SolutionMethod:  diagon       standard diagonalisation, O(N^3)",
        "#                  OMM          order-N, for systems > 500 atoms",
        "#                  transiesta   non-equilibrium transport",
    ]
    out.append(f"SolutionMethod    {cfg.solution_method}")

    if v: out += [
        "",
        "# DM.MixingWeight: density-matrix mixing weight (0.001 - 0.5).",
        "#   Smaller = more conservative, stable, slower.",
        "#   Larger  = aggressive, may oscillate.",
        "# Tuning hints:",
        "#   - SCF oscillating?     reduce to 0.005",
        "#   - SCF stalled?         increase or add Pulay history",
        "#   - Metals:              0.005 - 0.02",
        "#   - Insulators:          0.05 - 0.10 is often fine",
    ]
    out.append(f"DM.MixingWeight   {cfg.mixing_weight}")

    if v: out += [
        "",
        "# DM.NumberPulay: # of past SCF iterations kept for Pulay mixing.",
        "# Range 2-10.  More = better convergence + more memory.",
        "#   3      fine for most cases",
        "#   5-8    hard cases (metals, magnetic systems)",
    ]
    out.append(f"DM.NumberPulay    {cfg.pulay_history}")

    if v: out += [
        "",
        "# DM.Tolerance: density-matrix element convergence threshold",
        "# for the inner SCF loop.  Forces are derived from the",
        "# converged density -- sloppy SCF -> noisy forces -> optimiser",
        "# thrashes.  Per-tier:",
        "#   1e-3    screening (sanity-check only)",
        "#   1e-4    loose preopt / publishable",
        "#   1e-5    tight (vib / IR / accurate forces)",
        "#   1e-6    very-tight (band structure, phonons)",
        "# Rule of thumb: keep SCF tol ~10x tighter than the force-",
        "# precision target you want at convergence.  See",
        "# docs/engines/tuning.md sect. 2.5.",
    ]
    out.append(f"DM.Tolerance      {cfg.dm_tolerance:.0e}")

    if v: out += [
        "",
        "# DM.Energy.Tolerance: redundant energy-based SCF check (eV).",
        "# Catches the rare case where DM is converged but energy keeps",
        "# drifting -- usually triggered by ill-conditioned mixing.",
    ]
    out.append(f"DM.Energy.Tolerance {cfg.dm_energy_tolerance:.0e} eV")

    if v: out += [
        "",
        "# MaxSCFIterations: SCF iteration cap.  500 is generous for the",
        "# first geometry; well-mixed systems converge in 30-100.",
    ]
    out.append(f"MaxSCFIterations  {cfg.max_scf_iter}")

    if v: out += [
        "",
        "# ElectronicTemperature: Fermi-Dirac smearing temperature.",
        "#   25 K     2 meV  -- molecular / cold properties",
        "#   300 K    25 meV -- room temperature default (ok for most)",
        "#   1000-2000 K     metals; helps SCF convergence",
        "#   < 100 K  for very accurate band-edge properties",
    ]
    out.append(f"ElectronicTemperature {cfg.electronic_temperature} K")

    if cfg.use_save_dm:
        if v: out += [
            "",
            "# DM.UseSaveDM: read .DM from previous run if present.  Free",
            "# warm-start; SIESTA silently ignores if no file exists.",
        ]
        out.append("DM.UseSaveDM      .true.")

    # ---- Spin polarisation ---------------------------------------
    # Targeted SIESTA version range: 4.1 -- 5.x.
    #
    # v5 introduced a unified `Spin <option>` keyword that supersedes
    # the older `SpinPolarized true` form.  Recognised options
    # include `non-polarized`, `polarized`, `non-collinear`, `spin-orbit`.
    # The single-line `Spin polarized` form is what current docs
    # recommend; v4 back-compat keepers still accept `SpinPolarized
    # true` but the v5 manual marks it deprecated (gap #2).
    #
    # The total-spin pin requires TWO lines, not one (gap #1):
    #   `Spin.Fix true`           -- enable the constraint (otherwise
    #                                Spin.Total below is silently ignored)
    #   `Spin.Total <value>`      -- target total spin moment in mu_B
    # Pre-fix the generator emitted a single `SpinTotal <v>` token
    # which is NOT a real SIESTA keyword -- the parser silently
    # ignored it and the user got the spin-unrestricted ground state
    # despite asking for a constrained multiplicity.
    if cfg.spin_polarized:
        if v: out += [
            "",
            "# Spin polarized: open-shell DFT (collinear).  Required for",
            "# any system with unpaired electrons.  SIESTA's default is",
            "# closed-shell -- omitting this for a radical / transition-",
            "# metal / triplet system gives the wrong electronic state.",
            "#",
            "# We use the v4 ``SpinPolarized .true.`` form (not the v5",
            "# ``Spin polarized`` single-line form) on purpose: as of",
            "# SIESTA 5.4.2 (verified 2026-05-24) the v5 unified parser",
            "# DOES NOT read the auxiliary ``Spin.Fix`` / ``Spin.Total``",
            "# keys below -- so open-shell metals like Fe abort at",
            "# initial-DM construction with ``propor: ERROR: IMAX = 0``",
            "# because no spin target reaches the constructor.  v4",
            "# syntax is marked deprecated in the manual but is still",
            "# fully honored AND triggers the auxiliary spin reads.",
        ]
        out.append("SpinPolarized .true.")
        if cfg.spin_total is not None:
            if v: out += [
                "# Spin.Fix + Spin.Total: target total spin moment in mu_B",
                "# (= number of unpaired electrons).  Spin.Fix true MUST",
                "# accompany Spin.Total or the constraint is silently ignored.",
                "# Helps SIESTA's initial guess converge to the right",
                "# multiplicity; without it SIESTA may settle into a wrong",
                "# spin state.",
            ]
            if v and cfg.spin_total == 0.0:
                # SP-A: a constrained singlet ON TOP of open-shell DFT
                # is unusual -- the cheaper path is spin_polarized=False
                # (spin-restricted Kohn-Sham).  Surface this so a user
                # who landed here by accident sees the contradiction.
                out += [
                    "# NOTE: spin_total = 0.0 with spin_polarized=True asks",
                    "# for a constrained singlet via open-shell DFT (broken-",
                    "# symmetry capable).  Most users wanting a singlet are",
                    "# better served by spin_polarized=False -- the",
                    "# spin-restricted formalism is cheaper and gives the",
                    "# same answer.  Keep this if you specifically want",
                    "# anti-ferromagnetic / broken-symmetry singlet.",
                ]
            out.append("Spin.Fix          .true.")
            out.append(f"Spin.Total        {cfg.spin_total}")

        # Spin-state-sweep template when an open-shell metal is in
        # the structure.  The "right" spin state for a transition
        # metal complex isn't computable from element identity alone
        # (depends on coordination chemistry + axial ligand field);
        # the practical resolution is to run with each plausible
        # spin and pick the lowest-energy convergence.
        from ..chemistry import detect_open_shell_metals
        _metals = detect_open_shell_metals(struct)
        if v and _metals:
            out += [
                "",
                f"# --- Spin-state sweep template ({', '.join(_metals)}) ---",
                "# The right spin state for an open-shell metal complex",
                "# depends on the axial ligand field, not just element",
                "# identity.  Standard practice: run with each plausible",
                "# Spin.Total, pick the lowest-energy convergence.",
                "#",
                "# Fe(II) candidates (Z=26, d6):",
                "#   Spin.Total 0.0   low-spin   (CO / CN heme, strong-field)",
                "#   Spin.Total 2.0   intermediate (4-coord Fe-porphyrin, FeTPP)",
                "#   Spin.Total 4.0   high-spin  (deoxy-heme, bis-thiolate)",
                "# Fe(III) candidates (d5):",
                "#   Spin.Total 1.0   low-spin   (bis-imidazole)",
                "#   Spin.Total 3.0   intermediate (cyt P450)",
                "#   Spin.Total 5.0   high-spin  (met-myoglobin)",
                "#",
                "# Workflow: rename SystemLabel per run (so .XV / .DM don't",
                "# stomp), run each, compare the converged E_KS values.",
                "# Verify the winning state against Mossbauer / EPR / UV-Vis",
                "# data; calc-energy minimum and experimental ground state",
                "# don't always agree for borderline cases (spin crossover).",
            ]

    # ---- NetCharge -----------------------------------------------
    # Either user-specified (cfg.net_charge != None) or auto-detected
    # from phosphate protonation state.  SIESTA defaults to neutral and
    # silently adds compensating electrons; we MUST set NetCharge for
    # any non-zero charge or the electronic structure is wrong.
    if auto_charge != 0:
        if v: out += [
            "",
            f"# NetCharge: {auto_charge:+d} ({charge_source}).",
            "# Note: SIESTA adds a uniform compensating background charge",
            "# for periodic-cell consistency.  For vacuum calcs of charged",
            "# molecules set the structure's vacuum >= 25 A per side (Modify ->",
            "# Cell) to suppress image-image Coulomb interactions; molbuilder",
            "# warns if it's thinner.  To make a neutral system instead,",
            "# either build with protonate_phosphates=True or pass a",
            "# Config(net_charge=0) override.",
            "#",
            "# IMPORTANT: residual image-charge artefact (Makov-Payne).",
            "# Padding alone does NOT remove the leading image-charge",
            "# error.  With q != 0 in a finite supercell SIESTA's total",
            "# energy carries a systematic bias from the molecule's",
            "# interaction with its own periodic replicas, going as",
            "#     E_bias ~ q^2 * alpha / (2 * L * eps_r)",
            "# where alpha is the Madelung constant (~2.84 for simple",
            "# cubic), L = V^(1/3) is the effective supercell side,",
            "# and eps_r is the relative permittivity of the medium",
            "# (1 in vacuum).  For q = +/- 1 at L ~ 15-25 A this is",
            "# 0.5-1.5 eV -- much larger than chemical accuracy.",
            "#",
            "# molbuilder emits ``makov_payne_correction.py`` next to",
            "# this FDF.  After SIESTA finishes, run:",
            "#     python3 makov_payne_correction.py",
            "# The script reads the .out, extracts the converged total",
            "# energy and the final lattice vectors, computes",
            "# DeltaE_MP, and prints the corrected total in eV.  Pass",
            "# --epsilon <eps_r> if your medium isn't vacuum.",
            "# See Makov & Payne, Phys. Rev. B 51, 4014 (1995).",
        ]
        out.append(f"NetCharge       {auto_charge:+d}")
    out.append("")

    # k-grid
    kx, ky, kz = cfg.kgrid
    out.append(f"# --- k-points ({kx}x{ky}x{kz}) ---")
    if v: out += [
        "# Monkhorst-Pack mesh.  Cost scales linearly with # of k-points.",
        "#   1x1x1               vacuum / molecule (only Gamma matters)",
        "#   4x4x4 to 8x8x8      periodic 3D crystals",
        "#   kx x ky x 1         2D slabs (no k along the vacuum direction)",
        "# Convergence test: rerun with 1.5x density on each axis -> total",
        "# energy should change < 1 meV/atom.",
    ]
    out.append("%block kgrid_Monkhorst_Pack")
    out.append(f"{kx} 0 0 0.0")
    out.append(f"0 {ky} 0 0.0")
    out.append(f"0 0 {kz} 0.0")
    out.append("%endblock kgrid_Monkhorst_Pack")
    out.append("")

    # ---- Parallel execution (MPI) -------------------------------
    # Always emit explicit values so the FDF behaves the same across
    # SIESTA versions and across system sizes.  Without an explicit
    # `BlockSize`, SIESTA auto-picks one as `ceil(Norb / Nrank)`,
    # which is fine for distributing the orbital matrix but is
    # *also* used elsewhere in the pipeline -- including a per-atom
    # distribution step right before mesh setup.  When the system
    # is small (e.g., a 14-atom molecule -> Norb~134, auto-picked
    # BlockSize=34 with 4 MPI ranks), the auto value is much larger
    # than the atom count, and that earlier step bails with
    #     propor: ERROR: IMAX = 0
    # An explicit smaller BlockSize keeps every distribution step
    # well-conditioned regardless of system size.
    out.append("# --- Parallel execution (MPI) ---")
    if v: out += [
        "# These settings matter only with `mpirun -np N siesta`",
        "# (single-rank runs ignore them).",
        "#",
        "# BlockSize: ScaLAPACK orbital-distribution block.  Affects",
        "# cache efficiency for the diagonaliser; does NOT fix the",
        "# propor IMAX=0 crash (an earlier claim was wrong -- an",
        "# empirical sweep confirmed BlockSize = 1, 2, 4 all crash",
        "# at the same mpi_np; propor is a matel_table proportionality",
        "# check, not a BLACS distribution check).  If your run dies",
        "# at startup with ``propor: ERROR: IMAX = 0``: lower mpi_np",
        "# via the wrapper's ``-np`` flag, not BlockSize.  Larger",
        "# BlockSize gives marginally better diag throughput on big",
        "# systems (>1000 atoms / >=16 ranks); for smaller jobs the",
        "# default is fine.  Override only for hand-tuned perf work.",
        "#",
        "# Diag.ParallelOverK: parallelise the diagonaliser over",
        "# k-points (.true.) or over orbitals (.false.).  Auto-",
        "# selected here from the kgrid above: .false. for 1x1x1",
        "# (molecule / vacuum), .true. for multi-k periodic runs.",
    ]
    if cfg.parallel_block_size is None:
        # GPU and CPU have fundamentally different BlockSize optima
        # (GPU targets the ELPA-CUDA 64-block sweet spot; CPU keeps
        # the historical n_atoms/mpi_np formula).  Branch the picker
        # via ``gpu_mode`` rather than hand-rolling it here.
        block_size = _auto_block_size(
            struct.n_atoms, cfg.mpi_np, gpu_mode=bool(cfg.enable_gpu),
        )
    else:
        # User-set BlockSize is honored verbatim.  Earlier code
        # auto-downgraded when ``BlockSize * mpi_np > n_atoms`` on
        # the theory that it caused propor IMAX=0; empirical sweep
        # (2026-05-28) disproved that theory -- propor is a
        # matel_table issue, not a BlockSize issue.  The auto-
        # downgrade is gone; user's value passes through.
        block_size = int(cfg.parallel_block_size)
    out.append(f"BlockSize          {block_size}")
    if cfg.parallel_over_k is None:
        over_k = (kx, ky, kz) != (1, 1, 1)
    else:
        over_k = bool(cfg.parallel_over_k)
    out.append(f"Diag.ParallelOverK {'.true.' if over_k else '.false.'}")
    # Diagonalizer (engines/siesta.md § 13).  The solver choice
    # (``diag_algorithm``) is INDEPENDENT of the GPU toggle; ELPA runs on
    # CPU and GPU alike, and ``enable_gpu`` only moves an ELPA solve onto
    # the GPU.
    #   * ScaLAPACK -> emit nothing (SIESTA's built-in Divide-and-Conquer).
    #   * ELPA-* -> emit ``Diag.Algorithm`` (required: Diag.ELPA.GPU alone
    #     is ignored without it, Src/diag_option.F90:213-225) AND
    #     ``Diag.ELPA.GPU .true./.false.``.  The explicit ``.false.`` for
    #     CPU-ELPA is load-bearing: the source ELPA defaults to the GPU
    #     codepath, so an omitted flag crashes a CPU run (Sol job 57852378).
    _algo = (cfg.diag_algorithm or "ScaLAPACK").strip()
    _is_elpa = _algo.upper().startswith("ELPA")
    if cfg.enable_gpu and not _is_elpa:
        raise ValueError(
            "enable_gpu requires an ELPA diagonalizer (diag_algorithm = "
            "ELPA-1STAGE or ELPA-2STAGE); GPU acceleration does not apply to "
            f"the {_algo} solver.  Pick an ELPA algorithm or turn GPU off "
            "(engines/siesta.md § 13).")
    if _is_elpa:
        out.append(f"Diag.Algorithm     {_algo}")
        out.append(f"Diag.ELPA.GPU      {'.true.' if cfg.enable_gpu else '.false.'}")
    out.append("")

    # Relaxation / dynamics.  In SIESTA 5.4.2 the step-count and
    # displacement-cap fdf keywords are UNIVERSAL across relax types
    # despite the CG-prefixed names -- ``MD.NumCGsteps`` and
    # ``MD.MaxCGDispl`` are recognized for CG, Broyden, AND FIRE.
    #
    # HISTORY: pre-2026-06-23, this branch emitted made-up per-
    # algorithm keywords (``MD.NumBroydenSteps``, ``MD.MaxDispl``)
    # which SIESTA 5.4.2 silently dropped -- with NO warning -- so a
    # Broyden / FIRE relaxation ran as a Single-point calculation.
    # The user surfaced the bug in TJ-BDT-Au111 when stage 2 (Broyden)
    # "finished in one step" with max-force 0.18 vs threshold 0.02 --
    # SIESTA never took a Broyden step at all.  See decision-log
    # 2026-06-23 in design.md for the full failure analysis.
    #
    # Empirical proof of the universal mapping (small H2 against
    # SIESTA 5.4.2 with ``MD.TypeOfRun Broyden`` + ``MD.NumCGsteps 5``
    # + ``MD.MaxCGDispl 0.1 Ang``):
    #   redata: Dynamics option        = Broyden coord. optimization
    #   redata: Maximum number of optimization moves = 5
    #   redata: Max atomic displ per move = 0.1000 Ang
    # Identical echo lines stage1 (CG) already produces in real jobs.
    #
    # Verlet / Nose (NVE / NVT dynamics, not relaxation) use distinct
    # step-control keywords -- ``MD.FinalTimeStep`` + the temperature
    # block.  They never reached this branch with the broken mapping
    # because no test ever ran them; today they're handled below too
    # for completeness, with the universal MD.NumCGsteps NOT emitted
    # (it would be a no-op + visual noise in the fdf).
    if cfg.relax_type and cfg.relax_type.lower() != "none":
        relax_kind = cfg.relax_type.strip().upper()
        is_md = relax_kind in ("VERLET", "NOSE")
        # Universal step-count keyword for CG / Broyden / FIRE.
        # Verlet / Nose use MD.FinalTimeStep instead (the loop is
        # time-based, not step-count-based) -- handled below.
        _STEP_KW = {
            "CG":      "MD.NumCGsteps",
            "BROYDEN": "MD.NumCGsteps",
            "FIRE":    "MD.NumCGsteps",
            "VERLET":  "MD.FinalTimeStep",
            "NOSE":    "MD.FinalTimeStep",
        }
        step_kw = _STEP_KW.get(relax_kind, "MD.NumCGsteps")
        # Universal displacement-cap keyword for CG / Broyden / FIRE;
        # Verlet / Nose have no per-step displacement cap (forces +
        # masses drive the timestep instead).
        displ_kw = "MD.MaxCGDispl" if not is_md else None

        out.append("# --- Geometry optimisation / dynamics ---")
        if v: out += [
            "# MD.TypeOfRun -- algorithm tier guidance:",
            "#   CG       Conjugate Gradients.  Best for loose warm-up",
            "#            stages (far from minimum, large forces).  No",
            "#            memory.  OSCILLATES near a minimum on stiff /",
            "#            coupled systems (metals, organic-on-metal,",
            "#            vdW stacks) -- if max-force fluctuates instead",
            "#            of descending, switch to Broyden.",
            "#   Broyden  Quasi-Newton (BFGS-like).  Best for publishable",
            "#            / tight stages, especially where CG oscillates.",
            "#            Memory: keeps ~5 history vectors.",
            "#   FIRE     Fast Inertial Relaxation Engine.  Robust on",
            "#            rough energy landscapes (random-built initial",
            "#            geometries).  Slower than Broyden near a minimum.",
            "#   Verlet   NVE molecular dynamics (NOT relax).",
            "#   Nose     Nose-Hoover NVT molecular dynamics (NOT relax).",
            "# Recipe: stage 1 CG -> stage 2 Broyden (refine).  See",
            "# docs/engines/tuning.md sect. 2.1 for full",
            "# algorithm comparison + citations.",
        ]
        out.append(f"MD.TypeOfRun {cfg.relax_type}")

        if v: out += [
            "",
            f"# {step_kw}: number of {'MD time' if is_md else 'relaxation'} steps."
            f"  Relaxation runs typically",
            "# converge in 30-150; 200+ is a safety cap.  MD runs scale",
            "# with the timescale you want sampled (steps * dt).",
        ]
        out.append(f"{step_kw} {cfg.relax_steps}")

        if not is_md:
            # Force-based convergence + displacement cap apply only to
            # the relaxation modes; SIESTA silently ignores them in
            # Verlet / Nose dynamics.
            if v: out += [
                "",
                "# MD.MaxForceTol: max-atomic-force convergence threshold.",
                "# Per-tier (eV/Ang) -- 2026-06-23 realignment:",
                "#   0.10     screening (sanity-check only)",
                "#   0.05     loose preopt",
                "#   0.04     publishable (Gaussian-OPT default, molecule + bulk)",
                "#   0.01     tight (CRYSTAL/SURFACE production -- VASP",
                "#            EDIFFG=-0.01 standard; safe for 100+ atom metals)",
                "#   0.001    very-tight (MOLECULE vib/IR/TS/NEB only --",
                "#            Gaussian GAU_TIGHT; DO NOT use on 100+ atom",
                "#            metal systems, chases SCF noise + never converges)",
                "# SIESTA only checks max force; geomeTRIC / Gaussian check 5",
                "# criteria.  See docs/engines/tuning.md sect. 2.3",
                "# for the cross-engine + system-type-aware tier framework.",
            ]
            out.append(f"MD.MaxForceTol {cfg.relax_force_tol} eV/Ang")
            if v: out += [
                "",
                f"# {displ_kw}: maximum atom displacement per optimiser step (Ang).",
                "# Hard ceiling that catches line-search over-shoot.",
                "# Per-tier (Ang) -- 2026-06-23 realignment:",
                "#   0.30     screening",
                "#   0.20     loose preopt (SIESTA default)",
                "#   0.05     publishable",
                "#   0.02     tight (crystal/surface production)",
                "#   0.01     very-tight (molecule vib/IR only)",
                "# Symptom of too-large cap: max-force oscillates instead of",
                "# descending (e.g. 0.09 -> 0.44 -> 0.13 -> 0.31 -> ...).",
                "# Halve the cap and continue.  See docs/engines/tuning.md",
                "# sect. 2.2 + sect. 2.3 design considerations.",
            ]
            out.append(f"{displ_kw} {cfg.relax_max_displ} Ang")
        else:
            # Verlet / Nose dynamics need an initial-velocity seed;
            # without MD.InitialTemperature SIESTA starts with zero
            # velocities, producing a steepest-descent-like trajectory
            # mislabelled as MD.  Nose-Hoover NVT also needs
            # MD.TargetTemperature for the thermostat target -- without
            # it SIESTA defaults the target to 0 K and the trajectory
            # cools monotonically (a quench mislabelled as NVT).  All
            # three values come from cfg fields so the user can tune
            # them without editing the generated FDF (S1).
            target_T = (cfg.md_target_temperature
                        if cfg.md_target_temperature is not None
                        else cfg.md_initial_temperature)
            if v: out += [
                "",
                "# MD.InitialTemperature: initial atomic-velocity seed (K).",
                "# Without this, SIESTA starts at 0 K -- not real dynamics.",
            ]
            out.append(f"MD.InitialTemperature {cfg.md_initial_temperature} K")
            if relax_kind == "NOSE":
                if v: out += [
                    "",
                    "# MD.TargetTemperature: Nose-Hoover NVT target (K).",
                    "# Required for the thermostat; without it SIESTA",
                    "# defaults the target to 0 K and the run quenches",
                    "# instead of equilibrating.",
                ]
                out.append(f"MD.TargetTemperature  {target_T} K")
            if v: out += [
                "",
                "# MD.LengthTimeStep: integration timestep (fs).",
                "# 1.0 fs is SIESTA's default and works for systems without H;",
                "# bonded H typically needs 0.5 fs for stable energy conservation.",
            ]
            out.append(f"MD.LengthTimeStep {cfg.md_length_timestep} fs")

        if cfg.use_save_cg or cfg.use_save_xv:
            if v: out += [
                "",
                "# MD.UseSaveCG / UseSaveXV: read .CG / .XV from previous run",
                "# if present.  Restart-friendly for long jobs.",
            ]
            if cfg.use_save_cg and not is_md:
                # MD.UseSaveCG is CG-only; Broyden / FIRE / dynamics modes
                # ignore it.  Only emit when meaningful.
                out.append("MD.UseSaveCG      .true.")
            if cfg.use_save_xv: out.append("MD.UseSaveXV      .true.")
        out.append("")

    # Output
    out.append("# --- Output ---")
    if v: out += [
        "# WriteForces      forces in .FA (required for relaxation)",
        "# WriteCoorStep    coords at every MD step in main .out",
        "# WriteCoorXmol    .xyz at every step (movie viewer)",
        "# WriteMDhistory   trajectory to .ANI (xcrysden / vmd / OVITO)",
        "# SaveHS           H + S matrices to .HSX (needed for TranSIESTA",
        "#                  electrode reuse + DOS post-processing).  In",
        "#                  SIESTA 5.4.2 the keyword is SaveHS; the older",
        "#                  WriteHS is silently dropped (no warning) so a",
        "#                  pre-2026-06-23 generator that emitted WriteHS",
        "#                  always got the default (SaveHS = T) regardless",
        "#                  of cfg.write_hs.  See decision-log 2026-06-23.",
    ]
    out.append(f"WriteForces        {'.true.' if cfg.write_forces else '.false.'}")
    out.append(f"WriteCoorStep      {'.true.' if cfg.write_coor_step else '.false.'}")
    out.append(f"WriteCoorXmol      {'.true.' if cfg.write_coor_xmol else '.false.'}")
    out.append(f"WriteMDhistory     {'.true.' if cfg.write_md_history else '.false.'}")
    # Always emit SaveHS so user choice (T or F) is explicit + auditable.
    # Pre-2026-06-23: only emitted when cfg.write_hs=True (and as the
    # wrong keyword ``WriteHS``).  The default-T behavior masked the bug
    # whenever the user wanted T anyway; the day someone sets
    # cfg.write_hs=False to skip the .HSX overhead, the override silently
    # did nothing.  Emit unconditionally now so the fdf-echo shows the
    # user's actual choice + this generator's intent.
    out.append(f"SaveHS             {'.true.' if cfg.write_hs else '.false.'}")

    # Troubleshooting block at the end (verbose mode only).  We only
    # emit the relaxation-specific tips when an MD block is actually
    # being written -- otherwise mentioning `MD.TypeOfRun` here would
    # surprise downstream code that scans the FDF.
    if v:
        out += [
            "",
            "# ============================================================",
            "# TROUBLESHOOTING / TUNING HINTS                              ",
            "# ============================================================",
            "#",
            "# SCF doesn't converge:",
            "#   * lower DM.MixingWeight to 0.005",
            "#   * increase DM.NumberPulay to 5-8",
            "#   * raise ElectronicTemperature to 1000-2000 K (metals)",
            "#   * verify all .psml pseudopotentials are in this directory",
            "#",
            "# Forces look noisy / break symmetry:",
            "#   * raise MeshCutoff to 400-500 Ry",
            "#   * tighten DM.Tolerance to 1e-6",
            "#   * smaller PAO.EnergyShift (0.005 Ry, more diffuse basis)",
            "#",
            "# Calculation is too slow:",
            "#   * try PAO.BasisSize SZP for screening runs",
            "#   * reduce MeshCutoff to 200 Ry  (forces ~0.05 eV/Ang noisier)",
            "#   * reduce k-grid (periodic systems)",
            "#   * SolutionMethod OMM for >500 atoms",
            "#",
            "# Energy fluctuates during SCF:",
            "#   * lower DM.MixingWeight to 0.005",
            "#   * raise DM.NumberPulay to 6",
            "#   * tighten DM.Energy.Tolerance to 1e-5 eV",
            "#",
            "# 'propor: ERROR: IMAX = 0' on parallel run:",
            "#   * caused by SIESTA's auto-picked BlockSize (which",
            "#     scales with Norb / Nrank) being too large for an",
            "#     unrelated per-atom distribution step earlier in",
            "#     the pipeline.  Setting BlockSize explicitly above",
            "#     overrides the auto-choice -- if it still fails,",
            "#     drop BlockSize to 4 or 1.",
            "#   * for 1x1x1 (Gamma) k-grids, also confirm that",
            "#     Diag.ParallelOverK is .false. (we set it above",
            "#     based on the kgrid).",
            "#   * single-rank `siesta` (no mpirun) is often fastest",
            "#     for molecules under ~50 atoms anyway, since",
            "#     ScaLAPACK overhead dominates.",
        ]
        if cfg.relax_type and cfg.relax_type.lower() != "none":
            out += [
                "#",
                "# Relaxation oscillates near minimum:",
                "#   * shrink MD.MaxCGDispl to 0.02 Ang",
                "#   * loosen MD.MaxForceTol to 0.04 eV/Ang",
                "#   * switch MD.TypeOfRun to Broyden (often robust on",
                "#     flat regions) or FIRE (better for >100 atoms)",
            ]

    # Post-processing hook (gap #6).  Commented templates for the
    # follow-up analyses users typically want after a successful
    # relaxation.  Default-disabled so the script's behaviour is
    # unchanged; uncomment + tune to enable.
    out.append("")
    out.append("# === Post-processing hook (commented templates) ===")
    if v:
        out += [
            "# Common follow-ups after a successful relaxation.  All",
            "# four are independent: enable any subset.  Each block",
            "# adds at most one extra (cheap) SCF or one parsing pass",
            "# over the saved DM, so the cost is negligible compared",
            "# to the optimisation that just ran.",
        ]
    out += [
        "#",
        "# 1. Mulliken population analysis (per-atom charge breakdown):",
        "# WriteMullikenPop    1     # 0=off, 1=atom, 2=atom+orbital",
        "#",
        "# 2. Band structure along high-symmetry path (set kgrid > 1):",
        "# %block BandLines",
        "#    1   0.0  0.0  0.0   \\Gamma",
        "#   30   0.5  0.0  0.0   X",
        "#   30   0.5  0.5  0.0   M",
        "#   30   0.0  0.0  0.0   \\Gamma",
        "# %endblock BandLines",
        "#",
        "# 3. Projected DOS (per-orbital DOS, energy window in eV):",
        "# %block ProjectedDensityOfStates",
        "#   -10.0  5.0  0.05  500  eV",
        "# %endblock ProjectedDensityOfStates",
        "#",
        "# 4. Charge-density grid (volumetric file for visualisation):",
        "# SaveRho             .true.",
        "# SaveDeltaRho        .true.",
        "# SaveElectrostaticPotential  .true.",
    ]
    # ----- Wrap engine body with script-contract blocks -----
    # See docs/protocols/script-contract.md.  Per-emission rules:
    #   - PROVENANCE: always emitted (cheap, always meaningful).
    #   - BENCH-MARKS: always emitted for .fdf.  The MD.NumCGsteps
    #     anchor is UNIVERSAL across CG / Broyden / FIRE (post
    #     2026-06-23 SIESTA keyword fix); the bench picks it up
    #     regardless of cfg.relax_type.
    #   - ATOM-METADATA: emit_atom_metadata returns None when both
    #     regions and frozen_atoms are empty -- per the contract's
    #     emission rule, absence is the honest signal.
    #   - USER-CUSTOM placeholder: empty in 2a; Step 2b adds the
    #     round-trip preservation of user edits.
    from .. import script_emit as _sc
    _provenance = _sc.emit_provenance(
        generator_version=_sc.molbuilder_git_sha(),
        generated_at=_sc.generated_at_now(),
        resolved_defaults={
            "enable_gpu": str(bool(cfg.enable_gpu)).lower(),
            "BlockSize": (
                f"auto -> {block_size}" if cfg.parallel_block_size is None
                else f"user-set -> {block_size}"
            ),
            "mpi_np": (
                "auto" if cfg.mpi_np is None else str(cfg.mpi_np)
            ),
            "omp_threads": (
                "auto" if cfg.omp_threads is None else str(cfg.omp_threads)
            ),
        },
    )
    _bench_marks = _sc.emit_bench_marks(
        metadata={
            "n_atoms":        struct.n_atoms,
            "n_orbitals_est": 10 * struct.n_atoms,
            "gpu_mode":       str(bool(cfg.enable_gpu)).lower(),
        },
        fields=_sc.SIESTA_BENCH_FIELDS,
        defaults={
            "BlockSize":         block_size,
            "MaxSCFIterations":  cfg.max_scf_iter,
            "MD.NumCGsteps":     cfg.relax_steps,
            "MeshCutoff":        cfg.mesh_cutoff,
        },
    )
    _atom_metadata = _sc.emit_atom_metadata(
        regions=dict(getattr(struct, "regions", {}) or {}),
        frozen_atoms=list(getattr(struct, "frozen_atoms", []) or []),
        annotations=dict(getattr(struct, "annotations", {}) or {}),
        n_atoms_total=int(struct.n_atoms),
        created_by="molbuilder render_fdf",
        created_at=_sc.generated_at_now(),
    )
    _engine_body = "\n".join(out)
    _user_custom = _sc.emit_user_custom_placeholder()
    _top_blocks = [_provenance, _bench_marks]
    if _atom_metadata is not None:
        _top_blocks.append(_atom_metadata)
    return (
        "\n\n".join(_top_blocks)
        + "\n\n"
        + _engine_body
        + "\n\n"
        + _user_custom
        + "\n"
    )


# --------------------------------------------------------------------- #
#  File -> (Structure, cell) loader                                     #
# --------------------------------------------------------------------- #


def _emit_dispersion_template(xc_authors: str, v: bool) -> List[str]:
    """Commented-out Grimme-D2 dispersion-correction template emitted
    when XC.functional is non-vdW (PBE / BLYP / hybrids).  See gap #3
    in docs/design.md.

    The template is commented so the default behaviour is unchanged
    (the user opts in by uncommenting).  Parameter values are
    placeholders -- per-species C6 / R0 come from the Grimme-D2 table
    (Grimme 2006, J. Comput. Chem. 27, 1787); for biomolecule-typical
    species (C, N, O, H, P, S) the 21-pair grid is small enough to
    paste in by hand once you've decided you want it.
    """
    out: List[str] = []
    out.append("# --- Dispersion correction (commented template) ---")
    if v:
        out += [
            f"# {xc_authors} is a non-dispersive XC: long-range vdW",
            "# (C6/r^6) is missing.  Organic / biomolecule consequences:",
            "#   * DNA stacking under-bound by 5-10 kcal/mol per pair",
            "#   * peptide folding favours wrong conformers",
            "#   * molecular crystals: lattice constants too long by 0.1-0.3 A",
            "#   * surface adsorption (benzene/graphite, etc.) off ~10x",
            "# Two ways to fix.  Pick ONE:",
            "#",
            "# 1) Switch to a vdW-aware XC (cheapest correctness):",
            "#      XC.functional VDW",
            "#      XC.authors    DRSLL    (or KBM, LMKLL, BH, VV)",
            "#    The non-local correlation lives in the functional;",
            "#    no MM.Potentials block needed.",
            "#",
            "# 2) Add Grimme-D2 empirical dispersion ON TOP of the",
            "#    current XC (cheap, additive, no XC change).  Fill in",
            "#    one row per atom-species pair from Grimme-D2 tables.",
            "#    Uncomment to enable:",
        ]
    out += [
        "# %block MM.Potentials",
        "#   # species_i  species_j  type     C6 (Eh*Bohr^6)  R0 (Bohr)",
        "#   #   C           C         Grimme   1.75            1.452",
        "#   #   C           H         Grimme   ...             ...",
        "#   #   N           N         Grimme   ...             ...",
        "#   # See SIESTA manual sec. 5.20 (MM.Potentials) and Grimme",
        "#   # (2006) Tables 1+2 for C6 / R0 per species.",
        "# %endblock MM.Potentials",
    ]
    return out


def _struct_from_file(path: str) -> Tuple[Structure, Optional[np.ndarray]]:
    """Read an XYZ or PDB and return ``(Structure, cell_or_None)``.

    Format is detected from the file extension.  XYZ files may carry
    a periodic cell in the comment line (ASE's extended XYZ format);
    if present, it is returned alongside the structure so the caller
    can preserve it in the FDF.  PDB has no native cell concept here,
    so the cell is always ``None``.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdb":
        return Structure.from_pdb(p), None
    if ext in (".xyz", ""):
        # Use ASE for XYZ -- it understands extended-XYZ headers and
        # gives us the lattice when present, which our hand-rolled
        # parser doesn't.
        atoms = _ase_read(path)
        elements = [a.symbol for a in atoms]
        positions = atoms.get_positions()
        cell = atoms.cell.array if hasattr(atoms.cell, "array") else atoms.cell
        cell = np.asarray(cell, dtype=float)
        return (
            Structure(elements=elements, positions=positions, title=p.stem),
            cell if cell.any() else None,
        )
    raise ValueError(
        f"unsupported input extension {ext!r}; expected .xyz or .pdb"
    )


def convert(
    input_path: str,
    fdf_path: str,
    config: Optional["SiestaConfig"] = None,
    vacuum: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Read an XYZ or PDB file, write an FDF, optionally copy psml files.

    ``vacuum`` (Å, per-side gap) sets the structure's isolation padding -- the
    CLI/convert equivalent of the Modify -> Cell tab, since vacuum comes with the
    STRUCTURE (structure-periodicity.md), not the config.  Applied only when the
    input file carries no explicit cell (an imported cell wins).  Without it a
    flat/linear molecule loaded from a bare XYZ has vacuum 0 -> a degenerate cell
    (render_fdf raises with an actionable message).

    Returns a summary dict with keys: ``fdf``, ``n_atoms``, ``species``,
    ``missing_psml``.
    """
    cfg = config or SiestaConfig()
    struct, cell = _struct_from_file(input_path)
    if vacuum is not None and cell is None:
        struct.vacuum = tuple(float(v) for v in vacuum)

    species = (list(cfg.species_order) if cfg.species_order
               else _detect_species(struct.elements))
    fdf_text = render_fdf(struct, cfg, cell=cell)

    fdf_p = Path(fdf_path)
    fdf_p.parent.mkdir(parents=True, exist_ok=True)
    fdf_p.write_text(fdf_text)

    summary = {
        "fdf": str(fdf_p),
        "n_atoms": struct.n_atoms,
        "species": species,
        "missing_psml": [],
    }

    # Makov-Payne correction script.  Emitted whenever the input
    # carries a non-zero net charge so the user can run a single
    # post-process command after SIESTA finishes and get the
    # finite-size-corrected total energy.  The FDF header already
    # tells the user about the artefact; the script makes the
    # correction numeric instead of "go do the arithmetic
    # yourself".
    from .makov_payne import emit_correction_script
    from ..chemistry import resolve_net_charge
    try:
        _q = resolve_net_charge(struct, getattr(cfg, "net_charge", None))
    except Exception:
        _q = 0
    if _q != 0:
        emitted = emit_correction_script(
            fdf_path=fdf_p,
            system_label=cfg.system_label,
            q=_q,
        )
        if emitted is not None:
            summary["makov_payne_script"] = str(emitted)

    if cfg.psml_lib and cfg.copy_psml:
        lib = Path(cfg.psml_lib).expanduser()
        if not lib.is_dir():
            print(f"  WARN: --psml-lib {lib} is not a directory; skipping psml copy",
                  file=sys.stderr)
        else:
            summary["missing_psml"] = copy_pseudopotentials(species, lib, fdf_p.parent)

    # Drop a preview <basename>[-stage<N>].molwatch.log next to the
    # .fdf so molwatch can render the initial geometry the moment the
    # user loads it -- no waiting for SIESTA to write its first
    # outcoor block.  The file is static (one preview block, no live
    # updates); for live updates while SIESTA is running, point
    # molwatch at the .out file instead.
    #
    # Filename derives from cfg.system_label (the protocol basename)
    # plus the optional stage suffix -- NOT from the FDF's stem.  This
    # way a user who names the FDF "anything.fdf" still gets the
    # canonical preview-log name that the Watch tab discovery chain
    # recognises.  See docs/protocols/job-layout.md.
    if cfg.write_molwatch_log:
        from ..trajectory_log import molwatch_log_basename, write_initial_preview
        mw_path = fdf_p.parent / molwatch_log_basename(
            cfg.system_label, cfg.stage)
        write_initial_preview(
            struct,
            mw_path,
            job=cfg.system_label,
            engine="siesta",
        )
        summary["molwatch_log"] = str(mw_path)

    return summary


# --------------------------------------------------------------------- #
#  Multi-stage emitters (#542 / C1 commit 2)                            #
# --------------------------------------------------------------------- #
#                                                                       #
#  The two functions below are the SIESTA counterpart to PySCF's        #
#  _emit_stages_loop().  PySCF runs all stages inside ONE Python        #
#  process (warm-start via ``dm0=dm_prev``); SIESTA runs each stage     #
#  as a separate ``siesta`` invocation (warm-start via the .XV file     #
#  that the prior invocation wrote, auto-picked-up because every        #
#  per-stage .fdf shares the same SystemLabel).                         #
#                                                                       #
#  Two-file emission shape:                                             #
#    <system_label>_<stage.name>.fdf   one per enabled stage; same      #
#                                      SystemLabel across all files.    #
#    <runner>                          bash driver, looped over the     #
#                                      enabled-stage list, runs each    #
#                                      .fdf through ``siesta``, and     #
#                                      polices ``on_nonconvergence``.   #
#                                                                       #
#  The runner is intentionally minimal: NO conda activation, NO MPI     #
#  wrapping.  Those layers belong to runwrap.py (C1 commit 3) which     #
#  composes this runner into the final job script.  Keeping that out    #
#  of input.py means the same emitter is usable from a manual SIESTA    #
#  rig (user already activated the env).                                #


# Bash runner template -- format-substituted via str.format.  Every
# literal ``{`` and ``}`` in the bash source is doubled; the five real
# placeholders are {basename}, {stages}, {policies}, {n_stages},
# {siesta_cmd}.  Tested end-to-end via ``bash -n`` in the C1 tests.
_STAGES_RUNNER_TEMPLATE = r"""#!/usr/bin/env bash
# Generated by molbuilder for staged SIESTA optimisation
# (molbuilder.siesta.input.render_siesta_stages_runner).
#
# Run from the directory containing the per-stage .fdf files:
#   {basename}_<stage>.fdf  e.g.  {basename}_stage1.fdf
#
# Override behaviour via environment:
#   MOLBUILDER_FORCE=1   skip the warm-restart consistency prompt
#                        (use in --yes / batch / cron runs)
set -euo pipefail

BASENAME='{basename}'
STAGES=({stages})
ON_NONCONV=({policies})
FORCE="${{MOLBUILDER_FORCE:-0}}"

_warm_check() {{
    # At stage 2+ entry, if a *.XV file exists in the directory but
    # ${{BASENAME}}.XV is missing, the user probably renamed the
    # project since the prior stage ran -- SIESTA will NOT pick up
    # the stray .XV (filename keyed on SystemLabel), it will silently
    # re-initialise atoms from the .fdf.  Warn loud + offer abort.
    local idx=$1
    (( idx == 0 )) && return
    [[ -f "${{BASENAME}}.XV" ]] && return
    shopt -s nullglob
    local stray=( *.XV )
    shopt -u nullglob
    (( ${{#stray[@]}} == 0 )) && return
    cat >&2 <<EOM
[WARN] stage $((idx+1)) entry: ${{BASENAME}}.XV is missing, but other
       *.XV files exist here: ${{stray[*]}}
       SIESTA will NOT warm-restart from those -- it discards them and
       re-initialises atoms from the .fdf.  Likely cause: project
       rename since the prior stage ran.  Fix: rename the stray .XV
       to ${{BASENAME}}.XV, OR re-run the missing stage.
EOM
    if [[ "${{FORCE}}" == "1" ]]; then
        echo "[WARN] MOLBUILDER_FORCE=1 -- continuing without restart" >&2
        return
    fi
    if [[ ! -t 0 ]]; then
        echo "[ABORT] non-interactive shell; re-run with MOLBUILDER_FORCE=1" >&2
        echo "        to acknowledge + continue without restart." >&2
        exit 2
    fi
    read -r -p "Continue without warm-restart? [y/N] " resp
    case "$resp" in
        [yY]|[yY][eE][sS]) : ;;
        *) echo "[ABORT] user declined" >&2; exit 2 ;;
    esac
}}

for i in "${{!STAGES[@]}}"; do
    stage="${{STAGES[$i]}}"
    policy="${{ON_NONCONV[$i]}}"
    fdf="${{BASENAME}}_${{stage}}.fdf"
    log="${{BASENAME}}_${{stage}}.out"

    if [[ ! -f "$fdf" ]]; then
        echo "[ABORT] missing $fdf" >&2
        exit 1
    fi

    _warm_check "$i"

    echo "[stage $((i+1))/{n_stages}] ${{stage}}: ${{fdf}} -> ${{log}}"
    if ! {siesta_cmd} < "$fdf" > "$log"; then
        rc=$?
        echo "[stage $((i+1))] FAILED (rc=$rc)" >&2
        if [[ "$policy" == "halt" ]]; then
            exit "$rc"
        fi
        echo "[stage $((i+1))] policy=${{policy}}: continuing to next stage" >&2
    fi
done

echo "[done] all enabled stages completed: ${{STAGES[*]}}"
"""


def _enabled_stages(cfg: "SiestaConfig") -> List:
    """Return cfg.stages filtered to enabled entries, or raise."""
    stages = list(getattr(cfg, "stages", []) or [])
    enabled = [s for s in stages if s.enabled]
    if not enabled:
        raise ValueError(
            "render_siesta_stage_fdfs / render_siesta_stages_runner: "
            "cfg.stages has no enabled entries.  At least one stage "
            "must be enabled."
        )
    return enabled


def render_siesta_stage_fdfs(
    struct: Structure,
    config: Optional["SiestaConfig"] = None,
    *,
    cell: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Render one .fdf per enabled stage in ``config.stages``.

    Returns a mapping ``{filename: fdf_text}``.  All emitted files share
    the same ``SystemLabel`` (== ``cfg.system_label``) so SIESTA's
    auto-read of ``<SystemLabel>.XV`` warm-restarts each stage from
    the prior stage's final geometry without any file-copying step.

    Filename convention: ``f"{cfg.system_label}_{stage.name}.fdf"``.

    The LAST enabled stage's ``on_nonconvergence`` is intentionally
    NOT mutated here -- this function only renders fdfs.  The runner
    (:func:`render_siesta_stages_runner`) applies the
    "last-stage-must-halt" policy.

    Raises ``ValueError`` if no stage is enabled.
    """
    cfg = config or SiestaConfig()
    enabled = _enabled_stages(cfg)
    out: Dict[str, str] = {}
    for stage in enabled:
        staged_cfg = dataclasses.replace(
            cfg,
            relax_type=stage.relax_type,
            relax_steps=stage.relax_steps,
            relax_force_tol=stage.relax_force_tol,
            relax_max_displ=stage.relax_max_displ,
        )
        filename = f"{cfg.system_label}_{stage.name}.fdf"
        out[filename] = render_fdf(struct, staged_cfg, cell=cell)
    return out


def render_siesta_stages_runner(
    config: "SiestaConfig",
    *,
    siesta_cmd: str = "siesta",
) -> str:
    """Render the bash runner that drives the per-stage SIESTA loop.

    ``siesta_cmd`` is injected verbatim into the per-stage invocation
    line, BEFORE the ``< $fdf > $log`` redirection.  Pass ``"siesta"``
    for a serial run (the default), or e.g. ``"mpirun -np 8 siesta"``
    for an MPI run.  runwrap.py (C1 commit 3) composes the full
    command -- this emitter doesn't try to second-guess MPI / OpenMP /
    GPU flags.

    The LAST enabled stage's ``on_nonconvergence`` is forced to
    ``"halt"`` in the rendered runner regardless of what the
    SiestaStageSpec carried.  Rationale: the final tier of any
    ladder is the publishable / production result; falling through
    silently is a bug.  Earlier stages keep their declared policy
    (``proceed`` / ``continue`` / ``halt``).

    Raises ``ValueError`` if no stage is enabled.
    """
    enabled = _enabled_stages(config)
    # Force-halt last enabled stage -- see docstring.
    last_idx = len(enabled) - 1
    policies = [
        "halt" if i == last_idx else s.on_nonconvergence
        for i, s in enumerate(enabled)
    ]
    # Stage names are constrained to ``[A-Za-z0-9_]+`` by the
    # SiestaStageSpec name pattern, so they're shell-safe unquoted in
    # the bash array literal.
    return _STAGES_RUNNER_TEMPLATE.format(
        basename=config.system_label,
        stages=" ".join(s.name for s in enabled),
        policies=" ".join(policies),
        n_stages=len(enabled),
        siesta_cmd=siesta_cmd,
    )
