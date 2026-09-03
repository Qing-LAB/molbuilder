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
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ase.data import atomic_numbers
    from ase.io import read as _ase_read   # noqa: F401 -- probe, see below
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "molbuilder.siesta needs ASE; install with `pip install ase`"
    ) from exc
# ``_ase_read`` is imported and not called: it is a PROBE.  ``ase.data`` is a
# plain table and imports even from a half-installed ASE, while ``ase.io``
# pulls the reader machinery -- so asking for both is what turns a broken
# install into this message instead of an AttributeError several hundred lines
# into a render.

from ..structure import Structure
# SiestaConfig is the L1 dataclass; this module imports it for use by
# the generator below.  External callers can import it from either
# molbuilder.config.siesta (the canonical location) or from
# molbuilder.siesta (re-exported by siesta/__init__.py).
from ..config.siesta import SiestaConfig
# § 4 rule 2's reading of `restart`, shared with PySCF -- one field, one
# rule, one place that reads it.
from ..identity import continues
# Module level: `deck_note` is called from the body emitter, and script_emit
# imports no engine (its own `template` import is lazy), so there is no cycle.
from .. import script_emit as _sc







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
    ``BlockSize <= floor(n_atoms / Nrank)``.  An empirical sweep on
    2026-05-28 disproved it: SIESTA crashes IDENTICALLY with
    BlockSize = 1, 2, 4 at mpi_np = 15 on hemeC-dithiol.  THIS NOTE
    is the durable record of that result -- the probe ran in a /tmp
    scratch that no longer exists, so the numbers here are the
    artifact (softened 2026-08-12; a citation pointing at deleted
    scratch read as if a repo artifact backed it).

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
    leaves some ranks idle on the diag step.  The per-rank cap is
    stated in ORBITALS, not atoms: SIESTA distributes ORBITALS
    across ranks (BlockSize is a block of the ScaLAPACK orbital
    distribution -- SIESTA's own auto-pick is ``ceil(Norb /
    Nrank)``), so the bound is ``floor(n_orbitals_est / mpi_np)``
    with ``n_orbitals_est = 10 * n_atoms`` -- the SAME rough DZP
    estimate the deck's BENCH-MARKS block records as
    ``n_orbitals_est`` (job-contracts.md § 3.2 provenance example
    + § 3.3; the atoms-based cap ``floor(n_atoms / mpi_np)`` was
    retired U18, 2026-08-12).  It is an upper bound for the diag
    block, not the propor-fixing constraint it was once advertised
    as (see HISTORICAL NOTE above).

    Strategy
    --------
    Two regimes -- CPU mode and GPU mode.  Since U18 (2026-08-12)
    both derive the cap from the same orbital estimate; they still
    differ in floor and in the no-rank-info fallback, because the
    optimal BlockSize differs on the two solvers.

    CPU mode (default)
      * mpi_np is None or 1: size-only ladder (1, 2, 4, 8 by
        n_atoms), capped at 8.  No rank count means the per-rank
        derivation cannot be stated; this conservative baseline
        predates the contract and is out of its scope (single-rank
        runs ignore BlockSize anyway).
      * mpi_np >= 2: largest power of 2 satisfying
        ``BlockSize <= min(256, floor(10 * n_atoms / mpi_np))``.
        The 256 ceiling is the LOAD-BALANCE ceiling shared with GPU
        mode below: past it, too few blocks circulate per rank for
        BLACS to balance, with no cache gain to pay for it.  (Until
        2026-08-12 this bullet also propped 256 on "the top of the
        BENCH-MARKS legal override window, range=[16,256]" -- a
        constant § 3.3 retired 2026-08-10; the deck's declared window
        is now per-deck ``[1, floor(n_orbitals_est/mpi_np)]`` and the
        auto pick lands inside it by the same min() above.)

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
        # Orbital-aware cap: ``floor(10 * n_atoms / mpi_np)`` (the
        # 10x is a rough DZP-basis heuristic; underestimates for
        # heavy elements like Au where DZP gives ~25 orb/atom).
        # Since U18 (2026-08-12) the CPU branch below derives from
        # the same orbital estimate per job-contracts.md § 3.2/
        # § 3.3; GPU keeps its own branch for the floor of 8 and
        # the mpi_np=None default of 4 (GPU+MPS policy).
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
    # CPU mode with mpi_np >= 2.  Rank constraint in ORBITALS, not
    # atoms: SIESTA distributes ORBITALS across ranks, and BlockSize
    # is a block of that orbital distribution, so every rank must
    # get >= 1 ORBITAL block, i.e. ``BlockSize <=
    # floor(n_orbitals_est / mpi_np)`` with ``n_orbitals_est =
    # 10 * n_atoms`` -- the SAME estimate the deck's BENCH-MARKS
    # block records (job-contracts.md § 3.2 provenance example +
    # § 3.3; atoms-based ``floor(n_atoms / mpi_np)`` retired U18,
    # 2026-08-12).  The 256 ceiling: top of the BENCH-MARKS legal
    # override window (``range=[16,256]``, § 3.3) and the
    # load-balance ceiling shared with the GPU branch -- above 256
    # a low-rank run drops below ~12 blocks/rank and tail-effect
    # imbalance bites.  Take the LARGEST power of 2 under the cap.
    orbital_estimate = 10 * max(1, n_atoms)
    cap = max(1, min(256, orbital_estimate // int(mpi_np)))
    pow2 = 1
    while pow2 * 2 <= cap:
        pow2 *= 2
    return pow2


def _block_size_bounds(n_atoms: int,
                       mpi_np: Optional[int] = None,
                       gpu_mode: bool = False,
                       *,
                       emitted: Optional[int] = None) -> Tuple[int, int]:
    """The power-of-two window a ``BlockSize`` override may use **for this
    deck** — what BENCH-MARKS declares as ``range=[lo,hi]``.

    ``job-contracts.md`` § 3.3 calls ``range`` *"advisory bounds for
    validating a requested override"*.  Advice about a value that was derived
    from the launch has to be derived from the same launch, and until
    2026-08-10 it was not: the range was the module constant ``(16, 256)``
    while the default came from :func:`_auto_block_size`.  The two disagreed
    routinely rather than exceptionally — under the ATOMS-era derivation of
    the day, ``_auto_block_size(200, mpi_np=16)`` was 8 and ``(20,
    mpi_np=32)`` was 1, both below the declared floor (U18's orbital
    derivation gives 64 and 4; the history keeps the old numbers because
    they are what motivated the fix) — so the block advised a *validator*
    that its own emitted value was illegal, and advised a *bench tool* it
    could climb past this deck's own rank constraint.  Climbing is the
    dangerous direction: above ``floor(n_orbitals_est / mpi_np)`` some
    ranks get no block at all.

    **One derivation, not two.**  The upper bound IS
    :func:`_auto_block_size`'s answer, because that function already picks the
    largest legal power of two — so "the generator's choice" and "the top of
    the window" are the same number by construction, and cannot drift apart
    the way a second constant did.  That gives the block a checkable
    invariant: ``lo <= default <= hi``, always.

    The floor is the picker's own: 1 on CPU (the empirical sweep in
    :func:`_auto_block_size` swept 1, 2, 4), 8 in GPU mode, where the ELPA-CUDA
    branch never goes below 8.

    ``emitted`` is the value the deck actually carries.  It differs from the
    derived one only when the user set ``block_size``, which
    :func:`render_fdf` honours verbatim; the window is widened to contain it,
    because a block whose range excludes its own default is the defect this
    function exists to end — the user's number is a *decision*, not an error
    to advertise as out of bounds.
    """
    hi = _auto_block_size(n_atoms, mpi_np, gpu_mode=gpu_mode)
    lo = 8 if gpu_mode else 1
    if emitted is not None:
        lo = min(lo, int(emitted))
        hi = max(hi, int(emitted))
    return (lo, hi)


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


def _restart_keys(cfg) -> tuple:
    """Which members of the declared group this run mode has a use for.

    The group itself is the catalogue's — ``[item.restart].expands``, read
    through the one API (`template.md` § 8.0).  This says only which of them
    *mean* anything for the run being written, and there is exactly one such
    distinction: ``MD.UseSaveCG`` names the optimizer's own history, so a run
    that does not relax with an optimizer has none.

    **It read ``SIESTA_RESTART_GROUP.keys`` for one morning** (2026-08-18) and
    that was the wrong home. *Which keywords `restart` writes* was declared in
    THREE places — the catalogue's ``expands``, that tuple, and
    ``warm-files.toml``'s ``honoured_by`` rows — carrying the same three names
    in three different orders. A generator asked to *"pull what it needs from
    the source it knows"* cannot, when there are three; whichever an author
    reaches for is the one their layer then disagrees from. ``expands`` is the
    parameter's own statement of what it writes into the deck, it is floor-2
    data, and it is already in the read API every other item question goes
    through.

    ``none`` (a static stage) and the dynamics modes are the cases: a Verlet
    or Nosé run integrates rather than optimizes.  Broyden and FIRE **do** get
    it — that is what the condition has always done, and narrowing it is a
    SIESTA-semantics question wanting the manual and a science review, not
    something to change while fixing how the group is written.
    """
    keys = _sc.parameter("restart", "siesta").writes
    relax = str(getattr(cfg, "relax_type", "") or "none").strip().upper()
    if relax in ("NONE", "VERLET", "NOSE"):
        return tuple(k for k in keys if k != "MD.UseSaveCG")
    return keys


def _restart_group_lines(cfg) -> List[str]:
    """The whole restart group, in one place, for BOTH answers.

    **`clean` writes `.false.`; it does not stay silent.**  Until 2026-08-18
    this group was emitted only when the run continued, on the premise that a
    key left out is a key not honoured.  It is not: SIESTA reads
    ``<SystemLabel>.DM`` when the file is there whatever the deck omits
    (measured — see ``SIESTA_RESTART_GROUP.mechanism``).  A stage told to
    start clean therefore warm-started from whatever the directory held, and a
    benchmark trial — forced clean precisely so every point measures the same
    thing — measured a continued run whenever the wrapper retried it.

    **One site, from the declaration.**  The members were written by hand at
    two points in this file, several hundred lines apart, each with its own
    ``if continues(cfg)``.  Reading them from
    the catalogue's ``expands`` is what makes *"one field, one group"* true
    of the code rather than of a comment: a member cannot now be emitted with
    one answer while its sibling carries the other, and an engine that grows a
    fourth member gets it written without touching this function.
    """
    on = ".true." if continues(cfg) else ".false."
    return [f"{k:<18}{on}" for k in _restart_keys(cfg)]


def _stage_science(cfg: "SiestaConfig") -> str:
    """One line saying what this stage actually computes, in its own units.

    Decided 2026-08-10 (user): *"have comments somewhere for explaining what it
    is as scientific notation for each run."*  A stage's **name** says which
    rung it is; this says what the rung IS -- the numbers that make a coarse
    stage coarse.

    **Derived from the config being rendered, never from the description.**
    That is the whole design of it: the values quoted here are the same objects
    the keyword lines below are written from, so the comment cannot drift from
    the deck it sits in.  A prose note typed per stage could say
    *"tight convergence"* over a deck someone had since loosened; this cannot.
    It also keeps ``Stage`` at the three fields ``engines/stages.md`` § 2
    allows -- a note would have been a fourth.

    Units are the ones the deck itself writes a few lines down (``MeshCutoff
    … Ry``, ``MD.MaxForceTol … eV/Ang``), so the comment and the keywords read
    as one document.
    """
    bits = [f"MeshCutoff {cfg.mesh_cutoff} Ry",
            f"force tol {cfg.relax_force_tol} eV/Ang"]
    if getattr(cfg, "relax_type", None):
        bits.append(str(cfg.relax_type))
    if getattr(cfg, "relax_steps", None):
        bits.append(f"max {cfg.relax_steps} steps")
    return " \u00b7 ".join(bits)


def _bench_marks_for(struct, cfg, block_size, bs_range) -> dict:
    """The BENCH-MARKS block's arguments — the values only SIESTA can supply.

    Kept beside the writer rather than in the framework because every entry is
    a SIESTA fact: which keyword carries the step count, that ``BlockSize`` is
    derived from a launch quantity, that a defaults row for an absent keyword
    would be the block lying.
    """
    # BlockSize's row carries the range this deck's rank count makes legal, and
    # drops out entirely when the deck carries no BlockSize (state three).
    fields = [
        (dataclasses.replace(f, range_=bs_range) if f.anchor == "BlockSize"
         else f)
        for f in _sc.SIESTA_BENCH_FIELDS
        if not (f.anchor == "BlockSize" and block_size is None)
    ]
    return dict(
        metadata={
            "n_atoms":        struct.n_atoms,
            "n_orbitals_est": 10 * struct.n_atoms,
            "gpu_mode":       str(bool(cfg.use_gpu)).lower(),
            # The launch quantity BlockSize was derived FROM.  § 5.2's whole
            # point is that a later change of launch can re-derive the coupled
            # lines "instead of silently leaving them stale" -- and
            # ``_auto_block_size`` takes three inputs while this block used to
            # record two, so re-derivation was not actually possible from what
            # the deck carried.
            "mpi_np":         ("auto" if cfg.mpi_np is None
                               else str(int(cfg.mpi_np))),
        },
        fields=fields,
        defaults={
            # State three omits the row too: a BENCH-MARKS line claiming a
            # value the deck does not carry would be the block lying.
            **({"BlockSize": block_size} if block_size is not None else {}),
            "MaxSCFIterations":  cfg.max_scf_iter,
            # only when the DECK carries the line (CG/Broyden/FIRE): relax
            # "none" and the Verlet/Nose dynamics emit no MD.Steps, and a
            # defaults row for an absent keyword is the block lying -- the same
            # rule as BlockSize's state three one entry up (R11, 2026-08-12).
            **({"MD.Steps": cfg.relax_steps}
               if (cfg.relax_type
                   and cfg.relax_type.strip().upper()
                   in ("CG", "BROYDEN", "FIRE"))
               else {}),
            "MeshCutoff":        cfg.mesh_cutoff,
        },
    )


def _spin_facts(cfg) -> dict:
    """Whether this deck constrains the spin, and whether it pins a value.

    ``spin_total`` EXPANDS to two keywords -- SIESTA ignores the number
    without ``Spin.Fix`` -- so *which* items the section carries depends
    on the answer, and the layout cannot be read until it exists.
    """
    polarized = cfg.spin_treatment != "non-polarized"
    return {
        "spin_polarized": polarized,
        "spin_fixed": (polarized and cfg.spin_total is not None
                       and cfg.spin_treatment == "polarized"),
    }


def _parallel_facts(cfg) -> dict:
    """How the work is split across ranks, and which solver reads which knob.

    **Derivation, not emission** -- and at SPEC time, for the same reason
    `_relaxation_facts` is: the MPI section's membership depends on these
    answers (no ``BlockSize`` line at all when the value is unset, no
    ``Diag.*`` when the solver is ScaLAPACK), so the layout cannot be read
    until they exist (`script-preparation.md` § 4.1).

    ``over_k`` read the k-grid out of the writer's locals until 2026-08-19;
    it is ``cfg.kgrid`` either way, and reading the config is what let this
    move ahead of the deck.
    """
    # ---- Parallel execution (MPI) -------------------------------
    # BlockSize is a THROUGHPUT knob, not a crash guard: the empirical
    # sweep recorded in the HISTORICAL NOTE above (2026-05-28, hemeC)
    # showed the ``propor: ERROR: IMAX = 0`` startup crash identical at
    # BlockSize 1, 2 and 4 -- it is matel_table's proportionality check
    # against the rank count, and the remedy is a smaller -np.  The
    # paragraph that stood here until 2026-08-12 still taught the
    # pre-sweep theory ("an explicit smaller BlockSize keeps every
    # distribution step well-conditioned") -- the OPPOSITE of the deck
    # text emitted ten lines below, in the same function.
    # TWO STATES (tuning.md § 2.11, revised 2026-08-15).  Unset means
    # AUTO, and auto means SIESTA'S OWN automatic -- the keyword is simply
    # not emitted.  The manual declares it: ``BlockSize [integer]
    # <automatic>``.  Omitting a keyword is a real answer, the same shape
    # as ``Diag.Algorithm ScaLAPACK`` emitting nothing (siesta.md § 7).
    #
    # A THIRD state stood here until 2026-08-15: unset made molbuilder
    # DERIVE a value (``_auto_block_size``) and write it into the deck,
    # while SIESTA's own automatic hid behind the sentinel ``0``.  So the
    # ordinary user got a guess and the engine's answer needed a magic
    # number to request -- and the guess contradicted § 2.11's own opening
    # decision (2026-08-11): *"not a value molbuilder derives and hands
    # you"*.  It also produced ``BlockSize 1`` below four atoms, which is
    # legal and the exact opposite of the cache blocking the parameter
    # exists for.
    #
    # ``_auto_block_size`` itself is NOT deleted -- it is still the upper
    # bound of the BENCH-MARKS window (``_block_size_bounds``), which is
    # where a power-of-two constraint belongs: the benchmark sweeps them.
    if cfg.block_size is None:
        block_size = None
    else:
        # Honoured verbatim -- hand-set, or a benched result.  Earlier code
        # auto-downgraded when ``BlockSize * mpi_np > n_atoms`` on the
        # theory that it caused propor IMAX=0; an empirical sweep
        # (2026-05-28) disproved that -- propor is a matel_table issue, not
        # a BlockSize issue.  Under a GPU-ELPA target a non-power-of-two
        # value is realigned by `prep`, which is the layer that knows the
        # GPU flag and the rank count (§ 2.11); it is not second-guessed
        # here, and never silently.
        block_size = int(cfg.block_size)

    if cfg.parallel_over_k is None:
        over_k = tuple(cfg.kgrid) != (1, 1, 1)
    else:
        over_k = bool(cfg.parallel_over_k)

    # Diagonalizer (engines/siesta.md § 7).  The solver choice
    # (``diag_algorithm``) is INDEPENDENT of the GPU toggle; ELPA runs on
    # CPU and GPU alike, and ``use_gpu`` only moves an ELPA solve onto
    # the GPU.
    #   * ScaLAPACK -> emit nothing (SIESTA's built-in Divide-and-Conquer).
    #   * ELPA-* -> emit ``Diag.Algorithm`` (required: Diag.ELPA.GPU alone
    #     is ignored without it, Src/diag_option.F90:213-225) AND
    #     ``Diag.ELPA.GPU .true./.false.``.  The explicit ``.false.`` for
    #     CPU-ELPA is load-bearing: the source ELPA defaults to the GPU
    #     codepath, so an omitted flag crashes a CPU run (Sol job 57852378).
    _algo = (cfg.diag_algorithm or "ScaLAPACK").strip()
    _is_elpa = _algo.upper().startswith("ELPA")
    if cfg.use_gpu and not _is_elpa:
        raise ValueError(
            "use_gpu requires an ELPA diagonalizer (diag_algorithm = "
            "ELPA-1STAGE or ELPA-2STAGE); GPU acceleration does not apply to "
            f"the {_algo} solver.  Pick an ELPA algorithm or turn GPU off "
            "(engines/siesta.md § 7).")
    # THE FOUR MPI VALUES go through the one door.  None of them is a config
    # field read straight through -- the block size is computed from the atom
    # count and the rank count, ParallelOverK defaults from whether the k-mesh
    # is more than Gamma, the algorithm is normalised -- so they reach the door
    # through ``parameter(..., value=)`` and a DERIVED number still arrives
    # with its declaration, its range and its note.
    return {"block_size": block_size, "over_k": over_k,
            "algorithm": _algo, "gpu": bool(cfg.use_gpu)}


def _relaxation_facts(cfg) -> Optional[dict]:
    """What SIESTA spells this run's geometry loop with, or ``None`` for a
    single point.

    **Derivation, not emission**, and it runs when the SPEC is built rather
    than while the deck is written.  Which items the geometry section
    carries depends on these answers, and a layout whose MEMBERS are only
    known once the writer has run is a layout the framework cannot read --
    which is the whole reason the seam carries a form
    (`script-preparation.md` § 4.1).  Every value here is a function of
    ``cfg`` alone, so there was never anything to wait for.
    """
    # Relaxation / dynamics.  In SIESTA 5.4.2 the step-count and
    # displacement-cap fdf keywords are UNIVERSAL across relax types
    # despite the CG-prefixed names -- ``MD.Steps`` and
    # ``MD.MaxDispl`` are recognized for CG, Broyden, AND FIRE.
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
    # SIESTA 5.4.2 with ``MD.TypeOfRun Broyden`` + ``MD.Steps 5``
    # + ``MD.MaxDispl 0.1 Ang``):
    #   redata: Dynamics option        = Broyden coord. optimization
    #   redata: Maximum number of optimization moves = 5
    #   redata: Max atomic displ per move = 0.1000 Ang
    # Identical echo lines a CG first stage already produces in real jobs.
    #
    # Verlet / Nose (NVE / NVT dynamics, not relaxation) use distinct
    # step-control keywords -- ``MD.FinalTimeStep`` + the temperature
    # block.  They never reached this branch with the broken mapping
    # because no test ever ran them; today they're handled below too
    # for completeness, with the universal MD.Steps NOT emitted
    # (it would be a no-op + visual noise in the fdf).
    if not (cfg.relax_type and cfg.relax_type.lower() != "none"):
        return None
    relax_kind = cfg.relax_type.strip().upper()
    is_md = relax_kind in ("VERLET", "NOSE")
    # Universal step-count keyword for CG / Broyden / FIRE.
    # Verlet / Nose use MD.FinalTimeStep instead (the loop is
    # time-based, not step-count-based) -- handled below.
    _STEP_KW = {
        "CG":      "MD.Steps",
        "BROYDEN": "MD.Steps",
        "FIRE":    "MD.Steps",
        "VERLET":  "MD.FinalTimeStep",
        "NOSE":    "MD.FinalTimeStep",
    }
    # REFUSED, never defaulted.  This fell back to ``MD.Steps`` for any
    # unrecognised type, and SIESTA's geometry loop is bounded by a
    # DIFFERENT keyword depending on the run type (siesta_init.F: idyn 0
    # bounds on MD.Steps, idyn 1-5 on MD.InitialTimeStep..FinalTimeStep).
    # So adding any of SIESTA's other MD ensembles -- ParrinelloRahman,
    # NoseParrinelloRahman, Anneal, all idyn 3-5 -- to the choice list
    # without touching this map would emit a keyword the run ignores,
    # leaving MD.FinalTimeStep at its default of 1: a one-step MD that
    # looks like it ran.  A loud refusal here costs one line; that bug
    # costs a wasted allocation and is invisible in the output.
    if relax_kind not in _STEP_KW:
        raise ValueError(
            f"relax_type {cfg.relax_type!r} has no step-count keyword "
            f"mapping. SIESTA bounds a relaxation with MD.Steps and an MD "
            f"run with MD.FinalTimeStep, and guessing wrong gives a "
            f"one-step run that reports success. Add it to _STEP_KW in "
            f"siesta/input.py with the keyword its MD.TypeOfRun family "
            f"uses (known: {', '.join(sorted(_STEP_KW))}).")
    step_kw = _STEP_KW[relax_kind]
    # Universal displacement-cap keyword for CG / Broyden / FIRE;
    # Verlet / Nose have no per-step displacement cap (forces +
    # masses drive the timestep instead).
    displ_kw = "MD.MaxDispl" if not is_md else None

    # A Nose thermostat with no target temperature set holds the run at
    # the temperature it started from.  The fallback is the ENGINE's rule,
    # so it is resolved here and handed to the door rather than left for
    # the door to guess.
    target_T = (cfg.md_target_temperature
                if cfg.md_target_temperature is not None
                else cfg.md_initial_temperature)
    return {
        "relax_kind": relax_kind,
        "is_md":      is_md,
        "step_kw":    step_kw,
        "displ_kw":   displ_kw,
        "target_t":   target_T if is_md else None,
    }



def spec_for(struct: Structure, config: Optional["SiestaConfig"] = None,
               *, cell: Optional[np.ndarray] = None,
               stage_token: Optional[str] = None,
               calculation: str = "optimization") -> "_sc.RenderedDeck":
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
    if calculation != "optimization":
        raise ValueError(
            f"SIESTA has no {calculation!r} deck yet; the vibration kind "
            f"is PySCF-first (spectra-migration plan § 2 -- the "
            f"engine-agnostic shape admits a SIESTA arm later).")
    # WHAT THIS DECK WRITES is not collected here.  It is read off the
    # LAYOUT below by the framework, which is the only reading that can
    # close the check gate's loop: a list this writer kept would say what
    # the writer believed, and the gate exists because a writer can be
    # wrong.  This function kept such a list until 2026-08-19 -- filled at
    # eight call sites and read at none, because the sections it collected
    # from were rendered inside a block where the framework could not see
    # them (`script-preparation.md` § 4.1).
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
        # LAST LINE OF DEFENCE: SIESTA builds reciprocal vectors from the cell,
        # so a zero-volume lattice fails outright -- it must never be emitted.
        #
        # A flat / linear molecule with NO vacuum set used to land here, but the
        # 6.1 default now gives every ISOLATED axis 3 A per side, so that path
        # is gone.  Two things still reach here: a vacuum a user SET to zero on
        # a flat axis (their value is honoured, never overridden), and an
        # axis where vacuum does not apply: a TRANSPORT axis, whose length is the
        # captured device length rather than bbox + padding.  The message names
        # the offending axes and their kinds, because telling the user to "set a
        # vacuum" would be wrong advice for exactly the case that gets here.
        from molbuilder.cell import ZERO_VOLUME_TOL
        if abs(float(np.linalg.det(cell))) < ZERO_VOLUME_TOL:
            _kinds = struct.axis_kind or ("isolated",) * 3
            _thin = [(i, _kinds[i]) for i in range(3)
                     if abs(float(cell[i, i])) < ZERO_VOLUME_TOL]
            _detail = ", ".join(f"axis {i} (kind '{k}')" for i, k in _thin) \
                or "no single axis -- the three vectors are not independent"
            raise ValueError(
                f"the cell derived from the structure is degenerate (zero "
                f"volume), and SIESTA cannot run without a real box: {_detail}. "
                f"Vacuum padding applies only to an 'isolated' axis -- on a "
                f"'transport' axis the length IS the device length captured at "
                f"construction, and a structure with zero extent along it cannot "
                f"define that length. Set an explicit unit cell for that "
                f"direction (Modify -> Cell tab), or correct the axis kind. The "
                f"geometry is never changed for you.")
        origin = struct.resolve_cell_origin()
        if origin is not None:
            positions = positions - np.asarray(origin, dtype=float)
        # Vacuum adequacy is checked by the VALIDATOR
        # (validation/siesta.py:_check_siesta_vacuum_adequacy), which the
        # report(validate(...)) call below runs -- so the finding reaches the
        # web panel and this stderr report alike.  It used to be a
        # warnings.warn here, which no web user could ever see (contract R5,
        # science/validation.md 4.1).
        sizes = np.diag(cell)
        # Report the EFFECTIVE vacuum: where the user set none, the § 6.1
        # default supplied one, and the provenance comment must describe the
        # box that was actually emitted -- printing the stored value (``None``)
        # would make the note disagree with the LatticeVectors above it.
        _eff = struct.effective_vacuum()
        _defaulted = struct.defaulted_vacuum_axes()
        # Named from the model rather than repeated as a literal here.
        from molbuilder.structure import _DEFAULT_ISOLATED_VACUUM as _GAP
        _DEFAULT_GAP_TEXT = f"{_GAP:g} A/side"
        cell_note = (
            f"# (vacuum cell derived from the structure: "
            f"{sizes[0]:.2f} x {sizes[1]:.2f} x {sizes[2]:.2f} A; "
            f"vacuum = {tuple(round(float(v), 2) for v in _eff)} A/side"
            + (f"; no vacuum was set, so the default {_DEFAULT_GAP_TEXT} "
               f"was used on isolated axes {_defaulted}"
               if _defaulted else "")
            + f"; atoms centred)"
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
        # regions carries every label, reserved ones included -- so the frozen
        # set rides along and cannot be dropped between the Build endpoint that
        # loaded the sidecar and the validator meant to consume it.
        # struct.regions is Dict[str, List[int]] per Structure's
        # declaration; the previous list-comprehension assumed an
        # iterable of lists and crashed at __post_init__ when the
        # dict was non-empty (caught by task #303's Pattern-B test).
        regions       = {
            k: list(v)
            for k, v in (getattr(struct, "regions", {}) or {}).items()
        },
        # Periodicity metadata rides into validation too -- without it a
        # genuine crystal validated as isolated³/vacuum-0 and produced
        # spurious "kgrid on an isolated axis" + image-distance warnings
        # (review finding, 2026-07-29).
        cell          = (struct.cell.copy()
                         if struct.cell is not None else None),
        cell_origin   = (struct.cell_origin.copy()
                         if struct.cell_origin is not None else None),
        axis_kind     = struct.axis_kind,
        vacuum        = struct.vacuum,
        pbc           = struct.pbc,
    )
    # The gate is NOT run here.  `render_deck` owns step 3.3 and applies it
    # to the subject this spec names -- the wrapped coordinates and the
    # resolved cell, which is what the deck actually expresses.  Running it
    # here as well gave the step two owners judging two different
    # structures (`script-preparation.md` § 4.3).

    # WHAT THE DECK DERIVED — the per-render context, filled in as the writer
    # works each value out and read by everything downstream of it: the ONE
    # syntax door when a section is rendered, and the RECORD blocks when the
    # framework writes them afterwards.  The block size, the k-parallel
    # default, the algorithm, the step and displacement keywords and the Nosé
    # target are none of them a config field read straight through.
    #
    # It is what makes the body LAZY: the record needs numbers the body works
    # out, so before this the two could only be assembled in one pass, which is
    # why the deck's text — and not its spec — had to cross the seam (§ 4.3).
    v = cfg.verbose_comments
    from . import layout as _layout
    # WHAT THIS DECK DERIVED -- filled here, before the layout is built,
    # because the layout's MEMBERS depend on some of it: a geometry section
    # exists only for a run that moves atoms, and which items it carries
    # depends on whether that run is dynamics.  A layout whose membership is
    # only known once the writer has run is a layout the framework cannot
    # read, and then the check gate has nothing to compare the file against
    # (`script-preparation.md` § 4.1).  Every value below is a function of
    # `(struct, cfg)` alone, so there was never anything to wait for.
    #
    # The blocks fill in the rest as they render -- the block size, the
    # k-parallel default -- and the syntax door and the record blocks read
    # the same dict.  ONE channel, not one argument list per reader.
    _derived: dict = {**_spin_facts(cfg),
                      **_parallel_facts(cfg),
                      **(_relaxation_facts(cfg) or {})}

    def _deck_line(param):
        # ONE channel: whatever this deck has worked out so far.  The door
        # took a keyword per value until 2026-08-19, so the context could
        # hold nothing the door did not also declare -- and the layout's
        # own facts (is this a relaxation? an MD run?) had nowhere to live.
        return _layout.line(_derived)(param)

    # THE DECK'S SHAPE, declared.  Read down it and you have read what a
    # SIESTA deck contains and in what order.
    #
    # The blocks are wrapped in lambdas because they are defined BELOW --
    # a bare name here is looked up now, and the point of defining them
    # after is that a block runs when the framework walks to it, not when
    # the spec is built.  That laziness is what lets a FORM cross the seam
    # instead of finished text (`script-preparation.md` § 4.3).
    spec = _sc.DeckSpec(
        engine="siesta",
        layout=(
            _sc.Block("system, structure and constraints",
                      lambda s, c: _science(s, c)),
            _layout.BASIS_SECTION,
            _layout.XC_SECTION,
            _sc.Block("dispersion template",
                      lambda s, c: _dispersion_template(s, c)),
            _layout.SCF_SECTION,
            _layout.FREE_ENERGY_SECTION,
            _layout.SCF_TAIL_SECTION,
            _sc.Block("the restart group",
                      lambda s, c: _restart_group(s, c)),
            *((_layout.spin_section(
                   polarized=True, fixed=_derived["spin_fixed"]),)
              if _derived["spin_polarized"] else ()),
            _sc.Block("spin notes, net charge and k-points",
                      lambda s, c: _after_spin(s, c)),
            _layout.mpi_section(block_size=_derived["block_size"],
                                algorithm=_derived["algorithm"]),
            *((_layout.geometry_section(
                   is_md=_derived["is_md"],
                   is_nose=_derived["relax_kind"] == "NOSE"),)
              if _derived.get("relax_kind") else ()),
            _sc.Block("after the geometry settings",
                      lambda s, c: _after_geometry(s, c)),
            _layout.OUTPUT_SECTION,
            _sc.Block("troubleshooting",
                      lambda s, c: _troubleshooting(s, c)),
        ),
        line=_deck_line,
        # W10's context, DECLARED rather than only closed over: the same dict
        # `_deck_line`, the layout and the record blocks read.  Handing it to
        # the form costs nothing and is what lets a reader outside this module
        # see where `block_size` came from.
        derived=_derived,
        note_lead=_layout.note_lead,
        # section_title: the framework's default.  Both engines write a
        # heading as a `#` comment, so both restated the default verbatim
        # until 2026-08-19 -- two more copies of one string, and a slot
        # that LOOKED exercised.  It stays a slot because the comment
        # character is genuinely an engine's syntax; it is simply not one
        # these two differ on.
        provenance_defaults=lambda c: {
            "use_gpu": str(bool(c.use_gpu)).lower(),
            # TWO states, matching the emitter above: unset omits the
            # keyword, a value is written verbatim.  A `== 0` arm stood
            # here saying "omitted (SIESTA's own)" for the retired
            # sentinel -- and it would have LIED: with 0 the emitter
            # writes `BlockSize 0` into the deck (0 is not None), so
            # PROVENANCE claimed an omission the deck contradicts.  The
            # validator refuses 0 outright now (`_validate_block_size`),
            # so the state has no way in and no arm here.
            "BlockSize": (
                f"auto -> {_derived.get('block_size')}"
                if c.block_size is None
                else f"user-set -> {_derived.get('block_size')}"
            ),
            "mpi_np": ("auto" if c.mpi_np is None else str(c.mpi_np)),
            "omp_threads": ("auto" if c.omp_threads is None
                            else str(c.omp_threads)),
        },
        bench_marks=lambda st, c: _bench_marks_for(
            st, c, _derived.get("block_size"),
            _block_size_bounds(st.n_atoms, c.mpi_np,
                               gpu_mode=bool(c.use_gpu),
                               emitted=_derived.get("block_size"))),
        created_by="molbuilder render_fdf",
        check_rules=_layout.check_rules,
        # WHAT the settings gate judges: the structure as this deck
        # expresses it, not as it arrived.
        validate_subject=lambda s, c: (validation_struct, {"cell": cell}),
    )
    def _science(struct, cfg) -> str:
        """The deck, built in the order SIESTA reads it.

        **Lazy, like PySCF's**: it runs when the framework walks the
        layout, not before.  That is what lets the SPEC exist without
        rendering first -- and therefore what lets a spec, rather than
        finished text, cross the seam (§ 4.3).
        """
        out: List[str] = []

        # job-layout v1 hint -- the basename ``cfg.system_label`` is the
        # protocol's "job name"; every output / restart file shares it.
        # Suggest the canonical ``mpirun`` invocation so the user redirects
        # stdout to ``<basename>.out`` (the Watch tab's discovery chain
        # also looks for that filename).  See docs/execution/job-contracts.md.
        #
        # Stage-aware filenames: ``stage_token`` (``01_coarse`` --
        # ``identity.stage_token``) arrives as a RENDER ARGUMENT from the caller
        # that holds the StageRef (C7, 2026-08-12 -- it rode a config field
        # until then, which was the emitter learning the word stages.md § 1.1
        # forbids).  Every name MOLBUILDER chooses picks it up, so a ladder
        # produces ``<label>_01_coarse.fdf``, ``…_01_coarse.out`` and
        # ``…_01_coarse.molwatch.log`` and a stage's deck matches its own log.
        #
        # The SystemLabel itself stays unsuffixed, which is the whole reason the
        # ladder works: SIESTA writes and reads ``<SystemLabel>.XV`` / ``.DM`` /
        # ``.CG``, so the next stage finds the last one's geometry with no copying
        # and no instruction (decision 26 -- engine-named files stay bare).
        #
        # This wrote ``-stage<N>`` until 2026-08-10.  ``-`` announces *a counter
        # follows* (``job-contracts.md`` § 6.3) and a stage is not a counter, and a
        # bare position silently reassigns outputs when the ladder grows (R5).
        from ..trajectory_log.format import molwatch_log_basename
        _stage_suffix = f"_{stage_token}" if stage_token else ""
        _fdf_name  = f"{cfg.system_label}{_stage_suffix}.fdf"
        _out_name  = f"{cfg.system_label}{_stage_suffix}.out"
        _mw_name   = molwatch_log_basename(cfg.system_label, stage_token)
        if cfg.verbose_comments:
            out.append("# === Run with (job-layout v1) ===")
            out.append(
                "# Run from this directory -- all outputs share the "
                "SystemLabel basename below.")
            # the deck's OWN rank count (R11 -- a hardcoded 4 contradicted
            # the BENCH-MARKS mpi_np three blocks down; "auto" says the
            # wrapper decides)
            _np_hint = "auto" if cfg.mpi_np is None else str(int(cfg.mpi_np))
            out.append(f"#     mpirun -np {_np_hint} siesta "
                       f"< {_fdf_name} > {_out_name}"
                       + ("   # -np auto: the wrapper resolves it"
                          if cfg.mpi_np is None else ""))
            if stage_token:
                out.append(f"# Stage {stage_token} -- {_stage_science(cfg)}")
                # WHAT THIS STAGE ACTUALLY DOES WITH THE PREVIOUS ONE'S STATE.
                # This said "SIESTA reads .XV / .DM from the previous stage"
                # unconditionally, on every staged deck -- including one whose own
                # restart group two hundred lines below says `.false.` three times.
                # A deck that contradicts itself at its first screenful is worse
                # than one that says nothing: the banner is what a person reads to
                # decide whether the ladder is chaining, and it answered yes for
                # every rung including the ones that start fresh.
                if continues(cfg):
                    out.append(
                        "# This stage CONTINUES: SIESTA reads the .XV / .DM the "
                        "previous run left under the same")
                    out.append(
                        "# SystemLabel, in this directory.  See the Watch tab's "
                        "'Staged relaxation workflow' panel.")
                else:
                    out.append(
                        "# This stage starts CLEAN: the restart group below is "
                        "'.false.', so any .XV / .DM in this")
                    out.append(
                        "# directory is left unread and the run starts from the "
                        "coordinates in this file.")
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

        # Extensible annotation channels (model/structure-annotations.md § 4b -- the
        # one-way translation into engine input): emit fdf for
        # any Structure.annotations channel that carries a REGISTERED fdf
        # strategy.  No registered strategies / no annotations -> no-op (the
        # frozen/region built-ins above are untouched).
        from ..annotations_fdf import emit_channels as _emit_channels
        _channel_lines = _emit_channels(struct)
        if _channel_lines:
            out += _channel_lines
            out.append("")

        return "\n".join(out)

    def _dispersion_template(struct, cfg) -> Optional[str]:
        """The MM.Potentials stub, for an XC that has no dispersion of its
        own.  Between the exchange-correlation settings and the SCF ones,
        which is where a reader looking at the functional will be.
        """
        out: List[str] = []
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
        return "\n".join(out) if out else None

    def _restart_group(struct, cfg) -> Optional[str]:
        """The three warm-start keys, written from one declaration.

        A block and not a section because it is not a run of catalogue
        items: ONE field (``restart``) expands into three keywords, and
        the expansion is the catalogue's ``[item.restart].expands`` rather
        than a layout of three rows.
        """
        out: List[str] = []

        # ---- the restart group, whole, from its declaration ---------------
        # ONE field decides it (`restart`) and ONE object declares its members
        # (the catalogue's `[item.restart].expands`).  Both answers are written:
        # continue, `.false.` to start clean -- SIESTA reads the files when they
        # are present unless told otherwise, so omission says nothing.  See
        # `_restart_group_lines`.
        if v: out += [
            "",
            "# Start from: whether this run reads the state a previous one left",
            "# under the same SystemLabel -- the geometry (.XV), the converged",
            "# density (.DM) and, for a relaxation, the optimizer's own history",
            "# (.CG).  One field in the description decides all of them",
            "# (run-identity.md § 4), and every member is written either way:",
            "# SIESTA reads these files when they are there unless a deck says",
            "# .false., so leaving a key out is not a way to decline it.",
            f"# This run: {'continue' if continues(cfg) else 'clean'}.",
        ]
        out += _restart_group_lines(cfg)

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
        return "\n".join(out) if out else None

    def _after_spin(struct, cfg) -> Optional[str]:
        """The spin notes, the net charge and the k-grid.

        The notes open with the same guard the section above them is chosen
        by: a note about a keyword the deck did not write would be a claim
        with nothing behind it.  The charge and the k-grid follow because
        neither is a run of catalogue items -- one is auto-detected from the
        structure, the other is an fdf ``%block``.
        """
        out: List[str] = []
        if cfg.spin_treatment != "non-polarized":
            if cfg.spin_total is not None and cfg.spin_treatment == "polarized":
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
                    # is unusual -- the cheaper path is Spin non-polarized
                    # (spin-restricted Kohn-Sham).  Surface this so a user
                    # who landed here by accident sees the contradiction.
                    out += [
                        "# NOTE: spin_total = 0.0 with Spin polarized asks",
                        "# for a constrained singlet via open-shell DFT (broken-",
                        "# symmetry capable).  Most users wanting a singlet are",
                        "# better served by Spin non-polarized -- the",
                        "# spin-restricted formalism is cheaper and gives the",
                        "# same answer.  Keep this if you specifically want",
                        "# anti-ferromagnetic / broken-symmetry singlet.",
                    ]
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

        # k-grid.  The block's fourth column is SIESTA's ``displ(3)`` -- the grid
        # ORIGIN, in units of one mesh spacing.  It was hard-coded 0.0 here until
        # 2026-08-14; it is now a config item, so the classic Monkhorst-Pack shift
        # (0.5 on an even mesh) is expressible.
        kx, ky, kz = cfg.kgrid
        dx, dy, dz = cfg.kgrid_displacement
        shifted = any(float(d) != 0.0 for d in (dx, dy, dz))
        _shift_note = f", displaced {dx} {dy} {dz}" if shifted else ""
        out.append(f"# --- k-points ({kx}x{ky}x{kz}{_shift_note}) ---")
        # THE REASONS COME FROM THE DECLARATIONS.  Both items write one
        # `%block kgrid_Monkhorst_Pack` -- the counts and the shift -- so the block
        # itself is emitted below rather than through the per-parameter door; a
        # multi-line structure is not a `key value` line.  What does go through the
        # catalogue is why each is what it is, and the nine hand-typed lines that
        # stood here were a thinner copy of exactly that: they lost the equivalent
        # cutoff (the number that makes two different cells comparable), the
        # per-axis independence, and SIESTA's transport-direction override.
        if v:
            out += _sc.parameter("kgrid", "siesta").note()
            out += _sc.parameter("kgrid_displacement", "siesta").note()
        out.append("%block kgrid_Monkhorst_Pack")
        out.append(f"{kx} 0 0 {float(dx)}")
        out.append(f"0 {ky} 0 {float(dy)}")
        out.append(f"0 0 {kz} {float(dz)}")
        out.append("%endblock kgrid_Monkhorst_Pack")
        # No blank line here: the section that follows opens with one, because
        # the framework separates every section from what precedes it.

        # No blank line here either: every section opens with one.

        return "\n".join(out) if out else None

    def _after_geometry(struct, cfg) -> Optional[str]:
        """The blank line that closes the geometry settings.

        ``None`` for a single point, where the section above it is not in
        the layout at all and there is nothing to close.
        """
        if _derived.get('relax_kind') is None:
            return None
            # MD.UseSaveCG is NOT emitted here any more.  It is a member of the
            # restart group like the other two, and the group is written in ONE
            # place from its declaration (`_restart_group_lines`, above the
            # k-grid).  It was written here, several hundred lines from its
            # siblings, each site testing `continues` for itself -- which is how
            # "one field, one group" stayed true of the prose and not of the deck.
            #
            # Which run modes have a use for it is unchanged and lives in
            # `_restart_keys`: every RELAXATION (CG, Broyden, FIRE) and neither
            # dynamics mode, because a Verlet or Nosé run integrates rather than
            # optimizes and has no optimizer history to reload.  Whether that is
            # the right SIESTA semantics is still open -- it wants the manual and
            # a science review -- and moving where the group is written was not
            # the moment to answer it.
        return ""

    def _troubleshooting(struct, cfg) -> str:
        """Tuning hints, after the output settings.

        A LAYOUT MEMBER, not a tail this writer appends: the section
        above it is one too, and a block that swallowed both would put
        the output settings back where the framework cannot see them
        (`script-preparation.md` § 4.1).
        """
        out: List[str] = []

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
                "#   * lower SCF.Mixer.Weight to 0.005",
                "#   * increase SCF.Mixer.History to 5-8",
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
                "#   * lower SCF.Mixer.Weight to 0.005",
                "#   * raise SCF.Mixer.History to 6",
                "#   * tighten DM.EnergyTolerance to 1e-5 eV",
                "#",
                "# 'propor: ERROR: IMAX = 0' on parallel run:",
                "#   * too many MPI ranks for this molecule's radial-",
                "#     function tables (matel_table's proportionality",
                "#     check; the tables come from the .psml",
                "#     pseudopotentials, so a bad or mismatched .psml is",
                "#     the usual root cause -- molbuilder validates the",
                "#     .psml set before rendering).  Retry with a smaller",
                "#     rank count -- the wrapper's ``-np`` flag.",
                "#     BlockSize does NOT fix this: an empirical sweep",
                "#     crashed identically at BlockSize 1, 2, 4",
                "#     (2026-05-28).",
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
                    "#   * shrink MD.MaxDispl to 0.02 Ang",
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
        # ----- ONE DeckSpec, and the framework runs the step -----
        # The reader's section, the record blocks and the banner are the
        # framework's (`script-preparation.md` § 4.2a).  This writer assembled them
        # itself until 2026-08-18 -- as did PySCF's, which made them two copies of
        # one idea and left `render_deck` with no caller.  What stays here is what
        # only SIESTA can say: which VALUES its provenance and bench-marks rows
        # carry.
        #
        # BENCH-MARKS is always emitted for a `.fdf`; the `MD.Steps` anchor is
        # universal across CG / Broyden / FIRE.  ATOM-METADATA emits nothing when
        # regions and frozen atoms are both empty -- absence is the honest signal.
        return "\n".join(out) if out else None

    return spec


def render_fdf(struct: Structure, config: Optional["SiestaConfig"] = None,
               *, cell=None, stage_token: Optional[str] = None) -> str:
    """Format a Structure as SIESTA .fdf text.

    **A thin call over :func:`spec_for`.**  The engine describes its deck; the
    framework renders it.  This name survives because twenty test files and two
    shipped routes point at it -- what moved is what it does, not what it is
    called (`archive/2026-08-18-preparation-backend-plan.md` § 3.1a).

    Prefer ``spec_for`` + ``script_emit.prepare_deck`` where a deck is being
    WRITTEN: that runs validate -> render -> write -> check in one place, and
    the order then has one owner rather than one per caller (§ 4.3).
    """
    spec = spec_for(struct, config, cell=cell, stage_token=stage_token)
    cfg = config or SiestaConfig()
    return _sc.render_deck(spec, struct, cfg,
                           verbose=cfg.verbose_comments)
# --------------------------------------------------------------------- #
#  File -> (Structure, cell) loader                                     #
# --------------------------------------------------------------------- #


def _emit_dispersion_template(xc_authors: str, v: bool) -> List[str]:
    """Commented-out dispersion-correction template emitted when
    XC.functional is non-vdW (PBE / BLYP / hybrids).  See gap #3 in
    docs/design.md and docs/engines/siesta.md § "Reference sources".

    The template is commented so the default behaviour is unchanged
    (the user opts in by uncommenting).  Three routes are offered: a
    vdW-aware XC; ``DFTD3 T`` (one line, parameters matched to the
    functional) for a SIESTA built against s-dftd3; and the older D2
    pair potential via ``MM.Potentials`` for one that is not.

    Every value here is checked against the SIESTA manual entry for
    ``MM.Potentials`` (Docs/tex/sections/Options/Auxiliary_force_field
    .tex) and against the block parser in Src/molecularmechanics.F90.
    The example rows are ``Util/Grimme``'s own output (installed as
    ``fdf2grimme``), so they are what SIESTA would write for itself.
    """
    # The fdf lines themselves -- identical in both the terse and the
    # annotated deck, because they are what the user actually pastes.
    # Copied from `fdf2grimme`'s output for a C/H deck.
    d2_block = [
        "# MM.UnitsEnergy   eV     # units of the C6 column",
        "# MM.UnitsDistance Ang    # units of the R0 column",
        "# MM.Grimme.S6     0.75   # PBE 0.75, BLYP 1.20, B3LYP 1.05",
        "# MM.Grimme.D      20.    # damping steepness d (D2 uses 20)",
        "# %block MM.Potentials",
        "#   1  1  Grimme   18.14   2.904   # C / C",
        "#   1  2  Grimme    5.13   2.453   # C / H",
        "# %endblock MM.Potentials",
    ]

    out: List[str] = ["# --- Dispersion correction (commented template) ---"]
    if not v:
        out += [
            "# DFT-D3 on top of the current XC -- one line, no table:",
            "#      DFTD3 T",
            "# ...or Grimme-D2 pair potentials, which SIESTA's own",
            "# `fdf2grimme <this file>` writes for you:",
        ]
        return out + d2_block

    out += [
        f"# {xc_authors} is a non-dispersive XC: long-range vdW",
        "# (C6/r^6) is missing.  Organic / biomolecule consequences:",
        "#   * DNA stacking under-bound by 5-10 kcal/mol per pair",
        "#   * peptide folding favours wrong conformers",
        "#   * molecular crystals: lattice constants too long by 0.1-0.3 A",
        "#   * surface adsorption (benzene/graphite, etc.) off ~10x",
        "# Three ways to fix.  Pick ONE:",
        "#",
        "# 1) Switch to a vdW-aware XC (cheapest correctness):",
        "#      XC.functional VDW",
        "#      XC.authors    DRSLL    (or KBM, LMKLL, BH, VV)",
        "#    The non-local correlation lives in the functional;",
        "#    no dispersion block needed.",
        "#",
        "# 2) DFT-D3 on top of the current XC -- one line, no table:",
        "#      DFTD3 T",
        "#    SIESTA then picks D3 parameters matched to the",
        "#    functional (DFTD3.UseXCDefaults, default true: PBE,",
        "#    PBESol, RevPBE, RPBE, LYP, BLYP -- and HSE06 / PBE0",
        "#    with LibXC).  Newer and better than D2: the C6 depend",
        "#    on each atom's coordination, and a 3-body term is",
        "#    included.  Requires a SIESTA built against s-dftd3;",
        "#    if yours is not, use route 3 below.",
        "#",
        "# 3) Grimme-D2 via the molecular-mechanics pair potential.",
        "#    Do NOT type this table by hand -- SIESTA ships a utility",
        "#    that writes it from this very deck:",
        "#      fdf2grimme <this file>",
        "#    (SIESTA's Util/Grimme; installed under that name in",
        "#    molbuilder's siesta env).  Paste its output, which looks",
        "#    like this:",
    ]
    out += d2_block
    out += [
        "#",
        "#    Set MM.Grimme.S6 yourself: SIESTA's default is 1.66, a",
        "#    DZ-basis value, NOT your functional's.",
        "#    Column by column:",
        "#      1,2  the two SPECIES NUMBERS from ChemicalSpeciesLabel",
        "#           -- integers.  Element symbols are NOT accepted:",
        "#           SIESTA counts the line but cannot parse it, then",
        '#           stops with "Too many lines in MM.Potentials block',
        '#           are not read".',
        "#      3    the potential name, Grimme",
        "#      4    C6 for the PAIR, in eV*Ang^6:",
        "#             C6_ij = sqrt(C6_i * C6_j)",
        "#      5    R0 for the PAIR, in Ang -- the SUM of the two",
        "#           atomic vdW radii, already carrying Grimme's 1.1",
        "#           correction factor:  R0_ij = R0_i + R0_j",
        "#    SIESTA evaluates",
        "#      E = -s6 * C6_ij / r^6 * 1/(1+exp(-d*(r/R0_ij - 1)))",
        "#    so R0_ij is the distance where damping is 1/2.  Halving",
        "#    it (passing a per-atom radius) stops suppressing the",
        "#    term between BONDED atoms, double-counting binding that",
        "#    DFT already describes.",
        "#",
        "#    Grimme 2006 (J. Comput. Chem. 27, 1787) tabulates the",
        "#    PER-ATOM values in J*nm^6/mol and nm.  To convert:",
        "#      C6[eV*Ang^6] = C6[J*nm^6/mol] * 10.36",
        "#      R0[Ang]      = R0[nm] * 10",
        "#    Carbon: 1.75 -> 18.14 eV*Ang^6, and 0.1452 nm -> 1.452",
        "#    Ang, so a C-C PAIR takes R0 = 2.904.",
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
    from ..workingcopy_structure import StructureCodec
    if ext == ".pdb":
        return StructureCodec().load(p), None
    if ext in (".xyz", ""):
        # THROUGH THE ONE READER.  This used to call ``ase.io.read`` here,
        # with a comment explaining that ASE "understands extended-XYZ headers
        # and gives us the lattice when present, which our hand-rolled parser
        # doesn't" -- a correct diagnosis fixed in the wrong place, by adding a
        # SECOND reader beside the lossy one instead of fixing it.  Every other
        # caller kept the lossy one.  ``Structure.from_xyz`` is ASE now, so the
        # lattice arrives on the structure and there is one reader again.
        # THROUGH THE CODEC, which is the one reader of a structure AND the
        # sidecar beside it (`model/structure.md` § 2.4).  The comment below
        # records this lesson being learned once already -- a second reader
        # added beside the lossy one instead of fixing it -- and it stopped one
        # level short: `from_xyz` became the one COORDINATE reader, while the
        # PAIR has a different one, and this route kept the half that drops the
        # other file.  So a structure carrying frozen atoms, region labels, an
        # explicit cell or a stated vacuum went through here and came out
        # without them: the `.molstruct.json` sat unread beside the `.xyz`, and
        # the script relaxed every atom of a structure whose author had frozen
        # two.  `prep` and the web route have always used the codec; only the
        # single-shot converters did not.
        #
        # The codec applies the sidecar when it is there and changes nothing
        # when it is not, so a bare `.xyz` reads exactly as before.
        struct = StructureCodec().load(p)
        cell = struct.cell
        return struct, (np.asarray(cell, dtype=float)
                        if cell is not None else None)
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
    fdf_p = Path(fdf_path)
    fdf_p.parent.mkdir(parents=True, exist_ok=True)
    # STEP 3, WHOLE, IN ONE CALL -- the same call `prep` makes.
    # THE CHECK GATE RUNS ON EVERY ROUTE THAT WRITES A DECK: `prep` is not the
    # only door, this is the CLI's, and a deck naming a keyword twice is no
    # less wrong for having been produced here.  The order is the framework's
    # and is not restated per route (`script-preparation.md` § 4.3).
    _sc.prepare_deck(spec_for(struct, cfg, cell=cell),
                     struct, cfg, fdf_p)

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
        # The one anchor rule (job-contracts.md § 2.5a), anchored on the
        # calculation the .fdf is being written into.  A bare
        # `Path(...).expanduser()` stood here until 2026-08-21 and made every
        # relative spelling working-directory-relative -- so `convert` and
        # `prep` disagreed about what the same template meant.
        from ..pseudos import (PsmlLibError, describe_psml_anchor,
                               resolve_psml_lib)
        try:
            lib = resolve_psml_lib(str(cfg.psml_lib), dest_dir=fdf_p.parent)
        except PsmlLibError as exc:
            lib = None
            print(f"  WARN: skipping psml copy -- {exc}", file=sys.stderr)
        if lib is not None and not lib.is_dir():
            print(f"  WARN: skipping psml copy -- "
                  f"{describe_psml_anchor(str(cfg.psml_lib), dest_dir=fdf_p.parent)}",
                  file=sys.stderr)
        elif lib is not None:
            summary["missing_psml"] = copy_pseudopotentials(species, lib, fdf_p.parent)

    # Drop a preview <basename>.molwatch.log next to the
    # .fdf so molwatch can render the initial geometry the moment the
    # user loads it -- no waiting for SIESTA to write its first
    # outcoor block.  The file is static (one preview block, no live
    # updates); for live updates while SIESTA is running, point
    # molwatch at the .out file instead.
    #
    # Filename derives from cfg.system_label (the protocol basename) --
    # NOT from the FDF's stem (`convert` is the single-shot path and has
    # no stage; a ladder's logs are seeded by `prep`, which holds the
    # token).  This
    # way a user who names the FDF "anything.fdf" still gets the
    # canonical preview-log name that the Watch tab discovery chain
    # recognises.  See docs/execution/job-contracts.md.
    if cfg.write_molwatch_log:
        from ..trajectory_log import molwatch_log_basename, write_initial_preview
        mw_path = fdf_p.parent / molwatch_log_basename(
            cfg.system_label, None)
        write_initial_preview(
            struct,
            mw_path,
            job=cfg.system_label,
            engine="siesta",
        )
        summary["molwatch_log"] = str(mw_path)

    return summary
