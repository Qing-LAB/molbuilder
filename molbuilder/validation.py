"""Pre-emission validation for SIESTA / PySCF input generation.

The check list lives in ``docs/design.md`` § "Validation pass
(pre-emission)".  This module is the in-code expression of that
table.  Generators (siesta.input.render_fdf, pyscf.input.render_script)
call :func:`validate` before writing output; errors block emission,
warnings print to stderr.

Design principles realised by this module:

  * **Principle #1** ("the dataclass is the lingua franca"):
    config-field validators (range / `validate=` callable) are read
    off ``dataclasses.field(metadata=...)`` -- no parallel lookup
    table, so adding a new field with a range is a one-line change
    to the dataclass.

  * **Principle #6** ("pre-emission geometry validation"): every
    structure-side check from the table is enforced here, before any
    SIESTA / PySCF text is emitted.

The output is a ``List[Issue]``.  Callers decide what to do with it;
:func:`report` is the "raise errors, print warnings to stderr" helper
that ``render_fdf`` / ``render_script`` use.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from .issues import Issue, ValidationError
from .structure import Structure


# --------------------------------------------------------------------- #
#  Engine-validator registry                                            #
#                                                                        #
#  Type-keyed dispatch.  Each engine config class registers an          #
#  engine-specific validator; `validate()` looks it up by              #
#  isinstance().  Adding a new engine is a `_ENGINE_VALIDATORS[T] = fn` #
#  line, not a string-compare in `validate()`.                          #
# --------------------------------------------------------------------- #


_ENGINE_VALIDATORS: Dict[Type, Callable[[Structure, object, Optional[np.ndarray]], List[Issue]]] = {}


def _register_engine_validator(cfg_cls: Type):
    """Decorator: register a validator for a specific config class.

    The validator receives (struct, cfg, cell) and returns a list of
    Issues.  ``cell`` may be None for engines that don't have a
    periodic cell concept (PySCF gas-phase / PCM); each registered
    validator decides what to do with it.
    """
    def deco(fn):
        _ENGINE_VALIDATORS[cfg_cls] = fn
        return fn
    return deco


# --------------------------------------------------------------------- #
#  Top-level entry point                                                #
# --------------------------------------------------------------------- #


def validate(struct: Structure, cfg, *,
             cell: Optional[np.ndarray] = None) -> List[Issue]:
    """Run every applicable validation check and return the findings.

    Parameters
    ----------
    struct
        The Structure about to be emitted.
    cfg
        SiestaConfig or PySCFConfig (or any dataclass; the generic
        config-field metadata pass runs on anything dataclass-shaped).
    cell
        Optional pre-computed (3, 3) lattice the generator is going
        to use.  If None, cell-dependent checks are skipped.  The
        SIESTA generator computes the cell anyway, so it should pass
        the same matrix here.

    The returned list is in deterministic order: generic geometry
    checks first, generic config-field checks next, then engine-
    specific checks.  Callers can sort / filter as they please.
    """
    issues: List[Issue] = []
    issues += validate_geometry(struct, cell)
    issues += _validate_config_metadata(cfg)

    # Engine-specific dispatch via the registry.  isinstance() picks
    # up subclasses too, so a future engine config that subclasses
    # an existing one inherits its validator unless it registers its
    # own.
    for cfg_cls, fn in _ENGINE_VALIDATORS.items():
        if isinstance(cfg, cfg_cls):
            issues += fn(struct, cfg, cell)
            break
    return issues


def report(issues: List[Issue], *,
           raise_on_error: bool = True,
           stream=None) -> None:
    """Print warnings to stderr; raise ValidationError on errors.

    The two-pass shape (warnings first, then maybe-raise) lets the
    user see *all* the warnings even when an error is also present --
    helpful when triaging a misconfigured run.
    """
    if stream is None:
        stream = sys.stderr
    for i in issues:
        if i.severity == "warn":
            tag = f" [{i.where}]" if i.where else ""
            print(f"warn{tag}: {i.message}", file=stream)
    errors = [i for i in issues if i.severity == "error"]
    if errors and raise_on_error:
        raise ValidationError(issues)


# --------------------------------------------------------------------- #
#  Generic geometry checks                                              #
# --------------------------------------------------------------------- #


def validate_geometry(struct: Structure,
                      cell: Optional[np.ndarray] = None) -> List[Issue]:
    """Run only the geometry-side checks (no config / engine dispatch).

    Useful for surfaces that don't have a cfg yet -- e.g. the web
    Build page wants to flag a heavy-atom-only structure as soon as
    the user clicks Build, before they even pick SIESTA vs PySCF.

    Cell-dependent checks (volume / image distance / determinant) are
    skipped when ``cell is None``; the always-applicable checks
    (min atom distance, H/heavy ratio) still run.
    """
    issues: List[Issue] = []
    pos = struct.positions
    n   = len(pos)

    # Min atom-atom distance.  An O(N^2) pass is fine at the scale
    # this package targets (< 10k atoms); KD-tree only helps once the
    # constant overhead is amortised.
    if n >= 2:
        # Pairwise distance matrix without the diagonal.
        d  = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        min_d = float(d.min())
        if min_d < 0.3:
            issues.append(Issue(
                "error",
                f"closest atom pair is {min_d:.3f} Å apart -- atoms are "
                f"effectively on top of each other; SCF will diverge",
                "geometry.min_distance",
            ))
        elif min_d < 0.7:
            issues.append(Issue(
                "warn",
                f"closest atom pair is {min_d:.3f} Å apart -- this is "
                f"too short for any real bond; check for failed "
                f"protonation or backend output corruption",
                "geometry.min_distance",
            ))

    # H/heavy ratio.  Heavy-atom-only skeletons (typically X3DNA's `fiber`
    # output with `add_hydrogens=False`, or a hand-loaded heavy-atom PDB)
    # produce the wrong total electron count in DFT and are missing the
    # Watson-Crick / hydrogen-bond donors that hold the chemistry together.
    # Typical organic molecules sit at H/heavy ~ 0.6-1.5; nucleic acids
    # ~ 0.6.  A ratio below 0.3 is unambiguously a heavy-atom skeleton.
    #
    # **Severity: warn, not error.**  The user may legitimately want to
    # inspect or hand-process the heavy-atom skeleton (e.g., feed it to
    # an external protonator with different residue assumptions).  The
    # warning surfaces the issue prominently so they don't accidentally
    # ship a broken structure to a calculation.
    if n >= 1:
        n_h     = sum(1 for e in struct.elements if e == "H")
        n_heavy = sum(1 for e in struct.elements if e != "H")
        if n_heavy > 0 and (n_h / n_heavy) < 0.3:
            issues.append(Issue(
                "warn",
                f"H/heavy ratio is {n_h}/{n_heavy}={n_h/n_heavy:.2f} -- "
                f"structure looks like a heavy-atom skeleton (typical "
                f"organic molecules: H/heavy ~ 0.6-1.5).  DFT will "
                f"compute the wrong electron count without explicit H. "
                f"Did you mean to add hydrogens?",
                "geometry.h_ratio",
            ))

    issues += _check_polymer_orientation(struct)

    if cell is None:
        return issues

    # Cell determinant: <= 0 means left-handed or degenerate.
    det = float(np.linalg.det(cell))
    if det <= 0:
        issues.append(Issue(
            "error",
            f"cell determinant is {det:.3f} -- cell is degenerate or "
            f"left-handed (right-handed lattice vectors required)",
            "cell.determinant",
        ))
        # Don't run the volume / image checks if the cell is broken.
        return issues

    # Cell volume vs atom-bounding-volume: warn when the cell is so
    # tight that the molecule fills most of it (= guaranteed
    # image-image contact in PBC).
    if n >= 1:
        extent = pos.max(axis=0) - pos.min(axis=0)
        atom_box = float(np.prod(np.maximum(extent, 1.0)))   # min 1 Å on each side
        ratio = det / atom_box
        if ratio < 3:
            issues.append(Issue(
                "warn",
                f"cell volume / atom-bounding-volume = {ratio:.2f} -- cell "
                f"is suspiciously tight; expect image-image interactions",
                "cell.volume",
            ))

    # Atom-to-nearest-image (PBC minimum-image distance).
    # For each atom, find the closest image of any OTHER atom under
    # the lattice translations spanning the 27 nearest cells.  When
    # this is < 6 Å (a generous "atoms close enough to interact"
    # threshold; below the typical electronic exchange-correlation
    # cutoff), warn -- the user is likely seeing image-image overlap.
    if n >= 2:
        try:
            inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            inv = None
        if inv is not None:
            min_image = _min_image_distance(pos, cell, inv)
            if min_image < 6.0:
                issues.append(Issue(
                    "warn",
                    f"min atom-to-nearest-image distance is "
                    f"{min_image:.2f} Å -- molecule images interact through "
                    f"the periodic boundary; increase cell vectors or "
                    f"cell_padding so this exceeds ~6 Å",
                    "cell.image_distance",
                ))

    return issues


def _min_image_distance(positions: np.ndarray,
                        cell: np.ndarray,
                        inv:  np.ndarray) -> float:
    """Closest distance between any atom and any atom in a NEIGHBOURING
    cell (translation != (0, 0, 0)).

    The zero-translation case is excluded entirely: in-cell distances
    are real bonds, not images.  We only care about how close the
    molecule sits to its periodic copies.

    For each atom i, distance to (atom j shifted by t) is computed
    over all 26 non-identity lattice translations and over all atoms
    j (including j == i, which is "this atom seeing its own copy").
    """
    n = positions.shape[0]
    if n == 0:
        return float("inf")
    # 26 non-identity translations spanning the immediate neighbour shell.
    shifts = [(a, b, c)
              for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
              if (a, b, c) != (0, 0, 0)]
    translations = np.asarray(shifts, dtype=float) @ cell   # (26, 3)
    best = float("inf")
    for i in range(n):
        # Vector from atom i to every (atom j + every non-zero translation):
        deltas = (positions[None, :, :] + translations[:, None, :]
                  - positions[i, None, None, :])
        d = float(np.linalg.norm(deltas, axis=2).min())
        if d < best:
            best = d
    return best


# --------------------------------------------------------------------- #
#  Polymer orientation                                                  #
#                                                                       #
#  For nucleic acids, every backend builds residues 5' -> 3' (lowest    #
#  residue_id at the 5' end).  If a future backend (or a user-loaded    #
#  PDB from an external tool) lists residues 3' -> 5', the polymer is  #
#  chemically the same but the residue listing is reversed -- which     #
#  silently breaks any downstream code that infers orientation from    #
#  residue_ids[0] (CIF / PDB writers, web "Watch this run" handoff,    #
#  the X3DNA 5'-phosphate strip in `_threedna._strip_5prime_phosphate`).#
#                                                                       #
#  This check looks at the actual P-O3' bridges to find the structural  #
#  5' end (the residue with NO incoming bridge) and warns when it       #
#  doesn't match the lowest-numbered residue.                          #
# --------------------------------------------------------------------- #


def _check_polymer_orientation(struct: Structure) -> List[Issue]:
    if struct.residue_ids is None or struct.atom_names is None:
        return []
    # Locate P and O3' positions per residue.  If neither is present
    # this isn't a nucleic-acid polymer (or it's a heavy-atom-only
    # build with no backbone atoms named) -- silently skip.
    P_pos:  Dict[int, np.ndarray] = {}
    O3_pos: Dict[int, np.ndarray] = {}
    for i in range(struct.n_atoms):
        rid = struct.residue_ids[i]
        nm  = struct.atom_names[i]
        if nm == "P":
            P_pos[rid] = struct.positions[i]
        elif nm == "O3'":
            O3_pos[rid] = struct.positions[i]
    if not P_pos and not O3_pos:
        return []

    rids = sorted(set(struct.residue_ids))
    # has_predecessor[r] = True if residue r's P bonds to residue (r-1)'s O3'.
    has_predecessor = set()
    for r in rids:
        if r in P_pos and (r - 1) in O3_pos:
            d = float(np.linalg.norm(P_pos[r] - O3_pos[r - 1]))
            if d < 1.8:                                # bridged
                has_predecessor.add(r)

    five_prime_ends = [r for r in rids if r not in has_predecessor]
    if not five_prime_ends:
        # cyclic polymer -- nothing to orient against
        return []
    if len(five_prime_ends) > 1:
        # Multiple chains, branched, or multiple disconnected pieces.
        # If the structure has a single chain_id this is a polymer
        # integrity issue worth surfacing; multi-chain inputs (e.g.,
        # a duplex) get a pass.
        if struct.chain_ids is not None and len(set(struct.chain_ids)) <= 1:
            return [Issue(
                "warn",
                f"polymer has {len(five_prime_ends)} residues with no "
                f"preceding O3'-P bridge (residues {five_prime_ends}); "
                f"single-chain input expected exactly one 5' end.  "
                f"Possible disconnected backbone or unintended branching.",
                "polymer.orientation",
            )]
        return []

    # Exactly one 5' end -- it should be the lowest-numbered residue
    # (every backend builds 5' -> 3', so residue_ids[0] = 5' terminus).
    if five_prime_ends[0] != rids[0]:
        return [Issue(
            "warn",
            f"residue listing appears reversed: structural 5' end is "
            f"residue {five_prime_ends[0]} but residue_ids start at "
            f"{rids[0]}.  Backends should list 5' -> 3' (lowest residue_id "
            f"at the 5' end); a mismatch breaks downstream orientation-"
            f"sensitive code (terminal-phosphate stripping, FDF residue "
            f"numbering).  Likely a backend regression.",
            "polymer.orientation",
        )]
    return []


# --------------------------------------------------------------------- #
#  Generic config-field metadata pass                                   #
#                                                                        #
#  Reads `range` and `validate` off dataclass field metadata and        #
#  produces Issues.  This is what makes Principle #1 load-bearing:      #
#  field metadata IS the source of truth; CLI / web / validators all   #
#  read from the same place.                                            #
# --------------------------------------------------------------------- #


def _validate_config_metadata(cfg) -> List[Issue]:
    issues: List[Issue] = []
    if not is_dataclass(cfg):
        return issues
    for f in fields(cfg):
        meta = f.metadata or {}
        value = getattr(cfg, f.name)
        # range = (lo, hi) inclusive
        rng = meta.get("range")
        if rng is not None and value is not None:
            lo, hi = rng
            try:
                if value < lo or value > hi:
                    label = meta.get("label", f.name)
                    unit = f" {meta['unit']}" if meta.get("unit") else ""
                    issues.append(Issue(
                        "warn",
                        f"{label} = {value}{unit} is outside the "
                        f"recommended range [{lo}, {hi}]{unit}",
                        f"config.{f.name}",
                    ))
            except TypeError:
                # Field metadata claims a numeric range but the value
                # isn't comparable -- silently skip rather than crash;
                # a misconfigured metadata entry shouldn't break runs.
                pass
        # Optional callable: meta["validate"] -> Issue or None
        validator = meta.get("validate")
        if validator is not None:
            try:
                result = validator(value, cfg)
            except Exception:
                continue
            if isinstance(result, Issue):
                issues.append(result)
            elif isinstance(result, list):
                issues.extend(i for i in result if isinstance(i, Issue))
    return issues


# --------------------------------------------------------------------- #
#  SIESTA-specific checks                                               #
# --------------------------------------------------------------------- #


def _check_peptide_protonation(struct: Structure,
                               cfg_charge) -> List[Issue]:
    """Hint at the gap between gas-phase neutral build and
    physiological charge state for peptides with charged side chains.

    PeptideBuilder + AddHs produces a neutral molecule by default
    (Asp / Glu protonated, Lys / Arg uncharged amines).  At pH 7 the
    charged side chains carry a net charge.  Most users don't realise
    the script is silently using the gas-phase neutral form.

    Triggered only when:
      * the structure looks like a peptide (has standard amino-acid
        residue names);
      * the estimated pH-7 charge is non-zero;
      * the user hasn't explicitly set cfg.charge to a non-zero
        value (None or 0 means "auto / default neutral").

    Severity: warn (not error).  The neutral build may be exactly
    what the user wants -- the surface emits SIESTA / PySCF input
    that runs without modification.  This warning surfaces the
    INFORMATION gap, not a bug.
    """
    from .chemistry import expected_pH7_peptide_charge
    expected = expected_pH7_peptide_charge(struct)
    if expected is None or expected == 0:
        return []
    # cfg_charge None -> auto-detection path; cfg_charge == 0 -> the
    # user explicitly forced neutral.  Both paths produce the same
    # gas-phase build, so both deserve the warning telling them about
    # the side-chain mismatch.  An explicit non-zero cfg_charge means
    # the user already accounted for this -- skip the warning.
    if cfg_charge not in (None, 0):
        return []
    return [Issue(
        "warn",
        f"peptide has charged side chains (estimated charge at "
        f"pH 7.4: {expected:+d}) but cfg.charge = 0; the script "
        f"will build the gas-phase neutral form (Asp/Glu protonated, "
        f"Lys/Arg neutral).  For physiological-state runs set "
        f"cfg.charge = {expected} (and adjust spin / basis: open "
        f"shells need diffuse functions like aug-cc-pVDZ for anions)",
        "config.charge",
    )]


def _validate_siesta(struct: Structure, cfg,
                     cell: Optional[np.ndarray]) -> List[Issue]:
    """SIESTA-specific checks.

    Registered with the engine-validator dispatch at module bottom
    (the decorator is applied after the SiestaConfig type is
    importable -- avoids the import cycle between validation.py and
    siesta/input.py at definition time).
    """
    issues: List[Issue] = []

    # Peptide protonation hint -- same as PySCF side; see
    # _check_peptide_protonation for the full rationale.
    issues += _check_peptide_protonation(struct, getattr(cfg, "net_charge", None))

    # Spin.Total set without spin polarised: SIESTA silently ignores it.
    if cfg.spin_total is not None and not cfg.spin_polarized:
        issues.append(Issue(
            "warn",
            f"spin_total = {cfg.spin_total} is set but spin_polarized "
            f"is False; SIESTA will silently ignore the total-spin pin",
            "config.spin_total",
        ))

    if cell is None:
        return issues

    # k-grid vs cell extent: distinguish three cases per axis:
    #   * vacuum direction (atoms span < 85% of axis) -> k=1 correct,
    #     k>1 wasted
    #   * periodic direction (atoms span > 85% of axis) -> k=1
    #     under-converged when other axes are sampled
    #   * indeterminate (no atoms or single-axis tiny molecule) ->
    #     fall back to the cell-extent heuristic
    #
    # Pre-fix the heuristic used cell-extent alone, which mis-flagged
    # vacuum-padded long axes (e.g. a 12-mer DNA in an 80 Å cell with
    # kgrid (4, 4, 1) along the molecular axis is correct vacuum, not
    # periodic).  Atoms spanning < 85% of an axis means there's
    # vacuum padding at the ends -> the user opted for vacuum on
    # that axis, k=1 is right.
    diag_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    if struct.n_atoms > 0:
        atom_extent = struct.positions.max(axis=0) - struct.positions.min(axis=0)
    else:
        atom_extent = np.zeros(3)
    for axis, (k, length) in enumerate(zip(cfg.kgrid, diag_lengths)):
        # Span ratio: how much of the cell axis the atoms cover.
        # Near 1.0 -> atoms reach edge -> periodic intent.
        # Near 0.0 -> atoms cluster, edges are vacuum -> vacuum intent.
        span_ratio = (atom_extent[axis] / length) if length > 0 else 0.0
        is_periodic_axis = span_ratio > 0.85

        if k != 1 and not is_periodic_axis and length >= 5.0:
            # User asked for k-points on a vacuum-padded axis; rare
            # and almost always wasted cost.  Don't warn for tiny
            # cells (length < 5 Å) where the heuristic is unreliable.
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = {k} along an axis of {length:.1f} Å "
                f"where atoms span only {atom_extent[axis]:.1f} Å "
                f"({span_ratio*100:.0f}%); this looks like a vacuum-padded "
                f"axis -- k>1 there adds cost without improving accuracy",
                "config.kgrid",
            ))
        elif k == 1 and is_periodic_axis and any(kk > 1 for kk in cfg.kgrid):
            # An axis where atoms span the full extent (slab / wire /
            # crystal direction) with k=1 while another axis is
            # sampled -- almost always a forgotten k-grid value.
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = 1 along an axis where atoms span "
                f"{atom_extent[axis]:.1f} of {length:.1f} Å "
                f"({span_ratio*100:.0f}%, looks periodic) while "
                f"another axis uses k > 1; likely under-converged "
                f"sampling on this axis",
                "config.kgrid",
            ))

    # Net dipole > 1 D in vacuum (no dipole correction).  Image-image
    # dipole interactions in PBC shift molecular energies by an amount
    # that scales with the dipole magnitude squared and inversely with
    # the cell size.  We use a heuristic EN-based partial-charge
    # estimate (see chemistry.estimate_dipole_moment_debye) -- not a
    # research-grade dipole, but enough to flag "polar molecule in a
    # finite vacuum cell" and recommend a larger cell or an explicit
    # dipole correction.
    #
    # Triggered only when the cell looks like the auto-vacuum case:
    # all kgrid axes == 1 (Gamma-only sampling, no PBC physics
    # intended).  A genuine periodic crystal with k>1 is meant to
    # carry image-image interactions and shouldn't trip this warning.
    if all(k == 1 for k in cfg.kgrid) and len(struct.positions) > 0:
        try:
            from .chemistry import (estimate_dipole_moment_debye,
                                    formal_charge_from_phosphates)
            net_charge = (cfg.net_charge if cfg.net_charge is not None
                          else formal_charge_from_phosphates(struct))
            dipole = estimate_dipole_moment_debye(struct,
                                                  total_charge=float(net_charge))
        except Exception:
            dipole = 0.0
        if dipole > 1.0:
            issues.append(Issue(
                "warn",
                f"estimated net dipole = {dipole:.1f} D in a vacuum cell "
                f"with no dipole correction -- image-image dipole "
                f"interactions will shift energies; consider a larger "
                f"cell_padding or enable SIESTA's SlabDipoleCorrection "
                f"(estimate from EN-based partial charges; rough +/- 50%)",
                "geometry.dipole",
            ))

    # Atoms outside [0, 1) fractional coords with wrap_into_cell=False
    # mean the visualiser will see the molecule in the neighbour cell.
    if not cfg.wrap_into_cell and len(struct.positions) > 0:
        try:
            inv  = np.linalg.inv(cell)
            frac = struct.positions @ inv
            outside = np.any((frac < 0) | (frac >= 1), axis=1)
            n_out = int(outside.sum())
            if n_out > 0:
                issues.append(Issue(
                    "warn",
                    f"{n_out} of {len(struct.positions)} atoms have "
                    f"fractional coords outside [0, 1) but wrap_into_cell "
                    f"= False; visualisations will show the molecule in "
                    f"the neighbour cell",
                    "config.wrap_into_cell",
                ))
        except np.linalg.LinAlgError:
            pass   # Singular cell -- already flagged by determinant check

    return issues


# --------------------------------------------------------------------- #
#  PySCF-specific checks                                                #
# --------------------------------------------------------------------- #


def _validate_pyscf(struct: Structure, cfg,
                    cell: Optional[np.ndarray] = None) -> List[Issue]:
    """PySCF-specific checks.  ``cell`` is unused (PySCF jobs are gas-
    phase or PCM-solvent here); accepted for signature uniformity
    with the engine-validator registry."""
    issues: List[Issue] = []

    # spin = 2S; must be a non-negative integer.  PySCFConfig exposes
    # spin as an int with default 0; a negative value is meaningless
    # (2S is the count of unpaired electrons, never negative).
    if getattr(cfg, "spin", 0) < 0:
        issues.append(Issue(
            "error",
            f"spin = {cfg.spin} is negative; spin counts unpaired "
            f"electrons (2S), must be 0 or positive",
            "config.spin",
        ))

    # spin > 0 with a closed-shell method (RKS/RHF) is silently wrong:
    # the user wanted open-shell.
    method = (getattr(cfg, "method", "") or "").upper()
    if cfg.spin > 0 and method.startswith("R") and method in ("RKS", "RHF"):
        issues.append(Issue(
            "warn",
            f"spin = {cfg.spin} is set but method = {method} is closed-shell; "
            f"use UKS / UHF for open-shell systems (otherwise SCF "
            f"either fails to converge or quietly returns wrong "
            f"electronic structure)",
            "config.method",
        ))

    # Peptide protonation: PeptideBuilder + AddHs builds the gas-phase
    # NEUTRAL form (Asp / Glu protonated, Lys / Arg neutral, etc.).
    # For sequences containing charged side chains, the physiological
    # charge differs.  Surface the gap so the user knows the script
    # is using neutral defaults; they can override with cfg.charge.
    issues += _check_peptide_protonation(struct, getattr(cfg, "charge", None))

    # Inverse case: UKS / UHF with spin = 0 is almost always a mistake.
    # The unrestricted formalism on a closed-shell system runs at ~2x
    # the SCF cost (separate alpha / beta blocks), is more numerically
    # fragile (broken-symmetry saddle points are reachable), and gives
    # the same answer as RKS / RHF unless the user specifically wanted
    # broken-symmetry (e.g. anti-ferromagnetic singlet).  Warn so the
    # default-of-RKS user who flipped to UKS to "be safe" is told it's
    # the wrong default-of-safe.
    if cfg.spin == 0 and method in ("UKS", "UHF"):
        issues.append(Issue(
            "warn",
            f"method = {method} (unrestricted) with spin = 0 (closed shell) "
            f"runs the unrestricted formalism at ~2x the SCF cost of the "
            f"corresponding R{method[1:]}; switch to R{method[1:]} unless you "
            f"specifically want a broken-symmetry singlet "
            f"(e.g. anti-ferromagnetic system)",
            "config.method",
        ))

    # Hybrid functional (B3LYP, PBE0, M06-2X, wB97X) with grid_level < 4:
    # forces become noisy at the ~1e-4 Ha/Bohr scale the optimizer
    # cares about.  The molecule WILL relax, but the "converged" forces
    # may have a noisy floor that prevents tight convergence.  Warn but
    # allow -- the user may be doing screening at level 3 deliberately.
    grid_level = getattr(cfg, "grid_level", None)
    functional = (getattr(cfg, "functional", "") or "").upper()
    is_hybrid = any(functional.startswith(p) for p in (
        "B3", "PBE0", "M06", "WB97", "BHANDH", "X3LYP", "TPSS0", "MN15",
    ))
    if grid_level is not None and grid_level < 4 and is_hybrid:
        issues.append(Issue(
            "warn",
            f"grid_level = {grid_level} with a hybrid functional "
            f"({functional}): hybrid-DFT forces are noisy at this grid "
            f"density (forces look ~1e-4 Ha/Bohr noise floor).  Bump to "
            f"grid_level = 4 for production geometry optimisation; level "
            f"3 is fine for energies / screening only",
            "config.grid_level",
        ))

    return issues


# --------------------------------------------------------------------- #
#  Engine-validator registration                                        #
#                                                                        #
#  Done at module bottom rather than via decorators on _validate_siesta #
#  / _validate_pyscf because the config classes import from this        #
#  module in some code paths (lift would create an import cycle).  A   #
#  late lookup in this module is fine; both engines' renderers import  #
#  validation.py before they call validate(), so by then the registry  #
#  is populated.                                                        #
# --------------------------------------------------------------------- #


def _register_default_engines() -> None:
    """Late binding to avoid an import cycle: SiestaConfig /
    PySCFConfig live in modules that themselves import from
    validation.  Importing them here at module-import time would loop;
    importing inside a function called from validate() is safe because
    by then both modules are fully loaded."""
    try:
        from .siesta import SiestaConfig
        _ENGINE_VALIDATORS[SiestaConfig] = _validate_siesta
    except ImportError:
        pass
    try:
        from .pyscf import PySCFConfig
        _ENGINE_VALIDATORS[PySCFConfig] = _validate_pyscf
    except ImportError:
        pass


_register_default_engines()


__all__ = ["validate", "report"]
