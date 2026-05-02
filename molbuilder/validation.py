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
from typing import List, Optional, Tuple

import numpy as np

from .issues import Issue, ValidationError
from .structure import Structure


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
    issues += _validate_geometry(struct, cell)
    issues += _validate_config_metadata(cfg)

    cls_name = type(cfg).__name__
    if cls_name == "SiestaConfig":
        issues += _validate_siesta(struct, cfg, cell)
    elif cls_name == "PySCFConfig":
        issues += _validate_pyscf(struct, cfg)
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


def _validate_geometry(struct: Structure,
                       cell: Optional[np.ndarray]) -> List[Issue]:
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

    return issues


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
    cls_name = type(cfg).__name__
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


def _validate_siesta(struct: Structure, cfg,
                     cell: Optional[np.ndarray]) -> List[Issue]:
    issues: List[Issue] = []

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

    # k-grid vs cell extent: vacuum direction with k != 1 is wasted;
    # periodic direction (extent > 10 Å) with k == 1 is under-converged.
    diag_lengths = [float(np.linalg.norm(cell[i])) for i in range(3)]
    for axis, (k, length) in enumerate(zip(cfg.kgrid, diag_lengths)):
        if k != 1 and length < 10.0:
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = {k} along an axis of {length:.1f} Å "
                f"(< 10 Å, looks like vacuum); k-points along a vacuum "
                f"direction add cost without improving accuracy",
                f"config.kgrid",
            ))
        elif k == 1 and length > 10.0 and any(kk > 1 for kk in cfg.kgrid):
            # The any-other-axis-has-k>1 guard limits this warning to
            # mixed-PBC systems (slabs / wires) where one axis is
            # genuinely periodic and the user might have forgotten to
            # set k>1 on another periodic axis.  A pure vacuum (all
            # k=1) shouldn't trigger.
            issues.append(Issue(
                "warn",
                f"kgrid[{axis}] = 1 along an axis of {length:.1f} Å "
                f"(> 10 Å, looks periodic) while another axis uses "
                f"k > 1; likely under-converged sampling on this axis",
                f"config.kgrid",
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


def _validate_pyscf(struct: Structure, cfg) -> List[Issue]:
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

    return issues


__all__ = ["validate", "report"]
