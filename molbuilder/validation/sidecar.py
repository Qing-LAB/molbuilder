"""Sidecar-aware validators (frozen-atoms / region labels).

Per docs/model/structure-molstruct.md the user attaches metadata to
a structure via a ``.molstruct.json`` sidecar: regions, frozen-atom
indices, generator-input echo.  Engines that don't consume one of
these labels MUST surface an INFO Issue so the user can see the
absorption was noticed but not silently dropped.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/science/validation.md  Function body +
signature are identical to the pre-split version.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def _check_frozen_atoms_consumed(struct: Structure, *,
                                   engine: str,
                                   honored: bool,
                                   reason_when_dropped: str = "",
                                   ) -> List[Issue]:
    """Three-stage contract: warn when ``Structure.frozen_atoms`` is
    populated (the user / sidecar asked SOMETHING be held fixed) but
    the engine emission path is going to silently drop the constraint.

    Two callsites:
      * SIESTA: ``honored`` is False when ``cfg.relax_type == 'none'``
        (no MD block emitted, so Geometry.Constraints does nothing).
      * PySCF: ``honored`` is False when ``cfg.optimize == False``
        (single-point energy, no relaxation) OR
        ``cfg.optimizer != 'geometric'`` (the only PySCF optimizer
        with constraint support).

    When honored is True we emit an INFO-severity Issue so the user
    sees an explicit "N atoms held fixed during relaxation" line in
    the preflight panel, not just a silent emission.
    """
    n = len(getattr(struct, "frozen_atoms", []) or [])
    if n == 0:
        return []
    if not honored:
        return [Issue(
            "warn",
            (f"Structure has {n} frozen atom(s) from /modify "
             f"(struct.frozen_atoms), but {engine} won't honor them: "
             f"{reason_when_dropped}.  Either change the config to a "
             f"mode that supports constraints, or clear the frozen "
             f"atoms in /modify if you want a free relaxation."),
            "config.frozen_atoms",
        )]
    return [Issue(
        "info",
        (f"{n} atom(s) held fixed during {engine} relaxation "
         f"(from struct.frozen_atoms / /modify sidecar)."),
        "config.frozen_atoms",
    )]


def check_unconsumed_region_labels(struct: Structure, *,
                                   engine: str) -> List[Issue]:
    """Pattern B, re-homed (validation.md § 5; C-shared 2026-08-21): every
    region label the current engine does NOT consume is named explicitly.

    The /modify selection panel writes ``regions`` for transport workflows
    (L-electrode, bridge, ...); an optimization deck reads none of them,
    and silence would let a user believe their labels shaped the run.  The
    reserved frozen label is EXCLUDED: both engines consume it (SIESTA's
    ``Geometry.Constraints``, PySCF's geomeTRIC ``$freeze``), so warning
    about it would be the same false alarm E-M7.1 fixed on the vibration
    route.  This ran in two web endpoints until they were deleted; living
    HERE puts it on every deck route through the one settings gate.
    """
    from ..structure import FROZEN_LABEL
    regions = getattr(struct, "regions", None) or {}
    inert = sorted(name for name, idxs in regions.items()
                   if idxs and name != FROZEN_LABEL)
    if not inert:
        return []
    return [Issue(
        "warn",
        (f"this structure carries region label(s) {inert}, which the "
         f"{engine} run does NOT consume -- they stay in the sidecar "
         f"for /transport but do not shape this calculation. "
         f"If you meant those atoms to be held fixed, assign them to "
         f"\"frozen_atoms\" in /modify."),
        "structure.regions",
    )]
