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
