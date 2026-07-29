"""SIESTA's engine-parameter adapter for the chemistry analyzer.

Translates a ``ChemistryAnalysis`` (engine-agnostic conclusions)
into a typed ``SiestaSuggestedParams`` dataclass whose fields
match the SIESTA web-form / ``SiestaConfig`` field names.

See ``docs/science/validation.md`` § 4 for the
adapter contract.  The adapter is registered at import time via
``@register_adapter``; the canonical import site is
``molbuilder/web/blueprints/__init__.py`` so the registry is
populated before any HTTP request hits ``/api/structure/analyze``.
"""
from __future__ import annotations

from dataclasses import dataclass

from molbuilder.chemistry import (
    ChemistryAnalysis,
    EngineParameterAdapter,
    register_adapter,
)


@dataclass(frozen=True)
class SiestaSuggestedParams:
    """Engine-specific suggested defaults for SIESTA.

    Field names match the SIESTA web-form fields (and the
    ``SiestaConfig`` dataclass) so the UI's "apply suggestion" path
    is a 1:1 spread into form values.

    ``spin_total`` is in μB (Bohr magnetons), matching SIESTA's
    ``Spin.Total`` keyword.  The analyzer carries 2S as an int; we
    cast to float here so the wire shape matches what SIESTA's deck
    will eventually receive.
    """
    net_charge:     int
    spin_polarized: bool
    spin_total:     float
    rationale:      str


@register_adapter("siesta")
class SiestaAdapter:
    """SIESTA adapter — single translation point from
    ChemistryAnalysis to SiestaSuggestedParams.

    No chemistry logic here; only translation.  If you find
    yourself reaching for ``detect_open_shell_metals`` or
    ``check_spin_charge_parity`` from this file, the work belongs
    in ``analyze_structure`` instead.
    """
    name = "siesta"

    @classmethod
    def to_params(cls, analysis: ChemistryAnalysis) -> SiestaSuggestedParams:
        return SiestaSuggestedParams(
            net_charge     = analysis.suggested_charge,
            spin_polarized = analysis.suggested_treatment == "open",
            spin_total     = float(analysis.suggested_spin),
            rationale      = analysis.rationale,
        )
