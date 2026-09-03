"""PySCF's engine-parameter adapter for the chemistry analyzer.

Translates a ``ChemistryAnalysis`` (engine-agnostic conclusions)
into a typed ``PyscfSuggestedParams`` dataclass whose fields
match the PySCF web-form / ``PySCFConfig`` field names.

See ``docs/science/validation.md`` § 4 for the
adapter contract.  The adapter is registered at import time via
``@register_adapter``; the canonical import site is
``molbuilder/web/blueprints/__init__.py`` so the registry is
populated before any HTTP request hits ``/api/structure/analyze``.

The Spectra tab also emits PySCF scripts; it reuses this adapter
through the registry — no separate ``SpectraAdapter`` needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from molbuilder.chemistry import (
    ChemistryAnalysis,
    register_adapter,
)


@dataclass(frozen=True)
class PyscfSuggestedParams:
    """Engine-specific suggested defaults for PySCF.

    Field names match the PySCF web-form fields (and the
    ``PySCFConfig`` dataclass) so the UI's "apply suggestion" path
    is a 1:1 spread into form values.

    ``spin`` is 2S (= number of unpaired electrons), matching
    PySCF's ``gto.M(spin=...)`` convention (NOT multiplicity 2S+1).
    ``method`` is the SCF method string PySCF accepts —
    ``"UKS"`` for open-shell DFT, ``"RKS"`` for closed-shell DFT.
    """
    net_charge: int
    spin:      int
    method:    str       # "UKS" | "RKS"
    rationale: str


@register_adapter("pyscf")
class PyscfAdapter:
    """PySCF adapter — single translation point from
    ChemistryAnalysis to PyscfSuggestedParams.

    Open-shell treatment → UKS; closed-shell → RKS.  Doesn't pick
    between DFT and HF (the user picks the functional); the
    analyzer's job is the (charge, spin, treatment) triplet only.
    See ``science/validation.md`` § 8 for what is intentionally out
    of scope.
    """
    name = "pyscf"

    @classmethod
    def to_params(cls, analysis: ChemistryAnalysis) -> PyscfSuggestedParams:
        method = "UKS" if analysis.suggested_treatment == "open" else "RKS"
        return PyscfSuggestedParams(
            net_charge = analysis.suggested_charge,
            spin      = analysis.suggested_spin,
            method    = method,
            rationale = analysis.rationale,
        )
