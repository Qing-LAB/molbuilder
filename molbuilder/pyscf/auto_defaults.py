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

from typing import Literal
from dataclasses import dataclass

from molbuilder.chemistry import (
    ChemistryAnalysis,
    register_adapter,
)


#: The SCF method strings PySCF accepts from us: UKS for open-shell DFT,
#: RKS for closed-shell.  One home -- `PyscfSuggestedParams.method` and
#: anything that validates it read this tuple.
SCF_METHODS: "tuple[str, ...]" = ("UKS", "RKS")
SCFMethod = Literal["UKS", "RKS"]


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
    method:    SCFMethod
    rationale: str

    def __post_init__(self) -> None:
        # The Literal is for the reader; THIS is the enforcement.  Nothing in
        # this repo type-checks (no mypy / pyright), so an annotation alone
        # refuses nothing at runtime -- the same pairing `issues.Issue` uses.
        # Until 2026-09-09 the legal values lived in a COMMENT beside
        # `method: str`, and a route test asserted `py["method"] in {"UKS",
        # "RKS"}` to make up for it.
        if self.method not in SCF_METHODS:
            raise ValueError(
                f"PyscfSuggestedParams.method must be one of {SCF_METHODS}; "
                f"got {self.method!r}")
        # `spin: int` is an annotation and annotations are not checked; a str
        # constructed happily until 2026-09-09, covered by a route test
        # asserting `isinstance(py["spin"], int)`.  PySCF's `gto.M(spin=)` is
        # 2S -- a count of unpaired electrons -- so a non-integer is not a
        # rounding question, it is wrong.
        if isinstance(self.spin, bool) or not isinstance(self.spin, int):
            raise TypeError(
                f"PyscfSuggestedParams.spin is 2S, a whole number of unpaired "
                f"electrons; got {self.spin!r}")


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
