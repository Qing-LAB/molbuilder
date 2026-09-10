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
    spin_treatment: str          # one of `config.siesta.SPIN_TREATMENTS`
    spin_total:     float
    rationale:      str

    def __post_init__(self) -> None:
        # THE enforcement.  Nothing in this repo type-checks (no mypy /
        # pyright), so an annotation refuses nothing at runtime -- the pairing
        # `issues.Issue` uses is a declared vocabulary plus a constructor that
        # rejects.  The vocabulary is NOT re-spelled here: it is imported from
        # the module that declares the form field, so there is one home.
        from ..config.siesta import SPIN_TREATMENTS
        if self.spin_treatment not in SPIN_TREATMENTS:
            raise ValueError(
                f"SiestaSuggestedParams.spin_treatment must be one of "
                f"{SPIN_TREATMENTS}; got {self.spin_treatment!r}")
        # `spin_total: float` is an ANNOTATION, and annotations are not checked
        # -- `SiestaSuggestedParams(0, "polarized", "abc", "why")` constructed
        # happily until 2026-09-09, with a route test asserting
        # `isinstance(si["spin_total"], float)` to cover for it.  Coerce here,
        # so the wire value IS a float and a non-numeric raises at the point
        # the wrong value was written.
        if not isinstance(self.spin_total, float):
            try:
                object.__setattr__(self, "spin_total", float(self.spin_total))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"SiestaSuggestedParams.spin_total must be a number "
                    f"(mu_B); got {self.spin_total!r}") from exc


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
            spin_treatment = ("polarized"
                              if analysis.suggested_treatment == "open"
                              else "non-polarized"),
            spin_total     = float(analysis.suggested_spin),
            rationale      = analysis.rationale,
        )
