"""The vibration deck RUNS the spectra science gate -- and it refuses.

Between P1 and P3 the deck's ``validate(struct, view)`` call silently
skipped the spectra science (grid / amplitude / parity / method /
open-shell): the engine dispatch inside ``validate()`` is keyed on
``type(cfg)`` and the deck's config view is an ADAPTER over
PySCFConfig, so no registered validator matched -- a green E2E hid a
gate that never ran.  Found 2026-08-21 while retiring the engine
registry (spectra-migration plan P3).  The science moved whole to
``validation/spectra.py`` and the deck composes it BY NAME; these pins
hold the composition down at its two observable ends.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import spec_for
from molbuilder.script_emit import render_deck
from molbuilder.structure import Structure


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.119],
                            [0.0, 0.757, -0.477],
                            [0.0, -0.757, -0.477]]))


def _render(cfg: PySCFConfig) -> str:
    s = _water()
    return render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)


def test_parity_error_refuses_the_deck():
    """spin=1 on water's 10 electrons is impossible; the gate's
    parity check must refuse at RENDER, not at PySCF runtime."""
    with pytest.raises(Exception) as exc:
        _render(PySCFConfig(spin=1, method="UKS"))
    msg = str(exc.value).lower()
    assert "spin" in msg and "electron" in msg


def test_amplitude_advisory_reaches_the_person():
    """A 0.5 A displacement is outside the accepted window; the deck
    still renders (a warn is advice, not a refusal) and the warning
    goes where a person sees it."""
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        text = _render(PySCFConfig(displacement_amplitude_ang=0.5))
    finally:
        sys.stderr = old
    assert "Hessian" in text, "the deck did not render"
    assert "0.02-0.20" in err.getvalue(), (
        "the amplitude advisory never surfaced -- the science gate "
        "is not composed into the deck's render")


def test_registry_path_serves_a_real_spectra_config():
    """The registered validator (for callers holding a real
    SpectraConfig) runs the SAME body -- one gate, two doors."""
    from molbuilder.config.spectra import SpectraConfig
    from molbuilder.validation import validate
    issues = validate(_water(), SpectraConfig(es_mode_selection="top_n"))
    assert any("Raman-weak" in i.message for i in issues)
