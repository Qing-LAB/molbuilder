"""Every parameter the vibration form shows is HONORED by the deck.

The user's ruling (2026-08-21, vibration-parameter-integration-plan): a
form that shows a knob the calculation ignores is a lie -- the gap that
plan exists to close.  This gate measures honesty by RENDERING: for each
parameter the vibration form offers, setting a non-default value must
change the emitted deck text.  Parameters still awaiting their
integration phase sit on the explicit OPEN list below, each naming its
plan row -- the list only shrinks, and a NEW parameter leaking into the
form unhonored fails here on arrival.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import template as T
from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import spec_for
from molbuilder.script_emit import render_deck
from molbuilder.structure import Structure

#: Still awaiting their phase of the integration plan.  Each row names
#: the category that owns it; deleting a row is the proof its knob
#: landed.  (Category 2 = the physical model, pending the PySCF
#: support-matrix investigation; category 3 = workflow knobs.)
STILL_OPEN = {
    "solvent":               "category 2 -- solvation support matrix",
    "solvent_method":        "category 2 -- solvation support matrix",
    "symmetry":              "category 2 -- mode irrep labeling",
    "optimize":              "category 3 -- excluded by ruling (already_relaxed is the one skip)",
    "optimizer":             "category 3 -- relax block hardcodes geomeTRIC today",
    "geom_etol":             "category 3 -- relax convergence dict",
    "geom_continue_retries": "category 3 -- relax retry wrapper",
    "save_initial_xyz":      "category 3 -- standalone geometry files",
    "save_optimized_xyz":    "category 3 -- standalone geometry files",
    "write_molwatch_log":    "category 3 -- live-watch for the relax phase",
    "write_trajectory":      "category 3 -- relax trajectory file",
    "log_file":              "category 3 -- log naming",
}

_PROBES = {
    "bool":  lambda d: (not d),
    "int":   lambda d: (int(d or 0) + 3),
    "float": lambda d: (float(d or 0.0) * 2 + 0.031),
    "str":   lambda d: "probe-value",
    # list-typed items (e.g. ecp_atoms): a one-element list of a heavy
    # element -- the class of value the knob exists for.
    "strlist": lambda d: ["Au"],
    "intlist": lambda d: [7],
}


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.119],
                            [0.0, 0.757, -0.477],
                            [0.0, -0.757, -0.477]]))


def _render(**over) -> str:
    # density_fit ON in the baseline so auxbasis's ride is probeable.
    cfg = PySCFConfig(density_fit=True, **over)
    s = _water()
    return render_deck(spec_for(s, cfg, calculation="vibration"),
                       s, cfg, verbose=False)


def _vibration_items():
    items = [i for i in T.select(T.read_template(T.load_catalogue()),
                                 engine="pyscf")
             if i.group != "staging"
             and (not i.calculations or "vibration" in i.calculations)]
    return items


def _probe_value(item):
    if item.choices:
        others = [c for c in item.choices if c != item.default]
        return others[0] if others else None
    fn = _PROBES.get(item.type)
    if fn is None:
        return None
    v = fn(item.default)
    if item.range:
        lo, hi = item.range
        try:
            v = min(max(v, lo), hi)
            if v == item.default:
                v = hi if item.default != hi else lo
        except TypeError:
            pass
    return v


def test_every_shown_parameter_changes_the_deck_or_is_openly_pending():
    baseline = _render()
    silent = []
    skipped = []
    for item in _vibration_items():
        if item.name in STILL_OPEN:
            continue
        probe = _probe_value(item)
        if probe is None or probe == item.default:
            skipped.append(item.name)
            continue
        try:
            text = _render(**{item.name: probe})
        except Exception:
            # A REFUSAL is an honored parameter: the deck reacted.
            continue
        if text == baseline:
            silent.append(item.name)
    assert not skipped, (
        f"probe generator could not produce a distinct value for: "
        f"{skipped} -- extend _PROBES rather than skipping silently")
    assert not silent, (
        f"the vibration form shows these parameters and the deck "
        f"IGNORES them -- the honesty gap this test exists to close: "
        f"{silent}.  Honor each (pyscf.md § 7a is the door for SCF "
        f"machinery) or add it to STILL_OPEN with its plan row.")


def test_the_open_list_matches_the_plan_not_more():
    """The open list may only name parameters the plan actually tracks
    -- a name parked here without a plan row is hiding, not pending."""
    plan = open("docs/plans/vibration-parameter-integration-plan.md").read()
    unplanned = [n for n in STILL_OPEN if f"`{n}`" not in plan]
    assert not unplanned, (
        f"STILL_OPEN entries absent from the integration plan: {unplanned}")
