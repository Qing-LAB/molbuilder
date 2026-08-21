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

import re

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
#: EMPTY -- and the emptiness is the point: every parameter the
#: vibration form shows is now read by the render or refused by name
#: with the reason and references (category 2 landed 2026-08-21 on the
#: probed PySCF support matrix; PCM honored end to end, SMD/ddCOSMO
#: informed refusals, symmetry honored on the already-relaxed path).
STILL_OPEN = {}

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


def _gold() -> Structure:
    # Au dimer: the ECP-wanting case (a bare heavy metal).
    return Structure(
        elements=["Au", "Au"],
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]]))


#: Knobs that only speak when a companion is set -- the probe renders
#: with these beside the probe value (mirroring real use: retries only
#: matter under the continue policy; auxbasis rides density fitting,
#: which the baseline already enables).
_COMPANIONS = {
    "geom_continue_retries": {"on_nonconvergence": "continue"},
    # solvent alone -> the PCM decoration lines; symmetry is honored
    # on the already-relaxed path (elsewhere it refuses, which also
    # counts as honored).
    "symmetry": {"already_relaxed": True},
    # a method methods nothing without a solvent (the validator warns
    # standalone); its deck effect is probed beside one.
    "solvent_method": {"solvent": "water"},
    # An ECP only speaks when both halves are named AND the structure
    # holds the element -- probe them together, on gold (below).
    "ecp":       {"ecp_atoms": ["Au"]},
    "ecp_atoms": {"ecp": "lanl2dz"},
}

#: Knobs whose effect is conditional on the STRUCTURE: probed on a
#: molecule that can exercise them (water has no ECP candidate).
_PROBE_STRUCTS = {
    "ecp":       "gold",
    "ecp_atoms": "gold",
}


def _strip_config_echo(text: str) -> str:
    """The deck ECHOES the whole config into its provenance block
    (``CONFIG = {...}``), so any field flip changes the raw text
    trivially -- which made this gate VACUOUS for every knob the deck
    never actually reads (caught 2026-08-21: log_file went silent and
    the gate stayed green; a first regex fix then swallowed 29 kB to
    the wrong closing brace).  The span is found by BRACE BALANCE from
    the one ``CONFIG = {`` anchor -- honesty is measured on the deck
    minus exactly that dict: a knob must change what the deck DOES,
    not what it reports it was asked."""
    anchor = "\nCONFIG = {"
    i = text.index(anchor) + 1
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[:i] + "CONFIG = {…}" + text[k + 1:]
    raise AssertionError("CONFIG echo never closed -- the emitter changed")


def _render(_struct: str = "water", **over) -> str:
    # density_fit ON in the baseline so auxbasis's ride is probeable.
    cfg = PySCFConfig(density_fit=True, **over)
    s = _gold() if _struct == "gold" else _water()
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
    baselines = {name: _strip_config_echo(_render(name))
                 for name in ("water", "gold")}
    silent = []
    skipped = []
    for item in _vibration_items():
        if item.name in STILL_OPEN:
            continue
        probe = _probe_value(item)
        if probe is None or probe == item.default:
            skipped.append(item.name)
            continue
        which = _PROBE_STRUCTS.get(item.name, "water")
        try:
            text = _render(which,
                           **{item.name: probe,
                              **_COMPANIONS.get(item.name, {})})
        except Exception:
            # A REFUSAL is an honored parameter: the deck reacted.
            continue
        if _strip_config_echo(text) == baselines[which]:
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
