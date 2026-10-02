"""The electronic state's items, declared once for every engine's config.

``science/chemistry-correctness.md`` § 2a: a calculation's charge and spin are
four template items -- ``net_charge``, ``spin_treatment``,
``unpaired_electrons`` and ``method`` -- and a blank one means *work it out*,
which ``electronic_state.electronic_state`` does once for the form, the checks
and the deck alike.  Nothing reads these fields raw.

The first three are **merged** items: one question, asked of both engines,
spelled alike in both configs (``engines/template.md`` § 6.3).  A merge is
declared by the field name, and the two declarations must agree on kind, type,
default, category, allocation, optional and unit -- so they are written ONCE,
here, and each config takes a fresh field from these factories (a dataclass
``Field`` is named and typed by the class it lands in, so one object cannot be
shared).  ``net_charge`` had two copies until 2026-09-28, one carrying a
``tier`` the other did not.  ``method`` is PySCF's alone -- SIESTA is a
density-functional code -- and stays in ``config/pyscf.py``.
"""
from __future__ import annotations

from dataclasses import field

from ..electronic_state import COUNTS, TREATMENTS


def net_charge():
    """How many electrons short (+) or extra (−), in |e|.  Blank runs the
    phosphate rule (``model/chemistry.md`` § 1)."""
    return field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label": "Net charge",
        # The engine_key names both spellings because the item belongs to
        # both engines and neither spelling is THE answer (`template.md`
        # § 6.3).
        "engine_key":  "NetCharge (SIESTA) | gto.M(charge=...) (PySCF)",
        "item_kind": "deck",
        "expands": ("NetCharge", "gto.M"),
        "null_label": "(auto)",
        "range": (-10, 10),
        "tier": "basic",
    })


def spin_treatment():
    """How the two spin channels are solved -- the engine-neutral words
    (``electronic_state.TREATMENTS``).  SIESTA spells them ``Spin
    non-polarized`` / ``polarized`` / ``non-colinear`` / ``spin-orbit``
    (``siesta/layout.py``); PySCF composes its SCF class from them
    (``pyscf/layout.scf_class``).  A form offers only what the engine can run for the
    kind (ES4)."""
    return field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label": "Spin treatment",
        "choices": TREATMENTS,
        "engine_key": "Spin (SIESTA) | the SCF class's R / RO / U (PySCF)",
        "item_kind": "deck",
        "expands": ("Spin",),
        "null_label": "(auto)",
    })


def unpaired_electrons():
    """2S, the number of unpaired electrons -- NOT the multiplicity 2S+1 --
    or ``free``, a moment that floats to whatever the SCF finds (SIESTA
    only, ES6)."""
    return field(default=None, metadata={
        "category": ("system",),
        "workflow_group": "profile",
        "label": "Unpaired electrons (2S)",
        "choices": COUNTS,
        "engine_key": ("Spin.Fix + Spin.Total (SIESTA) | "
                       "gto.M(spin=...) (PySCF)"),
        "item_kind": "deck",
        "expands": ("Spin.Fix", "Spin.Total", "gto.M"),
        "null_label": "(auto)",
    })
