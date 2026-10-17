"""A VALUE A PERSON SETS ON A FORM, AND WHERE ITS EFFECT IS.

Two tests ask every field a form offers whether setting it changes what the
engine reads: the general form's (``tests/test_every_form_field_reaches_the_deck.py``)
and each transport rung's tab (``tests/test_transport_on_a_real_junction_e2e.py``).
A row of either needs the same two facts, kept here once: a different, legal
value for the item, and the fields whose effect is real but is not in the deck,
each with its reason.
"""
from __future__ import annotations

#: Fields whose effect is real but is NOT in what the engine reads, with the
#: reason.  Anything added here needs one -- "it does nothing" is not a
#: reason, it is the defect these tests look for.
NOT_IN_THE_DECK = {
    ("siesta", "psml_lib"):
        "names WHERE pseudopotentials are found; a deck never carries a host path",
    ("siesta", "verbose_comments"):
        "it is about the deck's COMMENTS -- what a person reads beside each "
        "keyword -- and these tests compare what SIESTA reads",
    ("pyscf", "verbose_comments"):
        "it is about the deck's COMMENTS -- what a person reads beside each "
        "setting -- and these tests compare what PySCF runs",
    ("siesta", "write_molwatch_log"):
        "SIESTA honours it at the PROMISES sub-step (3.12), not in the deck: "
        "`prep._seed_trajectory_log` skips seeding "
        "`<label>.molwatch.log` when it is off.  Exempt for SIESTA ONLY -- "
        "PySCF's script builds the emitter, so its deck does change, and the "
        "general form's test checks that it does",
}


def another_value(item, current, choices=None):
    """A different, legal value for the catalogue ``item`` than ``current``
    -- the next of ``choices`` (what the calculation's kind offers,
    `template.offered`; the item's own by default), a count one more, a
    number half as large again, a switch flipped; a k-point grid one more
    along x and y and its offset moved half a step there, z kept, which a
    transport rung fixes -- or ``None`` if none can be built."""
    choices = choices if choices is not None else getattr(item, "choices",
                                                          None)
    if choices:
        return next((c for c in choices if c != current), None)
    rng = getattr(item, "range", None)
    if item.type == "bool":
        return not bool(current)
    if item.type == "int":
        v = (current or 0) + 1
        return rng[0] if rng and not (rng[0] <= v <= rng[1]) else v
    if item.type == "float":
        v = current * 1.5 if isinstance(current, (int, float)) and current else 1.0
        return (rng[0] + rng[1]) / 2 if rng and not (rng[0] <= v <= rng[1]) else v
    if item.type == "int3":
        x, y, z = tuple(current) if current else (1, 1, 1)
        top = rng[1] if rng else None
        return (x + 1 if top is None or x < top else x - 1,
                y + 1 if top is None or y < top else y - 1, z)
    if item.type == "float3":
        x, y, z = tuple(current) if current else (0.0, 0.0, 0.0)
        return (0.5 if x == 0 else 0.0, 0.5 if y == 0 else 0.0, z)
    if item.type == "str":
        return "mb-probe"
    return None
