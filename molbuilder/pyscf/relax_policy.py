"""The relaxation's non-convergence policy — one home for the retry loop.

**The contract is `docs/engines/pyscf.md` § 7a's sibling rule:** the framework
never spells one behaviour twice.  Two PySCF decks relax a geometry — the
optimization deck, and the vibration deck for which relaxation is the
measurement's precondition — and both honour ``on_nonconvergence``.  Its
``continue`` arm is a retry loop, and until 2026-08-23 each deck spelled that
loop out: same budget, same `try`, same *"is this a convergence failure or a
real error"* test, same exhausted-raise, same countdown in the warning.  A fix
to one would not have reached the other.

**Why here and not in `layout.py`.**  That module is door 3a — one catalogue
item, one line — and its own docstring rules this out: *"behaviour flags with
no keyword at all … generate control flow rather than a setting, and belong to
the free-form body (door 3b)."*  A retry loop is control flow.

**Why a module of its own.**  `scf_setup.py` is the precedent and the shape is
identical: emitted PySCF code that both decks compose, living beside them
rather than inside either.  This is that, one concern over — the SCF dresser
there, the relaxation's policy here.

Stdlib-only and value-free: everything it needs arrives as an argument, so it
never learns which deck called it.
"""
from __future__ import annotations

from typing import List, Sequence


def emit_retry_loop(call: Sequence[str], *, retries: int, steps_var: str,
                    what: str = "", indent: str = "") -> List[str]:
    """The ``on_nonconvergence='continue'`` arm, as deck lines.

    ``call``      the statement to run inside the ``try``, as lines.  Line 0
                  carries no indent; a continuation line carries only its
                  alignment *relative to the statement's own start*, so the
                  caller states the shape of its call and this states the
                  shape of the loop.
    ``retries``   extra attempts beyond the first; the budget is ``1 + n``.
    ``steps_var`` the deck's own name for the step budget, quoted into the
                  message so it reports the number the run actually used.
    ``what``      an infix naming what did not converge (``"relaxation "``),
                  or empty.
    ``indent``    the base indent, for a caller emitting inside a block.

    **The two arms this does NOT own are `proceed` and `halt`**, and that is
    deliberate rather than an omission: they are one call each, and the two
    decks do genuinely different things around them — the vibration deck
    records `state['relaxation']` bookkeeping the optimization deck has no
    equivalent of.  Folding those in would mean parameterising a difference
    instead of sharing a sameness.
    """
    i = indent
    out: List[str] = [
        f"{i}_budget = 1 + {int(retries)}",
        f"{i}for _attempt in range(_budget):",
        f"{i}    try:",
    ]
    out += [f"{i}        {line}" for line in call]
    out += [
        f"{i}        break",
        f"{i}    except RuntimeError as _e:",
        f"{i}        if 'not converged' not in str(_e).lower():",
        f"{i}            raise            # a genuinely different error",
        f"{i}        if _attempt == _budget - 1:",
        f"{i}            raise            # exhausted -> halt",
        f'{i}        print(f"WARN: {what}did not converge in "',
        f'{i}              f"{{{steps_var}}} steps; retrying "',
        f'{i}              f"({{_budget - 1 - _attempt}} left)")',
    ]
    return out
