"""The relaxation's outcome and its non-convergence policy — one function,
spliced into both PySCF decks (`docs/engines/pyscf.md` § 3).

Two PySCF decks relax a geometry — the optimization deck, and the vibration
deck for which relaxation is the measurement's precondition — and both honour
``on_nonconvergence``.  **Both relax through :func:`relax`**, whose source is
spliced verbatim into each deck (:func:`emit_relax`, the way the HOMO rule is),
so one implementation runs and the tests exercise the one that runs.

**What it asks, and why it has to ask.**  geomeTRIC raises
``GeomOptNotConvergedError`` when its step cap is reached with its criteria
unmet; PySCF's driver catches it, and ``geometric_solver.kernel`` returns the
flag with the geometry.  ``geometric_solver.optimize`` returns ``kernel(...)[1]``
— the geometry alone — so a deck that calls ``optimize`` cannot know.  Until
2026-09-29 both decks called ``optimize`` and wired the policy to
``assert_convergence``, which guards only each step's SCF (PySCF 2.14
``geomopt/geometric_solver.py``): a rung that ran out of steps was recorded
converged and handed on under every policy (the M11 review, plan § 5w K6).

**Why a module of its own.**  `scf_setup.py` is the precedent: emitted PySCF
code that both decks compose, living beside them rather than inside either —
the SCF dresser there, the relaxation's policy here.  Value-free: everything
it needs arrives as an argument, so it never learns which deck called it.
"""
from __future__ import annotations

import inspect
from typing import List, Tuple


def policy_of(cfg) -> Tuple[str, int]:
    """``(on_nonconvergence, geom_continue_retries)`` read ONCE, the same way
    for both decks: the policy lower-cased with ``halt`` for a blank, the
    retry budget an integer with 0 for a blank.  Each deck spelled its own
    reading until 2026-09-29 (the K6 review, R13)."""
    policy = str(getattr(cfg, "on_nonconvergence", "") or "halt").strip().lower()
    retries = int(getattr(cfg, "geom_continue_retries", 0) or 0)
    return policy, retries


def relax(mf, policy, retries, **geometric_kw):
    """Relax ``mf``'s molecule with geomeTRIC and apply this rung's
    ``on_nonconvergence`` to what geomeTRIC reports — ``(mol, converged)``,
    ``converged`` geomeTRIC's own verdict on all of its criteria.

    ``policy``  ``halt`` · ``continue`` · ``proceed`` (`engines/pyscf.md` § 3):
                ``halt`` stops the run before the caller can write the relaxed
                geometry; ``continue`` re-enters from the geometry reached, up
                to ``retries`` more batches, then stops as ``halt`` does;
                ``proceed`` returns the geometry reached with ``False``.
    ``retries`` further batches of the step budget, under ``continue`` only.
    ``geometric_kw`` what geomeTRIC is handed — ``maxsteps`` (required: the
                step budget the messages name), the five ``convergence_*``
                criteria, ``constraints``, ``prefix``, ``callback`` — the same
                at every batch.

    **The stop is a ``RuntimeError``**, as PySCF's own failures are, not a
    ``SystemExit``: Python hands a ``SystemExit`` to no ``excepthook``, so the
    deck's live log would have closed as a clean end (the K6 review, R1).

    Every step's SCF must converge under every policy
    (``assert_convergence=True``): a gradient from an unconverged SCF is not a
    force, so PySCF's own error for one is let through, never retried.  A
    re-entry starts geomeTRIC's step history afresh from the geometry reached
    and evaluates that geometry again, so the live log, fed by ``callback``,
    shows it twice and counts it; the trajectory under ``prefix`` holds the
    last batch's steps.
    """
    from pyscf.geomopt.geometric_solver import kernel
    steps = int(geometric_kw["maxsteps"])
    batches = 1 + (int(retries) if policy == "continue" else 0)
    for batch in range(1, batches + 1):
        converged, mol = kernel(mf, assert_convergence=True, **geometric_kw)
        if converged:
            return mol, True
        if batch < batches:
            print(f"WARNING: the relaxation did not meet geomeTRIC's criteria "
                  f"in {steps} steps; continuing from the geometry it "
                  f"reached ({batches - batch} more batch(es) of {steps})")
            mf = mf.reset(mol)
    if policy == "proceed":
        print(f"WARNING: the relaxation did not meet geomeTRIC's criteria in "
              f"{steps} steps; on_nonconvergence = proceed keeps the "
              f"geometry it reached")
        return mol, False
    raise RuntimeError(
        f"the relaxation did not meet geomeTRIC's criteria in "
        f"{steps * batches} steps (on_nonconvergence = {policy}), so the "
        f"relaxed geometry is not written and nothing can start from it.  "
        + ("Raise geom_max_steps or geom_continue_retries"
           if policy == "continue" else
           "Raise geom_max_steps or choose on_nonconvergence = continue")
        + ", and prep this stage again.")


def emit_relax() -> List[str]:
    """:func:`relax`, as deck lines — its source, spliced rather than retyped."""
    return [ln.rstrip() for ln in inspect.getsource(relax).splitlines()]


__all__ = ["policy_of", "relax", "emit_relax"]
