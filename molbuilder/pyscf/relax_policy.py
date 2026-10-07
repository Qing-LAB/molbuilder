"""The relaxation's outcome and its non-convergence policy — one function,
imported by both PySCF decks (`docs/engines/pyscf.md` § 3).

Two PySCF decks relax a geometry — the optimization deck, and the vibration
deck for which relaxation is the measurement's precondition — and both honour
``on_nonconvergence``.  **Both relax through :func:`relax`**, which each
imports from ``mb_pyscf.pyz`` beside it (``runwrap.PYSCF_COMPANIONS``), so one
implementation runs and the tests exercise the one that runs.  This file
travels as itself: the standard library at load, PySCF inside :func:`relax`.

**What it asks, and why it has to ask.**  geomeTRIC raises
``GeomOptNotConvergedError`` when its step cap is reached with its criteria
unmet; PySCF's driver catches it, and ``geometric_solver.kernel`` returns the
flag with the geometry.  ``geometric_solver.optimize`` returns ``kernel(...)[1]``
— the geometry alone — so a deck that calls ``optimize`` cannot know.

**Why a module of its own.**  Code both decks run, living beside them rather
than inside either.  Value-free: everything it needs arrives as an argument,
so it never learns which deck called it.
"""
from __future__ import annotations

from typing import Tuple


def policy_of(cfg) -> Tuple[str, int]:
    """``(on_nonconvergence, geom_continue_retries)`` read ONCE, the same way
    for both decks, as the template states them -- the catalogue gives both
    their values (``halt``, 1).  A blank is refused (`generator.md`
    § 10.1)."""
    policy = getattr(cfg, "on_nonconvergence", None)
    retries = getattr(cfg, "geom_continue_retries", None)
    if policy in (None, "") or retries is None:
        raise ValueError(
            "on_nonconvergence and geom_continue_retries are stated in the "
            "template -- this relaxation's leaves "
            + ("on_nonconvergence" if policy in (None, "")
               else "geom_continue_retries") + " blank.")
    return str(policy).strip().lower(), int(retries)


def relax(mf, policy, retries, *, keep=None, resumable=False,
          **geometric_kw):
    """Relax ``mf``'s molecule with geomeTRIC and apply this rung's
    ``on_nonconvergence`` to what geomeTRIC reports — ``(mol, converged)``,
    ``converged`` geomeTRIC's own verdict on all of its criteria.

    ``policy``  ``halt`` · ``continue`` · ``proceed`` (`engines/pyscf.md` § 3):
                ``halt`` stops the run, an error -- so no later rung builds
                on it: a hand-over takes only a run the status door calls
                finished (`job-system.md` § 5.4); ``continue`` re-enters
                from the geometry reached, up
                to ``retries`` more batches, then stops as ``halt`` does;
                ``proceed`` returns the geometry reached with ``False``.
    ``retries`` further batches of the step budget, under ``continue`` only.
    ``keep``    writes the geometry each step reached -- the molecule
                geomeTRIC's callback is handed after every step's energy
                and gradient (PySCF's ``geometric_solver``: ``callback(
                locals())``, its ``mol`` at the step's coordinates) -- so a
                run stopped at its step limit or its wall leaves where it
                got to, as SIESTA's ``.XV`` does (user, 2026-10-06).
    ``resumable`` whether launching the stage again reads that geometry
                back (its ``restart`` is ``continue``): what the stop's
                message tells the person to do.
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
    if keep is not None:
        given = geometric_kw.get("callback")

        def _each_step(envs):
            if given is not None:
                given(envs)
            reached = envs.get("mol") if isinstance(envs, dict) else None
            if reached is not None:
                keep(reached)

        geometric_kw = {**geometric_kw, "callback": _each_step}
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
    # THE WAY ON, as `status` and `launch` say it: a prepped stage is not
    # prepped again, so a setting changes from the state saved before its
    # prep (`job-system.md` § 5.0).
    change = ("geom_max_steps or geom_continue_retries" if policy == "continue"
              else "geom_max_steps, or on_nonconvergence = continue")
    back = (f"to change {change}, go back to the state saved before this "
            f"stage's prep and prep it anew")
    raise RuntimeError(
        f"the relaxation did not meet geomeTRIC's criteria in "
        f"{steps * batches} steps (on_nonconvergence = {policy}).  "
        + (f"The geometry it reached is kept: launch this stage again to "
           f"continue from it -- or, {back}."
           if keep is not None and resumable else
           f"Nothing continues from it: {back}."))


__all__ = ["policy_of", "relax"]
