"""The refusals every floor of the job system raises -- floor 1
(`execution/architecture.md` § 2.1).

A plain type, so a lower floor can refuse in the person's own words without
importing the conductor that collects the refusal: ``PrepError`` lived in
`jobset/prep.py` until 2026-10-03, and each module that refused a prep --
the assembly, the stage namer, the engine seam -- imported the conductor to
reach it (W55 B7).
"""
from __future__ import annotations

from typing import Optional


class PrepError(Exception):
    """A prep refused -- in the reader's own words, which each surface shows
    as they are.

    What the one prep entry had already found when it refused rides with it
    (`prep.prep_stage`): ``findings`` (the description's preflight notes),
    ``notes`` (what its inputs said -- a bench's grid with every crossed-out
    cell, a run's sizing) and ``partial`` (the answer so far, once the five
    steps have written something).  A refusal that says *see the crossed-out
    list above* is only honest if the list is shown with it; the command line
    prints these before the error, the Task setup route returns them beside
    it.  Empty on a refusal raised anywhere else.
    """
    findings: tuple = ()
    notes: tuple = ()
    partial: "Optional[object]" = None


__all__ = ["PrepError"]
