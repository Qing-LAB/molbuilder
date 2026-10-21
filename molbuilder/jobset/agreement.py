"""A benchmark trial starts cold -- the one check a launch makes of a deck
(`job-system.md` R-L29; `execution/job-request.md` § 6).

prep bakes the INTENT -- the measurement pin resolves ``restart = clean``
into each trial's deck -- and the submission verifies the artifact it is
about to send: a trial that would warm-start measures a continued run, not
its point.  Nothing else of a deck is read as a launch: prep writes a deck
and its launch together, once (`architecture.md` A3), and launch sends what
prep wrote.  **Callers**: `jobset/submit`'s benchmark doors -- every trial,
sent by name or riding its shelf's walk.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..identity import checkpoint_words


class TrialNotCold(Exception):
    """A benchmark trial's deck would not start cold -- the launch declines
    to send it (`job-system.md` R-L29)."""


def _bench_anew(folder) -> str:
    """A trial's deck redone: a prepared benchmark is not prepared again, so
    the way back is the state saved before its prep (`job-system.md`
    § 5.0), the calculation's folder named (`identity.checkpoint_words`)."""
    return ("a prepared benchmark is not prepared again (job-system.md "
            "§ 5.0): go back to the state saved before its prep -- "
            + checkpoint_words(folder) + " -- and prep it anew.")


def _calculation_of(job_dir):
    """The calculation a job's folder belongs to, by the folder's own marks
    (`calcdirs.root_of`) -- ``None`` when it says none, and a way back then
    names no folder rather than the wrong one."""
    from ..calcdirs import root_of
    return root_of(job_dir)


def check_trial_starts_cold(job_dir, job) -> None:
    """A TRIAL measures from scratch, and SUBMISSION verifies it.

    User-settled 2026-08-21: prep bakes the INTENT (the measurement pin
    resolves ``restart = clean`` into the deck), but the submission is
    what determines the run's actual starting state -- so the door checks
    the artifact it is about to launch.  The clean restart group is
    written out, never omitted (`siesta/input.py`, 2026-08-18: ``clean``
    writes ``.false.``), so a cold deck PROVABLY carries ``UseSave`` keys
    reading false; a deck where any reads true would warm-start -- a
    continued run wearing a trial's label -- and one carrying none at all
    cannot be vouched for (hand-stripped, or rendered before the pin).
    A warm or group-stripped deck refuses by name (remedy: the benchmark
    prepared anew, from the state saved before its prep); an ABSENT deck
    passes silently -- absence says nothing,
    and the launch itself fails loudly without one.

    Raises :class:`TrialNotCold`: submission declines to send it.
    """
    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        # Absence says nothing (the sibling gate's doctrine: a deck that
        # says nothing cannot be disagreed with) -- and a launch without
        # its deck fails loudly on its own; this gate's question is only
        # whether an EXISTING deck would warm-start.
        return
    text = deck.read_text(encoding="utf-8", errors="replace")
    # THE RESTART GROUP, as the catalogue spells it (`[item.restart]
    # .expands`), read off the deck by the one door (`script_emit.
    # deck_values`) and each switch as fdf reads one (`parse.fdf.fdf_true`).
    from ..parse.fdf import fdf_true
    from ..script_emit import deck_values
    saves = deck_values("restart", "siesta", text)
    if not saves:
        raise TrialNotCold(
            f"trial {job.name!r}: {deck.name} carries no restart group at "
            f"all, so its cold start cannot be vouched for (prep always "
            f"writes the clean group).  " + _bench_anew(_calculation_of(job_dir)))
    warm = [k for k, v in saves.items() if fdf_true(str(v).split())]
    if warm:
        raise TrialNotCold(
            f"trial {job.name!r}: {deck.name} would WARM-start "
            f"({', '.join(warm)} true) -- that measures a continued run, "
            f"not its point (generator.md § 4.3a).  For the cold deck, "
            + _bench_anew(_calculation_of(job_dir)))
