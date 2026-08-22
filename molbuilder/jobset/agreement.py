"""Does the deck agree with the launch?  (P6 unit 2 · `project-layout.md`
§ 2.3.1 — *"step 3 cannot precede step 1"* · § 2.3.4 M5)

**Module**: the ONE comparison between what a deck was rendered for and the
launch it is about to get, plus the ONE wording of why a mismatch matters.
**Callers**: `jobset/submit._resolve_launch` (the refusal, M5's *"`submit`
decides nothing … refuses if they do not"*), `jobset/_cli._echo_resolved`
(the warning at `prep`, where changing your mind is still cheap), and --
since the cold-start ruling (2026-08-21) -- `jobset/submit`'s per-trial
gate calling :func:`check_trial_starts_cold` (the door VERIFIES the
run's starting state; prep's measurement pin is the one setter); the
fixture-level tests in `test_jobset` / `test_prep_calculation`.

**Why its own module** (U6, 2026-08-12): this comparison lived in `prep.py`
while its refusal vocabulary (`SubmitError`) lived in `submit.py`, so prep
reached down for the error and submit reached up for the checker — the one
import cycle in the jobset package, hidden behind a lazy import.  The
comparison serves BOTH surfaces and belongs to neither; giving it a floor
both import downward from dissolves the cycle instead of managing it.

The warning and the refusal cannot come to different conclusions because
both read :func:`launch_agreement`; after this module they cannot *say*
different things either, because both embed :func:`disagreement_note`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class LaunchAgreement:
    """Whether a deck was rendered for the launch it is about to get.

    ``verdict`` is one of:

    | ``"silent"`` | the deck makes no claim about its launch, so there is nothing to disagree with — and nothing to refuse |
    | ``"agrees"`` | the rank count it was rendered for is the one it will get |
    | ``"differs"`` | the two are different numbers, or one is `auto` and the other is not |

    ``rendered_for`` is what the deck's BENCH-MARKS block records (an int, or
    the string ``"auto"``); ``launching_at`` is the job's ``mpi_np`` (an int,
    or ``None`` meaning *let the wrapper decide*).
    """
    verdict:      str
    rendered_for: Optional[Union[int, str]] = None
    launching_at: Optional[int] = None

    @staticmethod
    def _fmt(v) -> str:
        return "auto" if v in ("auto", None) else str(v)

    @property
    def rendered_text(self) -> str:
        return self._fmt(self.rendered_for)

    @property
    def launch_text(self) -> str:
        return self._fmt(self.launching_at)


def launch_agreement(job_dir, job) -> LaunchAgreement:
    """Read the deck's launch claim and compare it with this job's.

    `project-layout.md § 2.3.1` states the five steps and says the order is
    **forced**: *"Step 3 cannot precede step 1, because a deck carries values
    that depend on how it will be launched — a block size derived from the rank
    count … A parameter that depends on the launch cannot be decided before the
    launch is known."*

    Until 2026-08-11, step 3 ran at `molbuilder fdf` on whatever machine typed
    it, and the rank count was resolved hours later by the wrapper — the two
    halves of one ordered sequence in different places, with nothing carrying
    the first's answer to the third. On 2026-08-10 that produced a deck
    rendered with no rank count — so ``BlockSize`` from the size-only branch —
    launched at ``-np 14``, and SIESTA refused at startup with *"You have too
    many processors for the system size"*. The migration then moved rendering
    into `prep` (ladder step 4), so both halves resolve on one machine — and
    this comparison stays, because a deck can still meet a launch it was not
    rendered for: a re-prep with a new allocation, a hand-carried deck, an
    edited wrapper.

    P4 unit 5 put the launch quantity **into** the deck, which is why that
    failure was diagnosable at all. **Recording is not agreeing**: this is the
    comparison, and a person is owed its answer at the moment they are still
    deciding (`prep`) — not only at the moment they are committing cluster
    time (`launch`).
    """
    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        return LaunchAgreement("silent")
    from ..parse.scripts.bench_marks import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
    if not marks or "mpi_np" not in marks:
        # A deck with no BENCH-MARKS block says nothing about its launch, so
        # there is nothing to disagree with.  The check is an agreement between
        # two statements, never a demand that every deck make one.
        return LaunchAgreement("silent")
    rendered_for = marks["mpi_np"]                 # an int, or the str "auto"
    launching_at = job.resources.mpi_np            # an int, or None == auto
    agree = ((rendered_for == "auto" and launching_at is None)
             or rendered_for == launching_at)
    return LaunchAgreement("agrees" if agree else "differs",
                           rendered_for, launching_at)


def disagreement_note(a: LaunchAgreement) -> str:
    """The ONE wording of why a mismatch matters and what to do — embedded
    by the `prep` warning and the `launch` refusal alike, so the two
    surfaces cannot drift into explaining one fact two ways."""
    return (f"a deck derives values from the rank count -- BlockSize above "
            f"all -- so one rendered for a different launch is wrong for "
            f"this one (project-layout.md § 2.3.1: a parameter that depends "
            f"on the launch cannot be decided before the launch is known).  "
            f"Re-render the deck for this launch, or launch it at "
            f"{a.rendered_text}.")


class DeckLaunchMismatch(Exception):
    """The deck was rendered for one launch and is meeting another.
    `launch` translates this into its own refusal (M5: the refusal is
    submit's); anything else may catch it by name without importing
    `launch` at all."""


def check_launch_matches_deck(job_dir, job) -> None:
    """Refuse a launch the deck was not rendered for (P6 unit 2).

    Three outcomes, and the middle one is the live defect:

    * deck ``auto`` + launch ``auto`` — both defer to the wrapper. Fine.
    * deck ``auto`` + launch ``N`` — the deck's launch-derived values were
      computed with **no** rank count, and now one is being imposed. Refused.
    * deck ``N`` + launch ``M`` — refused, with both numbers named.

    This is the last honest moment, not the first: :func:`launch_agreement`
    answers the same question at `prep`, where it is still cheap to change
    your mind.
    """
    a = launch_agreement(job_dir, job)
    if a.verdict != "differs":
        return
    deck = os.path.basename(job.script)
    raise DeckLaunchMismatch(
        f"job {job.name!r}: this deck ({deck}) was rendered for mpi_np "
        f"{a.rendered_text}, and you are launching it at "
        f"{a.launch_text}.\n  " + disagreement_note(a) + "\n"
        "  The deck records what it assumed in its BENCH-MARKS block, "
        "which is what made this checkable.")


__all__ = ["LaunchAgreement", "launch_agreement", "disagreement_note",
           "DeckLaunchMismatch", "check_launch_matches_deck",
           "check_trial_starts_cold"]


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
    A warm or group-stripped deck refuses by name (remedy: one
    re-prep); an ABSENT deck passes silently -- absence says nothing,
    and the launch itself fails loudly without one.

    Raises :class:`DeckLaunchMismatch` (the same family as the launch
    agreement -- submission declines to act on a deck that disagrees).
    """
    import re

    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        # Absence says nothing (the sibling gate's doctrine: a deck that
        # says nothing cannot be disagreed with) -- and a launch without
        # its deck fails loudly on its own; this gate's question is only
        # whether an EXISTING deck would warm-start.
        return
    text = deck.read_text(encoding="utf-8", errors="replace")
    saves = re.findall(r"^[ \t]*([A-Za-z.]*UseSave[A-Za-z.]*)[ \t]+(\S+)",
                       text, re.M)
    if not saves:
        raise DeckLaunchMismatch(
            f"trial {job.name!r}: {deck.name} carries no restart group at "
            f"all, so its cold start cannot be vouched for (the clean "
            f"group is always written -- a deck without one was edited or "
            f"predates the pin).  Re-prep the bench.")
    warm = [k for k, v in saves
            if v.strip().lower().strip(".") in ("t", "true", "yes", "1")]
    if warm:
        raise DeckLaunchMismatch(
            f"trial {job.name!r}: {deck.name} would WARM-start "
            f"({', '.join(warm)} true) -- that measures a continued run, "
            f"not its point (generator.md § 4.3a).  Re-prep the bench to "
            f"restore the cold deck.")
