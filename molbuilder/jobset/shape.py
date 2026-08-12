"""``Shape`` — where a stage's files live, in whichever layout was chosen.

**Module:** floor 4 (layout).  Imports exactly one thing of the tree —
``task.SHAPES``, floor 2's vocabulary constant — and callers hand it
tokens, so it stays reachable from anywhere that needs a path.  *(R8,
2026-08-12: this header claimed an ``identity`` import it does not make
and labelled ``task`` L1 where the floor table puts the description on
floor 2.)*

**Contract:** [`execution/project-layout.md`](?doc=execution/project-layout.md)
§ 1 (the two shapes, and the table this object is a transcription of) ·
[`engines/stages.md`](?doc=engines/stages.md) § 6.7 (*required, never
inferred*; *"`prep` **reads** it; it does not decide it"*) ·
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6.3 (the
names, in both shapes).

WHY THIS IS AN OBJECT AND NOT AN ``if shape ==`` IN FOUR MODULES.  The two
layouts do not differ by one branch — they differ in **what a stage is told
apart BY**:

    hierarchical   a PATH      01_coarse/run-0/…      every stage its own tree
    flat           a FILENAME  bdt_01_coarse-run0.out  one directory, all of them

Every layer built for the hierarchy assumes the distinction lives in the path,
because for the hierarchy it does. Handed a flat calculation those layers do not
fail — they answer about **the wrong stage**, or about all of them at once, and
say nothing. That asymmetry is the whole reason this is a type: a caller asks
*where does this stage's state live* and gets an answer that is correct in both,
instead of asking *which directory* and being right in one.

WHAT IT DELIBERATELY DOES NOT DO.  It does not read a description, and it does
not guess: `engines/stages.md` § 6.7 makes the shape a required field, and a
default here would be the inference that section refuses. The **surfaces**
resolve it — they are the ones holding a bundle directory — and hand it down.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..task import SHAPES


@dataclass(frozen=True)
class Shape:
    """One of the two layouts, and the questions a layout can answer.

    Construct with :meth:`named`, which refuses anything that is not one of
    ``project-layout.md § 1``'s two.
    """

    name: str

    @classmethod
    def named(cls, name: str) -> "Shape":
        """The shape called ``name``, or ``ValueError`` naming both."""
        if name not in SHAPES:
            raise ValueError(
                f"shape {name!r} is not one of {' / '.join(SHAPES)}. It is a "
                f"required field of the description and is never inferred "
                f"(engines/stages.md § 6.7)")
        return cls(name)

    # -- the two questions a layout answers ---------------------------- #

    def stage_dir(self, token: str) -> str:
        """Where this stage's work happens, **relative to the bundle root**.

        Hierarchical gives it a directory of its own (``01_coarse``); flat is
        **depth 1** (`project-layout.md` § 1) and everything happens in the
        bundle root, which is ``"."`` — a real path a caller can join, so no
        caller needs an ``if`` to handle "no directory".
        """
        return token if self.name == "hierarchical" else "."

    def stage_glob(self, token: str, label: str) -> str:
        """A glob matching the files **this stage** produced, inside
        :meth:`stage_dir`.

        This is the half that does not exist in the hierarchy's world view.
        There, the directory has already selected the stage, so anything in it
        belongs to it. In flat, one directory holds every stage, and they are
        told apart by the deck's token carried in each filename
        (`job-contracts.md § 6.3`) — so the *name* does the selecting.

        Returning a glob rather than a boolean is what lets one caller write
        one line: ``(base / sh.stage_dir(t)).glob(sh.stage_glob(t, label))``
        is correct in both, and there is no second code path to keep in step.
        """
        return "*" if self.name == "hierarchical" else f"{label}_{token}*"

    @property
    def keeps_attempts_as_directories(self) -> bool:
        """Whether a re-run makes a **directory** (``run-1``) or an index.

        `project-layout.md` § 1: hierarchical separates attempts by directory,
        flat by an **output index** (``-run0.out``) that the wrapper writes.
        So the whole attempt layer — ``prepare_attempt``, ``latest_attempt``,
        ``run.json`` — is hierarchical's, and in flat there is nothing to open:
        *"continuing: free — the next stage finds them lying there."*
        """
        return self.name == "hierarchical"

    def __str__(self) -> str:
        return self.name


__all__ = ["Shape"]
