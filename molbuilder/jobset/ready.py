"""THE READY DOOR -- whether a stage not yet prepared can be prepared now,
and what it would take (`job-system.md`, *The task*; `architecture.md`
§ 3.2).

**One question, answered by the doors that already decide it.**  What a
stage takes is decided in one place per kind: an independent stage's or a
force-constant stage's continuation (`continuation.continuation_answer` --
the stage before it, or `relax`, its newest run finished), a transport
rung's gather (`prep.gather_sources` -- each upstream rung's newest run,
finished, having run the deck it renders now).  A stage is **ready** when
that door answers without refusing; **waiting** when it refuses, its words
saying for what.  Nothing here decides a second way, and nothing is written.

Readers: `prep task`'s ladder and offer, `status`'s rows and next step,
Task setup's Prep -- and prep itself asks the same two doors when it
plans the stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Readiness:
    """One stage's answer: ``prepared`` (asked of the prepared door first),
    else ``ready`` with what it takes, or waiting with ``why``."""
    stage: str
    prepared: bool = False
    ready: bool = False
    #: What it would take, one line each -- the run it continues from, or
    #: each file a transport rung gathers and the run it comes from; ``[]``
    #: for a stage that builds on nothing.
    takes: List[str] = field(default_factory=list)
    #: The refusal of the door that decides what it takes -- what it waits
    #: for, and what to do -- for a stage that is waiting.
    why: Optional[str] = None

    @property
    def state(self) -> str:
        """``prepared`` / ``ready`` / ``waiting`` -- the word `status` shows."""
        return ("prepared" if self.prepared
                else "ready" if self.ready else "waiting")

    @property
    def detail(self) -> str:
        """One line beside the state -- what a ready stage takes (or that
        it builds on nothing), the first line of what one waiting waits
        for: `status`'s detail column and `prep task`'s ladder."""
        if self.prepared:
            return ""
        if not self.ready:
            return (self.why or "").splitlines()[0].rstrip(" -") if self.why \
                else "waiting"
        return "takes " + "; ".join(self.takes) if self.takes \
            else "builds on nothing"


def readiness(base, task, stage: str, *, template_text: Optional[str] = None,
              from_attempt: Optional[str] = None, cold: bool = False,
              verdict: bool = True) -> Readiness:
    """``stage``'s answer (`job-system.md`, *The task*) -- ``from_attempt``
    and ``cold`` asked as prep would take them: a run named is taken as
    said, so it makes a stage ready that waits for its default.
    ``verdict=False`` leaves a relaxation's convergence unread -- for
    `status`, read far more often than prep."""
    from .prep import (PrepError, _user_error_as_prep, gather_sources,
                       prepared_already)
    base = Path(base)
    if prepared_already(base, task, "task", stage):
        return Readiness(stage, prepared=True)
    if template_text is None:
        from ..template import find_template
        tpl = find_template(base, task.label)
        template_text = tpl.read_text(encoding="utf-8") if tpl else None
    if getattr(task, "calculation", None) == "transport":
        try:
            # THE REFUSALS PREP SAYS, translated as prep translates them.
            with _user_error_as_prep():
                got = gather_sources(base, task, stage,
                                     template_text=template_text)
        except PrepError as exc:
            return Readiness(stage, why=str(exc))
        return Readiness(stage, ready=True,
                         takes=[f"{fn} <- {src}" for _c, _v, inputs in got
                                for src, fn in inputs])
    from .continuation import continuation_answer
    cont, refused = continuation_answer(base, task, stage,
                                        from_attempt=from_attempt, cold=cold,
                                        verdict=verdict,
                                        template_text=template_text)
    if refused:
        return Readiness(stage, why=refused)
    return Readiness(stage, ready=True,
                     takes=([cont.line(would=True)] if cont is not None
                            else []))


def ladder(base, task, *, template_text: Optional[str] = None,
           verdict: bool = True) -> List[Readiness]:
    """Every stage of the description, in ladder order, with its answer --
    what `prep task` shows before it asks, and `status` beside each stage
    not prepared."""
    from .materialize import ladder_homes
    if template_text is None:
        from ..template import find_template
        tpl = find_template(Path(base), task.label)
        template_text = tpl.read_text(encoding="utf-8") if tpl else None
    return [readiness(base, task, h.name, template_text=template_text,
                      verdict=verdict)
            for h in ladder_homes(Path(base), task)]


def preselected(base, task, answers: List[Readiness]) -> List[str]:
    """The stages `prep task` offers pre-selected (D2, `job-system.md`, *The
    task*): for a kind whose stages each have a role -- transport, a
    vibration (`template.KIND_ROLES`) -- every ready stage that builds on
    nothing (`group.upstream_of`), which may share one job; for any other
    kind, the first ready stage."""
    from ..template import KIND_ROLES
    from .group import upstream_of
    ready = [a.stage for a in answers if a.ready]
    if not ready:
        return []
    if getattr(task, "calculation", None) in KIND_ROLES:
        free = [s for s in ready if not upstream_of(base, task, s)]
        return free or ready[:1]
    return ready[:1]


def ladder_text(answers: List[Readiness]) -> str:
    """The ladder as `prep task` shows it before it asks: each stage, where
    it stands, and for one waiting the first line of what it waits for."""
    width = max([len(a.stage) for a in answers] + [5])
    out = []
    for a in answers:
        out.append(f"  {a.stage:<{width}}  {a.state:<8}  {a.detail}")
    return "\n".join(out)


__all__ = ["Readiness", "readiness", "ladder", "preselected", "ladder_text"]
