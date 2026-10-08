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

Readers: `prep task`'s offer and its check, `status`'s rows and next step,
Task setup's Prep.
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


def readiness(base, task, stage: str, *, template_text: Optional[str] = None,
              from_attempt: Optional[str] = None,
              cold: bool = False) -> Readiness:
    """``stage``'s answer (`job-system.md`, *The task*) -- ``from_attempt``
    and ``cold`` asked as prep would take them: a run named is taken as
    said, so it makes a stage ready that waits for its default."""
    from .prep import PrepError, gather_sources, prepared_already
    base = Path(base)
    if prepared_already(base, task, "task", stage):
        return Readiness(stage, prepared=True)
    if template_text is None:
        from ..template import find_template
        tpl = find_template(base, task.label)
        template_text = tpl.read_text(encoding="utf-8") if tpl else None
    if getattr(task, "calculation", None) == "transport":
        try:
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
                                        template_text=template_text)
    if refused:
        return Readiness(stage, why=refused)
    return Readiness(stage, ready=True,
                     takes=([cont.line(would=True)] if cont is not None
                            else []))


def ladder(base, task, *, template_text: Optional[str] = None
           ) -> List[Readiness]:
    """Every stage of the description, in ladder order, with its answer --
    what `prep task` shows before it asks, and `status` beside each stage
    not prepared."""
    from .materialize import ladder_homes
    if template_text is None:
        from ..template import find_template
        tpl = find_template(Path(base), task.label)
        template_text = tpl.read_text(encoding="utf-8") if tpl else None
    return [readiness(base, task, h.name, template_text=template_text)
            for h in ladder_homes(Path(base), task)]


__all__ = ["Readiness", "readiness", "ladder"]
