"""What molbuilder prints for a person to type -- every `molbuilder jobset`
line a verb, a refusal, a next step or a status names, composed here once
(`execution/job-system.md` § 5.3: *what molbuilder prints, you can type*).

**Module:** L2 (jobset).  A line composed here is ONE command:

* the calculation named by its address from the projects root, every time
  (:func:`bundle_flag`);
* a launch's mode stated where this calculation's config sets none -- one
  line per mode its machine takes, each typeable (:func:`launch_lines`);
* any prose after ``#``.

What a text read LATER says about a launch -- a deck's header, a result's
remedy -- is `identity.launch_as_typed`: it is read wherever it was copied, so
it names no folder.  A layer below this one (a machine record's refusal, a
wrapper's banner) cannot import it and prints plain text by the same rules.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import List, Optional

#: The program as a person types it.
PROG = "molbuilder jobset"

#: THE KINDS a verb acts on (`job-system.md` § 5.3): the calculation as a
#: TASK, its groups named with ``--group`` and a stage alone with
#: ``--stage``; a BENCHMARK of one stage, named by position.
TASK, BENCH = "task", "bench"


def words_for(kind: str, *stages, groups=()) -> tuple:
    """A verb's words for ``kind`` and what it names -- the ONE spelling
    every printed line takes: ``task --group 1 --group 2`` (a task's
    groups, the answer to the question its verbs ask), ``task --stage a``
    (a member launched again alone, D4), ``bench a [trial]`` (a benchmark's
    stage and trial by position).  A stage that is ``None`` is left out."""
    named = [s for s in stages if s is not None]
    if kind == TASK:
        return (TASK, *(x for g in groups for x in ("--group", str(g))),
                *(x for s in named for x in ("--stage", s)))
    return (kind, *named)


def stage_group(base, stage: str) -> Optional[int]:
    """The group ``stage`` is in, as the calculation's description numbers
    it (`engines/stages.md` § 2), or ``None`` -- a description that does
    not read is its reader's to refuse, where it is read."""
    from ..task import FILENAME, read_task
    from .group import number_of
    try:
        task = read_task(Path(base) / FILENAME)
    except (OSError, ValueError):
        return None
    return number_of(task, stage)


def prep_group_lines(numbers, *, base) -> List[str]:
    """The prep of the groups ``numbers`` -- one command."""
    return [command("prep", *words_for(TASK, groups=list(numbers)),
                    base=base)]


def bundle_flag(base) -> str:
    """`` --bundle <address from the projects root>`` -- every time.  A line
    is pasted where its reader is, and the reader moves: an omitted
    ``--bundle`` means the working folder (`job-contracts.md` § 2.5b), so a
    line that left it out because it was printed inside the calculation
    acted on whatever calculation the next shell stood in.  A folder outside the projects tree is
    named by its full path, which `--bundle` refuses by name -- never a line
    with no calculation in it.  The address is quoted as the shell needs
    it: a folder name may hold a space."""
    from ..projects import projects_root
    here = Path(base).resolve()
    try:
        rel = here.relative_to(Path(projects_root()).resolve())
    except (ValueError, OSError):
        rel = here
    return f" --bundle {shlex.quote(str(rel))}"


def command(verb: str, *words, base, flags=()) -> str:
    """``molbuilder jobset <verb> <words> <flags>`` and the calculation's
    ``--bundle``: one command.  ``base`` is required -- the calculation a
    line acts on is never left to where it happens to be pasted (D1); pass
    ``None`` for a line that names none (`init`, `probe`).  A word that is
    ``None`` is left out -- the verb's form without it -- never printed as
    ``None``."""
    parts = [PROG, verb, *(str(w) for w in words if w is not None),
             *(str(f) for f in flags)]
    return " ".join(parts) + (bundle_flag(base) if base is not None else "")


def configured_mode() -> Optional[str]:
    """The launch mode this machine's config states, or ``None`` --
    `runtime_config.get_launch_mode`, the reader `launch` falls back on."""
    from ..runtime_config import RuntimeConfigError, get_launch_mode
    try:
        return get_launch_mode() or None
    except RuntimeConfigError:
        return None          # a malformed config is `launch`'s to refuse


def takes_a_queue(base, target: Optional[str] = None) -> bool:
    """Whether a launch of this calculation can go to a queue: the machine
    it launches on names one -- its record says ``slurm``, the test a
    header is written on (`runwrap._render_sbatch_for`) -- so a launch to it
    can be sent.  The calculation's own copy of its machine's record answers
    once it is set to one; before that, the record of the machine named for
    its first prep (``target``).  A machine that cannot be told -- no record
    yet, or several on file and none named -- is taken to have one: the prep
    that sets it answers first."""
    from ..scheduler import AmbiguousTarget, UnknownTarget, machine_for
    try:
        rec = machine_for(base, target=target)
    except (AmbiguousTarget, UnknownTarget):
        return True                     # the prep that sets it refuses first
    return rec is None or rec.scheduler == "slurm"


def launches_here(base, target: Optional[str] = None) -> bool:
    """Whether the calculation is launched on THIS machine -- the machine
    it is set to (`scheduler.record.calculation_machine`, M-3), or before
    its first prep the one named for it (``target``), is this one."""
    from ..scheduler.record import LOCAL_TARGET, calculation_machine
    return (calculation_machine(base) or target
            or LOCAL_TARGET) == LOCAL_TARGET


#: The launch flags only a queue reads -- what a scheduler is asked for,
#: and the side of a benchmark sent to one.  A run here refuses each by name
#: (`_cli._refuse_flags_without_effect`), and a line that offers the launch
#: here leaves them off (:func:`launch_with`).
QUEUE_FLAGS = ("--domain", "--time", "--mem", "--gpu-domain", "--only")


def launch_with(kind: str, *words, base, mode: str, typed=(),
                groups=()) -> str:
    """The launch of ``words`` with ``mode`` stated, and the flags it was
    typed with (``typed``, ``(flag, value)`` pairs as the verb records them,
    a switch's value ``True``)
    that ``mode`` reads -- a run here leaves off :data:`QUEUE_FLAGS`.  What a
    line says when it offers a launch again in another mode: the whole
    command, never *"the same command with --mode X"*, an edit to a line the
    person may never have typed -- a bare launch takes its mode from config
    (`job-system.md` § 5.3)."""
    kept = [x for flag, value in typed
            if not (mode == "direct" and flag in QUEUE_FLAGS)
            for x in ((flag,) if value is True
                      else (flag, shlex.quote(str(value))))]
    return command("launch", *words_for(kind, *words, groups=groups),
                   base=base, flags=(*kept, "--mode", mode))


def launch_lines(kind: str, *words, base,
                 target: Optional[str] = None, flags=(),
                 groups=()) -> List[str]:
    """The launch of ``words`` as a person types it: the bare line where the
    calculation is launched on this machine and this machine's config names
    a mode -- `launch` takes it -- else one line per mode the calculation's
    machine takes (:func:`takes_a_queue`), each a command (never
    ``--mode submit|direct``, which bash reads as a pipe).  A calculation
    set to another machine is launched there, whose config this one does
    not read: its lines state their mode.  ``flags`` are switches every
    line carries (``--cold``); ``groups`` the groups it names."""
    if configured_mode() and launches_here(base, target):
        return [command("launch", *words_for(kind, *words, groups=groups),
                        base=base, flags=flags)]
    typed = [(f, True) for f in flags]
    here = (launch_with(kind, *words, base=base, mode="direct", typed=typed,
                        groups=groups)
            + "   # here")
    if not takes_a_queue(base, target):
        return [here]
    return [here, launch_with(kind, *words, base=base, mode="submit",
                              typed=typed, groups=groups)
            + "   # to the queue"]


def lines(verb: str, kind: str, *words, base) -> List[str]:
    """The command lines for ``verb`` -- a launch's from
    :func:`launch_lines`, any other verb's one :func:`command` -- so a caller
    that names a verb it was handed cannot print a launch with no mode."""
    if verb == "launch":
        return launch_lines(kind, *words, base=base)
    return [command(verb, *words_for(kind, *words), base=base)]


def run_first(stage: str, *, base) -> List[str]:
    """Run ``stage`` first -- its group's prep, then its group's launch: the
    remedy every refusal that waits on a stage not prepared gives."""
    g = stage_group(base, stage)
    if g is None:
        return launch_lines(TASK, stage, base=base)
    return prep_group_lines([g], base=base) + launch_lines(
        TASK, base=base, groups=[g])


def name_a_stage(verb: str, kind: str, refs, *, base) -> str:
    """``name one -- coarse ('#1'), medium ('#2'):`` and the command for the
    first -- what every verb that acts on ONE stage says when none was named.
    ``refs`` are the stages the verb takes, first the one to offer."""
    from ..identity import render_stage_choices
    refs = list(refs)
    if not refs:
        return "name a stage."
    return (f"name one -- {render_stage_choices(refs)}:\n"
            + block(lines(verb, kind, refs[0].name, base=base)))


def block(text_lines, pad: str = "    ") -> str:
    """Lines indented as a command block under a sentence -- a sentence
    ending ``--`` that introduces them (``Launch it --``)."""
    return "\n".join(pad + ln for ln in text_lines)


def reason_of(refusal: str) -> str:
    """A refusal's reason in one line, for a row beside others: its first
    line, less the sentence that ends ``--`` introducing the commands of
    its :func:`block` below -- the commands are the next step's to print
    (`runstatus.next_step`)."""
    first = refusal.splitlines()[0] if refusal else ""
    if not first.rstrip().endswith("--"):
        return first
    reason, sep, _intro = first.rpartition(".  ")
    return reason + "." if sep else first.rstrip(" -")


def rollback(what: str, *, base) -> str:
    """How ``what`` is redone: the state saved before it, restored, and a
    prep anew -- a launched stage is not prepared again (user, 2026-10-02:
    *"refuse it, redo via rollback"*; `job-system.md` § 5.0); its way back
    is the state saved before its group's first launch.

    WHICH STATE is the person's to pick, so it is asked for in words, never
    a ``<state>``; the list that shows them is a command, and both verbs
    name the folder -- by its path, `checkpoint`'s ``-p`` -- every time,
    as every printed line names its calculation (:func:`bundle_flag`)."""
    return (f"go back to the state saved before {what}, and prep anew -- "
            + _saved_states(base))


def _saved_states(base) -> str:
    """The folder's saved states, and how one is restored -- the command
    that lists them and the one that restores the one picked, each naming
    the folder."""
    from ..identity import checkpoint_as_typed
    shown, restore = checkpoint_as_typed(Path(base).resolve())
    return ("the folder's saved states:\n" + block([shown])
            + f"\n  then `{restore}` with the id of the one you pick.")


def redo_launched(stage: Optional[str], *, base, task=None) -> str:
    """How a LAUNCHED stage is changed (`job-system.md` § 5.0): the state
    saved before its group's first launch restored -- every run it builds on
    is in it, and there it is prepared, never launched -- and a prep anew.
    The group is named by the members it was launched with, as the state's
    note named them (`group.named`, at the launch): the description may
    hold other members now, and a number shifts when a stage above it is
    removed.  A stage launched before 2.7.0, whose launch saved no state,
    goes back to the state saved before its prep.  ``task`` is the
    description as the caller read it; with none it is read here."""
    from .group import (group_members, members_of, named, number_of,
                        same_members)
    from .ready import job_of
    if task is None and stage is not None:
        from ..task import FILENAME, read_task
        try:
            task = read_task(Path(base) / FILENAME)
        except (OSError, ValueError):
            task = None
    g = (number_of(task, stage)
         if task is not None and stage is not None else None)
    job = job_of(base, stage) if stage is not None else None
    ran = members_of(job) if job is not None else None
    what = ("the first launch of its group" if g is None else
            f"the first launch of {named(task, g)}"
            if ran is None or same_members(ran, group_members(task, g)) else
            f"the first launch of its group as it was launched "
            f"({', '.join(ran)})")
    return (f"go back to the state saved before {what} -- for a stage "
            f"launched before 2.7.0, the one saved before its prep -- and "
            f"prep anew -- " + _saved_states(base))


def way_back(stage: str, *, base, task, launched: bool) -> str:
    """How a prepared stage is made as the description gives it now
    (`job-system.md`, *The task*) -- the one composer of a stage's way back:
    never launched, its group prepared again; launched,
    :func:`redo_launched`.  ``launched`` is the caller's answer from the one
    rule (`materialize.ever_launched`)."""
    from .group import number_of
    g = number_of(task, stage) if task is not None else None
    if not launched and g is not None:
        return ("prepare its group again -- it was never launched:\n"
                + block(prep_group_lines([g], base=base)))
    return redo_launched(stage, base=base, task=task)


def launch_pending(stage: str, group: Optional[int], attempt: Optional[str],
                   *, base) -> List[str]:
    """The launch that sends ``stage``'s run that is opened and not launched
    -- ``attempt``, its folder's name (``None`` on the flat layout, which
    opens no run before its launch): prep's first run goes with its group,
    which goes whole the first time; a run a refused send opened for a stage
    launched before goes with the stage alone, launched again (D4)."""
    from ..paths import attempt_index
    n = attempt_index(attempt) if attempt else None
    if n or group is None:
        return launch_lines(TASK, stage, base=base)
    return launch_lines(TASK, base=base, groups=[group])


def target_flags(base, target: Optional[str]) -> tuple:
    """``--target <name>`` where a prep needs one
    (`preparing-for-another-machine.md` § 4): none for a calculation already
    set to its machine -- its copy of the record answers, and the machine
    does not change (`configuration.md` M-3); none for this machine when it
    is the only one on file; the name otherwise."""
    from ..scheduler import choice_required, known_machines
    from ..scheduler.record import LOCAL_TARGET, calculation_record
    if not target or calculation_record(base).is_file():
        return ()
    if target == LOCAL_TARGET:
        return (("--target", LOCAL_TARGET)
                if choice_required(known_machines()) else ())
    return ("--target", target)


def stage_lines(kind: str, *, base, groups=(), stages=(),
                from_attempt=None, cold: bool = False,
                target: Optional[str] = None, prepared: bool = False,
                from_at_prep: bool = False) -> List[str]:
    """What a person types next -- the lines Task setup shows, composed here
    as the terminal composes its own (`task-setup.md` § 11; W55 B4: the page
    composes none).  For a task's ``groups``: their prep -- the machine,
    and a force-constant stage's `relax` run (``from_attempt`` with
    ``from_at_prep``) -- unless ``prepared``; then, for one group, its
    launch, with what a group of one continues from (``from_attempt``,
    ``cold``).  For
    ``stages`` (one): a benchmark's prep, launch and read; a task's member
    launched again alone, with what it continues from."""
    out: List[str] = []
    if kind == BENCH:
        if not prepared:
            out.append(command("prep", *words_for(kind, *stages), base=base,
                               flags=target_flags(base, target)))
        out += launch_lines(kind, *stages, base=base, target=target)
        out.append(command("summarize", BENCH, *stages, base=base))
        return out
    # WHAT IT CONTINUES FROM, said at its launch -- a force-constant
    # stage's `relax` run is its prep's.
    launch_flags = ((() if from_at_prep or not from_attempt
                     else ("--from", from_attempt))
                    + (("--cold",) if cold else ()))
    if stages:
        return launch_lines(kind, *stages, base=base, target=target,
                            flags=launch_flags)
    if not prepared:
        out.append(command(
            "prep", *words_for(kind, groups=list(groups)), base=base,
            flags=((("--from", from_attempt)
                    if from_attempt and from_at_prep else ())
                   + target_flags(base, target))))
    if len(groups) > 1:
        # SEVERAL GROUPS PICKED FOR ONE PREP: each launched as its own job.
        return out
    out += launch_lines(kind, base=base, target=target, groups=list(groups),
                        flags=launch_flags)
    return out


__all__ = ["PROG", "TASK", "BENCH", "words_for", "bundle_flag", "command",
           "configured_mode", "stage_group", "prep_group_lines",
           "launch_lines", "lines", "run_first", "name_a_stage",
           "block", "reason_of", "rollback", "redo_launched", "way_back",
           "launch_pending",
           "stage_lines", "takes_a_queue", "target_flags"]
