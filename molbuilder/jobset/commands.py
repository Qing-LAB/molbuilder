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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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


def stage_group(request, stage: str) -> Optional[int]:
    """The group ``stage`` is in, as the job request numbers it
    (`execution/job-request.md` § 3.3), or ``None`` with no request."""
    return request.group_of(stage) if request is not None else None


def prep_group_lines(numbers, *, base) -> List[str]:
    """The prep of the groups ``numbers`` -- one command."""
    return [command("prep", *words_for(TASK, groups=list(numbers)),
                    base=base)]


def prep_sequence_lines(numbers, *, base) -> List[str]:
    """The prep of the groups ``numbers`` as ONE job launching each in turn,
    its header written too (`job-system.md` § 6.0, *A sequence of
    groups*) -- one command."""
    return [command("prep", *words_for(TASK, groups=sorted(numbers)),
                    base=base, flags=("--one-job",))]


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
    ``None`` for a line that names none (`molbuilder describe`, `probe`).  A word that is
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
    it launches on names one -- its record's scheduler kind has a queue, the
    test a header is written on (`runwrap._render_sbatch_for`) -- so a launch
    to it can be sent.  The calculation's own copy of its machine's record answers
    once it is set to one; before that, the record of the machine named for
    its first prep (``target``).  A machine that cannot be told -- no record
    yet, or several on file and none named -- is taken to have one: the prep
    that sets it answers first."""
    from ..scheduler import AmbiguousTarget, UnknownTarget, machine_for
    try:
        rec = machine_for(base, target=target)
    except (AmbiguousTarget, UnknownTarget):
        return True                     # the prep that sets it refuses first
    return rec is None or rec.kind.queue


def launches_here(base, target: Optional[str] = None) -> bool:
    """Whether the calculation is launched on THIS machine -- the machine
    it is set to (`scheduler.record.calculation_machine`, M-3), or before
    its first prep the one named for it (``target``), is this one."""
    from ..scheduler.record import LOCAL_TARGET, calculation_machine
    return (calculation_machine(base) or target
            or LOCAL_TARGET) == LOCAL_TARGET


@dataclass(frozen=True)
class LaunchFlag:
    """One flag a launch reads -- a fact of how the launch SENDS what prep
    wrote, never a value of the job: every value is prep's, and the one a
    launch may restate is a submitted job's wall (`job-system.md` § 6.0;
    the user, 2026-10-17)."""
    #: As typed.
    flag: str
    #: The modes that read it.
    modes: Tuple[str, ...]
    #: The launches that read it: a task's, a benchmark's (a grouped one,
    #: when ``grouped``).
    kinds: Tuple[str, ...]
    #: What it is, in the words a refusal says it in.
    what: str
    #: A benchmark's grouped walk only -- a named trial is sent alone.
    grouped: bool = False
    #: The value of the scheduler's line it restates -- the kind's words
    #: for it (`scheduler/kinds.toml`, `restate`).
    says: Optional[str] = None


#: EVERY FLAG A LAUNCH READS beside the work it names and how it asks
#: (`--mode`, `--yes`, `--dry-run`): what the verb records as typed and
#: refuses where it would not be read (`_cli._refuse_flags_without_effect`),
#: what a line offering the launch in another mode keeps
#: (:func:`launch_with`), and what the line a launch sends restates
#: (`submit._sbatch_request`).
LAUNCH_FLAGS: Tuple[LaunchFlag, ...] = (
    LaunchFlag("--time", modes=("submit", "ask"), kinds=(TASK, BENCH),
               what="a submitted job's wall, restated on the line that "
                    "sends it",
               says="wall"),
    LaunchFlag("--only", modes=("submit", "ask"), kinds=(BENCH,),
               what="one side of a benchmark, sent to its queue",
               grouped=True),
    LaunchFlag("--trial-timeout", modes=("submit", "direct"),
               kinds=(BENCH,), what="a benchmark's per-trial bound",
               grouped=True),
    LaunchFlag("--foreground", modes=("direct",), kinds=(TASK, BENCH),
               what="waits for a run here"),
    # WHAT A STAGE'S RUN TAKES (`job-system.md` § 5.4) -- a benchmark's
    # trial starts cold always, from its deck (§ 7).
    LaunchFlag("--cold", modes=("submit", "ask", "direct"), kinds=(TASK,),
               what="a stage's run taking nothing from a run before"),
    LaunchFlag("--from", modes=("submit", "ask", "direct"), kinds=(TASK,),
               what="the run a stage's run continues from"),
    LaunchFlag("--gather", modes=("submit", "ask", "direct"), kinds=(TASK,),
               what="a transport rung's inputs, taken anew"),
)


# WHAT A LAUNCH RESTATES IS A VALUE EVERY KIND WITH A QUEUE STATES THE WORDS
# OF (`scheduler.kinds.RESTATES`, checked on every kind as it is read), or
# nothing here imports.
from ..scheduler.kinds import RESTATES as _RESTATES  # noqa: E402
if not {f.says for f in LAUNCH_FLAGS if f.says} <= set(_RESTATES):
    raise ImportError("a launch flag restates a value no scheduler kind is "
                      f"asked to state ({', '.join(_RESTATES)})")


def launch_flag(flag: str) -> LaunchFlag:
    """The row of :data:`LAUNCH_FLAGS` for ``flag``."""
    return next(f for f in LAUNCH_FLAGS if f.flag == flag)


def launch_with(kind: str, *words, base, mode: str, typed=(),
                groups=()) -> str:
    """The launch of ``words`` with ``mode`` stated, and the flags it was
    typed with (``typed``, ``(flag, value)`` pairs as the verb records them,
    a switch's value ``True``)
    that ``mode`` reads (:data:`LAUNCH_FLAGS`; a flag no row names --
    `--yes` -- is every mode's).  What a
    line says when it offers a launch again in another mode: the whole
    command, never *"the same command with --mode X"*, an edit to a line the
    person may never have typed -- a bare launch takes its mode from config
    (`job-system.md` § 5.3)."""
    read = {f.flag: f.modes for f in LAUNCH_FLAGS}
    kept = [x for flag, value in typed
            if mode in read.get(flag, (mode,))
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


def run_first(stage: str, *, calc) -> List[str]:
    """Run ``stage`` first -- its group's prep, then its group's launch: the
    remedy every refusal that waits on a stage not prepared gives; ``calc``
    the calculation as the verb read it (A16)."""
    base = calc.base
    g = stage_group(calc.request, stage)
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


def said_lines(text: str) -> List[str]:
    """A composed way on, as a step's lines: each of its lines, the
    commands among them, in order."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def reason_of(refusal: str) -> str:
    """A refusal's reason in one line, for a row beside others: its first
    line, less the sentence that ends ``--`` introducing the commands of
    its :func:`block` below -- the commands are the next step's to print
    (`runstatus.render_steps`)."""
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


def redo_launched(stage: Optional[str], *, calc) -> str:
    """How a LAUNCHED stage is changed (`job-system.md` § 5.0): the state
    saved before its group's first launch restored -- every run it builds on
    is in it, and there it is prepared, never launched -- and a prep anew;
    for a group copied in from a finished calculation, the state saved
    before the prep that copied it in.
    The group is named by the members it was launched with, as the state's
    note named them (`group.named`, at the launch): the job request may
    hold other members now, and a number shifts when a stage above it is
    removed.  A stage launched before 2.7.0, whose launch saved no state,
    goes back to the state saved before its prep.  ``calc`` is the
    calculation as the verb read it (A16): its job request and its plan."""
    from ..runrecord import copied_in
    from .group import members_of, named, same_members
    from .naming import stage_home
    base, request = calc.base, calc.request
    g = (request.group_of(stage)
         if request is not None and stage is not None else None)
    # A GROUP COPIED IN from the finished calculation a transport
    # calculation reuses was launched by no launch of this one: its way back
    # is the state saved before the prep that copied it in
    # (`engines/transport.md` § 2a.16).
    if (stage is not None and g is not None
            and copied_in(stage_home(base, calc.task, stage).dir)):
        return (f"go back to the state saved before the prep that copied "
                f"{named(request, g)} in from {Path(calc.task.reuses).name} "
                f"-- and prep anew -- " + _saved_states(base))
    job = calc.job(stage) if stage is not None else None
    ran = members_of(job) if job is not None else None
    what = ("the first launch of its group" if g is None else
            f"the first launch of {named(request, g)}"
            if ran is None or same_members(ran, request.members(g)) else
            f"the first launch of its group as it was launched "
            f"({', '.join(ran)})")
    return (f"go back to the state saved before {what} -- for a stage "
            f"launched before 2.7.0, the one saved before its prep -- and "
            f"prep anew -- " + _saved_states(base))


def way_back(stage: str, *, calc, launched: bool) -> str:
    """How a prepared stage is made as its records give it now
    (`job-system.md`, *The task*) -- the one composer of a stage's way back:
    never launched, its group prepared again; launched on the hierarchical
    layout, its group prepared again as long as what it computes is
    unchanged, else :func:`redo_launched` (`execution/job-request.md` § 5);
    launched on the flat layout, :func:`redo_launched`.  ``launched`` is the
    caller's answer from the one rule (`materialize.ever_launched`)."""
    base, request = calc.base, calc.request
    g = request.group_of(stage) if request is not None else None
    if g is not None and not launched:
        return ("prepare its group again -- it was never launched:\n"
                + block(prep_group_lines([g], base=base)))
    if g is not None and calc.shape.keeps_attempts_as_directories:
        return ("prepare its group again -- a launched stage is prepared "
                "again for its job request, each deck the same calculation "
                "as its runs ran:\n" + block(prep_group_lines([g], base=base))
                + "\n  For a change of what it computes, "
                + redo_launched(stage, calc=calc))
    return redo_launched(stage, calc=calc)


def order_recorded(*, base) -> str:
    """How a transport calculation's junction whose sidecar states no order
    record has it recorded (`engines/transport.md` § 1.1, *The first prep*):
    `prep task`, with a group left to prepare or none, checks the junction
    against its citation and records its order -- a sentence ending ``--``
    and its line."""
    return ("`prep task` records it, checking the junction against its "
            "citation -- with a group left to prepare or none --\n"
            + block([command("prep", *words_for(TASK), base=base)]))


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
           "prep_sequence_lines",
           "launch_lines", "lines", "run_first", "name_a_stage",
           "block", "said_lines", "reason_of", "rollback", "redo_launched", "way_back",
           "order_recorded", "launch_pending",
           "stage_lines", "takes_a_queue", "target_flags"]
