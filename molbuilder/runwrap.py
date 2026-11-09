"""Shell-wrapper emission — `prep` step 4 (render the wrapper).

Called by ``jobset/prep``, the one route that writes a run's files.

Each generated script (``.fdf`` or ``.py``) gets a sibling run script,
named by its stage's names (``runfiles.RunNames``), that activates the
right conda env and runs the engine.  ``jobset launch`` starts it with its
counts, once in its run's folder: a run is one start
(`project-layout.md` § 1.6.1).

The wrapper is small and human-readable, so a person can see the command
a run is about to execute.  It is written once, at the stage's prep, and
copied into each run.

Testing hook: tests inject a synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities`.
"""

from __future__ import annotations

#: THE LAUNCH CLAIM is the scheduler protocol's (`scheduler/kinds.py`:
#: `LAUNCH_CLAIM_VAR`, `LAUNCH_CLAIM`), read here by the wrapper's gate.
from .scheduler.kinds import LAUNCH_CLAIM, LAUNCH_CLAIM_VAR

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from .diagnostics import get_capabilities
from .envs import hints as _hints
from .runfiles import (RANK_LAUNCHER, ROLE_CONCLUDED, ROLE_SESSION_LOG,
                       deck_role, stderr_apart as _stderr_apart, tail)
# The channel-name rule, from the module that owns the file those names
# have to match.  It cannot import upward -- it ships to a compute node
# alone -- so it is the end of the exchange that gets to own the rule.
from .config_dir import CHANNEL_NAME_RULE, is_channel_name
# The way back a refusal names: the checkpoint commands, the folder in them;
# and the one door a run script takes a start from.
from .identity import (command_stage, holds_a_start, next_run_words,
                       started_by_launch, way_back_words)
from .jobset.errors import WrapperError
# The thread chain's rungs, one list for the script's chain and the run
# script's (`running-a-job.md` § 3.2).
from .runpoints import AGAIN, CONTINUE
from .parse.dirs.job import STOPPED
from .runrecord import conclude_line
from .runtime_info import (JOB_STARTED_VAR, NODE_CORES_VAR, THREAD_SOURCES,
                           WALL_SLACK_VAR, WALL_VAR)
# THE SESSION LOG'S LINES are its one module's, which travels beside the
# job: the wrapper renders them, every reader reads with them.
from .wrapper_log import (BINARY_LABEL, CONDA_ENV_LABEL, CWD_LABEL,
                          HOSTNAME_LABEL, LOG_CLOCK, LOG_LINE, NODE_LINE,
                          PYTHON_LABEL, RANKS_OMP_LINE,
                          USER_LABEL, VERSION_LABEL, WALL_LINE,
                          WRAPPER_LOG_START)

if TYPE_CHECKING:                       # floor 5 reading floor 3's object
    # Under TYPE_CHECKING because the annotation is all this module needs:
    # the renderers read a `Resources` whole, they never build one (A4 --
    # `resolve.py` is the one builder).  A runtime import would also be legal
    # -- floor 5 may read floor 3 -- but it would be an import nothing calls.
    from .jobset.model import Resources

# Shell-safety guard for wrapper emission.  The wrapper interpolates
# ``basename`` and ``script_name`` (the script's filename stem and
# the filename itself) into many bash strings -- some inside ``"..."``,
# some unquoted in glob lists, some inside ``$(...)`` substitutions.
# A filename containing ``"``, ``$``, ``` ` ```, ``\\``, ``;``, ``|``,
# ``&``, newlines, etc. would either break the wrapper or execute
# arbitrary code at run time.
#
# Rather than escape every interpolation (fragile + makes the bash
# unreadable for the user, who is meant to be able to read and edit
# the wrapper), we restrict the input to a safe alphabet at
# emission time: letters, digits, dot, underscore, hyphen -- a FILE
# name's alphabet, the dot included for the deck's own suffix
# (``<stem>.fdf``).  Wider than a label's (`runfiles`: no dot), which the
# names passed here already keep; this is the shell's guard, not the
# grammar.
_SAFE_WRAPPER_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


def _phys_cores_probe_block() -> str:
    """Bash that sets ``_phys_cores``, ``_n_sockets`` and ``_cps``.

    Prefer ``lscpu`` because it reports PHYSICAL cores; ``nproc`` counts
    hyperthread siblings, which would make ``_cps`` too large and lead to
    over-binding.  Falls back to ``nproc / 2`` (conservative 2-way-HT
    assumption) when lscpu is unavailable.

    ``|| true`` sits INSIDE each substitution on purpose: under
    ``set -e`` an assignment whose command substitution fails aborts the
    script, so without it a box with no ``lscpu`` would die HERE -- before
    reaching the fallback written for exactly that case.

    Shared by both engines.  SIESTA's GPU rank policy divides the socket
    among MPI ranks; a PySCF script reports the count, exported to it
    (`runtime_info.NODE_CORES_VAR`).  One probe, because "how many cores
    does this machine have" has one answer and two engines -- or a run
    script and its engine's python -- asking it separately is how they
    come to disagree.
    """
    return (
        '_phys_cores=$(LANG=C lscpu -p=Core,Socket 2>/dev/null '
        '| grep -v "^#" | sort -u | wc -l 2>/dev/null || true)\n'
        'if [ -z "$_phys_cores" ] || [ "$_phys_cores" -lt 1 ]; then\n'
        '    _logical=$(nproc --all 2>/dev/null '
        '|| getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)\n'
        '    _phys_cores=$(( _logical / 2 ))\n'
        '    [ "$_phys_cores" -lt 1 ] && _phys_cores=$_logical\n'
        'fi\n'
        '_n_sockets=$(LANG=C lscpu -p=Socket 2>/dev/null '
        '| grep -v "^#" | sort -u | wc -l 2>/dev/null || true)\n'
        'if [ -z "$_n_sockets" ] || [ "$_n_sockets" -lt 1 ]; '
        'then _n_sockets=1; fi\n'
        '_cps=$(( _phys_cores / _n_sockets ))\n'
        '[ "$_cps" -lt 1 ] && _cps=1\n'
        # THE RUN'S ONE COUNT, handed to its engine's own python
        # (`runtime_info.physical_core_count`), which reports it.
        f'export {NODE_CORES_VAR}="$_phys_cores"\n'
        # ONE PROBE, ONE REPORT: whoever measures it says what it
        # measured, so a CPU sweep records its node shape too
        # (`bench/result.py`'s `node_phys_cores`).
        'echo "' + NODE_LINE.format(phys="$_phys_cores", sockets="$_n_sockets",
                                    cps="$_cps") + '" >&2\n'
    )


def _stdout_role_of(engine: str) -> str:
    """The role a run of ``engine`` writes its stdout to -- the catalogue's
    (`runfiles.stdout_roles`): ``.out`` for SIESTA, ``.pyscf.log`` for PySCF,
    and neither is spelled here (`model/parse.md` § 5.5, R-RO1 -- the
    vocabulary binds WRITERS too).

    Refuses rather than falling back: a fallback would have a third engine's
    run script redirect into another engine's filename in silence.
    """
    from .runfiles import stdout_roles
    roles = stdout_roles(engine)
    if len(roles) != 1:
        raise WrapperError(
            f"no single stdout role for engine {engine!r} (candidates "
            f"{roles!r}).  The catalogue is `runfiles.WRITTEN`: the engine's "
            f"`output=\"stdout\"` row names the file.")
    return roles[0]


#: WHAT A NAME TEMPLATE'S PLACEHOLDERS BECOME IN THE RUN SCRIPT, which fills
#: them as it runs: the script's own process id (`runfiles.FIELDS["pid"]`).
_BASH_FILLS = {"pid": "$$"}


def _bash_fill(name: str) -> str:
    """A catalogue template -- a fixed-name row's
    (`runfiles.RANK_LAUNCHER`) -- as the run script spells it: each field
    filled by its run-time value."""
    unfilled = set(re.findall(r"\{([a-z_]+)\}", name)) - set(_BASH_FILLS)
    if unfilled:
        raise WrapperError(f"{name!r}: the run script has no value for "
                           f"{', '.join(sorted(unfilled))}.")
    for field, value in _BASH_FILLS.items():
        name = name.replace("{" + field + "}", value)
    return name


def _env_python_block(names, stdout_role: str, target_env: str) -> str:
    """Bash that names the run's output -- every writer of it below writes
    this name (`project-layout.md` § 1.6.3); the conclusion marker's is
    named with the session log's (:func:`_one_start_block`); ``names`` are
    the stage's (`runfiles.RunNames`), ``stdout_role`` the
    engine's stdout role (:func:`_stdout_role_of`).

    **Then the one check every job makes before anything runs**
    (`running-a-job.md` § 2): the activation gave this job its env --
    ``CONDA_PREFIX`` names ``target_env`` -- and that env's own python,
    ``$_mb_py``, is there.  Refused otherwise, the conclusion saying why
    (`parse.dirs.job.NO_ENV_PYTHON`), whatever the engine -- a dry run
    refused alike, but concluding nothing: it starts nothing."""
    from .parse.dirs.job import NO_ENV_PYTHON
    env_name = str(target_env).rstrip("/").rsplit("/", 1)[-1]
    return (
        f'_out_file="{names.name(stdout_role)}"\n'
        f"# --- The job's python: its env's own (running-a-job.md 2) ---\n"
        + env_python_check(env_name)
        + f'    echo "ERROR: this job has no python of its env \'{env_name}\' -- '
        f'CONDA_PREFIX is ${{CONDA_PREFIX:-<unset>}}, and '
        f'${{_mb_py:-<no python>}} is not there.  The activation did not '
        f'take, or the env holds no python of its own: rebuild it on this '
        f'machine ({_hints.fix_cmd("install", env_name)}).  Nothing was '
        f'started." >&2\n'
        + '    [ "$_dry_run" = "1" ] || ' + conclude_line(
            1, f"{NO_ENV_PYTHON} (%s)",
            '"${_mb_py:-${CONDA_PREFIX:-no CONDA_PREFIX}}"') + "\n"
        "    exit 1\n"
        "fi\n"
        "\n"
    )


def _monitor_block(names, notify_on_scf: bool,
                   notify_every_hours: float, notify_channels,
                   notify_report, *, cores: str, gpu: bool,
                   member: bool = False) -> str:
    """Bash that launches the background monitor -- the SAME block for every
    engine (`running-a-job.md` § 4.1, `run-reports.md` § 2.3).

    It tells the monitor WHICH RUN it watches -- its stage's names
    (`runfiles.RunNames`) -- and never a path: the monitor names its log, its utilisation CSV and every file it
    reads through `runfiles`, and reads them through the framework's readers
    that travel beside it (:func:`monitor_companions`).  ``member`` makes it
    a MEMBER of a run of points' monitor (`run-reports.md` § 2.4;
    `execution/runs-of-points.md` § 1.4): it finds its run by its folder's
    record (`calcdirs.where`), counts the wall from the run's launch record
    and its clock from the run's notification record, and sends no `start`
    and no `finish` of its own -- the run reports as one run, those sent by
    its points' report steps.

    It tells it WHAT THE JOB HOLDS too (`run-reports.md` § 2.1a): ``cores``
    is the shell arithmetic for the cores this engine is launched on -- a run
    started directly has no allocation, so its percentages are fractions of
    these -- and ``gpu`` whether the run uses a GPU, without which none is
    sampled or judged.

    It runs with the JOB's OWN python (``$_mb_py``, the activated env's,
    set after the activation) from the working dir -- needing no molbuilder
    install, nothing beyond the standard library and no repo on PATH --
    sleeps between wakes, and runs at
    `nice -n 19` so it never competes with the compute ranks.  Opt out with
    ``MB_MONITOR=0``.  Stopped by ``_mb_stop_monitor``; it also exits when
    this run script's PID (``$$``) disappears.
    """
    from .monitor import NOTIFY_CHANNELS_VAR, NOTIFY_REPORT_VAR
    head = (
        # A '# ---' header in the EMITTED text names the block as
        # `job-contracts.md` § 2.6 lists it.
        "# --- Background job monitor ---------------------------\n"
        '_monitor_pid=""\n'
        # THE POLICY, in the environment too, for a run of points' own
        # python to send the run's events with -- its start, milestones,
        # failed parts and finish (`mb_monitor.pyz notify`,
        # `engines/vibration.md` § 5.11 § 7a) -- the same two answers the
        # flags below carry.
        + ("" if notify_channels is None
           else f'export {NOTIFY_CHANNELS_VAR}="{",".join(notify_channels)}"\n')
        + ("" if notify_report is None
           else f'export {NOTIFY_REPORT_VAR}="{",".join(notify_report)}"\n')
    )
    return head + (
        f'if [ "${{MB_MONITOR:-1}}" = "1" ] '
        f'&& command -v nice >/dev/null 2>&1 '
        f'&& [ -f {MONITOR_BUNDLE} ]; then\n'
        f'    nice -n 19 "$_mb_py" {MONITOR_BUNDLE} '
        f'--label "{names.label}" --stage "{names.stage}" '
        + f'--util --cores "{cores}" '
        + ("--gpu " if gpu else "")
        + ("--member " if member else "")
        + '--interval "${MB_MONITOR_INTERVAL:-10}" '
        + ('--notify-on-scf ' if notify_on_scf else "")
        + (f'--notify-every-hours {notify_every_hours:g} '
           if notify_every_hours > 0 else "")
        # ABSENT is "nothing set up, nothing sent", `*` is every channel this
        # machine has and `""` is none -- so the flag is emitted whenever the
        # description has a `notify` block, including when what it names is
        # nothing at all (`run-reports.md` § 3.0).
        + ("" if notify_channels is None
           else f'--notify-channels "{",".join(notify_channels)}" ')
        # ABSENT is "every field the monitor can determine" and "" is "the
        # summary line alone" (`stages.md` 6.9).  BAKED HERE, so a running
        # job's report format cannot change because `task.json` was edited
        # while it sat in the queue.
        + ("" if notify_report is None
           else f'--notify-report "{",".join(notify_report)}" ')
        # ITS STDERR GOES TO THE RUN'S SESSION LOG, its stdout nowhere: the
        # monitor opens its `.monitor.log` only once it has loaded, so its
        # START is said there -- *starting*, then *started* or the error
        # that stopped it (`run-reports.md` § 2.6).
        + f'--watch-pid $$ >/dev/null 2>>"$_runwrap_log" &\n'
        f'    _monitor_pid=$!\n'
        f'    _log INFO "monitor: pid=$_monitor_pid (nice 19, interval '
        f'${{MB_MONITOR_INTERVAL:-10}}s, util-sampling; '
        f'reads the run through the framework shipped beside it) -> the '
        f'run\'s .monitor.log + .util.csv"\n'
        f'else\n'
        f'    _log INFO "monitor: not started (set MB_MONITOR=1; needs '
        f'nice + {MONITOR_BUNDLE} beside the job)"\n'
        f'fi\n'
    )


@dataclass(frozen=True)
class PointOfRun:
    """THIS RUN IS A POINT OF A RUN OF POINTS (`execution/runs-of-points.md`
    § 1.1): its monitor a member of the run's, its two report steps around
    its engine -- and, where its kind works between its points, the kind's
    own steps, which the report steps run in their place: ``steps``, a verb
    of the job's bundle (``finish``) per moment, ``before``, ``after`` and
    ``finish`` (the kind table's record, `kind_table.RunOfPoints.steps`).
    ``relaunch`` is what a launch again does with a point short of done --
    ``again`` or ``continue`` (`PointsOf.relaunch_of`): what the run
    script says after a failure (:func:`_launch_again`)."""
    relaunch: str = AGAIN
    steps: Optional[Dict[str, str]] = None
    #: The one role this point plays, where its stage plays several -- a
    #: SIESTA vibration's phase (`Rung.role_at`) -- so its record of what
    #: the engine will read lists that role's settings; ``None``, the
    #: stage's.
    role: Optional[str] = None

    def step(self, moment: str) -> Optional[str]:
        return (self.steps or {}).get(moment)


def _point_block(names, verb: str, point: Optional[PointOfRun],
                 finish: Optional[str], script_name: str, *,
                 failed: bool = False) -> str:
    """ONE OF A POINT'S TWO REPORT STEPS (`execution/runs-of-points.md`
    § 3, operations 9 and 10): the monitor's bundle, run with the job's own
    python in the point's folder -- ``point started`` before the engine,
    ``point ended`` after it, success or failure, told the engine's exit and
    the output -- which stamp the run's index, run the kind's own steps in
    their place and say the run's events.  **The report never fails the
    point**: what it cannot do is logged here.  **The kind's own step
    does**, as it always has: one before the engine that fails starts no
    engine, and one after a clean engine that fails is the job's -- the
    conclusion names it (`FINISH_FAILED`), the script exits with its code;
    after a failed engine its failure is logged and the conclusion stays the
    engine's (``failed``).  Empty for a run that is no point."""
    if point is None:
        return ""
    from .monitor import POINT_ENDED, POINT_VERB
    from .parse.dirs.job import FINISH_FAILED
    ended = verb == POINT_ENDED
    own = [(m, point.step(m)) for m in (("after", "finish") if ended
                                        else ("before",)) if point.step(m)]
    words = f"{POINT_VERB} {verb}"
    call = (f'"$_mb_py" {MONITOR_BUNDLE} {words} --label "{names.label}" '
            f'--stage "{names.stage}" --deck "{script_name}"'
            + (' --exit "$_engine_exit" --output "$_out_file"' if ended
               else "")
            + (f' --bundle "{finish}"' if own else "")
            + "".join(f' --{m} "{v}"' for m, v in own))
    # WHOSE FAILURE THIS IS: the kind's step's, where it runs one that
    # decides the point -- before the engine, or after a clean one.
    decides = bool(own) and not failed and (point.step("before") if not ended
                                            else point.step("after"))
    pad = "    " if failed else ""
    head = (f"{pad}# --- The run's report step: {words} "
            f"(execution/runs-of-points.md 3) ---\n"
            f"{pad}# This point's {'end, exit and verdict' if ended else 'start'}"
            f" into the run's index"
            + (f", its own step{'s' if len(own) > 1 else ''} "
               f"({', '.join(f'{finish} {v}' for _m, v in own)}) run in "
               f"{'their' if len(own) > 1 else 'its'} place" if own else "")
            + ", and the run's message it calls for.\n"
            f"{pad}# THE REPORT NEVER FAILS THIS POINT: what it cannot do is "
            f"logged here.\n")
    if not decides:
        return head + (
            f"{pad}if [ -f {MONITOR_BUNDLE} ]; then\n"
            f"{pad}    {call} \\\n"
            f'{pad}        || _log WARN "report step: {words} exited $? -- '
            f"the run's index or its message may lack this point\"\n"
            f"{pad}else\n"
            f'{pad}    _log WARN "report step: {MONITOR_BUNDLE} is not beside '
            f'the deck -- {words} not run"\n'
            f"{pad}fi\n")
    step = point.step("after" if ended else "before")
    what = f"{finish} {step}"
    return head + (
        "# ITS OWN STEP'S FAILURE IS THE POINT'S"
        + (": the engine is not started, and the step after it is told how "
           "the point ended.\n" if not ended else
           ": the conclusion names it.\n")
        + "set +e\n"
        f"if [ -f {MONITOR_BUNDLE} ]; then\n"
        f"    {call}\n"
        f"    _mb_point_rc=$?\n"
        f"else\n"
        f'    _log ERROR "report step: {MONITOR_BUNDLE} is not beside the deck '
        f'-- {words} not run, nor this point\'s own step ({what})"\n'
        f"    _mb_point_rc=1\n"
        f"fi\n"
        "set -e\n"
        f'if [ "$_mb_point_rc" -ne 0 ]; then\n'
        f'    echo "===== this point\'s own step ({what}) exited with code '
        f'$_mb_point_rc'
        + (": the engine was not started" if not ended else "")
        + ' -- see $_runwrap_log =====" >&2\n'
        + ('    _engine_exit="$_mb_point_rc"\n'
           + _point_block(names, POINT_ENDED, point, finish, script_name,
                          failed=True)
           if not ended else "")
        + "    " + conclude_line('"$_mb_point_rc"',
                                  f"{FINISH_FAILED} ({what})") + "\n"
        '    exit "$_mb_point_rc"\n'
        "fi\n")


def _finish_check_block(finish: Optional[str]) -> str:
    """Before the engine starts: can the job's finish run here?  Asked with
    the python that will run it (``$_mb_py``, the activated env's, checked
    once the arguments are read, `running-a-job.md` § 2.2a), so a job
    env without what the finish imports stops before the expensive run it
    would otherwise throw away -- and SAYS SO in the conclusion, which reads
    failed (`engines/vibration.md` § 5.5, `running-a-job.md` § 4.2).  A dry
    run has already exited.  Empty when the job has no finish."""
    if finish is None:
        return ""
    from .parse.dirs.job import FINISH_CANNOT_LOAD
    return (
        f"# --- Can the job finish itself? (engines/vibration.md 5.5) -------\n"
        f"# The bundle's `loads` verb imports the finish on the job's own\n"
        f"# python and answers; its words are the load error itself.\n"
        f'_mb_fin_said=""\n'
        f'if [ ! -f {finish} ] '
        f'|| ! _mb_fin_said=$("$_mb_py" {finish} loads 2>&1); then\n'
        f'    _log ERROR "finish: {finish} cannot run on the job\'s python ($_mb_py) -- '
        f'${{_mb_fin_said:-{finish} is not beside the deck}}"\n'
        f'    echo "ERROR: this job cannot finish itself: {finish} does not '
        f'load on the job\'s python ($_mb_py) -- the engine '
        f'was not started, so no run is lost.  Its imports are the standard '
        f'library, numpy and netCDF4, which the job env carries '
        f'(envs/recipes.py)." >&2\n'
        f'    [ -n "$_mb_fin_said" ] && echo "$_mb_fin_said" >&2\n'
        + "    " + conclude_line(1, f"{FINISH_CANNOT_LOAD} ({finish})")
        + "\n"
        "    exit 1\n"
        "fi\n"
    )


#: A count refused unless it is a positive integer -- one spelling for every
#: count a run script reads: ``_mb_positive NAME VALUE`` for an argument,
#: read before anything is written; ``_mb_positive NAME VALUE SOURCE`` for
#: the count resolved once the start has begun, which names where the value
#: came from and concludes the run (a dry run concludes nothing).
_POSITIVE_FUNCTION = (
    "_mb_positive() {\n"
    '    if ! printf %s "$2" | grep -qE \'^[1-9][0-9]*$\'; then\n'
    '        echo "ERROR: $1 must be a positive integer; got: \'$2\'${3:+ '
    '(from $3)}" >&2\n'
    '        if [ -n "${3:-}" ] && [ "$_dry_run" = "0" ]; then\n'
    "            " + conclude_line(
        1, "a count that is not a positive integer (%s)", '"$1"') + "\n"
    "        fi\n"
    "        exit 1\n"
    "    fi\n"
    "}\n")


def _one_start_usage(trial: bool) -> str:
    """The run script's usage says what a run is -- one start; a trial's,
    that it measures its point once (`project-layout.md` § 1.6.1)."""
    return ("Starts once in its run's folder: a run is one start, and to\n"
            "continue it is to launch the stage again.\n\n" if not trial else
            "Starts once in its run's folder: a run is one start, and a\n"
            "trial measures its point once.\n\n")


def _args_parse(*, stated: str, omp_aliases: str, usage: str,
                inits: str = "", arms: str = "") -> str:
    """THE RUN SCRIPT'S ARGUMENTS, read once, before anything is written --
    one parser for every engine (`job-contracts.md` § 2.6, *Arguments*): a
    help request answers and exits, an argument that does not read is
    refused, and a dry run is known before the one-start guard.  ``stated``
    the counts baked at prep; ``omp_aliases`` the thread flag's spellings;
    ``inits`` and ``arms`` an engine's own flags; ``usage`` the help's
    text."""
    return (
        "# --- Arguments: read once, before anything is written ---\n"
        "# A help request answers and exits, an argument that does not\n"
        "# read is refused, and a dry run is known before the one-start\n"
        "# guard -- none of them writes a file (job-contracts.md 2.6).\n"
        + stated + _POSITIVE_FUNCTION
        + '_np_flag=""\n'
        '_omp_flag=""\n'
        + inits
        + '_dry_run=0\n'
        # THE ARGUMENTS AS THE LAUNCH GAVE THEM, kept for the log before
        # the loop below consumes them.
        '_mb_argv="$*"\n'
        'while [ $# -gt 0 ]; do\n'
        '    case "$1" in\n'
        "        -np|--np)\n"
        '            _mb_positive -np "${2:-}"\n'
        '            _np_flag="$2"; shift 2 ;;\n'
        f"        {omp_aliases})\n"
        '            _mb_positive -omp "${2:-}"\n'
        '            _omp_flag="$2"; shift 2 ;;\n'
        "        --dry-run|--dryrun)\n"
        '            _dry_run=1; shift ;;\n'
        + arms
        + "        -h|--help)\n"
        "            cat <<USAGE\n"
        + usage
        + "USAGE\n"
        "            exit 0 ;;\n"
        "        *)\n"
        '            echo "ERROR: unknown argument: $1 (use -h)" >&2\n'
        "            exit 1 ;;\n"
        "    esac\n"
        "done\n"
        "\n")


def _again_line(again: str, said: str = "what a launch again will do") -> str:
    """The run script's line after a failure: what a launch again will do
    (`running-a-job.md` § 3.5), ``said`` the cause where it is a stop at the
    wall (§ 3.6)."""
    return f'echo "{said}: {again}" >&2\n'


#: HOW LONG THE EXIT TRAP WAITS FOR ITS MONITOR'S CLOSING LINES, in seconds:
#: a monitor not gone by then is left to its own watch-pid exit.  A launch
#: waiting on a run here it was stopped over waits past it
#: (`submit._ended_within`).
MONITOR_STOP_S = 10

#: The exit trap's conclusion, for a failure of the script's own that
#: reached no marker (`project-layout.md` § 1.6.3): the script's exit code,
#: read as the trap began.
_STOPPED_BEFORE_ITS_END = conclude_line('"$_mb_rc"', STOPPED)

#: THE SIGNALS THAT STOP A RUN, each caught by its one handler
#: (``_mb_on_signal``), so none reaches the exit trap as an end of the
#: script's own: a scheduler's stop, Ctrl-C, a lost terminal, a limit
#: reached.  A pipe its log's tee left and a warning (``USR1``, ``USR2``)
#: are caught apart -- neither stops the run (:data:`_SIGNALS_APART`).
_STOPPING_SIGNALS = ("HUP", "INT", "QUIT", "ALRM", "TERM", "XCPU", "XFSZ",
                     "VTALRM", "PROF")

#: ...AND THE TWO CAUGHT APART.  A PIPE its log's tee left -- the tee gone
#: with the process group a stop signals -- marks the run stopped by a
#: signal and is ignored from then on, its write failing instead of ending
#: the script, so the stop that took the tee is the one said; a USR1 or
#: USR2 -- a warning a scheduler may be asked to send -- is said in the
#: session log, and the run goes on.
_SIGNALS_APART = (
    "trap '_mb_signalled=1; trap \"\" PIPE' PIPE\n"
    "for _mb_s in USR1 USR2; do\n"
    "    trap \"_log WARN 'caught SIG$_mb_s -- a warning, not a stop; the "
    "run goes on' || true\" \"$_mb_s\"\n"
    "done\n")


def _one_start_block(names, trial: bool) -> str:
    """Bash that refuses a second start in a run folder, first of all,
    before anything is written (`project-layout.md` § 1.6.1): a run is one
    start, and its session log is that start's -- a folder that holds one
    refuses this start, naming `jobset launch`, which opens the next run.
    A help request has answered before it, and a dry run starts nothing
    (``_dry_run``, read with the arguments): it passes, and writes no
    session log (`job-contracts.md` § 2.6).  The log's name is set here,
    the one variable every later line writes and reads -- and the
    conclusion marker's (`project-layout.md` § 1.6.3), which the exit trap
    may write before any later block runs.  ``trial``: a benchmark trial,
    which measures its point once."""
    log = names.name(ROLE_SESSION_LOG)
    way = (_TRIAL_ONCE if trial else
           next_run_words(command_stage(names.stage)))
    return (
        "# --- One start per run (project-layout.md 1.6.1) --------\n"
        "# A run folder holds one start, and its session log says it has\n"
        "# one.  To continue a run is to launch it again, which opens the\n"
        "# next run folder.  ABSOLUTE, the log's name: every later line --\n"
        "# the ending read over it (_mb_ending) among them --\n"
        "# resolves it wherever it runs from.\n"
        f'_runwrap_log="$PWD/{log}"\n'
        f'_concluded="{names.name(ROLE_CONCLUDED)}"\n'
        f'if [ "$_dry_run" = "0" ] && [ -e "$_runwrap_log" ]; then\n'
        f'    echo "ERROR: {holds_a_start(log, way)}." >&2\n'
        "    exit 2\n"
        "fi\n"
    )


def _orbitals_per_rank_notice(n_orbitals: Optional[int]) -> str:
    """A NOTICE about occupancy, checked against the rank count that is
    actually used -- never a limit *(user ruling, 2026-09-03)*.

    **SIESTA distributes ORBITALS across ranks**, not atoms, so the number
    that says whether the ranks have anything to hold is

        n_orbitals / mpi_np

    and it wants to be greater than one.  At or below it, ranks are idle by
    arithmetic, and the user is told exactly that -- *"your CPUs are not going
    to be fully used"* -- with both numbers shown, so the claim is checkable
    rather than a verdict.

    **The orbital count is an ESTIMATE and says so**: the deck's own, its
    BENCH-MARKS block's ``n_orbitals_est`` (`job-contracts.md` § 3.3), read
    by :func:`_n_orbitals_est` -- one estimate, one place, or the deck and the
    notice would disagree.  The true count needs the basis of every species
    and is known only once the engine starts.

    Empty when the deck states none: a notice needs a number, and inventing
    one is what this function exists to stop.
    """
    if n_orbitals is None:
        return ""
    return (
        f'# Occupancy notice: orbitals over ranks (running-a-job.md 3.1a).\n'
        f'_norb_est={int(n_orbitals)}   # the deck\'s own estimate '
        f'(BENCH-MARKS n_orbitals_est)\n'
        f'if [ "$_mpi_np" -ge "$_norb_est" ]; then\n'
        f'    echo "molbuilder: NOTE -- $_mpi_np ranks for ~$_norb_est '
        f'orbitals is <= 1 orbital per rank." >&2\n'
        f'    echo "  Your CPUs are not going to be fully used.  '
        f'This is a notice, not a limit -- the run proceeds." >&2\n'
        f'fi\n'
    )


def _runtime_status_block(
    *,
    held: str,
    label: str,
    warm: Sequence[str],
    resumes: bool = True,
    reads_prior: Optional[bool] = None,
) -> str:
    """Bash that reads what this run starts from and the atoms its deck
    holds, for the status banner -- the same for every engine.

    ``warm``: the restart files the deck reads, under the label the engine
    names them on (`run-identity.md` § 3.2) -- ``label`` is ``task.label``,
    baked, never re-read (`execution/gpu.md` G7).  ``reads_prior``: the
    deck's own answer about prior state, where its keywords state one
    (:func:`_deck_restart`) -- ``False``, and the Mode line says the deck
    reads none of what lies here; ``None`` where the deck states none: a
    script deck reads what launch copies in, and launch copies nothing into
    a stage that takes nothing.

    Prints, at run time, BEFORE the engine launches:

      * **Mode** -- what the run starts from, one of `job-contracts.md`
        § 4.4's lines: a clean start (no restart file here); the restart
        files launch copied in, which the engine loads; restart files the
        deck does not read; restart files on a kind that does not resume.

      * **Held atoms** -- ``held``, the words :func:`_held_words` composed
        at prep from the deck's own record of the atoms it holds: one reader
        for every engine's deck, never a pattern per engine typed in bash.

    Silently warm-restarting from contaminated restart files would
    re-contaminate every downstream run, so users have to be able to see
    at a glance where the SCF / geometry will start from and which atoms
    are held.
    """
    # THE RESTART FILES LYING HERE, from the restart list in effect
    # (``warm``), under the label the engine names them on
    # (`run-identity.md` § 3.2) -- so the Mode line names what is there,
    # never the whole list (a run whose only warm file
    # is ``<JOB>_optimized.xyz`` does not announce a clean start).  BRACED:
    # a suffix may start with "_" (`_optimized.xyz`), and an unbraced
    # "$_warm_label_optimized.xyz" is one unbound variable under set -u.
    warm_names = " ".join(f'"${{_warm_label}}{tail(suf, roles=tuple(warm))}"'
                          for suf in warm)
    return (
        "# --- Runtime status banner --------------------------\n"
        "# Reports what the run starts from -- the restart files here, read\n"
        "# at run time -- and the atoms its deck holds, stated at prep, so\n"
        "# the user can see what is about to happen BEFORE the engine starts\n"
        "# (job-contracts.md 4.4).\n"
        f'_warm_label="{label}"\n'
        + '# WARM STATE IS CONTENT, not mere existence: a zero-byte\n'
        + '# restart file is nothing to resume from, and a launch that\n'
        + '# announced "CONTINUES (engine will load ...)" over one\n'
        + '# would be telling the person something false.\n'
        + '_mb_has_state() { [ -s "$1" ]; }\n'
        + '_warm_found=""\n'
        + (f'for _f in {warm_names}; do\n'
           f'    if _mb_has_state "$_f"; then '
           f'_warm_found="${{_warm_found:+$_warm_found }}$_f"; fi\n'
           f'done\n' if warm else "")
        + f'_mode="{_CLEAN_START_MODE}"\n'
        'if [ -n "$_warm_found" ]; then\n'
        + (f'    _mode="{_FIRST_STEP_MODE}"\n' if not resumes else
           f'    _mode="{_CLEAN_DECK_MODE}"\n' if reads_prior is False else
           f'    _mode="{_CONTINUES_MODE}"\n')
        + "fi\n"
        f'_held="{held}"\n'
        f"\n"
    )


def _held_words(deck_text: Optional[str]) -> str:
    """THE ATOMS THE DECK HOLDS, in words for the run script's banner -- as
    the deck's own record states them: its ATOM-METADATA block, read by its
    one reader (`deck_record.extract_atom_metadata`), the held atoms asked of
    the sidecar's one door (`sidecars.molstruct.frozen_atoms`).  Every
    engine's deck carries the block (`job-contracts.md` § 3.4), and it is
    written at prep: read here, at prep, from the deck the run script ships
    beside.  A record that does not read raises `DeckRecordUnreadable`, which
    the caller refuses."""
    from .deck_record import extract_atom_metadata
    from .sidecars.molstruct import frozen_atoms
    meta = extract_atom_metadata(deck_text or "")
    if meta is None:
        return "not stated -- the deck carries no atom record"
    held = frozen_atoms(meta)
    return (f"{len(held)} atom{'' if len(held) == 1 else 's'} held (the "
            f"deck's atom record)" if held else "none held -- every atom free")


def _bash_numa_from_gpu(gpu_expr: str, out_var: str,
                        bus_var: str = "_bus", indent: str = "") -> str:
    """Emit bash that resolves the NUMA node of the GPU at index
    ``gpu_expr`` into ``out_var`` (``-1`` when unknown).

    Single source for the sysfs lookup so the per-rank launcher (running-a-job.md § 3.3)
    and the ``--dry-run`` placement report don't each carry their own
    copy of the subtle bit: ``nvidia-smi`` prints an **8-hex-digit** PCI
    domain (``00000000:02:00.0``) but the kernel sysfs path uses **4**
    (``0000:02:00.0``), so we strip the leading 4 chars with
    ``${bus:4}``.  Lowercased + space-stripped because sysfs is
    lowercase.  Any read failure leaves ``out_var=-1``.  ``indent`` is a
    leading-whitespace string applied to every line so the snippet lands
    aligned inside an indented block (e.g. the dry-run loop).
    """
    return (
        f'{indent}{bus_var}=$(nvidia-smi --id={gpu_expr} '
        f'--query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null '
        f"| tr 'A-Z' 'a-z' | tr -d ' ' || true)\n"
        f'{indent}{out_var}=-1\n'
        f'{indent}[ -n "${bus_var}" ] && {out_var}=$(cat '
        f'/sys/bus/pci/devices/${{{bus_var}:4}}/numa_node 2>/dev/null '
        f'|| echo -1)\n'
    )


def _gpu_loadbalance_block() -> str:
    """Bash that derives the rank<->GPU load balance at RUN time.

    GOAL: count the GPUs the job was given (``$_ngpu``, which the MPS
    decision reads) and turn the resolved rank count into a ranks-per-GPU
    figure, which the banner and the dry run report; the per-rank launcher
    block-distributes the ranks over the GPUs itself.

    READS  ``$_mpi_np``  (rank count; set by the args block).
    SETS   ``$_ngpu``           -- visible GPU count (0 if none),
           ``$_ranks_per_gpu``  -- ``mpi_np / ngpu`` (clamped >= 1).
    EMITS  one stderr line ``molbuilder: GPU load-balance -- N ranks over
           G GPU(s) = K rank(s)/GPU`` -- the human-checkable confirmation
           that K matches the allocation (validation pinpoint for the
           benchmark sweep, generator.md § 4.3a).
    WHY: K is the knob the benchmark sizes; it must come from the ACTUAL
    runtime allocation (the scheduler may grant a different GPU count than
    generation assumed).  1 GPU degenerates to all-ranks-on-GPU0 (the
    existing single-GPU behavior) with no code fork.

    ORDERING: emit AFTER ``$_mpi_np`` resolves and BEFORE the MPS block
    (which reads ``$_ngpu``).  running-a-job.md § 3.3
    """
    return (
        "# --- GPU load-balance: rank <-> GPU matching "
        "(running-a-job.md § 3.3) ---\n"
        "# Count ALLOCATED GPUs now and split ranks across them.  The\n"
        "# per-rank launcher maps rank -> GPU as\n"
        "# local_rank*ngpu/localsize, so K=mpi_np/ngpu ranks share\n"
        "# each GPU via MPS.  1 GPU -> all ranks on GPU0 (the existing\n"
        "# behavior); N GPUs -> block-distributed, no code fork.\n"
        "# Prefer CUDA_VISIBLE_DEVICES (the scheduler's allocated set) over\n"
        "# nvidia-smi -L (which over-counts on a shared node without\n"
        "# device-cgroup isolation) -- must agree with the per-rank\n"
        "# launcher's own count.\n"
        'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then\n'
        '    _ngpu=$(printf %s "$CUDA_VISIBLE_DEVICES" '
        '| tr , "\\n" | grep -c . || true)\n'
        'else\n'
        '    _ngpu=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)\n'
        'fi\n'
        'case "$_ngpu" in ""|*[!0-9]*) _ngpu=0 ;; esac\n'
        'if [ "$_ngpu" -ge 1 ]; then\n'
        '    _ranks_per_gpu=$(( _mpi_np / _ngpu ))\n'
        '    [ "$_ranks_per_gpu" -lt 1 ] && _ranks_per_gpu=1\n'
        'else\n'
        '    _ranks_per_gpu=$_mpi_np\n'
        'fi\n'
        'echo "molbuilder: GPU load-balance -- $_mpi_np ranks over '
        '$_ngpu GPU(s) = $_ranks_per_gpu rank(s)/GPU '
        '(MPS when ranks exceed GPUs)" >&2\n'
        "\n"
    )


def _gpu_socket_affinity_block() -> str:
    """Bash (rank-time, inside the quoted heredoc) implementing the
    GPU↔CPU socket co-location framework fix (running-a-job.md § 3.3).

    Each rank already knows its GPU's NUMA node (``$_numa``).  This block
    resolves that NUMA node's **socket** (``physical_package_id``) and, by
    walking the rank's own ``Cpus_allowed_list``, the set of sockets it
    owns cores on AND the GPU-socket NUMA nodes it actually owns
    (``$_pin_numas``), then:

      * **pins** (``numactl --cpunodebind/--membind`` to the **owned**
        GPU-socket NUMA nodes only) when the cpuset spans MORE than one
        socket -- the whole-node / ``--exclusive`` (or partial multi-
        socket) case where a rank would otherwise roam cross-socket;
      * **no-ops** when the cpuset is already confined to the GPU's socket
        (the scheduler co-located us -- nothing to do);
      * **warns, never fails** when we own NO cores on the GPU's socket
        (a shared cross-socket allocation we cannot fix from here).

    Binding only to OWNED NUMA nodes is the B-1 fix (audit 2026-06-27): a
    partial multi-NUMA allocation on the 8-NUMA Sol GPU nodes must never
    ``--membind`` to a socket-mate NUMA node it wasn't allocated.

    Sets ``$_pin`` (empty or the ``numactl`` prefix) for the ``exec``.
    All sysfs reads are guarded; missing topology/``numactl`` -> no pin.
    """
    return (
        '# --- GPU<->CPU socket co-location (running-a-job.md § 3.3) ---\n'
        '# Disable the whole pin (trust the scheduler) with '
        'MB_NO_SOCKET_PIN=1 -- useful to A/B whether the pin actually '
        'helps on this machine.\n'
        '_pin=""\n'
        'if [ "${MB_NO_SOCKET_PIN:-0}" = "1" ]; then\n'
        '    echo "molbuilder[rank ${_lr}/${_ls}]: socket co-location '
        'DISABLED (MB_NO_SOCKET_PIN=1) -- trusting scheduler placement" '
        '>&2\n'
        'else\n'
        '_gpu_sock=""\n'
        'if [ "${_numa:--1}" -ge 0 ] && '
        '[ -r "/sys/devices/system/node/node${_numa}/cpulist" ]; then\n'
        '    _gcpu=$(sed "s/[,-].*//" '
        '"/sys/devices/system/node/node${_numa}/cpulist")\n'
        '    _gpu_sock=$(cat '
        '"/sys/devices/system/cpu/cpu${_gcpu}/topology/physical_package_id" '
        '2>/dev/null)\n'
        'fi\n'
        '# Walk EVERY cpu this rank owns: collect the sockets we span and\n'
        '# the GPU-socket NUMA nodes we actually own -- a contiguous "0-47"\n'
        '# owns every NUMA node of both sockets, not its two endpoints\'.\n'
        '_my_socks=""; _pin_numas=""\n'
        '_cl=$(awk "/Cpus_allowed_list/{print \\$2}" /proc/self/status '
        '2>/dev/null)\n'
        '_oldifs=$IFS; IFS=,\n'
        'for _rng in $_cl; do\n'
        '    _a=${_rng%%-*}; _b=${_rng##*-}\n'
        '    case "$_a$_b" in ""|*[!0-9]*) continue ;; esac\n'
        '    for ((_e=_a; _e<=_b; _e++)); do\n'
        '        _ps=""\n'
        '        read -r _ps 2>/dev/null '
        '< "/sys/devices/system/cpu/cpu${_e}/topology/physical_package_id"\n'
        '        [ -n "$_ps" ] || continue\n'
        '        case ",$_my_socks," in *",$_ps,"*) : ;; '
        '*) _my_socks="${_my_socks:+$_my_socks,}$_ps" ;; esac\n'
        '        # If this owned cpu is on the GPU\'s socket, record its\n'
        '        # NUMA node (the node* symlink under the cpu dir).\n'
        '        if [ -n "$_gpu_sock" ] && [ "$_ps" = "$_gpu_sock" ]; then\n'
        '            for _nd in '
        '/sys/devices/system/cpu/cpu${_e}/node[0-9]*; do\n'
        '                _nn=${_nd##*node}\n'
        '                case "$_nn" in ""|*[!0-9]*) continue ;; esac\n'
        '                case ",$_pin_numas," in *",$_nn,"*) : ;; '
        '*) _pin_numas="${_pin_numas:+$_pin_numas,}$_nn" ;; esac\n'
        '                break\n'
        '            done\n'
        '        fi\n'
        '    done\n'
        'done\n'
        'IFS=$_oldifs\n'
        'if [ -n "$_gpu_sock" ] && [ -n "$_my_socks" ]; then\n'
        '    case ",$_my_socks," in\n'
        '      *",$_gpu_sock,"*)\n'
        '        case "$_my_socks" in\n'
        '          *,*)\n'
        '            if command -v numactl >/dev/null 2>&1 && '
        '[ -n "$_pin_numas" ]; then\n'
        '                _pin="numactl --cpunodebind=$_pin_numas '
        '--membind=$_pin_numas"\n'
        '                echo "molbuilder[rank ${_lr}/${_ls}]: socket-pin '
        '-> GPU socket $_gpu_sock, owned numa $_pin_numas via numactl '
        '(was multi-socket)" >&2\n'
        '            fi\n'
        '            ;;\n'
        '        esac\n'
        '        ;;\n'
        '      *)\n'
        '        echo "molbuilder[rank ${_lr}/${_ls}]: WARN cross-socket '
        '-- GPU numa=$_numa is on socket $_gpu_sock but this rank owns '
        'cores only on socket(s) $_my_socks; host<->device traffic and '
        'its OpenMP threads pay a remote hop; the run goes on '
        '(running-a-job.md § 3.3)." >&2\n'
        '        ;;\n'
        '    esac\n'
        'fi\n'
        'fi\n'                                    # close MB_NO_SOCKET_PIN else
    )


def _gpu_per_rank_launcher_block(program: str) -> str:
    """Bash that writes the per-rank GPU launcher + picks the CPU-bind
    policy.  Emit ONLY where ranks are placed on GPUs; each rank starts
    ``program``, the job's own (`Job.program`, else its engine's).

    GOAL: give every MPI rank its own GPU and (under a scheduler) defer CPU/mem
    placement to the scheduler.  Implements the general load-balance
    model -- 1 GPU degenerates to "all ranks on GPU0", N GPUs
    block-distribute -- with no code fork on GPU count (running-a-job.md § 3.3).

    WRITES a runtime helper ``.mb-rank-launch-$$.sh`` (``trap``-removed on
    EXIT) in which each rank computes ``gpu = local_rank*ngpu/localsize``,
    resolves that GPU's NUMA node, sets ``CUDA_VISIBLE_DEVICES``, logs its
    GPU+NUMA+cpuset, then ``exec``s the program.  The heredoc is QUOTED
    (``'HELPEREOF'``) so the rank-time variables resolve when the rank
    runs, not when the wrapper writes the file.
    SETS  ``$_program_target="bash $_rank_helper"`` (the launch target the
          ``_launch_cmd`` assembly interpolates in place of the bare
          program).
    UNDER A SCHEDULER (``$MB_JOB_ID`` set): clears ``$_numa_wrap_gpu`` and
          ``$_mpirun_bind`` so we do NOT double-bind against the scheduler's
          cgroup cpuset (running-a-job.md § 3.3; the benchmark logs the
          actual cpuset to confirm the scheduler bound near the GPU).
    WHY a helper FILE (not ``bash -c``): the per-rank logic can't survive
    as a word-split ``_launch_cmd`` string -- the ``bash -c`` quoting
    breaks under the later unquoted expansion -- so a tiny temp script is
    the robust idiom.
    """
    return (
        f'_rank_helper="{_bash_fill(RANK_LAUNCHER)}"\n'
        "cat > \"$_rank_helper\" <<'HELPEREOF'\n"
        '#!/bin/bash\n'
        '_lr=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMIX_RANK:-0}}\n'
        '_ls=${OMPI_COMM_WORLD_LOCAL_SIZE:-${MB_NTASKS:-1}}\n'
        '[ "${_ls:-0}" -lt 1 ] && _ls=1\n'
        '# Allocated GPU list: prefer CUDA_VISIBLE_DEVICES (a scheduler sets it\n'
        '# to the allocated set -- robust whether or not device-cgroups\n'
        '# isolate the node, and it carries the real physical indices);\n'
        '# fall back to all visible GPUs only when unset (local / no\n'
        '# scheduler).  nvidia-smi -L alone would over-count on a shared\n'
        '# node without cgroup isolation.\n'
        'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then\n'
        '    IFS=, read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"\n'
        'else\n'
        '    _n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)\n'
        '    case "$_n" in ""|*[!0-9]*) _n=1 ;; esac\n'
        '    [ "$_n" -lt 1 ] && _n=1\n'
        '    _gpus=(); _i=0\n'
        '    while [ "$_i" -lt "$_n" ]; do _gpus+=("$_i"); _i=$((_i+1)); done\n'
        'fi\n'
        '_ngpu=${#_gpus[@]}\n'
        '[ "$_ngpu" -lt 1 ] && { _gpus=(0); _ngpu=1; }\n'
        '_idx=$(( _lr * _ngpu / _ls ))\n'
        '_gpu=${_gpus[$_idx]}\n'
        # Resolve NUMA from the chosen GPU index BEFORE narrowing
        # CUDA_VISIBLE_DEVICES (keeps the sysfs lookup unambiguous).
        + _bash_numa_from_gpu("$_gpu", "_numa")
        + 'export CUDA_VISIBLE_DEVICES=$_gpu\n'
        "_cpus=$(awk '/Cpus_allowed_list/{print $2}' "
        '/proc/self/status 2>/dev/null)\n'
        'echo "molbuilder[rank ${_lr}/${_ls}]: '
        'CUDA_VISIBLE_DEVICES=$_gpu (numa=$_numa) '
        'cpus=${_cpus:-?}" >&2\n'
        + _gpu_socket_affinity_block()
        + f'exec $_pin {program} "$@"\n'
        'HELPEREOF\n'
        'chmod +x "$_rank_helper"\n'
        # Cleanup is handled by the single unified EXIT trap (_mb_cleanup,
        # set near the top): it rm's $_rank_helper.  A local ``trap ...
        # EXIT`` here would CLOBBER the MPS cleanup trap (bash keeps only
        # the last EXIT trap) -- hence the centralised function.
        '_program_target="bash $_rank_helper"\n'
        # Under a scheduler, its cgroup cpuset (the GPU binding + the cores
        # per rank) governs CPU/memory placement.  Do NOT also numactl-wrap
        # / --map-by -- that double-binds and can fight the scheduler.  P1
        # in running-a-job.md § 3.3
        'if [ -n "${MB_JOB_ID:-}" ]; then\n'
        '    _numa_wrap_gpu=""\n'
        '    _mpirun_bind=""\n'
        '    echo "molbuilder: under the scheduler (job $MB_JOB_ID) -- '
        'trusting scheduler cpuset for CPU/mem binding '
        '(no manual numactl)" >&2\n'
        # Off a scheduler with >1 GPU in play, the whole-job wrap would set
        # every rank\'s memory on GPU0\'s node, and its cpuset is not what
        # places the ranks (OpenMPI\'s map overrides it -- measured,
        # running-a-job.md § 3.3).  The per-rank helper pins each rank
        # beside ITS GPU whenever ranks span GPUs; the whole-job wrap
        # stays only for the single-GPU case.
        'elif [ "${_ngpu:-0}" -ge 2 ] && [ -n "$_numa_wrap_gpu" ]; then\n'
        '    _numa_wrap_gpu=""\n'
        '    echo "molbuilder: $_ngpu GPUs in play -- per-rank NUMA '
        'binding via the rank helper (the whole-job wrap would set '
        'every rank\'s memory on GPU0\'s node)" >&2\n'
        'fi\n'
    )


def _resolved_log_block(launch_line: str, gpu_ranks: bool) -> str:
    """Bash that records the fully-resolved launch command + placement
    into the session log -- ALWAYS, on every run.

    GOAL (user request 2026-06-26): the log must carry the exact command
    that ran and the GPU/NUMA context, so a post-mortem never has to
    guess what was launched.  These ``_log INFO`` lines ARE the audit
    trail that pins what the run did -- the validation anchor when a run
    is slow or wrong.

    ``launch_line`` is the command as the banner shows it; READS
    ``$_launch_note`` / ``$_mpi_np`` / ``$_omp_threads`` and, where ranks are
    placed on GPUs (``gpu_ranks``), ``$_ngpu`` / ``$_ranks_per_gpu`` /
    ``$_numa_wrap_gpu``.
    """
    return (
        f'# --- Record resolved launch command + placement (log) ---\n'
        f'_log INFO "resolved launch : {launch_line}"\n'
        f'_log INFO "launch mode     : $_launch_note"\n'
        f'_log INFO "{RANKS_OMP_LINE.format(ranks="$_mpi_np", threads="$_omp_threads")}"\n'
        + (
            '_log INFO "gpu placement   : ${_ngpu:-0} GPU(s), '
            '${_ranks_per_gpu:-?} rank(s)/GPU, '
            'numa-wrap=\'${_numa_wrap_gpu:-none}\'"\n'
            if gpu_ranks else ""
        )
    )


def _scheduler_context_block() -> str:
    """Bash that logs what the scheduler told this job, when set -- in
    molbuilder's own names (`kinds.JOB_VARIABLES`), which the job's header
    says once for everything it starts (`kinds.Kind.job_lines`); none is set
    for a run here.  The same lines on every machine: the run script is one
    file with or without a scheduler (`architecture.md` § 9)."""
    from .scheduler.kinds import JOB_VARIABLES
    return (
        "# What the scheduler told this job -- only logged if set.  Read for\n"
        "# diagnostic logging + launch tuning (running-a-job.md 2.1-2.2a);\n"
        "# they do NOT alter activation or preamble.\n"
        f"for _v in {' '.join(JOB_VARIABLES)}; do\n"
        '    _v_val="${!_v:-}"\n'
        '    [ -n "$_v_val" ] && _log INFO "$_v=$_v_val"\n'
        "done\n")


def _dry_run_block(launch_line: str, gpu_ranks: bool, words: str) -> str:
    """Bash implementing ``--dry-run``: print the resolved command, each
    count's source, the environment and -- where ranks are placed on GPUs
    (``gpu_ranks``) -- the per-rank GPU/NUMA mapping that WOULD be used, then
    ``exit 0`` WITHOUT launching the engine (``words``, as its record names
    it).

    GOAL (user request 2026-06-26): let a user send the header with
    ``--dry-run`` (or run it here) and read the log to confirm the
    generated command matches what the scheduler granted (``MB_NTASKS``,
    ``MB_CPUS_PER_TASK``, ``MB_JOB_GPUS``), WITHOUT spending a real run.
    This is the cheap pre-flight validation of the whole resolution
    chain.

    The GPU map loops ranks ``0..mpi_np-1`` and prints ``rank R -> GPU G
    (numa=N)`` -- the block-distribution the per-rank launcher will apply
    (divisor is ``$_mpi_np`` here since LOCAL_SIZE isn't set without mpirun;
    equal on single-node v1).  SIDE-EFFECT-FREE: the MPS daemon start is
    separately gated on ``_dry_run != 1`` so a dry run touches no GPU state.
    """
    return (
        # The emitted header names the block as `job-contracts.md` § 2.6
        # lists it.
        f"# --- Dry-run preview (--dry-run) ----------------------\n"
        f'if [ "$_dry_run" = 1 ]; then\n'
        f'    echo ""\n'
        f'    echo "===== molbuilder DRY RUN (no {words} launch) ====="\n'
        f'    echo "  Resolved cmd : {launch_line}"\n'
        f'    echo "  Launch mode  : $_launch_note"\n'
        f'    echo "  MPI ranks    : $_mpi_np   '
        f'(source: ${{_np_source:-?}})"\n'
        f'    echo "  OMP threads  : $_omp_threads   '
        f'(source: ${{_omp_source:-?}})"\n'
        f'    echo "  Env          : ${{CONDA_DEFAULT_ENV:-<none active>}}"\n'
        f'    echo "  Scheduler    : job=${{MB_JOB_ID:-<none>}} '
        f'ntasks=${{MB_NTASKS:-?}} cpus/task=${{MB_CPUS_PER_TASK:-?}} '
        f'gpus=${{MB_JOB_GPUS:-?}}"\n'
        + (
            '    echo "  Visible GPUs : ${_ngpu:-0}"\n'
            '    echo "  Ranks/GPU    : ${_ranks_per_gpu:-?} '
            '(MPS ${_use_mps_str:-?})"\n'
            '    echo "  Rank -> GPU mapping (block-distributed):"\n'
            '    _dr_ng=${_ngpu:-0}\n'
            '    if [ "$_dr_ng" -lt 1 ]; then\n'
            '        echo "      (no GPUs visible here -- on the '
            'compute node each rank maps as rank*ngpu/ntasks)"\n'
            '    else\n'
            '        _dr_r=0\n'
            '        while [ "$_dr_r" -lt "$_mpi_np" ]; do\n'
            '            _dr_g=$(( _dr_r * _dr_ng / _mpi_np ))\n'
            + _bash_numa_from_gpu("$_dr_g", "_dr_numa",
                                  bus_var="_dr_bus", indent="            ")
            + '            echo "      rank $_dr_r -> GPU $_dr_g '
            '(numa=$_dr_numa)"\n'
            '            _dr_r=$(( _dr_r + 1 ))\n'
            '        done\n'
            '    fi\n'
            if gpu_ranks else ""
        )
        + '    echo "================================================"\n'
        '    _log INFO "dry-run complete; nothing launched"\n'
        '    exit 0\n'
        'fi\n'
        "\n"
    )


def _scf_tee_func(row_ere: str, prefix_ere: str) -> str:
    """Bash defining ``_mb_scf_tee`` -- the SCF per-iteration timing
    instrument (running-a-job.md § 4.1), for an output the file catalogue
    cuts a timing file from (`runfiles.cut_from_output`):
    ``row_ere`` a row, ``prefix_ere`` the prefix before its iteration
    number, the grammar's own patterns (`scf_timing_rows.scf_row_patterns`).

    GOAL: an engine that prints **no usable per-iteration wall time** --
    SIESTA's ``scf:`` lines carry energies + dDmax but no time.  This filter
    IS the benchmark's measurement instrument: it tees the engine's stdout
    to the output AND timestamps every SCF row into a per-run
    ``.scf-timing.log`` as ``<epoch.ns> <iter#> <full row>``, so
    per-iteration wall time = consecutive-stamp delta (first delta dropped,
    rest averaged -- one clean sample under the capped 3-iteration bench
    trial; running-a-job.md § 4.1).

    WHY this shape: a single ``awk`` process copies every line at C speed
    (``fflush(out)`` keeps the output live for ``tail -f``); ``date
    +%s.%N`` is spawned only on the infrequent SCF rows.  Portable across
    gawk/mawk (getline-from-command + close + fflush).  The caller pipes the
    launch through ``_mb_scf_tee "$_out_file" "$_scf_timing_log"`` and reads
    ``${PIPESTATUS[0]}`` for the engine's exit (awk never masks it).
    """
    return (
        "# --- SCF per-iteration timing instrument "
        "(running-a-job.md § 4.1) ---\n"
        "# Tees the engine's stdout to its output AND stamps each SCF row\n"
        "# into the per-run .scf-timing.log, so per-iter wall time =\n"
        "# consecutive-epoch delta (the engine prints no per-iter time).\n"
        "_mb_scf_tee() {\n"
        "    awk -v out=\"$1\" -v tlog=\"$2\" '\n"
        "        { print > out; fflush(out) }\n"
        "        /" + row_ere + "/ {\n"
        "            _cmd=\"date +%s.%N\"; _cmd | getline _ts; close(_cmd)\n"
        "            _l=$0; sub(/" + prefix_ere + "/, \"\", _l)\n"
        "            split(_l, _f, /[ \\t]+/)\n"
        "            print _ts, _f[1], $0 > tlog; fflush(tlog)\n"
        "        }\n"
        "    '\n"
        "}\n"
    )


def _ending_question_func(stderr_apart: bool) -> str:
    """Bash defining ``_mb_ending`` -- the run script's one door onto how the
    run ended (`execution/run-reports.md` § 2.3).

    THE RUN SCRIPT ASKS, IT DOES NOT GREP.  ``_run_ending.py`` travels beside
    every job (:func:`monitor_companions`), so the run script runs it with
    the job's own python over the output -- read by the reader its role
    names -- and, where the output keeps its engine's stderr apart
    (``stderr_apart``, `runfiles.stderr_apart`: SIESTA's), over the
    session log that holds it; the markers keep their one home, each engine
    family's table.

    ``_mb_ending QUESTION`` answers by exit status (0 yes, 1 no, 2 the
    ending cannot be read); ``_mb_ending advice`` prints the reader's advice
    for the cause, exit 0 where it gives some; ``_mb_ending`` alone prints
    the ending in words.  With the bundle missing, not loading on the job's
    python, or not reading the output, it answers 2 -- no hint.

    ``_mb_ending_able`` ASKS ONCE whether the ending can be read here -- the
    ending in words, read by the bundle on the job's python -- and remembers
    the answer and the words (``_mb_ending_words``), so a bundle that cannot
    load, or a reading that stops, prints its reason once and the one place
    that asks first says the ending cannot be read (user: "ask the bundle
    once").  It is asked in THIS shell before any question: an assignment
    keeps what its ``$( )`` read, where an ask inside ``$( )`` alone could
    set nothing its caller sees.
    """
    stderr = ' --stderr "$_runwrap_log"' if stderr_apart else ""
    return (
        "# How the run ended, asked of the framework: `_run_ending`, in\n"
        f"# {MONITOR_BUNDLE} beside the job, reads the output"
        + (" and the engine's\n# stderr (this run's session log)" if stderr_apart
           else "\n#")
        + " with its engine family's own table (run-reports.md 2.3).\n"
        "# _mb_ending QUESTION: exit 0 yes, 1 no, 2 cannot read;\n"
        "# _mb_ending advice: the advice for the cause; alone: the ending.\n"
        "_mb_ending_able() {\n"
        '    if [ -z "${_mb_ending_state:-}" ]; then\n'
        f'        if [ -f {MONITOR_BUNDLE} ] && _mb_ending_words=$("$_mb_py" \\\n'
        f'               {MONITOR_BUNDLE} ending "$_out_file"{stderr}); then\n'
        "            _mb_ending_state=able\n"
        "        else\n"
        "            _mb_ending_state=unable\n"
        "        fi\n"
        "    fi\n"
        '    [ "$_mb_ending_state" = able ]\n'
        "}\n"
        "_mb_ending() {\n"
        "    _mb_ending_able || return 2\n"
        f'    "$_mb_py" {MONITOR_BUNDLE} ending "$_out_file"{stderr} "$@"\n'
        "}\n"
    )


def _gpu_runtime_block() -> str:
    """Bash for a GPU run's PLACEMENT: whether MPS is available, and the
    NUMA node the GPU is attached to.  It sets no rank or thread count.

    **The counts are the stated ones** *(`architecture.md` § 5.2)*.

    Outputs, consumed by the rest of the run script:

      * ``_have_mps`` / ``_use_mps_default``: NVIDIA's Multi-Process Service
        is on PATH, and whether the run may start it (``--mps`` /
        ``--no-mps`` and ``MOLBUILDER_USE_MPS`` say; unsaid, allowed).  The
        MPS block below decides from both and from ranks sharing a GPU.
      * ``_gpu_numa``: the NUMA node GPU 0 is attached to, resolved at run
        time on the machine the job runs on (:func:`_bash_numa_from_gpu`:
        nvidia-smi's bus id + the kernel's sysfs); overridable via
        ``MOLBUILDER_GPU_NUMA``.  An integer string, or ``"unknown"``.
      * ``_numa_wrap_gpu``: the ``numactl`` prefix that sets a single-GPU
        run's memory on its GPU's node, where the box has two sockets and
        numactl is there -- it does not hold the ranks there: OpenMPI's map
        overrides its cpuset (measured, `running-a-job.md` § 3.3).
    """
    return (
        "# --- GPU mode: placement (MPS, NUMA) -- the counts are stated ---\n"
        # ---- MPS availability ----
        # NVIDIA Multi-Process Service: a daemon that lets multiple CUDA
        # client processes share one GPU CONCURRENTLY via Hyper-Q instead of
        # serialising through the driver context.  The binary lives on the
        # HOST DRIVER side (nvidia-cuda-mps-control); not a conda package.
        '_have_mps=0\n'
        'if command -v nvidia-cuda-mps-control >/dev/null 2>&1; '
        'then _have_mps=1; fi\n'
        # WHETHER MPS IS ALLOWED: --mps / --no-mps win (the args block);
        # the env var second; allowed when nothing says.  Whether it STARTS
        # is the MPS block's one decision -- allowed, on PATH, and ranks
        # sharing a GPU.
        '_use_mps_default="${MOLBUILDER_USE_MPS:-1}"\n'
        '_have_mps_str="no"; '
        '[ "$_have_mps" = "1" ] && _have_mps_str="yes"\n'
        'echo "molbuilder: mps_available=$_have_mps_str" >&2\n'
        # ---- GPU NUMA proximity, RESOLVED AT RUN TIME ----
        # On the machine the job runs on, from GPU 0's PCI bus id and the
        # kernel's sysfs -- the same lookup the per-rank launcher makes
        # (`_bash_numa_from_gpu`).  A machine fact is measured where the job
        # runs, never on the machine that prepared it (`configuration.md`
        # § 5 M-1).  The runtime override
        # ``MOLBUILDER_GPU_NUMA=N`` wins; 0 is a valid node.
        + _bash_numa_from_gpu("0", "_gpu_numa_probed")
        + '_gpu_numa="${MOLBUILDER_GPU_NUMA:-$_gpu_numa_probed}"\n' +
        '# Defence in depth: whatever the lookup or the override gave,\n'
        '# use it as a NUMA target only if it parses as a non-negative\n'
        '# integer (the lookup says -1 for unknown).\n'
        'case "$_gpu_numa" in\n'
        '    ""|*[!0-9]*) _gpu_numa="unknown" ;;\n'
        'esac\n'
        # ---- numactl availability + NUMA-pin decision ----
        # mpirun is wrapped whole: its memory policy is inherited by every
        # rank OpenMPI forks, while the ranks' cores are OpenMPI's map's
        # (it overrides the wrap's cpuset).  Three conditions: dual-socket+
        # box, GPU NUMA known, numactl on PATH.
        '_numa_wrap_gpu=""\n'
        'if [ "$_gpu_numa" != "unknown" ] && '
        '[ "$_n_sockets" -ge 2 ] && '
        'command -v numactl >/dev/null 2>&1; then\n'
        '    _numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa '
        '--membind=$_gpu_numa"\n'
        'fi\n'
        "\n"
    )


def _wrapped(text: str, prefix: str, width: int = 62) -> str:
    """``text`` as lines of at most ``width`` columns, each begun with
    ``prefix`` and ended with a newline -- a sentence composed elsewhere
    (`identity.way_back_words`), set in this script's comments or its usage
    text."""
    import textwrap
    return "".join(f"{prefix}{line}\n" for line in textwrap.wrap(
        text, width=width - len(prefix), break_on_hyphens=False))


#: THE MODE LINES (`job-contracts.md` § 4.4): what a run starts from --
#: no restart file here; the restart files launch copied in, which the
#: engine loads (``$_warm_found`` names them); restart files the deck reads
#: none of -- its restart answer is ``clean`` (`run-identity.md` § 4); and
#: restart files on a kind that does not resume (`Job.resumes`).
_CLEAN_START_MODE = "CLEAN START (no restart file here)"
_CONTINUES_MODE = "CONTINUES (engine will load $_warm_found)"
_CLEAN_DECK_MODE = ("CLEAN (this deck reads no prior state; the files here "
                    "are not read)")
_FIRST_STEP_MODE = ("FROM THE FIRST STEP (this kind of run does not "
                    "resume)")


#: What a benchmark trial's run script says of a launch again: a trial
#: measures its point once (`submit._plan_trial`, `project-layout.md`
#: § 1.5a), so its benchmark's launch does not start it again.
_TRIAL_ONCE = ("a trial measures its point once, and its benchmark's launch "
               "does not start it again -- to measure it again, go back to "
               "the state saved before the benchmark's prep and prep anew")


def _launch_again(resumes: bool, continues: bool,
                  restart_honoured: Optional[bool],
                  relaunch: Optional[str] = None,
                  trial: bool = False) -> str:
    """WHAT A LAUNCH AGAIN WILL DO, said by the run script after a failure
    (`running-a-job.md` § 3.5) -- from the facts the launch itself decides
    by, which prep bakes in: ``continues`` (`Job.relaunch_continues`: the
    kind resumes and the stage takes something from a run -- the launch
    copies this run's restart files into the next run) and ``resumes``
    (`Job.resumes`, the kind's warm-files fact), with the deck's restart
    answer: a stage that continues goes on warm from the files this run
    saved; a deck that reads no prior state starts over from its own
    coordinates; a kind that does not resume, or a stage that carries
    nothing from one run to the next, starts over from its deck, nothing of
    this run copied in.  ``trial``: a benchmark trial, never launched
    again (:data:`_TRIAL_ONCE`).

    **A POINT OF A RUN OF POINTS** (``relaunch`` -- its declared relaunch,
    `execution/runs-of-points.md` § 1.2): said after its engine failed --
    the point *failed* (§ 1.3) -- a launch again takes the run's points
    done over and runs this one ``again`` from its start or ``continue``s
    it from its own last state."""
    back = "to change a setting, " + way_back_words(True)
    if trial:
        return _TRIAL_ONCE
    if relaunch is not None:
        return ("this point failed: a launch again takes the run's "
                "points done over and "
                + ("continues this one from its own last state"
                   if relaunch == CONTINUE else
                   "runs this one again from its start")
                + " -- " + back)
    if restart_honoured is False:
        return ("this deck reads no prior state, so a launch again starts "
                "over from its own coordinates -- " + back)
    if not continues:
        return ("a launch again starts this run over from its deck, nothing "
                "of it copied in -- "
                + ("this kind of run does not resume" if not resumes else
                   "this stage carries nothing from one run to the next")
                + "; " + back)
    return ("launch the stage again to continue warm from the files this run "
            "saved -- or, " + back)


def _deck_restart(engine: str, text: Optional[str]
                  ) -> Tuple[Optional[bool], Dict[str, Tuple[str, bool]]]:
    """What a deck whose settings are KEYWORDS (`engine_table`'s
    ``keyword_reader``) says of the state a previous run left -- read from
    its text by that reader, because the deck is what the engine obeys:
    ``(honoured, reads)``; ``(None, {})`` for a script deck.

    ``honoured``: its ``restart`` item's answer (the catalogue's
    ``expands``, `script_emit.parameter`), read by the one reader of a
    deck's restart group (`script_emit.restart_group`) -- ``None`` where the
    deck states none or cannot be read.  ``reads``: FILE BY FILE, ``{suffix: (keyword, reads)}``
    for each restart file whose deck keyword (`warm-files.toml`'s
    ``honoured_by``) the deck states.  One restart group may say yes to one
    file and no to another: a force-constant call reads the relaxed density,
    never a geometry (`siesta/vibration_deck.start_state_lines`).

    The run script ships beside exactly one deck, so what its texts say of
    restart files -- the header, the Mode line, what a launch again will do
    -- is what that deck says.  First match wins, as the engine's keyword
    reader reads its deck.
    """
    from .engine_table import keywords
    from .script_emit import restart_group
    from .warmfiles import warm_list
    group = restart_group(engine, text)
    if group is None:
        return None, {}
    key = keywords(engine).key
    said = {key(k): reads for k, reads in group.items()}
    # The item's answer is its first keyword's (`script_emit.parameter`).
    honoured = next(iter(group.values()), None)
    return honoured, {
        r.suffix: (r.honoured_by, said[key(r.honoured_by)])
        for r in warm_list(engine).rules
        if r.honoured_by and key(r.honoured_by) in said}


def _reads_words(words: str, reads: Dict[str, Tuple[str, bool]]) -> str:
    """What :func:`_deck_restart` found, as the run script says it -- the
    engine named in ``words``: a group that reads everything it names is the
    description's *start from: continue*; one that reads some and not others
    is a call that fixes its own start -- *sets DM.UseSaveDM .true., so
    SIESTA reads the .DM left under its label -- and MD.UseSaveXV .false.:
    never the .XV*; one that reads none starts from the deck's own
    coordinates."""
    yes = [(s, k) for s, (k, r) in reads.items() if r]
    no = [(s, k) for s, (k, r) in reads.items() if not r]
    if not yes:
        return ("sets " + " / ".join(k for _, k in no) + " .false., so "
                + words + " does NOT load the " + "/".join(s for s, _ in no)
                + " left under its label -- the run starts from the "
                "coordinates in the deck.")
    out = (("" if no else "says 'start from: continue' -- it ")
           + "sets " + " / ".join(k for _, k in yes) + " .true., so "
           + words + " reads the " + "/".join(s for s, _ in yes)
           + " left under its label")
    if no:
        out += (" -- and " + " / ".join(k for _, k in no) + " .false.: "
                "never the " + "/".join(s for s, _ in no))
    return out + "."


def _n_orbitals_est(text: Optional[str]) -> Optional[int]:
    """The deck's own orbital estimate -- its BENCH-MARKS block's
    ``n_orbitals_est`` (`job-contracts.md` § 3.3), read by the block's one
    reader (`deck_record.read_bench_marks_block`) -- or ``None``: a deck with
    no such block, or one that states none.  It feeds the occupancy NOTICE
    (:func:`_orbitals_per_rank_notice`) and clamps nothing (user ruling,
    2026-09-03)."""
    if not text:
        return None
    from .deck_record import read_bench_marks_block
    got = (read_bench_marks_block(text) or {}).get("n_orbitals_est")
    try:
        return int(got) if got is not None else None
    except (TypeError, ValueError):
        return None


def _effective_parameters_block(script_name: str, engine: str,
                                 calculation: str, stage: str,
                                 role: Optional[str] = None) -> str:
    """What the engine will be given, echoed into the run log at launch --
    for a deck whose settings are KEYWORDS the engine reads, by the keyword
    reader its record names (`engine_table`'s ``keyword_reader``); a script
    deck writes its own record from inside itself, where it can read its
    objects back.

    **The one block both engines write** (`model/parse.md` § 5d.3a): a row per
    catalogue item THIS RUNG READS -- of ``engine``, carried by a
    ``calculation`` and read by the rung its ``stage`` token names -- or,
    given ``role``, by the one role its point plays, a SIESTA vibration's
    phase (`script_emit.declarations`, by the one rule `template.reads`;
    the record's reader lists what this block names): a transmission deck lists
    TBtrans's settings, never a relaxation's --
    ``[item, default, asked, used]``,
    `script_emit`'s format and its reader -- then the deck as the engine
    will read it.  The engine is a separate process that has not started
    when this runs, so the rows carry the item's catalogue DEFAULT only:
    what the deck asks for is the deck itself, which the record reads from
    the attempt, and what the engine used is its own record of it (SIESTA's
    ``fdf.<stamp>.log``).  The default is baked here because it is a fact of
    the catalogue this run script was rendered from -- the run's own record
    of it, never today's catalogue read later.

    **The deck is read at LAUNCH** by the engine's keyword reader, beside
    the job (``mb_monitor.pyz deck READER DECK``: its ``read_deck``), each
    line the engine parses -- so a deck somebody hand-edited after `prep` is
    recorded as the engine will see it, and a block read from a file beside
    the deck (``%block Label < Filename``) is echoed with that file's rows.
    """
    import shlex as _shlex
    from . import script_emit as _sc
    from .deck_record import BLOCK_PARAMETERS, begin_marker, end_marker

    lines = [
        "",
        "# --- What the engine will read ------------------------------",
        f'echo "{begin_marker(BLOCK_PARAMETERS)}"',
        "echo " + _shlex.quote(_sc.PARAMETER_COLUMNS_LINE),
    ]
    from .engine_table import engine as _engine_rec
    reader = _engine_rec(engine).keyword_reader
    for item in _sc.declarations(engine=engine, calculation=calculation,
                                 stage=stage, role=role):
        param = _sc.parameter(item.name, engine)
        if param.writes:
            lines.append("echo " + _shlex.quote(
                _sc.parameter_row(item.name, param.default)))
    lines += [
        f'if [ -f {MONITOR_BUNDLE} ]; then',
        f'    "$_mb_py" {MONITOR_BUNDLE} deck "{reader}" "{script_name}" '
        f'2>>"$_runwrap_log" | sed "s/^/#   /" \\',
        f'        || _log WARN "what the engine will read: {script_name} did '
        f'not read -- the record holds none of its lines"',
        "else",
        f'    _log WARN "what the engine will read: {MONITOR_BUNDLE} is not '
        f'beside the deck -- the record holds none of its lines"',
        "fi",
        f'echo "{end_marker(BLOCK_PARAMETERS)}"',
        "",
    ]
    return "\n".join(lines) + "\n"


def _preamble_source_targets(chunks) -> List[str]:
    """Absolute paths the baked preamble will ``source``, in order.

    Only ABSOLUTE ones, and only from a bare ``source X`` / ``. X`` at the
    start of a line: those are the ones that can silently refer to a
    machine that is not this one.  A relative path or one built from a
    variable is the author's own business and is left alone -- guessing at
    it would put a check in front of a line this cannot actually read.

    Deliberately not a shell parser.  It reads the one form that caused a
    real failure (`source /abs/path/conda.sh`) and ignores everything
    else, because a wrong guard is worse than no guard: it would refuse a
    run that would have worked.
    """
    import re as _re
    out: List[str] = []
    pat = _re.compile(r"^\s*(?:source|\.)\s+(?P<q>[\"']?)(?P<path>/[^\s\"';|&]+)"
                      r"(?P=q)\s*(?:#.*)?$")
    for _scope, text in (chunks or []):
        for line in (text or "").splitlines():
            m = pat.match(line)
            if m:
                pth = m.group("path")
                if pth not in out:
                    out.append(pth)
    return out


#: The run script's stated launch counts, baked at prep -- the ranks and the
#: threads per rank (`running-a-job.md` § 3.1-3.2) -- by these names, which
#: :func:`stated_counts` reads back.
_STATED_COUNTS = ("_mpi_np_default", "_omp_threads_default")


def stated_counts(run_script_text: str) -> List[str]:
    """A rendered run script's stated counts, ``name=value`` as it carries
    them -- what a surface shows for what a job will be launched with on a
    machine with no scheduler (A13, `execution/architecture.md` § 5.2), read
    back by the writer's side and never worked out again."""
    return [ln.strip() for ln in run_script_text.splitlines()
            if ln.split("=", 1)[0].strip() in _STATED_COUNTS]


def _thread_chain(default: str) -> str:
    """The run script's thread count, as one shell expansion: the first of
    :data:`runtime_info.THREAD_SOURCES` set, else ``default`` -- the count
    stated at prep (`running-a-job.md` § 3.2)."""
    expr = default
    for var in reversed(THREAD_SOURCES):
        expr = f"${{{var}:-{expr}}}"
    return expr


#: How the ``--dry-run`` report names each rung of :data:`runtime_info.
#: THREAD_SOURCES` when the count came from it -- the report's own words,
#: as the rank count's are ("MB_NTASKS (the ranks the scheduler granted)").
#: A rung added to the list without words here stops the render by name
#: (KeyError).
_THREAD_SOURCE_WORDS = {
    "OMP_NUM_THREADS":  "OMP_NUM_THREADS env",
    "MB_CPUS_PER_TASK": "MB_CPUS_PER_TASK (the cores per rank the scheduler "
                        "granted)",
}


def env_entry(machine_record, env_name: str) -> Tuple[str, str]:
    """HOW A SHELL ENTERS ``env_name`` ON THE MACHINE A SCRIPT IS FOR -- the
    one rendering every generated script enters an environment by: a run
    script its engine's, a sequence of groups' walk molbuilder's own
    (`job-system.md` § 6.0, *The walk*) -- off THAT MACHINE'S RECORD
    (``machine_record.env_init``, `configuration.md` § 4; `running-a-job.md`
    § 5.2).  Returns ``(preamble, activation)``: the record's preamble,
    every absolute path it sources checked before it is sourced (``exit
    78``, naming it), and the record's activation of ``env_name``, the line
    to run.  The script defines ``_log`` (:data:`LOG_FUNCTION`).  Refused
    when no record states the activation: it has no default."""
    env_init = dict(getattr(machine_record, "env_init", None) or {})
    if not env_init.get("activation"):
        # REFUSED BY THE ONE DOOR that refuses it at prep, in its words
        # (`jobset.machine.require_activation`): the activation has no
        # default, and prep asked it before anything was written.
        from .jobset.errors import PrepError
        from .jobset.machine import require_activation
        try:
            require_activation(getattr(machine_record, "machine", None),
                               machine_record)
        except PrepError as exc:
            raise WrapperError(str(exc)) from None
    _preamble_chunks = ([("target", env_init["preamble"].rstrip("\n"))]
                        if env_init.get("preamble") else [])
    _activation_form = env_init["activation"]
    _scope_labels = {
        "target": "TARGET PREAMBLE (from the target machine's record)",
    }
    if _preamble_chunks:
        _rendered_chunks = [
            f"# === {_scope_labels.get(scope, scope.upper())} ===\n{text}\n"
            for scope, text in _preamble_chunks
        ]
        # EVERY ABSOLUTE PATH THE PREAMBLE SOURCES IS CHECKED BEFORE IT IS
        # SOURCED (2026-08-24, live hit on Sol).
        #
        # The preamble is baked VERBATIM from the machine that ran `prep`.
        # A bundle travels -- that is the whole point of it naming no
        # machine -- so an absolute path that exists on the workstation
        # need not exist on the cluster, and nothing at prep time can know
        # that: on the preparing machine the file is right there.  The only
        # code that runs on the TARGET is this script, so the check has to
        # be here.
        #
        # Without it bash says
        #     line 196: /home/.../conda.sh: No such file or directory
        # and dies -- naming neither the config key that put the path
        # there, nor the machine it was baked on, nor what to do.
        _srcs = _preamble_source_targets(_preamble_chunks)
        _guard = ""
        if _srcs:
            _lines = ["# --- Preamble preflight (paths baked elsewhere) ---"]
            for _pth in _srcs:
                _lines += [
                    f'if [ ! -r "{_pth}" ]; then',
                    '    _log ERROR "baked preamble sources a file that '
                    'does not exist on this machine:"',
                    f'    _log ERROR "    {_pth}"',
                    # SINGLE-quoted: bash runs backticks inside a
                    # double-quoted string.
                    "    _log ERROR 'It was baked verbatim from the "
                    "preamble of the record prep read, and this machine "
                    "is not the one that record describes.'",
                    "    _log ERROR 'Fix: environment.json beside task.json, "
                    "the record this calculation was prepared with, carries "
                    "this preamble -- edit its env_init.preamble (for "
                    "example: module load mamba), or correct the record of "
                    "that machine and prep the calculation anew, from the "
                    "state saved before its first prep; or edit the source "
                    "line below.'",
                    '    exit 78',           # EX_CONFIG
                    "fi",
                ]
            _guard = "\n".join(_lines) + "\n\n"
        _preamble_block = (
            "# --- Baked preamble (verbatim from the target's record) ---\n"
            + _guard
            + "_log STAGE \"running baked preamble\"\n"
            + "\n".join(_rendered_chunks)
            + "\n"
        )
    else:
        _preamble_block = (
            "# --- Baked preamble (none) ---\n"
            "# (the target's record states no preamble)\n"
            "\n"
        )

    return _preamble_block, f"{_activation_form} {env_name}"


#: The structured log helper every generated script defines -- its lines in
#: the session log's own format (`wrapper_log.LOG_LINE`), on stderr.
LOG_FUNCTION = (
    f"_log() {{\n"
    f"    printf '{LOG_LINE}\\n' \"$(date '+{LOG_CLOCK}')\" \"$1\" \"$2\" >&2\n"
    f"}}\n")

#: THE JOB'S PYTHON -- the activated env's own, never the first ``python3``
#: on PATH (`running-a-job.md` § 2.2a): set right after the activation.
JOB_PYTHON = '_mb_py="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python3}"\n'


def env_bootstrap(machine_record, env_name: str) -> str:
    """THE ENVIRONMENT ENTERED -- the one bootstrap every generated script
    runs before its own logic: a run script into its engine's environment,
    a sequence of groups' walk into molbuilder's own (`job-system.md`
    § 6.0, *The walk*).  Nounset off while the site's scripts run, the
    record's preamble and its activation of ``env_name``
    (:func:`env_entry`), nounset back on; then the job's python,
    ``$_mb_py`` (:data:`JOB_PYTHON`), and what the activation gave, in the
    log.  It runs under ``set -e``, which the caller holds: a preamble line
    that fails stops the script at that line (`running-a-job.md` § 2.0a).
    A run script's help request never reaches it: it answers with the
    arguments, first of all.  The script defines ``_log``
    (:data:`LOG_FUNCTION`)."""
    preamble, activation = env_entry(machine_record, env_name)
    return (
        "# The env bootstrap (preamble ``module load``, conda/mamba\n"
        "# ``activate``) sources EXTERNAL scripts not under our control.\n"
        "# Conda activate.d hooks (e.g. cuda-nvcc's, which references an\n"
        "# unset NVCC_PREPEND_FLAGS) abort under the run script's ``set -u``.\n"
        "# Disable nounset for the bootstrap ONLY; restore it before our\n"
        "# own logic (where -u still catches real bugs).\n"
        "#\n"
        "# NOTE `set -e` IS NOT DISABLED HERE, AND MUST NOT BE.  This\n"
        "# script runs unattended on a compute node, on an allocation.  A\n"
        "# preamble that fails means the job is about to run in the wrong\n"
        "# environment with nobody reading the output -- dying at the\n"
        "# failed line is the cheap outcome (running-a-job.md 2.0a).\n"
        "set +u\n"
        + preamble
        + "# --- Activation (verbatim from the target's record) ---\n"
        + f'_log STAGE "{activation}"\n'
        + f"{activation}\n"
        + "set -u\n"
        + "\n"
        # THE JOB'S PYTHON -- the activated env's own, never the first
        # `python3` on PATH (`running-a-job.md` § 2): a site's activation can
        # leave its own python ahead of the env's (`source activate` under a
        # mamba 2.x module, Sol, 2026-08-21 and 2026-10-24), which ran the
        # job's monitor, its ending reader, its finish and a PySCF deck on an
        # interpreter without the env's packages.  Every python a script
        # runs is this one, checked after (:func:`env_python_check`).
        + JOB_PYTHON
        + f'_log INFO "{CONDA_ENV_LABEL}${{CONDA_DEFAULT_ENV:-<unset>}}"\n'
        + '_log INFO "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"\n'
        + f'_log INFO "{PYTHON_LABEL} '
          f'${{_mb_py:-(none: no CONDA_PREFIX after activation)}}"\n'
        + "\n")


def env_python_check(env_name: str) -> str:
    """The head of the one check every script makes after its activation
    (`running-a-job.md` § 2.2a): ``CONDA_PREFIX`` names ``env_name`` and its
    python, ``$_mb_py``, is there -- the ``if`` whose body the script
    writes, its own way of stopping."""
    return (f'_mb_env="${{CONDA_PREFIX:-}}"\n'
            f'if [ "${{_mb_env##*/}}" != "{env_name}" ] '
            f'|| [ ! -x "$_mb_py" ]; then\n')


def host_env_name() -> str:
    """molbuilder's own environment's name -- its recipe's, the one with no
    engine category (`envs.recipes`, `configuration.md` § 2.1c)."""
    from .envs.recipes import builtin_recipes
    return next(r.name for r in builtin_recipes() if r.category is None)


def _mps_block() -> str:
    """Bash for NVIDIA MPS -- the Multi-Process Service, which lets several
    MPI ranks share one GPU at once (`running-a-job.md` § 3.3).  ONE
    DECISION (``_mps_wanted``): allowed (``--mps`` / ``--no-mps``, else
    ``MOLBUILDER_USE_MPS``), its binary on PATH, a GPU visible, and the ranks
    outnumber the GPUs -- ANY shared GPU gets the funnel (user decision
    2026-08-13).  A floor-division gate (``_ranks_per_gpu >= 2``) would miss
    the uneven split: 3 ranks over 2 GPUs floors to 1, yet GPU0 hosts 2
    ranks.  A ``--dry-run`` starts no daemon (a real GPU side-effect) and
    says what the run would do; the banner and the dry run read
    ``_use_mps_str``, set from this decision and the start's own outcome.
    Per-job pipe and log directories, so concurrent runs don't trample each
    other's daemon; the one EXIT trap tears it down.  Emit ONLY where ranks
    are placed on GPUs."""
    return (
                '# --- MPS daemon (Hyper-Q GPU sharing) -------------------\n'
                '_mps_wanted=0\n'
                '_mps_why=""\n'
                'if [ "$_use_mps_default" != "1" ]; then\n'
                '    _mps_why="not allowed (--no-mps, or MOLBUILDER_USE_MPS other '
                'than 1)"\n'
                'elif [ "$_have_mps" != "1" ]; then\n'
                '    _mps_why="nvidia-cuda-mps-control is not on PATH"\n'
                'elif [ "${_ngpu:-0}" -lt 1 ]; then\n'
                '    _mps_why="no GPU visible"\n'
                'elif [ "$_mpi_np" -le "$_ngpu" ]; then\n'
                '    _mps_why="no GPU is shared: $_mpi_np rank(s) over '
                '$_ngpu GPU(s)"\n'
                'else\n'
                '    _mps_wanted=1\n'
                'fi\n'
                'if [ "$_mps_wanted" = "1" ] '
                '&& [ "$_dry_run" != "1" ]; then\n'
                '    export CUDA_MPS_PIPE_DIRECTORY="/tmp/mb-mps-$$"\n'
                '    export CUDA_MPS_LOG_DIRECTORY="/tmp/mb-mps-$$-log"\n'
                '    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" '
                '"$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null || true\n'
                '    # Start MPS only if the control socket for THIS\n'
                '    # pipe dir is not already present (a global daemon\n'
                '    # started outside this run uses a different dir,\n'
                '    # so this check is independent).\n'
                '    if [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then\n'
                '        # || true: under set -e a failed daemon start\n'
                '        # (perms, global daemon holding the device) killed\n'
                '        # the run script HERE -- before the readiness loop\n'
                '        # below, whose whole job is timing out to the\n'
                '        # graceful no-MPS fallback (R9).\n'
                '        nvidia-cuda-mps-control -d 2>/dev/null || true\n'
                '        # Daemon readiness signal: the control UNIX\n'
                '        # SOCKET file appears in the pipe directory.\n'
                '        # It appears as soon as the daemon binds\n'
                '        # (typically <100 ms); ``get_server_list`` is\n'
                '        # empty until a client first connects, so it\n'
                '        # cannot say the daemon is ready.\n'
                '        _mps_wait=0\n'
                '        while [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; do\n'
                '            sleep 0.1\n'
                '            _mps_wait=$((_mps_wait + 1))\n'
                '            if [ "$_mps_wait" -gt 50 ]; then\n'
                '                echo "molbuilder: MPS control socket '
                'did not appear within 5 s ('
                '$CUDA_MPS_PIPE_DIRECTORY/control); falling back '
                'to no-MPS." >&2\n'
                '                _use_mps_default=0\n'
                # D18b: the exports must not outlive the fallback -- a
                # CUDA client finding these set talks to a daemon-less
                # pipe dir and hangs at init.  The dirs go here too:
                # the EXIT trap reads these very vars, so after the
                # unset nothing else would remove them.
                '                rm -rf "$CUDA_MPS_PIPE_DIRECTORY" '
                '"$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null || true\n'
                '                unset CUDA_MPS_PIPE_DIRECTORY '
                'CUDA_MPS_LOG_DIRECTORY\n'
                '                break\n'
                '            fi\n'
                '        done\n'
                '    fi\n'
                '    # Mark MPS as started so the SINGLE unified EXIT trap\n'
                '    # (_mb_cleanup, set near the top) stops the daemon +\n'
                '    # removes the per-job dirs on ANY exit (success, error,\n'
                '    # signal).  Set regardless of whether the daemon bound\n'
                '    # -- a partially-started daemon still wants cleanup.\n'
                '    # NB: a local ``trap ... EXIT`` here would be CLOBBERED\n'
                '    # by the per-rank launcher trap; the unified function\n'
                '    # is why teardown is centralised.\n'
                '    _mps_started=1\n'
                # Gate the "MPS enabled" message on the daemon-bind
                # result: otherwise the readiness-poll fallback above and
                # the line below would print two contradictory messages,
                # with the run continuing without MPS but the banner
                # claiming otherwise.
                '    if [ "$_use_mps_default" = "1" ]; then\n'
                '        echo "molbuilder: MPS enabled '
                '(pipe=$CUDA_MPS_PIPE_DIRECTORY)" >&2\n'
                '    fi\n'
                'elif [ "$_mps_wanted" != "1" ]; then\n'
                '    echo "molbuilder: MPS off ($_mps_why)" >&2\n'
                'fi\n'
                # WHAT THE BANNER AND THE DRY RUN SAY: on where the decision
                # is on and its start bound -- a start that timed out falls
                # back and clears `_use_mps_default`; a dry run starts
                # nothing, so it says what the run would do.
                '_use_mps_str="off"\n'
                'if [ "$_mps_wanted" = "1" ] '
                '&& [ "$_use_mps_default" = "1" ]; then '
                '_use_mps_str="on"; fi\n'
    )


def _program_launcher_block(prog: str, words: str, target_env: str, *,
                            ranks: bool, gpu_ranks: bool) -> str:
    """Bash that finds the program ``prog`` -- the job's, else its engine's
    own (``words``, as its record names it) -- asks what it was built with
    and decides how it is launched (`running-a-job.md` § 2.3).

    ``<program> --version`` (SIESTA 5.x +) self-reports the parallelisation
    the binary was compiled with -- for a typical conda-forge build::

        Version         : 5.4.2
        Parallelisations: MPI

    For an engine that runs ranks (``ranks``) the launcher follows it:

        MPI present      ->  mpirun -np <N> <program>   (always)
        OMP present      ->  bare <program>             (OMP env vars take effect)
        both             ->  mpirun -np <N> <program>   (hybrid)
        probe failed     ->  mpirun -np <N> <program>   (safe default for
                                                         MPI-compiled binaries)
        serial build     ->  bare <program>

    An engine of one process runs the program alone, its threads.  The probe
    runs ONCE per start and prints what it found before the launch, so the
    user sees the build and the launcher choice in the log -- adapting if
    the program is rebuilt, with no new prep.  Where ranks are placed on
    GPUs (``gpu_ranks``) the per-rank helper starts the program
    (:func:`_gpu_per_rank_launcher_block`).  Sets ``$_launch_cmd``,
    ``$_launch_note`` and ``$_mb_cores`` -- the cores this launch uses, the
    monitor's cpu% denominator (`run-reports.md` § 2.1a).
    """
    return (
            # The block NAME is job-contracts.md § 2.6's row.
            "# --- Probe the program's build at runtime ---\n"
            f'_program_bin="$(command -v {prog} || echo \"\")"\n'
            f'if [ -z "$_program_bin" ]; then\n'
            f"    echo \"ERROR: '{prog}' not on PATH after activating "
            f"'{target_env}'.  Is {words} installed in this env?\" >&2\n"
            + '    [ "$_dry_run" = "1" ] || '
            + conclude_line(1, f"{prog} not on PATH") + "\n"
            f"    exit 1\n"
            f"fi\n"
            # THE PROBE CANNOT BLOCK THE JOB (2026-08-25).  Each of these
            # is needed and none alone is enough; this is the order they
            # were found in, live on the dev workstation:
            #
            #  1. `|| true` catches a probe that FAILS.  It cannot catch
            #     one that never returns -- and this one can.  SIESTA reads
            #     its deck from STANDARD INPUT, so a build that does not
            #     know `--version` does not error: it waits for a deck,
            #     forever, holding the job it was about to describe.  A
            #     2023 `/usr/local/bin/siesta` does exactly this.
            #  2. A CLOCK, because closing stdin is not enough -- measured:
            #     that binary blocks with stdin at /dev/null too.
            #  3. A FILE, not a pipe, because the clock is not enough
            #     either.  `timeout` signals the process it started; the
            #     probe FORKS, the child outlives the signal still holding
            #     the write end, and `$( )` waits on a pipe that never sees
            #     EOF.  The wrapper then hangs AFTER the clock has already
            #     fired.  A file has no
            #     EOF to wait on, so a surviving child cannot hold it.
            #  4. `-k`, because a clock that only ASKS is not a bound.
            #     Plain `timeout` sends TERM and then waits -- indefinitely,
            #     against anything that ignores it, which MPI launchers do.
            #     `-k 2` follows with KILL, and KILL cannot be declined.
            #  5. The answer is READ from that file by awk, never piped
            #     into it.  awk stops at its first match; bash's printf
            #     writes a value line by line, so a writer still holding
            #     the next line dies of SIGPIPE, and under pipefail +
            #     set -e the wrapper ended there -- silently, before the
            #     engine (2026-10-06: a direct launch under the full test
            #     batch left a run with no output and no error).
            #
            # An unanswered probe leaves the file empty, which the launcher
            # choice below already treats as *probe failed* -> mpirun, the
            # documented safe default.  Five seconds is a thousand times any
            # real answer and nothing against a wall.  Without coreutils
            # `timeout` the bare call stands: nothing portable bounds it,
            # and refusing to probe at all would cost every machine its
            # version banner to guard against one broken binary.
            f'_mb_probe_out="$(mktemp 2>/dev/null '
            f'|| echo "${{TMPDIR:-/tmp}}/mb-probe.$$")"\n'
            f'if command -v timeout >/dev/null 2>&1; then\n'
            f'    timeout -k 2 5 {prog} --version >"$_mb_probe_out" 2>/dev/null '
            f'</dev/null || true\n'
            f'else\n'
            f'    {prog} --version >"$_mb_probe_out" 2>/dev/null '
            f'</dev/null || true\n'
            f'fi\n'
            f"_program_ver=\"$(awk -F': *' '/^Version/ {{print $2; exit}}' "
            f'"$_mb_probe_out" 2>/dev/null || true)"\n'
            f"_program_par=\"$(awk -F': *' '/^Parallelisations/ {{print $2; exit}}' "
            f'"$_mb_probe_out" 2>/dev/null || true)"\n'
            f'rm -f "$_mb_probe_out"\n'
            f"# Decide launcher from probe.  Default to mpirun (safe\n"
            f"# for any MPI-compiled binary) when the probe can't\n"
            f"# tell us anything.\n"
            f'_has_mpi=0; _has_omp=0\n'
            f'# Word-boundary match.  ``*MPI*`` alone would falsely\n'
            f'# catch ``NoMPI`` / ``pre-MPI`` / ``nompi``.  Strategy:\n'
            f'# focus on CONTENT, not formatting -- normalise ANY\n'
            f'# whitespace (space, tab, vertical-tab, ...) AND the\n'
            f'# comma/semicolon list-separators to single spaces, then\n'
            f'# space-anchor the token match.  ``tr "[:space:]" " "``\n'
            f'# rewrites every POSIX whitespace char; ``tr ",;" "  "``\n'
            f'# absorbs the list separators; ``tr -s " "`` collapses\n'
            f'# runs.  Robust against any SIESTA build that prints the\n'
            f'# Parallelisations line with tabs / extra spaces /\n'
            f'# mixed list separators -- the value semantics carry.\n'
            f'_par_norm=" $(echo \"$_program_par\" '
            f'| tr "[:space:]" " " | tr ",;" "  " | tr -s " ") "\n'
            f'case "$_par_norm" in *" MPI "*) _has_mpi=1 ;; esac\n'
            f'case "$_par_norm" in *" OMP "*|*" OpenMP "*) _has_omp=1 ;; esac\n'
            # GPU mode: pin threads to cores in the same package so
            # OpenMP doesn't spill across sockets (SIESTA performance-
            # options guidance), and tell OpenMPI to place ranks one
            # per package.  On OpenMPI 5.x "package" is canonical;
            # "socket" still works as an alias.
            + (
                # PE counting hazard (measured 2026-06-16 in a live
                # 212-atom Au-BDT run): a ``ppr:K:package:PE=$_omp``
                # form, on Intel HT boxes, allocates PE=2 *processing
                # units* (PUs) per rank mapped as HT-sibling pairs of
                # ONE physical core -- 4 ranks x PE=2 used only 4
                # physical cores (not 8).
                #
                # Hence the canonical "N physical cores per rank,
                # packed onto packages" form:
                #
                #   --map-by package:PE=$_omp_threads
                #     map ranks across packages, PE counts physical
                #     CORES (not PUs) by default in OpenMPI 5 without
                #     ``--use-hwthread-cpus``.  The whole-job numactl
                #     wrap does NOT hold the ranks to its node:
                #     measured 2026-10-17 with OpenMPI 5.0.10 on a
                #     two-socket box, this map overrides the wrap's
                #     cpuset, so a single GPU's second rank lands on
                #     the far socket -- on cores of its own, which
                #     the rank helper then warns about (`gpu.md`
                #     G10).  Binding nothing instead put every rank's
                #     OpenMP threads on the same cores, which is
                #     worse.
                #   --bind-to core
                #     bind each rank to its PE cores (one cpuset
                #     per rank covering all its cores; OS scheduler
                #     places OMP threads on those cores).
                #
                # NB: NO ``--rank-by core``.  OpenMPI 4.x rejects it
                # ("Valid directives: slot:node:fill:span"); rank
                # ordering defaults to map order which is already
                # deterministic for our use.  Caught 2026-06-16 by
                # bench rc=213 on the user's box.
                '_mpirun_bind="--bind-to core --map-by '
                'package:PE=$_omp_threads"\n'
                if gpu_ranks else
                '_mpirun_bind=""\n'
            )
            # ``_numa_wrap_gpu`` is set by _gpu_runtime_block
            # in GPU mode; default to empty here so CPU-mode wrappers
            # (which never inject that block) still see a defined var
            # in the launch_cmd interpolation below.  This is also the
            # safe-default branch when GPU mode runs on single-socket
            # boxes / boxes where numactl isn't installed -- the block
            # leaves the var empty in those cases.
            + '_numa_wrap_gpu="${_numa_wrap_gpu:-}"\n'
            # Default launch target is the bare binary; GPU mode swaps in
            # the per-rank launcher (assigns each rank its GPU + picks the
            # CPU-bind policy).  See _gpu_per_rank_launcher_block.__doc__.
            + f'_program_target="{prog}"\n'
            + (_gpu_per_rank_launcher_block(prog) if gpu_ranks else "")
            # ``_mb_cores``: the cores THIS launch uses -- what the job holds
            # when it was started directly, the monitor's cpu% denominator
            # (`run-reports.md` § 2.1a).  Decided here, with the launcher:
            # ranks x threads only for a hybrid build, and one core for a
            # serial one, whatever -np and -omp said.
            + ((
                # RANKS: the launcher the build calls for.
                f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_program_target"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ($_mpi_np ranks x $_omp_threads OMP threads)"\n'
            f'        _mb_cores=$(( _mpi_np * _omp_threads ))\n'
            f'    else\n'
            f'        _launch_note="pure MPI ($_mpi_np ranks; OMP setting irrelevant to this binary)"\n'
            f'        _mb_cores=$_mpi_np\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="{prog}"\n'
            f'    _launch_note="OMP-only build ($_omp_threads threads)"\n'
            f'    _mb_cores=$_omp_threads\n'
            f'elif [ -z "$_program_par" ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_program_target"\n'
            f'    _launch_note="MPI fallback (probe inconclusive; safe default for an MPI-compiled binary)"\n'
            f'    _mb_cores=$_mpi_np\n'
            f'else\n'
            f'    _launch_cmd="{prog}"\n'
            f'    _launch_note="serial build (no parallelisation compiled in)"\n'
            f'    _mb_cores=1\n'
            f"fi\n"
            f"\n"
            ) if ranks else (
                # ONE PROCESS: the program alone, its threads.
                f'_launch_cmd="{prog}"\n'
                '_launch_note="one process ($_omp_threads OpenMP threads)"\n'
                '_mb_cores=$_omp_threads\n'
                '\n'
            ))
        )


def _prior_state_words(eng, resumes: bool, continues: bool,
                       honoured: Optional[bool],
                       reads: Dict[str, Tuple[str, bool]]) -> str:
    """The run script's header paragraph on what its deck reads of the
    restart files launch copies in (`job-contracts.md` § 4.1) -- ONE
    composer for every engine (``eng``, its record): a deck whose keywords
    say it, file by file (:func:`_deck_restart`); any other from the facts
    its launch decides by -- whether its kind resumes (`Job.resumes`) and
    whether its stage takes something from a run
    (`Job.relaunch_continues`)."""
    head = f"* {eng.words}: this deck "
    if eng.keyword_reader is not None:
        if honoured is None:
            text = (head + "states no restart keyword that could be read; "
                    "whether the restart files are read is the deck's to "
                    "say.  To restore the restart group the generator "
                    "writes, " + way_back_words(None) + ".")
        else:
            text = head + _reads_words(eng.words, reads)
            if honoured is False:
                text += ("  To continue instead, "
                         + way_back_words(None, change="change `restart` in "
                                                       "the description")
                         + ".")
    elif not resumes:
        text = (head + "reads no prior engine state: this kind of run does "
                "not resume, so each run starts from its deck.")
    elif continues:
        text = (head + "reads the restart files launch copies in: its "
                "stage continues from a run.")
    else:
        text = (head + "is given none: its stage takes nothing from a run, "
                "so launch copies none in.")
    return _wrapped(text, "#  ")


def render_run_wrapper(script_path: Path, *,
                       names,
                       engine: str,
                       calculation: str,
                       resources: "Resources",
                       program: Optional[str] = None,
                       env: Optional[str] = None,
                       machine_record,
                       finish: Optional[str] = None,
                       resumes: bool,
                       relaunch_continues: bool,
                       trial: bool,
                       warm: Optional[Sequence[str]] = None,
                       deck_text: Optional[str] = None,
                       notes: Optional[list] = None,
                       point: Optional[PointOfRun] = None) -> str:
    """Return the bash text of the run script that runs ``script_path``.

    **ONE RUN SCRIPT, EVERY ENGINE** (`running-a-job.md` § 1): what differs
    between two runs is a fact of their engine's record (`engine_table`,
    `architecture.md` § 3.3) or of their job, never the engine's name or the
    deck's suffix --

      * ``one_process``: ranks to resolve, launch under ``mpirun`` and place
        on GPUs -- or one process with threads, ``-np`` accepted and said to
        be ignored;
      * ``programs``: a program to probe, launch and cap the memory of
        (``program``, the job's, else the engine's own) -- or none, a deck
        that is a script the job's python runs;
      * ``keyword_reader``: a deck of keywords, whose lines the log records
        as the engine will read them (that reader, beside the job), and whose
        restart keywords say file by file what it reads of the restart
        files;
      * ``environment`` / ``gpu_environment``: the environment it enters;

    and asked by the output's role: whether the run script cuts a timing
    file from it (the file catalogue's `cut_from_output`), its SCF rows'
    patterns (`scf_timing_rows.scf_row_patterns`), and whether the engine's
    stderr is kept apart from it (`runfiles.stderr_apart`).

    ``engine`` is prep's -- the calculation's (`JobSet.engine`); the deck is
    one of its decks.  ``notes`` receives what the rendering has to say and
    does not refuse (`gpu.md` G6: the target's record lists no GPU build) --
    the same out-list `prep` hands every step, so a sentence reaches its
    answer.  ``point`` makes the run a POINT OF A RUN OF POINTS
    (:class:`PointOfRun`, `execution/runs-of-points.md` § 1.1): its monitor a
    member of the run's, its two report steps around the engine
    (:func:`_point_block`) -- which run the kind's own steps, verbs of the
    ``finish`` bundle, where it has them -- and its relaunch said after a
    stop.  ``deck_text``: the deck as it will be written, for one `prep` has
    planned and not yet written (`jobset.planned`); read from
    ``script_path`` when omitted.

    **The allocation arrives whole** — `architecture.md` § 3.1, rule A8.
    ``env`` stays loose: a per-invocation override (``prep --env``).
    """
    from .engine_table import EngineTableError, engine as _engine
    from .parse.engines._run_ending import (advice_of as _advice_of,
                                            wall_causes as _wall_causes)
    from .parse.instruments.scf_timing_rows import scf_row_patterns
    from .runfiles import cut_from_output
    try:
        eng = _engine(engine)
    except EngineTableError as exc:
        raise WrapperError(f"`{Path(script_path).name}`: {exc}") from None
    if deck_text is None:
        from .engine_table import read_deck
        try:
            deck_text = read_deck(engine, script_path)
        except (OSError, ValueError):
            deck_text = None
    r = resources
    mpi_np, omp_threads = r.mpi_np, r.cpus_per_task
    max_memory_mb = r.max_memory_mb
    # THE WALL SLACK (`running-a-job.md` § 3.6): the job's own (checked
    # below with every run setting the script reads).
    wall_slack = r.wall_slack
    # WHEN this calculation should speak up -- carried on the allocation
    # like `wall_slack`, and rendered as monitor flags rather than
    # scheduler ones.  Absent means the flags are not emitted at all, so a
    # run script for a description that asked for nothing carries none.
    # Read straight off the object, not via getattr-with-a-default: this
    # function is typed for `Resources` and a defaulting read would turn a
    # field that went missing into a run script that silently stops
    # notifying.
    notify_on_scf = bool(r.notify_on_scf)
    notify_every_hours = float(r.notify_every_hours or 0.0)
    notify_channels = r.notify_channels
    from .config_dir import ALL_CHANNELS
    if notify_channels is not None and tuple(notify_channels) != (ALL_CHANNELS,):
        # VALIDATED HERE because this is where a name becomes SHELL.  Every
        # name that arrives through `task.json` was checked at parse, but
        # `Resources` can be built directly, and a value that reaches a
        # generated script unchecked is a quoting bug waiting for the one
        # caller that does not go through the description.
        bad = [n for n in notify_channels if not is_channel_name(n)]
        if bad:
            raise WrapperError(
                f"notify channel name(s) {', '.join(map(repr, bad))}: only "
                f"{CHANNEL_NAME_RULE}.  A name is written into a "
                f"description and rendered into the monitor's command line, "
                f"and anything else would mean one thing in the file and "
                f"another in the shell.")
    notify_report = r.notify_report
    if notify_report is not None:
        # SAME REASON as the channel names above: this becomes SHELL.
        # And the vocabulary is closed, so a name outside it is a typo
        # the person can still fix rather than a field that silently
        # never arrives -- the one declaration's sentence (`stages.md` 6.9).
        from .report_fields import refusal
        why = [w for w in (refusal(n) for n in notify_report) if w]
        if why:
            raise WrapperError(f"notify report field: {why[0]}")
    script_path = Path(script_path)
    suffix = script_path.suffix.lower()
    # THE DECK IS ITS ENGINE'S (`runfiles.deck_role`): a run script runs its
    # own engine's deck, and its stage's names name it.
    if suffix != deck_role(engine):
        raise WrapperError(
            f"`{script_path.name}` is no deck of {eng.words}'s -- its deck "
            f"is a {deck_role(engine)}; a run script runs its own engine's "
            f"deck.")
    if script_path.name != names.name(suffix):
        raise WrapperError(
            f"{script_path.name} is not the deck {names.name(suffix)} its "
            f"stage's names name -- a run script runs its own stage's deck.")
    # THE JOB CARRIES EVERY RUN SETTING THIS SCRIPT READS -- the
    # engine's wrapper items with no unset (the wall's slack): the run
    # card's, or the catalogue's default `resolve`
    # lays (`engines/template.md` § 5.4).  The script falls back to no
    # number of its own, so a job without one is refused.
    from .template import catalogue as _catalogue, select as _select
    for _it in _select(_catalogue(), engine=engine, kind="wrapper"):
        if not _it.optional and getattr(r, _it.name) is None:
            raise WrapperError(
                f"the job states no {_it.name} -- every job carries it, its "
                f"run card's or the catalogue's default, laid at prep; "
                f"prepare it again.")
    # A FINISH THE LAUNCHER CANNOT SHIP IS REFUSED, never dropped: a job
    # told to finish that ran without finishing would conclude rc=0 with no
    # result (`engines/vibration.md` § 5.5).
    if finish is not None and finish not in _FINISH_BUNDLES:
        raise WrapperError(
            f"`{script_path.name}`: a job finishes with "
            f"{', '.join(sorted(_FINISH_BUNDLES))}, never {finish!r}")
    from .monitor import POINT_ENDED, POINT_STARTED
    if point is not None and point.steps and finish is None:
        raise WrapperError(
            f"`{script_path.name}`: a point's own steps are verbs of its "
            f"job's bundle, and the job names none")

    # WHETHER THIS RUN USES THE GPU is its request -- `jobset.model.
    # gpu_request`, the one door (`execution/architecture.md` § 3.2) -- read
    # once, here, for the env, the GPU placement and the monitor alike.
    from .jobset.model import GpuRequestError, gpu_request
    try:
        gpus = gpu_request(resources)
    except GpuRequestError as exc:
        raise WrapperError(f"`{script_path.name}`: {exc}") from None

    # WHAT THE ENGINE'S RECORD SAYS OF HOW ITS RUNS RUN.
    ranks = not eng.one_process
    prog = program or eng.program
    try:
        words = eng.program_words(prog)
    except EngineTableError as exc:
        raise WrapperError(f"`{script_path.name}`: {exc}") from None
    # GPU PLACEMENT IS A RANK'S: an engine of one process uses the device
    # from its own deck (`execution/gpu.md` G8).
    gpu_ranks = gpus.uses and ranks

    # THE ENVIRONMENT IS THE RECORD'S (`running-a-job.md` § 2.3): its
    # `environment`, or -- for a job that uses a GPU -- its
    # `gpu_environment` where it names one.  An env the USER named always
    # wins.
    #
    # THE TWO SIESTA ENVS SPLIT ON PROVENANCE, NOT ON HARDWARE (2026-08-13).
    # ``molbuilder-siesta`` is installable from packages on ANY machine;
    # ``molbuilder-siesta-gpu`` must be BUILT FROM SOURCE, which some HPC
    # sites do not permit.  CPU-ELPA needs NEITHER: the packaged build
    # carries ELPA through ELSI and runs both stages on CPU (measured on H2,
    # `engines/siesta.md` § 7.2) -- only the GPU entry is absent from it, a
    # build capability, not a missing device.  So the ONE thing that
    # decides which env a run enters is whether it uses the GPU -- never
    # the solver.
    category = (eng.gpu_environment
                if env is None and gpus.uses and eng.gpu_environment
                else eng.environment)
    caps = get_capabilities()
    target_env = (env if env is not None
                  else caps.env_for_category(category))
    if target_env is None:
        raise WrapperError(
            f"engine_table.toml [{engine}] names the environment category "
            f"{category!r}, and no environment is registered for it -- the "
            f"categories are diagnostics.DEFAULT_ENV_NAMES; pass --env to name "
            f"one.")

    # THE GPU BUILD'S PRESENCE IS SAID, NEVER A VETO (`gpu.md` G6; the
    # user, 2026-10-17: the record is shown, the person decides).  When the
    # job's GPU request routes it to its engine's GPU environment and the
    # target's record lists none, the run script's activation would stop
    # the run at its start with a conda-side error that names no fix.  So
    # prep says it here -- the env, that the run stops without it, and the
    # command that installs it -- and prepares the job as asked.  Only when
    # we routed (env not user-passed) and the inventory lists at least one
    # env: an empty inventory is "unknown", never "none".
    #
    # WHICH MACHINE IS ASKED WHETHER THE ENV EXISTS: the one that will run
    # this, by its record (`configuration.md` § 5 M-1) -- including when
    # that record cannot answer.
    _known_envs = [str(e) for e in (machine_record.conda_envs or [])]
    if (env is None and category != eng.environment and _known_envs
            and target_env not in _known_envs):
        _probed = machine_record.detected_at
        _where = (f"the record of "
                  f"{machine_record.machine or 'the target machine'}"
                  + (f" (probed {_probed})" if _probed else ""))
        _there = "on " + (machine_record.machine or "that machine")
        _said = (
            f"`{script_path.name}`'s job asks for the GPU (its GPU "
            f"request), and {_where} lists no ``{target_env}`` environment, "
            f"{eng.words}'s GPU build.  The run script enters it at its "
            f"start and stops there if it is absent -- so install it before "
            f"launching, {_there}: ``{_hints.fix_cmd('install', target_env)}`` "
            f"-- or turn ``Use GPU`` off and prep again.")
        if notes is not None and _said not in notes:
            notes.append(_said)

    # WHAT THE DECK READS OF THE RESTART FILES -- from the deck where its
    # settings are keywords, so the run script's own texts cannot
    # contradict the file it ships beside; a script deck reads what launch
    # copies in, and a stage set to start clean takes nothing
    # (`Job.relaunch_continues`).
    _restart_honoured, _reads = _deck_restart(engine, deck_text)
    # WHAT A LAUNCH AGAIN WILL DO -- one description (`_launch_again`),
    # said after a failure (`running-a-job.md` § 3.5), from the facts the
    # launch decides by and the deck's own restart answer.
    _again = _launch_again(resumes, relaunch_continues, _restart_honoured,
                           relaunch=(point.relaunch if point is not None
                                     else None),
                           trial=trial)

    basename = names.stem
    script_name = script_path.name
    run_sh = names.run_script
    # Shell-safety: both basename and script_name are interpolated
    # raw into bash f-strings throughout this module (inside
    # ``"..."``, inside glob lists, inside ``$(...)``, etc.).
    # Reject anything outside ``_SAFE_WRAPPER_NAME_RE`` to prevent
    # shell injection via a malicious filename.
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for run-script emission: "
            f"{basename!r}.  Allowed characters: letters, digits, "
            f"``.``, ``_``, ``-``.  Rename the script and prep "
            f"again (`molbuilder jobset prep`)."
        )
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(script_name):
        raise WrapperError(
            f"unsafe script filename for run-script emission: "
            f"{script_name!r}."
        )

    # ---- WHAT THE PARSE LAYER SAYS OF THE OUTPUT, asked by its role ----
    stdout_role = _stdout_role_of(engine)
    # THE TIMING FILE CUT FROM THE OUTPUT, where the file catalogue names
    # one, its rows stamped by the grammar's patterns for that output.
    timing_role = cut_from_output(stdout_role)
    stamp = (scf_row_patterns(stdout_role) if timing_role is not None
             else None)
    stderr_apart = _stderr_apart(stdout_role)
    # THE COMMAND, as it runs and as the banner shows it: a program its
    # launcher starts, or a script deck on the job's own python.
    if prog is not None:
        launch_words = '$_launch_cmd ${_mb_wall_args[@]+"${_mb_wall_args[@]}"}'
        launch_shown = "$_launch_cmd"
    else:
        # QUOTED where it runs; shown bare, inside the banner's own quotes.
        launch_words, launch_shown = '"$_mb_py"', "$_mb_py"
    to_output = (" > $_out_file" if stderr_apart else " > $_out_file 2>&1")
    launch_line = f"{launch_shown} {script_name}{to_output}"

    # THE WALL'S WORDS FOR THIS PROGRAM -- the catalogue's `launch` for
    # `wall_slack` (template.md § 5), rendered with what the script counts:
    # never a flag spelled here.  A program the item names no words for
    # (tbtrans) checks no clock; a script deck reads the wall from its
    # environment, exported below (`running-a-job.md` § 3.6).
    from .script_emit import parameter as _parameter
    _wall_decl = _parameter("wall_slack", engine).declaration
    _wall_words = (tuple(_wall_decl.launch.get(prog, ()))
                   if _wall_decl is not None and prog is not None else ())
    takes_wall = bool(_wall_words) or (
        _wall_decl is not None
        and _wall_decl.limit_applied_by[engine] == "deck")
    # ASKED ONLY OF AN OUTPUT WHOSE FAMILY NAMES A STOP AT THE WALL, and its
    # advice for a cause (its catalogue row's `stops`; `running-a-job.md` § 3.5).
    asks_wall = takes_wall and bool(_wall_causes(stdout_role))
    asks_advice = bool(_advice_of(stdout_role))
    # THE MEMORY CAP, by its item's declaration alone: the run script's
    # `ulimit -v` before the engine starts, unless its deck applies the cap
    # itself (`limit_applied_by`).
    _mem_decl = _parameter("max_memory_mb", engine).declaration
    caps_memory = (max_memory_mb is not None and int(max_memory_mb) > 0
                   and (_mem_decl is None
                        or _mem_decl.limit_applied_by[engine] != "deck"))
    _wall_array = " ".join(
        '"' + w.replace("{left}", "${_mb_wall_left}")
               .replace("{slack}", "${" + WALL_SLACK_VAR + "}")
        + '"'
        for w in _wall_words)
    _slack_export = f"export {WALL_SLACK_VAR}={int(wall_slack)}\n"

    # ---- THE ARGUMENTS: one parser; the ranks where the engine runs them ----
    stated = ("# The stated counts, baked at prep (running-a-job.md § 3.1-3.2).\n"
              + (f"{_STATED_COUNTS[0]}={int(mpi_np)}\n" if ranks else "")
              + f"{_STATED_COUNTS[1]}={int(omp_threads)}\n")
    args_parse = _args_parse(
        stated=stated,
        omp_aliases="-omp|--omp|-t|--threads",
        inits=('_mps_flag=""\n' if gpu_ranks else ""),
        # WHETHER MPS IS ALLOWED.  Unsaid, the MOLBUILDER_USE_MPS env var
        # says, else allowed (the GPU placement block).  These flags ALWAYS
        # win, and they allow or forbid the daemon and nothing else: the
        # rank count is the stated one either way, and the MPS block starts
        # it only where the ranks outnumber the GPUs.
        arms=("        --mps)\n"
              '            _mps_flag=1; shift ;;\n'
              "        --no-mps)\n"
              '            _mps_flag=0; shift ;;\n' if gpu_ranks else ""),
        usage=(
            'Usage: bash $(basename "$0") [-np N] [-omp N] [--dry-run]'
            + (" [--mps|--no-mps]" if gpu_ranks else "")
            + " [-h]\n"
            "\n"
            + _one_start_usage(trial)
            + ("  -np N            override the MPI rank count.  Stated\n"
               "                   at prep: $_mpi_np_default.\n" if ranks else
               f"  -np N            accepted and IGNORED -- {eng.words} runs\n"
               "                   one process; N>1 prints a note.\n")
            + "  -omp N, -t N     override the OpenMP threads"
            + (" per MPI rank" if ranks else "") + ".\n"
            "                   Stated at prep: $_omp_threads_default.\n"
            + ("  --mps, --no-mps  allow or forbid NVIDIA MPS; it starts only\n"
               "                   where ranks outnumber GPUs.\n"
               if gpu_ranks else "")
            + "  --dry-run        resolve + log the launch command"
            + (" and the\n                   rank->GPU/NUMA placement" if gpu_ranks
               else "")
            + ", then exit\n"
            f"                   WITHOUT running {words}.\n"
            "  -h               this help.\n"
            "\n"
            "Environment variables:\n"
            + ("  MB_NP=N            same as -np N.\n" if ranks else "")
            + "  OMP_NUM_THREADS=N  same as -omp N (standard OMP toolchain\n"
              "                     convention; honored if set in env).\n"))

    # ---- THE COUNTS, resolved on the node (running-a-job.md § 3.1-3.2) ----
    # THE ATOMS THE DECK HOLDS, stated at prep from its own record: a deck
    # prep wrote whose record does not read is refused here, never shown as
    # words on the banner.
    from .deck_record import DeckRecordUnreadable
    try:
        held = _held_words(deck_text)
    except DeckRecordUnreadable as exc:
        raise WrapperError(
            f"`{script_path.name}`: its atom record does not read ({exc}) "
            f"-- prepare the stage again") from None
    counts = (
        "# --- The counts, resolved (running-a-job.md § 3.1-3.2) ---\n"
        + ("# MPI rank-count precedence (highest first): the -np flag,\n"
           "# MB_NP, MB_NTASKS (the ranks the scheduler granted, said in\n"
           "# molbuilder's name by the job's header), the stated value.\n"
           '_mpi_np="${_np_flag:-${MB_NP:-${MB_NTASKS:-$_mpi_np_default}}}"\n'
           '_np_source="stated at prep"\n'
           'if [ -n "$_np_flag" ]; then _np_source="-np flag"\n'
           'elif [ -n "${MB_NP:-}" ]; then _np_source="MB_NP env"\n'
           'elif [ -n "${MB_NTASKS:-}" ]; then '
           '_np_source="MB_NTASKS (the ranks the scheduler granted)"\n'
           'fi\n' if ranks else
           "# One process: no rank count (running-a-job.md § 3.1).\n"
           '_mpi_np=1\n'
           '_np_source="one process"\n'
           'if [ -n "$_np_flag" ] && [ "$_np_flag" != "1" ]; then\n'
           f'    echo "molbuilder: -np $_np_flag ignored -- {eng.words} runs '
           'one process (OpenMP threads); use -omp for the thread count" >&2\n'
           'fi\n')
        # THE THREADS, by the one chain (`runtime_info.THREAD_SOURCES`;
        # `running-a-job.md` § 3.2): the -omp flag, else OMP_NUM_THREADS,
        # else the scheduler's grant, else the count stated at prep.
        + f'_omp_threads="${{_omp_flag:-{_thread_chain("$_omp_threads_default")}}}"\n'
        + ('if [ -n "$_mps_flag" ]; then _use_mps_default="$_mps_flag"; fi\n'
           if gpu_ranks else "")
        # WHERE each number came from, computed once the values are final --
        # the banner and the --dry-run report print these, so a wrong scale
        # is caught BEFORE a queue slot is spent (user design, 2026-08-13).
        + '_omp_source="stated at prep"\n'
        'if [ -n "$_omp_flag" ]; then _omp_source="-omp flag"\n'
        + "".join(f'elif [ -n "${{{var}:-}}" ]; then '
                  f'_omp_source="{_THREAD_SOURCE_WORDS[var]}"\n'
                  for var in THREAD_SOURCES)
        + 'fi\n'
        "\n"
        + _env_python_block(names, stdout_role, target_env)
        # THE COUNTS AS RESOLVED, once the start has begun: a value from
        # the environment that is no count is refused naming where it came
        # from, and the run concludes.
        + ('_mb_positive -np "$_mpi_np" "$_np_source"\n' if ranks else "")
        + '_mb_positive -omp "$_omp_threads" "$_omp_source"\n'
        + _runtime_status_block(
            held=held, label=names.label, resumes=resumes,
            reads_prior=_restart_honoured,
            # the files the deck says it reads
            warm=tuple(s for s in _warm_in_effect(engine, warm)
                       if _reads.get(s, ("", True))[1]))
    )

    env_prefix = (
        # THE NODE'S SHAPE, every run: provenance for any trial, and a CPU
        # sweep is the case that most needs it (`plan.md` E4).
        _phys_cores_probe_block()
        + (_gpu_runtime_block() if gpu_ranks else "")
        + counts
        # GPU load-balance: count the GPUs the job was given and derive
        # ranks-per-GPU from the resolved rank count -- after the counts
        # (needs $_mpi_np), before the MPS block (reads $_ngpu).
        + (_gpu_loadbalance_block() if gpu_ranks else "")
        + (_orbitals_per_rank_notice(_n_orbitals_est(deck_text))
           if ranks else "")
        + "# --- Thread / BLAS pinning ------------------------------\n"
        "#   * OMP_NUM_THREADS: the stated cores"
        + (" per rank" if ranks else "")
        + f" ({int(omp_threads)}); override with -omp N\n"
        "#     or by exporting OMP_NUM_THREADS before invoking.\n"
        + ("#   * BLAS pinned to 1 thread per rank, so ranks x OMP x BLAS\n"
           "#     doesn't oversubscribe the cores.\n" if ranks else
           "#   * BLAS: the deck sizes its own (one process).\n")
        + "export OMP_NUM_THREADS=$_omp_threads\n"
        + ("export MKL_NUM_THREADS=1\n"
           "export OPENBLAS_NUM_THREADS=1\n" if ranks else "")
        + (
            # Hybrid MPI+OMP needs the OMP runtime told to bind --
            # ``mpirun --bind-to core`` only binds the rank (cpuset), not
            # the threads inside it.  ``close`` keeps threads near the
            # rank's master; ``cores`` says "one place per core".
            "export OMP_PROC_BIND=close\n"
            "export OMP_PLACES=cores\n" if gpu_ranks else "")
        + (_mps_block() if gpu_ranks else "")
        # THE MEMORY CAP, before the engine starts: a soft guard against a
        # runaway allocation -- unless the engine's deck applies it itself
        # (`limit_applied_by`; `running-a-job.md` § 3.2).
        + (f"# Memory cap (max_memory_mb): {max_memory_mb} MB\n"
           f"ulimit -v {int(max_memory_mb) * 1024} || true  "
           f"# soft cap; ignored if the shell can't set it\n"
           if caps_memory else "")
        + "\n"
        + (_program_launcher_block(prog, words, target_env, ranks=ranks,
                                   gpu_ranks=gpu_ranks)
           if prog is not None else
           "# --- The deck is a script, run on the job's python ---\n"
           '_launch_note="a script deck on the job\'s python '
           '($_omp_threads OpenMP threads)"\n'
           "_mb_cores=$_omp_threads\n"
           "\n")
    )

    # THE BANNER, printed at run time so the user sees what the run is
    # about to do -- counts, threading, what it starts from, the command --
    # before the engine spends its first seconds reading the deck.  Its
    # labels line up on the longest one it prints, which is the program's
    # where it names one (``TBtrans version``).
    labels = (("Date", "Host", "Cwd", "Conda env")
              + ((f"{words} {BINARY_LABEL}", f"{words} {VERSION_LABEL}",
                  "Build paral.") if prog is not None else ())
              + ("Launch mode",) + (("Ranks",) if ranks else ())
              + ("Threading",)
              + (("GPU resources", "GPU monitor") if gpu_ranks else ())
              + ("Mode", "Held atoms", "Command", "Time limit", "Stdout"))
    width = max(map(len, labels))

    def lead(label: str) -> str:
        return f"  {label:<{width}} : "

    under = " " * len(lead(""))
    env_prefix += (
        f'echo "===== molbuilder run script: {words} ====="\n'
        f'echo "{lead("Date")}$(date -Iseconds)"\n'
        f'echo "{lead("Host")}$(hostname)"\n'
        f'echo "{lead("Cwd")}$(pwd)"\n'
        f'echo "{lead("Conda env")}${{CONDA_DEFAULT_ENV:-?}}"\n'
        + (f'echo "{lead(words + " " + BINARY_LABEL)}$_program_bin"\n'
           f'echo "{lead(words + " " + VERSION_LABEL)}'
           '${_program_ver:-unknown}"\n'
           f'echo "{lead("Build paral.")}${{_program_par:-unknown}}"\n'
           if prog is not None else "")
        + f'echo "{lead("Launch mode")}$_launch_note"\n'
        + (f'echo "{lead("Ranks")}$_mpi_np (from $_np_source)"\n'
           if ranks else "")
        + f'echo "{lead("Threading")}OMP_NUM_THREADS=$_omp_threads '
        '(from $_omp_source)' + (", OPENBLAS=1, MKL=1" if ranks else "")
        + '"\n'
        + ((
            # THE single authoritative GPU-resource summary, printed with
            # the RESOLVED launch values so it always matches what runs;
            # $_gpu_numa is GPU 0's NUMA node, looked up at run time.
            f'echo "{lead("GPU resources")}$_mpi_np ranks x $_omp_threads '
            'threads ($(( _mpi_np * _omp_threads )) cores); '
            'mps=$_use_mps_str; ranks/GPU=${_ranks_per_gpu:-?}; '
            'GPU0 NUMA=$_gpu_numa"\n'
            # TUNE BY MEASURING, not by asking for a guess.
            f'echo "{under}# tune: -np / -omp / --mps / --no-mps "'
            '"(or measure it: prep bench, for this stage)"\n'
            # The command on its own line, so it pastes into a shell; the
            # annotation on the next, a comment.
            f'echo "{lead("GPU monitor")}nvidia-smi dmon -s pucvmet -d 1"\n'
            f'echo "{under}# columns: sm%, mem%, clk, temp, power"\n'
            f'echo "{under}# if sm% bounces 0->100->0 across ranks, '
            'MPS may help"\n'
        ) if gpu_ranks else "")
        # What the run starts from -- read at run time from the files beside
        # the script -- and the atoms its deck holds, stated at prep (the
        # status block).
        + f'echo "{lead("Mode")}$_mode"\n'
        + f'echo "{lead("Held atoms")}$_held"\n'
        + f'echo "{lead("Command")}{launch_line}"\n'
        # WHAT THE ENGINE DOES WITH THE WALL: stops before it -- told on its
        # command line, or a script deck from its environment -- or checks
        # no clock and runs to it (`running-a-job.md` § 3.6).
        + f'if [ -n "${{{WALL_VAR}:-}}" ]; then '
          f'echo "{lead("Time limit")}${{{WALL_VAR}}}s from the job\'s start; '
        + (f'{words} stops ${{{WALL_SLACK_VAR}}}s before it ' if takes_wall
           else f'{words} checks no clock and runs to it ')
        + '"; '
          f'else echo "{lead("Time limit")}none stated -- runs until done"; fi\n'
        + f'echo "{lead("Stdout")}$_out_file (live; tail -f to follow)"\n'
        + 'echo "========================================="\n'
        "\n"
    )

    # Per docs/execution/running-a-job.md § 5.2: the run script is a
    # self-contained shell script.  At generate time the generator reads the
    # TARGET's preamble and activation off its record and bakes them
    # VERBATIM into the run script.  At runtime the wrapper does no discovery,
    # no probing, no config-file reads, no env-var-driven behaviour
    # switching.  If anything fails, ``set -euo pipefail`` aborts with the
    # real bash error.  The activation has no default: no record stating
    # one, no wrapper.
    # WHOSE MACHINE IS THIS SCRIPT FOR?
    #
    # (The parameter is `machine_record`, NOT `target_env`: this function
    # already binds a local `target_env` meaning *the conda env NAME to
    # activate*.  Same word, two meanings, one scope.)
    #
    # A run script is generated on one machine and executed on another -- that
    # is what a bundle is for -- and the two enter their environment
    # differently: `module load mamba` + `source activate` on ASU Sol, a
    # `conda.sh` hook on the workstation.  So when the caller names a
    # TARGET, the answer comes off THE TARGET'S RECORD, which is the thing
    # that travels (`scheduler.record.Environment.env_init`).
    #
    # Reading the local config for a remote target is the 2026-08-24
    # failure: `prep --target sol` took Sol's queues and topology from its
    # record and the WORKSTATION's preamble from `molbuilder.json`, so every
    # job on Sol died on `source /home/.../conda.sh`.  Two doors for one
    # fact -- which machine is this for -- answered out of two files.
    # THE RECORD IS WHAT THIS READS -- this machine's too (`configuration.md`
    # § 4): a machine declares its `env_init` in its own `molbuilder.json`,
    # and `jobset probe` copies it into the record it writes.  Prep hands
    # the one it read (A16): read once, never again here.

    env_activation = (
        "# Structured log helper.\n"
        + LOG_FUNCTION
        + f"\n"
        f"# Single unified EXIT cleanup (one trap -- a second ``trap ...\n"
        f"# EXIT`` would REPLACE the first, so all teardown lives here).\n"
        f"# A failure of its own that wrote no conclusion concludes\n"
        f"# here first (project-layout.md 1.6.3); then it stops the\n"
        f"# monitor and cleans (a) the per-rank GPU launcher temp file and\n"
        f"# (b) the MPS daemon + its pipe/log dirs, each a no-op unless\n"
        f"# its variable was set.\n"
        # STOPPING THE MONITOR, in one place: TERM at the job's end
        # (`run-reports.md` § 2) -- and the wait lets it write its closing
        # [STATUS] / [UTIL-SUMMARY] before this shell exits and a scheduler
        # reaps the job's processes.  Bounded (MONITOR_STOP_S): a monitor
        # that does not exit by then is left to its own watch-pid exit.
        f"_mb_stop_monitor() {{\n"
        f'    [ -n "${{_monitor_pid:-}}" ] || return 0\n'
        f'    kill -TERM "$_monitor_pid" 2>/dev/null || return 0\n'
        f"    _mb_w=0\n"
        f'    while kill -0 "$_monitor_pid" 2>/dev/null '
        f'&& [ "$_mb_w" -lt {MONITOR_STOP_S * 5} ]; do\n'
        f"        sleep 0.2\n"
        f"        _mb_w=$((_mb_w + 1))\n"
        f"    done\n"
        f"}}\n"
        f"_mb_cleanup_ran=0\n"
        f"_mb_signalled=0\n"
        f"_mb_cleanup() {{\n"
        # THE SCRIPT'S EXIT CODE, read before any command here replaces it.
        f"    _mb_rc=$?\n"
        # Idempotence guard: a caught signal runs cleanup and exits,
        # which fires the EXIT trap and would run it AGAIN (D17).
        f'    [ "${{_mb_cleanup_ran:-0}}" = "1" ] && return 0 || true\n'
        f"    _mb_cleanup_ran=1\n"
        # A FAILURE OF THE SCRIPT'S OWN CONCLUDES THE RUN (`project-layout.md`
        # § 1.6.3): an activation or a step that failed under `set -e`, an
        # exit before the engine -- the main line never reached its marker,
        # so the marker is written here, with the exit code, and BEFORE the
        # monitor is stopped, which reads it for its closing status, as the
        # main line writes its own first.  A signal leaves none: every
        # signal that would end the script is caught (below), and a forced
        # stop is told by its scheduler and its monitor, never by a marker
        # -- nor by a code 0, which no end the main line did not reach can
        # have.  A dry run starts nothing, and concludes nothing.
        f'    if [ "$_mb_signalled" = "0" ] && [ "$_dry_run" = "0" ] '
        f'&& [ "$_mb_rc" -ne 0 ] && [ ! -e "$_concluded" ]; then\n'
        f"        {_STOPPED_BEFORE_ITS_END} 2>/dev/null || true\n"
        f"    fi\n"
        # ``|| true`` on every arm: this trap runs UNDER set -e, and bash
        # exits on the failure of the command following the final ``&&`` --
        # a monitor already dead (the COMMON case at cleanup) would kill the
        # trap mid-body and skip the MPS teardown below (R9).  Same for a
        # vanished MPS control daemon under pipefail.
        f"    _mb_stop_monitor || true\n"
        f'    [ -n "${{_rank_helper:-}}" ] && rm -f "$_rank_helper" '
        f"2>/dev/null || true\n"
        # BOTH conditions (E-3, 2026-08-13): _mps_started says a daemon
        # was launched, and the pipe dir still being set says OUR pipe is
        # the one the control client will talk through.  The readiness
        # fallback unsets the pipe dirs on its way out, and with them
        # unset `nvidia-cuda-mps-control` addresses the DEFAULT pipe --
        # so the trap could quit a user's GLOBAL daemon this run never
        # started.  A partially-started daemon still gets cleaned up: the
        # fallback path removes the per-job dirs itself before unsetting.
        f'    if [ "${{_mps_started:-0}}" = "1" ] '
        f'&& [ -n "${{CUDA_MPS_PIPE_DIRECTORY:-}}" ]; then\n'
        f"        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 "
        f"|| true\n"
        f'        rm -rf "${{CUDA_MPS_PIPE_DIRECTORY:-}}" '
        f'"${{CUDA_MPS_LOG_DIRECTORY:-}}" 2>/dev/null || true\n'
        f"    fi\n"
        f"}}\n"
        f"trap _mb_cleanup EXIT\n"
        f"_mb_on_signal() {{\n"
        f"    _mb_signalled=1\n"
        # THE LINE GOES TO THE LOG FILE FIRST, AND PIPE IS IGNORED: `_log`
        # writes through the tee, and a kill of the whole process group
        # takes the tee with it -- a write to the dead pipe then kills bash
        # by SIGPIPE before anything else runs, so nothing landed in the
        # session log of a killed point (the 2026-10-08 road walk, plan
        # § 5x.7 F11).  The direct append cannot die that way.
        f"    trap '' PIPE\n"
        f"    printf '{LOG_LINE}\\n' \"$(date '+{LOG_CLOCK}')\" WARN "
        f'"caught SIG$1 -- {STOPPED}" '
        f'>> "$_runwrap_log" 2>/dev/null || true\n'
        f'    _log WARN "caught SIG$1 (a scheduler\'s stop, Ctrl-C, a lost '
        f'terminal) -- cleaning up" 2>/dev/null || true\n'
        f"    _mb_cleanup\n"
        f'    exit $((128 + $(kill -l "$1" 2>/dev/null || echo 15)))\n'
        f"}}\n"
        # EVERY SIGNAL THAT STOPS A RUN, caught: one bash ends on untrapped
        # runs the EXIT trap with the last command's status, which would
        # read as an end of its own.
        f"for _mb_s in {' '.join(_STOPPING_SIGNALS)}; do\n"
        f'    trap "_mb_on_signal $_mb_s" "$_mb_s"\n'
        f"done\n"
        + _SIGNALS_APART
        + f"\n"
        f"_log STAGE \"{WRAPPER_LOG_START}\"\n"
        f'_log INFO "timestamp:  $(date \'+%Y-%m-%d %H:%M:%S %Z\')"\n'
        f'_log INFO "{HOSTNAME_LABEL:<12}$(hostname)"\n'
        f'_log INFO "{USER_LABEL:<12}${{USER:-?}}"\n'
        f'_log INFO "{CWD_LABEL:<12}$(pwd)"\n'
        f'_log INFO "script:     $0"\n'
        f'_log INFO "argv:       $0 $_mb_argv"\n'
        f'_log INFO "log file:   $_runwrap_log"\n'
        + _scheduler_context_block()
        + "\n"
        + env_bootstrap(machine_record, target_env)
    )

    # THE LAUNCH -- the same block for every engine.  The command is run,
    # not exec'd: the CONCLUSION MARKER is the run script's LAST ACT
    # (`project-layout.md` § 1.6.3 -- absent means still running or
    # force-stopped), and `exec` would replace this shell so nothing could
    # run afterwards -- the ending asked, its advice, what a launch again
    # will do, a point's step after its engine.
    launch_block = (
        _resolved_log_block(launch_line, gpu_ranks)
        + _dry_run_block(launch_line, gpu_ranks, words)
        + (_scf_tee_func(*stamp) if stamp is not None else "")
        + "# --- How a run ended, asked of the framework (_mb_ending) ---\n"
        + _ending_question_func(stderr_apart)
        + _monitor_block(names, notify_on_scf,
                         notify_every_hours, notify_channels,
                         notify_report,
                         cores="$_mb_cores", gpu=gpus.uses,
                         member=point is not None)
        + _finish_check_block(finish)
        # THE POINT'S START, AND ITS OWN STEP BEFORE ITS ENGINE, where its
        # kind has one -- a vibration call's coordinates
        # (`execution/runs-of-points.md` § 3, op. 9).
        + _point_block(names, POINT_STARTED, point, finish, script_name)
        # THE RECORD OF WHAT THE ENGINE WILL READ, once every step before
        # it has written what it reads -- a call's plan step its
        # coordinates (`engines/vibration.md` § 5.11 § 4) -- for a deck
        # whose settings are keywords; a script deck writes its own from
        # inside itself, where it can read its objects back.
        + (_effective_parameters_block(
            script_name, engine, calculation, names.stage,
            role=point.role if point is not None else None)
           if eng.keyword_reader is not None else "")
        # THE WALL REACHES THE ENGINE (running-a-job.md § 3.6; siesta.md
        # § 8.9): what is left of the job's wall when the program starts,
        # in the words the catalogue's `wall_slack` item gives this program
        # (`launch`, template.md § 5) -- the deck, written at prep, never
        # carries a wall the launch stated after it.  A script deck reads
        # the wall from its environment, exported above.
        + "# --- Launch the engine + capture exit -------------------\n"
        "# `set +e` lets us inspect the exit code; on a failure the ending\n"
        "# is asked of the framework (_mb_ending), with the advice for its\n"
        "# cause, and what a launch again will do is said.  Then we re-exit\n"
        "# with the engine's code.\n"
        + (f'_scf_timing_log="{names.name(timing_role)}"\n'
           f'_log INFO "scf timing  : per-iteration stamps -> '
           f'$_scf_timing_log"\n' if stamp is not None else "")
        + ("_mb_wall_args=()\n" if prog is not None else "")
        + (f'if [ -n "${{{WALL_VAR}:-}}" ] && [ -n "${{{JOB_STARTED_VAR}:-}}" ]; then\n'
           f"    _mb_wall_left=$(( {WALL_VAR} - ( $(date +%s) - {JOB_STARTED_VAR} ) ))\n"
           f'    [ "$_mb_wall_left" -lt 1 ] && _mb_wall_left=1\n'
           + (f"    _mb_wall_args=({_wall_array})\n" if _wall_words else "")
           + f'    _log INFO "time limit: ${{_mb_wall_left}}s left of ${{{WALL_VAR}}}s; '
           f'{words} stops ${{{WALL_SLACK_VAR}}}s '
           f'before it"\n'
           f"fi\n" if takes_wall else "")
        + "_t_start=$(date +%s.%N)\n"
        "set +e\n"
        + (f'{launch_words} {script_name}'
           + ("" if stderr_apart else " 2>&1")
           + ' | _mb_scf_tee "$_out_file" "$_scf_timing_log"\n'
           if stamp is not None else
           f'{launch_words} {script_name}{to_output}\n')
        + "_engine_exit=${PIPESTATUS[0]}\n"
        "set -e\n"
        "_t_end=$(date +%s.%N)\n"
        '_engine_wall=$(awk -v a="$_t_start" -v b="$_t_end" '
        "'BEGIN{printf \"%.1f\", b-a}')\n"
        # The engine's wall time -- and, where the run script stamps its
        # SCF rows, how many it took, of both phases.  The seconds PER
        # ITERATION are the timing instrument's, one phase at a time
        # (`parse/instruments/scf_timing.py`, `model/parse.md` § 5c.1).
        + ((f'if [ -f "$_scf_timing_log" ]; then\n'
            f'    _n_scf=$(wc -l < "$_scf_timing_log" | tr -d " ")\n'
            f'else\n'
            f'    _n_scf=0   # the engine stopped before any SCF row\n'
            f'fi\n'
            f'case "$_n_scf" in ""|*[!0-9]*) _n_scf=0 ;; esac\n'
            f'_log INFO "{WALL_LINE.format(program=words, seconds="${_engine_wall}")}, '
            f'${{_n_scf}} SCF rows in $_scf_timing_log"\n')
           if stamp is not None else
           f'_log INFO "{WALL_LINE.format(program=words, seconds="${_engine_wall}")}"\n')
        + f"\n"
        f'if [ "$_engine_exit" -ne 0 ]; then\n'
        f"    echo \"\"\n"
        f'    echo "===== {words} exited with code $_engine_exit =====" >&2\n'
        f'    if _mb_ending_able; then\n'
        f'        echo "how it ended: $_mb_ending_words" >&2\n'
        # THE ADVICE FOR ITS CAUSE, where the output's family keeps some --
        # SIESTA's propor: its two causes, in the order to check them
        # (`running-a-job.md` § 3.5, its catalogue row's `stops`).
        + ('        if _mb_advice=$(_mb_ending advice); then\n'
           "            printf '\\n%s\\n\\n' \"$_mb_advice\" >&2\n"
           '        fi\n' if asks_advice else "")
        + f'    else\n'
        f'        _log WARN "{_ENDING_UNREADABLE}"\n'
        f'    fi\n'
        f'    echo "the output: $_out_file -- this run\'s session log: '
        f'$_runwrap_log" >&2\n'
        # WHAT A LAUNCH AGAIN WILL DO, from the kind's fact
        # (`running-a-job.md` § 3.5): this run is one start, and it
        # concludes now.  A stop at the wall -- asked as such, where the
        # output's family names one -- says so first (§ 3.6).
        + (f'    if _mb_ending out-of-time; then\n'
           f'        {_again_line(_again, "stopped at its time limit")}'
           f'    else\n'
           f'        {_again_line(_again)}'
           f'    fi\n'
           if asks_wall else f'    {_again_line(_again)}')
        + _point_block(names, POINT_ENDED, point, finish, script_name,
                       failed=True)
        + '    # CONCLUDED -- an error is a conclusion (project-\n'
        '    # layout.md 1.6.3): the engine returned and this process\n'
        '    # says goodbye with its code.  A forced stop leaves NO\n'
        '    # marker: its signal handler writes none.\n'
        + "    " + conclude_line('"$_engine_exit"') + "\n"
        + '    exit "$_engine_exit"\n'
        "fi\n"
        + f'echo "{words} completed: {launch_line}"\n'
        + _point_block(names, POINT_ENDED, point, finish, script_name)
        + conclude_line(0) + "\n"
    )

    # ----- Script-contract PROVENANCE block -----
    # See docs/execution/job-contracts.md § 3.2.  PROVENANCE only for the
    # run script -- no BENCH-MARKS (run-script parameters are overridden via
    # existing env vars per the contract) and no ATOM-METADATA (lives in
    # the deck, not the run script).  An engine of one process states no
    # rank count: prep refuses one before it reaches here.
    from . import script_emit as _sc
    _resolved_defaults = {
        "target_env":    target_env,
        # THE STATED COUNTS -- unstated, the run was refused at prep, so
        # there is no "auto" to record (2026-10-02).
        "omp_threads": str(omp_threads),
        "max_memory_mb": (
            "n/a" if max_memory_mb is None else str(max_memory_mb)
        ),
        "mpi_np": (str(mpi_np) if ranks
                   else f"n/a ({eng.words} runs one process)"),
    }
    _provenance = _sc.emit_provenance(
        generator_version=_sc.generator_version(),
        generated_at=_sc.generated_at_now(),
        resolved_defaults=_resolved_defaults,
        # The engine this run script launches -- the same answer its deck's
        # own PROVENANCE gives (`script_emit.render_deck`): prep's.
        engine=engine,
    )
    _user_custom = _sc.emit_user_custom_placeholder()

    return (
        f"#!/usr/bin/env bash\n"
        f"{_provenance}\n"
        f"#\n"
        f"# molbuilder run script -- {words} run"
        + (f", -np {int(mpi_np)} as stated" if ranks else "") + "\n"
        f"# Script: {script_name}\n"
        f"# Target env: {target_env}\n"
        f"#\n"
        f"# Generated at prep (`molbuilder jobset prep`).  Edit freely;\n"
        f"# molbuilder will not regenerate this file until the next prep\n"
        f"# of the same script.  `molbuilder jobset launch` starts it,\n"
        f"# once in its run's folder, with its counts, its engine's output\n"
        f"# going to {names.name(stdout_role)} (project-layout.md 1.6.1); it takes\n"
        f"# no other start: "
        f"{started_by_launch(command_stage(names.stage), trial)}.\n"
        f"#\n"
        "# What the deck reads of the restart files launch copies in\n"
        "# (job-contracts.md 4.1):\n"
        + _prior_state_words(eng, resumes, relaunch_continues,
                             _restart_honoured, _reads)
        + f"#\n"
        f"# IMPORTANT: this run script does NOT change cwd (see\n"
        f"# docs/execution/running-a-job.md § 2.2a).  Its own files (the\n"
        f"# session log, the output) land in the CALLER'S current working\n"
        f"# directory.  Sent to a scheduler this is the folder it was\n"
        f"# sent from; run here, the folder `jobset launch` starts it\n"
        f"# in, its run's own.  The engine's files land in the same\n"
        f"# directory: a program writes where it runs, and a script deck\n"
        f"# writes beside itself, which this run script runs by its bare\n"
        f"# name from here (docs/execution/job-contracts.md § 4.2).\n"
        f"#\n"
        f"# It runs in its run's folder, run-<n>/, where launch put it\n"
        f"# beside its deck and inputs and where the run's outputs belong\n"
        f"# (project-layout.md 1.5).  Do NOT add ``cd`` lines or modify the\n"
        f"# cwd from inside the deck -- code layered on top would write into\n"
        f"# the wrong place if you chdir to elsewhere.\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"\n"
        # HELP IS ANSWERABLE BY ANYONE: the arguments are read first, so
        # asking what a script does never meets the launch-door gate (U10,
        # 2026-08-12), and a dry run is known before the one-start guard.
        f"{args_parse}"
        # THE WALL, counted from the job's start (running-a-job.md § 3.6):
        # a job's header said when it started; a run here says it now.  The
        # slack travels to the engine beside it -- a program on its command
        # line below, a script deck from its environment.
        f"# The time limit: the launch told this job the seconds it was sent with\n"
        f"# ({WALL_VAR}); its start is the header's, or now; "
        + (f"the engine stops\n# itself {WALL_SLACK_VAR} before the time limit "
           if takes_wall else
           "this engine checks\n# no clock and runs to the time limit ")
        + "(running-a-job.md 3.6).\n"
        f': "${{{JOB_STARTED_VAR}:=$(date +%s)}}"\n'
        f"export {JOB_STARTED_VAR}\n"
        f"{_slack_export}"
        f"\n"
        + _one_start_block(names, trial)
        + f"\n"
        f"# --- Launch-door gate (one door: jobset launch) ----------\n"
        f"# `molbuilder jobset launch` starts this script, and says so in\n"
        f"# {LAUNCH_CLAIM_VAR} (direct: the child's environment; sent to a\n"
        f"# scheduler: on the line it is sent with).  It starts nothing\n"
        f"# else: a run's records are its launch's, and a start without\n"
        f"# them would be read by nothing (job-contracts.md 2.6).  A help\n"
        f"# request and a dry run start nothing, and pass.\n"
        f'if [ "$_dry_run" = "0" ] '
        f'&& [ "${{{LAUNCH_CLAIM_VAR}:-}}" != "{LAUNCH_CLAIM}" ]; then\n'
        f'    echo "ERROR: {run_sh} was not started by its launch: '
        f'{started_by_launch(command_stage(names.stage), trial)}." >&2\n'
        f'    echo "launched-by: ${{{LAUNCH_CLAIM_VAR}:-NONE}} -- refused at '
        f'the launch-door gate"\n'
        f"    exit 2\n"
        f"fi\n"
        f"# --- The run's session log (running-a-job.md 2.2a) ------\n"
        f"# It opens once the gate lets a start through, and only for a\n"
        f"# start: a help request, a dry run and a refused start write\n"
        f"# none, so the folder stays free for the start it was opened for\n"
        f"# (job-contracts.md 2.6).\n"
        f'if [ "$_dry_run" = "0" ]; then\n'
        f'    exec > >(tee -a "$_runwrap_log") '
        f'2> >(tee -a "$_runwrap_log" >&2)\n'
        f"else\n"
        f"    _runwrap_log=/dev/null\n"
        f"fi\n"
        f"\n"
        f"# The verdict, in the JOB'S OWN OUTPUT: sent to a scheduler this line is in\n"
        f"# the job's .out, and for a start it is in the session log too; a\n"
        f"# dry run passes the gate with no claim, and says so.\n"
        f'echo "launched-by: ${{{LAUNCH_CLAIM_VAR}:-NONE}}"\n'
        f"# Per docs/execution/running-a-job.md § 2.2a: the run script does NOT change cwd.\n"
        f"# A scheduler lands the job in the folder it was sent from; run\n"
        f"# here, `jobset launch` starts it in the run's folder.  The\n"
        f"# caller's cwd is the contract -- outputs (the session log, the output)\n"
        f"# land where the run script was invoked.\n"
        f"\n"
        f"{env_activation}"
        f"{env_prefix}"
        f"{launch_block}"
        f"\n{_user_custom}\n"
    )


#: THE MONITOR, beside a job, as ``{name beside the job: the module whose
#: source it is}`` -- the one member named apart, ``mb_monitor.py``.  Every
#: other member travels because a member states it needs it beside it (its
#: ``TRAVELS_WITH``: the monitor its own imports, each of those theirs) or a
#: record names it (:func:`_members`): no module is listed here by hand.
#:
#: **A verbatim copy of each module's own file**, not a copy of its logic:
#: each imports the next two ways -- from the package, or from beside the job
#: -- as `config_dir` always has, so the monitor reads a run with the readers
#: the Results tab's directory door reads it with.  The rule for joining is
#: `configuration.md` § 2.3's: *stdlib-only AND travels*, and
#: `tests/test_monitor_watches_a_live_run_e2e.py` runs the whole set with molbuilder
#: absent.
#:
#: They travel as ONE file, :data:`MONITOR_BUNDLE`, which `render_wrappers`
#: writes next to the wrapper and `materialize` brings into every attempt.
_MONITOR_BASE: Dict[str, str] = {"mb_monitor.py": "molbuilder.monitor"}


def _members() -> Dict[str, str]:
    """EVERY MEMBER BUT THE MONITOR -- what the monitor imports beside the job
    (its ``TRAVELS_WITH``), and the members the records name
    (`execution/architecture.md` § 3.3):
    each kind's words, the run-of-points parts its pair names (the kind
    table's ``points.count`` and ``points.outcome`` -- a vibration's count, a
    transport rung's count and a round's outcome), and each engine's
    keyword reader, which reads its deck for its run script (the engine
    table's ``keyword_reader``, `mb_monitor.pyz deck`), and the readers of
    each output's role a job asks -- how it ended, its stops, its live pass,
    its loop's words, its timing (the file catalogue row's
    ``reads``, `runfiles.READ_BESIDE_A_JOB`) -- each with every module it
    says it needs beside it (its own ``TRAVELS_WITH``).  So a new kind,
    engine or output travels by its record, never an edit of this module."""
    import importlib
    from .engine_table import engine as _engine, engines
    from .kind_table import kind as _kind, kinds, run_of_points
    from .runfiles import READ_BESIDE_A_JOB, ON_THE_LABEL
    refs = [ref for k in kinds() for e in _kind(k).engines
            for points in [run_of_points(k, e)] if points is not None
            for ref in (points.count, points.outcome)]
    refs += [_engine(e).keyword_reader for e in engines()
             if _engine(e).keyword_reader]
    refs += [ref for a in ON_THE_LABEL for question, ref in a.reads
             if question in READ_BESIDE_A_JOB]
    todo = [str(ref).partition(":")[0] for ref in refs]
    # ...AND WHAT THE MONITOR ITSELF IMPORTS beside the job, by its own word.
    todo += list(importlib.import_module(
        _MONITOR_BASE["mb_monitor.py"]).TRAVELS_WITH)
    out: Dict[str, str] = {}
    while todo:
        module = todo.pop()
        name = module.rsplit(".", 1)[-1] + ".py"
        held = out.get(name) or _MONITOR_BASE.get(name)
        if held == module:
            continue
        if held is not None:
            # TWO MODULES, ONE NAME BESIDE THE JOB: the bundle holds one
            # file of each name, so the second would be the first's copy.
            raise WrapperError(
                f"{module} and {held} both travel as {name} in "
                f"{MONITOR_BUNDLE}; a module the records name travels "
                f"under its own name, which another already holds.")
        out[name] = module
        todo += list(getattr(importlib.import_module(module),
                             "TRAVELS_WITH", ()))
    return out


@functools.lru_cache(maxsize=1)
def monitor_companions() -> Dict[str, str]:
    """EVERY MODULE THAT TRAVELS IN :data:`MONITOR_BUNDLE`, ``{name beside
    the job: the module whose source it is}`` -- the monitor and the
    framework's readers it imports, and the members the records name
    (:func:`_members`)."""
    return {**_MONITOR_BASE, **_members()}


#: THE ONE FILE that travels beside every job for its monitor: a Python zip
#: application (PEP 441) holding every module of :func:`monitor_companions`,
#: each its own file verbatim under its shipped name, and a ``__main__`` that
#: runs the monitor.  ``"$_mb_py" mb_monitor.pyz ...`` is the monitor;
#: ``"$_mb_py" mb_monitor.pyz ending OUTPUT ...`` is `_run_ending`'s door, which
#: `monitor.main` hands on.  One file holds them all (user: *"can't we just
#: put it in one single Python file?"*): still the package's own files, so still no second copy of any
#: reader.
from .runfiles import MONITOR_BUNDLE  # noqa: E402 -- the catalogue's name
from .monitor import BUNDLE_VERBS  # noqa: E402 -- the bundle's own verbs
from .report_cards import FILENAME as _CARDS_FILE, SHIPPED as _CARDS_SHIPPED  # noqa: E402

#: The bundle's ONE DATA MEMBER (`run-reports.md` § 4.1c): the chat card's
#: layout, the package's own file under its shipped name, beside its reader
#: `report_cards.py`, which finds it there through ``__loader__.get_data``.
#: ``{name beside the job: the package file}`` -- the path is the owner's.
MONITOR_DATA: Dict[str, Path] = {_CARDS_FILE: _CARDS_SHIPPED}

#: What the wrapper's log says when the ending cannot be asked: no bundle,
#: a bundle that does not load on the job's python, or a reading of the
#: output that stops -- the last two's reason printed once, by the one ask
#: (`_mb_ending_able`).
_ENDING_UNREADABLE = (f"the ending cannot be read here ({MONITOR_BUNDLE} "
                      f"is not beside the job, does not load on its python, "
                      f"or could not read the output): no failure hint")

#: The bundle's entry: the monitor, whose exit status is the process's --
#: `_run_ending`'s questions answer by it.  A bundle this python cannot
#: import answers `ending` with 2, *cannot read* (`job-contracts.md` § 2.6),
#: never 1, which would say *no*.
#:
#: IT SAYS ITS LOAD, in the session log's own line (`run-reports.md` § 2.6):
#: ``monitor: starting ...`` before the import and ``... started`` after it,
#: so a ``starting`` with no ``started`` is a monitor that died loading -- and
#: the error follows it, one line and the traceback.  The verbs that are no
#: monitor print no pair (`monitor.BUNDLE_VERBS`): `ending`, whose answer
#: the run script waits on and reads by its exit status, `deck`, the deck as
#: its engine reads it, and the run's own steps, `notify` and a point's
#: report steps, `point` (`execution/runs-of-points.md` § 3).
#:
#: WRITTEN IN THE PYTHON EVERY INTERPRETER PARSES -- no f-strings, no print
#: function -- because its whole job is to report a python that cannot load
#: the members, and it has to parse on that python to say so.
_BUNDLE_MAIN = (
    "import sys, time\n"
    "def _say(level, message):\n"
    f"    sys.stderr.write({(LOG_LINE + chr(10))!r} % "
    f"(time.strftime({LOG_CLOCK!r}), level, message))\n"
    "    sys.stderr.flush()\n"
    f"_asked = sys.argv[1:2] in {[[v] for v in BUNDLE_VERBS]!r}\n"
    "if not _asked:\n"
    f"    _say('INFO', 'monitor: starting {MONITOR_BUNDLE} on python %s (%s)'"
    " % (sys.version.split()[0], sys.executable))\n"
    "try:\n"
    "    import mb_monitor\n"
    "except Exception as _e:\n"
    "    import traceback\n"
    f"    _say('ERROR', 'monitor: {MONITOR_BUNDLE} did not load -- %s: %s'"
    " % (type(_e).__name__, _e))\n"
    "    traceback.print_exc()\n"
    "    raise SystemExit(2 if _asked else 1)\n"
    "if not _asked:\n"
    f"    _say('INFO', 'monitor: {MONITOR_BUNDLE} started')\n"
    "raise SystemExit(mb_monitor.main())\n")


def monitor_bundle() -> bytes:
    """The bytes of :data:`MONITOR_BUNDLE` -- each module's own file, read
    (:func:`companion_source`), zipped beside the entry.  The same sources
    give the same bytes (fixed member times), so a re-prep writes the file it
    found."""
    return _zip_bundle(_BUNDLE_MAIN, monitor_companions(), MONITOR_DATA)


def _zip_bundle(main: str, companions: Dict[str, str],
                data: Optional[Dict[str, Path]] = None) -> bytes:
    """ONE builder for every bundle that travels beside a job: ``main`` as
    the zip application's ``__main__.py``, then each of ``companions`` as its
    module's own file, read (:func:`companion_source`), then each of
    ``data`` -- a file that is not a module, read as text -- under its
    member name."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        members = [("__main__.py", main)] + [
            (name, companion_source(name, companions)) for name in companions]
        for name, path in (data or {}).items():
            try:
                members.append((name, path.read_text(encoding="utf-8")))
            except OSError as exc:
                raise WrapperError(
                    f"could not read {name!r} to ship beside the job: "
                    f"{exc}") from None
        for name, text in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, text)
    return buf.getvalue()


def companion_source(name: str,
                     companions: Optional[Dict[str, str]] = None) -> str:
    """The text that travels as ``name`` -- its module's own file, read --
    from ``companions`` (the monitor's by default).

    Read rather than restated: *"a comment is not a mechanism"*.

    **Returned as text rather than copied to a destination.**  Step 4 is on
    floor 3, and floor 3 does not touch the disk (`script-preparation.md` § 5,
    W7).
    """
    import importlib
    table = monitor_companions() if companions is None else companions
    try:
        module = importlib.import_module(table[name])
        return Path(module.__file__).read_text(encoding="utf-8")
    except (KeyError, ImportError, OSError) as exc:
        raise WrapperError(
            f"could not read {name!r} to ship beside the job: {exc}") from None


#: Every file that travels beside each call of a SIESTA vibration for its
#: steps and its FINISH (`engines/vibration.md` § 5.11 § 3), as ``{name beside the job: the module
#: whose source it is}``: the SIESTA route (`spectra.siesta_vibration`, whose
#: ``main`` the bundle runs), the engine-neutral analysis and its math, the
#: result's class, its activity classes and its one writer, the Methods
#: prose, and every reader the finish reads the attempt through -- the fdf
#: reader and the unit words it reads with, the deck's block reader
#: (`deck_record`), the ``.FC`` reader and its
#: error, SIESTA's reading pass with its grammar and rule engine, the
#: permutation record, the one door for engine atom numbering, the session
#: log's line with the run-file grammar it is built on (`runfiles`), the
#: commands a remedy names (`identity`: the launch and the way back, as a
#: text read later says them) -- and the constants all of them convert
#: with.
#:
#: **Each module's own file, imported two ways** (package, or beside the
#: job), like the monitor's; unlike the monitor's, the set needs **numpy
#: and netCDF4**, which the SIESTA job envs carry for it (`envs/recipes.py`):
#: the rule for joining is *imports only the standard library, numpy,
#: netCDF4 and the other members* (`engines/vibration.md` I23).  The SIESTA end-to-end run proves the set whole: its
#: job's python cannot import molbuilder (`tests/test_siesta_vibration_e2e.py`).
VIBRATION_COMPANIONS: Dict[str, str] = {
    "siesta_vibration.py":     "molbuilder.spectra.siesta_vibration",
    "vibrational_analysis.py": "molbuilder.spectra.vibrational_analysis",
    "normal_modes.py":         "molbuilder.spectra.normal_modes",
    "results.py":              "molbuilder.spectra.results",
    "activity.py":             "molbuilder.spectra.activity",
    "derived.py":              "molbuilder.spectra.derived",
    "methods.py":              "molbuilder.spectra.methods",
    "spectra_sidecar.py":      "molbuilder.sidecars.spectra",
    "fdf.py":                  "molbuilder.parse.fdf",
    "units.py":                "molbuilder.units",
    "deck_record.py":          "molbuilder.deck_record",
    "siesta_fc.py":            "molbuilder.parse.engines.siesta_fc",
    "errors.py":               "molbuilder.parse.errors",
    "siesta_reader.py":        "molbuilder.parse.engines.siesta_reader",
    "siesta_grammar.py":       "molbuilder.parse.engines.siesta_grammar",
    # AN OPTICAL CALL'S SPECTRUM, read by its one reader, and the optical
    # calls' data file (`engines/vibration.md` § 5.11 § 8).
    "siesta_epsimg_reader.py": "molbuilder.parse.engines.siesta_epsimg_reader",
    "optical_data.py":         "molbuilder.spectra.optical_data",
    "transiesta.py":           "molbuilder.parse.engines.transiesta",
    "_section_rules.py":       "molbuilder.parse.engines._section_rules",
    "molwatch_grammar.py":     "molbuilder.parse.engines.molwatch_grammar",
    "emitter.py":              "molbuilder.trajectory_log.emitter",
    "keys.py":                 "molbuilder.trajectory_log.keys",
    "atom_permutation.py":     "molbuilder.atom_permutation",
    # THE STRUCTURE, WHOLE, in the result (engines/vibration.md § 6.10): the
    # result type builds and checks it through the one serialiser.
    "structure.py":            "molbuilder.structure",
    "structure_pair.py":       "molbuilder.structure_pair",
    # THE RESULT'S TWO FILES (engines/vibration.md § 6.10.5): the data
    # file's writer, the variable record it shares with the transport data
    # file, and the one atomic writer both files go through.
    "spectra_datafile.py":     "molbuilder.spectra.datafile",
    "ncdata.py":               "molbuilder.ncdata",
    "persist.py":              "molbuilder.persist",
    "engine_atom_index.py":    "molbuilder.engine_atom_index",
    "wrapper_log.py":          "molbuilder.wrapper_log",
    "runfiles.py":             "molbuilder.runfiles",
    "identity.py":             "molbuilder.identity",
    "constants.py":            "molbuilder.constants",
    # A RUN OF POINTS (`engines/vibration.md` § 5.11): the force constants'
    # record step makes each optical call's folder and its files through
    # the same doors this machine's launch uses -- the plan, the place record and the names' rules, the
    # copier -- and picks the modes with PySCF's own selector.
    "planned.py":              "molbuilder.jobset.planned",
    "calcdirs.py":             "molbuilder.calcdirs",
    "paths.py":                "molbuilder.paths",
    "selection.py":            "molbuilder.spectra.selection",
    "deck_lines.py":           "molbuilder.siesta.deck_lines",
    "vibration_phases.py":     "molbuilder.siesta.vibration_phases",
    # THE RUN'S INDEX, which the steps read and write (`runpoints`).
    "runpoints.py":            "molbuilder.runpoints",
    "runrecord.py":            "molbuilder.runrecord",
    # THE RELAXATION RECORD'S ONE FUNCTION (`parse/contract.py`, the
    # relax call's record in the index); the keys it reads are a member
    # above.
    "contract.py":             "molbuilder.parse.contract",
    # A CALL'S VERDICT AND ITS DONE RULE, the status door's own
    # (`_run_ending.verdict_of`, `not_done_why`; § 5.11 § 6), with the end
    # lines its dispatch imports.
    "_run_ending.py":          "molbuilder.parse.engines._run_ending",
    "end_lines.py":            "molbuilder.pyscf.end_lines",
}

#: THE ONE FILE that is a SIESTA vibration's Python: a Python zip
#: application of :data:`VIBRATION_COMPANIONS` whose ``__main__`` runs
#: `spectra.siesta_vibration.main` -- its verbs ``plan``, ``record`` and
#: ``finish-run`` are the vibration's own steps, which each call's report
#: steps run in their place (the kind table's ``steps``,
#: `execution/runs-of-points.md` § 1.2): before a call's engine, after it,
#: and the run's finish once no call is left, which writes the result at
#: the run's root (`engines/vibration.md` § 5.11 § 3).  Every phase's deck
#: names it (`DeckSpec.finish`), `prep` copies that onto the job
#: (`Job.finish`) and writes it beside each phase's run script, and every
#: call is brought a copy.
from .runfiles import VIBRATION_BUNDLE  # noqa: E402 -- the catalogue's name

#: The finish bundle's entry.  A set that cannot load on the job's python --
#: most often an env without numpy -- says so in the session log's own
#: line, with the traceback, and exits 1: the job then fails, because the
#: spectrum IS this stage's result (`engines/vibration.md` § 5.5).
_VIBRATION_MAIN = (
    "import sys, time\n"
    "try:\n"
    "    import siesta_vibration\n"
    "except Exception as _e:\n"
    "    import traceback\n"
    f"    sys.stderr.write({(LOG_LINE + chr(10))!r} % "
    f"(time.strftime({LOG_CLOCK!r}), 'ERROR', "
    f"'vibration: {VIBRATION_BUNDLE} did not load -- %s: %s (it needs "
    f"numpy and netCDF4 in the job env: envs/recipes.py)' "
    "% (type(_e).__name__, _e)))\n"
    "    traceback.print_exc()\n"
    "    raise SystemExit(1)\n"
    # `loads`: the wrapper's question before the engine starts -- does the
    # finish load on this python?  Answered by the import above.
    "if sys.argv[1:2] == ['loads']:\n"
    "    raise SystemExit(0)\n"
    "raise SystemExit(siesta_vibration.main())\n")


def vibration_bundle() -> bytes:
    """The bytes of :data:`VIBRATION_BUNDLE`, from the one builder."""
    return _zip_bundle(_VIBRATION_MAIN, VIBRATION_COMPANIONS)


#: What a job's ``finish`` may name, and the builder of each -- the one
#: lookup `render_wrappers` ships a finish bundle by.
_FINISH_BUNDLES = {VIBRATION_BUNDLE: vibration_bundle}


#: Every file the PySCF script IMPORTS molbuilder's code from
#: (`engines/pyscf.md` § 3), as ``{name beside the job: the module whose
#: source it is}``: the run's thread and GPU set-up, its threads sized
#: before numpy is imported (`runtime_info`, standard library only); the
#: progress-log writer and the unit factors it writes with; the structure
#: codec, with the structure module (its one XYZ reader among it), the order
#: record its ``atom_order`` field holds (`model/overview.md` § 2.2a) and the
#: sidecar module its ``write_moved`` reaches; the relaxation; and a
#: vibration's rules -- the PySCF route's, the harmonic path and
#: thermochemistry, the structure the result states and the result's writer
#: (its JSON and its data file) -- and its mode selector.
#: The script imports them by these names -- `pyscf.input.emit_bundle_imports`
#: reads them here.
#:
#: **Each module's own file, imported two ways**, like the monitor's and the
#: finish's; the rule for joining is *imports the standard library, numpy
#: and netCDF4 at load (netCDF4 in `ncdata`, the data-file framework), and
#: in the functions the script calls also ASE -- which the PySCF env carries,
#: with netCDF4, for it (`envs/recipes.py`) -- and PySCF itself, with gpu4pyscf and cupy for a run on the GPU; nothing else but
#: the other members*.
PYSCF_COMPANIONS: Dict[str, str] = {
    "runtime_info.py":          "molbuilder.runtime_info",
    "molwatch_emitter.py":      "molbuilder.trajectory_log.emitter",
    "keys.py":                  "molbuilder.trajectory_log.keys",
    "constants.py":             "molbuilder.constants",
    "workingcopy_structure.py": "molbuilder.workingcopy_structure",
    "structure.py":             "molbuilder.structure",
    "atom_permutation.py":      "molbuilder.atom_permutation",
    "engine_atom_index.py":     "molbuilder.engine_atom_index",
    "molstruct.py":             "molbuilder.sidecars.molstruct",
    "structure_pair.py":        "molbuilder.structure_pair",
    "relax_policy.py":          "molbuilder.pyscf.relax_policy",
    "pyscf_vibration.py":       "molbuilder.spectra.pyscf_vibration",
    "normal_modes.py":          "molbuilder.spectra.normal_modes",
    "spectra_sidecar.py":       "molbuilder.sidecars.spectra",
    # THE RESULT'S RECORD, one with the SIESTA finish's: the script builds its
    # whole shape and each mode by it (`engines/vibration.md` § 6.5).
    "results.py":               "molbuilder.spectra.results",
    # THE RESULT'S TWO FILES (engines/vibration.md § 6.10.5): the data
    # file's writer, the variable record it shares with the transport data
    # file, the one atomic writer both files go through, and the catalogue
    # the data file's name follows from.
    "spectra_datafile.py":      "molbuilder.spectra.datafile",
    "ncdata.py":                "molbuilder.ncdata",
    "persist.py":               "molbuilder.persist",
    "runfiles.py":              "molbuilder.runfiles",
    # THE DERIVED VALUES, stated by the one writer at every write
    # (`spectra.derived`, `engines/vibration.md` § 6.6), and the activity
    # rule they ask.
    "derived.py":               "molbuilder.spectra.derived",
    "activity.py":              "molbuilder.spectra.activity",
    "mode_selection.py":        "molbuilder.spectra.selection",
}

#: THE ONE FILE the PySCF script imports molbuilder's code from: a Python
#: zip of :data:`PYSCF_COMPANIONS`.  `render_wrappers` writes it beside every
#: PySCF script and `materialize` brings it into every attempt; the script
#: puts it on its import path on its first lines.
from .runfiles import PYSCF_BUNDLE  # noqa: E402 -- the catalogue's name

#: Its entry, for a person who runs it: the bundle is imported, not run.
_PYSCF_MAIN = (
    "import sys\n"
    f"sys.stderr.write({PYSCF_BUNDLE!r} + ' is not run: the PySCF script "
    "beside it imports molbuilder code from it\\n')\n"
    "raise SystemExit(2)\n")


def pyscf_bundle() -> bytes:
    """The bytes of :data:`PYSCF_BUNDLE`, from the one builder."""
    return _zip_bundle(_PYSCF_MAIN, PYSCF_COMPANIONS)


def bundles_for(engine: str, calculation: str,
                finish: Optional[str] = None
                ) -> Tuple[Tuple[str, Callable[[], bytes]], ...]:
    """The bundles that travel beside a deck, each ``(its name, its
    builder)``: `render_wrappers` writes them beside the deck, and
    `materialize` brings them into every attempt with the rest of what
    travels (`runfiles.beside_the_deck`).

    The file catalogue's travelling rows for the deck's engine and kind that
    are BUILT here (`runfiles.travelling`) -- the monitor's
    (:data:`MONITOR_BUNDLE`), and beside a PySCF script the molbuilder code
    it imports (:data:`PYSCF_BUNDLE`, `engines/pyscf.md` § 3) -- and the
    job's finish when it has one (`Job.finish`, `engines/vibration.md`
    § 5.5)."""
    from .runfiles import travelling
    builders = {MONITOR_BUNDLE: monitor_bundle, PYSCF_BUNDLE: pyscf_bundle,
                **_FINISH_BUNDLES}
    names = [a.name for a in travelling(engine, calculation)
             if a.name in builders]
    if finish is not None:
        names.append(finish)
    return tuple((name, builders[name]) for name in names)


@dataclass(frozen=True)
class RenderedWrapper:
    """**Step 4's product, before anything reaches the disk** —
    `script-preparation.md` § 5, W7.

    Named texts in the order they are written, plus which of them the shell
    must be able to execute, and the binary files beside them: the bundles
    -- the monitor with its readers (:data:`MONITOR_BUNDLE`), a job's
    finish, the code a PySCF script imports.  The wrapper is always
    ``files[0]``: it is what step 4 exists to produce, and the ``.sbatch`` and
    the bundle are things it needs beside it.

    **Why a set rather than one string.**  A deck is one file; a wrapper comes
    with others, and which of them exist depends on facts only this layer
    holds -- whether the machine has a queue.  Returning the set keeps those
    decisions on floor 3 where they are made, and leaves the disk to the
    caller.
    """
    files: Tuple[Tuple[str, str], ...]
    executable: Tuple[str, ...] = ()
    #: Named BYTES -- the bundles, which are zips and no text.
    blobs: Tuple[Tuple[str, bytes], ...] = ()
    #: WHAT THE RENDERING SAYS AND DOES NOT REFUSE (`gpu.md` G6): sentences
    #: for prep's answer, printed with its other notes.
    notes: Tuple[str, ...] = ()

    @property
    def wrapper_name(self) -> str:
        return self.files[0][0]


def render_wrappers(script_path: Path, *,
                    names,
                    engine: str,
                    calculation: str,
                    resources: "Resources",
                    program: Optional[str] = None,
                    env: Optional[str] = None,
                    emit_sbatch: bool = True,
                    machine_record,
                    domain_pq: Optional[Tuple[str, str]] = None,
                    finish: Optional[str] = None,
                    resumes: bool,
                    relaunch_continues: bool,
                    trial: bool,
                    warm: Optional[Sequence[str]] = None,
                    deck_text: Optional[str] = None,
                    point: Optional[PointOfRun] = None) -> RenderedWrapper:
    """Render everything step 4 produces for *script_path*, and write nothing.

    ``deck_text`` is the deck as it will be written, for one `prep` has
    planned and not yet written (`jobset.planned`) -- the wrapper is
    rendered from it, and the file need not exist yet.  ``domain_pq`` is the
    queue a run was placed on at prep -- its job's placement, which the
    header renders rather than binding the name again (`job-system.md`
    § 6.0).

    **W7 — floor 3 returns text.**  The deck writers hand back a string and the
    conductor writes it, and step 4 does the same, so *"what would a run of
    this deck look like?"* is answerable without producing files.

    **The allocation arrives whole** — `execution/architecture.md` § 3.1, rule
    A8.  With one object there is no subset to choose, and it is unpacked ONCE here so the
    two renderings cannot be given different answers (A9).

    **``omp_threads`` is not a parameter.**  `job-contracts.md` § 6.2 keeps the
    two names distinct because different layers read them: ``cpus_per_task`` is
    what the scheduler is asked for, and the launcher's OMP default is derived
    from it here, in one place, rather than supplied twice by callers who may
    agree.

    ``env`` and ``emit_sbatch`` stay loose because neither is a per-job fact:
    ``env`` is a per-invocation override (``prep --env``) and ``emit_sbatch``
    is a surface's choice about what to write.  The test is ownership — a field
    with a home in § 3's table arrives in that home or not at all.

    ``engine`` is the calculation's (`JobSet.engine`), whose record decides
    every block of the run script (:func:`render_run_wrapper`) and whether
    the header asks one task (``one_process``).

    Both shell texts go through ``bash -n`` (parse-only, no execution) before
    they are returned, so a caller never receives malformed shell to write.
    That gate exists because the 2026-06-20 pentanedithiol incident shipped an
    unterminated quote and the user found out by running it.
    """
    script_path = Path(script_path).resolve()
    if deck_text is None:
        if not script_path.is_file():
            raise WrapperError(f"script not found: {script_path}")
        # THE DECK AS ITS ENGINE READS IT, a block read from a file beside
        # it inlined where its settings are keywords
        # (`engine_table.read_deck`).
        from .engine_table import read_deck
        deck_text = read_deck(engine, script_path)
    r = resources
    _notes: list = []
    text = render_run_wrapper(
        script_path, names=names, engine=engine, calculation=calculation,
        resources=r,
        program=program, env=env,
        machine_record=machine_record,
        finish=finish, resumes=resumes,
        relaunch_continues=relaunch_continues, trial=trial, warm=warm,
        deck_text=deck_text, notes=_notes, point=point)
    _validate_rendered_wrapper(text, script_path)
    # THE STAGE'S NAMES NAME IT (`runfiles.RunNames`), as every file of it.
    files = [(names.run_script, text)]

    # THE MONITOR TRAVELS WITH EVERY JOB, and the framework it reads the run
    # through with it: it runs under the job's own python
    # (`running-a-job.md` § 4.1, `run-reports.md` § 2.3).
    #
    # ONE FILE PER BUNDLE, the catalogue's travelling rows that have a
    # builder (`bundles_for`, `runfiles.travelling`) -- the rows `materialize`
    # brings into the attempt (`runfiles.beside_the_deck`).
    # `render_run_wrapper` has refused
    # a finish it cannot ship and a suffix it cannot run.
    blobs = tuple((name, build()) for name, build
                  in bundles_for(engine, calculation, finish))

    # The submission layer (`scheduler.md` § 6a): a ``.sbatch`` only when the
    # machine has a queue -- every value in it the job's own (its resources,
    # its GPU request among them).
    if emit_sbatch:
        from .engine_table import engine as _engine
        sbatch = _render_sbatch_for(
            script_path, names=names, resources=r,
            domain_pq=domain_pq, machine_record=machine_record,
            one_process=_engine(engine).one_process)
        if sbatch is not None:
            _validate_rendered_wrapper(sbatch, script_path)
            files.append((names.header, sbatch))

    return RenderedWrapper(files=tuple(files), executable=(files[0][0],),
                           blobs=blobs, notes=tuple(_notes))


def _warm_in_effect(engine: str, warm: Optional[Sequence[str]]
                    ) -> Tuple[str, ...]:
    """The restart files this script knows: ``warm``, the list in effect for
    its calculation that ``prep`` hands in (`warmfiles.warm_list`,
    `job-contracts.md` § 4.2a), or -- for a script rendered with no
    calculation behind it -- the engine's own file, every section: the same
    door asked with no folder.  PySCF's are SUFFIXES rather than extensions
    -- ``_optimized.xyz`` -- which no plain extension match would catch."""
    if warm is not None:
        return tuple(warm)
    from .warmfiles import warm_list
    return warm_list(engine).suffixes


def write_run_wrapper(script_path: Path, *,
                      names,
                      engine: str,
                      calculation: str,
                      resources: "Resources",
                      program: Optional[str] = None,
                      env: Optional[str] = None,
                      emit_sbatch: bool = True,
                      machine_record,
                      domain_pq: Optional[Tuple[str, str]] = None,
                      finish: Optional[str] = None,
                      resumes: bool,
                      relaunch_continues: bool,
                      trial: bool,
                      warm: Optional[Sequence[str]] = None,
                      plan=None, notes: Optional[list] = None,
                      point: Optional[PointOfRun] = None) -> Path:
    """Write what :func:`render_wrappers` produced, and return the wrapper's path.

    ``plan`` (`jobset.planned.Plan`) receives the files instead of the disk,
    rendered from the deck the plan holds: `prep` decides everything before
    it writes (`job-system.md` § 5.0).  ``notes`` receives what the
    rendering said (`RenderedWrapper.notes`), each sentence once.

    **This function renders nothing.**  It is the writing half of step 4, kept
    beside the rendering half so every writer of a wrapper set -- `jobset/prep`
    is the one today -- writes it the same way rather than deciding for itself
    what a wrapper needs beside it.

    **Through the one writer** (`script-preparation.md` § 3.2, W4), which keeps
    the reader's USER-CUSTOM block.  The wrapper EMITS that block — it invites
    a person to put their own lines in it — and an invitation the next prep
    silently revokes is worse than no invitation.

    Modes: the wrapper is 0o755, a program its launch runs (and a person
    asks for ``-h`` or ``--dry-run``); the ``.sbatch`` is 0o644, read by
    its scheduler rather than run.
    Overwrites whatever is there.
    """
    from . import script_emit as _sc_write
    from .engine_table import read_deck
    rendered = render_wrappers(script_path, names=names, engine=engine,
                               calculation=calculation,
                               resources=resources, program=program,
                               machine_record=machine_record,
                               domain_pq=domain_pq,
                               env=env, emit_sbatch=emit_sbatch,
                               finish=finish,
                               resumes=resumes,
                               relaunch_continues=relaunch_continues,
                               trial=trial, warm=warm,
                               point=point,
                               # THE DECK AS THE PLAN HOLDS IT, as its
                               # engine reads it.
                               deck_text=(read_deck(engine, script_path,
                                                    read=plan.read_text)
                                          if plan is not None else None))
    if notes is not None:
        notes.extend(n for n in rendered.notes if n not in notes)
    parent = Path(script_path).resolve().parent
    for name, text in rendered.files:
        _sc_write.write_script(
            parent / name, text, plan=plan,
            mode=0o755 if name in rendered.executable else 0o644)
    for name, data in rendered.blobs:
        if plan is not None:
            plan.bytes(parent / name, data, mode=0o644)
        else:
            from .persist import write_bytes
            write_bytes(parent / name, data, mode=0o644)
    return parent / rendered.wrapper_name


def _render_sbatch_for(script_path: Path, *, names,
                       resources: "Resources",
                       domain_pq: Optional[Tuple[str, str]] = None,
                       machine_record, one_process: bool,
                       where: str = "") -> Optional[str]:
    """The ``.sbatch`` text for this job when its target has a scheduler;
    ``None`` when it does not.

    Returns text rather than writing, so the whole of step 4 can be rendered
    before anything is on disk (`script-preparation.md` § 5, W7).

    **Every value in it is one the job stated** (`architecture.md` § 5.2):
    the queue prep placed it on (``domain_pq``, its job's placement,
    `placement.placed`), its wall, its memory, its cores per rank, its GPU
    count and -- for an engine that runs ranks -- its rank count.
    ``one_process`` (PySCF, with OpenMP threads) asks for one task: that is
    what the engine is, not a value anyone picks -- its engine record's
    ``one_process``, as the caller read it.
    ``where`` is the folder, from the one the job is submitted in, that its
    run script and the scheduler's own output are in (a walk's, `launch/`).
    The header is written in the target's scheduler kind's language, and
    none for a kind with no queue (`scheduler.md` § 6a).  Whether
    each is stated was asked at prep, before anything was written
    (`placement.launch_refusal`).  Nothing is filled in: not the target's
    width, not a rank per GPU, not a queue's ceiling, not a default from any
    config.
    """
    r = resources
    # THE MACHINE IS DECIDED ONCE, AND IT WAS DECIDED BEFORE THIS: prep
    # hands over the target's record, and that IS the answer -- resolving it
    # a second time here is how the two artifacts of one render could come
    # to describe two different machines.
    if machine_record is None:
        raise WrapperError(
            f"`{script_path.name}`: a header is rendered for the target's "
            f"record, and none was handed over")
    if not machine_record.kind.queue:
        return None  # no queue on this machine -> only .run.sh is meaningful

    # THE QUEUE PREP PLACED IT ON (`placement.placed`), as its job
    # records it: the header renders it and binds nothing.
    if not domain_pq:
        raise WrapperError(
            f"`{script_path.name}`: a header is rendered for the queue prep "
            f"placed it on, and none was handed over")
    partition, qos = domain_pq
    return render_sbatch(
        names, scheduler=machine_record.scheduler,
        partition=partition, qos=qos, resources=r,
        one_process=one_process, where=where)


def walk_header(base: Path, names, *, resources: "Resources",
                one_process: bool, machine_record,
                domain_pq: Tuple[str, str]) -> Optional[str]:
    """THE ONE HEADER OF A WALK -- a group's one job, a benchmark's shelf, a
    run of points (`job-system.md` § 6.0): rendered at prep for its one
    request on the queue prep placed it on (``domain_pq``), submitted from
    ``base``, the folder the walk runs from -- the calculation's for a
    group, the stage's for a run of points, the benchmark's container for a
    shelf -- its walk script and the scheduler's own output in `launch/`
    beside it
    -- or ``None`` where the target's record (``machine_record``) names no
    scheduler.  ``names`` are the walk's own (`runfiles.GroupNames`): the
    header's path and its folder, named from ``base``, are theirs
    (`GroupNames.path`)."""
    from .runfiles import GROUP_HEADER
    return _render_sbatch_for(
        names.path(base, GROUP_HEADER), names=names,
        resources=resources, domain_pq=domain_pq,
        machine_record=machine_record, one_process=one_process,
        where=f"{names.walked()}/")


# --------------------------------------------------------------------- #
#  The scheduler header (docs/execution/scheduler.md § 6a)              #
# --------------------------------------------------------------------- #


def render_sbatch(names, *, scheduler,
                  partition: str, qos: Optional[str],
                  resources: "Resources", one_process: bool,
                  where: str = "") -> str:
    """Render the ``<basename>.sbatch`` submission script, in the language of
    ``scheduler``'s kind (the target record's `record.Scheduler`;
    `scheduler.md` § 6a).

    The thin two-layer model (scheduler.md § 6a, *Two layers*): a header that
    allocates resources, then a one-line body that delegates to the
    UNCHANGED launcher ``bash <basename>.run.sh "$@"``.  The launcher still
    owns env activation + the ``mpirun`` launch; the ``.sbatch`` never
    re-implements ``module load`` / ``source activate``.

    **Every value is the job's own, and every one is required**
    (`architecture.md` § 5.2): the queue it named, bound on the target's
    record; and its ``resources``, WHOLE (`architecture.md` § 3.1, rule A8)
    -- its ranks, cores per rank, wall and memory, its GPU request.  There is
    no site configuration to fall back on: `molbuilder.json`'s `scheduler`
    block is refused.

    The ranks asked are the MPI rank count for CPU **and** GPU jobs -- the
    GPU ask carries the GPU count separately, and K ranks may share one GPU
    via MPS -- or one task where the engine runs ``one_process``.  Under the
    scheduler the run script reads the ranks it granted (``MB_NTASKS``), so
    this and ``mpirun -np`` agree by construction (running-a-job.md § 3.1).
    The GPU request (`jobset.model.gpu_request`) is consistent by
    construction: a GPU run with no count, or a count for a CPU run, never
    reaches here (`execution/gpu.md` G5); its binding asked unless
    ``gpu_binding`` is False (G9); the whole node only for a GPU job whose
    resources say so.
    """
    from .jobset.model import GpuRequestError, gpu_request
    r = resources
    try:
        gpus = gpu_request(r)
    except GpuRequestError as exc:
        raise WrapperError(f"render_sbatch: {exc}") from None
    ntasks = 1 if one_process else r.mpi_np
    cpus_per_task, time, mem = r.cpus_per_task, r.time, r.mem
    gpu_binding, exclusive = r.gpu_binding, r.exclusive
    for name, value in (("ntasks", ntasks), ("cpus_per_task", cpus_per_task)):
        if not isinstance(value, int) or value < 1:
            raise WrapperError(
                f"render_sbatch: {name} must be a positive int; got "
                f"{value!r}.")
    # THE QoS IS THE PAIR'S: a domain the record states no QoS for writes no
    # QoS line, and the scheduler applies the account's default
    # (`scheduler.md` § 4; `probe.derive_domains`).
    for name, value in (("partition", partition), ("time", time),
                        ("mem", mem)):
        if not (isinstance(value, str) and value.strip()):
            raise WrapperError(
                f"render_sbatch: {name} is required and was not stated.")
    if qos is not None and not (isinstance(qos, str) and qos.strip()):
        raise WrapperError(
            "render_sbatch: qos is the domain's, a name or None; got "
            f"{qos!r}.")

    basename = names.stem
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for sbatch emission: {basename!r}."
        )

    count = gpus.count if gpus is not None and gpus.uses else None
    if count:
        exclusive = bool(exclusive)
    else:
        exclusive = False  # CPU jobs never request a whole node here
    from .runfiles import SCHEDULER_STDERR, SCHEDULER_STDOUT
    from .scheduler.kinds import kind as _kind_named
    kind = _kind_named(scheduler.kind)

    # HOW IT IS SENT (`job-contracts.md` § 2.6): by `jobset launch` alone,
    # which opens each run and records it -- the run script it starts takes
    # no other start -- and, for a walk (``where``), writes the walk script
    # this header runs, its members' counts in it.
    lines: List[str] = [
        "#!/bin/bash",
        f"# === molbuilder scheduler header (scheduler: {kind.name}) ===",
        "# Generated by prep from the job's own stated values.",
        "# Authoritative design: docs/execution/scheduler.md § 6a.",
        "# Sent by `molbuilder jobset launch`, which opens each run and",
        *(["# records it, and writes the walk script this header runs --",
           "# each member's counts in it."] if where else
          ["# records it."]),
        "# It is not sent by hand: the run script it starts takes no",
        "# other start (job-contracts.md 2.6).",
        "#",
        *kind.directive("name", name=basename),
        *kind.directive("nodes"),
    ]
    # THE ONE EMITTER (scheduler.md R1): the queue, the wall, the ranks, the
    # cores, the devices and the memory rule, rendered by `scheduler.emit`.
    from .scheduler.emit import Directives
    _d = Directives(partition=partition, qos=qos, walltime=time,
                    ntasks=ntasks, cpus_per_task=cpus_per_task,
                    gpus=count,
                    gpu_binding=gpu_binding is not False,
                    mem=mem, exclusive=bool(exclusive))
    if exclusive:
        # The ignored value, said out loud so it is never a silent surprise
        # (the whole node, `scheduler.md` § 6a).  The RULE
        # itself lives in the emitter; this is the explanation beside it.
        lines.append(
            f"# The whole node is owned -> ALL its memory.  The stated "
            f"mem ({mem}) is IGNORED.")
    lines.extend(_d.header_lines(kind))
    lines += kind.directive("stdout", where=where,
                            file=kind.output_file(SCHEDULER_STDOUT))
    lines += kind.directive("stderr", where=where,
                            file=kind.output_file(SCHEDULER_STDERR))

    body = (
        "\n"
        # WHAT THE SCHEDULER TELLS THE JOB, said once in molbuilder's names
        # for everything this starts (`kinds.Kind.job_lines`) -- here, in the
        # kind's own file, so the run script is one file on every machine
        # (`architecture.md` § 9).
        + "".join(f"{ln}\n" for ln in kind.job_lines())
        + "\n"
        # WHEN THE JOB STARTED, said once for everything it runs: each run's
        # monitor counts the job's wall down from it -- a walk's later
        # member too (`run-reports.md` § 2.1b).
        "# When this job started, for everything it runs: each run's monitor\n"
        "# counts the job's time limit down from it (run-reports.md 2.1b).\n"
        f'export {JOB_STARTED_VAR}="$(date +%s)"\n'
        + "\n"
        "# The scheduler lands us in the folder it was sent from;\n"
        "# the launcher never cd's (running-a-job.md § 2.2a).  The launcher's\n"
        "# preamble + activation are baked into it, so it needs nothing\n"
        "# from the submitting shell.  \"$@\" forwards what `jobset launch`\n"
        + ("# hands it -- nothing: each member's counts are in the walk.\n"
           if where else
           "# hands the run script: its -np / -omp.\n")
        + f"bash {where}{names.run_script} \"$@\"\n"
    )
    return "\n".join(lines) + "\n" + body


def _validate_rendered_wrapper(text: str, script_path: Path) -> None:
    """Run ``bash -n`` (parse-only) on the rendered wrapper text.
    Raises :exc:`WrapperError` if bash rejects it as malformed shell.

    The text goes to ``bash -n`` on its standard input -- nothing is
    written anywhere: `prep` checks a wrapper it has planned and not yet
    written (`job-system.md` § 5.0, rule 3), in a folder that may not exist
    yet.  No execution happens; bash only checks shell-syntax
    validity.

    Cheap: a few ms per render; the user's wait is dominated by the
    upstream form-submit roundtrip anyway.  The alternative — only
    finding out at run time — costs the user a full re-render cycle
    (or worse, a confused "why doesn't my script run?" support
    request like the 2026-06-20 PDT incident).
    """
    import subprocess
    cp = subprocess.run(["bash", "-n"], input=text,
                        capture_output=True, text=True, timeout=15)
    if cp.returncode != 0:
        raise WrapperError(
            f"generator produced malformed shell for {script_path.name}; "
            f"bash -n rejected the rendered run script.  This is a "
            f"molbuilder bug -- the run script's template emitted invalid "
            f"syntax.  bash stderr:\n{cp.stderr}"
        )


__all__ = [
    "render_run_wrapper",
    "write_run_wrapper",
    "render_sbatch",
]
