"""Shell-wrapper emission — `prep` step 4 (render the wrapper).

Called by ``jobset/prep`` (the described route), ``bench/generate`` and the
web Build endpoint.  The ``molbuilder run`` verb that originally fronted
this module was deleted 2026-08-11 (C1, `process/conventions.md` § 3).

Each generated script (``.fdf`` or ``.py``) gets a sibling
``<basename>.run.sh`` that activates the right conda env and executes
the tool.  The user runs the ``.sh`` manually (foreground / background
/ cluster scheduler -- their call); molbuilder does **not** manage
processes.

The wrapper is intentionally small and human-readable:

* A user can read it to understand what command they're about to run.
* They can edit it to add custom flags (MPI options, env vars, ulimit).
* They can copy chunks into SLURM / PBS / GNU parallel scripts.

The wrapper is regenerated freshly on every `prep` (it's per-invocation
output, not state); edits between regenerations are lost.

Testing hook: tests inject a synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities`.  Production call
sites pass only the script path + optional ``env`` / ``mpi_np``
overrides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from .diagnostics import EXTENSION_TO_CATEGORY, get_capabilities
# The channel-name rule, from the module that owns the file those names
# have to match.  It cannot import upward -- it ships to a compute node
# alone -- so it is the end of the exchange that gets to own the rule.
from .config_dir import is_channel_name
# The thread chain's rungs, one list for the script's chain and the run
# script's (`running-a-job.md` § 3.2).
from .runtime_info import THREAD_SOURCES
# THE SESSION LOG'S LINES are its one module's, which travels beside the
# job: the wrapper renders them, every reader reads with them.
from .wrapper_log import LOG_CLOCK, LOG_LINE, RUN_INDEX_LINE, WRAPPER_LOG_START

if TYPE_CHECKING:                       # floor 5 reading floor 3's object
    # Under TYPE_CHECKING because the annotation is all this module needs:
    # `write_run_wrapper` unpacks a `Resources`, it never builds one (A4 --
    # `resolve.py` is the one builder).  A runtime import would also be legal
    # -- floor 5 may read floor 3 -- but it would be an import nothing calls.
    from .jobset.model import GpuRequest, Resources

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
# emission time.  Matches the ``_BASENAME_RE`` rule in
# ``molbuilder/config/siesta.py`` (SystemLabel validation) and
# ``molbuilder/config/pyscf.py`` (job_name): letters, digits, dot,
# underscore, hyphen.  The dot is allowed because PySCF emits files
# like ``<basename>.molwatch.log`` and SIESTA's user-named
# SystemLabel often contains dots.
_SAFE_WRAPPER_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")



class WrapperError(Exception):
    """Wrapper cannot be generated -- unsupported extension, no env
    routing, missing script file, ..."""


def _phys_cores_probe_block() -> str:
    """Bash that sets ``_phys_cores``, ``_n_sockets`` and ``_cps``.

    Prefer ``lscpu`` because it reports PHYSICAL cores; ``nproc`` counts
    hyperthread siblings, which would make ``_cps`` too large and lead to
    over-binding.  Falls back to ``nproc / 2`` (conservative 2-way-HT
    assumption) when lscpu is unavailable.

    ``|| true`` sits INSIDE each substitution on purpose: under
    ``set -e`` an assignment whose command substitution fails aborts the
    script, so without it a box with no ``lscpu`` died HERE -- before
    reaching the fallback written for exactly that case, and before
    ``-h`` could print usage (R9).

    Shared by both engines.  SIESTA's GPU rank policy divides the socket
    among MPI ranks; PySCF has no MPI and uses ``_phys_cores`` as the
    last-resort thread count when no allocation states one.  One probe,
    because "how many cores does this machine have" has one answer and
    two engines asking it separately is how they come to disagree.
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
        # ONE PROBE, ONE REPORT.  The echo used to live in the GPU
        # block, so a CPU sweep -- which is most sweeps -- recorded no
        # node shape at all, and `bench/result.py`'s `node_phys_cores`
        # (the field that exists so a sweep spread over different node
        # types can SAY so) could never be filled for one.  It belongs
        # with the probe: whoever measures it says what it measured.
        'echo "molbuilder: detected phys_cores=$_phys_cores, '
        'n_sockets=$_n_sockets, cores_per_socket=$_cps" >&2\n'
    )


def _stdout_role_for(deck_suffix: str) -> str:
    """The role a run of THIS deck writes its stdout to — from the catalogue.

    ``.py`` -> ``.pyscf.log``, ``.fdf`` -> ``.out``, and neither is spelled
    here.  This was ``".pyscf.log" if suffix == ".py" else ".out"``, an
    engine-to-role map written a third time one layer below the two in
    `parse/contract.py` that quoted this very line back at it
    (`model/parse.md` § 5.5, R-RO1 -- the vocabulary binds WRITERS too).

    Refuses rather than falling back.  The old conditional's ``else`` branch
    answered ``.out`` for ANY suffix it did not recognise, so a third engine
    would have had its wrapper redirect into SIESTA's filename in silence --
    which is the failure mode this section exists to end, in the writer.
    """
    from .runfiles import engine_of_role, stdout_roles
    engine = engine_of_role(deck_suffix)
    roles = stdout_roles(engine) if engine else ()
    if len(roles) != 1:
        raise WrapperError(
            f"no single stdout role for a {deck_suffix!r} deck (engine "
            f"{engine!r}, candidates {roles!r}).  The catalogue is "
            f"`runfiles.WRITTEN`: the deck's row names the engine and the "
            f"engine's `output=\"stdout\"` row names the file.")
    return roles[0]


def _run_index_resolver(basename: str, ext: str) -> str:
    """Bash block that resolves ``_out_file`` to
    ``{basename}-runN{ext}``.

    **The only attempt mechanism a wrapper has**, since P7 unit 1 retired
    ``_attempt_dir_block`` (2026-08-10).  That was ~130 lines of generated
    bash resolving the next ``run-<n>/``, creating it, linking the inputs,
    copying warm state and ``cd``-ing in -- ``jobset/materialize.py`` written
    a second time, in bash, one level down, in the layer deliberately kept
    free of filesystem logic.  It was also **the only thing in the system
    that broke the cwd rule everything else holds** (`job-contracts.md § 2.1`:
    the caller's working directory is the contract, and neither the wrapper
    nor the engine ever navigates), so retiring it restores an invariant
    rather than tidying one.  The behaviour it established is right and stays
    -- an attempt per invocation, immutable once written, inputs linked, warm
    state copied -- in Python, where `prepare_attempt` owns it.

    This one indexes attempts inside ONE directory, which is exactly the flat
    shape's rule (`project-layout.md` § 1: attempts told apart by an output
    index).  The hierarchy tells them apart by directory, and that is the
    layout layer's job, not the wrapper's.

    ``ext`` is the output-file ROLE (with leading dot), and it is REQUIRED:
    it defaulted to ``.out`` until 2026-09-18, which made SIESTA's filename
    the silent answer for a caller that forgot to say.  Every caller now
    passes :func:`_stdout_role_for`\'s answer, so the role comes from the
    catalogue at both sites (`model/parse.md` § 5.5).

    Honours two shell variables that the caller (the engine-specific
    args block) is expected to set:

      ``_continue`` (0/1)  -- ``--continue`` was passed; in addition
        to the default index advance it asks the engine to warm-start
        from prior state (.DM/.CG/.XV).
      ``_force``    (0/1)  -- ``--force`` was passed; restart the
        sequence at -run0 (overwriting it) instead of advancing.

    DEFAULT (2026-06-26): when a prior ``-runN{ext}`` exists the
    resolver **auto-advances** to ``max(N)+1`` -- re-running NEVER
    errors and NEVER overwrites a prior result.  The old behaviour
    (refuse-with-exit-1 unless ``--continue``/``--force``) was the
    single biggest papercut in the iterative resubmit loop.

    The resolver is shared by SIESTA + PySCF wrappers so the run-index
    semantics are identical across engines; only the suffix differs.
    ``basename`` is the script stem (e.g. ``hemeC_gas_03_tight``)
    baked in at generation time -- the bash itself doesn't try to
    derive it.
    """
    return (
        f"# --- Run index resolution ------------------------------\n"
        f"# Outputs are ``{basename}-runN{ext}``.  First run produces\n"
        f"# -run0; any later run AUTO-ADVANCES to max(N)+1 by default so\n"
        f"# re-running never errors and never clobbers a prior result.\n"
        f"# --force restarts the sequence at -run0; --continue adds an\n"
        f"# engine warm-start on top of the (default) index advance.\n"
        f"# EVERY per-run file counts, not the output alone: an engine that\n"
        f"# dies before its first line leaves its -runN.concluded, monitor\n"
        f"# log and util.csv and no output, and the next run must not reuse N.\n"
        f"_existing_max=-1\n"
        f'shopt -s nullglob 2>/dev/null || true\n'
        f'for _f in "{basename}-run"*; do\n'
        f'    _n=${{_f#{basename}-run}}\n'
        f'    _n=${{_n%%.*}}\n'
        f'    case "$_n" in\n'
        f"        ''|*[!0-9]*) continue ;;\n"
        f"    esac\n"
        f'    if [ "$_n" -gt "$_existing_max" ]; then\n'
        f'        _existing_max=$_n\n'
        f"    fi\n"
        f"done\n"
        f"shopt -u nullglob 2>/dev/null || true\n"  # D18c: restore
        f"\n"
        f'if [ "$_force" = "1" ]; then\n'
        f"    # --force: explicitly restart the sequence at -run0 (SIESTA's\n"
        f"    # redirect clobbers the existing -run0).\n"
        f"    _run_n=0\n"
        f'elif [ "$_existing_max" -ge 0 ]; then\n'
        f"    # DEFAULT auto-continue (2026-06-26): a prior -runN exists, so\n"
        f"    # advance to the next free index.  Re-running the script NEVER\n"
        f"    # errors and NEVER overwrites a prior result -- the #1 papercut\n"
        f"    # in the iterative HPC loop (resubmit after OOM / propor / a\n"
        f"    # walltime hit) was the old refuse-with-exit-1 gate.\n"
        f"    # (--continue additionally asks the engine to warm-start from\n"
        f"    # .DM/.CG/.XV; the run-index advances either way.)\n"
        f"    _run_n=$((_existing_max + 1))\n"
        f'    if [ "$_continue" != "1" ]; then\n'
        f'        echo "[molbuilder] prior output present; auto-continuing '
        f'as -run${{_run_n}}{ext} (use --force to restart at -run0, '
        f'--cold to also drop warm-start state)." >&2\n'
        f"    fi\n"
        f"else\n"
        f"    _run_n=0   # first run\n"
        f"fi\n"
        f'_out_file="{basename}-run${{_run_n}}{ext}"\n'
        f'echo "{RUN_INDEX_LINE} $_run_n  ->  $_out_file"\n'
        f"\n"
    )


def _monitor_block(label: str, stage: Optional[str], notify_on_scf: bool,
                   notify_every_hours: float, notify_channels,
                   notify_report, *, cores: str, gpu: bool,
                   unwatchable: Optional[str] = None) -> str:
    """Bash that launches the background monitor -- the SAME block for every
    engine (`running-a-job.md` § 4.1, `run-reports.md` § 2.3).

    It tells the monitor WHICH RUN it watches -- the label, the stage token,
    the run index the resolver just chose -- and never a path: the monitor
    names its log, its utilisation CSV and every file it reads through
    `runfiles`, and reads them through the framework's readers that travel
    beside it (:data:`MONITOR_COMPANIONS`).  Until 2026-09-26 this block sat
    in the SIESTA branch alone, spelled four paths by hand and passed the
    grammar's patterns as flags.

    It tells it WHAT THE JOB HOLDS too (`run-reports.md` § 2.1a): ``cores``
    is the shell arithmetic for the cores this engine is launched on -- a run
    started directly has no allocation, so its percentages are fractions of
    these -- and ``gpu`` whether the run uses a GPU, without which none is
    sampled or judged.

    ``unwatchable`` is why this run cannot be watched, when it cannot: the
    block then says so in the log and starts nothing.

    It runs with the JOB's OWN python from the working dir -- no molbuilder
    install, no numpy, no repo on PATH -- sleeps between wakes, and runs at
    `nice -n 19` so it never competes with the compute ranks.  Opt out with
    ``MB_MONITOR=0``.  Stopped by ``_mb_stop_monitor``; it also exits when
    this wrapper's PID (``$$``) disappears.
    """
    head = (
        # A '# ---' header in the EMITTED text: § 2.6's anatomy guard reads
        # blocks by these headers, and this real compute-node work was
        # structurally invisible to it (D9, user decision 2026-08-13).
        f"# --- Background job monitor ---------------------------\n"
        f'_monitor_pid=""\n'
        # The interpreter is PROBED (python3 first, python second): bare
        # `python` does not exist on python3-only hosts, and the backgrounded
        # launch swallowed the 127 -- the log then said "monitor: pid=N" for
        # a monitor that died at exec (R9, 2026-08-12).  Probed whether or
        # not a monitor starts: the wrapper's `_mb_ending` asks with it.
        f'_mb_py="$(command -v python3 || command -v python || true)"\n'
    )
    if unwatchable:
        return head + f'_log INFO "monitor: not started -- {unwatchable}"\n'
    return head + (
        f'if [ "${{MB_MONITOR:-1}}" = "1" ] '
        f'&& command -v nice >/dev/null 2>&1 '
        f'&& [ -n "$_mb_py" ] '
        f'&& [ -f {MONITOR_BUNDLE} ]; then\n'
        f'    nice -n 19 "$_mb_py" {MONITOR_BUNDLE} '
        # INDEXED LIKE THE OUTPUT IS: the run index the resolver chose, so a
        # re-run's monitor log and util.csv never interleave with or
        # truncate an earlier run's (found 2026-08-27 by reading the write
        # mode).  The monitor composes both names from it.
        f'--label "{label}" '
        + (f'--stage "{stage}" ' if stage else "")
        + f'--run "$_run_n" --util --cores "{cores}" '
        + ("--gpu " if gpu else "")
        + f'--interval "${{MB_MONITOR_INTERVAL:-10}}" '
        + (f'--notify-on-scf ' if notify_on_scf else "")
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
        # that stopped it -- where one that died loading once left nothing
        # at all (`run-reports.md` § 2.6).
        + f'--watch-pid $$ >/dev/null 2>>"$_runwrap_log" &\n'
        f'    _monitor_pid=$!\n'
        f'    _log INFO "monitor: pid=$_monitor_pid (nice 19, interval '
        f'${{MB_MONITOR_INTERVAL:-10}}s, util-sampling; '
        f'reads the run through the framework shipped beside it) -> the '
        f'run\'s .monitor.log + .util.csv"\n'
        f'else\n'
        f'    _log INFO "monitor: not started (set MB_MONITOR=1; needs '
        f'nice + a python interpreter + {MONITOR_BUNDLE} beside the '
        f'job)"\n'
        f'fi\n'
    )


def _finish_block(finish: Optional[str], script_name: str,
                  basename: str) -> str:
    """The job's last step when its engine leaves no result: run the finish
    bundle beside the deck with the job's own python (``$_mb_py``, probed by
    the monitor block whether or not a monitor starts), after the engine
    exited cleanly and before the conclusion is written
    (`engines/vibration.md` § 5.5).  Its failure is the job's: the
    conclusion records its exit status and the wrapper exits with it.
    Empty when the job has no finish."""
    if finish is None:
        return ""
    # The marker's words for a failed finish, and the log line that says the
    # finish began, are their READERS' (`parse/dirs/job.py`,
    # `wrapper_log`, which travel and read them beside the job).
    from .parse.dirs.job import FINISH_FAILED
    from .wrapper_log import FINISH_STARTED
    return (
        f"\n"
        f"# --- The calculation's own result (engines/vibration.md 5.5) ---\n"
        f"# The engine left what the result is derived from, not the result:\n"
        f"# {finish}, beside this deck, derives it here with the job's own\n"
        f"# python and writes it beside the run.  ITS FAILURE IS THE JOB'S --\n"
        f"# the conclusion records its exit status and this wrapper exits\n"
        f"# with it; its lines and any traceback are in this session log.\n"
        f'_log INFO "{FINISH_STARTED} {finish} {script_name} $_out_file"\n'
        f"set +e\n"
        f'if [ -n "$_mb_py" ] && [ -f {finish} ]; then\n'
        f'    "$_mb_py" {finish} {script_name} "$_out_file"\n'
        f"    _mb_finish_rc=$?\n"
        f"else\n"
        f'    _log ERROR "finish: needs a python beside the job and {finish} '
        f'beside the deck -- the result was not derived"\n'
        f"    _mb_finish_rc=1\n"
        f"fi\n"
        f"set -e\n"
        f'if [ "$_mb_finish_rc" -ne 0 ]; then\n'
        f'    echo "===== the finish ({finish}) exited with code '
        f'$_mb_finish_rc: no result -- see $_runwrap_log =====" >&2\n'
        # THE MARKER NAMES THE FAILED FINISH, so `run_status` reads the job
        # as failed although the engine's own output ended cleanly -- an
        # engine that errs after its end line (an MPI teardown) carries no
        # such words and keeps the output's verdict (`parse/dirs/job.py`).
        f'    printf "rc=%s at %s; {FINISH_FAILED} ({finish})\\n" '
        f'"$_mb_finish_rc" "$(date)" '
        f'> "{basename}-run${{_run_n}}.concluded"\n'
        f'    exit "$_mb_finish_rc"\n'
        f"fi\n"
    )


def _finish_check_block(finish: Optional[str], basename: str) -> str:
    """Before the engine starts: can the job's finish run here?  Asked with
    the python that will run it (``$_mb_py``, probed by the monitor block
    whether or not a monitor starts), once the run index is known, so a job
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
        f'if [ -z "$_mb_py" ] || [ ! -f {finish} ] '
        f'|| ! _mb_fin_said=$("$_mb_py" {finish} loads 2>&1); then\n'
        f'    _log ERROR "finish: {finish} cannot run here -- '
        f'${{_mb_fin_said:-no python in the job env, or {finish} is not '
        f'beside the deck}}"\n'
        f'    echo "ERROR: this job cannot finish itself: {finish} does not '
        f'load on the job\'s python (${{_mb_py:-none found}}) -- the engine '
        f'was not started, so no run is lost.  Its imports are the standard '
        f'library, numpy and ASE, which the job env carries '
        f'(envs/recipes.py)." >&2\n'
        f'    [ -n "$_mb_fin_said" ] && echo "$_mb_fin_said" >&2\n'
        f'    printf "rc=1 at %s; {FINISH_CANNOT_LOAD} ({finish})\\n" '
        f'"$(date)" > "{basename}-run${{_run_n}}.concluded"\n'
        f"    exit 1\n"
        f"fi\n"
    )


def _continue_force_args_parser(name_for_usage: str) -> str:
    """Bash snippet declaring + parsing ``--continue`` / ``-c`` /
    ``--force`` / ``-f`` / ``--cold`` / ``--from-scratch``.

    Three orthogonal flags:

      * ``--continue`` / ``-c``: advance run-index AND let the
        engine warm-start from prior state files (SIESTA: .DM, .CG,
        .XV; PySCF: .chk).
      * ``--force`` / ``-f``: start a fresh run-index sequence
        (-run0) even when prior outputs exist.  Does NOT touch the
        engine warm-start files -- the engine still loads them.
      * ``--cold`` / ``--from-scratch``: start the engine strictly
        from the .fdf / .py, OVERWRITING everything named after the
        run's id as the run proceeds (`job-contracts.md` § 4.1: a
        list of extensions is a snapshot of one build, and a file
        nobody listed is a file --cold walks past, so the sweep is
        by name).  It NAMES those files and refuses; ``--force``
        proceeds.  Keeping them is `molbuilder checkpoint save`.

    Added 2026-06-14 after the BDT-stage-2 incident where stage 2
    ran without the user's frozen-atom constraints (contract bug
    in the form, fixed in the same release); the now-corrupt
    warm-start files would have contaminated every subsequent run
    until ``--cold`` was provided.

    Caller is responsible for the eventual ``--help`` text, EXCEPT for the
    ``--cold`` entry: that one fact has one writer,
    :func:`_cold_usage_entry`, because it is what drifted when each engine
    wrote its own.
    """
    return (
        # WHAT THE FLAGS DO TO THE RUN INDEX -- which is all this shared
        # block can honestly say.  It is emitted for BOTH engines and its
        # whole input is a name, so it cannot know what the deck beside it
        # instructs; until 2026-08-18 it asserted anyway ("warm-start from
        # prior .DM/.CG/.XV"; "the engine still loads them"), which was the
        # fifth copy of a claim that is false for every stage described
        # `clean`.  Whether the ENGINE also picks up prior state is said where
        # the deck is known -- the usage text in each engine's own branch.
        f"# --- Continuation flags (shared SIESTA / PySCF) --------\n"
        f"# ``--continue`` / ``-c``: advance the run index, and ask\n"
        f"#                          the engine to resume if its deck\n"
        f"#                          allows it (see -h).\n"
        f"# ``--force``    / ``-f``: reset the run index to -run0.\n"
        f"#                          Prior state files stay on disk;\n"
        f"#                          whether they are read is the\n"
        f"#                          deck's to say.\n"
        f"# ``--cold`` / ``--from-scratch``:\n"
        f"#                          start purely from the .fdf/.py,\n"
        f"#                          OVERWRITING everything named\n"
        f"#                          after the run's id -- minus what\n"
        f"#                          molbuilder itself wrote.  Names\n"
        f"#                          them and refuses; --force\n"
        f"#                          proceeds.  Use when the prior\n"
        f"#                          run was bad and its restart\n"
        f"#                          files would corrupt the next.\n"
        f"# Self-identity for the warm-retry re-exec: an ABSOLUTE path to\n"
        f"# this wrapper.  Under ``bash x.run.sh`` $0 is a bare relative\n"
        f"# name -- bash's ``exec`` PATH-searches slash-less words (never\n"
        f"# the cwd), so ``exec \"$0\"`` would die with 127.  readlink -f\n"
        f"# (GNU, cluster-standard) resolves it without cd'ing (the\n"
        f"# wrapper never changes cwd -- that is a tested contract).\n"
        f"# Captured with the ORIGINAL argv so a retry re-runs with the\n"
        f"# same -np/--omp.\n"
        f'_mb_self="$(readlink -f -- "$0" 2>/dev/null || echo "$0")"\n'
        f'_mb_orig_args=(${{@:+"$@"}})\n'
        f"_continue=0\n"
        f"_force=0\n"
        f"_cold=0\n"
        f"# We strip --continue / --force / --cold from $@ here,\n"
        f"# leaving the rest for the engine-specific arg loop below\n"
        f"# (-np for SIESTA; nothing for PySCF).\n"
        f"_argv_remaining=()\n"
        f'while [ $# -gt 0 ]; do\n'
        f'    case "$1" in\n'
        f"        --continue|-c)        _continue=1; shift ;;\n"
        f"        --force|-f)           _force=1;    shift ;;\n"
        f"        --cold|--from-scratch) _cold=1;    shift ;;\n"
        f'        *)                    _argv_remaining+=("$1"); shift ;;\n'
        f"    esac\n"
        f"done\n"
        f'set -- "${{_argv_remaining[@]+\"${{_argv_remaining[@]}}\"}}"\n'
        f"\n"
    )


#: THE RESTART FILES A RUN SCRIPT KNOWS are the list in effect for its
#: calculation (`job-contracts.md` § 4.2a, `warmfiles.warm_list`, every
#: section -- a hint about the directory, safe to over-include), handed in
#: when ``prep`` renders the script (:func:`render_run_wrapper`'s ``warm``).
#: They were read from the engine's file at import -- ``_SIESTA_WARM_SUFFIXES``
#: / ``_PYSCF_WARM_SUFFIXES`` -- until 2026-10-03, so a calculation's own copy
#: was followed by prep and ignored by its script's ``Mode :`` line (plan
#: W36 ⑧).  PySCF's are SUFFIXES rather than extensions -- ``_optimized.xyz``
#: -- which no plain extension match would catch; and since 2026-08-10 the
#: banner and the ``--cold`` help read the one list, so a run holding only
#: ``<JOB>_optimized.xyz`` no longer announces a clean start.


def _cold_usage_entry(*, warm_examples: str) -> str:
    """The ``--cold`` / ``--from-scratch`` entry of a wrapper's ``--help``.

    **ONE writer, because it is ONE fact.**  `job-contracts.md` § 4.1's sweep
    is engine-independent *by construction*: it reads no list of extensions,
    so there is nothing in it for an engine to differ about.  An engine
    contributes only ``warm_examples`` -- what its own runs happen to leave
    behind -- and the rule itself is written once.

    **It was written out per engine until 2026-08-19, and the two copies
    disagreed.**  SIESTA's still promised to move the files "into a
    timestamped backup dir BEFORE running", which the launcher stopped doing
    on 2026-08-18 when the aside directory was replaced by a refusal; and it
    cited `job-contracts.md` § 4.1 while stating that section's opposite.  A
    reader who believed it would pass ``--cold --force`` expecting a copy and
    get an overwrite.  PySCF's copy had been corrected, which is what made
    this the shape the twin-file rule exists to catch: one engine fixed, the
    other not, with no mechanism that could notice.
    """
    return (
        "  --cold,\n"
        "  --from-scratch   start the engine from the deck alone.\n"
        "                   Everything named after the run's id --\n"
        "                   minus what molbuilder itself wrote --\n"
        f"                   is OVERWRITTEN as the run proceeds\n"
        f"                   ({warm_examples}).  So --cold NAMES\n"
        "                   those files and REFUSES; --force then\n"
        "                   proceeds.  Nothing is moved or copied:\n"
        "                   keep a state with `molbuilder\n"
        "                   checkpoint save` before you discard it.\n"
        "                   Swept BY NAME, never by a list of\n"
        "                   extensions (job-contracts.md 4.1) -- a\n"
        "                   file nobody listed is a file --cold\n"
        "                   walks past.\n"
    )


# `_deck_label` DELETED 2026-09-17, the day it was added.
#
# It opened the deck and read `SystemLabel` / `JOB` back out of it, at prep
# instead of at launch -- which is the SAME defect the awk had, one step
# earlier.  `execution/gpu.md` G7 is about not re-reading the deck at all,
# not about when: the value was in the description the whole time
# (`task.label` -- "the SystemLabel / JOB literal, and the stem of every
# file", `task.py`), and it now travels here as `label`.
#
# A8 does NOT forbid that parameter, which is why the first version went the
# wrong way: A8 says a door taking one of § 3's objects "may not also name
# that object's FIELDS".  `label` is not a field of `Resources`, the only
# such object this door takes, so passing it destructures nothing.

#: The charset a wrapper may put in a filename.  This was a `case` pattern
#: inside the emitted bash (`*[!A-Za-z0-9._-]*`), guarding a value the awk
#: had just read out of the deck.  The value is now read at prep, so the
#: guard moves here with it -- one language, and a deck whose label is not
#: nameable falls back to the basename instead of warning at launch.
_WRAPPER_LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")


def _cold_restart_block(basename: str, *, engine: str, label: str) -> str:
    """Bash snippet that NAMES the prior state a cold run would overwrite.

    **It reports and stops; ``--force`` proceeds** *(user, 2026-08-18)*.  It
    MOVED the files into a timestamped aside directory until then, which read
    as helpful and was not: the launcher was deciding to keep something nobody
    asked it to keep, and it left two mechanisms for preserving a state with
    different shapes and different names.  Keeping one is
    `molbuilder checkpoint save`, it is never automatic
    (`checkpointing.md` § 2), and this message says so.

    **A NAME SWEEP, not a list** (`job-contracts.md` § 4.1, decided
    2026-08-08; implemented U17, 2026-08-12): everything matching the
    run's id is named, except the files molbuilder itself wrote.  The
    suffix enumeration that stood here (13 SIESTA extensions, 5 PySCF
    suffixes, each with its own hazard comment) was a snapshot of one
    build's behaviour, and its failure mode was silent in the worst
    direction -- a file nobody listed is a file ``--cold`` walks past,
    in the one operation whose entire purpose is leaving nothing behind.
    The engine's output set depends on its version and options;
    completeness was never purchasable by maintenance.

    The restart-file list (``warm``, the calculation's own, handed in at
    render) keeps its OTHER § 4.2 job -- the short hint list the banner
    tests -- and is not read here.

    What survives the sweep is § 4.1's exception — *what molbuilder
    wrote* — and since 2026-08-13 (E-1) the bash case list is DERIVED
    from ``identity.OUR_FILE_PATTERNS``, the one Python spelling of that
    enumeration, plus ``*.psml`` (element-named, defensive) and the aside
    dirs themselves.  One list, two languages, no second copy to drift:
    the hand list that stood here lacked ``*.out`` and the
    monitor/util/scf-timing logs, and its comment claimed prior outputs
    "survive by construction (hyphen-joined)" — false for a FLAT STAGED
    calculation, whose ``<id>_<NN>_<stage>-run<N>.out`` matches
    ``<id>_*``: ``--cold`` on stage 2 moved stage 1's stdout and timing
    history into the aside dir.

    Nothing is moved, copied or deleted here.  The engine overwrites what it
    overwrites, once the user has said to.
    """
    # THE LABEL IS BAKED, NOT RE-READ.  `execution/gpu.md` G7 -- *"the value
    # travels; the deck is not re-read for it"*.
    #
    # SIESTA names its warm files from `SystemLabel` and PySCF from `JOB`, and
    # neither is the wrapper's own basename: the deck label is UNSUFFIXED while
    # the wrapper is `<label>_<NN>_<stage>`, so the sweep genuinely needs it.
    # An awk one-liner read it at LAUNCH until 2026-09-17, on the stated ground
    # that a person may edit the deck in between.  Measured, that does not hold
    # up: the deck fences a `user-custom` zone and warns against editing the
    # rest, and the wrapper's own output naming below is ALREADY a baked
    # literal -- so a run whose deck label changed after prep is inconsistent
    # with itself whatever this does.
    #
    # THE VALUE TRAVELS.  `label` is `task.label` -- the description's own
    # name for this calculation, which is what the emitter wrote into the deck
    # in the first place.  Nothing here opens the deck: reading back a file we
    # just wrote, to recover a value we were holding when we wrote it, is the
    # defect G7 names, and doing it at prep rather than at launch does not
    # make it a different one.
    #
    if engine not in ("siesta", "pyscf"):   # pragma: no cover
        raise WrapperError(f"unknown engine for cold-restart: {engine!r}")

    # DECIDED HERE, TOLD AT LAUNCH -- `_orbitals_per_rank_notice`'s shape,
    # eight lines down, and for the same reason.
    #
    # `label` falls back to the basename when the deck states none or could
    # not be read, which is what the awk's `:-` default did.  The charset
    # check was a `case` in the emitted bash, guarding a value the awk had
    # just read; the value is read at prep now, so the check comes with it.
    #
    # **But the fallback must still be VISIBLE.**  The bash printed
    # "[molbuilder] warning: ... contained disallowed characters; falling
    # back to basename" and my first version dropped that, so a hand-edited
    # label that cannot be used in a filename silently swept under the wrong
    # name: `--cold` would find nothing, report nothing to clean, and the
    # engine would warm-start off files the person believed were gone.  The
    # notice is BAKED here rather than re-derived at launch -- G7 governs
    # where the DECISION is made, not whether the person hears about it.
    _lbl = label or basename
    _label_notice = ""
    if not _WRAPPER_LABEL_RE.fullmatch(_lbl):
        _keyword = "SystemLabel" if engine == "siesta" else "JOB"
        _label_notice = (
            # Single quotes around the name INSIDE the double-quoted echo:
            # an apostrophe is literal there, needs no escaping, and does not
            # end the string.  The first version wrote \" and relied on bash
            # re-joining the fragments -- which produced the right sentence by
            # accident and would not have survived a name with a space in it.
            f"echo \"molbuilder: NOTE -- {_keyword} in the deck is not "
            f"usable in a filename; using '{basename}' for the name sweep. "
            f"Files written under the deck's own name will NOT be found by "
            f"--cold.\" >&2\n"
        )
        _lbl = basename
    label_extract = _label_notice + '_warm_label="' + _lbl + '"\n'
    # § 4.1's "except what molbuilder wrote", derived from the ONE
    # enumeration (identity.OUR_FILE_PATTERNS) rather than hand-spelled
    # here in a second language (E-1, 2026-08-13).  ``{label}`` becomes a
    # glob star because the sweep's own globs already anchor on the id --
    # the exception only needs the SHAPE of our names.
    # ...anchored on the run's OWN id, not widened to a star.
    #
    # This read `{label}` -> `*` until 2026-08-17, on the argument that "the
    # exception only needs the SHAPE of our names, because the sweep's own
    # globs already anchor on the id".  That argues the widening is HARMLESS,
    # not that it is needed -- and it stopped being harmless the moment a
    # suffix was shared.  Every pattern here used to end in something only
    # molbuilder writes (`.fdf`, `.run.sh`, `.template.toml`, `.molwatch.log`);
    # `{label}.xyz` joined the list on 2026-08-16 so `prep` would stop calling
    # a hand-over's input structure an engine leftover, and `.xyz` is the first
    # suffix BOTH molbuilder and an engine write.  Widened, it read `*.xyz` and
    # made PySCF's `<JOB>_optimized.xyz` -- warm state, the whole reason
    # `--cold` exists -- look like a file molbuilder had written.
    #
    # Both spellings, because the sweep visits both: the label read out of the
    # deck (`$_warm_label`) and the wrapper's own basename.  A quoted expansion
    # inside a `case` pattern is matched literally, so `"$_warm_label".xyz`
    # protects exactly one file while `<label>_optimized.xyz` goes aside.
    #
    # The ONE enumeration still governs (E-1, 2026-08-13): this narrows how the
    # list is read, and adds no second list to drift from it.  `*.psml` is
    # the one glob of its own: a pseudopotential is named for its element,
    # not for the run.  *(`*-restart-aside-*` stood beside it until
    # 2026-10-04, for folders `--cold` filled before 2026-08-18 -- old runs
    # are not a design input; plan D27.)*
    from .identity import OUR_FILE_PATTERNS
    _exceptions = "|".join(sorted(
        {p.replace("{label}", anchor)
         for p in OUR_FILE_PATTERNS
         for anchor in ('"$_warm_label"', basename)}
        | {"*.psml"}))
    return (
        f"# --- Cold restart: SAY WHAT WOULD BE LOST, THEN STOP ------\n"
        f"# --cold starts the engine from the deck alone, so everything\n"
        f"# named after the run's id is about to be overwritten.  This\n"
        f"# NAMES those files and refuses; --force proceeds.\n"
        f"#\n"
        f"# It MOVED them into a timestamped aside/ folder until\n"
        f"# 2026-08-18 (user).  That was the launcher deciding to keep\n"
        f"# something nobody asked it to keep, and it left two ways to\n"
        f"# preserve a state with different shapes.  Keeping a state is\n"
        f"# `molbuilder checkpoint save` and it is never automatic\n"
        f"# (`checkpointing.md` § 2).\n"
        f"#\n"
        f"# No list of engine extensions: a list is a snapshot of one\n"
        f"# build, and a file nobody listed is a file --cold walks past.\n"
        # THE ONE label extraction (F13, 2026-08-13): outside the --cold
        # guard, so the status banner below reads THIS value instead of
        # re-extracting without the sanitizer and overwriting it.
        + label_extract +
        f'if [ "$_cold" = "1" ]; then\n'
        f"    _clobber=0\n"
        f"    shopt -s nullglob 2>/dev/null || true\n"
        f"    echo \"[molbuilder] --cold: name sweep over "
        f"\\\"$_warm_label\\\".* and \\\"{basename}\\\".* -- everything the id "
        f"names, minus what molbuilder wrote\" >&2\n"
        f'    for _f in "$_warm_label".* "${{_warm_label}}"_* '
        f'{basename}.* {basename}_*; do\n'
        f'        [ -e "$_f" ] || continue\n'
        f'        case "$_f" in\n'
        f"            # molbuilder-written, by name shape (4.1's exception;\n"
        f"            # derived from identity.OUR_FILE_PATTERNS)\n"
        f"            {_exceptions}) continue ;;\n"
        f"        esac\n"
        f'        if [ "$_clobber" = "0" ]; then\n'
        f'            echo "[molbuilder] --cold would OVERWRITE prior state:" >&2\n'
        f"            _clobber=1\n"
        f"        fi\n"
        f'        echo "[molbuilder]     $_f" >&2\n'
        f"    done\n"
        f"    shopt -u nullglob 2>/dev/null || true\n"  # D18c: restore
        f'    if [ "$_clobber" = "1" ]; then\n'
        f'        if [ "$_force" = "1" ]; then\n'
        f'            echo "[molbuilder] --force given: overwriting the files above." >&2\n'
        f"        else\n"
        f'            echo "[molbuilder] Refusing: nothing has been changed." >&2\n'
        f'            echo "[molbuilder] To keep this state, save it first:" >&2\n'
        f"            echo \"[molbuilder]     molbuilder checkpoint save -m "
        f"'before a clean rerun'\" >&2\n"
        f'            echo "[molbuilder] Then run again with --force to overwrite." >&2\n'
        f"            exit 1\n"
        f"        fi\n"
        f"    else\n"
        f'        echo "[molbuilder] --cold: nothing under this name; already a clean start" >&2\n'
        f"    fi\n"
        f"fi\n"
        f"\n"
    )


def _orbitals_per_rank_notice(n_atoms) -> str:
    """A NOTICE about occupancy, checked against the rank count that is
    actually used — never a limit *(user ruling, 2026-09-03)*.

    **SIESTA distributes ORBITALS across ranks**, not atoms, so the number
    that says whether the ranks have anything to hold is

        n_orbitals / mpi_np

    and it wants to be greater than one.  At or below it, ranks are idle by
    arithmetic, and the user is told exactly that — *"your CPUs are not going
    to be fully used"* — with both numbers shown, so the claim is checkable
    rather than a verdict.

    **It replaces a clamp that was not science.**  The wrapper used to lower
    an auto rank count to ``n_atoms`` and warn about a user-set one above it,
    citing the ``propor IMAX=0`` abort.  That abort came from a PSML problem,
    not from the system's size; the rule helped by accident on the systems
    where it fired and refused perfectly good rank counts on the others.

    **The orbital count is an ESTIMATE and says so.**  The true count needs
    the basis of every species and is known only once SIESTA starts, so this
    uses ``10 * n_atoms`` — the same rough double-zeta-polarised figure the
    BlockSize bound uses and the deck's BENCH-MARKS block already publishes
    as ``n_orbitals_est`` (`job-contracts.md` § 3.3).  One estimate, one
    place, or the deck and the notice would disagree.

    Empty when the deck states no atom count (``NumberOfAtoms`` is optional
    in SIESTA): a notice needs a number, and inventing one is what this
    function exists to stop.
    """
    if n_atoms is None:
        return ""
    n_orb = 10 * int(n_atoms)
    return (
        f'# Occupancy notice: SIESTA distributes ORBITALS across ranks.\n'
        f'_norb_est={n_orb}   # ~10 x {int(n_atoms)} atoms (DZP estimate)\n'
        f'if [ "$_mpi_np" -ge "$_norb_est" ]; then\n'
        f'    echo "molbuilder: NOTE -- $_mpi_np ranks for ~$_norb_est '
        f'orbitals is <= 1 orbital per rank." >&2\n'
        f'    echo "  Your CPUs are not going to be fully used.  '
        f'This is a notice, not a limit -- the run proceeds." >&2\n'
        f'fi\n'
    )


def _runtime_status_block(
    basename: str,
    *,
    engine: str,
    script_name: str,
    warm: Sequence[str],
    resumes: bool = True,
) -> str:
    """Bash snippet that detects and emits the execution status banner.

    Prints, at script launch time, AFTER cold/run-index resolution but
    BEFORE the engine launches:

      * **Mode** -- one of:
        - ``COLD`` (``--cold`` was confirmed with ``--force``; the prior
          state it named is overwritten as the run proceeds)
        - ``WARM-RESUME`` (``--continue`` with prior state on disk)
        - ``WARM-RESUME REQUESTED but no prior state`` (``--continue``
          with nothing to resume from -- the user probably intended
          this to be a fresh start)
        - ``WARM-RESTART`` (no ``--continue``, but warm-start files
          exist on disk; the engine will silently load them.  This is
          the silent-failure mode pre-2026-06-14: a stage-2 run with
          bad constraints contaminated stage-3's restart files and the
          user couldn't see why their numbers were wrong.  The flag
          now makes this case loudly visible.)
        - ``initial-run (clean state)`` (no prior state; no flags).

      * **Constraints** -- engine-specific:
        - SIESTA: counts ``position`` lines + total listed indices in
          ``%block Geometry.Constraints``; ``(none)`` when the block
          is absent or empty.
        - PySCF: counts indices from the ``# Source: Structure.
          frozen_atoms = [...]`` comment the generator emits.

    The detection logic reads the actual on-disk script at runtime
    (NOT the values baked in at emit time), so a user-edited .fdf
    or .py shows the EDITED values -- the script ``you see is
    what runs`` contract that the 2026-06-14 fix landed.

    Added 2026-06-14 as part of the BDT-stage-2 incident response:
    silently-warm-restarting from contaminated restart files would
    have re-contaminated every downstream run; users have to be able
    to see at a glance whether their constraints are honored and
    where the SCF / geometry will start from.
    """
    if engine == "siesta":
        # Warm-start file extensions matching the cold-restart block.
        # 2026-06-14 fix: SIESTA writes warm-start files keyed on the
        # ``SystemLabel`` from inside the .fdf -- NOT on the .fdf's
        # filename basename.  So a second-stage .fdf with
        # ``SystemLabel  foo`` writes ``foo.DM`` etc., regardless of
        # the script being named ``foo_02_tight.fdf``.  Test both label
        # patterns (SystemLabel-keyed AND basename-keyed) for the
        # warm-start detection so the Mode line is accurate.
        # Match the full SIESTA warm-start ext tuple used by the
        # cold-restart aside below — covers transport (.HSX/.TSHS/
        # .TSDE/.WFSX) and the geometry-checkpoint case
        # (STRUCT_NEXT_ITER) too.  Used only to detect whether the
        # banner should report "Mode: hot/warm" vs "Mode: cold".
        #
        # ⚠ That comment was already here and was NOT true: this list was
        # retyped and had drifted to 10 of the 13, missing .Bonds, .EIG and
        # .PARTIAL.  A directory holding only those got the banner
        # "initial-run (clean state)" while ``--cold`` would have moved them
        # aside as warm state -- the two halves of one contract disagreeing,
        # and the half that was wrong is the one `run-identity.md § 5` says
        # must never be weakened, because it is the one always present.
        # Found by P3's Review 2, whose checklist names this exact shape:
        # "a comment claiming one list sat above two lists".  Now derived.
        warmstart_exts = tuple(s.lstrip(".") for s in warm)
        warmstart_test_pieces = []
        for ext in warmstart_exts:
            # `_mb_has_state`, not `[ -e ]` -- warm state is CONTENT.  See
            # the helper's own comment in `_runtime_status_block`.
            warmstart_test_pieces.append(f'_mb_has_state "$_warm_label.{ext}"')
            warmstart_test_pieces.append(f'_mb_has_state "{basename}.{ext}"')
        warmstart_test = " || ".join(warmstart_test_pieces)
        # Awk that:
        #   * lower-cases the first token for case-insensitive
        #     ``%block`` / ``%endblock`` / ``position`` matching
        #     (portable; replaces gawk's IGNORECASE).
        #   * squashes ``.``, ``-`` and ``_`` out of the block name
        #     before comparing, which is fdf's keyword rule and what
        #     ``parse/fdf.py::_norm`` does on the Python side.  It
        #     spelled two of the four legal spellings until
        #     2026-09-18 and `_sidecar.py` spelled a different two.
        # Two passes so ``_ncon_lines`` and ``_ncon_indices`` come
        # from the same logical scan.  ``|| true`` keeps awk's exit
        # code from aborting under ``set -euo pipefail``.
        _awk_count_program = (
            r'{ k = tolower($1); nm = tolower($2); gsub(/[._-]/, "", nm) } '
            r'k == "%block" && nm == "geometryconstraints" '
            r'{ in_b = 1; next } '
            r'k == "%endblock" && nm == "geometryconstraints" '
            r'{ in_b = 0; next } '
            r'in_b && tolower($1) == "position"'
        )
        constraint_detection = (
            f'_constraints="(no Geometry.Constraints block -- all atoms free)"\n'
            f'_fdf_path="{script_name}"\n'
            # Grep gate: case-insensitive and tolerant of all four
            # legal spellings of the separator.  Same rule as the awk
            # below.
            f'if [ -e "$_fdf_path" ] && grep -qiE '
            f'\'^[[:space:]]*%block[[:space:]]+Geometry[._-]?Constraints\' '
            f'"$_fdf_path"; then\n'
            f'    _ncon_lines=$(awk \'BEGIN{{in_b=0;n=0}} '
            f'{_awk_count_program} '
            f'{{ n++ }} END{{ print n }}\' "$_fdf_path" 2>/dev/null '
            f'|| echo 0)\n'
            f'    _ncon_indices=$(awk \'BEGIN{{in_b=0;n=0}} '
            f'{_awk_count_program} '
            f'{{ for (i=2;i<=NF;i++) if ($i ~ /^[0-9]+$/) n++ }} '
            f'END{{ print n }}\' "$_fdf_path" 2>/dev/null '
            f'|| echo 0)\n'
            f'    if [ -z "$_ncon_lines" ] || [ "$_ncon_lines" = "0" ]; then\n'
            f'        _constraints="Geometry.Constraints block present but EMPTY -- all atoms free"\n'
            f'    else\n'
            f'        _constraints="$_ncon_lines position line(s), $_ncon_indices listed indices (range expansion not counted)"\n'
            f"    fi\n"
            f"fi\n"
        )
        # DERIVED from the one tuple, like the detection above it: a
        # hand-typed five-name label told a .TSHS-only directory the
        # engine "will load DM/CG/XV..." -- files that were not there --
        # while the detection had already keyed on all thirteen (R9).
        warm_files_label = "/".join(x.lstrip(".") for x in warm)
    elif engine == "pyscf":
        # PySCF's ``mf.chkfile`` is keyed on ``JOB`` (a Python
        # variable in the .py script), same naming-mismatch risk as
        # SIESTA's SystemLabel.  Mirror the dual test.
        # DERIVED, mirroring SIESTA: the banner and the ``--cold`` mover
        # read ONE list, so a run whose only warm file is
        # ``<JOB>_optimized.xyz`` can no longer announce a clean start and
        # then have that very file named by --cold as warm state.
        # BRACED, not bare: a suffix may start with "_" (`_optimized.xyz`),
        # so an unbraced "$_warm_label_optimized.xyz" parses as one variable
        # name -- unbound under set -u, killing EVERY fresh-directory run
        # before launch (a .chk on disk short-circuits the || chain and
        # hides it).  SIESTA's branch survives bare only because its "."
        # terminates the name; same brace lesson as bench/grid.py.
        warmstart_test = " || ".join(
            piece
            for suf in warm
            for piece in (f'_mb_has_state "${{_warm_label}}{suf}"',
                          f'_mb_has_state "{basename}{suf}"')
        )
        # PySCF embeds the canonical frozen-atom list as a single-line
        # comment.  The indices are comma-separated inside ``[...]``, so
        # the count is the digits INSIDE the brackets -- the line's tail
        # says "(0-based)", and counting the whole line reported one index
        # too many for every deck.
        constraint_detection = (
            f'_constraints="(no frozen_atoms -- all atoms free)"\n'
            f'_py_path="{script_name}"\n'
            f'if [ -e "$_py_path" ]; then\n'
            f'    _frozen_line=$(grep -E \'^[[:space:]]*#[[:space:]]*Source:[[:space:]]+Structure\\.frozen_atoms\' "$_py_path" | head -1 || true)\n'
            f'    if [ -n "$_frozen_line" ]; then\n'
            # || true: with ZERO digits in the list, grep -o exits 1 and
            # pipefail killed the wrapper -- making the very "lists 0
            # indices" branch below unreachable (R9).
            f'        _frozen_list=$(printf %s "$_frozen_line" | grep -oE \'\\[[^]]*\\]\' | head -1 || true)\n'
            f'        _ncon_indices=$(printf %s "$_frozen_list" | grep -oE \'[0-9]+\' | wc -l || true)\n'
            f'        if [ "$_ncon_indices" = "0" ]; then\n'
            f'            _constraints="frozen_atoms comment present but lists 0 indices -- all atoms free"\n'
            f"        else\n"
            f'            _constraints="frozen_atoms: $_ncon_indices listed indices (geomeTRIC constraints file)"\n'
            f"        fi\n"
            f"    fi\n"
            f"fi\n"
        )
        # DERIVED like SIESTA's (D11, 2026-08-12): hand-typed "chk", the
        # banner told a <JOB>_optimized.xyz-only directory "engine will
        # load existing chk" while the detection keyed on all five.
        warm_files_label = "/".join(x.lstrip(".") for x in warm)
    else:                                  # pragma: no cover
        raise WrapperError(f"unknown engine for status block: {engine!r}")

    # Extract the engine's canonical label (SystemLabel for SIESTA,
    # JOB for PySCF) UNCONDITIONALLY so the warmstart_test below has
    # ``$_warm_label`` in scope even when ``--cold`` was NOT passed
    # (the cold block also extracts it but inside its own ``if``).
    # The wrapper runs under ``set -euo pipefail``; an unbound
    # variable would otherwise abort the run.
    # Same robustness rules as the cold block's extraction:
    # portable case-insensitive matching (no gawk IGNORECASE),
    # quote-stripping for SystemLabel, ``|| true`` under set-e,
    # ``:-`` default under set-u.
    # NO second extraction (F13, 2026-08-13): ``$_warm_label`` is
    # extracted ONCE, sanitized, by the cold-restart block's prefix --
    # which runs unconditionally, before this banner, in both engines.
    # The unconditional re-read that stood here overwrote the sanitized
    # value with an unsanitized one on every --cold run.
    label_extract_unconditional = ""
    return (
        f"# --- Runtime status banner --------------------------\n"
        f"# Reads the actual on-disk script + state files at run\n"
        f"# time and reports the resulting MODE + CONSTRAINTS so\n"
        f"# the user can see what's about to happen BEFORE the\n"
        f"# engine starts.  See _runtime_status_block docstring.\n"
        + label_extract_unconditional
        + f'# WARM STATE IS CONTENT, not mere existence: a zero-byte\n'
        + f'# restart file is nothing to resume from, and a launch that\n'
        + f'# announced "WARM-RESUME ... engine will load ..." over one\n'
        + f'# would be telling the person something false.\n'
        + f'_mb_has_state() {{ [ -s "$1" ]; }}\n'
        + f'_warmstart_present=0\n'
        + f"if {warmstart_test}; then _warmstart_present=1; fi\n"
        f'_mode="initial-run (clean state)"\n'
        f'if [ "$_cold" = "1" ]; then\n'
        f'    _mode="COLD (--cold --force; prior state overwritten)"\n'
        f'elif [ "$_continue" = "1" ]; then\n'
        f'    if [ "$_warmstart_present" = "1" ]; then\n'
        + (f'        _mode="{_retry_texts(False, None)["mode"]}"\n'
           if not resumes else
           f'        _mode="WARM-RESUME (--continue; engine will load {warm_files_label})"\n')
        + f"    else\n"
        + (f'        _mode="RE-RUN REQUESTED but no prior state found -- starting from the first step with nothing to read back"\n'
           if not resumes else
           f'        _mode="WARM-RESUME REQUESTED but no prior state found -- starting cold by necessity"\n')
        + f"    fi\n"
        f'elif [ "$_warmstart_present" = "1" ]; then\n'
        f'    _mode="WARM-RESTART (silent; engine will load existing {warm_files_label}.  '
        f'Pass --cold to discard them.)"\n'
        f"fi\n"
        f"{constraint_detection}"
        f"\n"
    )


def _probe_gpu0_numa() -> Optional[int]:
    """Resolve the NUMA node that GPU 0 is attached to.

    Two paths, tried in order:

    1. ``nvidia-smi --query-gpu=pci.bus_id -i 0`` as a SUBPROCESS with a
       hard timeout.  This was an in-process ``pynvml`` call until
       2026-08-28 -- the day an in-process NVML call froze the whole web
       server (this function runs at wrapper render, and prep runs from
       the browser, so the request-thread path is real here too).  NVML
       has no timeout anywhere in its API; a child process is the one
       fence that lets the caller walk away (system_load.py carries the
       full story).
    2. Kernel sysfs at ``/sys/bus/pci/devices/<id>/numa_node``.  The
       stable ABI the Linux kernel itself uses for NUMA-aware
       allocation.  A single integer; "-1" for "no NUMA / single
       node system".

    Returns the int, or ``None`` when:
      * nvidia-smi is absent, fails, or times out (CPU-only host, or a
        held driver -- the probe is an optimisation, never worth a hang)
      * the sysfs file isn't readable
      * the sysfs value is "-1" (no NUMA affinity)

    Called once per wrapper render, at script-generation time.  The
    answer is baked into the generated bash as a literal.
    """
    import subprocess
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if cp.returncode != 0:
        return None
    bus_id = cp.stdout.strip().lower()
    if not bus_id:
        return None
    # nvidia-smi format: ``00000000:65:00.0`` (8-hex domain).
    # sysfs path:        ``0000:65:00.0``    (4-hex domain).
    # Strip the leading 4 hex digits of the domain.
    import re
    m = re.match(r"^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", bus_id)
    if not m:
        return None
    sysfs_id = bus_id[4:]
    try:
        with open(f"/sys/bus/pci/devices/{sysfs_id}/numa_node", "r") as fh:
            txt = fh.read().strip()
    except OSError:
        return None
    try:
        val = int(txt)
    except ValueError:
        return None
    if val < 0:
        return None  # "-1" = no NUMA affinity / single-node system
    return val


def _baked_numa_literal_line() -> str:
    """Format the ``_gpu_numa="${MOLBUILDER_GPU_NUMA:-<probed>}"``
    bash line with the probed value baked in.  Factored out so the
    "0 is a valid NUMA node, don't truthy-fallback it" logic stays
    in one obvious place."""
    probed = _probe_gpu0_numa()
    value = "unknown" if probed is None else str(probed)
    return f'_gpu_numa="${{MOLBUILDER_GPU_NUMA:-{value}}}"\n'


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
        f"| tr 'A-Z' 'a-z' | tr -d ' ')\n"
        f'{indent}{out_var}=-1\n'
        f'{indent}[ -n "${bus_var}" ] && {out_var}=$(cat '
        f'/sys/bus/pci/devices/${{{bus_var}:4}}/numa_node 2>/dev/null '
        f'|| echo -1)\n'
    )


def _gpu_loadbalance_block() -> str:
    """Bash that derives the rank<->GPU load balance at RUN time.

    GOAL: turn the resolved rank count into a ranks-per-GPU figure so the
    rest of the wrapper can (a) gate MPS on actual GPU sharing and (b)
    let the per-rank launcher block-distribute ranks across GPUs.

    READS  ``$_mpi_np``  (rank count; set by the args block).
    SETS   ``$_ngpu``           -- visible GPU count (0 if none),
           ``$_ranks_per_gpu``  -- ``mpi_np / ngpu`` (clamped >= 1).
    EMITS  one stderr line ``molbuilder: GPU load-balance -- N ranks over
           G GPU(s) = K rank(s)/GPU`` -- the human-checkable confirmation
           that K matches the allocation (validation pinpoint for the
           benchmark sweep, job-contracts.md § 6).
    WHY: K is the knob the benchmark sizes; it must come from the ACTUAL
    runtime allocation (SLURM may grant a different GPU count than
    generation assumed).  1 GPU degenerates to all-ranks-on-GPU0 (the
    existing single-GPU behavior) with no code fork.

    ORDERING: emit AFTER ``$_mpi_np`` resolves and BEFORE the MPS block
    (which reads ``$_ranks_per_gpu``).  running-a-job.md § 3.3
    """
    return (
        "# --- GPU load-balance: rank <-> GPU matching "
        "(running-a-job.md § 3.3) ---\n"
        "# Count ALLOCATED GPUs now and split ranks across them.  The\n"
        "# per-rank launcher maps rank -> GPU as\n"
        "# local_rank*ngpu/localsize, so K=mpi_np/ngpu ranks share\n"
        "# each GPU via MPS.  1 GPU -> all ranks on GPU0 (the existing\n"
        "# behavior); N GPUs -> block-distributed, no code fork.\n"
        "# Prefer CUDA_VISIBLE_DEVICES (SLURM's allocated set) over\n"
        "# nvidia-smi -L (which over-counts on a shared node without\n"
        "# device-cgroup isolation) -- must agree with the per-rank\n"
        "# launcher's own count.\n"
        'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then\n'
        '    _ngpu=$(printf %s "$CUDA_VISIBLE_DEVICES" '
        '| tr , "\\n" | grep -c .)\n'
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
        (SLURM co-located us -- nothing to do);
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
        '# Walk this rank\'s OWNED cpus: collect the sockets we span and the\n'
        '# GPU-socket NUMA nodes we actually own.  Sample both endpoints of\n'
        '# each range so a contiguous "0-47" is seen as two sockets.\n'
        '_my_socks=""; _pin_numas=""\n'
        '_cl=$(awk "/Cpus_allowed_list/{print \\$2}" /proc/self/status '
        '2>/dev/null)\n'
        '_oldifs=$IFS; IFS=,\n'
        'for _rng in $_cl; do\n'
        '    _a=${_rng%%-*}; _b=${_rng##*-}\n'
        '    for _e in "$_a" "$_b"; do\n'
        '        case "$_e" in ""|*[!0-9]*) continue ;; esac\n'
        '        _ps=$(cat '
        '"/sys/devices/system/cpu/cpu${_e}/topology/physical_package_id" '
        '2>/dev/null)\n'
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
        'cores only on socket(s) $_my_socks; host<->device + ELPA OpenMP '
        'pay a remote hop -- this machine\'s scheduler did not keep the '
        'job\'s cores beside its GPU (running-a-job.md § 3.3)." >&2\n'
        '        ;;\n'
        '    esac\n'
        'fi\n'
        'fi\n'                                    # close MB_NO_SOCKET_PIN else
    )


def _gpu_per_rank_launcher_block() -> str:
    """Bash that writes the per-rank GPU launcher + picks the CPU-bind
    policy.  Emit ONLY in GPU mode.

    GOAL: give every MPI rank its own GPU and (under SLURM) defer CPU/mem
    placement to the scheduler.  Implements the general load-balance
    model -- 1 GPU degenerates to "all ranks on GPU0", N GPUs
    block-distribute -- with no code fork on GPU count (running-a-job.md § 3.3).

    WRITES a runtime helper ``.mb-rank-launch-$$.sh`` (``trap``-removed on
    EXIT) in which each rank computes ``gpu = local_rank*ngpu/localsize``,
    resolves that GPU's NUMA node, sets ``CUDA_VISIBLE_DEVICES``, logs its
    GPU+NUMA+cpuset, then ``exec``s SIESTA.  The heredoc is QUOTED
    (``'HELPEREOF'``) so the rank-time variables resolve when the rank
    runs, not when the wrapper writes the file.
    SETS  ``$_siesta_target="bash $_rank_helper"`` (the launch target the
          ``_launch_cmd`` assembly interpolates in place of bare siesta).
    UNDER SLURM (``$SLURM_JOB_ID`` set): clears ``$_numa_wrap_gpu`` and
          ``$_mpirun_bind`` so we do NOT double-bind against SLURM's
          cgroup cpuset (running-a-job.md § 3.3; the benchmark logs the actual
          cpuset to confirm SLURM bound near the GPU).
    WHY a helper FILE (not ``bash -c``): the per-rank logic can't survive
    as a word-split ``_launch_cmd`` string -- the ``bash -c`` quoting
    breaks under the later unquoted expansion -- so a tiny temp script is
    the robust idiom.
    """
    return (
        '_rank_helper=".mb-rank-launch-$$.sh"\n'
        "cat > \"$_rank_helper\" <<'HELPEREOF'\n"
        '#!/bin/bash\n'
        '_lr=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMIX_RANK:-0}}\n'
        '_ls=${OMPI_COMM_WORLD_LOCAL_SIZE:-${SLURM_NTASKS:-1}}\n'
        '[ "${_ls:-0}" -lt 1 ] && _ls=1\n'
        '# Allocated GPU list: prefer CUDA_VISIBLE_DEVICES (SLURM sets it\n'
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
        + 'exec $_pin siesta "$@"\n'
        'HELPEREOF\n'
        'chmod +x "$_rank_helper"\n'
        # Cleanup is handled by the single unified EXIT trap (_mb_cleanup,
        # set near the top): it rm's $_rank_helper.  A local ``trap ...
        # EXIT`` here would CLOBBER the MPS cleanup trap (bash keeps only
        # the last EXIT trap) -- hence the centralised function.
        '_siesta_target="bash $_rank_helper"\n'
        # Under SLURM, the scheduler's cgroup cpuset (--gres-flags=
        # enforce-binding + -c) governs CPU/memory placement.  Do NOT
        # also numactl-wrap / --map-by -- that double-binds and can fight
        # SLURM.  P1 in running-a-job.md § 3.3
        'if [ -n "${SLURM_JOB_ID:-}" ]; then\n'
        '    _numa_wrap_gpu=""\n'
        '    _mpirun_bind=""\n'
        '    echo "molbuilder: under SLURM (job $SLURM_JOB_ID) -- '
        'trusting scheduler cpuset for CPU/mem binding '
        '(no manual numactl)" >&2\n'
        # Off SLURM with >1 GPU in play, the whole-job wrap is WRONG by
        # construction: it binds every rank to GPU0\'s node, and the
        # per-rank helper\'s own finer numactl (each rank pinned beside
        # ITS GPU) cannot escape the outer cpuset -- half the ranks were
        # locked cross-socket from the very GPU they serve, tripping the
        # helper\'s own WARN (R9/F7, 2026-08-12).  The helper owns
        # placement whenever ranks span GPUs; the whole-job wrap stays
        # only for the single-GPU case it was designed for.
        'elif [ "${_ngpu:-0}" -ge 2 ] && [ -n "$_numa_wrap_gpu" ]; then\n'
        '    _numa_wrap_gpu=""\n'
        '    echo "molbuilder: $_ngpu GPUs in play -- per-rank NUMA '
        'binding via the rank helper (whole-job numactl would pin '
        'every rank to GPU0\'s socket)" >&2\n'
        'fi\n'
    )


def _siesta_resolved_log_block(script_name: str, gpu_mode: bool) -> str:
    """Bash that records the fully-resolved launch command + placement
    into the wrapper log -- ALWAYS, on every run.

    GOAL (user request 2026-06-26): the log must carry the exact command
    that ran and the GPU/NUMA context, so a post-mortem never has to
    guess what was launched.  These ``_log INFO`` lines ARE the audit
    trail that pins what the run did -- the validation anchor when a run
    is slow or wrong.

    READS ``$_launch_cmd`` / ``$_launch_note`` / ``$_mpi_np`` /
    ``$_omp_threads`` and (GPU mode) ``$_ngpu`` / ``$_ranks_per_gpu`` /
    ``$_numa_wrap_gpu``.
    EMITS ``_log INFO`` lines (resolved launch / launch mode / ranks-omp /
    [gpu placement]).
    """
    return (
        f'# --- Record resolved launch command + placement (log) ---\n'
        f'_log INFO "resolved launch : $_launch_cmd {script_name} '
        f'> $_out_file"\n'
        f'_log INFO "launch mode     : $_launch_note"\n'
        f'_log INFO "ranks / omp     : $_mpi_np ranks x $_omp_threads '
        f'OMP threads"\n'
        + (
            f'_log INFO "gpu placement   : ${{_ngpu:-0}} GPU(s), '
            f'${{_ranks_per_gpu:-?}} rank(s)/GPU, '
            f'numa-wrap=\'${{_numa_wrap_gpu:-none}}\'"\n'
            if gpu_mode else ""
        )
    )


def _siesta_dry_run_block(script_name: str, gpu_mode: bool) -> str:
    """Bash implementing ``--dry-run``: print the resolved command and,
    in GPU mode, the per-rank GPU/NUMA mapping that WOULD be used, then
    ``exit 0`` WITHOUT launching SIESTA.

    GOAL (user request 2026-06-26): let a user ``sbatch job.sbatch
    --dry-run`` (or run locally) and read the log to confirm the
    generated command matches the SLURM-allocated resources, WITHOUT
    spending a real run.  This is the cheap pre-flight validation of the
    whole resolution chain.

    READS the same resolved vars as the launch.  In GPU mode it loops
    ranks ``0..mpi_np-1`` and prints ``rank R -> GPU G (numa=N)`` -- the
    block-distribution the per-rank launcher will apply (divisor is
    ``$_mpi_np`` here since LOCAL_SIZE isn't set without mpirun; equal on
    single-node v1).
    SIDE-EFFECT-FREE: the MPS daemon start is separately gated on
    ``_dry_run != 1`` so a dry run touches no GPU state.
    EXPECTED OUTCOME: a ``DRY RUN`` banner + mapping table on stdout/log,
    then ``exit 0`` -- nothing else runs.
    """
    _dr_stem = script_name.rsplit(".", 1)[0]
    return (
        # Emitted header: § 2.6's anatomy guard reads blocks by these
        # (D9, user decision 2026-08-13: document, don't soften).
        f"# --- Dry-run preview (--dry-run) ----------------------\n"
        f'if [ "$_dry_run" = 1 ]; then\n'
        f'    echo ""\n'
        f'    echo "===== molbuilder DRY RUN (no SIESTA launch) ====="\n'
        f'    echo "  Resolved cmd : $_launch_cmd {script_name} '
        f'> $_out_file"\n'
        f'    echo "  Launch mode  : $_launch_note"\n'
        f'    echo "  MPI ranks    : $_mpi_np   '
        f'(source: ${{_np_source:-?}})"\n'
        f'    echo "  OMP threads  : $_omp_threads   '
        f'(source: ${{_omp_source:-?}})"\n'
        f'    echo "  Env          : ${{CONDA_DEFAULT_ENV:-<none active>}}"\n'
        f'    echo "  SLURM        : job=${{SLURM_JOB_ID:-<none>}} '
        f'ntasks=${{SLURM_NTASKS:-?}} cpus/task=${{SLURM_CPUS_PER_TASK:-?}} '
        f'gpus=${{SLURM_JOB_GPUS:-${{SLURM_GPUS:-?}}}}"\n'
        # The sbatch-header cross-check (user design 2026-08-13): run
        # OUTSIDE a SLURM job with the sibling .sbatch present, a local
        # dry-run resolves through the policy and looks right -- while
        # the header's -n is what will actually rule once submitted
        # (SLURM_NTASKS outranks the baked default).  Read the header
        # back and WARN on disagreement, so the stale-header mistake is
        # caught before a queue slot is spent.
        f'    if [ -z "${{SLURM_JOB_ID:-}}" ] '
        f'&& [ -f "{_dr_stem}.sbatch" ] 2>/dev/null; then\n'
        f"        _hdr_all=$(sed -n 's/^#SBATCH[[:space:]]*//p' "
        f'"{_dr_stem}.sbatch" | tr "\\n" " " || true)\n'
        f'        echo "  sbatch header: $_hdr_all"\n'
        f"        _hdr_n=$(printf %s \"$_hdr_all\" | sed -n "
        f"'s/.*\\(-n\\|--ntasks\\)[= ]\\([0-9][0-9]*\\).*/\\2/p' "
        f'| head -1 || true)\n'
        f'        if [ -n "$_hdr_n" ] && [ "$_hdr_n" != "$_mpi_np" ]; then\n'
        f'            echo "  WARNING: submitted through that header, '
        f'SLURM_NTASKS=$_hdr_n will OVERRIDE"\n'
        f'            echo "           the $_mpi_np rank(s) resolved above.  '
        f'Scale with:"\n'
        f'            echo "               sbatch -n $_mpi_np '
        f'{_dr_stem}.sbatch"\n'
        f'            echo "           or regenerate so header and deck '
        f'agree."\n'
        f"        fi\n"
        f"    fi\n"
        + (
            f'    echo "  Visible GPUs : ${{_ngpu:-0}}"\n'
            f'    echo "  Ranks/GPU    : ${{_ranks_per_gpu:-?}} '
            f'(MPS ${{_use_mps_str:-?}})"\n'
            f'    echo "  Rank -> GPU mapping (block-distributed):"\n'
            f'    _dr_ng=${{_ngpu:-0}}\n'
            f'    if [ "$_dr_ng" -lt 1 ]; then\n'
            f'        echo "      (no GPUs visible here -- on the '
            f'compute node each rank maps as rank*ngpu/ntasks)"\n'
            f'    else\n'
            f'        _dr_r=0\n'
            f'        while [ "$_dr_r" -lt "$_mpi_np" ]; do\n'
            f'            _dr_g=$(( _dr_r * _dr_ng / _mpi_np ))\n'
            + _bash_numa_from_gpu("$_dr_g", "_dr_numa",
                                  bus_var="_dr_bus", indent="            ")
            + f'            echo "      rank $_dr_r -> GPU $_dr_g '
            f'(numa=$_dr_numa)"\n'
            f'            _dr_r=$(( _dr_r + 1 ))\n'
            f'        done\n'
            f'    fi\n'
            if gpu_mode else ""
        )
        + f'    echo "================================================"\n'
        f'    _log INFO "dry-run complete; nothing launched"\n'
        f'    exit 0\n'
        f'fi\n'
        f"\n"
    )


def _siesta_scf_timing_func() -> str:
    """Bash defining ``_mb_scf_tee`` — the SCF per-iteration timing
    instrument (running-a-job.md § 4.1).

    GOAL: SIESTA emits **no usable per-iteration wall time** (the ``scf:``
    lines carry energies + dDmax but no time; ``timer: IterSCF`` is
    cumulative). This filter IS the benchmark's measurement instrument:
    it tees SIESTA stdout to the ``.out`` AND timestamps every ``scf:``
    iteration line into a per-run ``.scf-timing.log`` as
    ``<epoch.ns> <iter#> <full scf line>``, so per-iteration wall time =
    consecutive-stamp delta (first delta dropped, rest averaged --
    one clean sample under the capped 3-iteration bench trial;
    running-a-job.md § 4.1).

    EXPECTED OUTPUT: `<basename>-runN.scf-timing.log`, one line per SCF
    iteration; subtracting adjacent epochs gives the steady-state
    per-iteration time — the number the CPU-vs-GPU / K-sweep comparison
    ranks on.

    WHY this shape: a single ``awk`` process copies every line at C speed
    (``fflush(out)`` keeps the ``.out`` live for ``tail -f``); ``date
    +%s.%N`` is spawned only on the infrequent ``scf:`` lines. Portable
    across gawk/mawk (getline-from-command + close + fflush). The caller
    pipes ``$_launch_cmd … | _mb_scf_tee "$_out_file" "$_scf_timing_log"``
    and reads ``${PIPESTATUS[0]}`` for SIESTA's exit (awk never masks it).

    EVERY SCF ROW OF EITHER PHASE, and the pattern is not this function's: it
    is ``parse.engines.siesta_grammar``'s, the one table the parser and the
    monitor read too (`model/parse.md` § 5d.5).  It matched ``scf:`` alone
    until 2026-09-26, so a TranSIESTA device's 1000 ``ts-scf:`` iterations
    were timed as its 7 periodic ones -- *"4049.94 s/iter"* against 27.5.
    """
    from .parse.engines import siesta_grammar as _G
    return (
        "# --- SCF per-iteration timing instrument "
        "(running-a-job.md § 4.1) ---\n"
        "# Tees SIESTA stdout to the .out AND stamps each SCF row -- scf:\n"
        "# and TranSIESTA's ts-scf: -- into the per-run .scf-timing.log so\n"
        "# per-iter wall time =\n"
        "# consecutive-epoch delta (SIESTA prints no per-iter time).\n"
        "_mb_scf_tee() {\n"
        "    awk -v out=\"$1\" -v tlog=\"$2\" '\n"
        "        { print > out; fflush(out) }\n"
        "        /" + _G.SCF_ROW_ERE + "/ {\n"
        "            _cmd=\"date +%s.%N\"; _cmd | getline _ts; close(_cmd)\n"
        "            _l=$0; sub(/" + _G.SCF_PREFIX_ERE + "/, \"\", _l)\n"
        "            split(_l, _f, /[ \\t]+/)\n"
        "            print _ts, _f[1], $0 > tlog; fflush(tlog)\n"
        "        }\n"
        "    '\n"
        "}\n"
    )


def _ending_question_func() -> str:
    """Bash defining ``_mb_ending`` -- the wrapper's one door onto how the
    run ended (`execution/run-reports.md` § 2.3).

    THE WRAPPER ASKS, IT DOES NOT GREP.  It decided its warm retries and its
    failure hints by grepping the output for strings it typed itself --
    ``SCF_NOT_CONV``, ``outcoor: Final (unrelaxed) ...``, ``propor: ERROR``,
    ``ERROR|aborted|Stopping`` -- until 2026-09-26.  ``_run_ending.py``
    travels beside every job (:data:`MONITOR_COMPANIONS`), so the wrapper runs
    it with the job's own python over the output and over SIESTA's stderr,
    which the session log holds, and the markers keep their one home: the
    SIESTA family's table.

    ``_mb_ending QUESTION [ARG]`` answers by exit status (0 yes, 1 no, 2 the
    ending cannot be read); ``_mb_ending`` alone prints the ending in words.
    With no python beside the job, or a bundle that does not load on it, it
    answers 2 -- no hint and no warm retry.

    ``_mb_ending_able`` ASKS THE BUNDLE ONCE whether it loads here
    (``mb_monitor.pyz loads``) and remembers the answer, so a bundle that
    cannot load prints its error once and the one place that asks first says
    the ending cannot be read -- where every question printed the error again
    until 2026-09-28 (user: "ask the bundle once").  It is asked in THIS shell
    before any question, because an ask inside ``$( )`` cannot set anything its
    caller sees.
    """
    return (
        "# How the run ended, asked of the framework: `_run_ending`, in\n"
        f"# {MONITOR_BUNDLE} beside the job, reads the output and\n"
        "# SIESTA's stderr (this wrapper's log) with the SIESTA family's own\n"
        "# table (run-reports.md 2.3).  _mb_ending QUESTION [ARG]: exit 0 yes,\n"
        "# 1 no, 2 cannot read; _mb_ending alone: the ending in words.\n"
        "_mb_ending_able() {\n"
        '    if [ -z "${_mb_ending_state:-}" ]; then\n'
        f'        if [ -n "$_mb_py" ] && [ -f {MONITOR_BUNDLE} ] \\\n'
        f'           && "$_mb_py" {MONITOR_BUNDLE} loads; then\n'
        "            _mb_ending_state=able\n"
        "        else\n"
        "            _mb_ending_state=unable\n"
        "        fi\n"
        "    fi\n"
        '    [ "$_mb_ending_state" = able ]\n'
        "}\n"
        "_mb_ending() {\n"
        "    _mb_ending_able || return 2\n"
        f'    "$_mb_py" {MONITOR_BUNDLE} ending "$_out_file" '
        '--stderr "$_runwrap_log" "$@"\n'
        "}\n"
    )




def _gpu_runtime_block() -> str:
    """Bash for a GPU run's PLACEMENT: whether MPS is available, and the
    NUMA node the GPU is attached to.  It sets no rank or thread count.

    **The counts are the stated ones** *(2026-10-02, `architecture.md`
    § 5.2)*.  This block computed its own until then -- about
    ``phys_cores / 4`` ranks with MPS, 2 or 1 without, the core budget divided
    among them as threads, overridable by ``MOLBUILDER_MPI_NP`` /
    ``MOLBUILDER_OMP_NUM_THREADS`` -- and ``--mps`` re-derived them after the
    flags were parsed, replacing even a stated, baked rank count.  Each was a
    number nobody stated for that run.

    Outputs, consumed by the rest of the SIESTA wrapper:

      * ``_have_mps`` / ``_use_mps_default``: NVIDIA's Multi-Process Service
        is on PATH, and whether this run starts it (``--mps`` / ``--no-mps``
        and ``MOLBUILDER_USE_MPS`` decide; the MPS block below gates it on
        ranks sharing a GPU).
      * ``_gpu_numa``: the NUMA node GPU 0 is attached to, baked at
        script-generation time by :func:`_probe_gpu0_numa` (NVML + kernel
        sysfs); overridable at run time via ``MOLBUILDER_GPU_NUMA``.  An
        integer string, or ``"unknown"``.
      * ``_numa_wrap_gpu``: the ``numactl`` prefix that keeps a single-GPU
        run on its GPU's socket, where the box has two and numactl is there.
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
        # --mps / --no-mps win (the args block); the env var second; on when
        # MPS is available.  The MPS block starts it only when ranks share
        # a GPU -- single-rank MPS is overhead with no concurrency benefit.
        '_use_mps_default="${MOLBUILDER_USE_MPS:-$_have_mps}"\n'
        '_have_mps_str="no"; '
        '[ "$_have_mps" = "1" ] && _have_mps_str="yes"\n'
        'echo "molbuilder: mps_available=$_have_mps_str" >&2\n'
        # ---- GPU NUMA proximity (probed at generation time) ----
        # Resolved by the Python generator via ``_probe_gpu0_numa()`` using
        # NVML + the kernel sysfs ABI.  No string-scraping of
        # ``nvidia-smi``'s tabular output -- that was the failure mode of
        # the 2026-06-16 run where libnuma rejected ``--cpunodebind=N/A``.
        # The runtime override ``MOLBUILDER_GPU_NUMA=N`` wins over the
        # baked value.  NB: an explicit ``is None`` check, because 0 is a
        # valid NUMA node.
        + _baked_numa_literal_line() +
        '# Defence in depth: even if the baked / override value got\n'
        '# garbage, refuse to use it as a NUMA target unless it parses\n'
        '# as a non-negative integer.\n'
        'case "$_gpu_numa" in\n'
        '    ""|*[!0-9]*) _gpu_numa="unknown" ;;\n'
        'esac\n'
        # ---- numactl availability + NUMA-pin decision ----
        # We wrap mpirun (not each rank) so the cpuset is inherited
        # uniformly by every rank OpenMPI forks.  Three conditions:
        # dual-socket+ box, GPU NUMA known, numactl on PATH.
        '_numa_wrap_gpu=""\n'
        'if [ "$_gpu_numa" != "unknown" ] && '
        '[ "$_n_sockets" -ge 2 ] && '
        'command -v numactl >/dev/null 2>&1; then\n'
        '    _numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa '
        '--membind=$_gpu_numa"\n'
        'fi\n'
        # ``mps=on`` / ``mps=off`` reads unambiguously next to the numeric
        # counts in the --dry-run banner (one daemon per GPU; no count).
        '_use_mps_str="off"; '
        '[ "$_use_mps_default" = "1" ] && _use_mps_str="on"\n'
        "\n"
    )


#: fdf's own truthy set for a logical keyword (fdf_get's accepted values) --
#: how `_fdf_honours_restart` reads the deck's ``DM.UseSaveDM`` as SIESTA
#: does.  *(It was the GPU toggle's too, read off the deck by
#: ``_fdf_requests_gpu`` until 2026-10-03: whether a run uses the GPU is its
#: request now, `jobset.model.gpu_request`, never a scan of what was
#: rendered.)*
_FDF_TRUTHY = (".true.", "true", "yes", "t", "y", "1")


# DELETED 2026-08-13: ``_fdf_requests_elpa``.  It read ``Diag.Algorithm``
# to route any ELPA deck to the source build, on the premise that the
# packaged SIESTA has no ELPA.  The premise was false -- ELPA is compiled
# into conda-forge's binary through ELSI and both stages run on CPU
# (measured; see the routing comment in ``write_run_wrapper``).  With the
# premise gone the function had no caller, and a scanner nothing routes on
# is a keyword the wrapper claims to care about and does not.
#
# The choice of solver stays entirely the user's: ``Diag.Algorithm`` is a
# deck keyword like any other, and it no longer decides an environment.


def _retry_texts(resumes: bool,
                 restart_honoured: Optional[bool]) -> Dict[str, Optional[str]]:
    """What a retry of this run does, in the words every retry text uses --
    ONE description (`running-a-job.md` § 3.5): a run whose kind cannot
    resume (`Job.resumes`, the kind's warm-files fact) re-runs from its first
    step and reads back only what that step saved; a deck that declines
    prior state re-runs cold; otherwise the retry resumes warm.

    ``does`` is the retry's own message, ``policy`` the banner's retry line;
    ``mode``, ``usage`` and ``after`` are the non-resuming run's Mode line,
    ``--continue`` help and after-budget advice (``None``: the warm and cold
    texts are the deck-restart ones, which already say what happens).
    Until 2026-09-29 a force-constant rung's banner, Mode line and help all
    called its retry a resume (the K6 review, R4)."""
    if not resumes:
        return {
            "does": ("re-running from its first step -- this kind of run "
                     "does not resume"),
            "policy": ("each re-runs the run from its first step, reading "
                       "back only what that step saved: this kind of run "
                       "does not resume (warm-files: resumes = false), so an "
                       "SCF that stopped the first step continues and one "
                       "that stopped later meets the same SCF again"),
            # what is true of every kind that does not resume: a SIESTA
            # force-constant run reads back its first step's density, and
            # the usage and retry lines say so (PySCF's wrapper words its
            # own, from the deck it ships beside)
            "mode": ("RE-RUN FROM THE START (--continue; this kind of run "
                     "does not resume)"),
            "usage": ("This kind of run does not resume (warm-files:\n"
                      "                   resumes = false): --continue "
                      "re-runs it from\n"
                      "                   its first step, reading back only "
                      "what\n"
                      "                   that step saved.\n"),
            "after": ("revisit mixing/smearing -- a re-run of this run "
                      "starts again from its first step"),
        }
    # THE LINE AFTER THE BUDGET, for every case, in molbuilder's own road:
    # a stage is launched again, never re-run inside its attempt, and a
    # setting changes from the state saved before its prep -- "re-run
    # with --continue to extend" stood here until 2026-10-06, false for a
    # deck that declines prior state (the unit 11 review).
    back = ("to change a setting, go back to the state saved before its "
            "prep and prep it anew")
    if restart_honoured is False:
        return {"does": "re-running cold -- this deck declines prior state",
                "policy": ("COLD, because this deck sets DM.UseSaveDM "
                           ".false.: a retry re-runs from the deck's "
                           "coordinates, it does not resume"),
                "mode": None, "usage": None,
                "after": ("this deck reads no prior state, so a re-run "
                          "starts again from its own coordinates -- "
                          + back)}
    return {"does": "warm-restarting",
            "policy": "each resumes warm from what the run banked (--continue)",
            "mode": None, "usage": None,
            "after": ("launch the stage again to continue from this run -- "
                      "or, " + back)}


def _fdf_honours_restart(text: Optional[str]) -> Optional[bool]:
    """Whether this deck lets SIESTA read the state a previous run left.

    Reads the deck's text, because the deck is what SIESTA obeys.  ``None``
    when it says nothing — a non-SIESTA script, an unreadable one, or one from
    before the restart group was written out.

    **Why the wrapper has to ask rather than assert.** Its ``--continue`` help
    said *"SIESTA reads .DM/.CG/.XV automatically when present (generator emits
    the flags by default)"*, which was true of a continuing deck and false of a
    clean one — and the wrapper ships beside exactly one deck, so it can simply
    look. A stage described `clean` now carries ``DM.UseSaveDM .false.``, and on
    that deck ``--continue`` advances the run index and starts cold; a help text
    promising otherwise is the wrapper telling the user something its own deck
    contradicts.

    First match wins, as libfdf does (`fdf_locate` stops at the first label).
    """
    from .script_emit import parameter
    if text is None:
        return None
    # THE ONE DOOR, deck-backed.  Which keyword answers for `restart` is the
    # catalogue's to say (`[item.restart].expands`), not this function's -- it
    # hand-spelled `DM.UseSaveDM` in a regex, which is the fourth copy of the
    # declaration and the habit this object exists to end.
    answer = parameter("restart", "siesta", deck_text=text).value
    if answer is None:
        return None
    return answer.strip().lower() in _FDF_TRUTHY


def _py_deck_reads_prior(text: Optional[str]) -> Optional[bool]:
    """Whether this PySCF deck reads prior state at start -- from the DECK.

    ``True``: the deck carries the restart-gated reads (the chkfile
    init-guess marker; the optimization deck described ``continue``).
    ``False``: an optimization deck described ``clean`` -- no read is
    emitted at all.  ``None``: a vibration deck (it computes its own
    relaxation and reads no prior engine state at start) or an
    unreadable file.  Keying the wrapper's continuation story on the
    deck text is what keeps the help from claiming reads the deck does
    not contain -- the same doctrine as ``_fdf_honours_restart``.
    """
    if text is None:
        return None
    if ".spectra.json" in text:
        return None                     # the vibration deck
    return 'init_guess = "chkfile"' in text


def _fdf_n_atoms(text: str) -> Optional[int]:
    """Read the ``NumberOfAtoms`` line off a SIESTA deck's text, or None.

    The wrapper needs the atom count to state its occupancy NOTICE
    (`_orbitals_per_rank_notice`); it clamps nothing with it since the
    2026-09-03 ruling.  Parsing the .fdf keeps the wrapper self-contained
    (the .fdf IS the source of truth for what SIESTA will see) and avoids
    plumbing n_atoms through every caller.  Returns None if the line isn't
    found -- ``NumberOfAtoms`` is OPTIONAL in SIESTA, the coordinates block
    being authoritative -- and the notice is then simply not emitted.
    """
    from molbuilder.parse.fdf import _parse_fdf
    # THROUGH THE ONE READER (2026-09-17).  This was
    # `re.search(r"(?im)^\s*NumberOfAtoms\b\s+(\d+)")` with a comment saying
    # *"SIESTA FDF parsing is whitespace-insensitive + case-insensitive on
    # labels.  Match defensively."* -- which knew the rule and implemented
    # half of it: `\b` on the literal word does not match `Number.Of.Atoms`
    # or `number_of_atoms`, and fdf treats all three as one keyword.
    got = _parse_fdf(text)[0].get("numberofatoms")
    if not got:
        return None
    try:
        return int(got[0])
    except ValueError:
        return None


def _effective_parameters_block(script_path: "Path") -> str:
    """What SIESTA will be given, echoed into the run log at launch.

    **The one block both engines write** (`model/parse.md` § 5d.3a): a row per
    catalogue item -- ``[item, default, asked, used]``, `script_emit`'s format
    and its reader -- then the deck as the engine will read it.  SIESTA is a
    separate process that has not started when this runs, so the rows carry
    the item's catalogue DEFAULT only: what the deck asks for is the deck
    itself, which the record reads from the attempt, and what SIESTA used is
    its own ``fdf.<stamp>.log``.  The default is baked here because it is a
    fact of the catalogue this wrapper was rendered from -- the run's own
    record of it, never today's catalogue read later.

    **The deck is read at LAUNCH**, comments and blank lines stripped -- the
    lines libfdf parses -- so a deck somebody hand-edited after `prep` is
    recorded as the engine will see it.
    """
    import shlex as _shlex
    from . import script_emit as _sc
    from .deck_record import BLOCK_PARAMETERS, begin_marker, end_marker

    script_name = script_path.name
    lines = [
        "",
        "# --- What the engine will read ------------------------------",
        f'echo "{begin_marker(BLOCK_PARAMETERS)}"',
    ]
    for item in _sc.declarations(engine="siesta"):
        param = _sc.parameter(item.name, "siesta")
        if param.writes:
            lines.append("echo " + _shlex.quote(
                _sc.parameter_row(item.name, param.default)))
    lines += [
        f'grep -v "^[[:space:]]*#" "{script_name}" | grep -v "^[[:space:]]*$"'
        ' | sed "s/^/#   /"',
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


def render_run_wrapper(script_path: Path, *,
                       label: str = "",
                        resources: "Resources",
                        env: Optional[str] = None,
                        n_atoms: Optional[int] = None,
                        project_dir: Optional[Path] = None,
                        machine_record=None,
                        finish: Optional[str] = None,
                        resumes: bool = True,
                        warm: Optional[Sequence[str]] = None,
                        deck_text: Optional[str] = None) -> str:
    """Return the bash text for a wrapper running ``script_path``.

    **The allocation arrives whole** — `architecture.md` § 3.1, rule A8.  This
    named four of ``Resources``' fields in its own signature until 2026-08-18,
    so every caller re-derived which subset mattered; ``max_memory_mb`` was
    lost that way once and the ranks/cores pair once more four days later.
    With the object passed whole there is no subset to choose, and *which*
    name this function uses internally — ``omp_threads`` for what the object
    calls ``cpus_per_task`` — is its own business (`job-contracts.md` § 6.2
    keeps the two names because different layers read them).

    ``env`` and ``n_atoms`` stay loose because neither is part of the
    allocation: the first is a per-invocation override, the second a fact
    read off the DECK.  (A ``mem_audit`` rode here until 2026-08-24 -- a
    baked memory MODEL re-estimated at launch.  Deleted, not unwired:
    memory is what the user states, never what a model guesses.)

    Routing by file extension:

    * ``.fdf``  → SIESTA.  Uses ``mpirun -np <N>`` when ``mpi_np`` is
                  given and ≥ 2; redirects stdout to the dynamic
                  ``<basename>-runN.out`` (the run index N is resolved
                  by the wrapper at run time; first run is -run0,
                  ``--continue`` advances to next free N).
    * ``.py``   → PySCF.  Runs ``python <script>`` with the same
                  ``-runN`` redirect, but the suffix is
                  ``.pyscf.log`` instead of ``.out`` (Phase C
                  rename, 2026-06-07) so the Results-tab inspector
                  dispatcher can distinguish PySCF stdout from
                  SIESTA's.  The script's own progress-log writer
                  (``MolwatchEmitter``) writes the ``.molwatch.log``.

    Both wrappers accept ``--continue`` / ``-c`` and ``--force`` /
    ``-f``.  See the wrapper's ``-h`` for the full flag inventory.

    Args:
      script_path: the ``.fdf`` or ``.py`` to wrap.
      env: override the routed env name for this invocation.  Default
        is whatever ``Capabilities.env_for_category(<category>)`` returns.
      resources: the job's allocation, whole.  ``mpi_np`` is the SIESTA
        rank count and is ignored for ``.py`` scripts.
      n_atoms: SIESTA atom count.  It no longer clamps anything (user
        ruling, 2026-09-03); it feeds the occupancy NOTICE, which needs an
        orbital estimate.  Auto-parsed from the .fdf by ``render_wrappers``
        when omitted; ``None`` simply means no notice can be stated.
      deck_text: the deck as it will be written, for one `prep` has planned
        and not yet written (`jobset.planned`); read from ``script_path``
        when omitted, and ``None`` when that cannot be read -- the wrapper
        then says what the deck instructs is the deck's to say.
    """
    if deck_text is None:
        try:
            deck_text = Path(script_path).read_text()
        except OSError:
            deck_text = None
    r = resources
    mpi_np, omp_threads = r.mpi_np, r.cpus_per_task
    max_memory_mb, continue_retries = (r.max_memory_mb,
                                       r.continue_retries)
    # WHEN this calculation should speak up -- carried on the allocation
    # like `continue_retries`, and rendered as monitor flags rather than
    # scheduler ones.  Absent means the flags are not emitted at all, so a
    # wrapper for a description that asked for nothing looks exactly as it
    # did before this existed.
    # Read straight off the object, not via getattr-with-a-default: this
    # function is typed for `Resources` and a defaulting read would turn a
    # field that went missing into a wrapper that silently stops notifying.
    # That failure -- a field lost between a job and its wrapper -- is the
    # one this module has already had twice.
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
                f"letters, digits, '-' and '_'.  A name is written into a "
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
    category = EXTENSION_TO_CATEGORY.get(suffix)
    if category is None:
        raise WrapperError(
            f"`{script_path.name}`: unsupported script extension "
            f"`{suffix}`.  Supported: "
            f"{', '.join(sorted(EXTENSION_TO_CATEGORY))}."
        )
    # A FINISH THE LAUNCHER CANNOT HONOUR IS REFUSED, never dropped: a job
    # told to finish that ran without finishing would conclude rc=0 with no
    # result (`engines/vibration.md` § 5.5).  Only a SIESTA job has one today
    # -- a PySCF deck writes its own result.
    if finish is not None and (category != "siesta"
                               or finish not in _FINISH_BUNDLES):
        raise WrapperError(
            f"`{script_path.name}`: this launcher finishes a SIESTA job with "
            f"{', '.join(sorted(_FINISH_BUNDLES))}; it cannot finish a "
            f"{category} job with {finish!r}")

    # WHETHER THIS RUN USES THE GPU is its request -- `jobset.model.
    # gpu_request`, the one door (`execution/architecture.md` § 3.2) -- read
    # once, here, for the env, the GPU placement and the monitor alike.
    # Prep asks it before anything is written, so a job of ours always
    # answers; the refusal below is the door's own words for one that does
    # not.
    from .jobset.model import GpuRequestError, gpu_request
    try:
        gpus = gpu_request(resources)
    except GpuRequestError as exc:
        raise WrapperError(f"`{script_path.name}`: {exc}") from None

    # SIESTA env routing: the ONE thing that decides which env to run in is
    # whether the run uses the GPU -- never the solver.
    #
    # THE TWO ENVS SPLIT ON PROVENANCE, NOT ON HARDWARE (2026-08-13).
    # ``molbuilder-siesta`` is installable from packages on ANY machine;
    # ``molbuilder-siesta-gpu`` must be BUILT FROM SOURCE, which some HPC
    # sites do not permit.  That is the whole reason there are two.
    #
    # CPU-ELPA needs NEITHER.  Measured, not assumed -- an H2 deck run in
    # the packaged env:
    #     Diag.Algorithm ELPA-2stage  -> exit 0, E = -30.136019 eV
    #     Diag.Algorithm ELPA-1stage  -> exit 0, E = -30.136019 eV
    #     Divide-and-Conquer          -> exit 0, E = -30.136019 eV
    #     ELPA-2stage + Diag.ELPA.GPU .true. -> EXIT 1,
    #         "diag: ELPA error on gpu set" / ELPA_ERROR_ENTRY_NOT_FOUND
    # conda-forge's SIESTA links no external ``libelpa``, but ELPA is
    # compiled INTO the binary through ELSI (279 defined ELPA symbols,
    # zero undefined), so both stages run on CPU.  Only the GPU entry is
    # absent from that build -- which is a build capability, not a
    # missing device.
    #
    # Until 2026-08-13 this routed EVERY ELPA deck to the source build.
    # On a site that cannot compile, that refused a CPU-ELPA run --
    # telling the user to install an env they cannot build -- for a
    # solver the installed baseline already runs.  Knowing a keyword is
    # not providing the capability, and the two are now kept apart.
    #
    # IMPORTANT: ``category`` drives every downstream ``if category ==
    # "siesta":`` branch in this module (MPI launch, .out filename,
    # log extension, runtime-status block).  We must NOT change it
    # here -- only the env LOOKUP needs to differ.  The earlier shape
    # of this fix mutated ``category`` to ``"siesta-gpu"`` and silently
    # disabled the entire SIESTA branch, which leaked a ``.pyscf.log``
    # filename and an unbalanced-quote template into the wrapper.
    # ``env is None`` guards this: an env the USER named always wins, so
    # choosing the source build for its external ELPA stays available
    # without molbuilder guessing on their behalf.
    env_lookup_category = category
    if category == "siesta" and env is None and gpus.uses:
        env_lookup_category = "siesta-gpu"

    caps = get_capabilities()
    target_env = (env if env is not None
                  else caps.env_for_category(env_lookup_category))
    if target_env is None:
        raise WrapperError(
            f"category `{category}`: no env name registered.  Pass "
            f"env=... explicitly or add a default to "
            f"molbuilder.diagnostics.DEFAULT_ENV_NAMES."
        )

    # GPU-env presence gate.  When the .fdf opted into GPU diagonalization
    # at generate time but ``molbuilder-siesta-gpu`` isn't installed,
    # ``source activate molbuilder-siesta-gpu`` would fail at run time
    # with a conda-side error -- the user-facing message would not point
    # at the real fix (install the env or turn the toggle off).  Raise
    # here with the install hint so the wrong env is caught at script-
    # generation time.  Only fires when we AUTO-routed (env not user-
    # passed) AND the host snapshot lists at least one env (an empty
    # snapshot means the conda probe never ran -- can't gate on that).
    # WHICH MACHINE IS ASKED WHETHER THE ENV EXISTS.
    #
    # *Which* env you want is a preference and comes from your own config
    # above; whether it EXISTS is a fact about the machine that will run
    # this, so the question goes to that machine's record when there is one
    # (`configuration.md` § 5 M-1).  Asking the box you are standing on is
    # the same mistake as baking its activation: `molbuilder-siesta-gpu`
    # installed here says nothing about ASU Sol, and the answer arrives as
    # a `conda activate` failure on a compute node after a queue wait.
    #
    # An EMPTY inventory is "unknown", never "none" -- a record written
    # before the field, or a machine with no conda on PATH -- so it cannot
    # refuse anything.
    # A RECORD, ONCE GIVEN, IS THE ONLY MACHINE ASKED -- including when it
    # cannot answer.  Falling back to this box's inventory for a record
    # that carries none would re-ask the wrong machine by a different
    # route: a workstation without `molbuilder-siesta-gpu` would refuse a
    # bundle for a cluster that has it, and one WITH it would wave through
    # a bundle for a cluster that does not.  Unknown stays unknown.
    if machine_record is not None:
        _known_envs = [str(e) for e in (
            getattr(machine_record, "conda_envs", None) or [])]
        _rec_envs = _known_envs
    else:
        _known_envs = list(caps.conda_envs or ())
        _rec_envs = []
    _env_is_known = bool(_known_envs)
    _env_present = target_env in _known_envs
    if (env is None
            and env_lookup_category == "siesta-gpu"
            and _env_is_known
            and not _env_present):
        # Only ONE ask reaches here now: GPU diagonalization.  The
        # message used to branch on GPU-vs-CPU-ELPA and tell a CPU-ELPA
        # user to install a source build for a solver the packaged env
        # already runs.  Name what is actually missing, and the way out
        # that does not require compiling anything.
        raise WrapperError(
            f"`{script_path.name}` requests GPU diagonalization "
            f"(``Diag.ELPA.GPU .true.``) but the ``{target_env}`` env is "
            f"not installed "
            + ("on the machine this bundle is being prepared FOR (read "
               "from its probed record.  If you installed the env since, "
               "re-probe that machine and prep again -- a calculation is "
               "set to the record of its first prep, so one prepped "
               "before goes back to the state saved before that prep "
               "first: `molbuilder checkpoint list` shows the folder's "
               "states).  "
               if _rec_envs else "on this machine.  ")
            + f"GPU support is the one thing the packaged "
            f"SIESTA does not have -- its ELPA is built without the GPU "
            f"entry -- so it lives only in that source build.  Either "
            f"install it with ``python -m molbuilder envs install "
            f"{target_env}`` (source build, ~10 minutes; some HPC sites "
            f"do not allow it), or turn off the SIESTA ``Use GPU`` "
            f"toggle and regenerate the .fdf.  Keeping ``Diag.Algorithm "
            f"ELPA-1stage/ELPA-2stage`` is fine either way: the packaged "
            f"env runs both on CPU."
        )

    # What this deck instructs about prior state -- read from the deck, so the
    # wrapper's own help cannot contradict the file it ships beside.
    _restart_honoured = (_fdf_honours_restart(deck_text)
                         if suffix == ".fdf" else None)
    # WHAT A RETRY OF THIS RUN DOES -- one description (`_retry_texts`), read
    # by every text that speaks of a retry: the banner's retry line, the
    # retry's own message, a retried run's Mode line, the --continue usage
    # and the line after the budget is spent (`running-a-job.md` § 3.5).
    _retry = _retry_texts(resumes, _restart_honoured)
    _py_reads_prior = (_py_deck_reads_prior(deck_text)
                       if suffix == ".py" else None)

    # THE BUDGET IS NOT OVERRIDDEN HERE, and that is deliberate.
    #
    # This zeroed `continue_retries` on a deck that declines prior state, on
    # the reasoning that a retry which cannot warm-start just repeats an
    # identical cold run.  The reasoning is right about the RUN and wrong about
    # whose call it is: `job-contracts.md` § 6.2 says the value *"rides the
    # element's Resources; prep bakes it into the wrapper"*, and
    # `test_the_warm_retry_budget_travels_the_described_route` exists because
    # it once did not -- *"a value that travels correctly and is dropped at the
    # last hop"*, which is precisely what dropping it here recreated.
    #
    # So the budget travels, and what changes is that the wrapper stops
    # DESCRIBING a cold re-run as a warm resume: the retry banner and the
    # comment beside the loop read the deck (`_restart_honoured`) and say what
    # a retry will actually do on this one.  Say it, do not decide it.

    basename = script_path.stem
    script_name = script_path.name
    # WHICH RUN THE MONITOR WATCHES, as the grammar reads this deck's name
    # (`run-reports.md` § 2.3): the label it was told and the stage token.
    # Without a label the deck's stem stands for it -- every file of the
    # rung begins with it, so the names compose the same.
    from .runfiles import RunFileError as _RunFileError
    from .runfiles import compose as _rf_compose
    from .runfiles import parse as _rf_parse
    _named = _rf_parse(script_name, label) if label else None
    watch_label, watch_stage = ((label, _named.stage) if _named is not None
                                else (basename, None))
    # A NAME THE MONITOR CANNOT COMPOSE is said here, not found there: it
    # names every file through `runfiles`, and a deck pointed at by hand as
    # `my.relaxation.fdf` gives a stem no run label can be -- the monitor
    # then died at start with its stderr at /dev/null, leaving no log at all.
    try:
        _rf_compose(watch_label, ".monitor.log", watch_stage)
        unwatchable = None
    except _RunFileError:
        unwatchable = (f"the deck's name {basename} is not a run label the "
                       f"file catalogue can read back (letters, digits, "
                       f"- and _ only)")
    # Shell-safety: both basename and script_name are interpolated
    # raw into bash f-strings throughout this module (inside
    # ``"..."``, inside glob lists, inside ``$(...)``, etc.).
    # Reject anything outside ``_SAFE_WRAPPER_NAME_RE`` to prevent
    # shell injection via a malicious filename.  The same alphabet
    # the SIESTA SystemLabel / PySCF job_name validators enforce
    # in ``molbuilder/config/*``.
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for wrapper emission: "
            f"{basename!r}.  Allowed characters: letters, digits, "
            f"``.``, ``_``, ``-``.  Rename the script and prep "
            f"again (`molbuilder jobset prep`)."
        )
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(script_name):
        # script_name is basename + suffix; if basename passed but
        # script_name fails, the suffix carries the offending char
        # (shouldn't be reachable since suffix is fixed to .fdf/.py,
        # but defence in depth).
        raise WrapperError(
            f"unsafe script filename for wrapper emission: "
            f"{script_name!r}."
        )

    # Pre-command env exports.  Shared anti-oversubscription recipe
    # with PySCF / spectra (see molbuilder/runtime_info.py): BLAS is
    # ALWAYS pinned to 1 thread per rank so OMP * BLAS doesn't
    # multiply.  OMP defaults differ by engine:
    #
    #   * SIESTA: ``OMP_NUM_THREADS = 1`` (mainline SIESTA is not
    #     reliably OMP-aware; pure MPI is the standard recipe).  User
    #     overrides via cfg.omp_threads only when running an
    #     OMP-compiled SIESTA build (hybrid MPI+OMP).
    #
    #   * PySCF: the WRAPPER resolves and exports OMP_NUM_THREADS
    #     (P1b, 2026-08-13) -- ``-omp`` flag, else OMP_NUM_THREADS,
    #     else the scheduler's allocation, else the count stated at prep
    #     (it ended at this node's physical cores until 2026-10-02).  No
    #     division by a rank count: PySCF is OpenMP-only.
    #
    #     It used to leave the variable unset so the script's own
    #     setdefault would win, and the script counted the whole NODE.
    #     Correct on a workstation, where the node IS the allocation;
    #     wrong under a scheduler, where a job holding 8 of 128 cores
    #     started 128 threads and time-sliced them onto its 8.  The
    #     script keeps the same chain for the case that still needs it
    #     -- ``python job.py`` run by hand, with no wrapper.
    env_prefix = ""
    if category == "siesta":
        # GPU mode is the run's own answer (its request, read above).  It
        # decides the GPU placement block below (MPS, NUMA) and the env --
        # and nothing about the counts, which are stated in either mode.
        gpu_mode = gpus.uses
        # THE RANKS AND THE THREADS ARE STATED (user, 2026-10-02: "explicit
        # job config is the only way allowed"; `architecture.md` § 5.2) --
        # and baked as stated.  An unstated count became the target's width
        # (ranks), one (threads), or -- on a GPU -- this script's own policy,
        # worked out at launch; each was a number nobody stated.  Whether
        # both are stated is asked ONCE, before anything is written:
        # `placement.launch_refusal` for a run, the grid's own point for a
        # benchmark (`MachineTranslation`).
        #
        # THE RANK COUNT IS THE USER'S (user ruling, 2026-09-03), and nothing
        # lowers it either: what is objective -- orbitals per rank -- is a
        # NOTICE (`_orbitals_per_rank_notice`), against the count actually
        # resolved at run time.
        resolved_mpi = int(mpi_np)
        resolved_omp = int(omp_threads)

        # NOTE: the actual launch command is computed at RUN time by
        # the probe block below, NOT here -- the wrapper picks
        # ``mpirun -np $_mpi_np siesta`` vs bare ``siesta`` based on
        # what ``siesta --version`` reports for the currently-installed
        # binary AND the runtime $_mpi_np value (from -np / MB_NP /
        # the stated value).  ``inner`` is the post-probe shell
        # expression; the launch_block at the bottom of this function
        # wraps it in ``set +e`` + propor-detection.  The
        # ``description`` string here is for the wrapper file header
        # only -- the user's actual -np at run time may differ.
        inner = f"$_launch_cmd {script_name} > $_out_file"
        description = f"SIESTA run, -np {resolved_mpi} as stated"

        # ---- Argument-parsing prelude (SIESTA only) ----
        # The wrapper accepts ``-np N`` (or env var ``MB_NP=N``) so
        # users can experiment with MPI rank counts WITHOUT
        # regenerating.  Background: SIESTA can crash with
        # ``propor: ERROR: IMAX = 0`` at startup for certain
        # mpi_np / molecule combinations.  The crash is data-
        # dependent on the ProcessorY x ProcessorX grid SIESTA auto-
        # picks for that rank count; predicting it from rank count
        # alone is not robust.  Allowing a runtime override means
        # the user can try ``./run.sh -np 8`` after a crash with
        # ``-np 15`` without regenerating the .fdf or the wrapper.
        # The post-run diagnostic at the bottom of this wrapper
        # catches the crash and prints retry suggestions.
        # Two-stage argument parsing:
        #
        #   1. Shared --continue / --force consumption (engine-
        #      agnostic; strips those flags and leaves the rest in $@).
        #   2. SIESTA-specific -np / -h.
        #
        # ORDER matters: --continue/--force are recognised first so
        # callers can combine them with -np in any order
        # (``--continue -np 8`` and ``-np 8 --continue`` both work).
        #
        # THE STATED COUNTS, BAKED -- the same two in GPU and CPU mode.  A
        # GPU-mode policy value stood here until 2026-10-02, and the deck was
        # re-read at LAUNCH to choose between the two sets: a wrapper that
        # answered differently from the deck it was rendered with, for a
        # hand-edited deck.
        _mpi_np_default_assignment = (
            f"# The stated rank and thread counts, baked at prep\n"
            f"# (running-a-job.md § 3.1-3.2).\n"
            f"{_STATED_COUNTS[0]}={resolved_mpi}\n"
            f"{_STATED_COUNTS[1]}={resolved_omp}\n"
        )
        siesta_args_block = (
            _continue_force_args_parser("SIESTA wrapper")
            + f"# --- SIESTA-specific argument parsing -----------\n"
            f"# Override the stated counts with: ``-np N`` /\n"
            f"# ``-omp N`` flags, or ``MB_NP`` / ``OMP_NUM_THREADS`` env\n"
            f"# vars.  Useful for retrying after a propor crash (see the\n"
            f"# diagnostic at the bottom of this wrapper) or for bench\n"
            f"# sweeps WITHOUT regenerating the .fdf / wrapper.\n"
            + _mpi_np_default_assignment
            + f"# MPI rank-count precedence (highest first):\n"
            f"#   1. ``-np N`` flag on the wrapper invocation\n"
            f"#   2. ``MB_NP`` env var (manual override)\n"
            f"#   3. ``SLURM_NTASKS`` (scheduler-allocated under sbatch)\n"
            f"#   4. ``PBS_NP`` (scheduler-allocated under qsub)\n"
            f"#   5. the stated value, baked at prep ($_mpi_np_default)\n"
            f"# Per docs/execution/running-a-job.md § 5: reading scheduler env vars for\n"
            f"# launch tuning is part of the wrapper contract -- the user\n"
            f"# reserved ``--ntasks=N`` from SLURM, the wrapper honors it.\n"
            f'_mpi_np="${{MB_NP:-${{SLURM_NTASKS:-${{PBS_NP:-$_mpi_np_default}}}}}}"\n'
            # OMP precedence: -omp flag > OMP_NUM_THREADS env >
            # SLURM_CPUS_PER_TASK (the sbatch ``-c`` allocation) > the
            # stated value, baked at prep.  Honoring a user-set OMP_NUM_THREADS matches the
            # standard OMP-toolchain convention; the prior wrapper
            # unconditionally clobbered it, which surprised users
            # benching with ``OMP_NUM_THREADS=8 ./run.sh``.  Under sbatch
            # the scheduler reserved ``-c`` cores/rank (the OMP width per
            # the running-a-job.md § 3.3 sizing) -- honor it so the
            # Sol allocation drives OMP automatically without a manual
            # -omp (running-a-job.md § 5: reading scheduler env for launch
            # tuning is part of the wrapper contract).
            f'_omp_threads="${{OMP_NUM_THREADS:-'
            f'${{SLURM_CPUS_PER_TASK:-$_omp_threads_default}}}}"\n'
            f'_dry_run=0\n'
            # Explicit-flag markers, read by the source report below
            # (`--dry-run` names where each count came from).
            f'_np_from_flag=0\n'
            f'_omp_from_flag=0\n'
            + f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        -np|--np)\n"
            f'            if [ $# -lt 2 ]; then\n'
            f'                echo "ERROR: -np requires a value" >&2\n'
            f"                exit 1\n"
            f"            fi\n"
            f'            _mpi_np="$2"; _np_from_flag=1; shift 2 ;;\n'
            f"        -omp|--omp|-t|--threads)\n"
            f'            if [ $# -lt 2 ]; then\n'
            f'                echo "ERROR: -omp requires a value" >&2\n'
            f"                exit 1\n"
            f"            fi\n"
            f'            _omp_threads="$2"; _omp_from_flag=1; shift 2 ;;\n'
            f"        --dry-run|--dryrun)\n"
            f"            # Resolve + LOG the launch command and the\n"
            f"            # rank<->GPU/NUMA placement, then exit WITHOUT\n"
            f"            # running SIESTA.  Lets you sbatch a preview and\n"
            f"            # read the log to confirm the command matches\n"
            f"            # the allocation (running-a-job.md § 3.3).\n"
            f'            _dry_run=1; shift ;;\n'
            # MPS toggle.  Default state is decided in the GPU placement
            # block based on (a) ``nvidia-cuda-mps-control`` binary
            # presence and (b) the MOLBUILDER_USE_MPS env var.  These flags
            # ALWAYS win, and they switch the daemon and nothing else: the
            # rank count is the stated one either way.  (They re-derived it
            # from a GPU policy until 2026-10-02, replacing even a stated,
            # baked count.)  Single-rank runs auto-disable below (MPS has
            # overhead with no concurrency benefit when only one process
            # touches the GPU).
            + (
                "        --mps)\n"
                '            _use_mps_default=1; shift ;;\n'
                "        --no-mps)\n"
                '            _use_mps_default=0; shift ;;\n'
                if gpu_mode else ""
            ) +
            f"        -h|--help)\n"
            f'            cat <<USAGE\n'
            f'Usage: bash $(basename "$0") [--continue|-c] [--force|-f] [--cold] '
            f"[-np N] [-omp N] [--dry-run]"
            + (" [--mps|--no-mps]" if gpu_mode else "")
            + " [-h]\n"
            f"\n"
            f"  --continue, -c   resume from prior run.  Scans existing\n"
            f"                   -runN.out files and writes -run(N+1).\n"
            + (
                f"                   {_retry['usage']}"
                if not resumes else
                f"                   This deck says 'start from: continue'\n"
                f"                   (DM.UseSaveDM / MD.UseSaveXV /\n"
                f"                   MD.UseSaveCG .true.), so SIESTA also\n"
                f"                   reads the .DM/.XV/.CG left under this\n"
                f"                   SystemLabel.\n"
                if _restart_honoured else
                f"                   This deck says 'start from: clean'\n"
                f"                   (DM.UseSaveDM .false.), so SIESTA will\n"
                f"                   NOT read prior .DM/.XV/.CG: the run\n"
                f"                   index advances and the calculation\n"
                f"                   starts cold.  To resume: go back to the\n"
                f"                   state saved before this stage's prep,\n"
                f"                   change 'restart' in the description,\n"
                f"                   and prep it anew.\n"
                if _restart_honoured is False else
                f"                   Whether the engine also reads prior\n"
                f"                   state is the deck's to say.\n"
            )
            + f"  --force, -f      start over from -run0 even if prior\n"
            f"                   runs exist.  Old files are NOT deleted;\n"
            f"                   the existing -run0.out is overwritten.\n"
            f"                   Prior .DM/.CG/.XV warm-start files STAY\n"
            + (
                f"                   on disk -- and this deck reads them.\n"
                f"                   Use --cold --force to discard them.\n"
                if _restart_honoured else
                f"                   on disk, but this deck declines them\n"
                f"                   (.false.), so they are not read.\n"
                if _restart_honoured is False else
                f"                   on disk; whether they are read is the\n"
                f"                   deck's to say.\n"
            )
            + _cold_usage_entry(warm_examples=".DM/.CG/.XV among them")
            + f"  -np N            override the MPI rank count.  Stated\n"
            f"                   at prep: $_mpi_np_default.\n"
            f"  -omp N,          override OpenMP threads per MPI rank.\n"
            f"  -t N, --threads  Aliased.  Stated at prep:\n"
            f"                   $_omp_threads_default.\n"
            f"  --dry-run        resolve + log the launch command and the\n"
            f"                   rank->GPU/NUMA placement for the current\n"
            f"                   allocation, then exit WITHOUT running\n"
            f"                   SIESTA.  Use to preview/validate a job\n"
            f"                   (e.g. ``sbatch job.sbatch --dry-run``).\n"
            f"  -h               this help.\n"
            f"\n"
            f"Environment variables:\n"
            f"  MB_NP=N            same as -np N (useful for SLURM/PBS:\n"
            f"                     ``export MB_NP=\\$SLURM_NTASKS``).\n"
            f"  OMP_NUM_THREADS=N  same as -omp N (standard OMP toolchain\n"
            f"                     convention; honored if set in env).\n"
            f"\n"
            f"On 'propor: ERROR: IMAX = 0' crashes at startup, retry\n"
            f"with a smaller -np.  See the diagnostic the wrapper prints\n"
            f"on failure for specific suggestions.\n"
            f"USAGE\n"
            f"            exit 0 ;;\n"
            f"        *)\n"
            f'            echo "ERROR: unknown argument: $1 (use -h)" >&2\n'
            f"            exit 1 ;;\n"
            f"    esac\n"
            f"done\n"
            f'if ! printf %s "$_mpi_np" | grep -qE \'^[1-9][0-9]*$\'; then\n'
            f'    echo "ERROR: -np must be a positive integer; got: '
            f'\'$_mpi_np\'" >&2\n'
            f"    exit 1\n"
            f"fi\n"
            f'if ! printf %s "$_omp_threads" | grep -qE \'^[1-9][0-9]*$\'; then\n'
            f'    echo "ERROR: -omp must be a positive integer; got: '
            f'\'$_omp_threads\'" >&2\n'
            f"    exit 1\n"
            f"fi\n"
            +
            # WHERE each number came from, computed once the values are
            # final -- the --dry-run report prints these so a wrong scale
            # is caught BEFORE a queue slot is spent (user design,
            # 2026-08-13: dry-run is the pre-submission inspection).
            '_np_source="stated at prep"\n'
            'if [ "$_np_from_flag" = "1" ]; then _np_source="-np flag"\n'
            'elif [ -n "${MB_NP:-}" ]; then _np_source="MB_NP env"\n'
            'elif [ -n "${SLURM_NTASKS:-}" ]; then '
            '_np_source="SLURM_NTASKS (the sbatch reservation)"\n'
            'elif [ -n "${PBS_NP:-}" ]; then '
            '_np_source="PBS_NP (the qsub reservation)"\n'
            'fi\n'
            '_omp_source="stated at prep"\n'
            'if [ "$_omp_from_flag" = "1" ]; then _omp_source="-omp flag"\n'
            'elif [ -n "${OMP_NUM_THREADS:-}" ]; then '
            '_omp_source="OMP_NUM_THREADS env"\n'
            'elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then '
            '_omp_source="SLURM_CPUS_PER_TASK (the sbatch -c reservation)"\n'
            'fi\n'
            f"\n"
            # SIESTA's stdout role, asked rather than taken from a default.
            + _run_index_resolver(basename, ext=_stdout_role_for(".fdf"))
            + _cold_restart_block(basename, engine="siesta", label=label)
            + _runtime_status_block(basename, engine="siesta",
                                    resumes=resumes,
                                    warm=_warm_in_effect("siesta", warm),
                                     script_name=script_name)
        )

        env_prefix = (
            # EVERY path, not just the GPU one: the node's shape is
            # provenance for any trial, and a CPU sweep is the case that
            # most needs it (`plan.md` E4).
            _phys_cores_probe_block()
            + (_gpu_runtime_block() if gpu_mode else "")
            + siesta_args_block
            # GPU load-balance: derive ranks-per-GPU from the resolved
            # rank count so MPS gates on real sharing + the per-rank
            # launcher can block-distribute.  Must sit AFTER the args
            # block (needs $_mpi_np) and BEFORE the MPS block (reads
            # $_ranks_per_gpu).  See _gpu_loadbalance_block.__doc__.
            + (_gpu_loadbalance_block() if gpu_mode else "")
            + f"# MPI rank count: $_mpi_np (stated at prep: "
            f"$_mpi_np_default)\n"
            + _orbitals_per_rank_notice(n_atoms)
            + f"# --- Thread / BLAS pinning ------------------------------\n"
            f"#   * OMP_NUM_THREADS: the stated cores per rank\n"
            f"#     ({resolved_omp}); override with -omp N or by\n"
            f"#     exporting OMP_NUM_THREADS before invoking.\n"
            f"#   * BLAS pinned to 1 per rank so OMP * BLAS doesn't\n"
            f"#     oversubscribe.\n"
            f"export OMP_NUM_THREADS=$_omp_threads\n"
            f"export MKL_NUM_THREADS=1\n"
            + (
                # Hybrid MPI+OMP needs the OMP runtime told to bind --
                # ``mpirun --bind-to core`` only binds the rank
                # (cpuset), not the threads inside it.  Without these
                # two env vars SIESTA prints "OpenMP NOT bound (please
                # bind threads!)" and the OMP runtime is free to
                # migrate threads across cores, causing cache thrash
                # + cross-package traffic that defeats the binding
                # we set on mpirun.  ``close`` keeps threads near the
                # rank's master; ``cores`` says "one place per core".
                "export OMP_PROC_BIND=close\n"
                "export OMP_PLACES=cores\n"
                if gpu_mode else ""
            )
            + f""
            f"export OPENBLAS_NUM_THREADS=1\n"
            # MPS setup: enable Hyper-Q GPU sharing when (a) the binary
            # is on PATH, (b) the user hasn't opted out, and (c) we'll
            # actually have multiple ranks (single-rank MPS is pure
            # overhead).  Per-job pipe / log directories so concurrent
            # molbuilder runs don't trample each other's MPS daemon.
            # Trap on EXIT cleans up after siesta returns.
            + (
                # ``_dry_run != 1`` guard: a --dry-run must not start the
                # MPS daemon (a real GPU side-effect).  The dry-run report
                # still shows the would-be MPS state from _use_mps_str.
                # GATE: ranks > GPUs -- ANY shared GPU gets the funnel
                # (user decision 2026-08-13).  The floor-division gate
                # (`_ranks_per_gpu >= 2`) missed the uneven split: 3
                # ranks over 2 GPUs floors to 1, yet GPU0 hosts 2 ranks
                # -- sharing by driver TIME-SLICING, kernels taking
                # turns, without the concurrency MPS exists to provide.
                '# --- MPS daemon (Hyper-Q GPU sharing) -------------------\n'
                'if [ "$_use_mps_default" = "1" ] '
                '&& [ "$_mpi_np" -gt "${_ngpu:-0}" ] '
                '&& [ "${_ngpu:-0}" -ge 1 ] '
                '&& [ "${_dry_run:-0}" != "1" ]; then\n'
                '    export CUDA_MPS_PIPE_DIRECTORY="/tmp/mb-mps-$$"\n'
                '    export CUDA_MPS_LOG_DIRECTORY="/tmp/mb-mps-$$-log"\n'
                '    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" '
                '"$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null\n'
                '    # Start MPS only if the control socket for THIS\n'
                '    # pipe dir is not already present (a global daemon\n'
                '    # started outside this run uses a different dir,\n'
                '    # so this check is independent).\n'
                '    if [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then\n'
                '        # || true: under set -e a failed daemon start\n'
                '        # (perms, global daemon holding the device) killed\n'
                '        # the wrapper HERE -- before the readiness loop\n'
                '        # below, whose whole job is timing out to the\n'
                '        # graceful no-MPS fallback (R9).\n'
                '        nvidia-cuda-mps-control -d 2>/dev/null || true\n'
                '        # Daemon readiness signal: the control UNIX\n'
                '        # SOCKET file appears in the pipe directory.\n'
                '        # 2026-06-16 audit fix: the prior probe polled\n'
                '        # ``echo get_server_list | nvidia-cuda-mps-control\n'
                '        # | grep -q .`` -- BUT MPS servers are spawned\n'
                '        # by the daemon only when a CLIENT FIRST\n'
                '        # CONNECTS.  Pre-launch, ``get_server_list``\n'
                '        # returns an empty string regardless of daemon\n'
                '        # health, so the loop always timed out at 5 s\n'
                '        # and falsely reported "daemon failed to bind"\n'
                '        # on perfectly healthy hosts.  The control\n'
                '        # socket appears as soon as the daemon binds\n'
                '        # (typically <100 ms) -- that is the correct\n'
                '        # readiness signal.\n'
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
                # result.  Before this gate the readiness-poll fallback
                # at the loop above would print "MPS daemon failed to
                # bind ... falling back to no-MPS" AND THEN the line
                # below would print "MPS enabled (pipe=...)" -- two
                # contradictory messages, with the run continuing
                # without MPS but the banner claiming otherwise.
                '    if [ "$_use_mps_default" = "1" ]; then\n'
                '        echo "molbuilder: MPS enabled '
                '(pipe=$CUDA_MPS_PIPE_DIRECTORY)" >&2\n'
                '    fi\n'
                'else\n'
                '    if [ "$_use_mps_default" = "1" ] '
                '&& [ "$_ranks_per_gpu" -lt 2 ]; then\n'
                '        echo "molbuilder: MPS auto-disabled '
                '(1 rank/GPU; MPS adds overhead with no '
                'concurrency benefit)" >&2\n'
                '    fi\n'
                'fi\n'
                if gpu_mode else ""
            )
        )
        if max_memory_mb is not None and int(max_memory_mb) > 0:
            kb = int(max_memory_mb) * 1024
            env_prefix += (
                f"# Memory cap (cfg.max_memory_mb): {max_memory_mb} MB\n"
                f"ulimit -v {kb} || true  # soft cap; ignored if shell can't set it\n"
            )
        env_prefix += "\n"

        # Runtime SIESTA build probe + launcher selection.
        #
        # ``siesta --version`` (5.x +) self-reports the parallelisation
        # the binary was compiled with.  Example for a typical conda-
        # forge build:
        #
        #   Version         : 5.4.2
        #   Parallelisations: MPI
        #
        # We parse the ``Parallelisations:`` line and pick the launcher
        # accordingly:
        #
        #   MPI present      ->  mpirun -np <N> siesta   (always)
        #   OMP present      ->  bare siesta             (OMP env vars take effect)
        #   both             ->  mpirun -np <N> siesta   (hybrid)
        #   probe failed     ->  mpirun -np <N> siesta   (safe default for
        #                                                 MPI-compiled binaries)
        #   serial build     ->  bare siesta
        #
        # The probe runs ONCE per wrapper invocation and prints what
        # it found before exec, so the user sees the actual build
        # capability + the launcher choice in the log.  This adapts
        # automatically if you rebuild SIESTA with different flags --
        # no need to regenerate the wrapper.
        # WHICH binary this wrapper launches.  The engine default is
        # ``siesta``; ``resources.program`` overrides it -- the transport
        # composite's transmission stage runs ``tbtrans`` over the SAME
        # deck text, so the deck cannot carry the answer and the
        # allocation road does (jobset/model.Resources.program).
        _prog = getattr(r, "program", None) or "siesta"
        # WHAT THE LOG CALLS IT: a transmission run is TBtrans.  Its log said
        # "SIESTA binary" over a tbtrans path, and "SIESTA wall" / "SIESTA
        # exited" after it, until 2026-09-26.
        _prog_label = "TBtrans" if _prog == "tbtrans" else "SIESTA"
        env_prefix += (
            # The block NAME is job-contracts.md § 2.6's row and stays
            # stable; the binary inside is _prog.
            f"# --- Probe SIESTA build at runtime ---\n"
            f'_siesta_bin_path="$(command -v {_prog} || echo \"\")"\n'
            f'if [ -z "$_siesta_bin_path" ]; then\n'
            f"    echo \"ERROR: '{_prog}' not on PATH after activating "
            f"'{target_env}'.  Is SIESTA installed in this env?\" >&2\n"
            f"    exit 1\n"
            f"fi\n"
            # THE PROBE CANNOT BLOCK THE JOB (2026-08-25).  Three things
            # are needed and each one alone is insufficient; this is the
            # order they were found in, live on the dev workstation:
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
            #     fired -- which is what the first attempt at this fix did,
            #     and why the failing tests did not move.  A file has no
            #     EOF to wait on, so a surviving child cannot hold it.
            #  4. `-k`, because a clock that only ASKS is not a bound.
            #     Plain `timeout` sends TERM and then waits -- indefinitely,
            #     against anything that ignores it, which MPI launchers do.
            #     `-k 2` follows with KILL, and KILL cannot be declined.
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
            f'    timeout -k 2 5 {_prog} --version >"$_mb_probe_out" 2>/dev/null '
            f'</dev/null || true\n'
            f'else\n'
            f'    {_prog} --version >"$_mb_probe_out" 2>/dev/null '
            f'</dev/null || true\n'
            f'fi\n'
            f'_siesta_version_out="$(cat "$_mb_probe_out" 2>/dev/null || true)"\n'
            f'rm -f "$_mb_probe_out"\n'
            f'_siesta_ver="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Version/ {{print $2; exit}}')\"\n"
            f'_siesta_par="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Parallelisations/ {{print $2; exit}}')\"\n"
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
            f'_par_norm=" $(echo \"$_siesta_par\" '
            f'| tr "[:space:]" " " | tr ",;" "  " | tr -s " ") "\n'
            f'case "$_par_norm" in *" MPI "*) _has_mpi=1 ;; esac\n'
            f'case "$_par_norm" in *" OMP "*|*" OpenMP "*) _has_omp=1 ;; esac\n'
            # GPU mode: pin threads to cores in the same package so
            # OpenMP doesn't spill across sockets (SIESTA performance-
            # options guidance), and tell OpenMPI to place ranks one
            # per package.  On OpenMPI 5.x "package" is canonical;
            # "socket" still works as an alias.
            + (
                # PE counting hazard caught 2026-06-16 in a live
                # 212-atom Au-BDT run: the previous
                # ``ppr:K:package:PE=$_omp`` form, on Intel HT boxes,
                # allocated PE=2 *processing units* (PUs) per rank
                # mapped as HT-sibling pairs of ONE physical core.
                # Observed binding: rank 0 cpus={0,20} (core 0
                # threads), rank 1 cpus={2,22}, etc.  So 4 ranks x
                # PE=2 used only 4 physical cores (not 8), with each
                # rank's 2 OMP threads sharing one core's execution
                # units -- socket 0 idle at 20% while it should have
                # been driving 80%.
                #
                # Replace with the canonical "N physical cores per
                # rank, packed onto packages" form:
                #
                #   --map-by package:PE=$_omp_threads
                #     map ranks across packages, PE counts physical
                #     CORES (not PUs) by default in OpenMPI 5 without
                #     ``--use-hwthread-cpus``.  When the cpuset
                #     restricts to one package (numactl wrap),
                #     OpenMPI packs all ranks onto that single
                #     visible package -- correct.
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
                if gpu_mode else
                f'_mpirun_bind=""\n'
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
            + f'_siesta_target="{_prog}"\n'
            + (_gpu_per_rank_launcher_block() if gpu_mode else "")
            # ``_mb_cores``: the cores THIS launch uses -- what the job holds
            # when it was started directly, the monitor's cpu% denominator
            # (`run-reports.md` § 2.1a).  Decided here, with the launcher:
            # ranks x threads only for a hybrid build, and one core for a
            # serial one, whatever -np and -omp said.
            + f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_siesta_target"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ($_mpi_np ranks x $_omp_threads OMP threads)"\n'
            f'        _mb_cores=$(( _mpi_np * _omp_threads ))\n'
            f'    else\n'
            f'        _launch_note="pure MPI ($_mpi_np ranks; OMP setting irrelevant to this binary)"\n'
            f'        _mb_cores=$_mpi_np\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="{_prog}"\n'
            f'    _launch_note="OMP-only build ($_omp_threads threads)"\n'
            f'    _mb_cores=$_omp_threads\n'
            f'elif [ -z "$_siesta_par" ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_siesta_target"\n'
            f'    _launch_note="MPI fallback (probe inconclusive; safe default for MPI-compiled SIESTA)"\n'
            f'    _mb_cores=$_mpi_np\n'
            f'else\n'
            f'    _launch_cmd="{_prog}"\n'
            f'    _launch_note="serial build (no parallelisation compiled in)"\n'
            f'    _mb_cores=1\n'
            f"fi\n"
            f"\n"
        )

        # Human-readable banner printed at run time so the user sees
        # the rank count / threading / cwd / command + BUILD probe
        # results before SIESTA spends 30 seconds reading the .fdf.
        env_prefix += (
            f'echo "===== molbuilder {_prog_label} run-wrapper ====="\n'
            f'echo "  Date          : $(date -Iseconds)"\n'
            f'echo "  Host          : $(hostname)"\n'
            f'echo "  Cwd           : $(pwd)"\n'
            f'echo "  Conda env     : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'echo "  {_prog_label + " binary":<14}: $_siesta_bin_path"\n'
            f'echo "  {_prog_label + " version":<14}: ${{_siesta_ver:-unknown}}"\n'
            f'echo "  Build paral.  : ${{_siesta_par:-unknown}}"\n'
            f'echo "  Launch mode   : $_launch_note"\n'
            # WHAT A RETRY WILL ACTUALLY DO ON THIS DECK.  It said
            # "--continue warm-resume" whatever the deck instructed, so a
            # `restart: clean` stage announced a warm resume and then re-ran
            # cold.  The budget is the user's (it travels; see the note in
            # `render_run_wrapper`); the description is the deck's.
            + (f'echo "  Retry policy  : up to {continue_retries} '
               f'retry(s) on non-convergence -- {_retry["policy"]}"\n'
               if continue_retries and continue_retries > 0 else
               f'echo "  Retry policy  : none (halt on non-convergence)"\n')
            + f'echo "  Threading     : OMP_NUM_THREADS=$_omp_threads, '
            f'OPENBLAS=1, MKL=1"\n'
            # GPU mode: print a brief monitoring hint so the user has
            # nvidia-smi commands at hand when they start the run.
            + ((
                # THE single authoritative GPU-resource summary, printed
                # with the RESOLVED launch values ($_mpi_np / $_omp_threads
                # / final $_use_mps_default / $_ranks_per_gpu) so it always
                # matches what runs -- replacing the old pre-resolution
                # probe advisory that could contradict it (one unified line
                # for the user).  $_gpu_numa is the generation-time GPU0
                # NUMA probe (per-rank placement is logged per rank below).
                '_mps_str_now="off"; '
                '[ "$_use_mps_default" = "1" ] && _mps_str_now="on"\n'
                'echo "  GPU resources : GPU mode (ELPA-CUDA, no NCCL) -- '
                'chosen $_mpi_np ranks × $_omp_threads threads '
                '($(( _mpi_np * _omp_threads )) cores); mps=$_mps_str_now; '
                'ranks/GPU=${_ranks_per_gpu:-?}; GPU0 NUMA=$_gpu_numa"\n'
                # TUNE BY MEASURING, not by asking for a guess.  This
                # named `molbuilder envs advise siesta-gpu` until 2026-09-12;
                # that command guessed the answer `jobset prep bench`
                # MEASURES, and probed whichever host it ran on -- the login
                # node on a cluster, which is the machine the job will not
                # run on.  The guess is gone, and so is the GPU policy the
                # MOLBUILDER_* knobs overrode (2026-10-02).
                'echo "                # tune: -np / -omp / --mps / --no-mps "'
                '"(or measure it: prep bench, for this stage)"\n'
                # IMPORTANT: keep the command on its own line so the
                # user can copy-paste it directly into a shell.  An
                # earlier banner shape put ``(sm%, mem%, ...)`` after
                # the command on the same line and bash interpreted
                # the ``(`` as a subshell open + ``%`` as a format op
                # when the user pasted it -- "syntax error near
                # unexpected token \`(\`".  Annotation goes on the
                # NEXT line, prefixed with ``# `` so even if it's
                # included accidentally in a paste the shell treats
                # it as a comment.
                'echo "  GPU monitor   : nvidia-smi dmon -s pucvmet -d 1"\n'
                'echo "                # columns: sm%, mem%, clk, temp, power"\n'
                'echo "                # if sm% bounces 0->100->0 across ranks, MPS may help"\n'
                'echo "                # (this ELPA build has no NCCL; multi-rank-per-GPU benefits from MPS)"\n'
            ) if gpu_mode else "")
            # ---- Mode + constraints (post-cold, post-run-index) ----
            # Surfaces the silent-warm-restart class explicitly so the
            # user can see whether the engine is starting clean,
            # resuming from prior state, or being asked to honor
            # frozen-atom constraints.  Both lines are read from the
            # on-disk script + restart files at runtime so a manual
            # .fdf edit shows the EDITED state, not the generation-
            # time snapshot (the "what you see is what runs" rule).
            + f'echo "  Mode          : $_mode"\n'
            + f'echo "  Constraints   : $_constraints"\n'
            + f'echo "  Command       : $_launch_cmd {script_name} > $_out_file"\n'
            + f'echo "  Stdout        : $_out_file (live; tail -f to follow)"\n'
            + f'echo "========================================="\n'
            f"\n"
        )
    else:                                          # pyscf
        inner = f"python {script_name} > $_out_file 2>&1"
        description = "PySCF run"
        # THE THREADS ARE STATED (user, 2026-10-02; `architecture.md`
        # § 5.2) -- the run card's `threads`, or `--cpus-per-task` on the
        # prep, asked before anything is written (`prep_inputs.
        # launch_refusal`).  The chain below ended at THIS NODE'S PHYSICAL
        # CORES until then: a thread count nobody stated.
        resolved_omp = int(omp_threads)

        # Argument parsing: PySCF gets --continue / --force +
        # a -h.  No engine-specific flags (PySCF doesn't have an
        # MPI rank knob; its thread count is the stated one).
        pyscf_args_block = (
            _continue_force_args_parser("PySCF wrapper")
            + f"# --- PySCF wrapper argument parsing -------------\n"
            f'_dry_run=0\n'
            f'_omp_flag=""\n'
            f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        --dry-run|--dryrun)\n"
            f'            _dry_run=1; shift ;;\n'
            # -omp / -np are what `jobset launch` hands EVERY .run.sh
            # (submit._run_sh_args).  This parser used to reject them as
            # unknown and exit 1, so `submit --mode direct` on a PySCF
            # job with resources set died before Python started -- on the
            # workstation posture, where direct mode is the normal way to
            # run.  -omp is the thread count and is honoured; -np is
            # accepted and reported, because PySCF is OpenMP-only and a
            # silently swallowed rank count would let a user believe they
            # had asked for something.
            f"        -omp|--omp)\n"
            f'            _omp_flag="$2"; shift 2 ;;\n'
            f"        -np|--np)\n"
            f'            if [ "${{2:-1}}" != "1" ]; then\n'
            f'                echo "molbuilder: -np $2 ignored -- PySCF is '
            f'OpenMP-only (no MPI ranks); use -omp for thread count" >&2\n'
            f'            fi\n'
            f'            shift 2 ;;\n'
            f"        -h|--help)\n"
            f'            cat <<USAGE\n'
            f'Usage: bash $(basename "$0") [--continue|-c] [--force|-f] [--cold] [--dry-run] [-h]\n'
            f"\n"
            f"  --continue, -c   resume from prior run.  Scans existing\n"
            f"                   -runN.pyscf.log files and writes\n"
            f"                   -run(N+1).pyscf.log.\n"
            + (
                f"                   This deck reads prior state: its\n"
                f"                   chkfile init-guess and geometry\n"
                f"                   reads are gated on its described\n"
                f"                   ``continue`` (run-identity.md § 4\n"
                f"                   rule 2).\n"
                if _py_reads_prior else
                f"                   This deck emits NO prior-state\n"
                f"                   read (described ``clean``), so the\n"
                f"                   run index advances and the\n"
                f"                   calculation starts from the deck's\n"
                f"                   own coordinates.\n"
                if _py_reads_prior is False else
                f"                   This vibration deck reads no prior\n"
                f"                   engine state at start: each run\n"
                f"                   recomputes from its own relaxation\n"
                f"                   (or your already_relaxed\n"
                f"                   assertion); --continue only\n"
                f"                   advances the run index.\n"
            )
            + f"  --force, -f      start over from -run0 even if prior\n"
            f"                   runs exist.  Old files are NOT deleted;\n"
            f"                   the existing -run0.pyscf.log is\n"
            f"                   overwritten.  Prior ``.chk`` warm-start\n"
            f"                   files STAY on disk -- "
            + ("and this deck's\n"
               f"                   ``continue`` reads them.\n"
               if _py_reads_prior else
               "and this deck\n"
               f"                   does not read them.\n")
            + _cold_usage_entry(
                warm_examples=".chk and _optimized.xyz among them")
            + f"  -omp N           OpenMP threads.  Highest precedence;\n"
            f"                   otherwise OMP_NUM_THREADS, else the\n"
            f"                   scheduler's allocation, else the\n"
            f"                   {resolved_omp} stated at prep.\n"
            f"  -np N            accepted and IGNORED -- PySCF is\n"
            f"                   OpenMP-only.  Present because\n"
            f"                   `jobset launch` passes it to every\n"
            f"                   run script; N>1 prints a note.\n"
            f"  --dry-run        resolve + log the launch command, then\n"
            f"                   exit WITHOUT running PySCF.\n"
            f"  -h               this help.\n"
            f"USAGE\n"
            f"            exit 0 ;;\n"
            f"        *)\n"
            f'            echo "ERROR: unknown argument: $1 (use -h)" >&2\n'
            f"            exit 1 ;;\n"
            f"    esac\n"
            f"done\n"
            f"\n"
            # --- Thread sizing: the WRAPPER decides ------------------
            # Resolved here and EXPORTED, so the deck's own chain sees it
            # and the banner can state the number before Python starts.
            #
            # The last rung is the STATED count, baked at prep -- never the
            # node.  Asking the machine when a scheduler has granted a slice
            # of it is how a job on a 128-core node claimed 128 threads for
            # the 8 cores it owned; and with no scheduler it was a thread
            # count nobody stated (2026-10-02).
            + "# --- OpenMP thread sizing (allocation first) ---\n"
            + _phys_cores_probe_block()
            # THE STATED COUNT, BY ITS NAME (`_STATED_COUNTS`): what
            # `stated_counts` reads back for A13 -- the SIESTA script's name
            # for the same fact.  PySCF runs one process, so it states no
            # rank count.  It was baked inline in the last rung until
            # 2026-10-05, and a PySCF run's preview showed no end point.
            + f"{_STATED_COUNTS[1]}={resolved_omp}\n"
            + 'if [ -n "$_omp_flag" ]; then\n'
              '    _omp_threads="$_omp_flag"; _omp_from="-omp flag"\n'
            # THE SAME RUNGS THE SCRIPT'S OWN CHAIN READS, from the one list
            # (`runtime_info.THREAD_SOURCES`; `running-a-job.md` § 3.2).
            # Spelled here by hand until 2026-10-05, it lacked PBS_NCPUS
            # and NSLOTS while a comment called the chains identical -- and
            # because this branch EXPORTS OMP_NUM_THREADS, the script's
            # chain, which had them, never reached them: under qsub the
            # engine got the whole node, the 128-threads-for-8-cores bug
            # this block exists to prevent.  The last rung is this run
            # script's alone: the count stated at prep, which the deck it
            # runs always receives exported.
            + "".join(f'elif [ -n "${{{var}:-}}" ]; then\n'
                      f'    _omp_threads="${var}"; _omp_from="{var}"\n'
                      for var in THREAD_SOURCES)
            + 'else\n'
              f'    _omp_threads="${_STATED_COUNTS[1]}"; '
              '_omp_from="stated at prep"\n'
              'fi\n'
              'export OMP_NUM_THREADS="$_omp_threads"\n'
              '\n'
            # PySCF's stdout role, from the catalogue -- it is not ``.out``
            # so the Results-tab inspector dispatcher can tell PySCF output
            # apart from SIESTA's.  Per docs/web/tabs.md (Phase C,
            # 2026-06-07); the literal left on 2026-09-18.
            + _run_index_resolver(basename, ext=_stdout_role_for(".py"))
            + _cold_restart_block(basename, engine="pyscf", label=label)
            + _runtime_status_block(basename, engine="pyscf",
                                    warm=_warm_in_effect("pyscf", warm),
                                     script_name=script_name)
        )

        # Same human-readable banner pattern as SIESTA -- the script
        # itself logs its own runtime info but the wrapper covers
        # the "did it even start" window before Python imports.
        env_prefix = (
            pyscf_args_block
            + f'echo "===== molbuilder PySCF run-wrapper ====="\n'
            f'echo "  Date        : $(date -Iseconds)"\n'
            f'echo "  Host        : $(hostname)"\n'
            f'echo "  Cwd         : $(pwd)"\n'
            f'echo "  Conda       : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'echo "  OMP threads : $_omp_threads (from $_omp_from; '
            f'node has $_phys_cores physical cores)"\n'
            # ---- Mode + constraints (mirrors SIESTA; see siesta
            # banner above for the rationale). ----
            f'echo "  Mode        : $_mode"\n'
            f'echo "  Constraints : $_constraints"\n'
            f'echo "  Command     : python {script_name} > $_out_file"\n'
            f'echo "  Stdout      : $_out_file"\n'
            f'echo "  Logs        : see <basename>.molwatch.log (script writes its own)"\n'
            f'echo "========================================"\n'
            f"\n"
        )

    # Per docs/execution/running-a-job.md § 5.2: the wrapper is a
    # self-contained shell script.  At generate time the generator reads the
    # TARGET's preamble and activation off its record and bakes them
    # VERBATIM into the wrapper.  At runtime the wrapper does no discovery,
    # no probing, no config-file reads, no env-var-driven behaviour
    # switching.  If anything fails, ``set -euo pipefail`` aborts with the
    # real bash error.  The activation has no default: no record stating
    # one, no wrapper.
    # The BUNDLE'S scope, stated by the caller since the layout repair
    # (roadmap 7.10 M1): the script is born in its job directory now, and
    # a scope derived from its parent would look for environment.json in
    # the job dir -- one level below the file.  The
    # parent stays as the fallback for a caller that points at a script
    # sitting wherever its bundle root is (tests).
    _project_dir = project_dir if project_dir is not None else (
        script_path.parent if script_path.parent.exists() else None)
    # WHOSE MACHINE IS THIS SCRIPT FOR?
    #
    # (The parameter is `machine_record`, NOT `target_env`: this function
    # already binds a local `target_env` meaning *the conda env NAME to
    # activate*.  A parameter of that name is silently overwritten by it a
    # thousand lines above, which turned this branch into a string test and
    # made every render refuse.  Same word, two meanings, one scope.)
    #
    # A wrapper is generated on one machine and executed on another -- that
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
    # and `jobset probe` copies it into the record it writes.
    _rec_sg = machine_record
    if _rec_sg is None and _project_dir is not None:
        from .scheduler import machine_for
        _rec_sg = machine_for(_project_dir)
    _tsg = dict(getattr(_rec_sg, "env_init", None) or {})
    if not _tsg.get("activation"):
        from .scheduler.record import probe_command
        raise WrapperError(
            "no record says how a shell enters an environment on the machine "
            "this script is for, so no wrapper can be written -- the "
            "activation has no default (docs/configuration.md § 4).  On "
            "that machine, declare it in molbuilder.json --\n"
            "    \"env_init\": {\"activation\": \"conda activate\", "
            "\"preamble\": \"source <conda root>/etc/profile.d/conda.sh\"}\n"
            "  (or \"source activate\" after \"module load mamba\" where a "
            "module gives the toolchain) -- then "
            f"`{probe_command(None)}`, and copy its record here when that "
            "machine is not this one.")
    _preamble_chunks = ([("target", _tsg["preamble"].rstrip("\n"))]
                        if _tsg.get("preamble") else [])
    _activation_form = _tsg["activation"]
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
        # that: on the prepping machine the file is right there.  The only
        # code that runs on the TARGET is this script, so the check has to
        # be here.
        #
        # Without it bash says
        #     line 196: /home/.../conda.sh: No such file or directory
        # and dies -- naming neither the config key that put the path
        # there, nor the machine it was baked on, nor what to do.  The
        # generate-time warning above does not catch this and never could:
        # it fires when the preamble does NOT name a conda hook, and this
        # preamble names one.
        _srcs = _preamble_source_targets(_preamble_chunks)
        _guard = ""
        if _srcs:
            _lines = ["# --- Preamble preflight (paths baked elsewhere) ---"]
            for _pth in _srcs:
                _lines += [
                    f'if [ ! -r "{_pth}" ]; then',
                    f'    _log ERROR "baked preamble sources a file that '
                    f'does not exist on this machine:"',
                    f'    _log ERROR "    {_pth}"',
                    # SINGLE-quoted: bash runs backticks inside a
                    # double-quoted string, and the first version of this
                    # message had `prep` and `module load mamba` in it --
                    # so the guard fired and printed a sentence with two
                    # holes in it.  Caught by RUNNING the generated
                    # script, not by reading it.
                    f"    _log ERROR 'It was baked verbatim from the "
                    f"preamble of the record prep read, and this machine "
                    f"is not the one that record describes.'",
                    f"    _log ERROR 'Fix: environment.json beside task.json, "
                    f"the record this calculation was prepped with, carries "
                    f"this preamble -- edit its env_init.preamble (for "
                    f"example: module load mamba), or correct the record of "
                    f"that machine and prep the calculation anew, from the "
                    f"state saved before its first prep; or edit the source "
                    f"line below.'",
                    f'    exit 78',           # EX_CONFIG
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

    env_activation = (
        f"# Structured log helper.\n"
        f"_log() {{\n"
        f"    printf '{LOG_LINE}\\n' \"$(date '+{LOG_CLOCK}')\" \"$1\" \"$2\" >&2\n"
        f"}}\n"
        f"\n"
        f"# Single unified EXIT cleanup (one trap -- a second ``trap ...\n"
        f"# EXIT`` would REPLACE the first, so all teardown lives here).\n"
        f"# No-ops unless the relevant vars were set, so it is safe for\n"
        f"# CPU / PySCF / non-MPS runs.  Cleans (a) the per-rank GPU\n"
        f"# launcher temp file and (b) the MPS daemon + its pipe/log dirs.\n"
        # STOPPING THE MONITOR, in one place: the signal says why -- TERM at
        # the job's end, USR1 when one attempt is retried in place, which is
        # not an ending (`run-reports.md` § 2) -- and the wait lets it write
        # its closing [STATUS] / [UTIL-SUMMARY] before this shell exits and a
        # scheduler reaps the job's processes.  Bounded (~10 s): a monitor
        # that does not exit by then is left to its own watch-pid exit.
        f"_mb_stop_monitor() {{\n"
        f'    [ -n "${{_monitor_pid:-}}" ] || return 0\n'
        f'    kill "-$1" "$_monitor_pid" 2>/dev/null || return 0\n'
        f"    _mb_w=0\n"
        f'    while kill -0 "$_monitor_pid" 2>/dev/null '
        f'&& [ "$_mb_w" -lt 50 ]; do\n'
        f"        sleep 0.2\n"
        f"        _mb_w=$((_mb_w + 1))\n"
        f"    done\n"
        f"}}\n"
        f"_mb_cleanup_ran=0\n"
        f"_mb_cleanup() {{\n"
        # Idempotence guard: a caught signal runs cleanup and exits,
        # which fires the EXIT trap and would run it AGAIN (D17,
        # 2026-08-12 -- before the signal trap below, a walltime
        # SIGTERM ran no cleanup at all: MPS daemon + pipe dirs leaked
        # and neither log said "killed").
        f'    [ "${{_mb_cleanup_ran:-0}}" = "1" ] && return 0 || true\n'
        f"    _mb_cleanup_ran=1\n"
        # (a ``_mb_claim_runwrap_log`` hook call sat here until U19,
        # guarded by ``command -v`` -- for a function NO emitter ever
        # defined.  A hook nothing defines is dead weight in every
        # wrapper and a false lead in every debugging session.)

        # ``|| true`` on every arm: this trap runs UNDER set -e, and bash
        # exits on the failure of the command following the final ``&&`` --
        # a monitor already dead (the COMMON case at cleanup) killed the
        # trap mid-body and skipped the MPS teardown below (R9,
        # 2026-08-12; the warm-retry's identical kill was already
        # guarded).  Same for a vanished MPS control daemon under
        # pipefail.
        f"    _mb_stop_monitor TERM || true\n"
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
        f'    _log WARN "caught SIGTERM/SIGINT (scheduler kill or '
        f'Ctrl-C) -- cleaning up" || true\n'
        f"    _mb_cleanup\n"
        f"    exit 143\n"
        f"}}\n"
        f"trap _mb_on_signal TERM INT\n"
        f"\n"
        f"_log STAGE \"{WRAPPER_LOG_START}\"\n"
        f'_log INFO "timestamp:  $(date \'+%Y-%m-%d %H:%M:%S %Z\')"\n'
        f'_log INFO "hostname:   $(hostname)"\n'
        f'_log INFO "user:       ${{USER:-?}}"\n'
        f'_log INFO "cwd:        $(pwd)"\n'
        f'_log INFO "script:     $0"\n'
        f'_log INFO "argv:       $0 $*"\n'
        f'_log INFO "log file:   $_runwrap_log"\n'
        f"# Scheduler context -- only emit if the var is set.  These are\n"
        f"# read for diagnostic logging + launch tuning (running-a-job.md\n"
        f"# 2.1-2.2a); they do NOT alter activation or preamble.\n"
        f'for _v in SLURM_JOB_ID SLURM_NTASKS SLURM_CPUS_PER_TASK \\\n'
        f"          SLURM_JOB_NODELIST SLURM_GPUS SLURM_JOB_GPUS \\\n"
        f"          PBS_JOBID PBS_NP PBS_NODEFILE; do\n"
        f"    _v_val=\"${{!_v:-}}\"\n"
        f'    [ -n "$_v_val" ] && _log INFO "$_v=$_v_val"\n'
        f"done\n"
        f"\n"
        f"# The env bootstrap (preamble ``module load``, conda/mamba\n"
        f"# ``activate``) sources EXTERNAL scripts not under our control.\n"
        f"# Conda activate.d hooks (e.g. cuda-nvcc's, which references an\n"
        f"# unset NVCC_PREPEND_FLAGS) abort under the wrapper's ``set -u``.\n"
        f"# Disable nounset for the bootstrap ONLY; restore it before our\n"
        f"# own logic (where -u still catches real bugs).\n"
        f"#\n"
        f"# NOTE `set -e` IS NOT DISABLED HERE, AND MUST NOT BE.  This\n"
        f"# script runs unattended on a compute node, on an allocation.  A\n"
        f"# preamble that fails means the job is about to run in the wrong\n"
        f"# environment with nobody reading the output -- dying at the\n"
        f"# failed line is the cheap outcome (running-a-job.md 2.0a).\n"
        f"set +u\n"
        f"# A help request runs NONE of the bootstrap: asking what a script\n"
        f"# does must not require a working preamble or activation -- a\n"
        f"# broken module line would kill -h at the activation, three\n"
        f"# screens before the usage text it asked for (U10, 2026-08-12).\n"
        f'if [ "$_mb_help" = "0" ]; then\n'
        f"{_preamble_block}"
        f"# --- Activation (verbatim from the target's record) ---\n"
        f'_log STAGE "{_activation_form} {target_env}"\n'
        f"{_activation_form} {target_env}\n"
        f"fi\n"
        f"set -u\n"
        f"\n"
        f'if [ "$_mb_help" = "0" ]; then\n'
        f'  _log INFO "CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-<unset>}}"\n'
        f'  _log INFO "CONDA_PREFIX=${{CONDA_PREFIX:-<unset>}}"\n'
        f'  _log INFO "which python: $(command -v python 2>/dev/null || echo \'(not on PATH)\')"\n'
        # DID THE ACTIVATION TAKE?  Same block, because it is the same
        # concern: what the environment actually IS after the bootstrap.
        # Logging it was only half the job -- a line a human reads
        # afterwards, if anyone looks.
        #
        # This wrapper is what the scheduler runs on a compute node, in a
        # fresh non-interactive shell that inherits nothing, with nobody
        # watching.  A manager can return 0 and leave the system python on
        # PATH (`source activate` under a mamba 2.x module does exactly
        # that), and the cost of finding out later is a queue wait plus MPI
        # start-up.  SIESTA has refused by name since the build probe
        # landed (`command -v siesta`); PySCF ran the deck regardless and
        # surfaced it as `ModuleNotFoundError: pyscf` from inside the
        # script -- the same failure the jobset launcher produced on Sol
        # (2026-08-21).  Both branches now make the same promise.
        + (f'  if ! python -c "import pyscf" >/dev/null 2>&1; then\n'
           f'    echo "ERROR: python cannot import pyscf after activating '
           f'\'{target_env}\' -- the environment did not take, or PySCF is '
           f'not installed in it." >&2\n'
           f'    echo "    CONDA_DEFAULT_ENV : ${{CONDA_DEFAULT_ENV:-<unset>}}" >&2\n'
           f'    echo "    CONDA_PREFIX      : ${{CONDA_PREFIX:-<unset>}}" >&2\n'
           f'    echo "    which python      : $(command -v python 2>/dev/null '
           f'|| echo \'(none)\')" >&2\n'
           f"    exit 1\n"
           f"  fi\n"
           if category == "pyscf" else "")
        + f"fi\n"
        f"\n"
    )

    # Launch + diagnostics.  For SIESTA we run the command (not
    # exec) so we can inspect the .out for ``propor: ERROR: IMAX = 0``
    # on failure and print a targeted retry hint.  Layer-on-top
    # cost: one extra bash process for the wrapper's lifetime; cheap.
    # PySCF ran through `exec` until 2026-09-08 -- "the original exec is
    # preserved, no diagnostic surface there yet".  The CONCLUSION MARKER is
    # the reason that "yet" ran out: `exec` replaces this shell, so nothing
    # can run afterwards, and the marker is by definition the wrapper's LAST
    # ACT.  `job-contracts.md` § 2.2 and `project-layout.md` § 1.6 state it
    # for the wrapper with no engine qualifier -- and "absent means killed",
    # so a PySCF run that finished cleanly was signalling that it had been
    # force-stopped.  MEASURED 2026-09-08: the marker's reader answered
    # 'rc=0 at ...' for a finished SIESTA attempt and None for an identically
    # finished PySCF one, which is what `submit.py` refuses a ladder on.
    # The cost is the one already accepted above: one extra bash process.
    if category == "siesta":
        # Always-on launch-command audit log + the --dry-run preview, both
        # extracted into named block-emitters (see their docstrings for
        # the goal/contract).  Order: log the resolved command, then the
        # dry-run guard (exits before launch), then the real launch.
        # The cause a failure is asked about is the table's own marker.
        from .parse.engines import siesta_grammar as _G
        launch_block = (
            _siesta_resolved_log_block(script_name, gpu_mode)
            + _siesta_dry_run_block(script_name, gpu_mode)
            + _siesta_scf_timing_func()
            + f"# --- Launch SIESTA + capture exit -----------------------\n"
            f"# `set +e` lets us inspect the exit code; on a failure the\n"
            f"# ending is asked of the framework (_mb_ending) and a stop by\n"
            f"# propor gets its retry suggestion.  Then we re-exit with\n"
            f"# SIESTA's code.\n"
            f"# stdout is piped through _mb_scf_tee, which writes the .out\n"
            f"# AND the per-iteration .scf-timing.log (running-a-job.md § 4.1); SIESTA's\n"
            f"# stderr stays on the wrapper's stderr (runwrap log).  We read\n"
            f"# ${{PIPESTATUS[0]}} so awk never masks SIESTA's exit code.\n"
            f'_scf_timing_log="${{_out_file%.out}}.scf-timing.log"\n'
            f'_log INFO "scf timing  : per-iteration stamps -> '
            f'$_scf_timing_log"\n'
            + _ending_question_func()
            + _monitor_block(watch_label, watch_stage, notify_on_scf,
                             notify_every_hours, notify_channels,
                             notify_report,
                             cores="$_mb_cores", gpu=gpu_mode,
                             unwatchable=unwatchable)
            + _finish_check_block(finish, basename)
            + (f'_siesta_retry=${{MB_RETRY_N:-0}}\n'
               f'_siesta_retry_max={continue_retries}\n'
               f'# Retry: re-exec this wrapper with --continue (advance the\n'
               f'# run-index; what the engine reads back is the banner\'s\n'
               f'# retry line -- warm, cold, or from the first step of a run\n'
               f'# that does not resume).  Original args are preserved MINUS the\n'
               f'# continuation flags: --force would reset the run-index\n'
               f'# sequence and --cold would move aside the very warm-start\n'
               f'# files the retry needs.  MB_RETRY_N is exported so it\n'
               f'# survives the exec -> bounded recursion.  The monitor is\n'
               f'# stopped first, as a retry and not an ending (exec skips\n'
               f'# the EXIT trap; the retried run starts its own).\n'
               f'_mb_warm_retry() {{\n'
               f'    _mb_next=$((_siesta_retry + 1))\n'
               f'    echo "" >&2\n'
               f'    echo "=== $1; {_retry["does"]} '
               f'(retry $_mb_next/$_siesta_retry_max) with --continue ===" >&2\n'
               f'    echo "" >&2\n'
               f'    _mb_stop_monitor USR1 || true\n'
               f'    export MB_RETRY_N=$_mb_next\n'
               f'    _mb_retry_args=()\n'
               f'    for _mb_a in ${{_mb_orig_args[@]+"${{_mb_orig_args[@]}}"}}; do\n'
               f'        case "$_mb_a" in\n'
               f'            --continue|-c|--force|-f|--cold|--from-scratch) ;;\n'
               f'            *) _mb_retry_args+=("$_mb_a") ;;\n'
               f'        esac\n'
               f'    done\n'
               f'    exec bash "$_mb_self" --continue '
               f'${{_mb_retry_args[@]+"${{_mb_retry_args[@]}}"}}\n'
               f'}}\n'
               if continue_retries and continue_retries > 0 else "")
            + f"_t_start=$(date +%s.%N)\n"
            f"set +e\n"
            f'$_launch_cmd {script_name} | _mb_scf_tee "$_out_file" '
            f'"$_scf_timing_log"\n'
            f"_siesta_exit=${{PIPESTATUS[0]}}\n"
            f"set -e\n"
            f"_t_end=$(date +%s.%N)\n"
            f'_siesta_wall=$(awk -v a="$_t_start" -v b="$_t_end" '
            f"'BEGIN{{printf \"%.1f\", b-a}}')\n"
            # The engine's wall time, and how many SCF rows the tee took --
            # of both phases.  The seconds PER ITERATION are the timing
            # instrument's, one phase at a time (`parse/instruments/
            # scf_timing.py`, `model/parse.md` § 5c.1); a total/N here was a
            # second answer, and across a device's two phases it is neither.
            f'if [ -f "$_scf_timing_log" ]; then\n'
            f'    _n_scf=$(wc -l < "$_scf_timing_log" | tr -d " ")\n'
            f'else\n'
            f'    _n_scf=0   # the engine stopped before any SCF row\n'
            f'fi\n'
            f'case "$_n_scf" in ""|*[!0-9]*) _n_scf=0 ;; esac\n'
            f'_log INFO "benchmark: {_prog_label} wall ${{_siesta_wall}}s, '
            f'${{_n_scf}} SCF rows in $_scf_timing_log"\n'
            f"\n"
            f'if [ "$_siesta_exit" -ne 0 ]; then\n'
            f"    echo \"\"\n"
            f'    echo "===== {_prog_label} exited with code $_siesta_exit =====" >&2\n'
            f'    _mb_ending_able || _log WARN "{_ENDING_UNREADABLE}"\n'
            f'    _mb_said=$(_mb_ending) || _mb_said=""\n'
            f'    [ -z "$_mb_said" ] || echo "how it ended: $_mb_said" >&2\n'
            f'    if _mb_ending stopped-by "{_G.PROPOR_MARKER}"; then\n'
            f"        cat <<HINT >&2\n"
            f"\n"
            f"SIESTA crashed with 'propor: ERROR: IMAX = 0' during startup.\n"
            f"\n"
            f"IMAX=0 means one of SIESTA's per-orbital / per-rank tables\n"
            f"came out EMPTY.  It has two common causes -- check them in\n"
            f"this order, most fundamental first.  Do NOT just lower -np\n"
            f"reflexively: that masks a defective pseudopotential and gives\n"
            f"silently-wrong physics.\n"
            f"\n"
            f"1) DEFECTIVE / MISMATCHED PSEUDOPOTENTIAL  (check this FIRST)\n"
            f"   A pseudo with a null Kleinman-Bylander projector (ekb=0 on\n"
            f"   a whole l-channel) trips IMAX=0.  In $_out_file find each\n"
            f"   species' 'PSML: Kleinman-Bylander projectors' table and look\n"
            f"   for 'rc=  0.000010   Ekb=  0.000000' rows -- a valence\n"
            f"   element missing its p or d channel is broken.  Fix: replace\n"
            f"   that .psml with a vetted one (PseudoDojo; match the rest of\n"
            f"   your set's generator version + XC).  Screen the whole set:\n"
            f"     python -m molbuilder pseudo check <pseudo-dir>\n"
            f"\n"
            f"2) MPI RANK COUNT  (np IS a legitimate tunable here)\n"
            f"   If the pseudos are clean, some -np values leave trailing\n"
            f"   ranks empty in the orbital/projector distribution.  Retry\n"
            f"   at a different -np (lower; powers of 2 are safest):\n"
            f"     bash {basename}.run.sh -np 8\n"
            f"     bash {basename}.run.sh -np 4\n"
            f"   The same clean .fdf can fail at one -np and pass at another.\n"
            f"\n"
            f"HINT\n"
            f"    fi\n"
            f'    echo "the output: $_out_file -- this wrapper\'s log: '
            f'$_runwrap_log" >&2\n'
            + (f'    # SCF_NOT_CONV abort: with SCF.MustConverge (SIESTA\n'
               f'    # default true) an unconverged SCF stops the run\n'
               f'    # NON-zero after banking the density matrix, and SIESTA\n'
               f'    # says so -- "(required)" on its SCF_NOT_CONV line, then\n'
               f'    # ABNORMAL_TERMINATION (Src/siesta_forces.F90) -- so the\n'
               f'    # retry asks whether THAT stopped it.  A --continue\n'
               f'    # restart resumes the SCF from that .DM with a fresh\n'
               f'    # iteration budget -- unless the run cannot resume: a\n'
               f'    # force-constant run restarts at its first step, and the\n'
               f'    # banner and the retry message say so.  Crash classes (propor\n'
               f'    # IMAX, generic aborts) are NOT retried -- rerunning\n'
               f'    # cannot fix a defective pseudo or a bad rank count.\n'
               f'    if [ "$_siesta_retry" -lt "$_siesta_retry_max" ] \\\n'
               f'       && _mb_ending stopped-by "{_G.SCF_NOT_CONV_MARKER}"; then\n'
               f'        _mb_warm_retry "SCF did not converge '
               f'(SIESTA aborted under SCF.MustConverge)"\n'
               f'    elif [ "$_siesta_retry" -gt 0 ] \\\n'
               f'       && _mb_ending stopped-by "{_G.SCF_NOT_CONV_MARKER}"; then\n'
               f'        echo "SCF still unconverged after '
               f'$_siesta_retry_max retry(s); '
               + f'{_retry["after"]}.'
               + '" >&2\n'
               f'    fi\n'
               if continue_retries and continue_retries > 0 else "")
            + f'    # CONCLUDED -- an error is a conclusion (project-\n'
            f'    # layout.md 1.6, "the other file"): the engine returned\n'
            f'    # and this process gets to say goodbye.  MAIN LINE ONLY,\n'
            f'    # never the cleanup trap: a walltime SIGTERM runs the\n'
            f'    # trap, and a forced stop must leave NO marker.\n'
            f'    printf "rc=%s at %s\\n" "$_siesta_exit" "$(date)" '
            f'> "{basename}-run${{_run_n}}.concluded"\n'
            + f'    exit "$_siesta_exit"\n'
            f"fi\n"
            + (f"\n"
               f"# --- Geometry-cap check + warm-retry "
               f"(up to {continue_retries} retries) ---\n"
               f"# A relaxation that exhausts its MD step budget UNCONVERGED\n"
               f"# still exits 0 and prints its final geometry as unrelaxed\n"
               f"# (a converged one as relaxed; Src/siesta_analysis.F90).\n"
               f"# Warm --continue resumes from the banked .XV/.DM/.CG with\n"
               f"# a fresh step budget.  A single point prints neither, so\n"
               f"# it cannot false-trigger.\n"
               f'_mb_ending_able || _log WARN "{_ENDING_UNREADABLE}"\n'
               f'if [ "$_siesta_retry" -lt "$_siesta_retry_max" ] \\\n'
               f'   && _mb_ending relaxation-capped; then\n'
               f'    _mb_warm_retry "geometry relaxation hit its step cap '
               f'unconverged"\n'
               f'elif _mb_ending relaxation-capped; then\n'
               f'    echo "WARNING: geometry still unconverged after '
               f'$_siesta_retry_max retry(s); {_retry["after"]}." >&2\n'
               f'elif [ "$_siesta_retry" -gt 0 ]; then\n'
               f'    echo "SIESTA converged after $_siesta_retry '
               f'retry(s)."\n'
               f'fi\n'
               f'\n'
               if continue_retries and continue_retries > 0 else "")
            + f'echo "{_prog_label} completed: $_launch_cmd {script_name} -> '
            + f'$_out_file"\n'
            + _finish_block(finish, script_name, basename)
            + f'printf "rc=0 at %s\\n" "$(date)" '
            f'> "{basename}-run${{_run_n}}.concluded"\n'
        )
    else:
        launch_block = (
            f'_log INFO "resolved launch : {inner}"\n'
            f"# --- Dry-run preview (--dry-run) ----------------------\n"
            f'if [ "$_dry_run" = 1 ]; then\n'
            f'    echo ""\n'
            f'    echo "===== molbuilder DRY RUN (no PySCF launch) ====="\n'
            f'    echo "  Resolved cmd : {inner}"\n'
            f'    echo "  Conda env    : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'    echo "==============================================="\n'
            f'    _log INFO "dry-run complete; no PySCF launched"\n'
            f'    exit 0\n'
            f'fi\n'
            + _monitor_block(watch_label, watch_stage, notify_on_scf,
                             notify_every_hours, notify_channels,
                             notify_report,
                             # OpenMP only: `-np` is accepted and ignored.
                             cores="$_omp_threads",
                             # the run's GPU request, read above -- its
                             # script's GPU probe runs on the same value
                             gpu=gpus.uses,
                             unwatchable=unwatchable)
            # NOT `exec`: the shell has to outlive the engine to conclude.
            + f"set +e\n"
            f"{inner}\n"
            f"_pyscf_exit=$?\n"
            f"set -e\n"
            f'    # CONCLUDED -- an error is a conclusion (project-layout.md\n'
            f'    # 1.6, "the other file"): the engine returned and this\n'
            f'    # process gets to say goodbye.  MAIN LINE ONLY, never the\n'
            f'    # cleanup trap: a walltime SIGTERM runs the trap, and a\n'
            f'    # forced stop must leave NO marker.\n'
            f'printf "rc=%s at %s\\n" "$_pyscf_exit" "$(date)" '
            f'> "{basename}-run${{_run_n}}.concluded"\n'
            f'exit "$_pyscf_exit"\n'
        )

    # Engine-specific output role, ASKED rather than branched on.  The
    # banner below shows the suffix the user will actually see so they don't
    # go hunting for the wrong filename after the first run (BOMB-6 fix) --
    # which is precisely why it must not be a second opinion.
    _ext = _stdout_role_for(suffix)
    # ----- Script-contract PROVENANCE block -----
    # See docs/execution/job-contracts.md.  PROVENANCE only for the
    # wrapper -- no BENCH-MARKS (wrapper-side parameters are overridden
    # via existing env vars per the contract) and no ATOM-METADATA
    # (lives in the engine input file, not the wrapper).
    #
    # For PySCF (.py) wrappers, mpi_np is meaningless (PySCF is OMP-
    # only) and must not surface as a per-call value -- the
    # test_render_pyscf_ignores_mpi_np invariant (same wrapper text
    # regardless of mpi_np input) is the contract here.
    from . import script_emit as _sc
    _is_pyscf = suffix == ".py"
    _resolved_defaults = {
        "target_env":    target_env,
        # THE STATED COUNTS -- both engines refuse an unstated one above,
        # so there is no "auto" to record (2026-10-02).
        "omp_threads": str(omp_threads),
        "max_memory_mb": (
            "n/a" if max_memory_mb is None else str(max_memory_mb)
        ),
    }
    if _is_pyscf:
        _resolved_defaults["mpi_np"] = "n/a (PySCF is OMP-only)"
    else:
        _resolved_defaults["mpi_np"] = str(mpi_np)
    _provenance = _sc.emit_provenance(
        generator_version=_sc.molbuilder_git_sha(),
        generated_at=_sc.generated_at_now(),
        resolved_defaults=_resolved_defaults,
        # The wrapper is the one artifact EVERY prepared run has, whatever
        # the engine and whatever the task -- a TranSIESTA run has no deck
        # PROVENANCE (§ 3.1's table) but always has this.  So the engine
        # declaration rides here too, and `_is_pyscf` is the same suffix
        # test that already decided this file's whole shape four lines up.
        engine="pyscf" if _is_pyscf else "siesta",
    )
    _user_custom = _sc.emit_user_custom_placeholder()

    return (
        f"#!/usr/bin/env bash\n"
        f"{_provenance}\n"
        f"#\n"
        f"# molbuilder run-wrapper -- {description}\n"
        f"# Script: {script_name}\n"
        f"# Target env: {target_env}\n"
        f"#\n"
        f"# Generated at prep (`molbuilder jobset prep`).  Edit freely;\n"
        f"# molbuilder will not regenerate this file until the next prep\n"
        f"# of the same script.  Run directly:\n"
        f"#\n"
        f"#     bash {basename}.run.sh              # first run -> -run0{_ext}\n"
        f"#     bash {basename}.run.sh --continue   # resume -> -run1, -run2, ...\n"
        f"#     bash {basename}.run.sh --force      # restart from -run0 (overwrite)\n"
        f"#     bash {basename}.run.sh -np 8        # override mpi_np (SIESTA only)\n"
        f"#     MB_NP=8 bash {basename}.run.sh      # same via env var (SLURM/PBS)\n"
        f"#     nohup ./{basename}.run.sh &         # background, detached\n"
        f"#\n"
        f"# Continuation contract:\n"
        # THE DECK DECIDES, AND THIS WRAPPER SHIPS BESIDE EXACTLY ONE DECK.
        # This block asserted that the generator emits the restart keywords
        # "by default" and that SIESTA "auto-loads .DM/.CG/.XV" -- the third
        # copy of one claim in this file, and false for every stage described
        # `clean`, whose deck now says `.false.` three times.  All three copies
        # read `_restart_honoured` now; there is no fourth.
        # KEYED ON THE SUFFIX FIRST (E-J3): which ENGINE's contract to
        # print is the script's own fact, not the restart probe's -- a
        # SIESTA deck whose restart answer could not be read used to
        # fall through this conditional's last arm and receive the
        # PySCF paragraph.
        + (
            f"#  * PySCF: this deck reads prior state -- its chkfile\n"
            f"#    init-guess and previous-rung geometry reads are gated\n"
            f"#    on its described ``continue`` (run-identity.md § 4\n"
            f"#    rule 2).\n"
            if suffix == ".py" and _py_reads_prior else
            f"#  * PySCF: this deck emits NO prior-state read (described\n"
            f"#    ``clean``): prior files on disk are simply not\n"
            f"#    consulted.\n"
            if suffix == ".py" and _py_reads_prior is False else
            f"#  * PySCF (vibration): this deck reads no prior engine\n"
            f"#    state at start -- each run recomputes from its own\n"
            f"#    relaxation (or the already_relaxed assertion).\n"
            if suffix == ".py" else
            f"#  * SIESTA: this deck sets ``DM.UseSaveDM`` /\n"
            f"#    ``MD.UseSaveXV`` / ``MD.UseSaveCG`` .true., so SIESTA\n"
            f"#    loads the .DM/.XV/.CG left under this SystemLabel.\n"
            if _restart_honoured else
            f"#  * SIESTA: this deck sets ``DM.UseSaveDM`` .false., so\n"
            f"#    SIESTA does NOT load prior .DM/.XV/.CG -- the run starts\n"
            f"#    from the coordinates in the deck.  To continue instead,\n"
            f"#    go back to the state saved before this stage's prep,\n"
            f"#    change `restart` in the description, and prep it anew.\n"
            if _restart_honoured is False else
            f"#  * SIESTA: this deck's restart keywords could not be read;\n"
            f"#    whether prior .DM/.XV/.CG are loaded is the deck's to\n"
            f"#    say.  Prep the stage anew, from the state saved before\n"
            f"#    its prep, to restore the restart group the generator\n"
            f"#    writes.\n"
        )
        + f"#\n"
        f"# IMPORTANT: this wrapper does NOT change cwd (see\n"
        f"# docs/execution/running-a-job.md § 5).  Its own files (the\n"
        f"# runwrap log, -runN.out) land in the CALLER'S current working\n"
        f"# directory.  Under sbatch this is ``SLURM_SUBMIT_DIR`` (the\n"
        f"# dir you ran sbatch from); under direct invocation it's\n"
        f"# wherever you ``cd``'d before running the wrapper.  The\n"
        f"# engine's files (.chk, trajectory XYZ, .molwatch.log,\n"
        f"# .spectra.json, .DM, ...) land in the same directory:\n"
        f"# SIESTA writes where it runs, and a PySCF deck writes beside\n"
        f"# itself, which this wrapper runs by its bare name from here\n"
        f"# (docs/execution/job-contracts.md § 4.2).\n"
        f"#\n"
        f"# Convention: invoke from the project directory (the dir\n"
        f"# holding the .fdf / .py and where you want outputs to\n"
        f"# accumulate).  Do NOT add ``cd`` lines or modify the cwd\n"
        f"# from inside the generated .py / .fdf -- code you might\n"
        f"# layer on top (PySCF's mol.log open(), geomeTRIC's optimize\n"
        f"# prefix) would write into the wrong place if you chdir to\n"
        f"# elsewhere.\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"\n"
        f"# Help is answerable by ANYONE: -h/--help is scanned before the\n"
        f"# launch-door gate, so asking what a script does never meets a\n"
        f"# permission prompt (U10, 2026-08-12).  The flag is only NOTED\n"
        f"# here -- the engine arg loop below owns the usage text.\n"
        f"_mb_help=0\n"
        f'for _mb_a in ${{@:+"$@"}}; do\n'
        f'    case "$_mb_a" in -h|--help) _mb_help=1 ;; esac\n'
        f"done\n"
        f"\n"
        f"# --- Per-run log file (current directory; see docs/execution/running-a-job.md § 5) -\n"
        f'_runwrap_log="$PWD/{basename}.runwrap-$(date +%Y%m%d-%H%M%S).log"\n'
        f"# ABSOLUTE, and it has to be: every later reference -- the\n"
        f"# ending read over it (_mb_ending), for SIESTA's stderr -- must\n"
        f"# resolve wherever it runs from.  (A comment here claimed the\n"
        f"# wrapper 'cd's into run-<n>/ after this point' -- the attempt-dir\n"
        f"# cd was retired 2026-08-10, and this wrapper twice states it\n"
        f"# never changes cwd; R9 removed the ghost.)\n"
        f"# It opens BEFORE the launch-door gate (U10), so a refusal is a\n"
        f"# fact on disk -- the log alone answers what happened, even when\n"
        f"# nothing ran.\n"
        f'exec > >(tee -a "$_runwrap_log") 2> >(tee -a "$_runwrap_log" >&2)\n'
        f"\n"
        f"# --- Launch-door gate (one door: jobset launch) ----------\n"
        f"# `molbuilder jobset launch` sets MB_LAUNCHED_BY when it launches\n"
        f"# this script (direct: child env; sbatch: --export on the command\n"
        f"# line).  A bare invocation is usually an accident: it skips the\n"
        f"# launch bookkeeping, the deck/launch agreement check and the\n"
        f"# mode/config resolution.  MB_LAUNCHED_BY=manual is the\n"
        f"# deliberate override, and the value is logged either way\n"
        f"# (job-contracts.md 2.6).  A help request skips the gate: it\n"
        f"# launches nothing.\n"
        f"if [ -z \"${{MB_LAUNCHED_BY:-}}\" ] && [ \"$_mb_help\" = \"0\" ]; then\n"
        f"  echo \"WARNING: {basename}.run.sh was called directly, not via\" >&2\n"
        f"  echo \"  'molbuilder jobset launch'.  Direct calls skip launch\" >&2\n"
        f"  echo \"  bookkeeping and the deck/launch agreement check.\" >&2\n"
        f"  if [ -t 0 ]; then\n"
        f"    printf \"  Proceed anyway? [y/N] \" >&2\n"
        f"    # `|| true`: EOF (Ctrl-D, a closed pty) must fall through to\n"
        f"    # the refusal below -- under set -e a failed `read` would kill\n"
        f"    # the script mid-prompt with NO verdict line (U10).\n"
        f"    _mb_answer=\"\"\n"
        f"    read -r _mb_answer || true\n"
        f"    case \"${{_mb_answer}}\" in\n"
        f"      # EXPORTED, not just set: the warm-retry re-execs this very\n"
        f"      # wrapper, and an unexported answer would re-prompt (or\n"
        f"      # refuse, stdin long gone) mid-retry at 3am (U10).  The\n"
        f"      # submit door already exports; the manual door must too.\n"
        f"      y|Y|yes|YES) export MB_LAUNCHED_BY=manual ;;\n"
        f"      *) echo \"  Aborted.  (MB_LAUNCHED_BY=manual skips this prompt.)\" >&2\n"
        f"         echo \"launched-by: NONE -- refused at the launch-door gate\"\n"
        f"         exit 2 ;;\n"
        f"    esac\n"
        f"  else\n"
        f"    echo \"  Non-interactive shell: refusing.  Launch via\" >&2\n"
        f"    echo \"  'molbuilder jobset launch', or override deliberately:\" >&2\n"
        f"    echo \"    MB_LAUNCHED_BY=manual bash {basename}.run.sh   # local\" >&2\n"
        f"    echo \"    sbatch --export=ALL,MB_LAUNCHED_BY=manual ...  # hand-sbatch\" >&2\n"
        f"    echo \"  (the sbatch form matters: a site or config with\" >&2\n"
        f"    echo \"  'export: NONE' strips a plain env var.)\" >&2\n"
        f"    echo \"launched-by: NONE -- refused at the launch-door gate\"\n"
        f"    exit 2\n"
        f"  fi\n"
        f"fi\n"
        f"# The verdict, in the JOB'S OWN OUTPUT: under sbatch this line is in\n"
        f"# the job's .out, and since the tee above it is ALSO in the runwrap\n"
        f"# log -- either alone answers \"was this launched properly, run by\n"
        f"# hand on purpose, or refused?\" -- no terminal needed, and nothing\n"
        f"# can sit waiting on one (no-TTY never prompts).\n"
        f'echo "launched-by: ${{MB_LAUNCHED_BY:-none (help)}}"\n'
        f"# Per docs/execution/running-a-job.md § 5: the wrapper does NOT change cwd.\n"
        f"# SLURM lands the job in SLURM_SUBMIT_DIR by default; direct\n"
        f"# callers ``cd`` to the project dir before invoking.  The\n"
        f"# caller's cwd is the contract -- outputs (log, -runN.out)\n"
        f"# land where the wrapper was invoked.\n"
        f"\n"
        f"{env_activation}"
        f"{env_prefix}"
        # THE RECORD, before the engine starts.  PySCF writes its own from
        # inside the script (it can read its objects back); SIESTA cannot, so
        # its wrapper echoes the deck it is about to hand over.
        f"{'' if _is_pyscf else _effective_parameters_block(script_path)}"
        f"{launch_block}"
        f"\n{_user_custom}\n"
    )


#: Every file that travels BESIDE a job for its monitor, as
#: ``{name beside the job: the module whose source it is}``: the monitor, the
#: one module it reads its channels through, and every framework module it
#: reads the RUN through -- the names (`runfiles`, `identity`), the output's
#: one parser (each engine family's reading pass, its grammar, the rule
#: engine they run on, and the physical constants a grammar converts a
#: residual with, `constants`), the timing instrument's rows, how a run ended (the
#: PySCF end lines, `_run_ending`, and the run's own records whose door
#: `run_status` builds its state on, `runrecord`), `run_status` itself, and
#: what a report may carry (`report_fields`) -- `execution/run-reports.md`
#: § 2.3.
#:
#: **A verbatim copy of each module's own file**, not a copy of its logic:
#: each imports the next two ways -- from the package, or from beside the job
#: -- as `config_dir` always has, so the monitor reads a run with the readers
#: the Results tab's directory door reads it with.  The rule for joining is
#: `configuration.md`'s: *stdlib-only AND travels*, and
#: `tests/test_monitor_bundle_runs_alone.py` runs the whole set with molbuilder
#: absent.
#:
#: They travel as ONE file, :data:`MONITOR_BUNDLE`, which `render_wrappers`
#: writes next to the wrapper and `materialize` brings into every attempt.
#: ONE table, because there were two lists.
MONITOR_COMPANIONS: Dict[str, str] = {
    "mb_monitor.py":       "molbuilder.monitor",
    "config_dir.py":       "molbuilder.config_dir",
    "constants.py":        "molbuilder.constants",
    "runfiles.py":         "molbuilder.runfiles",
    "identity.py":         "molbuilder.identity",
    "siesta_reader.py":    "molbuilder.parse.engines.siesta_reader",
    "siesta_grammar.py":   "molbuilder.parse.engines.siesta_grammar",
    "molwatch_reader.py":  "molbuilder.parse.engines.molwatch_reader",
    "molwatch_grammar.py": "molbuilder.parse.engines.molwatch_grammar",
    "_section_rules.py":   "molbuilder.parse.engines._section_rules",
    "scf_timing_rows.py":  "molbuilder.parse.instruments.scf_timing_rows",
    "end_lines.py":        "molbuilder.pyscf.end_lines",
    "_run_ending.py":      "molbuilder.parse.engines._run_ending",
    "report_fields.py":    "molbuilder.report_fields",
    "job.py":              "molbuilder.parse.dirs.job",
    "runrecord.py":        "molbuilder.runrecord",
    "wrapper_log.py":      "molbuilder.wrapper_log",
}


#: THE ONE FILE that travels beside every job for its monitor: a Python zip
#: application (PEP 441) holding every module of :data:`MONITOR_COMPANIONS`,
#: each its own file verbatim under its shipped name, and a ``__main__`` that
#: runs the monitor.  ``python mb_monitor.pyz ...`` is the monitor; ``python
#: mb_monitor.pyz ending OUTPUT ...`` is `_run_ending`'s door, which
#: `monitor.main` hands on.  Fourteen files stood beside the deck until
#: 2026-09-26 -- read as PySCF scripts, listed as runs of their own -- and one
#: file holds them all (user: *"can't we just put it in one single Python
#: file?"*): still the package's own files, so still no second copy of any
#: reader.
from .runfiles import MONITOR_BUNDLE  # noqa: E402,F401 -- the catalogue's name

#: What the wrapper's log says when the ending cannot be asked: no python
#: beside the job, no bundle, or a bundle that does not load on it -- whose
#: error the one ask printed just above (`_mb_ending_able`).
_ENDING_UNREADABLE = (f"the ending cannot be read here (it needs a python "
                      f"beside the job that loads {MONITOR_BUNDLE}): no "
                      f"failure hint, no warm retry")

#: The bundle's entry: the monitor, whose exit status is the process's --
#: `_run_ending`'s questions answer by it.  A bundle this python cannot
#: import answers `ending` with 2, *cannot read* (`job-contracts.md` § 2.6),
#: never 1, which would say *no*.
#:
#: IT SAYS ITS LOAD, in the session log's own line (`run-reports.md` § 2.6):
#: ``monitor: starting ...`` before the import and ``... started`` after it,
#: so a ``starting`` with no ``started`` is a monitor that died loading -- and
#: the error follows it, one line and the traceback.  Until 2026-09-27 the
#: error was caught and the process exited without a word, which undid the
#: session log keeping the monitor's stderr.  The `ending` and `loads`
#: verbs print no pair: the wrapper waits on their answers and reads their
#: exit status -- `loads` answers whether the bundle loads on this python and
#: nothing else, so the wrapper can ask it once (`_mb_ending_able`).
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
    "_asked = sys.argv[1:2] in (['ending'], ['loads'])\n"
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
    "if sys.argv[1:2] == ['loads']:\n"
    "    raise SystemExit(0)\n"
    "if not _asked:\n"
    f"    _say('INFO', 'monitor: {MONITOR_BUNDLE} started')\n"
    "raise SystemExit(mb_monitor.main())\n")


def monitor_bundle() -> bytes:
    """The bytes of :data:`MONITOR_BUNDLE` -- each module's own file, read
    (:func:`companion_source`), zipped beside the entry.  The same sources
    give the same bytes (fixed member times), so a re-prep writes the file it
    found."""
    return _zip_bundle(_BUNDLE_MAIN, MONITOR_COMPANIONS)


def _zip_bundle(main: str, companions: Dict[str, str]) -> bytes:
    """ONE builder for every bundle that travels beside a job: ``main`` as
    the zip application's ``__main__.py``, then each of ``companions`` as its
    module's own file, read (:func:`companion_source`)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        members = [("__main__.py", main)] + [
            (name, companion_source(name, companions)) for name in companions]
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

    Read rather than restated: three modules once spelled the config-dir rule
    independently and two said so in prose, *"a comment is not a mechanism"*.

    **Returned as text rather than copied to a destination.**  Step 4 is on
    floor 3, and floor 3 does not touch the disk (`script-preparation.md` § 5,
    W7).
    """
    import importlib
    table = MONITOR_COMPANIONS if companions is None else companions
    try:
        module = importlib.import_module(table[name])
        return Path(module.__file__).read_text(encoding="utf-8")
    except (KeyError, ImportError, OSError) as exc:
        raise WrapperError(
            f"could not read {name!r} to ship beside the job: {exc}") from None


#: Every file that travels beside a SIESTA force-constant job for its FINISH
#: (`engines/vibration.md` § 5.5), as ``{name beside the job: the module
#: whose source it is}``: the SIESTA route (`spectra.siesta_vibration`, whose
#: ``main`` the bundle runs), the engine-neutral analysis and its math, the
#: result's class, its activity classes and its one writer, the Methods
#: prose, and every reader the finish reads the attempt through -- the fdf
#: reader and the unit words it reads with, the deck's block reader
#: (`deck_record`), the ``.FC`` reader and its
#: error, SIESTA's reading pass with its grammar and rule engine, the
#: permutation record, the one door for engine atom numbering, the session
#: log's line with the run-file grammar it is built on (`runfiles`) -- and
#: the constants all of them convert with.
#:
#: **Each module's own file, imported two ways** (package, or beside the
#: job), like the monitor's; unlike the monitor's, the set needs **numpy and
#: ASE**, which the SIESTA job envs carry for it (`envs/recipes.py`): the
#: rule for joining is *imports only the standard library, numpy, ASE and
#: the other members*.  The SIESTA end-to-end run proves the set whole: its
#: job's python cannot import molbuilder (`tests/test_siesta_vibration_e2e.py`).
VIBRATION_COMPANIONS: Dict[str, str] = {
    "siesta_vibration.py":     "molbuilder.spectra.siesta_vibration",
    "vibrational_analysis.py": "molbuilder.spectra.vibrational_analysis",
    "normal_modes.py":         "molbuilder.spectra.normal_modes",
    "results.py":              "molbuilder.spectra.results",
    "activity.py":             "molbuilder.spectra.activity",
    "methods.py":              "molbuilder.spectra.methods",
    "spectra_sidecar.py":      "molbuilder.sidecars.spectra",
    "fdf.py":                  "molbuilder.parse.fdf",
    "units.py":                "molbuilder.units",
    "deck_record.py":          "molbuilder.deck_record",
    "siesta_fc.py":            "molbuilder.parse.engines.siesta_fc",
    "errors.py":               "molbuilder.parse.errors",
    "siesta_reader.py":        "molbuilder.parse.engines.siesta_reader",
    "siesta_grammar.py":       "molbuilder.parse.engines.siesta_grammar",
    "_section_rules.py":       "molbuilder.parse.engines._section_rules",
    "molwatch_grammar.py":     "molbuilder.parse.engines.molwatch_grammar",
    "end_lines.py":            "molbuilder.pyscf.end_lines",
    "atom_permutation.py":     "molbuilder.atom_permutation",
    "engine_atom_index.py":    "molbuilder.engine_atom_index",
    "wrapper_log.py":          "molbuilder.wrapper_log",
    "runfiles.py":             "molbuilder.runfiles",
    "constants.py":            "molbuilder.constants",
}

#: THE ONE FILE that finishes a SIESTA force-constant job: a Python zip
#: application of :data:`VIBRATION_COMPANIONS` whose ``__main__`` runs
#: `spectra.siesta_vibration.main` -- ``python mb_vibration.pyz <deck>
#: <output>`` writes ``<label>.spectra.json`` beside the run.  A deck names it
#: (`DeckSpec.finish`), `prep` copies that onto the job (`Job.finish`), the
#: wrapper runs it after SIESTA exits cleanly, and `materialize` brings it
#: into every attempt.
from .runfiles import VIBRATION_BUNDLE  # noqa: E402,F401 -- the catalogue's name

#: The finish bundle's entry.  A set that cannot load on the job's python --
#: most often an env without numpy or ASE -- says so in the session log's own
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
    f"numpy and ASE in the job env: envs/recipes.py)' "
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
#: codec, with the structure module (its one XYZ reader among it) and the
#: sidecar module its ``write_moved`` reaches; the relaxation; and a
#: vibration's rules -- the PySCF route's, the harmonic path and
#: thermochemistry, the result's hash and writer -- and its mode selector.
#: The script imports them by these names -- `pyscf.input.emit_bundle_imports`
#: reads them here.
#:
#: **Each module's own file, imported two ways**, like the monitor's and the
#: finish's; the rule for joining is *imports the standard library and numpy
#: at load, and in the functions the script calls also ASE -- which the
#: PySCF env carries for it (`envs/recipes.py`) -- and PySCF itself, with
#: gpu4pyscf and cupy for a run on the GPU; nothing else but the other
#: members*.
PYSCF_COMPANIONS: Dict[str, str] = {
    "runtime_info.py":          "molbuilder.runtime_info",
    "molwatch_emitter.py":      "molbuilder.trajectory_log.emitter",
    "constants.py":             "molbuilder.constants",
    "workingcopy_structure.py": "molbuilder.workingcopy_structure",
    "structure.py":             "molbuilder.structure",
    "molstruct.py":             "molbuilder.sidecars.molstruct",
    "relax_policy.py":          "molbuilder.pyscf.relax_policy",
    "end_lines.py":             "molbuilder.pyscf.end_lines",
    "pyscf_vibration.py":       "molbuilder.spectra.pyscf_vibration",
    "normal_modes.py":          "molbuilder.spectra.normal_modes",
    "spectra_sidecar.py":       "molbuilder.sidecars.spectra",
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
    "beside it imports molbuilder code from it (engines/pyscf.md 3)\\n')\n"
    "raise SystemExit(2)\n")


def pyscf_bundle() -> bytes:
    """The bytes of :data:`PYSCF_BUNDLE`, from the one builder."""
    return _zip_bundle(_PYSCF_MAIN, PYSCF_COMPANIONS)


def bundles_for(script, finish: Optional[str] = None
                ) -> Tuple[Tuple[str, Callable[[], bytes]], ...]:
    """The bundles that travel beside a deck, each ``(its name, its
    builder)`` -- THE ONE LIST: `render_wrappers` writes them beside the
    deck and `materialize` brings them into every attempt.

    The monitor's, beside every job (:data:`MONITOR_BUNDLE`); the job's
    finish when it has one (`Job.finish`, `engines/vibration.md` § 5.5); and
    beside a PySCF script, the molbuilder code it imports
    (:data:`PYSCF_BUNDLE`, `engines/pyscf.md` § 3)."""
    out = [(MONITOR_BUNDLE, monitor_bundle)]
    if finish is not None:
        out.append((finish, _FINISH_BUNDLES[finish]))
    if EXTENSION_TO_CATEGORY.get(Path(script).suffix.lower()) == "pyscf":
        out.append((PYSCF_BUNDLE, pyscf_bundle))
    return tuple(out)


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

    @property
    def wrapper_name(self) -> str:
        return self.files[0][0]


def render_wrappers(script_path: Path, *,
                    label: str = "",
                    n_atoms: Optional[int] = None,
                    resources: "Resources",
                    env: Optional[str] = None,
                    emit_sbatch: bool = True,
                    project_dir: Optional[Path] = None,
                    machine_record=None,
                    domain_pq: Optional[Tuple[str, str]] = None,
                    finish: Optional[str] = None,
                    resumes: bool = True,
                    warm: Optional[Sequence[str]] = None,
                    deck_text: Optional[str] = None) -> RenderedWrapper:
    """Render everything step 4 produces for *script_path*, and write nothing.

    ``deck_text`` is the deck as it will be written, for one `prep` has
    planned and not yet written (`jobset.planned`) -- the wrapper is
    rendered from it, and the file need not exist yet.  ``domain_pq`` is the
    queue a run was admitted on at prep -- its job's placement, which the
    header renders rather than binding the name again (`job-system.md`
    § 6.0).

    **W7 — floor 3 returns text.**  The deck writers hand back a string and the
    conductor writes it; step 4 held the opposite pattern until 2026-08-18,
    when rendering and writing were one function.  That made *"what would a run
    of this deck look like?"* unanswerable without producing files, and left
    the two halves of one floor with two shapes for the next engine to choose
    between.

    **The allocation arrives whole** — `execution/architecture.md` § 3.1, rule
    A8.  This took eleven loose keyword arguments until 2026-08-17, and its two
    callers passed ten and five of them: `jobset/prep.py` wrote a ``.sbatch``
    asking for ``-c 8`` beside a ``.run.sh`` whose OMP default was ``1``, while
    `web/blueprints/build.py` wrote a correct ``.run.sh`` beside a ``.sbatch``
    with no ``-c`` at all.  Neither pair was right, and neither call was wrong
    on its own terms — each had simply chosen a different subset.  With one
    object there is no subset to choose, and it is unpacked ONCE here so the
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

    For a ``.fdf`` the deck is parsed for ``NumberOfAtoms`` and the count is
    threaded into the wrapper, where it states the occupancy notice
    (`_orbitals_per_rank_notice`).  A parse failure reads as *unknown* and the
    notice is omitted rather than refusing -- ``NumberOfAtoms`` is optional in
    SIESTA.

    Both shell texts go through ``bash -n`` (parse-only, no execution) before
    they are returned, so a caller never receives malformed shell to write.
    That gate exists because the 2026-06-20 pentanedithiol incident shipped an
    unterminated quote and the user found out by running it.
    """
    script_path = Path(script_path).resolve()
    if deck_text is None:
        if not script_path.is_file():
            raise WrapperError(f"script not found: {script_path}")
        deck_text = script_path.read_text()
    r = resources
    # TOLD FIRST, READ SECOND.  The count is `len(struct.elements)`, which
    # `prep` holds when it renders a deck; reading it back out of the file we
    # just wrote is re-deriving a value we had.  The read stays for the caller
    # that has no structure to ask -- `prep_jobset` walks a job set of
    # scripts, not structures.
    if n_atoms is None and script_path.suffix.lower() == ".fdf":
        n_atoms = _fdf_n_atoms(deck_text)
    text = render_run_wrapper(
        script_path, label=label, resources=r, env=env, n_atoms=n_atoms,
        project_dir=project_dir, machine_record=machine_record,
        finish=finish, resumes=resumes, warm=warm, deck_text=deck_text)
    _validate_rendered_wrapper(text, script_path)
    # ``stem + ".run.sh"`` rather than ``with_suffix(".run.sh")``: the latter
    # replaces only the LAST suffix, so ``job.spectra.py`` would become
    # ``job.run.sh`` and lose the "spectra" tag.
    files = [(script_path.stem + ".run.sh", text)]

    # THE MONITOR TRAVELS WITH EVERY JOB, and the framework it reads the run
    # through with it: it runs under the job's own python
    # (`running-a-job.md` § 4.1, `run-reports.md` § 2.3).  It travelled only
    # beside a `.fdf` until 2026-09-26, so no PySCF run was ever watched.
    #
    # ONE FILE PER BUNDLE, AND ONE LIST OF THEM (`bundles_for`), which
    # `materialize` asks too when it brings the bundles into the attempt.
    # This writer and `materialize._bring`'s extras were two hand-kept lists
    # of "what travels with the monitor" until 2026-08-28; `config_dir.py`
    # was added here and never there, and every production run's monitor
    # died at import, stderr to /dev/null.  `render_run_wrapper` has refused
    # a finish it cannot ship and a suffix it cannot run.
    blobs = tuple((name, build()) for name, build
                  in bundles_for(script_path, finish))

    # The submission layer (`job-system.md` § 6): a ``.sbatch`` only when the
    # machine has a queue -- every value in it the job's own (its resources,
    # its GPU request among them).
    if emit_sbatch:
        sbatch = _render_sbatch_for(script_path, resources=r,
                                    project_dir=project_dir,
                                    domain_pq=domain_pq,
                                    machine_record=machine_record)
        if sbatch is not None:
            _validate_rendered_wrapper(sbatch, script_path)
            files.append((script_path.stem + ".sbatch", sbatch))

    return RenderedWrapper(files=tuple(files), executable=(files[0][0],),
                           blobs=blobs)


def _warm_in_effect(engine: str, warm: Optional[Sequence[str]]
                    ) -> Tuple[str, ...]:
    """The restart files this script knows: ``warm``, the list in effect for
    its calculation that ``prep`` hands in (`warmfiles.warm_list`), or --
    for a script rendered with no calculation behind it -- the engine's own
    file, every section: the same door asked with no folder."""
    if warm is not None:
        return tuple(warm)
    from .warmfiles import warm_list
    return warm_list(engine).suffixes


def write_run_wrapper(script_path: Path, *,
                      label: str = "",
                      n_atoms: Optional[int] = None,
                      resources: "Resources",
                      env: Optional[str] = None,
                      emit_sbatch: bool = True,
                      project_dir: Optional[Path] = None,
                      machine_record=None,
                      domain_pq: Optional[Tuple[str, str]] = None,
                      finish: Optional[str] = None,
                      resumes: bool = True,
                      warm: Optional[Sequence[str]] = None,
                      plan=None) -> Path:
    """Write what :func:`render_wrappers` produced, and return the wrapper's path.

    ``plan`` (`jobset.planned.Plan`) receives the files instead of the disk,
    rendered from the deck the plan holds: `prep` decides everything before
    it writes (`job-system.md` § 5.0).

    **This function renders nothing.**  It is the writing half of step 4, kept
    beside the rendering half so every writer of a wrapper set -- `jobset/prep`
    is the one today -- writes it the same way rather than deciding for itself
    what a wrapper needs beside it.

    **Through the one writer** (`script-preparation.md` § 3.2, W4), which keeps
    the reader's USER-CUSTOM block.  The wrapper EMITS that block — it invites
    a person to put their own lines in it — and this was a plain
    ``write_text``, so every re-prep deleted what they wrote.  An invitation
    the next run silently revokes is worse than no invitation.

    Modes: the wrapper is 0o755 so a person can ``./my-job.run.sh`` it; the
    ``.sbatch`` is 0o644 because you ``sbatch`` it rather than run it.
    Overwrites whatever is there.
    """
    from . import script_emit as _sc_write
    rendered = render_wrappers(script_path, label=label, n_atoms=n_atoms,
                               resources=resources,
                               machine_record=machine_record,
                               domain_pq=domain_pq,
                               env=env, emit_sbatch=emit_sbatch,
                               project_dir=project_dir, finish=finish,
                               resumes=resumes, warm=warm,
                               deck_text=(plan.read_text(script_path)
                                          if plan is not None else None))
    parent = Path(script_path).resolve().parent
    for name, text in rendered.files:
        _sc_write.write_script(
            parent / name, text, plan=plan,
            mode=0o755 if name in rendered.executable else 0o644)
    for name, data in rendered.blobs:
        if plan is not None:
            plan.bytes(parent / name, data, mode=0o644)
        else:
            (parent / name).write_bytes(data)
            (parent / name).chmod(0o644)
    return parent / rendered.wrapper_name


def _bound_queue(resources, domain_pq, env_rec, *, prefer_gpu=False):
    """The queue this job NAMES, bound on the target's record -- a
    :class:`~molbuilder.scheduler.place.Placement`, or ``None`` when the job
    names none.

    ``Placement`` is R1's "ONE decision" -- the header and the `sbatch`
    command line are two renderings of it -- so this returns one rather than
    a bare ``(partition, qos)``.

    Who names it, in order:

    1. ``domain_pq`` -- the caller already placed it (`launch` does, on the
       record it admitted the request against);
    2. the domain the job states (``Resources.domain`` -- `allocation.domain`,
       the run card's ``domain``, ``--domain``).

    **Nothing else** *(user, 2026-10-02: "explicit job config is the only way
    allowed")*.  An unnamed queue is not chosen here -- the menu's first row
    stood in until then, and a named one the record does not list fell back
    to it silently.  Prep refuses both before anything is written
    (`placement.launch_refusal`).

    **The binding, not the fit**: the request here is empty.  Whether the
    run fits the queue it names -- its cores, GPUs, memory and wall -- prep
    has asked already, at its checkpoint 4, of the same record by the same
    `place` (`placement.admitted`), and launch asks again of the
    machine as it stands then (R9).  *(This said `place` refused a GPU job
    naming a queue with no GPUs here until 2026-10-05: with no request it
    compares nothing, and nothing did before launch.)*
    """
    from .scheduler.place import Placement, Unplaceable, place
    from .scheduler import Request
    from .runtime_config import routing_of

    rows = routing_of(env_rec)
    if domain_pq and domain_pq[0] and domain_pq[1]:
        # Bind the ROW behind the pair, not just the two strings: callers
        # downstream want the domain itself, and re-finding it by matching
        # `(partition, qos)` against the menu is the lookup this function
        # exists to do once.
        _row = next((d for d in rows
                     if (d.partition, d.qos) == (domain_pq[0], domain_pq[1])),
                    None)
        return Placement(domain=_row, partition=domain_pq[0],
                         qos=domain_pq[1])
    want = getattr(resources, "domain", None)
    if not want:
        return None
    try:
        return place(rows, Request(), prefer_gpu=prefer_gpu, named=want)
    except Unplaceable as exc:
        raise WrapperError(
            f"this job names the queue {want!r}, which the target's record "
            f"cannot take it on:\n    "
            + "\n    ".join(r.message for r in exc.reasons)) from None


def _render_sbatch_for(script_path: Path, *,
                       project_dir: Optional[Path] = None,
                       resources: "Resources",
                       domain_pq: Optional[Tuple[str, str]] = None,
                       machine_record=None,
                       ) -> Optional[str]:
    """The ``.sbatch`` text for this job when its target has a scheduler;
    ``None`` when it does not.

    Returns text rather than writing, so the whole of step 4 can be rendered
    before anything is on disk (`script-preparation.md` § 5, W7).

    **Every value in it is one the job stated** (`architecture.md` § 5.2):
    the queue it names, its wall, its memory, its cores per rank, its GPU
    count and -- for SIESTA -- its rank count.  PySCF runs ONE process with
    OpenMP threads, so its header asks for one task: that is what the engine
    is, not a value anyone picks.  Whether each is stated is asked ONCE per
    moment, by `placement.launch_refusal` -- at prep, before anything is
    written, and at launch for a header written there (`submit.
    _sbatch_request`).  Nothing is filled in: not the target's width, not a
    rank per GPU, not a queue's ceiling, not a default from any config.

    """
    r = resources
    if project_dir is None:
        project_dir = (script_path.parent
                       if script_path.parent.exists() else None)

    # THE MACHINE IS DECIDED ONCE, AND IT WAS DECIDED BEFORE THIS: the
    # caller hands over the record, and that IS the answer -- resolving it a
    # second time here is how the two artifacts of one render could come to
    # describe two different machines.  A caller with none (the launch
    # doors) reads the calculation's own snapshot.
    if machine_record is not None:
        env_rec = machine_record
    else:
        from .scheduler import machine_for
        env_rec = machine_for(project_dir)
    if env_rec is None or env_rec.scheduler != "slurm":
        return None  # no queue on this machine -> only .run.sh is meaningful

    suffix = script_path.suffix.lower()
    # IS THIS A GPU JOB, AND HOW MANY -- the job's request, the one door for
    # every engine and every reader (`jobset.model.gpu_request`).  The
    # header counted a GPU job by its count OR by `use_gpu` until
    # 2026-10-03, so `--gpus 2` on a CPU run asked the queue for two GPUs.
    from .jobset.model import GpuRequestError, gpu_request
    try:
        gpus = gpu_request(r)
    except GpuRequestError as exc:
        raise WrapperError(f"`{script_path.name}`: {exc}") from None

    placement = _bound_queue(r, domain_pq, env_rec, prefer_gpu=gpus.uses)
    ntasks = 1 if suffix == ".py" else r.mpi_np

    return render_sbatch(
        script_path,
        partition=placement.partition, qos=placement.qos,
        ntasks=int(ntasks), cpus_per_task=int(r.cpus_per_task),
        time=r.time, mem=r.mem,
        gpus=gpus, gpu_binding=r.gpu_binding,
        exclusive=r.exclusive,
    )


# --------------------------------------------------------------------- #
#  SLURM .sbatch submission layer (docs/execution/job-system.md)  #
# --------------------------------------------------------------------- #


# `_mem_to_mb` and its `_MEM_RE` were DELETED 2026-08-24: defined once,
# called nowhere.  Reading SLURM's memory text is `scheduler.quantities`'s
# job (`parse_mem_gb`), where its human-dialect sibling can be seen beside it.


def render_sbatch(script_path: Path, *,
                  partition: str, qos: str,
                  ntasks: int, cpus_per_task: int,
                  time: str, mem: str,
                  gpus: "Optional[GpuRequest]" = None,
                  gpu_binding: Optional[bool] = None,
                  exclusive: Optional[bool] = None) -> str:
    """Render the ``<basename>.sbatch`` submission script.

    The thin two-layer model (job-system.md § 6, § 5): an ``#SBATCH`` header
    that allocates resources, then a one-line body that delegates to the
    UNCHANGED launcher ``bash <basename>.run.sh "$@"``.  The launcher still
    owns env activation + the ``mpirun`` launch; the ``.sbatch`` never
    re-implements ``module load`` / ``source activate`` (§ 2 principle 3).

    **Every value is the job's own, and every one is required**
    (`architecture.md` § 5.2): the queue it named, bound on the target's
    record; its ranks, cores per rank, wall and memory; its GPU count.  There
    is no site configuration to fall back on -- `molbuilder.json`'s
    `scheduler` block, which supplied a queue, `-c`/`-t`/`--mem` defaults,
    mail and export lines, is refused since 2026-10-02.

    Args:
      ntasks: ``-n``.  The MPI rank count for CPU **and** GPU jobs --
        ``--gres`` carries the GPU count separately, and K ranks may share one
        GPU via MPS, so ntasks = mpi_np, NOT the GPU count.  Under sbatch the
        launcher reads ``SLURM_NTASKS``, so this ``-n`` and ``mpirun -np``
        agree by construction (running-a-job.md § 3.1).
      gpus: the job's GPU request (`jobset.model.gpu_request`) -- for a run
        on the GPU, ``--gres=gpu:<count>`` and ``--gres-flags=enforce-binding``
        unless ``gpu_binding`` is False (`execution/gpu.md` G9).  ``None`` is
        a job that asks for none.  The request is consistent by construction:
        a GPU run with no count, or a count for a CPU run, never reaches
        here (G5).
      exclusive: ``--exclusive`` for a GPU job whose resources say so;
        always off for CPU.
    """
    for name, value in (("ntasks", ntasks), ("cpus_per_task", cpus_per_task)):
        if not isinstance(value, int) or value < 1:
            raise WrapperError(
                f"render_sbatch: {name} must be a positive int; got "
                f"{value!r}.")
    for name, value in (("partition", partition), ("qos", qos),
                        ("time", time), ("mem", mem)):
        if not (isinstance(value, str) and value.strip()):
            raise WrapperError(
                f"render_sbatch: {name} is required and was not stated "
                f"(docs/execution/architecture.md § 5.2).")

    basename = Path(script_path).stem
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for sbatch emission: {basename!r}."
        )

    gres = gpus.gres if gpus is not None else None
    if gres:
        exclusive = bool(exclusive)
    else:
        exclusive = False  # CPU jobs never request a whole node here

    lines: List[str] = [
        "#!/bin/bash",
        "# === molbuilder sbatch header (scheduler: slurm) ===",
        "# Generated at prep from the job's own stated values.",
        "# Authoritative design: docs/execution/job-system.md.",
        "# Submit with:  cd <projdir>; sbatch "
        f"{basename}.sbatch   (NOT bash -- sbatch reads the #SBATCH header; bash would ignore it)",
        "#",
        f"#SBATCH -J {basename}",
        "#SBATCH -N 1",
    ]
    # THE ONE EMITTER (2026-08-23, scheduler.md R1).  The queue, the wall, the
    # ranks, the cores, the devices and the memory rule are rendered by
    # `scheduler.emit`, which also renders them for the `sbatch` command line
    # -- so the header and the flags are two spellings of one placement rather
    # than two writers deciding separately.
    from .scheduler.emit import Directives
    _d = Directives(partition=partition, qos=qos, walltime=time,
                    ntasks=ntasks, cpus_per_task=cpus_per_task,
                    gres=gres,
                    gpu_binding=gpu_binding is not False,
                    mem=mem, exclusive=bool(exclusive))
    if exclusive:
        # The ignored value, said out loud so it is never a silent surprise
        # (the mem<->exclusive rule -- running-a-job.md § 5.3.1).  The RULE
        # itself lives in the emitter; this is the explanation beside it.
        lines.append(
            f"# --exclusive owns the whole node -> ALL its memory.  The stated "
            f"mem ({mem}) is IGNORED; --mem=0 = all node RAM.")
    lines.extend(_d.header_lines())
    lines.append("#SBATCH -o slurm.%j.out")
    lines.append("#SBATCH -e slurm.%j.err")

    body = (
        "\n"
        "# SLURM lands us in SLURM_SUBMIT_DIR = the project dir; the\n"
        "# launcher never cd's (running-a-job.md § 5).  The launcher's\n"
        "# preamble + activation are baked into it, so it needs nothing\n"
        "# from the submitting shell.  \"$@\" forwards what `jobset launch`\n"
        "# hands the run -- its own -np / -omp -- and --cold / --continue.\n"
        f"bash {basename}.run.sh \"$@\"\n"
    )
    return "\n".join(lines) + "\n" + body


def _validate_rendered_wrapper(text: str, script_path: Path) -> None:
    """Run ``bash -n`` (parse-only) on the rendered wrapper text.
    Raises :exc:`WrapperError` if bash rejects it as malformed shell.

    The text goes to ``bash -n`` on its standard input -- nothing is
    written anywhere: `prep` checks a wrapper it has planned and not yet
    written (`job-system.md` § 5.0, rule 3), in a folder that may not exist
    yet.  *(It went through a temp file in the script's own folder until
    2026-10-05.)*  No execution happens; bash only checks shell-syntax
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
            f"bash -n rejected the rendered wrapper.  This is a "
            f"molbuilder bug -- the wrapper template emitted invalid "
            f"syntax.  bash stderr:\n{cp.stderr}"
        )


__all__ = [
    "WrapperError",
    "render_run_wrapper",
    "write_run_wrapper",
    "render_sbatch",
]
