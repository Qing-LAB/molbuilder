"""`molbuilder jobs` -- the jobs launched from this machine, kept the way a
running server is (`execution/job-system.md` § 6.0a): `status` reads each
job's address and asks, fresh, what it is doing; `stop` stops one.

A surface (`execution/architecture.md` § 2, floor 7): the addresses are
`joblist`'s, the queue's answers `scheduler.queue`'s -- asked of the
scheduler the job's calculation's record names -- each member's state
the status door's (`jobset.runstatus.job_members`); the stop is written in
the calculation's ledger.
"""
from __future__ import annotations

import click


@click.group("jobs", short_help="the jobs launched from this machine: "
                                "what each is doing, and stop one")
def jobs_group() -> None:
    """The jobs launched from this machine -- a run here or a queue job --
    each kept by its address, the way `molbuilder serve status` keeps a
    server's (`job-system.md` § 6.0a).

    \b
        molbuilder jobs status          every job, and what each is doing
        molbuilder jobs stop <id>       stop one -- shown and asked first

    What a job is doing is read fresh each time: its process, or the
    scheduler, and each member's run as `jobset status` reads it.  A job
    that has ended is shown once, how each member ended, and then dropped --
    its runs and the calculation's ledger keep its history.
    """


class _Here:
    """The cluster a scheduler here answers for, as it names it -- asked
    once a command for each scheduler, and only when a queue job is asked
    of."""
    def __init__(self):
        self._asked = {}

    def cluster(self, scheduler):
        if scheduler not in self._asked:
            from .scheduler.queue import cluster
            self._asked[scheduler] = cluster(scheduler)
        return self._asked[scheduler]


def _scheduler_of(job):
    """The scheduler a queue job was sent to: the one its calculation's copy
    of its machine's record names (`configuration.md` M-3) -- ``None`` when
    that copy does not read here."""
    from .scheduler import read_environment
    from .scheduler.record import calculation_record
    rec = read_environment(calculation_record(job.calculation))
    return rec.scheduler if rec is not None else None


def _going(job, here: _Here):
    """``(going, said)`` -- whether the job is still going, its process or
    the scheduler's list, ``None`` where this machine cannot ask -- and
    that in words."""
    from .joblist import going_here, this_host
    if job.how == "here":
        going = going_here(job)
        return going, {True: "its process is going",
                       False: "its process has ended",
                       None: (f"on {job.host}: ask there"
                              if job.host != this_host() else
                              "this machine cannot tell")}[going]
    from .scheduler.queue import listed
    scheduler = _scheduler_of(job)
    if scheduler is None:
        from .scheduler.record import UnknownTarget, calculation_record
        if calculation_record(job.calculation).is_file():
            # THERE AND NOT READ, said as the record's own refusal says it
            # -- an older molbuilder's copy naming `jobset migrate`.
            return None, " ".join(
                str(UnknownTarget.unreadable_copy(job.calculation)).split())
        return None, ("its calculation's record of its machine is not here, "
                      "so its scheduler cannot be asked")
    answers = here.cluster(scheduler)
    if job.cluster is not None and answers != job.cluster:
        return None, (f"sent to cluster {job.cluster}: ask there"
                      if answers is not None else
                      "the scheduler cannot be asked here")
    # THE SCHEDULER'S OWN WORDS for a job it holds -- its state and reason,
    # so a job it took and will never start says why (`job-system.md`
    # § 6.0a; `scheduler.md` R6).
    answer = listed(scheduler, job.job_id)
    if answer is None:
        return None, "the scheduler cannot be asked here"
    going, said = answer
    if going:
        return True, "the scheduler lists it" + (f": {said}" if said else "")
    # ENDED, in its words while it still lists the job -- ``FAILED
    # (NonZeroExitCode)`` -- and only then forgotten.
    return False, (f"the scheduler lists it ended: {said}" if said
                   else "the scheduler no longer lists it")


def _shown(job, said: str) -> list:
    """The job, as `status` and `stop` show it: its id, how it went and what
    it is doing, when it went, and its calculation's status line."""
    from .jobset.commands import command
    how = (f"queue job {job.job_id}" if job.how == "queue" else
           f"run here on {job.host}, "
           f"{'in the background' if job.background else 'waited on'}, "
           f"pid {job.pid}")
    return [f"{job.id}  {how} -- {said}",
            f"  launched {job.launched_at}; its calculation:",
            f"  {command('status', base=job.calculation)}"]


def _record(job, decision: str, **facts) -> None:
    """A stop's decision in the calculation's ledger -- in a described
    calculation only, as a launch's (`submit._record`)."""
    from pathlib import Path

    from .jobset.ledger import record
    from .task import FILENAME
    if (Path(job.calculation) / FILENAME).is_file():
        record(job.calculation, "jobs stop", decision, job=job.id, **facts)


@jobs_group.command("status", short_help="every job launched from this "
                                         "machine, and what each is doing")
def cmd_jobs_status() -> None:
    """Every job's address, and what it is doing now -- its process or the
    scheduler, and each member's run as `jobset status` reads it.  A job
    that has ended is shown once, then its file is removed."""
    from .jobset.runstatus import job_members
    from .joblist import JobFileError, every, forget
    jobs = every()
    here = _Here()
    if not jobs:
        click.echo("no job launched from this machine is on file.")
        return
    for job in jobs:
        if isinstance(job, JobFileError):
            click.echo(f"a job's file that does not read: {job}")
            continue
        going, said = _going(job, here)
        for line in _shown(job, said):
            click.echo(line)
        for stage, run, state in job_members(job.calculation,
                                             job.members):
            click.echo(f"    {stage}  {run}  {state}")
        if going is False:
            click.echo("  ended -- said once; its address is removed")
            forget(job)


@jobs_group.command("stop", short_help="stop one job -- shown and asked "
                                       "first")
@click.argument("job_id")
@click.option("--yes", "-y", is_flag=True,
              help="Stop it without being asked -- the answer given in "
                   "advance.")
def cmd_jobs_stop(job_id: str, yes: bool) -> None:
    """Stop the job JOB_ID names (`molbuilder jobs status` lists them): a
    run here left in the background -- SIGTERM to its process group, which
    reaches the engine -- or a queue job, its scheduler's cancel.  Shown and asked
    first; the calculation's ledger records the stop."""
    from .jobset.ask import confirm
    from .joblist import JobFileError, NotGoing, find, signal_here
    try:
        job = find(job_id)
    except JobFileError as exc:
        raise click.ClickException(f"its file does not read: {exc}") \
            from None
    if job is None:
        raise click.ClickException(
            f"no job {job_id} is on file -- `molbuilder jobs status` lists "
            f"them")
    going, saying = _going(job, _Here())
    if going is None:
        raise click.ClickException(
            f"{job.id}: {saying} -- nothing was stopped")
    if going is False:
        raise click.ClickException(
            f"{job.id} has ended -- `molbuilder jobs status` says how")
    if job.how == "here" and not job.background:
        raise click.ClickException(
            f"{job.id} is waited on at its terminal: stop it there, with "
            f"Ctrl-C")
    said = confirm("\n".join(
        ["about to stop:", *_shown(job, saying),
         *(f"    {stage}  {m['run']}" for stage, m in job.members.items())]),
        auto_yes=yes, question="stop it?", default=False)
    # THE QUESTION AND ITS ANSWER, written down as a launch writes its own
    # -- a no, and nobody to ask, too (`job-system.md` § 6.0, step 5).
    _record(job, "question", about="stop it?",
            answer=(said.words if said.asked else
                    "no answer (not a terminal): nothing stopped"))
    if not said.asked:
        raise click.ClickException(
            "not a terminal, so there is nobody to ask -- nothing was "
            "stopped.  Pass --yes to stop it without being asked.")
    if not said:
        click.echo("nothing was stopped.")
        return
    try:
        if job.how == "here":
            # CHECKED AGAIN, NOW: the answer took a person's time.
            signal_here(job)
            how = f"SIGTERM to its process group, {job.pid}"
        else:
            from .scheduler.queue import cancel
            how = cancel(_scheduler_of(job), job.job_id)
    except NotGoing as exc:
        raise click.ClickException(
            f"{exc} -- `molbuilder jobs status` says how") from None
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"{job.id} was not stopped: {exc}") \
            from None
    _record(job, "stopped", how=how)
    click.echo(f"stopped: {how} -- `molbuilder jobs status` says how it "
               f"ended")
