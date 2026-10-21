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


def _shown(job, said: str) -> list:
    """The job, as `status` and `stop` show it: its id, how it went and what
    it is doing, when it went, and its calculation's status line."""
    from .jobset.commands import command
    how = (f"queue job {job.job_id}" if job.how == "queue" else
           f"run here on {job.host}, "
           f"{'in the background' if job.background else 'waited on'}, "
           f"pid {job.pid}")
    seq = (f", a sequence of groups "
           f"{', '.join(str(g['number']) for g in job.sequence['groups'])}"
           if job.sequence else "")
    return [f"{job.id}  {how}{seq} -- {said}",
            f"  launched {job.launched_at}; its calculation:",
            f"  {command('status', base=job.calculation)}"]


def _not_reached(job) -> list:
    """A SEQUENCE'S GROUPS NOT LAUNCHED YET -- none of their stages among
    its members -- each said (`job-system.md` § 6.0a)."""
    from .identity import stage_key
    if not job.sequence:
        return []
    launched = {stage_key(m) for m in job.members}
    return [f"    group {g['number']}  {', '.join(g['stages'])}  "
            f"not launched yet"
            for g in job.sequence["groups"]
            if not launched & {stage_key(s) for s in g["stages"]}]


def _record(job, decision: str, **facts) -> None:
    """A stop's decision in the calculation's ledger -- in a described
    calculation only: a job's address outlives a folder moved or deleted."""
    from .jobset.ledger import record
    from .calcdirs import described
    if described(job.calculation):
        record(job.calculation, "jobs stop", decision, job=job.id, **facts)


@jobs_group.command("status", short_help="every job launched from this "
                                         "machine, and what each is doing")
def cmd_jobs_status() -> None:
    """Every job's address, and what it is doing now -- its process or the
    scheduler, and each member's run as `jobset status` reads it.  A job
    that has ended is shown once, then its file is removed."""
    from .jobset.runstatus import job_members
    from .joblist import Asked, JobFileError, every, forget, is_going
    jobs = every()
    here = Asked()
    if not jobs:
        click.echo("no job launched from this machine is on file.")
        return
    for job in jobs:
        if isinstance(job, JobFileError):
            click.echo(f"a job's file that does not read: {job}")
            continue
        going, said = is_going(job, here)
        for line in _shown(job, said):
            click.echo(line)
        for stage, run, state in job_members(job.calculation,
                                             job.members):
            click.echo(f"    {stage}  {run}  {state}")
        for line in _not_reached(job):
            click.echo(line)
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
    from .joblist import (Asked, JobFileError, NotGoing, find,
                          is_going, scheduler_of, signal_here)
    try:
        job = find(job_id)
    except JobFileError as exc:
        raise click.ClickException(f"its file does not read: {exc}") \
            from None
    if job is None:
        raise click.ClickException(
            f"no job {job_id} is on file -- `molbuilder jobs status` lists "
            f"them")
    going, saying = is_going(job, Asked())
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
         *(f"    {stage}  {m['run']}" for stage, m in job.members.items()),
         *_not_reached(job)]),
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
            how = cancel(scheduler_of(job), job.job_id)
    except NotGoing as exc:
        raise click.ClickException(
            f"{exc} -- `molbuilder jobs status` says how") from None
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"{job.id} was not stopped: {exc}") \
            from None
    _record(job, "stopped", how=how)
    click.echo(f"stopped: {how} -- `molbuilder jobs status` says how it "
               f"ended")
