"""The calculation's machine -- its record read, checked, and set at its
first prep (`configuration.md` M-3).  Floor 1 (`execution/architecture.md`
§ 2.1): plain facts about a machine and the one refusal of a machine with no
record, which the conductor, the prep assembly and launch each ask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .errors import PrepError


#: What a prep says when no machine record answers.
#:
#: ONE SENTENCE OF WHY, then the command.  The rule is `running-a-job.md`
#: § 3.1's -- a machine's facts are read from a record and nowhere else, the
#: local box included -- and the reason it is worth a refusal rather than a
#: probe is that a probed-on-the-fly number is indistinguishable from a
#: recorded one once it is in a wrapper.
_NO_RECORD = (
    "no machine record for this machine, so there is nothing to prep "
    "against.\n"
    "  Record it, once:\n"
    "      {cmd}\n"
    "  A machine's cores, GPUs and queues are read from a record and never "
    "probed on the fly -- so the numbers in a wrapper can always be traced "
    "to a file you can look at (running-a-job.md 3.1)."
)


def set_machine(base_dir, target: Optional[str] = None, *,
                environment, plan) -> Path:
    """**Step 1 of the five: resolve the machine** (`project-layout.md`
    § 2.3.1) — the machine's record (its cores, GPUs, scheduler and
    environment), as the prep entry read it, snapshotted as
    ``environment.json`` beside the bundle.

    Written once per bundle and **not** overwritten on a later prep: the file
    records what this machine is, and re-reading on every stage would make two
    stages of one calculation disagree about their own target for no reason a
    user asked for.  It names the machine the calculation is set to -- this
    prep's ``--target`` -- and that does not change: another record reaches
    the calculation only through a new prep, from a state saved before this
    one (`configuration.md` M-3).  A preview reads without writing
    (:func:`machine_record`).

    **IT DOES NOT PROBE.  A machine that has no record is a REFUSAL**
    *(user, 2026-09-02: "all environments have to be explicitly probed and
    stored. no environment json, error")*, and it names the one command that
    fixes it -- the refusal of the record's one reader, :func:`machine_record`,
    which the prep entry asks before this is called.

    A probe here would be the guess `running-a-job.md` § 3.1 forbids: the
    numbers a wrapper carries would come from *whichever box happened to run
    prep*.  Probing is one command and it is the user's to run, so the record
    is always something they can point at and say where it came from.

    ``environment`` is the record the prep entry read, once, at its
    checkpoint 4, and checked the activation of there (`job-system.md`
    § 5.0); the copy is made from it.

    ``plan`` (`jobset.planned.Plan`) receives the copy, and is read for
    one it already holds: `prep` decides everything before it writes
    (`job-system.md` § 5.0).

    Returns the path to ``environment.json``.
    """
    from ..scheduler.record import calculation_record
    out = calculation_record(base_dir)
    if plan.is_file(out):
        return out
    # THE CALCULATION HAS NO COPY YET (we just early-returned if it did), so
    # the record handed over is the MACHINE's -- what `jobset probe` wrote,
    # read by the entry (`machine_record`).  Snapshotting that answer rather
    # than re-probing is what makes one probe serve every calculation here
    # (configuration.md § 5, M-3).  Nothing here detects anything.
    # THE MACHINE IT IS SET TO, named in the copy (M-3): a later prep's
    # `--target` is checked against it.  No target is this machine --
    # `machine_for` refuses the question when another is on file.
    from dataclasses import replace
    from ..scheduler.record import LOCAL_TARGET
    env = replace(environment, machine=target or LOCAL_TARGET)
    from ..persist import json_text
    plan.text(out, json_text(env.to_dict()))
    return out


def _no_record() -> PrepError:
    """The refusal of THIS machine with no record, naming the probe that
    writes one (`scheduler.record.probe_line`, W52).  Only this machine can
    have none: a named target's record is there, or `machine_for` refuses
    the name itself."""
    from ..scheduler.record import probe_line
    return PrepError(_NO_RECORD.format(cmd=probe_line(None)))


def machine_record(base: Path, target: Optional[str] = None):
    """Step 1's ANSWER, read -- the record `prep` snapshots
    (:func:`set_machine` writes it, after this is checked), and all a
    preview asks, which writes nothing (`web/task-setup.md` § 11.1).
    Refuses as step 1 does when no record answers."""
    from ..scheduler import machine_for
    env = machine_for(base, target=target)
    if env is None:
        raise _no_record()
    return env


def require_activation(target: Optional[str], environment,
                        base=None) -> None:
    """A record that does not state how a shell enters an environment there
    is refused HERE, for every target -- this machine included.

    **Prep reads the TARGET's record, and only it** (`configuration.md` § 4):
    each machine declares its ``env_init`` in its own ``molbuilder.json`` and
    `jobset probe` copies it into the record it writes.  No target is allowed
    a substitute: generating with THIS machine's activation for another
    machine succeeds at generate time and dies on the cluster hours later, on
    a path that exists only here (2026-08-24).
    """
    from ..scheduler.record import (LOCAL_TARGET, calculation_record,
                                    probe_command, probe_steps)
    from .commands import rollback
    if (getattr(environment, "env_init", None) or {}).get("activation"):
        return
    here = target in (None, LOCAL_TARGET)
    own = calculation_record(base) if base is not None else None
    if own is not None and own.is_file():
        # THE CALCULATION'S OWN COPY ANSWERED (`configuration.md` § 5 M-3):
        # taken at its first prep and never replaced, so no probe reaches it.
        raise PrepError(
            f"this calculation's record of its machine, {own}, does not say "
            f"how a shell enters an environment there -- it was copied at the "
            f"calculation's first prep and is never replaced, so a probe does "
            f"not reach it (a copy taken before 2026-10-02 names it "
            f"`script_generation`).  Rename that key to `env_init` in it, or "
            + rollback("the calculation's first prep", base=base)
            + ("" if here else f"  Name the machine again: --target {target}."))
    whose = "this machine's record" if here else f"the record of {target!r}"
    raise PrepError(
        f"{whose} does not say "
        f"how a shell enters an environment there, and nothing else may "
        f"(docs/configuration.md § 4).  Declare it in that machine's "
        f"molbuilder.json --\n"
        f"      \"env_init\": {{\"activation\": \"conda activate\", "
        f"\"preamble\": \"source <conda root>/etc/profile.d/conda.sh\"}}\n"
        f"  or \"source activate\" after \"module load mamba\" where a "
        f"module gives the toolchain -- then probe {probe_steps(target)}:\n"
        f"      {probe_command(target)}\n"
        f"  and prep again.  A copied record that is wrong for its machine "
        f"is edited by hand.")


__all__ = ["set_machine", "machine_record", "require_activation"]
