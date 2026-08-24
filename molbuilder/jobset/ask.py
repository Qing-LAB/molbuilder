"""What this job needs — asked once, answered once, shown once.

**One question, one answer, one interface, one output** (user, 2026-08-23).

A scientist knows what their calculation needs better than any rule this
framework can write.  So it asks, rather than deriving: *how much time, how
much memory.*  Everything else follows from the answer, and nothing has to be
explained afterwards because nothing was invented.

**What this replaces.**  The numbers used to arrive by themselves.  A job
asked for 128 GB because SLURM grants 2 GB a core and it had 64 of them; it
asked for 38 minutes because a per-trial default nobody set was multiplied by
a trial count nobody saw.  Both were arithmetic on inputs the person had never
been offered.  The first attempt at a fix was a provenance system — five
categories, an announcement rule, a display — machinery whose entire purpose
was to cope with numbers nobody chose.  **Asking removes the problem instead
of labelling it.**

Four things live here and nowhere else:

    Ask          the question, and the answer to it
    fits         whether this machine can honour that answer
    render       the one output — what is about to be requested
    confirm      the one interface — approve, change, or skip

The CLI and the browser call the same four.  Two surfaces asking one question
two ways is how they come to disagree about what was asked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: The benchmark launcher's own shape, stated once so the total shown is the
#: total requested.  Two spellings of one formula is how a displayed number
#: comes to differ from the one that reaches the scheduler.
GROUP_SLACK = 1.1
GROUP_STARTUP_S = 300


@dataclass(frozen=True)
class Ask:
    """What one job needs, as the person said it.

    ``None`` means *not answered* — never zero, never a default wearing a
    number's clothes.  A caller that must have a value asks; a caller that can
    proceed without one says so.
    """
    time_s: Optional[int] = None
    mem_gb: Optional[float] = None

    def __bool__(self) -> bool:
        return self.time_s is not None or self.mem_gb is not None


def parse_duration(text) -> Optional[int]:
    """``"4h"`` / ``"90m"`` / ``"45"`` -> seconds.  Bare numbers are minutes.

    Raises :class:`ValueError` with the forms it accepts, because a refusal
    that does not show the shape is a refusal you have to guess past.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().lower()
    mult = 60
    for suffix, m in (("h", 3600), ("m", 60), ("s", 1)):
        if t.endswith(suffix):
            mult, t = m, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration -- write 4h, 90m, "
                         f"or 45 (bare numbers are minutes)")
    if v <= 0:
        raise ValueError("a duration must be positive")
    return int(v * mult)


def parse_memory(text) -> Optional[float]:
    """``"128G"`` / ``"0.5T"`` / ``"128"`` -> gigabytes.  Bare numbers are GB.

    SLURM's own spelling, so what a person types here is what they would have
    typed into ``--mem``.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().upper()
    mult = 1.0
    for suffix, m in (("T", 1024.0), ("G", 1.0), ("M", 1 / 1024.0)):
        if t.endswith(suffix):
            mult, t = m, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not an amount of memory -- write "
                         f"128G, 0.5T, or 128 (bare numbers are GB)")
    if v <= 0:
        raise ValueError("memory must be positive")
    return v * mult


def bench_bound(total_s: int, n_trials: int) -> int:
    """A benchmark's TOTAL -> the per-trial bound that fits inside it.

    The person states the total — how long they are willing to wait — and the
    per-trial bound is arithmetic on top.  It ran the other way round: a
    per-trial default produced a total nobody saw.

    A bound below a minute measures nothing, so it floors there and the total
    is exceeded instead.  :func:`render` says so when that happens; honouring
    a budget into uselessness would be the worse answer.
    """
    n = max(1, int(n_trials))
    usable = max(0, int(total_s) - GROUP_STARTUP_S)
    return max(60, int(usable / (n * GROUP_SLACK)))


def bench_total(bound_s: int, n_trials: int) -> int:
    """The per-trial bound -> the total the allocation asks for.  The
    launcher's own formula, which :func:`bench_bound` inverts."""
    return int(max(1, int(n_trials)) * int(bound_s) * GROUP_SLACK) \
        + GROUP_STARTUP_S


def fits(ask: Ask, rows: Sequence) -> Tuple[bool, List[str]]:
    """Can this machine honour that answer?  ``(ok, reasons)``.

    Answers *before* anything is submitted, which is the whole point: a queue
    rejects an impossible ask after you have waited for it, and this rejects
    it while changing the number is free.

    ``True`` when the machine states no queues at all — nothing was promised,
    so there is nothing to contradict (the same rule placement follows).  An
    unstated ceiling never bars (R3): a row that does not say how much memory
    it has is not claiming to have none.
    """
    rows = list(rows or ())
    if not rows or not ask:
        return True, []
    from ..scheduler import domain_ceiling_s
    reasons: List[str] = []
    if ask.time_s is not None:
        best = max((domain_ceiling_s(r) or 0) for r in rows)
        if best and ask.time_s > best:
            reasons.append(
                f"{ask.time_s // 60} min exceeds every queue here; the "
                f"longest is {best // 60} min")
    if ask.mem_gb is not None:
        stated = [float(r.max_mem_gb) for r in rows if r.max_mem_gb]
        if stated and ask.mem_gb > max(stated):
            reasons.append(
                f"{ask.mem_gb:g} GB exceeds every queue here; the largest "
                f"holds {max(stated):g} GB")
    return (not reasons), reasons


def render(ask: Ask, *, placement=None, n_trials: Optional[int] = None,
           bound_s: Optional[int] = None, extra: Sequence[str] = ()) -> str:
    """**The one output** — what is about to be requested, in full.

    Every line is a number that will reach the scheduler, and a line the
    person can change while changing it is still free.  Shown by the CLI and
    by the browser from this one function, because two renderings of one
    request is how a surface comes to show something the scheduler was not
    told.
    """
    lines: List[str] = ["about to request:"]
    if ask.time_s is not None:
        lines.append(f"  time     {ask.time_s // 3600}h "
                     f"{(ask.time_s % 3600) // 60:02d}m")
    if ask.mem_gb is not None:
        lines.append(f"  memory   {ask.mem_gb:g} GB")
    if n_trials is not None and bound_s is not None:
        total = bench_total(bound_s, n_trials)
        lines.append(f"  {n_trials} trial(s), {bound_s // 60} min each "
                     f"-> {total // 60} min total")
        if ask.time_s is not None and total > ask.time_s:
            lines.append(
                f"  NOTE {n_trials} trials do not fit in "
                f"{ask.time_s // 60} min -- a bound under a minute measures "
                f"nothing, so the total is {total // 60} min.  Fewer trials, "
                f"or more time.")
    if placement is not None:
        lines.append(f"  queue    {placement.partition} / {placement.qos}")
    lines.extend(f"  {e}" for e in extra)
    return "\n".join(lines)


def confirm(text: str, *, auto_yes: bool = False, echo=None,
            prompt=None) -> bool:
    """**The one interface** — show it, then act on the answer.

    ``auto_yes`` is how a person says *I have decided to trust this*; its
    absence is not permission.  ``echo``/``prompt`` are injected so the same
    function serves a terminal, a test, and anything else that can show a
    string and read a yes — the browser included.
    """
    import sys
    import click
    echo = echo or click.echo
    echo(text)
    if auto_yes:
        echo("  (--yes)")
        return True
    if prompt is None and not sys.stdin.isatty():
        # NO TERMINAL TO ASK.  S4 says the absence of `--yes` is not
        # permission, so this declines -- but it declines by SAYING WHY and
        # naming the flag, because a scripted run that aborts with no
        # explanation is a worse failure than the one the gate prevents.
        echo("  not a terminal, so there is nobody to ask -- pass --yes to "
             "submit what is printed above without confirming.")
        return False
    prompt = prompt or (lambda: click.confirm("  submit this?",
                                              default=True))
    return bool(prompt())
