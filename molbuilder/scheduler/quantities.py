"""The two quantities a job asks for — a **wall time** and an **amount of
memory** — and every way each of them is written.

**One object, one module.**  These are not three unrelated helpers that
happen to take strings; they are the complete set of dialects a duration or
a memory size is spoken in, and they belong together because the whole cost
of getting this wrong is paid when two of them are used interchangeably.

Each quantity has three dialects, and they are NOT the same:

    the RECORD dialect   "0-04:00:00"  "80G"      what SLURM accepts, and
                                                  what every molbuilder file
                                                  stores
    the HUMAN dialect    "4h" "90m" "80GB"        what a person types at a
                                                  flag or into the browser
    PROSE                "1h30m"                  what a table shows a person
                                                  reading it

**They disagree, which is why the distinction is load-bearing.**  `04:30`
is four minutes thirty in the record dialect and four and a half hours in
the human one.  A field holding *whichever dialect arrived* cannot be read
correctly by anybody -- and on 2026-08-24 one did: the browser wrote `"4h"`
into `task.json`, `prep` copied it verbatim into a field documented as
SLURM's own, and `sbatch` was handed `-t 4h` and refused the tool's own
written value.

**Where these lived before, and why that was the bug's real cause.**  The
human-dialect readers sat in `jobset/ask.py` -- a workflow module whose own
docstring lists what lives there and does not mention them.  They were put
where they were first needed rather than where they belong, so the record
reader (`parse_walltime`, here, next to the `sinfo` text it was written for)
and the human reader (`parse_duration`, over in the job-submission package)
ended up in different subpackages with nothing to make their difference
visible.  Then `jobset/submit.py` called the record reader on a
human-dialect value, and nothing in the code's shape objected.

They are one subject, so they get one home, and it is at **L1**: this is a
codec on a basic unit, in the same sense as `identity` (how a run id is
written) or `persist` (how a versioned document is written).  Nothing here
touches a filesystem, a scheduler, or a workflow, which is what lets
`task.py` and `jobset/` both reach it without either importing the other
(`docs/design.md`, "Architecture").

Stdlib-only, like the rest of this package: a machine record is read on the
target inside a backend environment with no molbuilder installed.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "parse_walltime", "parse_gres",
    "parse_duration", "parse_memory",
    "slurm_time", "slurm_mem",
    "canonical_time", "canonical_mem", "parse_mem_gb",
    "human_wall",
]


# --------------------------------------------------------------------- #
#  The RECORD dialect — what SLURM's own tooling reports                #
# --------------------------------------------------------------------- #

def parse_walltime(s) -> int:
    """SLURM walltime string -> seconds.  Accepts the forms SLURM accepts:
    ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``
    (running-a-job.md § 5.3).  Empty -> 0.  Raises ValueError on garbage
    so a malformed config max_time fails loudly, not silently as 0.

    **This reads what a MACHINE wrote** -- a queue's ``max_time`` off
    ``sinfo``.  For what a PERSON wrote, use :func:`parse_duration`; the
    two differ on a two-part value (``04:30`` is 4m30s here and 4h30m
    there), so calling this one on a typed value is a silent mistranslation
    where it does not simply raise.
    """
    s = str(s).strip()
    if not s:
        return 0
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        days = int(d)
        parts = [int(x) for x in s.split(":")] if s else [0]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = parts[0], parts[1], parts[2]
    else:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0          # bare = minutes (SLURM rule)
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]   # MM:SS
        else:
            h, m, sec = parts[0], parts[1], parts[2]
    return ((days * 24 + h) * 60 + m) * 60 + sec


def parse_gres(text) -> "dict":
    """SLURM's own gres spelling -> ``{type: count}``.

    ``gpu:a100:4`` · ``gpu:a100:4(S:0-1)`` · ``gpu:a100:4,mps:400`` ·
    ``gpu:4`` (untyped -> type ``"gpu"``) · ``(null)``/``none`` -> ``{}``.

    **The type is POSITIONAL, and that is the whole point.**  A second
    reader of this same text matched the type against a hard-coded list of
    known GPU names, which on ASU Sol reported:

        gpu:a100.40gb:4  ->  a100     a MIG slice as a whole A100
        gpu:gh200:1      ->  h200     Grace-Hopper, by substring
        gpu:h200.35gb:4  ->  h200     a slice as the whole card
        gpu:hl225:8      ->  None     Habana, simply unknown

    `--gpus`' own help says the MIG slices "are separate askable types, not
    a smaller ask of the same one" -- and that reader conflated exactly
    those.  A list of names cannot keep up with a site's hardware; the
    token is already there to be read.

    (Matching against known names IS right for ``nvidia-smi``, which prints
    marketing names like ``NVIDIA A100-SXM4-40GB`` rather than gres tokens.
    `record._gpu_type_from_name` still does that, for that input.)

    Non-gpu resources are skipped, so a trailing ``mps:400`` is never read
    as a GPU count.  A non-integer count reads as 1 -- SLURM writes bare
    ``gpu:a100`` for a single device on some versions.  Duplicate types
    keep the LARGER count: a partition merged across node groups states the
    most any of its nodes offers.
    """
    out = {}
    g = (text or "").strip()
    if not g or g.lower() in ("(null)", "none"):
        return out
    g = re.sub(r"\(.*?\)", "", g)          # drop a "(S:0-1)" affinity tail
    for tok in g.split(","):
        tok = tok.strip()
        if not tok or not tok.lower().startswith("gpu:"):
            continue
        parts = tok.split(":")
        if len(parts) >= 3:
            gtype, count = parts[1], parts[2]
        elif len(parts) == 2:
            gtype, count = "gpu", parts[1]
        else:
            continue
        try:
            n = int(count)
        except ValueError:
            n = 1
        out[gtype] = max(out.get(gtype, 0), n)
    return out


# --------------------------------------------------------------------- #
#  The HUMAN dialect — what a person types                              #
# --------------------------------------------------------------------- #

def parse_duration(text) -> Optional[int]:
    """``"4h"`` / ``"90m"`` / ``"45"`` -> seconds.  Bare numbers are minutes.

    Raises :class:`ValueError` with the forms it accepts, because a refusal
    that does not show the shape is a refusal you have to guess past.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().lower()
    # SLURM'S OWN SPELLING, ``[D-]HH:MM[:SS]``, accepted too (2026-08-24).
    # It is not a nicety: it is how a queue's ``max_time`` is written in the
    # machine record, so it is what the browser fills a time field WITH --
    # and a person reading `7-00:00:00` out of their own `task.json` could
    # not type it back at `--time`, which refused the very value the tool
    # had just written.  One vocabulary, every surface.
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", t)
    if m:
        d, hh, mm, ss = (int(g or 0) for g in m.groups())
        total = d * 86400 + hh * 3600 + mm * 60 + ss
        if total <= 0:
            raise ValueError("a duration must be positive")
        return total
    mult = 60
    for suffix, mu in (("h", 3600), ("m", 60), ("s", 1)):
        if t.endswith(suffix):
            mult, t = mu, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration -- write 4h, 90m, 45 "
                         f"(bare numbers are minutes), or SLURM's own "
                         f"D-HH:MM:SS")
    if v <= 0:
        raise ValueError("a duration must be positive")
    return int(v * mult)


def parse_memory(text) -> Optional[float]:
    """``"128G"`` / ``"0.5T"`` / ``"128"`` -> gigabytes.  Bare numbers are GB.

    Reads the HUMAN dialect -- every form a person might type, SLURM's own
    among them.  What gets WRITTEN is :func:`slurm_mem`'s, which is a strict
    subset: this accepts ``0.5T`` and ``80GB``, and neither is a thing
    ``--mem`` takes.  The docstring said "SLURM's own spelling" until
    2026-08-24, which read as *"whatever comes out of here is fit for the
    command line"* -- and that is the assumption the `-t 4h` failure was
    made of.
    """
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip().upper()
    # A TRAILING ``B`` IS ACCEPTED (2026-08-24).  `prep --mem`'s own help
    # says *"e.g. 80GB"* and passes the string through unparsed, while
    # `launch --mem` parsed it and REFUSED -- two flags of one name
    # disagreeing about a spelling one of them advertises.  Also ``K``,
    # for completeness with SLURM's units.
    if t.endswith("B") and len(t) > 1 and not t[-2].isdigit():
        t = t[:-1]
    mult = 1.0
    for suffix, mu in (("T", 1024.0), ("G", 1.0),
                       ("M", 1 / 1024.0), ("K", 1 / 1048576.0)):
        if t.endswith(suffix):
            mult, t = mu, t[:-1]
            break
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{text!r} is not an amount of memory -- write "
                         f"128G, 128GB, 0.5T, or 128 (bare numbers are GB)")
    if v <= 0:
        raise ValueError("memory must be positive")
    return v * mult


#: SLURM's memory suffixes, as multiples of a gigabyte.
_MEM_UNITS = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def parse_mem_gb(text) -> Optional[float]:
    """``"390G"`` -> ``390.0``.  ``None`` when it says nothing usable.

    **Reads the RECORD dialect, where `parse_memory` reads the human one,
    and THEY DISAGREE ON A BARE NUMBER.**  SLURM's own default unit is
    megabytes, so ``"512"`` is half a gigabyte here; `--mem`'s help says
    *"bare numbers are GB"*, so ``"512"`` is five hundred and twelve
    gigabytes there.  A factor of 1024, and nothing about either name says
    which you are holding -- the same shape as ``04:30`` meaning four
    minutes thirty to SLURM and four and a half hours to a person.  They
    live in one module so that difference is on one screen; they lived in
    two packages until 2026-08-24, which is how the wrong one came to be
    called.

    **The unit R2 was missing.**  The record states memory as a number of
    gigabytes; a job states it as SLURM text.  Nothing converted between
    them, so ``max_mem_gb`` could be compared against nothing and was read
    by no code at all -- a limit that cannot be expressed in the same unit
    as the ask is a limit that will never be checked.

    ``--mem=0`` is SLURM for *all the memory on the node*, which is the
    opposite of asking for none, so it returns ``None``: an unbounded ask is
    not a small one, and R3 says an unstated limit never bars.  (That is a
    different answer from `canonical_mem`, which WRITES ``"0"`` through
    unchanged -- what to store and what to fit a queue against are two
    questions.)

    A trailing ``B`` is accepted, as it is by `parse_memory`: ``80GB`` is
    the spelling `prep --mem`'s own help advertises, and this returned
    ``None`` for it until 2026-08-24 -- which reads as *unstated*, and an
    unstated limit never bars, so an ask nobody could parse would have been
    admitted to a queue that could not hold it.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) or None
    t = str(text).strip().upper()
    if not t or t == "0":
        return None
    if t.endswith("B") and len(t) > 1 and not t[-2].isdigit():
        t = t[:-1]
    unit = _MEM_UNITS.get(t[-1])
    number = t[:-1] if unit is not None else t
    try:
        value = float(number)
    except ValueError:
        return None                      # unreadable is not small (R3)
    return (value * (unit if unit is not None else 1 / 1024)) or None


# --------------------------------------------------------------------- #
#  Writing the RECORD dialect — what every file stores                  #
# --------------------------------------------------------------------- #

def slurm_time(seconds: int) -> str:
    """Seconds -> ``D-HH:MM:SS``, the record's one spelling for a wall."""
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}"


def slurm_mem(gb: float) -> str:
    """Gigabytes -> SLURM's ``<n>G`` (or ``<n>M`` when not a whole GB).

    Megabytes for a fractional value rather than rounding: a ceiling asked
    for at 95% of a node lands on a fraction, and rounding a memory ask UP
    is how a request that fits becomes one the queue refuses.
    """
    if float(gb) <= 0:
        return "0"
    mb = round(float(gb) * 1024)
    # A POSITIVE ASK NEVER ROUNDS TO ZERO.  SLURM reads `--mem=0` as *all
    # the memory on the node*, so rounding a sliver down would turn the
    # smallest possible request into the largest one -- silently, and in
    # the direction that gets a job refused or a node monopolised.  One
    # megabyte is the smallest thing SLURM can be asked for; below that
    # there is nothing truthful to say.
    if mb <= 0:
        return "1M"
    return f"{mb // 1024}G" if mb % 1024 == 0 else f"{mb}M"


def canonical_time(text) -> Optional[str]:
    """A stated wall, in whatever dialect, -> the record's.  ``None`` when
    nothing was stated -- unstated is not zero (`submission.md` S1)."""
    secs = parse_duration(text)
    return None if secs is None else slurm_time(secs)


def canonical_mem(text) -> Optional[str]:
    """A stated memory, in whatever dialect, -> the record's.

    ``"0"`` passes through: it is SLURM's own spelling for *all the memory
    on the node*, which `--mem`'s own help advertises, and it is a stated
    ask rather than an absent one.  :func:`parse_memory` refuses it because
    zero gigabytes is not an amount to fit a queue against -- a different
    question from what to write in the file.
    """
    if text is None or str(text).strip() == "":
        return None
    if str(text).strip() == "0":
        return "0"
    gb = parse_memory(text)
    return None if gb is None else slurm_mem(gb)


# --------------------------------------------------------------------- #
#  PROSE — what a table shows a person                                  #
# --------------------------------------------------------------------- #

def human_wall(secs: Optional[int]) -> str:
    """A queue's ceiling, for a person reading the table.

    Whole units where they are whole, and the remainder where it is not:
    a 90-minute queue reads ``1h30m``, not ``1h``.  It rendered as ``1h``
    until 2026-08-24 -- integer division, no remainder -- which UNDER-reports
    the limit in the one table a person reads to decide what to ask for, so
    they would ask for less than the queue would have given them.

    Hours, not days: ``168h`` is what this column has always read and what
    `submission.md`'s worked example shows.

    Prose, not a record: what gets WRITTEN is :func:`slurm_time`'s spelling.
    """
    if not secs:
        return "-"
    h, rem = divmod(int(secs), 3600)
    m = rem // 60
    if h and m:
        return f"{h}h{m}m"
    return f"{h}h" if h else f"{m}m"
