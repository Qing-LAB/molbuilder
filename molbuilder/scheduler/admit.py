"""Admission — can this domain take this request, and if not, why not.

The contract is ``docs/execution/scheduler.md`` § 3 (the rules) and § 5 (the
graph this is the innermost branch of).

**The check the record was always missing.**  :class:`~molbuilder.scheduler.record.Domain`
has carried ``max_time``, ``max_cores``, ``max_mem_gb`` and ``gpu`` since it
was written, and nothing compared a REQUEST against them in one place.  Each
was instead handled wherever somebody noticed it: ``gpu`` got a selector,
``max_cores`` a single call site in `prep`, ``max_time`` nothing until a
grouped submission was routed into ASU Sol's 15-minute ``debug`` queue on
2026-08-23, and ``max_mem_gb`` was declared, serialised, round-tripped -- and
read by no code at all.

Four facts, four treatments, three moments, one never implemented.  That is
one missing function, and it lives here.

Split out of ``record.py`` at phase 2 (2026-08-23) so the CHECK cannot drift
away from the record it checks -- which is exactly what happened while they
shared a general-purpose module.

Stdlib-only, like the rest of the package: a record is read on the target
inside a backend env with no molbuilder installed.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional


def domain_serves_gpu(row: Mapping[str, Any]) -> bool:
    """Whether a routing row (a :class:`Domain` as `to_row` speaks it) can
    place a GPU job -- the ONE predicate for both consumers (`prep`'s
    per-family cap and `launch`'s side routing, `generator.md` § 4.3a).

    True when the row records a GPU inventory (the probe writes each
    partition's gres types onto its row) or declares a ``gpu_partition``
    (the hand-curated column `_resolve_domain` already honours).
    """
    return bool(row.get("gpu")) or bool(row.get("gpu_partition"))


def domain_admits(row, *, cores: Optional[int] = None,
                  walltime_s: Optional[int] = None,
                  mem_gb: Optional[float] = None,
                  gpus: Optional[int] = None) -> List[str]:
    """Why this domain would refuse this request — empty list means it fits.

    **The check the record was always missing.**  :class:`Domain` has carried
    ``max_time``, ``max_cores``, ``max_mem_gb`` and ``gpu`` since it was
    written, but nothing ever compared a REQUEST against them in one place.
    Each constraint was instead handled wherever somebody noticed it: ``gpu``
    got a selector, ``max_cores`` got a single call site in `prep`,
    ``max_time`` got nothing until a grouped submission was routed into ASU
    Sol's 15-minute ``debug`` queue on 2026-08-23, and ``max_mem_gb`` was
    declared, serialised, round-tripped — and read by no code at all.

    Four facts, four different treatments, three different moments, one never
    implemented.  That is not four bugs; it is one missing function, and this
    is it.  Callers ask what they know and leave the rest ``None``: `prep`
    knows cores and devices but not duration, `launch` knows all of them, and
    a caller asking about capability alone passes nothing.

    Returns REASONS rather than a bool because every caller has to explain
    itself to a user — a refusal that cannot say what was too big sends them
    to read ``scontrol`` for numbers already on disk.

    An unstated limit never bars: a row that does not say its ceiling is not
    claiming a small one.
    """
    from .probe import parse_walltime
    why: List[str] = []

    if walltime_s is not None and row.get("max_time"):
        try:
            ceiling = parse_walltime(str(row["max_time"]))
        except ValueError:
            ceiling = None               # unreadable is not small
        if ceiling is not None and ceiling < walltime_s:
            why.append(f"needs {walltime_s // 60} min but "
                       f"{row.get('name')} allows {row['max_time']}")

    if cores is not None and row.get("max_cores"):
        try:
            cap = int(row["max_cores"])
        except (TypeError, ValueError):
            cap = None
        if cap is not None and cap < cores:
            why.append(f"needs {cores} cores but {row.get('name')} "
                       f"allows {cap}")

    if mem_gb is not None and row.get("max_mem_gb"):
        try:
            cap_gb = float(row["max_mem_gb"])
        except (TypeError, ValueError):
            cap_gb = None
        if cap_gb is not None and cap_gb < mem_gb:
            why.append(f"needs {mem_gb:g} GB but {row.get('name')} "
                       f"allows {cap_gb:g} GB")

    if gpus:
        if not domain_serves_gpu(row):
            why.append(f"{row.get('name')} has no GPUs")
        else:
            have = row.get("gpu") or {}
            if isinstance(have, dict) and have:
                if max(int(v) for v in have.values()) < gpus:
                    why.append(f"needs {gpus} GPUs but {row.get('name')} "
                               f"offers at most "
                               f"{max(int(v) for v in have.values())}")
    return why


def domain_ceiling_s(row) -> Optional[int]:
    """This domain's stated wall in seconds, or ``None`` when it states none.

    The one place ``max_time`` is parsed for a caller that needs the NUMBER
    rather than a verdict — the header emitter, which must state a time the
    queue it names will accept.
    """
    from .probe import parse_walltime
    if not row or not row.get("max_time"):
        return None
    try:
        return parse_walltime(str(row["max_time"]))
    except ValueError:
        return None
