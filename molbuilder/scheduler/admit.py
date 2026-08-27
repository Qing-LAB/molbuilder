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

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple


def domain_serves_gpu(row: Mapping[str, Any]) -> bool:
    """Whether a routing row (a :class:`Domain` as `to_row` speaks it) can
    place a GPU job -- the ONE predicate for both consumers (`prep`'s
    per-family cap and `launch`'s side routing, `generator.md` § 4.3a).

    True when the row records a GPU inventory (the probe writes each
    partition's gres types onto its row) or declares a ``gpu_partition``
    (the hand-curated column `_resolve_domain` already honours).
    """
    return bool(row.gpu) or bool(row.gpu_partition)


def _compare(row, *, cores: Optional[int] = None,
                  walltime_s: Optional[int] = None,
                  mem_gb: Optional[float] = None,
                  gpus: Optional[int] = None) -> List[str]:
    """The comparison itself -- private; `admits` is the door.

    Kept as keywords rather than folded into :func:`admits` so each limit's
    branch reads beside the field it tests.  Callers do not reach it: one
    question gets one public door, or the two drift.

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
    from .quantities import parse_walltime
    why: List[str] = []

    if walltime_s is not None and row.max_time:
        try:
            ceiling = parse_walltime(str(row.max_time))
        except ValueError:
            ceiling = None               # unreadable is not small
        if ceiling is not None and ceiling < walltime_s:
            why.append(f"needs {walltime_s // 60} min but "
                       f"{row.name} allows {row.max_time}")

    if cores is not None:
        # REFUSE ONLY WHAT NO MACHINE HERE CAN HOLD.  A partition is a
        # queue, not a machine type: Sol's `htc` is 48-, 64- and 128-core
        # nodes under one name, and SLURM will not place a job on a node
        # too small -- it waits for one that fits.  So the ceiling is the
        # WIDEST node, and refusing on a floor would deny work the wide
        # nodes would run happily (caught 2026-08-27, when a floor refused
        # a declared 64-rank CPU trial on a partition whose CPU nodes have
        # 128 cores).
        #
        # R10 -- name what WOULD fit -- so the reason says which machine
        # is the biggest, not just that the ask is too large.
        cap, widest = _widest_node(row)
        if cap is not None and cap < cores:
            where = f" ({widest})" if widest else ""
            why.append(f"needs {cores} cores but {row.name}'s largest "
                       f"machine has {cap}{where}")

    if mem_gb is not None and row.max_mem_gb:
        try:
            cap_gb = float(row.max_mem_gb)
        except (TypeError, ValueError):
            cap_gb = None
        if cap_gb is not None and cap_gb < mem_gb:
            why.append(f"needs {mem_gb:g} GB but {row.name} "
                       f"allows {cap_gb:g} GB")

    if gpus:
        # R3 APPLIES TO DEVICES TOO.  A domain that states no inventory is not
        # claiming it has none -- plenty of records describe a queue without
        # enumerating its gres, and a hand-declared row often states only the
        # wall.  Refusing on silence made an explicitly named domain
        # unusable the moment its record was terse (caught 2026-08-23, when
        # R9 started admitting the named path).
        #
        # PREFERRING nodes that do have devices is a CHOICE, and choices live
        # in `place.candidates`; this only refuses what the record positively
        # rules out.
        most = _devices_offered(row)
        if most is not None and most < gpus:
            why.append(f"needs {gpus} GPUs but {row.name} offers at "
                       f"most {most}")
    return why


def _widest_node(row) -> Tuple[Optional[int], str]:
    """``(cores of the largest machine, how it is described)``.

    From ``node_types`` when the record lists them -- the measurement --
    and from ``max_cores`` otherwise, which is what every record written
    before 2026-08-27 carries.  ``None`` means the record does not say, and
    R3 then applies: an unstated limit never bars.
    """
    rows = getattr(row, "node_types", None) or []
    best, how = None, ""
    for r in rows:
        try:
            c = int(r.get("cores"))
        except (TypeError, ValueError):
            continue
        if best is None or c > best:
            n = r.get("nodes")
            best = c
            how = f"{n} node(s) of {c}" if n else f"{c} cores"
    if best is not None:
        return best, how
    try:
        return int(row.max_cores), ""
    except (TypeError, ValueError):
        return None, ""


def _devices_offered(row) -> Optional[int]:
    """The most devices one node of this domain offers, or ``None``.

    ``None`` means *the row does not say* -- an unreadable or absent column is
    not a domain with no devices (R3), and admission must refuse only what the
    record positively rules out.

    The two spellings the ``gpu`` column arrives in are `Domain.devices`'
    business, not this function's; it used to parse them here, which is how the
    descriptor form's key names came to be read as device names elsewhere
    (`scheduler/record._read_devices`).  Where several types are offered the
    largest count wins: the ask is *can this domain hold N devices*, and the
    richest node is the one that answers it.
    """
    counts = [d.per_node for d in row.devices if d.per_node is not None]
    return max(counts) if counts else None


def domain_ceiling_s(row) -> Optional[int]:
    """This domain's stated wall in seconds, or ``None`` when it states none.

    The one place ``max_time`` is parsed for a caller that needs the NUMBER
    rather than a verdict — the header emitter, which must state a time the
    queue it names will accept.
    """
    from .quantities import parse_walltime
    if not row or not row.max_time:
        return None
    try:
        return parse_walltime(str(row.max_time))
    except ValueError:
        return None


#: SLURM memory suffixes, in gigabytes.
# `parse_mem_gb` moved to `quantities.py` (2026-08-24): it is a reader of
# a dialect, not a rule about admission, and its human-dialect sibling
# `parse_memory` disagrees with it by 1024x on a bare number.  One object,
# one module (`docs/design.md`, "Architecture").
from .quantities import parse_mem_gb            # noqa: F401


@dataclass(frozen=True)
class Request:
    """What one job asks of a queue, in the units a domain states.

    Fields the caller does not know are ``None``, and ``None`` is never a
    refusal (R7): `prep` knows cores and devices but not duration, `launch`
    knows all of them, and a caller asking about capability alone passes
    nothing.  That is what lets one admission serve every caller instead of
    each growing its own variant.

    ``mem_gb`` is a NUMBER because the record's ceiling is one; build it with
    :func:`parse_mem_gb` from whatever SLURM text the caller holds.
    """
    ranks:      Optional[int] = None
    cpus_per_task: Optional[int] = None
    gpus:       Optional[int] = None
    mem_gb:     Optional[float] = None
    walltime_s: Optional[int] = None

    @property
    def cores(self) -> Optional[int]:
        """Cores this ask occupies on a node -- ranks x cpus-per-task.

        The number a domain's ``max_cores`` is stated against.  `prep`'s
        per-family cap already computed it as ``g * k * c``; stating it once
        here is what stops the two disagreeing about what "cores" means.
        """
        if self.ranks is None:
            return None
        return self.ranks * max(self.cpus_per_task or 1, 1)


def admits(domain, request: "Request") -> List[str]:
    """Why this domain would refuse this request -- empty list means it fits.

    The typed door, and the one the decision graph's innermost branch walks
    (`execution/scheduler.md` § 5).  Every limit the domain DECLARES is
    compared (R2); ``extra`` is not, by design.
    """
    return _compare(domain,
                    cores=request.cores,
                    walltime_s=request.walltime_s,
                    mem_gb=request.mem_gb,
                    gpus=request.gpus)
