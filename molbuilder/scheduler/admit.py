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

from .record import UNSET


def domain_serves_gpu(row: Mapping[str, Any]) -> bool:
    """Whether a routing row (a :class:`Domain` as `to_row` speaks it) can
    place a GPU job -- the ONE predicate for both consumers (`prep`'s
    per-family cap and `launch`'s side routing, `generator.md` § 4.3a).

    True when the row records a GPU inventory (the probe writes each
    partition's gres types onto its row) or declares a ``gpu_partition``
    (the hand-curated column `_resolve_domain` already honours).
    """
    return bool(row.gpu) or bool(row.gpu_partition)


def _types_offered(row) -> Tuple[str, ...]:
    """The gres type NAMES this domain's record positively claims.

    Empty when the record says nothing -- which R3 reads as *permission*,
    never as "this queue has no such card".
    """
    return tuple(d.type for d in row.devices if d.type)


def _compare(row, *, cores: Optional[int] = None,
                  walltime_s: Optional[int] = None,
                  mem_gb: Optional[float] = None,
                  gpus: Optional[int] = None,
                  gpu_type: Optional[str] = None) -> List[str]:
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
        #
        # AND THE CEILING IS AMONG THE MACHINES THAT OFFER WHAT WAS ASKED
        # (R3's second half, 2026-08-27).  "SLURM will not place a job on
        # a node too small" is exactly as true of devices: a job asking
        # for a GPU can only land on a node that has one.  On Sol's
        # `public` the two ceilings differ sharply -- 128 cores across
        # 107 standard nodes, 48 across the 52 with A100s -- so a 64-rank
        # GPU trial admitted against the 128 is unplaceable, and a
        # benchmark's GPU side hits it first because the rank axis is
        # sized against the CPU side.
        #
        # AND THE MACHINES THAT OFFER *THIS* DEVICE, when one is named.
        # "a node that has one" is a statement about a TYPE, not about
        # devices in general: on Sol, `general` holds a 128-core node
        # with an h200 and four 64-core nodes with a100.40gb, so a
        # 128-rank a100.40gb trial admitted against "the widest machine
        # with a device" is unplaceable -- it would have to land on the
        # h200 node, which carries no a100.40gb at all.
        cap, widest = _widest_node(row, needs_device=bool(gpus),
                                   device_type=gpu_type)
        if cap is not None and cap < cores:
            where = f" ({widest})" if widest else ""
            with_dev = (f" with {gpu_type}" if (gpus and gpu_type)
                        else " with a device" if gpus else "")
            why.append(f"needs {cores} cores{with_dev} but {row.name}'s "
                       f"largest machine{with_dev} has {cap}{where}")
        # THE POLICY CEILINGS, beside the hardware one (R13).  What the
        # widest machine HAS and what policy LETS one job take are two
        # facts; both are read and the smaller governs.  `lightwork` is
        # why: `max_cores: 128` (the nodes) beside a suspected 8-core cap
        # (the policy) with nothing able to compare either.
        for cap_field, phrase in (("max_cpus_per_job", "per job"),
                                  ("max_cpus_per_node", "per node")):
            pol = getattr(row, cap_field, None)
            if pol is UNSET:
                pol = None       # never asked -- no cap to enforce (R3)
            try:
                pol = int(pol) if pol is not None else None
            except (TypeError, ValueError):
                pol = None                 # unreadable is not small (R3)
            if pol is not None and pol < cores:
                why.append(f"needs {cores} cores but {row.name}'s policy "
                           f"allows {pol} {phrase}")

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
        # THE TYPE IS A LIMIT THE RECORD DECLARES, so R2 compares it.
        # `--gres=gpu:<type>:N` names a type SLURM matches literally: a
        # queue with no node registering that name answers *Requested
        # node configuration is not available*, which is the refusal this
        # framework exists to make before the scheduler does (R6).
        #
        # The types are NOT interchangeable and a suffix is not decoration
        # -- Sol registers `a100` on 48-core nodes and `a100.40gb` on
        # 64-core ones, disjoint groups, and `--gpus`' own help calls the
        # MIG slices "separate askable types, not a smaller ask of the
        # same one".  So this matches the token, never a prefix of it.
        offered = _types_offered(row)
        if gpu_type and offered and gpu_type not in offered:
            why.append(f"needs {gpu_type} but {row.name} offers "
                       f"{', '.join(sorted(offered))}")
        # THE COUNT, of the type that was asked for.  Reading the largest
        # count over ALL types answers a question nobody asked: on Sol's
        # `public` that is 16 (a100.20gb MIG slices), which admitted every
        # 4-device ask no matter which card it named.
        most = _devices_offered(row, device_type=gpu_type)
        if most is not None and most < gpus:
            named = f" {gpu_type}" if (gpu_type and gpu_type in offered) else ""
            why.append(f"needs {gpus}{named} GPUs but {row.name} offers at "
                       f"most {most}{named}")
    return why


def _widest_node(row, *, needs_device: bool = False,
                 device_type: Optional[str] = None
                 ) -> Tuple[Optional[int], str]:
    """``(cores of the largest machine, how it is described)``.

    From ``node_types`` when the record lists them -- the measurement --
    and from ``max_cores`` otherwise, which is what every record written
    before 2026-08-27 carries.  ``None`` means the record does not say, and
    R3 then applies: an unstated limit never bars.

    ``needs_device`` narrows the search to machines that carry one (R3's
    second half): a ``--gres`` job can only land on such a node, so the
    widest device-less machine is not a ceiling it could ever enjoy.  When
    the list names NO device-bearing machine the filter yields nothing and
    the unfiltered answer stands -- the record not saying which nodes hold
    the devices is silence, and silence never bars.

    ``device_type`` narrows it further, to the machines carrying THAT
    card.  A queue's device-bearing machines are not one pool: Sol's
    `general` holds four 64-core nodes with a100.40gb beside a 128-core
    node with an h200, and only the first four can take an a100.40gb job.
    Silence still permits -- a node group that lists no ``gpu`` map, or a
    record with no ``node_types`` at all, yields nothing to filter on and
    the wider answer stands.
    """
    rows = getattr(row, "node_types", None) or []
    if needs_device:
        with_dev = [r for r in rows
                    if isinstance(r, dict) and r.get("gpu")]
        # ...AND ONLY WHERE THE RECORD SAID WHICH NODES HOLD DEVICES.
        # ``with_dev`` empty is SILENCE, and narrowing silence by type
        # yields silence -- so the type filter is skipped and the wider
        # answer (the unfiltered widest node, else ``max_cores``) stands.
        # Without the ``and with_dev`` this returned "no ceiling" for every
        # record with no ``node_types`` at all -- i.e. every record written
        # before 2026-08-27, and every hand-declared row -- so NAMING a card
        # removed the core ceiling instead of tightening it, and a
        # 4096-rank trial was admitted on a 48-core queue.
        if device_type and with_dev:
            # THE ONE READER of a gpu column, per `record._read_devices`:
            # the column has two spellings and reading it here by key
            # would make the descriptor form's key names ("type",
            # "per_node") read as device names -- the exact bug that
            # reader was written to end.
            from .record import _read_devices
            of_type = [r for r in with_dev
                       if any(d.type == device_type
                              for d in _read_devices(r.get("gpu")))]
            # A LIST THAT NAMES DEVICES AND NOT THIS ONE IS AN ANSWER, and
            # the answer is *no machine here carries that card* -- so there
            # is no core ceiling to state, and stating the widest
            # OTHER-device machine would say "public's largest machine with
            # a100.40gb has 48" about a queue holding no a100.40gb at all.
            # The type comparison in `_compare` is what refuses; this axis
            # stays quiet rather than refusing the same fact in a false
            # sentence.  (An EMPTY ``with_dev`` is the different case: the
            # record never said which nodes hold devices, and silence never
            # bars -- the unfiltered answer stands, below.)
            if not of_type:
                return None, ""
            with_dev = of_type
        if with_dev:
            rows = with_dev
    best, best_n, how = None, None, ""
    for r in rows:
        try:
            c = int(r.get("cores"))
        except (TypeError, ValueError):
            continue
        # AMONG EQUALLY WIDE GROUPS, THE ONE WITH MORE MACHINES.  R10 asks
        # the reason to name what would fit, and Sol lists `htc`'s A100
        # nodes as a 1-node group and a 51-node group of the same width --
        # reporting "1 node(s) of 48" reads as a queue with one machine.
        n = _to_int_or_none(r.get("nodes"))
        if best is None or c > best or (c == best and (n or 0) > (best_n or 0)):
            best, best_n = c, n
            how = f"{n} node(s) of {c}" if n else f"{c} cores"
    if best is not None:
        return best, how
    try:
        return int(row.max_cores), ""
    except (TypeError, ValueError):
        return None, ""


def _to_int_or_none(v) -> Optional[int]:
    """A record field as an int, or ``None`` when it is not one -- an
    unreadable node count must not read as zero and lose a tie."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _devices_offered(row, device_type: Optional[str] = None) -> Optional[int]:
    """The most devices one node of this domain offers, or ``None``.

    ``device_type`` asks about ONE card: the most of *that* type a node
    here offers.  ``None`` when the row names no such type -- the caller
    has already refused on the name by then, and inventing a count for a
    card this queue does not have would refuse it twice.

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
    counts = [d.per_node for d in row.devices
              if d.per_node is not None
              and (device_type is None or d.type == device_type)]
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
    #: WHICH card, as ``--gres=gpu:<type>:N`` spells it -- the gres token,
    #: never a marketing name (`quantities.parse_gres` reads it, and its
    #: note says why a name-matching reader gets this wrong).  ``None``
    #: asks for a device without naming one, and names nothing to refuse.
    #:
    #: `scheduler.md` § 7 has shown this field in the caller's view since
    #: the contract was written; admission got it on 2026-08-30, after a
    #: bench asked ``gpu:a100.40gb:4`` on Sol's `public` -- which offers
    #: a100, a100.20gb and a30, and no a100.40gb anywhere.  Every declared
    #: limit but this one was compared, so the submission was admitted
    #: here and refused by sbatch (*Requested node configuration is not
    #: available*) after the group ahead of it had already gone out.
    gpu_type:   Optional[str] = None
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
                    gpus=request.gpus,
                    gpu_type=request.gpu_type)
