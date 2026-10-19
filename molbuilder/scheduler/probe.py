"""``molbuilder jobset probe`` -- what a live SLURM cluster offers, read on
its login node into that machine's record (`configuration.md` § 5).

Probes ``sinfo``/``sacctmgr`` and DERIVES the record's ``domains`` -- each
partition and QoS your account can reach, with its walls, caps and per-job
policy -- that a person would otherwise read off those tools by hand.  The
framework hardcodes NO partition names or limits: everything here comes from
the live system (the anti-hardcoding rule, made executable).

Pure parsing + derivation lives here (testable on captured text);
``record.probe_queues`` runs the commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional,
                    Set, Tuple)

if TYPE_CHECKING:
    from .record import Domain

from .quantities import parse_walltime


# sinfo timelimit tokens that mean "no ceiling".
_INFINITE = {"infinite", "unlimited", "n/a", ""}
# The threshold above which a partition's limit reads as none (sinfo's own
# `infinite` token, or a limit past thirty days).
_INFINITE_SECS = 30 * 24 * 3600



@dataclass(frozen=True)
class NodeGroup:
    """One kind of machine in a partition, as ``sinfo`` reports it.

    **A partition is a QUEUE, not a machine type** (measured 2026-08-27):
    ASU Sol's ``htc`` is 51 nodes of 48 cores with A100s, 3 of 64 with MIG
    slices, and 134 of 128 with no device at all.  ``general`` and
    ``public`` have the same shape, and what actually separates the three
    is the wall clock.

    So the groups are the fact, and any single number derived from them is
    an opinion.
    """
    cores:   Optional[int] = None
    mem_mb:  Optional[int] = None
    nodes:   int = 0
    gpu:     Tuple[Tuple[str, int], ...] = ()

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"cores": self.cores, "nodes": self.nodes}
        if self.mem_mb is not None:
            row["mem_gb"] = round(self.mem_mb / 1024.0, 1)
        if self.gpu:
            row["gpu"] = dict(self.gpu)
        return row


@dataclass
class Partition:
    """One SLURM partition, merged across its node groups."""
    name: str
    timelimit_str: str
    timelimit_secs: int
    nodes: int = 0
    #: Every distinct machine this partition holds -- ``sinfo`` reports one
    #: row PER NODE GROUP.  The MEASUREMENT; the scalars below summarise it.
    groups: List["NodeGroup"] = field(default_factory=list)
    gpu_types: Dict[str, int] = field(default_factory=dict)  # type -> max count
    #: MEMORY PER NODE, in MB, of the LARGEST node group -- the queue's one
    #: figure, as ``max_cpus`` is its widest (`scheduler.md` R0).
    mem_mb: Optional[int] = None
    #: DEFAULT MEMORY PER CORE, in MB -- what SLURM grants when a job states
    #: no ``--mem``.  From ``scontrol show partition``; ``sinfo`` does not
    #: report it.
    def_mem_per_cpu_mb: Optional[int] = None
    #: THE PARTITION'S OWN CPU CAP, per node per job -- ``MaxCPUsPerNode``
    #: from ``scontrol show partition``.  A POLICY ceiling, not a hardware
    #: one (`scheduler.md` R13): the nodes may be wide and the partition
    #: still refuses to give one job more than this many of their cores.
    #: ``None`` means the partition does not say (R3).
    max_cpus_per_node: Optional[int] = None
    #: Whether ``scontrol show partition`` ANSWERED for this partition --
    #: what lets the record write ``max_cpus_per_node: null`` (asked, no
    #: cap stated) instead of omitting the key (never asked).
    policy_queried: bool = False


# --------------------------------------------------------------------- #
#  parsing (pure -- operate on captured command text)                   #
# --------------------------------------------------------------------- #

def _to_secs(timelimit: str) -> int:
    """SLURM partition TIMELIMIT -> seconds; 'infinite'/'unlimited' -> a
    sentinel past thirty days, which `derive_domains` records as no wall."""
    t = (timelimit or "").strip().lower()
    if t in _INFINITE:
        return _INFINITE_SECS
    try:
        return parse_walltime(timelimit)
    except (ValueError, AttributeError):
        return _INFINITE_SECS


from .quantities import parse_gres as _parse_gres


def parse_sinfo(text: str) -> List[Partition]:
    """Parse ``sinfo -h -o '%P|%<w>l|%D|%<w>G|%c|%m'`` (pipe-delimited; fields
    may be space-padded by the width modifier -- we strip).  A partition
    appears once per node group; merge them (union GPU types, sum nodes,
    keep the time limit; each group its own machine, its MEMORY feeding
    ``mem_mb``, the largest machine's).
    The default-partition ``*`` marker is stripped.  A capture lacking the
    CPUS or MEMORY column leaves it ``None``: an absent column is not a small
    one (R3)."""
    parts: Dict[str, Partition] = {}
    for line in (text or "").splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4 or not cols[0]:
            continue
        name = cols[0].rstrip("*")
        tl_str, nodes_str, gres = cols[1], cols[2], cols[3]
        try:
            nodes = int(nodes_str)
        except ValueError:
            nodes = 0
        cpus: Optional[int] = None
        if len(cols) >= 5 and cols[4]:
            # sinfo prints "48+" when a group's nodes differ; the base is
            # the smallest, which every node of the group has.
            try:
                cpus = int(cols[4].rstrip("+"))
            except ValueError:
                cpus = None
        mem: Optional[int] = None
        if len(cols) >= 6 and cols[5]:
            # Same "48+" convention as the core column: the base is the
            # SMALLEST of a differing group, which every node of it has.
            try:
                mem = int(cols[5].rstrip("+"))
            except ValueError:
                mem = None
        gpus = _parse_gres(gres)
        p = parts.get(name)
        if p is None:
            p = Partition(name=name, timelimit_str=tl_str,
                          timelimit_secs=_to_secs(tl_str))
            parts[name] = p
        p.nodes += nodes
        p.groups.append(NodeGroup(cores=cpus, mem_mb=mem, nodes=nodes,
                                  gpu=tuple(sorted(gpus.items()))))
        for t, c in gpus.items():
            p.gpu_types[t] = max(p.gpu_types.get(t, 0), c)
        if mem is not None:
            # THE LARGEST MACHINE'S, as `max_cpus` is: the queue's single
            # memory figure stands only where no machines are listed, and
            # the fit reads each machine's own (`scheduler.md` R0, R3).
            p.mem_mb = mem if p.mem_mb is None else max(p.mem_mb, mem)
    return list(parts.values())


class PartitionPolicy(NamedTuple):
    """What one partition's policy block states -- both facts ``None`` when
    the partition does not say (R3)."""
    def_mem_per_cpu_mb: Optional[int] = None
    max_cpus_per_node:  Optional[int] = None


def parse_scontrol_partitions(text: str) -> Dict[str, "PartitionPolicy"]:
    """``scontrol show partition`` -> ``{partition: PartitionPolicy}``.

    **The number nobody chose.**  SLURM grants this much memory per core when
    a job states no ``--mem``, so a 64-core job silently asks for 64 x it.  On
    ASU Sol that is 2 GB.

    ``sinfo`` cannot report it -- there is no format code -- so this is a
    second command rather than a wider one.

    A partition that sets ``DefMemPerNode`` instead maps to ``None``: it is a
    per-NODE default and not per-core, so deriving a per-core figure from it
    would invent one.  ``None`` means *this partition does not say*, which a
    reader must not read as zero (R3).

    ``MaxCPUsPerNode`` rides in the same block (`scheduler.md` R13): the
    partition's POLICY cap on one job's cores per node, distinct from how
    many cores the nodes have.
    """
    out: Dict[str, PartitionPolicy] = {}
    name: Optional[str] = None
    for chunk in (text or "").split("PartitionName="):
        if not chunk.strip():
            continue
        name = chunk.split()[0].strip()
        mb: Optional[int] = None
        cap: Optional[int] = None
        for tok in chunk.split():
            if tok.startswith("DefMemPerCPU="):
                try:
                    mb = int(tok.split("=", 1)[1])
                except ValueError:
                    mb = None      # "UNLIMITED" and friends say no number
            elif tok.startswith("MaxCPUsPerNode="):
                try:
                    cap = int(tok.split("=", 1)[1])
                except ValueError:
                    cap = None     # UNLIMITED: no policy cap stated
        out[name] = PartitionPolicy(mb, cap)
    return out


class QosLimit(NamedTuple):
    """One QoS's per-job limits, as ``sacctmgr show qos`` states them.

    A named tuple because positional readers of a widening tuple are how a
    field gets read as its neighbour.  ``None`` everywhere
    means *this QoS does not say* (R3).
    """
    maxwall_str:  Optional[str] = None
    maxwall_secs: Optional[int] = None
    #: ``MaxTRESPerJob``'s ``cpu=N`` -- the POLICY cap on one job's cores,
    #: which no amount of hardware overrides (R13).
    max_cpus_per_job: Optional[int] = None
    #: ``MaxSubmitJobsPerUser`` -- HOW MANY JOBS this QoS lets one user have
    #: submitted at once (R14).
    #:
    #: It is a different KIND of ceiling from the two above, which is why it
    #: needed its own rule rather than joining R13.  Those cap ONE job and
    #: are answerable from the job alone; this one caps the SET, and whether
    #: an ask fits depends on what the user already has queued.  A bench
    #: sweep submits many jobs at once by construction, so this is exactly
    #: the limit a sweep meets.
    max_submit_jobs_pu: Optional[int] = None


_TRES_CPU = re.compile(r"(?:^|,)cpu=(\d+)")


def parse_qos(text: str) -> Dict[str, "QosLimit"]:
    """Parse ``sacctmgr -nP show qos
    format=Name,MaxWall,MaxTRES,Flags,MaxSubmitJobsPerUser`` ->
    ``{name: QosLimit}``.  Empty MaxWall -> None (no QoS-level wall
    ceiling; the partition limit governs); a MaxTRES with no ``cpu=`` term
    caps other resources and says nothing about cores.

    **The job-count column is read from the END of the row, and that is
    deliberate.** These rows are positional -- ``sacctmgr -nP`` prints no
    header -- so a column inserted in the middle shifts every reader after
    it, which is the failure :class:`QosLimit`'s own docstring warns about.
    Appending cannot move ``MaxWall`` or ``MaxTRES``, and a row from the
    older four-column format simply has no fifth cell, which reads as *this
    probe never asked* rather than as *no limit*.
    """
    out: Dict[str, QosLimit] = {}
    for line in (text or "").splitlines():
        cols = [c.strip() for c in line.split("|")]
        if not cols or not cols[0]:
            continue
        name = cols[0]
        mw = cols[1] if len(cols) > 1 else ""
        secs: Optional[int] = None
        if mw:
            try:
                secs = parse_walltime(mw)
            except (ValueError, AttributeError):
                secs = None
        cpus: Optional[int] = None
        m = _TRES_CPU.search(cols[2]) if len(cols) > 2 else None
        if m:
            cpus = int(m.group(1))
        # Column 5 (index 4), appended after Flags.  Blank means the QoS
        # states no per-user submit cap; a missing column means this row
        # came from the older format and the question was never asked.
        submit: Optional[int] = None
        if len(cols) > 4 and cols[4]:
            try:
                submit = int(cols[4])
            except ValueError:
                submit = None
        out[name] = QosLimit(mw or None, secs, cpus, submit)
    return out


def parse_allowed_qos(text: str) -> Dict[str, Set[str]]:
    """Parse ``sacctmgr -nP show assoc user=$USER format=Partition,QOS`` --
    one association per row, its partition (blank: the association holds on
    every partition) and its comma-separated QoS list -> ``{partition: the
    QoS names the account may submit with there}``, the blank partition
    under ``""``.  Drops the ``no_submit`` marker.  Empty when the command
    answered nothing: the record then states no QoS (`derive_domains`)."""
    allowed: Dict[str, Set[str]] = {}
    for line in (text or "").splitlines():
        if "|" not in line:
            continue
        part, qos_field = line.rsplit("|", 1)       # QOS is the last field
        part = part.split("|")[-1].strip()
        for q in qos_field.split(","):
            q = q.strip()
            if q and q != "no_submit":
                allowed.setdefault(part, set()).add(q)
    return allowed


def _qos_on(allowed: Dict[str, Set[str]], partition: str) -> Set[str]:
    """The QoS names the account may submit with on ``partition``: the
    associations that name it, and the ones that name every partition."""
    return set(allowed.get("", set())) | set(allowed.get(partition, set()))


def _domain_name(partition: str, qos: Optional[str],
                 qos_on: Set[str]) -> str:
    """The handle a description names a pair by (`allocation.domain`).  The
    bare partition name goes to ONE pair of the partition: the one whose
    QoS is ``public``, else the one named like the partition -- the two
    spellings sites give an account's ordinary QoS -- or the pair the record
    states no QoS for; every other pair is ``<partition>-<qos>``, so two
    pairs never share a name.  A name, not a preference: every pair is
    recorded, and the header is written from the pair's partition and QoS,
    never from the name (`scheduler.md` § 4)."""
    bare = ("public" if "public" in qos_on
            else partition if partition in qos_on else None)
    if qos is None or qos == bare:
        return partition
    return f"{partition}-{qos}"


# --------------------------------------------------------------------- #
#  derivation (pure)                                                     #
# --------------------------------------------------------------------- #

def derive_domains(
    partitions: List[Partition],
    qos: Dict[str, Tuple[Optional[str], Optional[int]]],
    allowed: Dict[str, Set[str]],
) -> Tuple[List["Domain"], List[str]]:
    """Live probes -> **every (partition, qos) this account may submit to**,
    with the wall each allows, plus human notes.  Facts only.

    **One domain per allowed pair, and no QoS preferred** (`configuration.md`
    § 5 M-1; the user, 2026-10-17: the record is shown, never a veto): a
    partition the account holds three QoS on is three domains, each with
    the wall its pair allows, named by `_domain_name`.  When the account's
    QoS list could not be read (``allowed`` empty), every partition is one
    domain stating no QoS -- the header then carries no QoS line and the
    scheduler applies the account's default -- and a note says so.  *(Until
    2026-10-19 the probe chose one QoS per partition, ``public`` first, and
    assumed ``public`` when the list was unreadable.)*

    **No GPU filter**: every reachable partition is listed; what a
    partition *has* is the topology's answer, and what a run *wants* is the
    person's.

    Ordered cheapest ceiling first, deduped by ``(partition, qos)``.

    **Returns ``Domain`` objects, not mappings.**  A policy key is ABSENT
    when the probe never asked and ``null`` when it asked and SLURM stated
    no cap; ``Domain`` holds that tri-state in the type: ``UNSET`` is the
    default and never reaches disk, ``None`` means asked-and-uncapped and
    lands as ``null``.
    """
    from .record import Domain          # the type this returns

    notes: List[str] = []
    if not partitions:
        notes.append("sinfo listed no partitions.")
        return [], notes
    if not allowed:
        notes.append("could not read your allowed QoS (sacctmgr show assoc "
                     "user=$USER): the record states no QoS for any queue, "
                     "and a job's header carries no QoS line -- the "
                     "scheduler applies your account's default.")

    parts = sorted(partitions, key=lambda p: p.timelimit_secs)
    domains: List[Domain] = []

    def _row(name, max_time, part):
        kw: Dict[str, Any] = {}
        # The partition's GPU INVENTORY rides the row (`generator.md`
        # § 4.3a, 2026-08-21): sinfo's gres column is a measurement, and
        # without it a login node could not enumerate the GPU grid family
        # for the cluster behind it.  Facts only -- which type a run WANTS
        # stays the person's.
        if part.gpu_types:
            kw["gpu"] = dict(part.gpu_types)
        # EVERY MACHINE THIS DOMAIN HOLDS, not a number derived from them
        # (2026-08-27, user: *list available machine types explicitly and
        # allow cpu request to fit that range instead of one lowest fit*).
        # A partition is a queue: `htc` is 48-, 64- and 128-core nodes
        # under one name, and any single figure over them is an opinion.
        # Listing them lets the fit read each machine and the person see the
        # trade (`scheduler.md` R0, R3).
        if part.groups:
            kw["node_types"] = [g.as_row() for g in part.groups
                                if g.cores]
        # ``max_cores`` is the WIDEST node, as ``max_mem_gb`` is the largest:
        # the figure the fit reads where a record lists no machines (R3).
        widest = max((g.cores for g in part.groups if g.cores), default=None)
        if widest:
            kw["max_cores"] = widest
        # THE TWO MEMORY FACTS (`execution/scheduler.md` § 2).
        #
        # `max_mem_gb` is the largest machine's memory (what a node has).
        # `default_mem_per_core_gb` is what SLURM GRANTS PER CORE when a job
        # states no --mem.  They are different facts and the code
        # that reads one must not read the other (submission.md § 1).
        if part.mem_mb:
            kw["max_mem_gb"] = round(part.mem_mb / 1024.0, 1)
        # THE POLICY CEILING, beside the hardware one (R13).  ``max_cores``
        # says what the widest machine HAS; this says what the partition
        # LETS one job take of it -- both read into the fit.
        #
        # WRITTEN AS NULL WHEN ASKED AND UNSTATED, absent only when never
        # asked.
        if part.policy_queried:
            kw["max_cpus_per_node"] = part.max_cpus_per_node
            # ...AND THE PER-CORE DEFAULT, from the same block: null when
            # asked and unstated, absent only when never asked (R13).
            kw["default_mem_per_core_gb"] = (
                round(part.def_mem_per_cpu_mb / 1024.0, 2)
                if part.def_mem_per_cpu_mb else None)
        # CONSTRUCTED, not assembled.  A column this probe emits that
        # ``Domain`` does not declare is a TypeError HERE, at the line that
        # writes it.  Omitting a keyword is how a policy column stays UNSET (never
        # asked); passing ``None`` is how it says asked-and-uncapped.
        return Domain(name=name, max_time=max_time, partition=part.name,
                      qos=None, **kw)

    # EVERY PAIR, or the partition alone when no QoS could be read.
    pairs = [(p, q) for p in parts
             for q in (sorted(_qos_on(allowed, p.name)) or [None])]
    for p, q in pairs:
        q_secs = qos.get(q, QosLimit()).maxwall_secs if q else None
        # the wall is the SMALLER of the partition limit and the QoS ceiling
        if q_secs is not None and q_secs < p.timelimit_secs:
            max_time = qos[q].maxwall_str
        else:
            # AN UNSTATED LIMIT IS NOT COUNTED (`scheduler.md` R3): an
            # infinite partition's domain states no wall, and the fit asks
            # no ceiling of it (`fit.domain_ceiling_s` answers None).  A
            # 30-day cap was invented here until 2026-10-18.
            max_time = (p.timelimit_str
                        if p.timelimit_secs < _INFINITE_SECS else None)
            if p.timelimit_secs >= _INFINITE_SECS:
                notes.append(f"partition {p.name!r} has no time limit "
                             f"(infinite); the record states no wall for "
                             f"it, and a job there is fit against none.")
        row = _row(_domain_name(p.name, q, _qos_on(allowed, p.name)),
                   max_time, p)
        row.qos = q
        # Same absent-vs-null rule as ``max_cpus_per_node`` above: the key
        # rides whenever the QoS table answered for this QoS, null meaning
        # *asked; it states no cpu cap* -- and R14, how many jobs the QoS
        # lets one user submit at once (ASU Sol caps `debug` at 2).
        if q is not None and q in qos:
            row.max_cpus_per_job = qos[q].max_cpus_per_job
            row.max_submit_jobs = qos[q].max_submit_jobs_pu
        domains.append(row)

    seen: Set[Tuple[str, str]] = set()
    uniq: List[Domain] = []
    for d in domains:
        key = (d.partition, d.qos)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    uniq.sort(key=lambda d: _to_secs(d.max_time or ""))

    if allowed.get(""):
        notes.append(
            "an association that names no partition is taken to hold on "
            "every one sinfo lists (SLURM's own reading of a blank "
            "Partition); a partition that also limits its QoS (AllowQos in "
            "`scontrol show partition`) refuses such a pair at submission, "
            "and the refusal is shown as the scheduler's.")
    return uniq, notes


__all__ = [
    "Partition", "PartitionPolicy", "QosLimit", "parse_sinfo", "parse_qos", "parse_allowed_qos",
    "derive_domains",
]
