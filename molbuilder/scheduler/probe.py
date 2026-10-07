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
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from .quantities import parse_walltime


# sinfo timelimit tokens that mean "no ceiling".
_INFINITE = {"infinite", "unlimited", "n/a", ""}
# A finite sentinel for an unbounded partition so a domain can still be built.
_INFINITE_SECS = 30 * 24 * 3600
_INFINITE_STR = "30-00:00:00"



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
    #: Every distinct machine this partition holds.  The MEASUREMENT; the
    #: scalars below are summaries over it, kept because callers already
    #: read them.
    groups: List["NodeGroup"] = field(default_factory=list)
    gpu_types: Dict[str, int] = field(default_factory=dict)  # type -> max count
    #: Cores per node of the GPU-carrying node group(s) -- the SMALLEST
    #: when they differ, because it feeds a cap (2026-08-21, user: "why
    #: not autodetected?").  ``sinfo`` reports one row PER NODE GROUP, so
    #: a partition mixing 128-core CPU nodes with 48-core GPU nodes shows
    #: the GPU nodes' own cores on the GPU row -- measurable after all.
    gpu_cores: Optional[int] = None
    #: The widest node's cores across every group (CPU rows included).
    max_cpus: Optional[int] = None
    #: MEMORY PER NODE, in MB, of the SMALLEST node group -- the safe reading
    #: for a ceiling, the same rule ``gpu_cores`` uses.
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

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_types)


# --------------------------------------------------------------------- #
#  parsing (pure -- operate on captured command text)                   #
# --------------------------------------------------------------------- #

def _to_secs(timelimit: str) -> int:
    """SLURM partition TIMELIMIT -> seconds; 'infinite'/'unlimited' -> a big
    finite sentinel so a domain can still be built (flagged by the caller)."""
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
    keep the time limit; per-group CPUS feed ``gpu_cores``/``max_cpus``, and
    per-group MEMORY feeds ``mem_mb``).
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
            # the smallest, which is the safe reading for a cap.
            try:
                cpus = int(cols[4].rstrip("+"))
            except ValueError:
                cpus = None
        mem: Optional[int] = None
        if len(cols) >= 6 and cols[5]:
            # Same "48+" convention as the core column: the base is the
            # SMALLEST of a differing group, which is the safe ceiling.
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
        if cpus is not None:
            if gpus:
                p.gpu_cores = cpus if p.gpu_cores is None \
                    else min(p.gpu_cores, cpus)
            p.max_cpus = cpus if p.max_cpus is None \
                else max(p.max_cpus, cpus)
        if mem is not None:
            # SMALLEST across groups: a partition whose nodes differ can only
            # promise the least of them, and a ceiling that over-promises is
            # the one that sends a job to a queue it does not fit.
            p.mem_mb = mem if p.mem_mb is None else min(p.mem_mb, mem)
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


def parse_allowed_qos(text: str) -> Set[str]:
    """Parse ``sacctmgr -nP show assoc user=$USER format=QOS`` (the QOS field
    is a comma-separated list; may span several association rows) -> the union
    set of QoS names the user may submit to.  Drops the ``no_submit`` marker."""
    allowed: Set[str] = set()
    for line in (text or "").splitlines():
        for q in line.split("|")[-1].split(","):   # QOS is the last field
            q = q.strip()
            if q and q != "no_submit":
                allowed.add(q)
    return allowed


# --------------------------------------------------------------------- #
#  derivation (pure)                                                     #
# --------------------------------------------------------------------- #

def _pick_qos(allowed: Set[str], partition_name: str) -> Optional[str]:
    """Choose a submit QoS: prefer ``public`` -> the partition-named QoS ->
    the first allowed QoS that is not ``debug``/``private``.  None if the
    user has no usable QoS."""
    if "public" in allowed:
        return "public"
    if partition_name in allowed:
        return partition_name
    for q in sorted(allowed):
        if q not in ("debug", "private"):
            return q
    return next(iter(sorted(allowed)), None)



def derive_domains(
    partitions: List[Partition],
    qos: Dict[str, Tuple[Optional[str], Optional[int]]],
    allowed: Set[str],
) -> Tuple[List["Domain"], List[str]]:
    """Live probes -> **every (partition, qos) this account may submit to**,
    with the wall each allows, plus human notes.  Facts only.

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
        notes.append("could not read your allowed QoS (sacctmgr assoc); "
                     "assuming 'public'. Verify with sacctmgr show assoc "
                     "user=$USER.")
        allowed = {"public"}

    parts = sorted(partitions, key=lambda p: p.timelimit_secs)
    domains: List[Domain] = []

    # A debug domain iff the user actually holds the debug QoS.  It rides on
    # the cheapest partition because a debug QoS is a wall, not a place.
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
        # under one name, and a single figure has to be either a floor
        # that refuses work the wide nodes would run, or a ceiling that
        # admits work most nodes cannot hold.  Listing them refuses
        # neither and lets the person see the trade.
        if part.groups:
            kw["node_types"] = [g.as_row() for g in part.groups
                                if g.cores]
        # ``max_cores`` stays, as the WIDEST node -- the honest ceiling for
        # a refusal, since `admits` "only refuses what the record
        # positively rules out" (R3) and SLURM will not place a job on a
        # node too small; it waits for one that fits.
        widest = max((g.cores for g in part.groups if g.cores), default=None)
        if widest:
            kw["max_cores"] = widest
        # THE TWO MEMORY FACTS (`execution/scheduler.md` § 2).
        #
        # `max_mem_gb` is the CEILING (what the node has).
        # `default_mem_per_core_gb` is what SLURM GRANTS PER CORE when a job
        # states no --mem.  They are different facts and the code
        # that reads one must not read the other (submission.md § 1).
        if part.mem_mb:
            kw["max_mem_gb"] = round(part.mem_mb / 1024.0, 1)
        # THE POLICY CEILING, beside the hardware one (R13).  ``max_cores``
        # says what the widest machine HAS; this says what the partition
        # LETS one job take of it, and the smaller governs.
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

    if "debug" in allowed and "debug" in qos:
        mw_str = qos["debug"].maxwall_str
        row = _row("debug", mw_str or "0-00:15:00", parts[0])
        row.qos = "debug"
        # Same absent-vs-null rule as the loop below -- and unconditional
        # here, because this arm's own guard just proved the QoS table
        # answered for ``debug``.
        row.max_cpus_per_job = qos["debug"].max_cpus_per_job
        # R14: HOW MANY JOBS this QoS lets one user submit at once (ASU
        # Sol caps `debug` at 2).
        row.max_submit_jobs = qos["debug"].max_submit_jobs_pu
        domains.append(row)

    for p in parts:
        q = _pick_qos(allowed, p.name)
        if not q:
            continue
        q_secs = qos.get(q, QosLimit()).maxwall_secs
        # the wall is the SMALLER of the partition limit and the QoS ceiling
        if q_secs is not None and q_secs < p.timelimit_secs:
            max_time = qos[q].maxwall_str
        else:
            max_time = (p.timelimit_str
                        if p.timelimit_secs < _INFINITE_SECS else _INFINITE_STR)
            if p.timelimit_secs >= _INFINITE_SECS:
                notes.append(f"partition {p.name!r} has no time limit "
                             f"(infinite); capped the domain at "
                             f"{_INFINITE_STR} -- adjust if needed.")
        row = _row(p.name, max_time, p)
        row.qos = q
        # Same absent-vs-null rule as ``max_cpus_per_node`` above: the key
        # rides whenever the QoS table answered for this QoS, null meaning
        # *asked; it states no cpu cap*.
        if q in qos:
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
    uniq.sort(key=lambda d: _to_secs(d.max_time))

    notes.append(
        "ASSUMPTION: a QoS allowed to your account is valid on any reachable "
        "partition (preferred 'public'). sinfo/assoc do not give the "
        "per-partition QoS list -- confirm each with `scontrol show "
        "partition` and its name (AllowQos), and drop any domain you cannot "
        "actually submit "
        "to (e.g. a privately-owned partition).")
    return uniq, notes


__all__ = [
    "Partition", "PartitionPolicy", "QosLimit", "parse_sinfo", "parse_qos", "parse_allowed_qos",
    "derive_domains",
]
