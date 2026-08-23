"""``molbuilder bench probe-scheduler`` -- turn a live SLURM cluster into a
proposed ``scheduler`` config block (job-system.md § 7).

Probes ``sinfo``/``sacctmgr`` on the target's login node and DERIVES the
``scheduler.{directives,gpu,routing}`` block the user would otherwise
hand-write by reading those tools (§ 4.1, § 4.3 — these and the section
references below cite the archived job-execution.md design record; the
live surface is `molbuilder bench probe-scheduler`, job-system § 7).  The
framework hardcodes
NO partition names or limits -- everything here comes from the live system
(the § 12 anti-hardcoding rule, made executable).

Pure parsing + derivation lives here (testable on captured text); the CLI
(``_cli.cmd_probe_scheduler``) runs the subprocesses and optionally merges
the result into ``.molbuilder.json`` via ``runtime_config.write_config_scope``.

Moved ``bench/probe.py`` -> ``molbuilder/scheduler_probe.py`` 2026-08-12,
and into the scheduler subsystem as ``scheduler/probe.py`` 2026-08-23
(follow-up to the U-program): what it produces is a ``scheduler`` CONFIG
block for molbuilder.json -- runtime_config's domain, floor 1 -- and
nothing it does is benchmarking.  It lives beside ``record.py``, the
other machine-probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# sinfo timelimit tokens that mean "no ceiling".
_INFINITE = {"infinite", "unlimited", "n/a", ""}
# A finite sentinel for an unbounded partition so a domain can still be built.
_INFINITE_SECS = 30 * 24 * 3600
_INFINITE_STR = "30-00:00:00"


def parse_walltime(s) -> int:
    """SLURM walltime string -> seconds.  Accepts the forms SLURM accepts:
    ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``
    (running-a-job.md § 5.3).  Empty -> 0.  Raises ValueError on garbage
    so a malformed config max_time fails loudly, not silently as 0."""
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


@dataclass
class Partition:
    """One SLURM partition, merged across its node groups."""
    name: str
    timelimit_str: str
    timelimit_secs: int
    nodes: int = 0
    gpu_types: Dict[str, int] = field(default_factory=dict)  # type -> max count
    #: Cores per node of the GPU-carrying node group(s) -- the SMALLEST
    #: when they differ, because it feeds a cap (2026-08-21, user: "why
    #: not autodetected?").  ``sinfo`` reports one row PER NODE GROUP, so
    #: a partition mixing 128-core CPU nodes with 48-core GPU nodes shows
    #: the GPU nodes' own cores on the GPU row -- measurable after all.
    gpu_cores: Optional[int] = None
    #: The widest node's cores across every group (CPU rows included).
    max_cpus: Optional[int] = None

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


def _parse_gres(gres: str) -> Dict[str, int]:
    """``gpu:a100:4,gpu:a100.20gb:16`` -> {'a100':4, 'a100.20gb':16}.
    Ignores non-gpu gres and the ``(null)`` placeholder.  Strips any
    ``(S:..)`` socket-affinity suffix SLURM may append."""
    out: Dict[str, int] = {}
    g = (gres or "").strip()
    if not g or g == "(null)":
        return out
    for tok in g.split(","):
        tok = tok.strip()
        if not tok.lower().startswith("gpu:"):
            continue
        parts = tok.split("(")[0].split(":")   # drop "(S:0)" affinity tail
        # gpu:<type>:<count>  OR  gpu:<count>  (untyped)
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


def parse_sinfo(text: str) -> List[Partition]:
    """Parse ``sinfo -h -o '%P|%<w>l|%D|%<w>G|%c'`` (pipe-delimited; fields
    may be space-padded by the width modifier -- we strip).  A partition
    appears once per node group; merge them (union GPU types, sum nodes,
    keep the time limit; per-group CPUS feed ``gpu_cores``/``max_cpus``).
    The default-partition ``*`` marker is stripped.  A 4-column capture
    (the pre-2026-08-21 format, no ``%c``) still parses -- the core
    columns simply stay ``None``."""
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
        gpus = _parse_gres(gres)
        p = parts.get(name)
        if p is None:
            p = Partition(name=name, timelimit_str=tl_str,
                          timelimit_secs=_to_secs(tl_str))
            parts[name] = p
        p.nodes += nodes
        for t, c in gpus.items():
            p.gpu_types[t] = max(p.gpu_types.get(t, 0), c)
        if cpus is not None:
            if gpus:
                p.gpu_cores = cpus if p.gpu_cores is None \
                    else min(p.gpu_cores, cpus)
            p.max_cpus = cpus if p.max_cpus is None \
                else max(p.max_cpus, cpus)
    return list(parts.values())


def parse_qos(text: str) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Parse ``sacctmgr -nP show qos format=Name,MaxWall,...`` ->
    ``{name: (maxwall_str|None, maxwall_secs|None)}``.  Empty MaxWall -> None
    (no QoS-level wall ceiling; the partition limit governs)."""
    out: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for line in (text or "").splitlines():
        cols = [c.strip() for c in line.split("|")]
        if not cols or not cols[0]:
            continue
        name = cols[0]
        mw = cols[1] if len(cols) > 1 else ""
        if mw:
            try:
                out[name] = (mw, parse_walltime(mw))
            except (ValueError, AttributeError):
                out[name] = (mw, None)
        else:
            out[name] = (None, None)
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

# ``best_gpu_type`` stood here until 2026-08-17 (N3).  It ranked GPU types --
# *"prefer FULL cards (no MIG slice) that are not l40, then the most plentiful"*
# -- to choose one to default to.  Ranking is a PREFERENCE, and a probe writes
# facts only (`configuration.md` § 5, M-1).  The GPU a run is sized against is
# the probed compute node's ``topology.gpu_type``, which is a measurement of the
# machine the job will land on rather than a vote across the cluster.  Deleted
# rather than left unused: its one caller went with it.


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
) -> Tuple[List[dict], List[str]]:
    """Live probes -> **every (partition, qos) this account may submit to**,
    with the wall each allows, plus human notes.  Facts only.

    Replaced ``derive_scheduler_block`` on 2026-08-17 (N3), which produced the
    same routing list wrapped in a whole ``scheduler`` config block -- ``kind``,
    ``gpu``, and a ``directives`` default whose partition was ``route_parts[0]``,
    *the cheapest*.  Cheapest is a preference, and the file it was written into
    was a person's (`configuration.md` § 5, M-1).  What survives is the part
    that was always a measurement.

    **No GPU filter**, and that is the second change.  The old derivation kept
    only partitions carrying the chosen full GPU type, which quietly assumed
    every question is a GPU question -- so a CPU-only partition was invisible
    even to a CPU benchmark.  Every reachable partition is listed; what a
    partition *has* is the topology's answer, and what a run *wants* is the
    person's.

    Ordered cheapest ceiling first, deduped by ``(partition, qos)``.
    """
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
    domains: List[dict] = []

    # A debug domain iff the user actually holds the debug QoS.  It rides on
    # the cheapest partition because a debug QoS is a wall, not a place.
    def _row(name, max_time, part):
        row = {"name": name, "max_time": max_time, "partition": part.name,
               "qos": None}
        # The partition's GPU INVENTORY rides the row (`generator.md`
        # § 4.3a, 2026-08-21): sinfo's gres column is a measurement, and
        # without it a login node could not enumerate the GPU grid family
        # for the cluster behind it.  Facts only -- which type a run WANTS
        # stays the person's.
        #
        # ``max_cores`` is probed too (2026-08-21, user: "why not
        # autodetected?"): sinfo reports one row PER NODE GROUP, so the
        # GPU nodes' own core count is on their row even inside a mixed
        # partition.  On a gpu-capable row it is the GPU nodes' cores --
        # exactly the cap the GPU grid checks -- and on a cpu-only row
        # the widest node's.  The row stays yours to edit.
        if part.gpu_types:
            row["gpu"] = dict(part.gpu_types)
            if part.gpu_cores:
                row["max_cores"] = part.gpu_cores
        elif part.max_cpus:
            row["max_cores"] = part.max_cpus
        return row

    if "debug" in allowed and "debug" in qos:
        mw_str, _ = qos["debug"]
        row = _row("debug", mw_str or "0-00:15:00", parts[0])
        row["qos"] = "debug"
        domains.append(row)

    for p in parts:
        q = _pick_qos(allowed, p.name)
        if not q:
            continue
        q_secs = qos.get(q, (None, None))[1]
        # the wall is the SMALLER of the partition limit and the QoS ceiling
        if q_secs is not None and q_secs < p.timelimit_secs:
            max_time = qos[q][0]
        else:
            max_time = (p.timelimit_str
                        if p.timelimit_secs < _INFINITE_SECS else _INFINITE_STR)
            if p.timelimit_secs >= _INFINITE_SECS:
                notes.append(f"partition {p.name!r} has no time limit "
                             f"(infinite); capped the domain at "
                             f"{_INFINITE_STR} -- adjust if needed.")
        row = _row(p.name, max_time, p)
        row["qos"] = q
        domains.append(row)

    seen: Set[Tuple[str, str]] = set()
    uniq: List[dict] = []
    for d in domains:
        key = (d["partition"], d["qos"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    uniq.sort(key=lambda d: _to_secs(d["max_time"]))

    notes.append(
        "ASSUMPTION: a QoS allowed to your account is valid on any reachable "
        "partition (preferred 'public'). sinfo/assoc do not give the "
        "per-partition QoS list -- confirm with `scontrol show partition "
        "<name>` (AllowQos) and drop any domain you cannot actually submit "
        "to (e.g. a privately-owned partition).")
    return uniq, notes


__all__ = [
    "Partition", "parse_sinfo", "parse_qos", "parse_allowed_qos",
    "derive_domains",
]
