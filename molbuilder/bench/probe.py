"""``molbuilder bench probe-scheduler`` -- turn a live SLURM cluster into a
proposed ``scheduler`` config block (slurm-integration.md § 4.5).

Probes ``sinfo``/``sacctmgr`` on the target's login node and DERIVES the
``scheduler.{directives,gpu,routing}`` block the user would otherwise
hand-write by reading those tools (§ 4.1, § 4.3).  The framework hardcodes
NO partition names or limits -- everything here comes from the live system
(the § 12 anti-hardcoding rule, made executable).

Pure parsing + derivation lives here (testable on captured text); the CLI
(``_cli.cmd_probe_scheduler``) runs the subprocesses and optionally merges
the result into ``.molbuilder.json`` via ``runtime_config.write_config_scope``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .adapters import parse_walltime

# sinfo timelimit tokens that mean "no ceiling".
_INFINITE = {"infinite", "unlimited", "n/a", ""}
# A finite sentinel for an unbounded partition so a domain can still be built.
_INFINITE_SECS = 30 * 24 * 3600
_INFINITE_STR = "30-00:00:00"


@dataclass
class Partition:
    """One SLURM partition, merged across its node groups."""
    name: str
    timelimit_str: str
    timelimit_secs: int
    nodes: int = 0
    gpu_types: Dict[str, int] = field(default_factory=dict)  # type -> max count

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
    """Parse ``sinfo -h -o '%P|%<w>l|%D|%<w>G'`` (pipe-delimited; fields may
    be space-padded by the width modifier -- we strip).  A partition appears
    once per node group; merge them (union GPU types, sum nodes, keep the
    time limit).  The default-partition ``*`` marker is stripped."""
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
        gpus = _parse_gres(gres)
        p = parts.get(name)
        if p is None:
            p = Partition(name=name, timelimit_str=tl_str,
                          timelimit_secs=_to_secs(tl_str))
            parts[name] = p
        p.nodes += nodes
        for t, c in gpus.items():
            p.gpu_types[t] = max(p.gpu_types.get(t, 0), c)
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

def best_gpu_type(partitions: List[Partition]) -> Optional[str]:
    """Pick the GPU type to default to: prefer FULL cards (no MIG ``.`` slice)
    that are not ``l40`` (crippled FP64 -- § 11/§ 8), then the most plentiful
    (by node count).  Mirrors decision D3 (a100 on Sol) without naming it."""
    counts: Dict[str, int] = {}
    for p in partitions:
        for t in p.gpu_types:
            counts[t] = counts.get(t, 0) + p.nodes
    if not counts:
        return None

    def quality(t: str) -> Tuple[bool, int]:
        full = "." not in t            # MIG slices look like a100.20gb
        ok = t != "l40"
        return (full and ok, counts[t])

    return max(counts, key=quality)


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


def derive_scheduler_block(
    partitions: List[Partition],
    qos: Dict[str, Tuple[Optional[str], Optional[int]]],
    allowed: Set[str],
) -> Tuple[Optional[dict], List[str]]:
    """Live probes -> proposed ``scheduler`` block + a list of human notes /
    assumptions (slurm-integration.md § 4.5).  Returns ``(block, notes)``;
    ``block`` is None when no usable GPU partition/QoS was found."""
    notes: List[str] = []
    gpu_parts = sorted((p for p in partitions if p.has_gpu),
                       key=lambda p: p.timelimit_secs)
    if not gpu_parts:
        notes.append("no GPU-bearing partition found in sinfo.")
        return None, notes
    gtype = best_gpu_type(gpu_parts)
    if not gtype:
        notes.append("no GPU type parsed from sinfo gres.")
        return None, notes
    # Only route to partitions that actually carry the chosen full GPU type
    # (drops MIG-only, l40-only, and exotic-accelerator partitions).
    route_parts = [p for p in gpu_parts if gtype in p.gpu_types]
    if not allowed:
        notes.append("could not read your allowed QoS (sacctmgr assoc); "
                     "assuming 'public'. Verify with sacctmgr show assoc "
                     f"user=$USER.")
        allowed = {"public"}

    routing: List[dict] = []
    # 1) a debug domain iff the user has the debug QoS.
    if "debug" in allowed and "debug" in qos:
        mw_str, _ = qos["debug"]
        routing.append({
            "name": "debug", "max_time": mw_str or "0-00:15:00",
            "partition": route_parts[0].name, "qos": "debug"})
    # 2) one domain per reachable GPU partition (cheapest -> most general).
    for p in route_parts:
        q = _pick_qos(allowed, p.name)
        if not q:
            continue
        q_secs = qos.get(q, (None, None))[1]
        # max_time = the smaller of the partition limit and the QoS ceiling.
        if q_secs is not None and q_secs < p.timelimit_secs:
            max_time = qos[q][0]
        else:
            max_time = (p.timelimit_str
                        if p.timelimit_secs < _INFINITE_SECS else _INFINITE_STR)
            if p.timelimit_secs >= _INFINITE_SECS:
                notes.append(f"partition '{p.name}' has no time limit "
                             f"(infinite); capped the domain at {_INFINITE_STR}"
                             " -- adjust if needed.")
        routing.append({"name": p.name, "max_time": max_time,
                        "partition": p.name, "qos": q})

    # de-dupe by (partition, qos) keeping the first (cheapest) occurrence,
    # then order by ceiling ascending (§ 4.3 first-fitting = cheapest).
    seen: Set[Tuple[str, str]] = set()
    uniq: List[dict] = []
    for d in routing:
        key = (d["partition"], d["qos"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    uniq.sort(key=lambda d: _to_secs(d["max_time"]))
    routing = uniq

    default_q = _pick_qos(allowed, route_parts[0].name)
    block = {
        "kind": "slurm",
        "directives": {"partition": route_parts[0].name, "qos": default_q},
        "gpu": {"partition": route_parts[0].name, "default_type": gtype},
        "routing": routing,
    }
    notes.append(
        "ASSUMPTION: a QoS allowed to your account is valid on any reachable "
        "partition (preferred 'public'). sinfo/assoc do not give the "
        "per-partition QoS list -- confirm with `scontrol show partition "
        "<name>` (AllowQos) and drop any domain you cannot actually submit "
        "to (e.g. a privately-owned partition).")
    notes.append(
        "exclusivity + memory are POLICY, not probed (§ 4.3.1): gpu.exclusive "
        "is left to you / the --exclusive prep flag, and gpu.mem is never "
        "written (memory is estimated per-job).")
    return block, notes


__all__ = [
    "Partition", "parse_sinfo", "parse_qos", "parse_allowed_qos",
    "best_gpu_type", "derive_scheduler_block",
]
