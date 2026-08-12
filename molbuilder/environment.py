"""Target-machine detection -> a portable ``Environment`` record.

The first half of the benchmark workflow's pluggable seam
(docs/execution/job-system.md, § 5): **probes** that learn the
target's scheduler + hardware topology + site facts, and the versioned
JSON record they produce (``environment@1`` — registry row: job-contracts
§ 6.1; produced by ``resolve_target`` at prep step 1) that every later stage
and any external tool reads.

**Section references below cite the archived job-execution.md design
record** (the live homes are job-system § 7 and job-contracts § 6.1; R8,
2026-08-12 — the numbers no longer resolve in the live doc set and are
kept as the design's own history).  Detection priority for topology
(§ 4.6 there): the **compute node**, not where
this runs.

  1. SLURM  -> ``scontrol show node`` (correct compute-node shape, askable
     from a login node);
  2. local  -> ``lscpu`` + ``nvidia-smi -L`` (valid only when run ON the
     target: a workstation or an interactive allocation);
  3. declared -> caller-supplied overrides / defaults.

**Stdlib-only** (subprocess + json + dataclasses): this module is meant to
also ship to the target and run in the backend env, which has no
molbuilder/numpy (the self-contained rule, § 2).  The pure parsers
(``_parse_*``) take text and are unit-tested; the ``detect_*`` wrappers
add the (guarded) subprocess calls.  Any command that is missing or fails
degrades to ``None`` fields -- never an exception.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA = "molbuilder/environment@1"

# Normalized GPU-type tokens we recognize in an nvidia-smi name string.
_GPU_TYPES = ("a100", "a30", "h100", "h200", "v100", "a40", "l40", "l4",
              "p100", "t4", "rtx")


# --------------------------------------------------------------------- #
#  Data model (§ 5.2)                                                   #
# --------------------------------------------------------------------- #


@dataclass
class Topology:
    """The target's hardware shape.  ``None`` = not detected (kept, never
    omitted, so a consumer can tell 'absent' from 'unknown')."""
    sockets:          Optional[int] = None
    cores_per_socket: Optional[int] = None
    threads_per_core: Optional[int] = None
    numa_per_socket:  Optional[int] = None
    gpus_per_node:    Optional[int] = None
    gpu_type:         Optional[str] = None
    mem_total_gb:     Optional[float] = None


@dataclass
class Site:
    """Scheduler-specific submission facts (empty on a workstation)."""
    partition: Optional[str] = None
    qos:       Optional[str] = None
    account:   Optional[str] = None


@dataclass
class Environment:
    """The portable target description (§ 5.2).  Produced by probes,
    consumed by adapters; neither knows the other's internals."""
    scheduler: str                                   # "slurm" | "workstation"
    topology:  Topology = field(default_factory=Topology)
    site:      Site = field(default_factory=Site)
    source:    Dict[str, str] = field(default_factory=dict)
    detected_at: Optional[str] = None
    tool:      str = "prep-bench@1"

    # ----- JSON round-trip (the persisted contract) ----------------- #

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "detected_at": self.detected_at,
            "scheduler": self.scheduler,
            "topology": asdict(self.topology),
            "site": asdict(self.site),
            "source": dict(self.source),
            "tool": self.tool,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Environment":
        from .persist import check_schema
        check_schema(str(d.get("schema", "")), SCHEMA,
                           label="environment")
        # Tolerant of unknown/extra keys; missing keys -> dataclass default.
        topo_fields = {f for f in Topology.__dataclass_fields__}
        site_fields = {f for f in Site.__dataclass_fields__}
        topo = Topology(**{k: v for k, v in (d.get("topology") or {}).items()
                           if k in topo_fields})
        site = Site(**{k: v for k, v in (d.get("site") or {}).items()
                       if k in site_fields})
        return cls(
            scheduler=str(d.get("scheduler", "workstation")),
            topology=topo, site=site,
            source=dict(d.get("source") or {}),
            detected_at=d.get("detected_at"),
            # the default names the LIVE writer; "prep-bench@1" (the deleted
            # verb) stood here until U19, stamping every re-read record
            # with a tool that no longer exists
            tool=str(d.get("tool", "jobset-prep@1")),
        )


# --------------------------------------------------------------------- #
#  Pure parsers (text -> values; unit-tested directly)                  #
# --------------------------------------------------------------------- #


def _to_int(s) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _gpu_type_from_name(name: str) -> Optional[str]:
    low = name.lower()
    for t in _GPU_TYPES:
        if t in low:
            return t
    return None


def _parse_gres(gres: str) -> Tuple[Optional[int], Optional[str]]:
    """``gpu:a100:4`` / ``gpu:a100:4(S:0-1)`` / ``gpu:4`` ->
    ``(count, type)``.

    A node's ``Gres`` can list MULTIPLE comma-separated resources (e.g.
    ``gpu:a100:4,mps:400``); split on ``,`` and parse only the ``gpu:``
    entry, else a trailing ``mps:0`` would be read as ``0`` GPUs."""
    g = gres.strip()
    if not g or g.lower() in ("(null)", "none"):
        return None, None
    g = re.sub(r"\(.*?\)", "", g)                    # drop (S:..) socket tag
    entries = [e for e in g.split(",") if e]
    gpu_entry = next((e for e in entries
                      if e.split(":", 1)[0].lower() == "gpu"), None)
    target = gpu_entry if gpu_entry is not None else (entries[0]
                                                      if entries else g)
    parts = target.split(":")
    count = _to_int(parts[-1])
    gtype = None
    for p in parts:
        t = _gpu_type_from_name(p)
        if t:
            gtype = t
            break
    return count, gtype


def _parse_scontrol_node(text: str) -> Topology:
    """Parse ``scontrol show node`` output (space-separated key=value
    tokens, possibly multi-line) into a Topology."""
    kv: Dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            kv.setdefault(k, v)
    t = Topology()
    t.sockets = _to_int(kv.get("Sockets"))
    t.cores_per_socket = _to_int(kv.get("CoresPerSocket"))
    t.threads_per_core = _to_int(kv.get("ThreadsPerCore"))
    rm = _to_int(kv.get("RealMemory"))               # MB
    if rm is not None:
        t.mem_total_gb = round(rm / 1024.0, 1)
    if "Gres" in kv:
        n, gt = _parse_gres(kv["Gres"])
        t.gpus_per_node, t.gpu_type = n, gt
    return t


def _parse_lscpu(text: str) -> Topology:
    """Parse ``lscpu`` (``key: value`` lines) into a Topology (no GPU/mem;
    those come from nvidia-smi / /proc)."""
    kv: Dict[str, str] = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if v:
            kv[k.strip()] = v.strip()
    t = Topology()
    t.sockets = _to_int(kv.get("Socket(s)"))
    t.cores_per_socket = _to_int(kv.get("Core(s) per socket"))
    t.threads_per_core = _to_int(kv.get("Thread(s) per core"))
    numa = _to_int(kv.get("NUMA node(s)"))
    if numa is not None and t.sockets and t.sockets > 0:
        t.numa_per_socket = max(1, numa // t.sockets)
    return t


def _parse_nvidia_smi_l(text: str) -> Tuple[Optional[int], Optional[str]]:
    """``nvidia-smi -L`` -> (gpu_count, normalized_type)."""
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("GPU ")]
    if not lines:
        return None, None
    gtype = None
    for ln in lines:
        # Match the model NAME only -- strip the "(UUID: GPU-a30...)" tail,
        # whose hex can false-match type tokens like a30/a40/a100.
        gtype = _gpu_type_from_name(ln.split("(", 1)[0])
        if gtype:
            break
    return len(lines), gtype


# --------------------------------------------------------------------- #
#  Detection (subprocess wrappers; all guarded)                         #
# --------------------------------------------------------------------- #


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    """Best-effort capture of ``cmd`` stdout; None on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_scheduler() -> Tuple[str, str]:
    """Return ``(scheduler, source)``: ``slurm`` if ``sbatch`` is on PATH
    or a ``SLURM_*`` var is set, else ``workstation``."""
    if shutil.which("sbatch"):
        return "slurm", "path:sbatch"
    if any(k.startswith("SLURM_") for k in os.environ):
        return "slurm", "env:SLURM_*"
    return "workstation", "no-sbatch"


def _slurm_pick_node(partition: Optional[str]) -> Optional[str]:
    """Pick a node from ``partition`` (prefer one advertising a GPU)."""
    args = ["sinfo", "-h", "-N", "-o", "%N %G"]
    if partition:
        args += ["-p", partition]
    out = _run(args)
    if not out:
        return None
    first = None
    for line in out.splitlines():
        toks = line.split()
        if not toks:
            continue
        if first is None:
            first = toks[0]
        if len(toks) > 1 and "gpu" in toks[1].lower():
            return toks[0]                            # prefer a GPU node
    return first


def detect_topology(scheduler: str, *,
                    partition: Optional[str] = None,
                    overrides: Optional[dict] = None
                    ) -> Tuple[Topology, str]:
    """Resolve the compute-node topology by the § 4.6 priority.  Returns
    ``(Topology, source)`` where source is ``scontrol`` | ``lscpu`` |
    ``flag`` | ``unknown``.  Overrides (declared flags) win field-by-field
    and mark the source ``flag`` when they supplied anything."""
    topo, source = Topology(), "unknown"

    if scheduler == "slurm":
        node = _slurm_pick_node(partition)
        if node:
            out = _run(["scontrol", "show", "node", node])
            if out:
                topo, source = _parse_scontrol_node(out), "scontrol"

    # Local probe ONLY when we are physically ON the target: a workstation,
    # or a SLURM job/allocation (``SLURM_JOB_ID`` set).  On a SLURM LOGIN
    # node ``lscpu`` would describe the *login* node, not the compute node
    # (§ 4.6) -- so if scontrol failed there, leave topology unknown and
    # let declared flags fill it, rather than report the wrong machine.
    on_node = (scheduler == "workstation"
               or bool(os.environ.get("SLURM_JOB_ID")))
    if source == "unknown" and on_node:
        lsc = _run(["lscpu"])
        if lsc:
            topo, source = _parse_lscpu(lsc), "lscpu"
        n, gt = _parse_nvidia_smi_l(_run(["nvidia-smi", "-L"]) or "")
        if n is not None:
            topo.gpus_per_node, topo.gpu_type = n, gt
        if topo.mem_total_gb is None:
            topo.mem_total_gb = _read_mem_total_gb()

    if overrides:
        applied = False
        for k, v in overrides.items():
            if v is not None and hasattr(topo, k):
                setattr(topo, k, v)
                applied = True
        if applied:
            source = "flag" if source == "unknown" else f"{source}+flag"

    return topo, source


def _read_mem_total_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1048576.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def detect_site(scheduler: str) -> Tuple[Site, str]:
    """SLURM **default partition** from ``sinfo`` (the one ``%P`` marks
    with ``*``); empty on a workstation.

    ``qos``/``account`` are intentionally left ``None`` -- they are site
    policy, not reliably derivable from ``sinfo``, so they come from the
    user's config (the SlurmAdapter / scheduler block), not detection."""
    if scheduler != "slurm":
        return Site(), "n/a"
    site = Site()
    out = _run(["sinfo", "-h", "-o", "%P"])          # PARTITION (default=*)
    if out:
        names = [t for t in out.split() if t]
        for name in names:
            if name.endswith("*"):                   # the default partition
                site.partition = name.rstrip("*")
                break
        if site.partition is None and names:         # else the first listed
            site.partition = names[0].rstrip("*")
    return site, ("sinfo" if site.partition else "unknown")


def resolve_environment(*, overrides: Optional[dict] = None,
                        now_iso: Optional[str] = None,
                        scheduler_override: Optional[str] = None
                        ) -> Environment:
    """Run the probes in order and assemble the Environment (§ 4.4).

    ``overrides`` is a flat dict of declared topology values (e.g.
    ``{"cores_per_socket": 24, "gpus_per_node": 4}``) that win over
    detection.  ``scheduler_override`` forces the scheduler.  ``now_iso``
    stamps ``detected_at`` (passed in so the module stays free of wall-
    clock calls for deterministic tests)."""
    if scheduler_override:
        scheduler, sch_src = scheduler_override, "flag"
    else:
        scheduler, sch_src = detect_scheduler()

    site, site_src = detect_site(scheduler)
    topo, topo_src = detect_topology(
        scheduler, partition=site.partition, overrides=overrides)

    return Environment(
        scheduler=scheduler, topology=topo, site=site,
        source={"scheduler": sch_src, "topology": topo_src, "site": site_src},
        detected_at=now_iso,
    )


__all__ = [
    "SCHEMA", "Topology", "Site", "Environment",
    "detect_scheduler", "detect_topology", "detect_site",
    "resolve_environment",
    "_parse_scontrol_node", "_parse_lscpu", "_parse_nvidia_smi_l",
    "_parse_gres",
]
