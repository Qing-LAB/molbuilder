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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SCHEMA = "molbuilder/environment@2"
#
# @1 -> @2 (N2, 2026-08-17): ``domains`` added, and ``site.qos`` became a field
# something actually writes.  A MAJOR bump rather than a minor, deliberately:
# ``from_dict`` tolerates missing keys, so an @1 record would parse -- and would
# read as *a cluster with no reachable domains*, which is indistinguishable from
# a real cluster where you hold no QoS.  The bump is what makes an old record
# say "I predate the probe" instead of answering the question wrongly.
# (`configuration.md` § 5.)

#: The record's filename, at every scope.  It was a string literal in three
#: modules (`jobset/prep.py`, `jobset/summarize.py`, and the door below) until
#: N1 -- the same defect `task.FILENAME` was created to fix.
FILENAME = "environment.json"

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
class Domain:
    """One **reachable** (partition, qos) pair, and what it allows.

    A fact, not a preference: it says *you may submit here, for this long*,
    never *submit here*.  Which domain a run wants is `molbuilder.json`'s
    (`configuration.md` § 5, M-1).

    **One type for both ways a fact arrives** (2026-08-17).  It carried four
    fields when only the prober built one, and a hand-declared row -- which is
    how a workstation states a cluster's capability -- went through
    `get_routing` as a **raw dict** instead, so the same function returned a
    4-key mapping or a 6-key one depending on which branch ran.  A caller could
    not rely on the shape of its own answer.  The columns below are the ones
    people actually write; anything else rides in ``extra``.

    ``extra`` is R10, 2026-08-12, made a property of the TYPE rather than of
    one code path: rebuilding a row from a known-key list made drafting a
    column indistinguishable from not writing it.  A reader owns only the keys
    it checks.
    """
    name:      str
    partition: str
    qos:       str
    max_time:  Optional[str] = None
    node_type: Optional[str] = None
    max_cores: Optional[int] = None
    max_mem_gb: Optional[float] = None
    gpu:       Optional[Dict[str, Any]] = None
    #: Columns this reader does not check, kept verbatim.
    extra:     Dict[str, Any] = field(default_factory=dict)

    #: The keys :meth:`from_row` recognises; everything else goes to ``extra``.
    _KNOWN = ("name", "partition", "qos", "max_time", "node_type",
              "max_cores", "max_mem_gb", "gpu")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Optional["Domain"]:
        """A mapping -> a Domain, or ``None`` when it is not one.

        The ONE parser, used by the probe, by the record reader and by a
        declared ``scheduler.routing`` row alike.  ``name``/``partition``/
        ``qos`` have no default: a row missing one is not a domain, and
        dropping it beats inventing a blank one that a `prep` check would then
        compare an ask against.
        """
        d = dict(row)
        if not {"name", "partition", "qos"} <= set(d):
            return None
        known = {k: d.pop(k) for k in list(d) if k in cls._KNOWN}
        return cls(**known, extra=d)

    def to_row(self) -> Dict[str, Any]:
        """A Domain -> the mapping it came from, unknown columns included."""
        row = {k: getattr(self, k) for k in self._KNOWN
               if getattr(self, k) is not None}
        row.update(self.extra)
        return row


@dataclass
class Environment:
    """The portable target description (§ 5.2).  Produced by probes,
    consumed by adapters; neither knows the other's internals."""
    scheduler: str                                   # "slurm" | "workstation"
    topology:  Topology = field(default_factory=Topology)
    site:      Site = field(default_factory=Site)
    #: Every (partition, qos) this account may actually submit to, with its
    #: wall.  Empty on a workstation, and empty on a cluster until `jobset
    #: probe` has run -- which is why @2 exists: on @1 those two were the same
    #: value and nothing could tell them apart.
    domains:   List["Domain"] = field(default_factory=list)
    source:    Dict[str, str] = field(default_factory=dict)
    detected_at: Optional[str] = None
    # the LIVE writer (resolve_target, prep step 1); "prep-bench@1" -- the
    # deleted verb -- stamped every fresh record until R10, 2026-08-12
    # (U19 fixed only from_dict's re-read default)
    tool:      str = "jobset-prep@1"

    # ----- JSON round-trip (the persisted contract) ----------------- #

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "detected_at": self.detected_at,
            "scheduler": self.scheduler,
            "topology": asdict(self.topology),
            "site": asdict(self.site),
            "domains": [d.to_row() for d in self.domains],
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
        domains = [d for d in (Domain.from_row(r)
                               for r in (d.get("domains") or []))
                   if d is not None]
        return cls(
            scheduler=str(d.get("scheduler", "workstation")),
            topology=topo, site=site, domains=domains,
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


def topology_field_types() -> Dict[str, type]:
    """The declared-fact vocabulary: each :class:`Topology` field and the
    type its value must parse as -- derived from the dataclass, so a field
    added to the schema is automatically declarable and one that is renamed
    cannot linger here (one home; M-1's ``flag`` door is typed by this).
    """
    import typing
    hints = typing.get_type_hints(Topology)
    out: Dict[str, type] = {}
    for f in fields(Topology):
        args = [a for a in typing.get_args(hints[f.name])
                if a is not type(None)]
        # A plain (non-Optional) annotation has no args -- the hint IS the
        # type.  str was the fallback for that branch too, which would have
        # silently text-typed the first plain field ever added (milestone
        # review N3; unreachable today, every field is Optional).
        hint = hints[f.name]
        out[f.name] = args[0] if args else (
            hint if isinstance(hint, type) else str)
    return out


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

    ``qos``/``account`` are left ``None`` **here** because ``sinfo`` cannot
    answer them -- not because nothing can.  This docstring claimed they were
    "site policy, not reliably derivable", and `scheduler_probe` disproves it
    in the same tree: ``parse_allowed_qos`` reads exactly your QoS from
    ``sacctmgr -nP show assoc user=$USER``.  Two modules disagreeing about
    whether one fact is detectable is what `configuration.md` § 5 M-1 was
    written to end.

    The split is by **command**, not by knowability: this function is the NODE
    probe and asks ``sinfo``/``scontrol``; the cluster probe (`jobset probe`)
    asks ``sacctmgr`` and fills ``site.qos`` and ``domains``.  What genuinely
    is policy, and stays in `molbuilder.json`, is which of them you *want*."""
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


# --------------------------------------------------------------------- #
#  The door (N1) -- configuration.md § 5, M-4                            #
# --------------------------------------------------------------------- #
#
# This module owned the SCHEMA, the dataclasses and the JSON round-trip, and
# not the FILE.  That gap is why three call sites grew three different shapes:
# a raw ``write_text``, a read returning an ``Environment``, and a second read
# returning a plain ``dict``.  Everything below is the missing layer, and no
# consumer opens the file itself any more.


def machine_scope_path() -> Path:
    """Where the MACHINE-scope record lives — ``jobset probe``'s target.

    ``~/.config/molbuilder/environment.json``, honouring ``XDG_CONFIG_HOME``.
    The convention is `runtime_config._per_user_fallback_path`'s, and it is
    **mirrored rather than imported**: this module is stdlib-only by contract
    (it ships to the target and runs in a backend env with no molbuilder on
    it), and `runtime_config` is not.

    **Per-user only, with no cwd step** — deliberately unlike
    `molbuilder.json`, whose lookup tries the working directory first.  A
    calculation is very often the working directory, so a cwd step here would
    make the machine scope and the calculation scope the same file whenever
    you happened to run from inside a bundle, and M-3's precedence would then
    compare a record against itself.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "molbuilder" / FILENAME


def environments_dir() -> Path:
    """Where NAMED target records live — ``<machine scope>/environments/``.

    One record describes one machine, and the machine you are preparing for is
    not always the machine you are on.  A benchmark prepped on a workstation
    for a cluster was silently measured against the workstation, because there
    was exactly one record and it was this box's.
    """
    return machine_scope_path().parent / "environments"


def named_environments() -> Dict[str, Path]:
    """``{name: path}`` for every declared or copied-back target record."""
    d = environments_dir()
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.json"))}


def record_scopes(bundle_dir=None,
                  target: Optional[str] = None) -> List[Tuple[str, Path]]:
    """**Where a machine record may live, in precedence order** — the one
    place that order is written down.

    Returns ``[(label, path), ...]``, first match wins.  Stated as data rather
    than as an if/elif chain inside the reader, so *"which file answers?"* is
    read off a list instead of traced through control flow -- the shape
    `runtime_config._SECTIONS` already uses for the config scopes next door.

    ``target`` names a machine explicitly and is resolved by the CALLER
    (:func:`machine_for`), because an unknown name is an error rather than a
    scope that failed to match.
    """
    scopes: List[Tuple[str, Path]] = []
    if bundle_dir is not None:
        scopes.append(("calculation", Path(bundle_dir) / FILENAME))
    if target is not None:
        known = named_environments()
        if target in known:
            scopes.append((f"target:{target}", known[target]))
    scopes.append(("machine", machine_scope_path()))
    return scopes


def read_environment(path) -> Optional["Environment"]:
    """The record at ONE path, or ``None``.

    Takes the **file**, not the directory holding it.  It took a directory and
    joined :data:`FILENAME`, which forced a second reader (``_read_named``) the
    moment named targets arrived, because a named record's name IS its
    filename.  Two readers of one format is the defect this door exists to
    remove, so the join moved out to :func:`record_scopes` where the paths are
    built.

    ``None`` covers every way there is no usable answer here — absent,
    unreadable, not JSON, wrong schema, malformed.  **Malformed is ``None``,
    not an exception**: a hand-edited file earns a fall-through to the next
    scope, and the caller has one thing to check instead of four.

    The narrowness of the second ``except`` is deliberate and was paid for.
    This read called a method that did not exist (``from_json``) from
    2026-08-11 to 2026-08-12, and a broad ``except Exception`` swallowed the
    ``AttributeError`` — so the persisted answer was never read back and every
    `prep` silently re-probed.  A bad file earns tolerance; a bug does not.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Environment.from_dict(raw)
    except (ValueError, TypeError, KeyError):
        return None


def write_environment(env: "Environment", path) -> Path:
    """The record to ONE path, atomically.  Returns the path written.

    Through ``persist.write_json``, the shared writer (L1, pure stdlib, so the
    ship-to-target rule holds).  It emits exactly what the hand-rolled
    ``write_text(env.to_json() + "\n")`` did — 2-space indent, trailing
    newline — while adding the unique-temp + ``os.replace`` that keeps two
    writers from colliding.

    Takes the **file** for the same reason the reader does; the ``filename=``
    escape hatch that briefly existed here was the one-filename rule being
    patched around rather than dropped.
    """
    from .persist import write_json
    return write_json(Path(path), env.to_dict())


class UnknownTarget(Exception):
    """``--target NAME`` cannot be honoured — unknown, or contradicted by the
    record this calculation was already prepped against."""

    def __init__(self, name: str, known):
        self.name, self.known = name, sorted(known)
        listed = ", ".join(self.known) or "(none)"
        super().__init__(
            f"no machine record named {name!r}.  Known targets: {listed}.  "
            f"Write one with `molbuilder jobset probe --write --name {name}` "
            f"on that machine, or declare it by hand in "
            f"{environments_dir() / (name + '.json')}.")

    @classmethod
    def conflict(cls, name: str, bundle_dir) -> "UnknownTarget":
        """This calculation already carries a DIFFERENT machine's record.

        Silently keeping the snapshot would make ``--target`` a no-op on every
        folder after the first prep -- the flag would appear to work and
        change nothing.  Silently re-snapshotting would let stage 2 of a
        ladder resolve against a different machine than stage 1, which is the
        exact disagreement the once-per-bundle rule exists to prevent.  So
        neither: say so, and let the person choose.
        """
        exc = cls.__new__(cls)
        exc.name, exc.known = name, []
        Exception.__init__(exc, (
            f"--target {name!r} does not match the machine this calculation "
            f"was already prepped for.  A calculation is snapshotted once so "
            f"two stages cannot resolve against different machines.\n"
            f"  To move it: delete {Path(bundle_dir) / FILENAME} and prep "
            f"again with --target {name}.\n"
            f"  To keep it: drop --target."))
        return exc


def machine_for(bundle_dir=None, *, target: Optional[str] = None,
                probe: bool = False) -> Optional["Environment"]:
    """**The precedence, entire** — the one function a caller asks.

    Walks :func:`record_scopes` and returns the first record that reads.  The
    first one found is the whole answer: **there is no field-level merge**, and
    that is the rule rather than an omission.  Two partial records blended at
    read time would describe a machine that exists in no file, and it would
    defeat the standing guarantee that two stages of one calculation cannot
    disagree about their own target.  A calculation that should follow a
    re-probed machine deletes its file.

    ``probe`` adds a fresh detection when no scope answered, and **defaults to
    off**.  It is opt-in because probing shells out to ``sinfo``, ``scontrol``,
    ``lscpu`` and ``nvidia-smi``, and a *read-only getter must not do that*:
    `get_routing` asks this on every call, so with the default the other way
    round every domain lookup ran four subprocesses -- 56 ms a call here, and a
    round trip to the scheduler on a login node.  `resolve_target` (prep step
    1) is the one caller that wants it, because it is the one that WRITES the
    answer down afterwards.
    """
    # A named target is validated FIRST, before any scope is consulted.  It
    # read the calculation's snapshot first and returned it when present, on
    # the reasoning that the snapshot *is* the answer already taken.  That made
    # a typo silent: `--target nope` on an already-prepped folder prepped
    # happily against whatever was snapshotted, which is precisely the mistake
    # this flag exists to catch.
    if target is not None and target not in named_environments():
        raise UnknownTarget(target, named_environments())

    named = None
    for label, path in record_scopes(bundle_dir, target):
        env = read_environment(path)
        if env is None:
            continue
        if label.startswith("target:"):
            named = env
        elif label == "calculation" and target is not None:
            # A NAMED target that contradicts the snapshot is a question
            # nobody can answer for the user: refuse rather than silently
            # keeping the old one or silently replacing it.
            want = read_environment(named_environments()[target])
            if want is not None and want.to_dict() != env.to_dict():
                raise UnknownTarget.conflict(target, bundle_dir)
        return env
    if named is not None:
        return named
    if not probe:
        return None
    try:
        return resolve_environment()
    except Exception:              # pragma: no cover - probing is optional
        return None


__all__ = [
    "SCHEMA", "FILENAME", "Topology", "Site", "Domain", "Environment",
    "detect_scheduler", "detect_topology", "detect_site",
    "resolve_environment",
    "machine_scope_path", "environments_dir", "named_environments",
    "record_scopes",
    "read_environment", "write_environment", "machine_for", "UnknownTarget",
    "_parse_scontrol_node", "_parse_lscpu", "_parse_nvidia_smi_l",
    "_parse_gres",
]
