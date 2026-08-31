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
    #: The instruction set: ``x86_64``, ``aarch64``, ...  As the machine
    #: spells it -- SLURM's ``Arch=`` or ``lscpu``'s ``Architecture:``,
    #: never normalised, because a name we invent is a name nothing else
    #: uses.
    #:
    #: Added 2026-08-26 (user: *x64 vs arm are different, don't mix them*).
    #: ASU Sol offers an ``arm`` partition -- Grace-Hopper, aarch64 -- in the
    #: same menu as eight x86 ones, and nothing here could tell them apart:
    #: an x86 conda env does not activate usefully on aarch64 and an
    #: AVX-512 binary does not run there at all, so the failure is total
    #: rather than slow.  ``None`` keeps its usual meaning (R3, *an unstated
    #: limit never bars*), which is what leaves every record written before
    #: this field filtering nothing.
    arch:             Optional[str] = None


@dataclass
class Site:
    """Scheduler-specific submission facts (empty on a workstation)."""
    partition: Optional[str] = None
    qos:       Optional[str] = None
    account:   Optional[str] = None


@dataclass(frozen=True)
class Device:
    """One kind of accelerator the nodes of a domain offer.

    The **interpreted** form of a domain's ``gpu`` column -- see
    :func:`_read_devices` for the two spellings that column arrives in and why
    both are read here rather than at each call site.
    """
    type:     str
    per_node: Optional[int] = None
    mem_gb:   Optional[float] = None


class _Unset:
    """The one instance below (``UNSET``) marks a policy column NO PROBE
    EVER ASKED -- distinct from ``None``, which means *asked; no cap
    stated*.  That distinction is the record's tri-state (`scheduler.md`
    R13, `checkpointing.md` S3's absent-vs-null): :meth:`Domain.to_row`
    drops ``UNSET`` but writes ``None`` as ``null``, so *asked, uncapped*
    survives the trip to disk.  Built 2026-08-28, the day a fresh Sol
    probe asked both policy questions and the written record could not
    show it -- ``to_row`` dropped every ``None`` field indiscriminately.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        # never truthy: a leaked UNSET must not pass an "is there a cap?"
        return False


UNSET: Any = _Unset()


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
    max_cores: Optional[int] = None
    max_mem_gb: Optional[float] = None
    #: What SLURM grants PER CORE when a job states no ``--mem`` -- the number
    #: that turns 64 cores into a 128 GB ask nobody made.  A DECLARED column
    #: since 2026-08-23, when the probe started measuring it: `asu-sol.md`
    #: § 5.3 has documented it since the row was designed, and it rode in
    #: ``extra`` until the probe could fill it.  ``None`` means the partition
    #: does not say, never zero (R3).
    default_mem_per_core_gb: Optional[float] = None
    #: POLICY ceilings on one job's cores (`scheduler.md` R13), beside the
    #: hardware one (``max_cores`` = the widest machine).  Both read, the
    #: smaller governs; a hardware ceiling cannot stand in for a policy
    #: one.  ``max_cpus_per_job`` is the QoS's ``MaxTRESPerJob`` ``cpu``
    #: term; ``max_cpus_per_node`` is the partition's ``MaxCPUsPerNode``.
    #:
    #: TRI-STATE, unlike every other column: a number caps, ``None`` means
    #: *asked; no cap stated* (written as ``null``), ``UNSET`` means the
    #: question was never asked (stays off the disk).  ``None`` here is a
    #: measurement, never *unlimited by assumption* (R3).
    max_cpus_per_job:  Any = UNSET
    max_cpus_per_node: Any = UNSET
    #: R14 -- how many jobs this domain's QoS lets ONE USER have submitted
    #: at once (``MaxSubmitJobsPerUser``).  Same tri-state as the two above,
    #: and the same reason it exists: the column was in the QoS table all
    #: along and the format list did not ask for it.
    #:
    #: It is a different KIND of ceiling from every other field here.  Those
    #: cap ONE job and are answerable from the job alone; this caps the SET,
    #: so whether an ask fits depends on what is already queued.  A bench
    #: sweep submits many jobs at once by construction -- which is how Sol's
    #: `debug` (cap 2) took two of six and refused four, with the record
    #: unable to have said so.
    max_submit_jobs:   Any = UNSET
    gpu:       Optional[Dict[str, Any]] = None
    #: Every distinct machine this domain holds: ``[{cores, nodes, mem_gb,
    #: gpu}, ...]``.  DECLARED since 2026-08-27, because a partition is a
    #: QUEUE and not a machine type -- Sol's ``htc`` is 48-, 64- and
    #: 128-core nodes under one name.
    #:
    #: Any single core figure over them is an opinion: a floor refuses work
    #: the wide nodes would run, a ceiling admits work most nodes cannot
    #: hold.  ``max_cores`` is kept as the widest (R3 -- refuse only what
    #: the record positively rules out) and this is what a person reads to
    #: choose.  Empty means the record does not say, never *no machines*.
    node_types: Optional[List[Dict[str, Any]]] = None
    #: Where a GPU job goes when that differs from ``partition``.  DECLARED
    #: since 2026-08-23 (scheduler.md § 4): it redirects real work, and until
    #: then it rode in ``extra`` -- the bag this reader documents as
    #: uninterpreted -- read by two call sites in routing that reached past
    #: the type to a raw key.  A value that changes where a job lands is a
    #: field or it is a bug waiting.
    gpu_partition: Optional[str] = None
    #: Columns this reader does not check, kept verbatim.
    extra:     Dict[str, Any] = field(default_factory=dict)

    #: The keys :meth:`from_row` recognises; everything else goes to ``extra``.
    #: The scalar ``node_type`` was RETIRED 2026-08-27 (`scheduler.md`
    #: R11): it said a queue has one machine type, which R0 measured to be
    #: false, and the S3 check reading it had never fired because the
    #: probe could not honestly write it.  A declared one now lands in
    #: ``extra``, uninterpreted; ``node_types`` is the machine list.
    _KNOWN = ("name", "partition", "qos", "max_time",
              "max_cores", "max_mem_gb", "default_mem_per_core_gb",
              "max_cpus_per_job", "max_cpus_per_node", "max_submit_jobs",
              "gpu", "gpu_partition", "node_types")

    #: The tri-state policy columns (see their field note): ``None`` is a
    #: real answer here and lands as ``null``; only ``UNSET`` stays off
    #: the disk.  Every other ``_KNOWN`` column keeps the record style --
    #: ``None`` says nothing and is not written.
    _NULLABLE = ("max_cpus_per_job", "max_cpus_per_node",
                 "max_submit_jobs")

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
        """A Domain -> the mapping it came from, unknown columns included.

        ``None`` is dropped -- the record says nothing by silence -- except
        in the ``_NULLABLE`` policy columns, where ``None`` means *asked;
        no cap stated* and must land as ``null``.  Dropping those too is
        the 2026-08-28 leak: a Sol probe asked both policy questions and
        the written record could not show it.
        """
        row: Dict[str, Any] = {}
        for k in self._KNOWN:
            v = getattr(self, k)
            if v is UNSET or (v is None and k not in self._NULLABLE):
                continue
            row[k] = v
        row.update(self.extra)
        return row

    @property
    def devices(self) -> Tuple[Device, ...]:
        """What this domain's ``gpu`` column says the nodes offer.

        Empty when it says nothing.  **Every reader of the GPU inventory goes
        through here** -- the column has two spellings, and reading it at the
        call site is what let two of them disagree (:func:`_read_devices`).
        """
        return _read_devices(self.gpu)


#: The keys that mark a ``gpu`` column as ONE device spelled out rather than a
#: map of types.  None of the three is a GPU gres type, so their presence is
#: unambiguous -- which is what makes the two spellings safe to accept.
_DEVICE_DESCRIPTOR_KEYS = ("type", "per_node", "mem_gb")


def _read_devices(gpu: Any) -> Tuple[Device, ...]:
    """A domain's ``gpu`` column -> the devices it names.  **The one reader.**

    The column arrives in two spellings, because two things write it:

      * a **probe** maps gres TYPE to per-node COUNT, ``{"a100": 4,
        "a100.20gb": 16}`` -- one entry per type ``sinfo`` reported, and no
        memory, because ``sinfo`` does not report it;
      * a person **declares** one device and describes it,
        ``{"type": "a100", "per_node": 4, "mem_gb": 80}`` -- the shape
        `execution/asu-sol.md` § 5.3 tells them to write.

    Both are one fact -- *what the nodes of this domain offer* -- so both parse
    to the same type here, and no caller re-decides.  That they were read at
    two call sites instead is how, until 2026-08-23, a hand-declared row made
    `prep bench` refuse with *"records several GPU types (mem_gb, per_node,
    type)"*: one reader knew only the map, and read the descriptor's key names
    as device names.

    An unreadable count is ``None``, never a raise and never a zero -- a column
    we cannot read is not a domain with no devices (R3), and admission must be
    able to tell those apart.  The user's own spelling is never rewritten: this
    interprets the column, `to_row` still returns what was written.
    """
    if not isinstance(gpu, dict) or not gpu:
        return ()
    if any(k in gpu for k in _DEVICE_DESCRIPTOR_KEYS):
        gtype = gpu.get("type")
        return (Device(type=str(gtype) if gtype else "gpu",
                       per_node=_to_int(gpu.get("per_node")),
                       mem_gb=_to_float(gpu.get("mem_gb"))),)
    return tuple(Device(type=str(name), per_node=_to_int(count))
                 for name, count in gpu.items())


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
    #: HOW A SHELL ON THIS MACHINE ENTERS THE ENVIRONMENT -- the preamble
    #: to run and the activation form to use, as that machine states them
    #: in its own ``molbuilder.json``.
    #:
    #: **It is a fact about the TARGET, so it travels on the target's
    #: record.**  A wrapper is generated on a workstation and executed on a
    #: cluster; the two activate differently (`module load mamba` +
    #: `source activate` on ASU Sol, a `conda.sh` hook on the workstation),
    #: and until 2026-08-24 nothing carried the difference.  `prep
    #: --target sol` resolved Sol's topology and queues from this record
    #: and then baked the WORKSTATION's preamble into every wrapper, so
    #: every job on Sol died with
    #: ``line 196: /home/.../conda.sh: No such file or directory``.
    #:
    #: ``{}`` means the record predates this field or was written by a
    #: probe that could not read a config.  Absent is NOT "use the local
    #: machine's" -- that substitution is the bug -- so a caller
    #: generating for a named target REFUSES and asks for a re-probe.
    #:
    #: Shape: ``{"preamble": str, "activation": str}``, both optional.
    script_generation: Dict[str, str] = field(default_factory=dict)
    #: WHICH ENVIRONMENTS EXIST ON THIS MACHINE, by name.
    #:
    #: A FACT, and the other half of the pair above: *which* env you want
    #: for a category is a preference and stays in `molbuilder.json`
    #: (`envs.<category>`); whether that name exists THERE is a property of
    #: the machine (`configuration.md` § 5 M-1).  A generator that checks
    #: the wanted env against the machine it is standing on answers the
    #: wrong question for a bundle bound elsewhere -- the same shape as
    #: baking that machine's activation.
    #:
    #: Enumerated, never entered: `conda env list --json` reports every env
    #: from inside any one of them, so a probe running in `molbuilder` sees
    #: `molbuilder-siesta-gpu` without activating it.  The apparent
    #: circularity -- *"probing needs an env"* -- is only about the probe's
    #: own env, never the ones a generated script will use.
    #:
    #: ``[]`` means the probe could not enumerate (no conda on PATH, or a
    #: record written before this field).  Empty is "unknown", not "none":
    #: a gate cannot refuse on it.
    conda_envs: List[str] = field(default_factory=list)
    #: The instruction set the environments above were enumerated ON, as
    #: ``platform.machine()`` spells it.  **The other half of the pair**
    #: (user, 2026-08-26: *we should know our compiled/installed
    #: architecture*): a name in ``conda_envs`` is not portable, and
    #: ``molbuilder-siesta`` built for x86_64 is different software from one
    #: built for aarch64 under the one string.
    #:
    #: A mismatch is what fails, so a check needs BOTH numbers -- this one
    #: and ``topology.arch``.  Without it a wrong-architecture failure
    #: arrives disguised: `envs/builds.py` looks for
    #: ``x86_64-conda-linux-gnu-gcc`` by name, so on aarch64 it simply finds
    #: nothing and reports an unknown compiler version rather than the
    #: actual cause.
    #:
    #: ``None`` means a record written before this field (R3).
    env_arch:   Optional[str] = None
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
            # Absent, not empty, when the probe could read no config: a key
            # that is missing and a key that is {} are different claims to
            # anything testing for it.
            **({"script_generation": dict(self.script_generation)}
               if self.script_generation else {}),
            **({"conda_envs": sorted(self.conda_envs)}
               if self.conda_envs else {}),
            **({"env_arch": self.env_arch} if self.env_arch else {}),
            "source": dict(self.source),
            "tool": self.tool,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Environment":
        from ..persist import check_schema
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
            script_generation={
                k: str(v) for k, v in
                (d.get("script_generation") or {}).items()
                if k in ("preamble", "activation") and v is not None},
            conda_envs=[str(e) for e in (d.get("conda_envs") or [])],
            env_arch=(str(d["env_arch"]) if d.get("env_arch") else None),
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


def _to_float(s) -> Optional[float]:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def _gpu_type_from_name(name: str) -> Optional[str]:
    low = name.lower()
    for t in _GPU_TYPES:
        if t in low:
            return t
    return None


def _parse_gres(gres: str) -> Tuple[Optional[int], Optional[str]]:
    """``gpu:a100:4`` -> ``(count, type)`` -- a Topology states ONE device
    kind, so this narrows what `quantities.parse_gres` reads in full.

    **The reading itself is not done here any more** (2026-08-24).  This
    matched the type against `_GPU_TYPES`, a hard-coded list, and on ASU Sol
    that turned `gh200` into `h200` by substring, `a100.40gb` into `a100`,
    and `hl225` into nothing at all -- so a machine record could state a
    device its nodes do not have.  The token carries the type; a list of
    names cannot keep up with a site's hardware.

    `_gpu_type_from_name` survives for the input it was RIGHT for:
    ``nvidia-smi`` prints marketing names (``NVIDIA A100-SXM4-40GB``), not
    gres tokens, and matching those against known names is the only way to
    read them.

    The first entry wins when a node lists several kinds, which is what a
    single ``gpu_type`` field can say.  An UNTYPED ``gpu:4`` yields type
    ``None`` rather than the literal ``"gpu"``: the record's field means
    *which device*, and "gpu" answers nothing.
    """
    from .quantities import parse_gres
    found = parse_gres(gres)
    if not found:
        return None, None
    gtype, count = next(iter(found.items()))
    return count, (None if gtype == "gpu" else gtype)


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
    t.arch = kv.get("Arch") or None                  # x86_64 / aarch64
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
    t.arch = kv.get("Architecture") or None
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
    "site policy, not reliably derivable", and `probe` disproves it
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
    The convention is :func:`molbuilder.config_dir.config_dir`'s and is now
    IMPORTED.  It was **mirrored** here, on the grounds that this module is
    stdlib-only and `runtime_config` is not -- true, and the reason the rule
    does not live in `runtime_config`.  It is not a reason to spell it twice:
    `config_dir` is L1 pure stdlib exactly like `persist`, which this module
    already imports (`write_environment` -> `..persist.write_json`).

    **Per-user only, with no cwd step.**  `molbuilder.json` had one until
    2026-08-31 and no longer does, so the two now agree rather than
    contrasting (`configuration.md` § 2.1a).  A
    calculation is very often the working directory, so a cwd step here would
    make the machine scope and the calculation scope the same file whenever
    you happened to run from inside a bundle, and M-3's precedence would then
    compare a record against itself.
    """
    from ..config_dir import config_dir
    return config_dir() / FILENAME


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


def known_machines() -> List[Dict[str, object]]:
    """Every machine a calculation could be prepared FOR, described once.

    One list, two readers: ``jobset machines`` prints it and
    ``GET /api/task-setup/machines`` serves it.  The terminal and the browser
    must not be able to disagree about which machines exist or which of them
    can be read -- that disagreement is the whole reason a user cannot tell
    whether the record they copied over actually arrived.

    Each entry: ``name``, ``kind`` (``target`` for a named record, ``local``
    for this machine), ``path``, ``readable``, ``summary``, ``detected_at``.

    **An unreadable record is described, never dropped** -- the user wrote it,
    and hiding it leaves them waiting for something that cannot happen
    (`preparing-for-another-machine.md` § 5).  ``readable`` is the flag; the
    summary says how to rewrite it.
    """
    from .admit import domain_ceiling_s          # one parser for max_time

    def _describe(env, name, kind, path):
        if env is None:
            # Absent and unreadable are one answer from `read_environment`,
            # so the message covers both -- and the REMEDY differs by scope:
            # this machine's record is written by a bare `probe --write`,
            # while a named target takes `--name` and must be probed on the
            # machine it describes.  (Said `--name (this machine)` until
            # 2026-08-22, which is not a name and not a command.)
            fix = ("`jobset probe --write` here"
                   if kind == "local"
                   else "`jobset probe --write --name %s` on that machine"
                        % name)
            return {"name": name, "kind": kind, "path": str(path),
                    "readable": False, "detected_at": "", "domains": [],
                    "mem_total_gb": None, "gpus_per_node": None,
                    "gpu_type": None,
                    "summary": "no record here yet, or it cannot be read -- "
                               "write one with " + fix}
        # CORES: the RANGE the machines span, when the record lists them.
        #
        # `sockets x cores_per_socket` describes ONE node -- whichever
        # `sinfo` printed first (`_slurm_pick_node`) -- and a partition is a
        # queue, not a machine type (`scheduler.md` R0).  On Sol that read
        # "64 cores" for a cluster whose machines are 48, 64 AND 128, in the
        # card a person picks a machine from.
        #
        # R3 keeps every older record working: one written before
        # `node_types` existed says nothing here, and falls back to the one
        # figure it does have.  (Caught in the browser, 2026-08-27.)
        from .quantities import core_range as _core_range
        from .quantities import machine_sizes as _machine_sizes
        cores = getattr(env.topology, "cores_per_socket", None)
        sockets = getattr(env.topology, "sockets", None)
        total = (cores * sockets) if (cores and sockets) else None
        spread = _core_range(_machine_sizes(env.domains))
        bits = [env.scheduler or "unknown scheduler"]
        if spread:
            bits.append(f"{spread} cores")
        elif total:
            bits.append(f"{total} cores")
        # MEMORY, which was measured and never shown (2026-08-24).  The
        # probe records `mem_total_gb` for every machine, and the summary
        # named the scheduler, the cores, the GPUs and the domain count --
        # so the one number a person sizing a job most wants sat in the
        # file unread.
        _mem = getattr(env.topology, "mem_total_gb", None)
        if _mem:
            bits.append(f"{float(_mem):g} GB")
        if getattr(env.topology, "gpus_per_node", None):
            bits.append(f"{env.topology.gpus_per_node}\u00d7 "
                        f"{env.topology.gpu_type or 'GPU'}")
        if env.domains:
            bits.append(f"{len(env.domains)} domain(s)")
        # THE QUEUES, with the ceilings a surface can default from
        # (2026-08-24, user): the browser's Task-setup tab lets a person
        # pick a machine, then one of ITS queues, and fills the time and
        # memory asks with what that queue actually allows.  Measured
        # values only -- `None` where the queue states nothing, never a
        # number invented here (R3: an unstated limit never bars).
        return {"name": name, "kind": kind, "path": str(path),
                "readable": True, "summary": " \u00b7 ".join(bits),
                "detected_at": env.detected_at or "",
                # The node's own ceiling, for a machine that states no
                # queues: a workstation has nothing to default a memory
                # ask from, though its RAM is a real limit and asking for
                # more than the box has is meaningless.
                "mem_total_gb": getattr(env.topology, "mem_total_gb", None),
                "gpus_per_node": getattr(env.topology, "gpus_per_node", None),
                "gpu_type": getattr(env.topology, "gpu_type", None),
                "domains": [{"name": d.name,
                             "partition": d.partition,
                             "qos": d.qos,
                             "max_time": d.max_time,
                             "max_time_s": domain_ceiling_s(d),
                             "max_cores": d.max_cores,
                             "max_mem_gb": d.max_mem_gb,
                             "default_mem_per_core_gb":
                                 d.default_mem_per_core_gb,
                             "gpu": bool(d.gpu)}
                            for d in (env.domains or [])]}

    out = [_describe(read_environment(path), name, "target", path)
           for name, path in sorted(named_environments().items())]
    here = machine_scope_path()
    out.append(_describe(read_environment(here), "(this machine)", "local",
                         here))
    return out


def choice_required(machines=None) -> bool:
    """Whether preparing must be told WHICH machine (`…-another-machine.md` 4).

    True exactly when a named record exists, because "this machine" is always
    a candidate too -- so any named record makes the question real.  The same
    rule the CLI refuses on, so the browser cannot offer a silent default the
    terminal would reject.
    """
    ms = known_machines() if machines is None else machines
    return any(m["kind"] == "target" for m in ms)


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
    from ..persist import write_json
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
    def unreadable(cls, name: str, path) -> "UnknownTarget":
        """The record exists under that name but does not read.

        Absent, unreadable, not JSON, wrong schema -- `read_environment`
        answers all of them with ``None`` so a caller has one thing to
        check.  For a NAMED target that single answer must not mean "try
        the next scope": the user named this machine, so silently prepping
        for a different one is the mistake the flag exists to catch.
        """
        exc = cls.__new__(cls)
        exc.name, exc.known = name, []
        Exception.__init__(exc, (
            f"--target {name!r} names a record that cannot be read: {path}.\n"
            f"  It is absent, not JSON, or a schema this molbuilder does not "
            f"know.  Re-write it with `molbuilder jobset probe --write "
            f"--name {name}` on that machine."))
        return exc

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


class AmbiguousTarget(Exception):
    """Several machines could be meant and nobody said which.

    The cost is asymmetric, which is the whole argument: being asked costs
    one flag; being given the wrong machine costs a queue wait, an
    allocation, and a set of numbers that look plausible.  That is the
    failure ``--target`` was introduced for -- a benchmark prepped for a
    cluster, silently measured against the workstation it was prepped on.

    Raised only when the question is real: no record beside the calculation
    yet (a snapshot IS the answer already taken), no ``--target``, and more
    than one machine to mean.  One record and silence still proceeds --
    there is no ambiguity to resolve, so there is no question to ask
    (`preparing-for-another-machine.md` § 4, C1).
    """

    def __init__(self, choices):
        self.choices = sorted(choices)
        listed = "\n".join(f"    --target {c}" for c in self.choices
                           if c != "(this machine)")
        listed += f"\n    --target {LOCAL_TARGET}   (this machine)"
        super().__init__(
            "several machines could be meant and none was named.  Say which "
            "this calculation is for:\n" + listed +
            "\n    (there is no default; name one of the above)\n"
            "  A record is written by `molbuilder jobset probe --write "
            "--name NAME` on the machine it describes.")


#: THE TYPEABLE NAME FOR THIS MACHINE (2026-08-24).  ``known_machines``
#: displays ``(this machine)``, which is a label and not something a person
#: can type at ``--target``; and with any named record on file, OMITTING
#: ``--target`` raised `AmbiguousTarget` -- whose own message said *"omit
#: --target only when this machine is the one"*.  So the instruction the
#: refusal gave was the action that produced it, and preparing for the box
#: you are sitting at became impossible the moment you saved one cluster
#: record.  C1 protects against SILENCE; naming this machine is not silence.
LOCAL_TARGET = "this"


def machine_for(bundle_dir=None, *, target: Optional[str] = None,
                probe: bool = False,
                local_only: bool = False) -> Optional["Environment"]:
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

    ``local_only`` asks a DIFFERENT question from everything else in this
    function: not *"which machine is this calculation for"* (bundle / target /
    C1's ambiguity guard -- all of that is precedence among CANDIDATES for the
    calculation's answer), but *"what does the box this process is literally
    running on know about itself, full stop."*  R9's second check
    (`jobset/submit.py::_reject_if_this_machine_says_no`) wants exactly that:
    a re-admission against THIS machine's own probe, independent of and in
    addition to whichever machine the calculation is prepped for.  Bug found
    2026-08-23 -- a workstation with named targets (``environments/sol.json``)
    but no local probe of its own raised ``AmbiguousTarget`` from inside a
    read-only re-check that never asked about a target at all.  The C1
    question and this one only LOOK alike because both start from "no target
    was named"; C1 protects the case where that silence would make a wrong
    machine's numbers travel into a wrapper.  Here there is no wrapper, no
    numbers travelling, and no target in the question -- so C1 does not apply
    and must not run.
    """
    if local_only:
        env = read_environment(machine_scope_path())
        if env is not None or not probe:
            return env
        try:
            return resolve_environment()
        except Exception:          # pragma: no cover - probing is optional
            return None
    # A named target is validated FIRST, before any scope is consulted.  It
    # read the calculation's snapshot first and returned it when present, on
    # the reasoning that the snapshot *is* the answer already taken.  That made
    # a typo silent: `--target nope` on an already-prepped folder prepped
    # happily against whatever was snapshotted, which is precisely the mistake
    # this flag exists to catch.
    if target == LOCAL_TARGET:
        # EXPLICIT "the box I am on".  Goes down the same door the R9
        # re-check uses, so there is one reader for "what does this
        # machine know", and C1 below is skipped because the question it
        # guards -- which machine is meant -- has just been answered.
        env = read_environment(machine_scope_path())
        if env is not None or not probe:
            return env
        try:
            return resolve_environment()
        except Exception:          # pragma: no cover - probing is optional
            return None

    _named = named_environments()
    if target is not None and target not in _named:
        raise UnknownTarget(target, _named)

    # A NAMED target is validated WHOLE, here, before any scope is walked --
    # both that the name exists and that its record READS.  `read_environment`
    # answers absent / not-JSON / wrong-schema with one `None` so callers have
    # one thing to check; for a named target that single answer must not mean
    # "try the next scope", because the user said which machine and silence
    # hands them another (C3).
    #
    # Validated here rather than inside the loop because the loop may never
    # REACH the target scope: `record_scopes` puts the calculation's snapshot
    # first, and an already-prepped bundle returns there.  A check placed in
    # the loop would fire for a fresh bundle and stay silent for a prepped
    # one -- the same flag, two behaviours.
    _want = read_environment(_named[target]) if target is not None else None
    if target is not None and _want is None:
        raise UnknownTarget.unreadable(target, _named[target])

    # C1 -- SEVERAL MACHINES, NOBODY SAID WHICH.  Only when the question is
    # real: a calculation that already carries a snapshot has its answer (and
    # a contradicting --target is caught below), so this asks nothing of a
    # re-prep.
    #
    # "This machine" is ALWAYS a candidate -- with no local record, `probe`
    # produces one -- so the presence of any named record is what makes the
    # question ambiguous.  Requiring a local record here first would let the
    # commonest cluster setup (named targets, nothing probed locally) fall
    # through to a fresh probe of the machine the user is sitting at, which
    # is the exact failure this refusal exists to stop.
    if target is None and _named:
        _snapshot = (Path(bundle_dir) / FILENAME) if bundle_dir is not None else None
        if _snapshot is None or not _snapshot.is_file():
            raise AmbiguousTarget(list(_named) + ["(this machine)"])

    for label, path in record_scopes(bundle_dir, target):
        env = read_environment(path)
        if env is None:
            continue
        if label == "calculation" and target is not None:
            # A NAMED target that contradicts the snapshot is a question
            # nobody can answer for the user: refuse rather than silently
            # keeping the old one or silently replacing it.
            if _want.to_dict() != env.to_dict():
                raise UnknownTarget.conflict(target, bundle_dir)
        return env
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
    "LOCAL_TARGET",
    "record_scopes",
    "topology_field_types",
    "read_environment", "write_environment", "machine_for", "UnknownTarget",
    "AmbiguousTarget",
    "_parse_scontrol_node", "_parse_lscpu", "_parse_nvidia_smi_l",
    "_parse_gres",
]
