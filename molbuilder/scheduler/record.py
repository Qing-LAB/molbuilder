"""Target-machine detection -> a portable ``Environment`` record.

The first half of the benchmark workflow's pluggable seam
(docs/execution/job-system.md, § 5): **probes** that learn the
target's scheduler + hardware topology + site facts, and the versioned
JSON record they produce (``environment@3`` — registry row: job-contracts
§ 6.1; written by ``jobset probe --write`` ON the machine it describes, and
copied beside a calculation at its first prep) that every later stage and any
external tool reads.

Detection priority for topology: the **compute node**, not where
this runs.

  1. SLURM  -> ``scontrol show node`` (correct compute-node shape, askable
     from a login node);
  2. local  -> ``lscpu`` + ``nvidia-smi -L`` (valid only when run ON the
     target: a workstation or an interactive allocation);
  3. declared -> caller-supplied overrides / defaults.

The pure parsers (``_parse_*``) take text and are unit-tested without a
scheduler; the ``detect_*`` wrappers add the (guarded) subprocess calls.
Any command that is missing or fails degrades to ``None`` fields -- never
an exception.

This module is NOT shipped beside a job.  The files that travel are
`runwrap.MONITOR_COMPANIONS` -- the monitor and the framework readers it
reads a run through; this one imports `..persist`, `..config_dir`,
`.quantities` and `.fit` where it needs them.  The RECORD is JSON and reads anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..identity import checkpoint_words, migrate_as_typed

SCHEMA = "molbuilder/environment@3"

#: The record's filename, at every scope.
from ..runfiles import MACHINE_RECORD_FILE as FILENAME  # noqa: E402 -- the catalogue's name

# Normalized GPU-type tokens we recognize in an nvidia-smi name string.
_GPU_TYPES = ("a100", "a30", "h100", "h200", "v100", "a40", "l40", "l4",
              "p100", "t4", "rtx")


# --------------------------------------------------------------------- #
#  Data model                                                           #
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
    #: (user, 2026-08-26: *x64 vs arm are different, don't mix them*)
    #: ASU Sol offers an ``arm`` partition -- Grace-Hopper, aarch64 -- in the
    #: same menu as eight x86 ones, and nothing here could tell them apart:
    #: an x86 conda env does not activate usefully on aarch64 and an
    #: AVX-512 binary does not run there at all, so the failure is total
    #: rather than slow.  Recorded with ``env_arch`` -- the two numbers a
    #: wrong-architecture refusal would compare; nothing compares them yet.
    arch:             Optional[str] = None


@dataclass
class Site:
    """Scheduler-specific submission facts (empty on a workstation)."""
    partition: Optional[str] = None
    qos:       Optional[str] = None


@dataclass(frozen=True)
class Device:
    """One kind of accelerator the nodes of a domain offer.

    The **interpreted** form of a domain's ``gpu`` column -- see
    :func:`_read_devices`, the one reader.
    """
    type:     str
    per_node: Optional[int] = None


class _Unset:
    """The one instance below (``UNSET``) marks a policy column NO PROBE
    EVER ASKED -- distinct from ``None``, which means *asked; no cap
    stated*.  That distinction is the record's tri-state (`scheduler.md`
    R13): :meth:`Domain.to_row` drops ``UNSET`` but writes ``None`` as
    ``null``, so *asked, uncapped* survives the trip to disk.
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
    never *submit here*.  Which domain a run wants is the job's own
    statement (`execution/architecture.md` § 5.2).

    The columns below are the declared ones; anything else rides in
    ``extra`` (R10).  A reader owns only the keys it checks.
    """
    name:      str
    partition: str
    #: ``None`` when the probe could not read the account's QoS list: the
    #: pair is the partition alone, and a header written for it carries no
    #: QoS line (`probe.derive_domains`).
    qos:       Optional[str]
    max_time:  Optional[str] = None
    max_cores: Optional[int] = None
    max_mem_gb: Optional[float] = None
    #: What SLURM grants PER CORE when a job states no ``--mem`` -- the number
    #: that turns 64 cores into a 128 GB ask nobody made.  Tri-state like the policy
    #: ceilings below (R13): ``None`` means *asked; the partition does not
    #: say* -- never zero (R3) -- and ``UNSET`` that it was never asked.
    default_mem_per_core_gb: Any = UNSET
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
    #: at once (``MaxSubmitJobsPerUser``).  Same tri-state as the two above.
    #:
    #: It is a different KIND of ceiling from every other field here.  Those
    #: cap ONE job and are answerable from the job alone; this caps the SET,
    #: so whether an ask fits depends on what is already queued.  A bench
    #: sweep submits many jobs at once by construction.
    max_submit_jobs:   Any = UNSET
    gpu:       Optional[Dict[str, Any]] = None
    #: Every distinct machine this domain holds: ``[{cores, nodes, mem_gb,
    #: gpu}, ...]``.  Declared because a partition is a
    #: QUEUE and not a machine type -- Sol's ``htc`` is 48-, 64- and
    #: 128-core nodes under one name.
    #:
    #: Any single figure over them is an opinion, so the fit reads each
    #: machine (`scheduler.md` R3) and this is what a person reads to
    #: choose; ``max_cores`` and ``max_mem_gb`` are the largest machine's.
    #: Empty means the record does not say, never *no machines*.
    node_types: Optional[List[Dict[str, Any]]] = None
    #: Columns this reader does not check, kept verbatim.
    extra:     Dict[str, Any] = field(default_factory=dict)

    #: The keys :meth:`from_row` recognises; everything else goes to ``extra``.
    _KNOWN = ("name", "partition", "qos", "max_time",
              "max_cores", "max_mem_gb", "default_mem_per_core_gb",
              "max_cpus_per_job", "max_cpus_per_node", "max_submit_jobs",
              "gpu", "node_types")

    #: The tri-state policy columns (see their field note): ``None`` is a
    #: real answer here and lands as ``null``; only ``UNSET`` stays off
    #: the disk.  Every other ``_KNOWN`` column keeps the record style --
    #: ``None`` says nothing and is not written.
    _NULLABLE = ("qos", "max_cpus_per_job", "max_cpus_per_node",
                 "max_submit_jobs", "default_mem_per_core_gb")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Optional["Domain"]:
        """A mapping -> a Domain, or ``None`` when it is not one.

        The ONE parser, used by the probe and by the record reader alike.
        ``name``/``partition``/
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
        no cap stated* and must land as ``null``.
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
    def core_range(self) -> str:
        """The core range across this domain's machines -- ``"48-128"``, one
        number when they are one size -- else ``max_cores``, the largest
        machine's, for a record that lists none (`scheduler.md` R0).  ONE
        spelling, read by the queue table and the machines listing."""
        from .quantities import core_range
        said = core_range([t.get("cores") for t in (self.node_types or [])
                           if isinstance(t, dict)])
        return said or (str(self.max_cores) if self.max_cores else "-")

    @property
    def mem_range(self) -> str:
        """The memory range across this domain's machines --
        ``"501-1511.2 GB"`` -- else ``max_mem_gb``, the largest machine's
        (`scheduler.md` R0)."""
        from .quantities import mem_range
        said = mem_range([t.get("mem_gb") for t in (self.node_types or [])
                          if isinstance(t, dict)])
        if said:
            return said
        return f"{float(self.max_mem_gb):g} GB" if self.max_mem_gb else "-"

    @property
    def devices(self) -> Tuple[Device, ...]:
        """What this domain's ``gpu`` column says the nodes offer.

        Empty when it says nothing.  **Every reader of the GPU inventory goes
        through here** (:func:`_read_devices`).
        """
        return _read_devices(self.gpu)


def _read_devices(gpu: Any) -> Tuple[Device, ...]:
    """A domain's ``gpu`` column -> the devices it names.  **The one reader.**

    The probe writes the column as gres TYPE to per-node COUNT, ``{"a100": 4,
    "a100.20gb": 16}`` -- one entry per type ``sinfo`` reported.

    An unreadable count is ``None``, never a raise and never a zero -- a column
    we cannot read is not a domain with no devices (R3), and the fit must be
    able to tell those apart.  This interprets the column; `to_row` still
    returns what was written.
    """
    if not isinstance(gpu, dict) or not gpu:
        return ()
    return tuple(Device(type=str(name), per_node=_to_int(count))
                 for name, count in gpu.items())


@dataclass(frozen=True)
class Scheduler:
    """THE MACHINE'S SCHEDULER: its kind (`kinds.toml`) and where each of
    the kind's commands is on that machine (`configuration.md` § 5 M-2) --
    found there by `jobset probe`, or declared to it (``--set
    scheduler.submit=PATH`` for a site's own wrapper).  The kind's language
    is never copied here: it is the kind's, one for every machine of that
    kind (`scheduler.md` § 6a).

    A command not found is ``None``.  A kind with a queue names its submit
    command -- a record that cannot send is no record of a scheduler -- and a
    kind with none names no command.
    """
    kind:    str
    submit:  Optional[str] = None
    query:   Optional[str] = None
    cancel:  Optional[str] = None
    cluster: Optional[str] = None

    def __post_init__(self):
        from .kinds import ROLES, kind, names
        k = kind(self.kind)                          # a kind the file states
        named = [r for r in ROLES if getattr(self, r)]
        if k.queue and not self.submit:
            raise ValueError(
                f"the machine record's {self.kind} scheduler names no submit "
                f"command -- probe where `{k.commands['submit']}` is on PATH, "
                f"or declare it: --set scheduler.submit=PATH "
                f"(configuration.md M-2)")
        if not k.queue and named:
            queues = [n for n in names() if kind(n).queue]
            raise ValueError(
                f"the machine record's scheduler is {self.kind}, which has no "
                f"queue, and names commands ({', '.join(named)}) -- name the "
                f"kind they are: "
                + " or ".join(f"--scheduler {n}" for n in queues))

    def commands(self) -> Dict[str, Optional[str]]:
        """Each command role -> where it is on the machine (`kinds.ROLES`)."""
        from .kinds import ROLES
        return {r: getattr(self, r) for r in ROLES}

    def said(self) -> str:
        """The block in words, as a probe and `jobset machines` say it."""
        found = [f"{r} {w}" for r, w in self.commands().items() if w]
        return self.kind + (f" ({', '.join(found)})" if found else "")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"kind": self.kind, **self.commands()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scheduler":
        from .kinds import ROLES
        if not isinstance(d, Mapping) or not isinstance(d.get("kind"), str):
            raise ValueError("the machine record's scheduler states its kind "
                             "(configuration.md M-2)")
        return cls(kind=d["kind"], **{r: (str(d[r]) if d.get(r) else None)
                                      for r in ROLES})


# THE RECORD'S COMMANDS ARE THE KINDS' ROLES, one list: the block's fields
# after its kind are `kinds.ROLES`, in order, or nothing here imports.
from .kinds import ROLES as _ROLES  # noqa: E402
if tuple(f.name for f in fields(Scheduler))[1:] != _ROLES:
    raise ImportError(f"record.Scheduler's commands are not kinds.ROLES "
                      f"({', '.join(_ROLES)})")


@dataclass
class Environment:
    """The portable target description.  Produced by probes,
    consumed by adapters; neither knows the other's internals."""
    #: The scheduler: its kind, and where its commands are there.
    scheduler: Scheduler
    topology:  Topology = field(default_factory=Topology)
    site:      Site = field(default_factory=Site)
    #: Every (partition, qos) this account may actually submit to, with its
    #: wall.  Empty on a workstation, and empty on a cluster until `jobset
    #: probe` has run.
    domains:   List["Domain"] = field(default_factory=list)
    #: HOW A SHELL ON THIS MACHINE ENTERS THE ENVIRONMENT -- the preamble
    #: to run and the activation form to use, as that machine states them
    #: in its own ``molbuilder.json`` (``env_init``, the same name there),
    #: copied in by `jobset probe`.
    #:
    #: **It is a fact about the TARGET, so it travels on the target's
    #: record.**  A wrapper is generated on a workstation and executed on a
    #: cluster; the two activate differently (`module load mamba` +
    #: `source activate` on ASU Sol, a `conda.sh` hook on the workstation).
    #:
    #: ``{}`` means the `molbuilder.json`
    #: of the machine it was probed on declared no ``env_init``.  Absent is
    #: NOT "use the local machine's" -- that substitution is the bug -- so
    #: prep REFUSES any target whose record carries no activation, saying
    #: where to declare it (`jobset.machine.require_activation`).
    #:
    #: Shape: ``{"preamble": str, "activation": str}``, both optional.
    env_init: Dict[str, str] = field(default_factory=dict)
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
    #: ``[]`` means the probe could not enumerate (no conda on PATH).
    #: Empty is "unknown", not "none":
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
    #: and ``topology.arch``; both are recorded, and nothing compares them
    #: yet (W54 R22).  The failure that check would catch arrives disguised:
    #: `envs/builds.py` looks for ``x86_64-conda-linux-gnu-gcc`` by name, so on
    #: aarch64 it simply finds nothing and reports an unknown compiler
    #: version rather than the actual cause.
    env_arch:   Optional[str] = None
    #: WHETHER THE MANAGER'S ``run`` ENTERS AN ENVIRONMENT THERE -- the fourth
    #: fact that travels (the user, 2026-10-18: *"remember the measurement in
    #: the machine record so it stops repeating"*).  Shape: ``{"manager": the
    #: binary the probe found there, "run_works": bool}`` -- measured by the
    #: probe the way the one door measures it (`envs.builds.measure_manager_run`,
    #: `installation.md` M4), and read by that door before it tries, so a
    #: broken ``run`` is not tried again on every command.  ``{}`` means not
    #: measured: no manager answered, or its ``run`` failed some other way
    #: than the stub -- the door measures as before.
    manager_run: Dict[str, Any] = field(default_factory=dict)
    source:    Dict[str, str] = field(default_factory=dict)
    detected_at: Optional[str] = None
    #: Who wrote it: the probe -- `jobset probe`, or `envs init-config`
    #: through it.  ``None`` for a record that does not say -- never a
    #: guessed writer.
    tool:      Optional[str] = "jobset-probe@1"
    #: THE MACHINE A CALCULATION IS SET TO -- the ``--target`` of its first
    #: prep, ``this`` for the machine it was prepared on -- written by that
    #: prep into the calculation's copy and nowhere else; a probe's record
    #: names none (``None``).  A later prep's ``--target`` is checked against
    #: it (`configuration.md` M-3; user, 2026-10-02: *"when a machine is set
    #: for a job, it is set, no changing"*).
    machine:   Optional[str] = None

    def __post_init__(self):
        # THE SCHEDULER IS STATED -- a kind `kinds.toml` holds, and where its
        # commands are (`configuration.md` M-2).
        if not isinstance(self.scheduler, Scheduler):
            raise ValueError(
                f"the machine record's scheduler is {self.scheduler!r} -- it "
                f"states its kind and where its commands are "
                f"(configuration.md M-2)")

    @property
    def kind(self):
        """The scheduler's kind (`kinds.Kind`): what every step asks -- has
        it a queue, how is a job sent, what does its header say -- never
        the word compared (`scheduler.md` § 6a)."""
        from .kinds import kind
        return kind(self.scheduler.kind)

    # ----- JSON round-trip (the persisted contract) ----------------- #

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "detected_at": self.detected_at,
            "scheduler": self.scheduler.to_dict(),
            "topology": asdict(self.topology),
            "site": asdict(self.site),
            "domains": [d.to_row() for d in self.domains],
            # Absent, not empty, when the probe could read no config: a key
            # that is missing and a key that is {} are different claims to
            # anything testing for it.
            **({"env_init": dict(self.env_init)}
               if self.env_init else {}),
            **({"conda_envs": sorted(self.conda_envs)}
               if self.conda_envs else {}),
            **({"env_arch": self.env_arch} if self.env_arch else {}),
            **({"manager_run": dict(self.manager_run)}
               if self.manager_run else {}),
            "source": dict(self.source),
            "tool": self.tool,
            "machine": self.machine,
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
            # AS WRITTEN: a record states its scheduler -- a missing one is
            # a record that does not read, never a workstation; a writer it
            # does not name is not guessed.
            scheduler=Scheduler.from_dict(d.get("scheduler")),
            topology=topo, site=site, domains=domains,
            env_init={
                k: str(v) for k, v in
                (d.get("env_init") or {}).items()
                if k in ("preamble", "activation") and v is not None},
            conda_envs=[str(e) for e in (d.get("conda_envs") or [])],
            env_arch=(str(d["env_arch"]) if d.get("env_arch") else None),
            manager_run=({"manager": str(d["manager_run"]["manager"]),
                          "run_works": bool(d["manager_run"]["run_works"])}
                         if isinstance(d.get("manager_run"), dict)
                         and "manager" in d["manager_run"]
                         and "run_works" in d["manager_run"] else {}),
            source=dict(d.get("source") or {}),
            detected_at=d.get("detected_at"),
            tool=(str(d["tool"]) if d.get("tool") else None),
            machine=(str(d["machine"]) if d.get("machine") else None),
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
    """``gpu:a100:4`` -> ``(count, type)`` -- a Topology states ONE device
    kind, so this narrows what `quantities.parse_gres` reads in full.

    The token carries the type; a list of names cannot keep up with a
    site's hardware.

    `_gpu_type_from_name` serves the input it is RIGHT for:
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
        # A NODE THAT STATES NO GRES HAS NONE -- measured, so 0
        # (`configuration.md` M-2: a consumer tells absent from unknown).
        if n is None and kv["Gres"].strip().lower() in ("(null)", "none"):
            n = 0
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
        # type.
        hint = hints[f.name]
        out[f.name] = args[0] if args else (
            hint if isinstance(hint, type) else str)
    return out


def detect_scheduler(*, declared: Optional[str] = None,
                     commands: Optional[Mapping[str, str]] = None
                     ) -> Tuple[Scheduler, str]:
    """Return ``(scheduler, source)`` -- this machine's scheduler, its kind
    and where each of its commands is here (`scheduler.md` § 6a).

    The first kind of `kinds.toml` with a queue whose submit command is on
    PATH here, each of its commands where PATH finds it (source ``path``);
    none found, the kind with no queue.  ``declared`` names the kind instead
    (``--scheduler``, source ``flag``).  ``commands`` are the commands
    declared to the probe (``--set scheduler.<role>=PATH``), which win over
    PATH and are said (``+flag``).  A kind with a queue whose submit command
    is neither found nor declared is refused by :class:`Scheduler`."""
    from .kinds import ROLES, kind, names
    commands = dict(commands or {})
    every = [kind(n) for n in names()]
    if declared:
        k, source = kind(declared), "flag"
    else:
        k = next((c for c in every
                  if c.queue and shutil.which(c.commands["submit"])),
                 None) or next(c for c in every if not c.queue)
        source = "path+flag" if commands else "path"
    # EVERY COMMAND DECLARED IS KEPT -- one for a kind with no queue among
    # them, which `Scheduler` refuses, naming the kinds it could be.
    found = {role: commands.get(role) or (shutil.which(k.commands[role])
                                          if role in k.commands else None)
             for role in ROLES}
    return Scheduler(kind=k.name, **found), source


# --------------------------------------------------------------------- #
#  Discovery -- a reader per kind (`scheduler.md` § 6a)                  #
# --------------------------------------------------------------------- #


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


def _slurm_node(scheduler: Scheduler,
                partition: Optional[str]) -> Tuple[Topology, str]:
    """A compute node's shape, asked of ``scontrol`` -- where the record
    says it is (the ``cluster`` command) -- for a node ``sinfo`` lists."""
    node = _slurm_pick_node(partition) if scheduler.cluster else None
    out = _run([scheduler.cluster, "show", "node", node]) if node else None
    return ((_parse_scontrol_node(out), "scontrol") if out
            else (Topology(), "unknown"))


def _slurm_site(scheduler: Scheduler) -> Tuple[Site, str]:
    """The **default partition**, from ``sinfo`` (the one ``%P`` marks with
    ``*``, else the first listed)."""
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


def _slurm_in_a_job() -> bool:
    """Whether this runs inside one of SLURM's jobs -- the variable its kind
    names for the job's id is set (`kinds.toml`, ``job``)."""
    from .kinds import kind
    return bool(os.environ.get(kind("slurm").job["MB_JOB_ID"]))


def detect_topology(scheduler: Scheduler, *,
                    partition: Optional[str] = None,
                    overrides: Optional[dict] = None
                    ) -> Tuple[Topology, str]:
    """Resolve the compute-node topology.  Returns
    ``(Topology, source)`` where source is ``scontrol`` | ``lscpu`` |
    ``flag`` | ``unknown``.  Overrides (declared flags) win field-by-field
    and mark the source ``flag`` when they supplied anything.

    A kind with a queue asks its own discovery (:data:`_DISCOVERY`) for the
    compute node's shape."""
    from .kinds import kind
    topo, source = Topology(), "unknown"
    queue = kind(scheduler.kind).queue
    if queue:
        topo, source = _DISCOVERY[scheduler.kind]["node"](scheduler,
                                                          partition)

    # Local probe ONLY when we are physically ON the target: a machine with
    # no queue, or inside one of its scheduler's jobs.  On a LOGIN node
    # ``lscpu`` would describe the *login* node, not the compute node -- so
    # if the scheduler could not be asked there, leave topology unknown and
    # let declared flags fill it, rather than report the wrong machine.
    on_node = not queue or _DISCOVERY[scheduler.kind]["in_a_job"]()
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


def detect_site(scheduler: Scheduler) -> Tuple[Site, str]:
    """The scheduler's **default partition**, asked of its kind's discovery
    (:data:`_DISCOVERY`); empty on a machine with no queue.

    ``qos`` is left ``None`` **here** because the node probe cannot answer
    it -- not because nothing can: the cluster probe (`jobset probe`) asks
    which QoS your account may use and writes each into its ``domains``
    row; ``site.qos`` is written by nothing."""
    from .kinds import kind
    if not kind(scheduler.kind).queue:
        return Site(), "n/a"
    return _DISCOVERY[scheduler.kind]["site"](scheduler)


def resolve_environment(*, overrides: Optional[dict] = None,
                        now_iso: Optional[str] = None,
                        scheduler_override: Optional[str] = None,
                        commands: Optional[Mapping[str, str]] = None
                        ) -> Environment:
    """Run the probes in order and assemble the Environment.

    ``overrides`` is a flat dict of declared topology values (e.g.
    ``{"cores_per_socket": 24, "gpus_per_node": 4}``) that win over
    detection.  ``scheduler_override`` names the scheduler's kind, and
    ``commands`` the scheduler's commands declared (:func:`detect_scheduler`).
    ``now_iso`` stamps ``detected_at`` (passed in so the module stays free of
    wall-clock calls for deterministic tests)."""
    scheduler, sch_src = detect_scheduler(declared=scheduler_override,
                                          commands=commands)

    site, site_src = detect_site(scheduler)
    topo, topo_src = detect_topology(
        scheduler, partition=site.partition, overrides=overrides)

    return Environment(
        scheduler=scheduler, topology=topo, site=site,
        source={"scheduler": sch_src, "topology": topo_src, "site": site_src},
        detected_at=now_iso,
    )


def probe_queues(env: Environment, user: str) -> Tuple[List[str], Optional[str]]:
    """Read this machine's queues into ``env.domains`` -- what ``sinfo``,
    ``sacctmgr`` and ``scontrol`` say -- and return ``(notes, summary)``:
    what a person should read before the record is written, and the one
    line that says what was read (``None`` when no ``sinfo`` answered).

    **THE ONE QUEUE PROBE** (`configuration.md` M-3: a machine is measured
    by `jobset probe`, or by `envs init-config` seeding its own record
    "through the same prober").

    Not a refusal on a machine with no scheduler -- M-2: it records its
    capability in the same shape a cluster does, with no domains.

    DISCOVERY, A READER PER KIND, keyed by the kind's name (`scheduler.md`
    § 6a): each parses its scheduler's own tools, whose output no format
    states (:data:`_DISCOVERY`).
    """
    if not env.kind.queue:
        return ([f"{env.kind.what} -- so no queues: this record is the "
                 f"machine's own (topology only)."], None)
    return _DISCOVERY[env.scheduler.kind]["queues"](env, user)


def _probe_slurm_queues(env: Environment, user: str
                        ) -> Tuple[List[str], Optional[str]]:
    """SLURM's queues, from ``sinfo``, ``sacctmgr`` and ``scontrol``."""
    from .probe import (derive_domains, parse_allowed_qos, parse_qos,
                        parse_scontrol_partitions, parse_sinfo)
    notes: List[str] = []
    # ``%m`` (memory per node, MB) is the ceiling `Domain.max_mem_gb`
    # (`execution/scheduler.md` § 2).
    sinfo_txt = _run(["sinfo", "-h", "-o", "%P|%30l|%D|%40G|%c|%m"])
    if sinfo_txt is None:
        notes.append("no sinfo reachable here, so no queues were "
                     "probed -- run the probe where `sinfo` answers "
                     "(a login node).")
        return notes, None
    parts = parse_sinfo(sinfo_txt)
    # MaxTRES (R13): a field you did not request is not an absence the
    # record may report as silence.
    #
    # MaxSubmitJobsPerUser (R14) -- the SAME rule, the next column over.
    #
    # APPENDED, never inserted: `sacctmgr -nP` prints no header, so these rows
    # are positional and a column in the middle would shift every reader
    # after it.
    #
    # AND ASKED SEPARATELY, with a fall-back.  `sacctmgr` rejects the WHOLE
    # query on one unknown format field, and `_run` reports any failure as
    # None -- so on a cluster whose Slurm spells this column differently, a
    # single combined query would lose MaxWall and MaxTRES as well, and every
    # domain would silently go back to having no QoS facts at all.  One new
    # question cannot be allowed to cost the answers we already had.
    qos_txt = _run(["sacctmgr", "-nP", "show", "qos",
                    "format=Name,MaxWall,MaxTRES,Flags,MaxSubmitJobsPerUser"])
    no_submit_cap = qos_txt is None
    if no_submit_cap:
        qos_txt = _run(["sacctmgr", "-nP", "show", "qos",
                        "format=Name,MaxWall,MaxTRES,Flags"])
    qos = parse_qos(qos_txt or "")
    allowed = parse_allowed_qos(
        _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
              "format=Partition,QOS"]) or "")
    # THE SECOND COMMAND, and it needs one: `sinfo` has no format code for
    # DefMemPerCPU, and that is the number SLURM grants per core when a job
    # states no --mem.  Absent scontrol, the rows simply carry no per-core default, which
    # reads as "this machine does not say" rather than as zero (R3).
    policy = parse_scontrol_partitions(
        (_run([env.scheduler.cluster, "show", "partition"])
         if env.scheduler.cluster else None) or "")
    for p in parts:
        pol = policy.get(p.name)
        if pol is not None:
            p.def_mem_per_cpu_mb = pol.def_mem_per_cpu_mb
            p.max_cpus_per_node = pol.max_cpus_per_node
            # the record writes null-when-asked vs absent-when-not
            p.policy_queried = True
    rows, notes = derive_domains(parts, qos, allowed)
    # EVERY NOTE AFTER `derive_domains`, which REASSIGNS `notes`.
    if no_submit_cap and qos_txt is not None:
        notes.append(
            "this Slurm did not accept MaxSubmitJobsPerUser in the QoS format "
            "list, so no domain records a submitted-job cap -- the wall and "
            "core caps were read as usual.  A sweep will find that limit at "
            "sbatch time instead of before it (scheduler.md R14).")
    if not policy:
        notes.append("scontrol was not reachable, so no per-core memory "
                     "default was measured -- a job that states no --mem "
                     "will get whatever SLURM decides and molbuilder cannot "
                     "show you the number in advance.")
    env.domains = rows          # already Domain objects (probe.py)
    env.source["domains"] = "sinfo+sacctmgr"
    return notes, (f"Probed (user={user}): {len(parts)} partitions; allowed "
                   f"QoS: {', '.join(sorted(allowed)) or '(unknown)'}")


#: DISCOVERY, A READER PER KIND (`scheduler.md` § 6a): each kind with a queue
#: -> how its own tools are read for the machine -- a compute node's shape,
#: the default partition, whether this runs inside one of its jobs, and its
#: queues.  Its tools' output, which no format states.  Adding a kind adds
#: its row here and its block in `kinds.toml`.
_DISCOVERY = {"slurm": {"node": _slurm_node, "site": _slurm_site,
                        "in_a_job": _slurm_in_a_job,
                        "queues": _probe_slurm_queues}}


# --------------------------------------------------------------------- #
#  The door (N1) -- configuration.md § 5, M-4                            #
# --------------------------------------------------------------------- #


def machine_scope_path() -> Path:
    """Where the MACHINE-scope record lives — ``jobset probe``'s target.

    ``~/.config/molbuilder/environment.json``, honouring ``XDG_CONFIG_HOME``.
    The convention is :func:`molbuilder.config_dir.config_dir`'s.

    **Per-user only, with no cwd step.**  A
    calculation is very often the working directory, so a cwd step here would
    make the machine scope and the calculation scope the same file whenever
    you happened to run from inside a bundle, and M-3's precedence would then
    compare a record against itself.
    """
    from ..config_dir import config_dir
    return config_dir() / FILENAME


def calculation_record(bundle_dir) -> Path:
    """The record a calculation carries, at its root.

    A message naming a file the code does not read is worse than no message,
    so the spelling has one home.
    """
    return Path(bundle_dir) / FILENAME


def calculation_machine(bundle_dir) -> Optional[str]:
    """The machine a calculation is set to -- the name its copy of its
    machine's record carries (`configuration.md` M-3: set at its first prep,
    and it does not change), :data:`LOCAL_TARGET` for the machine it was
    prepared on -- or ``None`` before its first prep, or for a copy that does
    not read or names none.  The one reader
    of that name: Task setup's folder answer, the launch lines molbuilder
    prints and a re-probe's remedy ask it."""
    env = read_environment(calculation_record(bundle_dir))
    return getattr(env, "machine", None) if env is not None else None


def environments_dir() -> Path:
    """Where NAMED target records live — ``<config dir>/environments/``.

    One record describes one machine, and the machine you are preparing for is
    not always the machine you are on.

    **From `config_dir`, not from `machine_scope_path().parent`** (A11).  A file's resolver answers *"where is
    this file"*; climbing off it to reach the directory is the same two-spellings
    drift as re-typing a path, and it means moving `environment.json` silently
    moves this directory too.
    """
    from ..config_dir import config_dir
    return config_dir() / "environments"


def named_environment_path(name: str) -> Path:
    """The record for the target called `name` — the ONE join for that file.

    A reader and a writer that disagree here look at different
    files and both report success.
    """
    return environments_dir() / f"{name}.json"


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
    from .fit import domain_ceiling_s            # one parser for max_time

    def _describe(env, name, kind, path):
        if env is None:
            # Absent and unreadable are one answer from `read_environment`,
            # so the message covers both -- and the REMEDY differs by scope:
            # this machine's record is written by a bare `probe --write`,
            # while a named target takes `--name` and must be probed on the
            # machine it describes.
            fix = (f"`{probe_command(None)}` here" if kind == "local"
                   else f"`{probe_command(name)}` {probe_steps(name)}")
            why = why_unread(path)
            return {"name": name, "kind": kind, "path": str(path),
                    "readable": False, "detected_at": "", "domains": [],
                    "mem_total_gb": None, "gpus_per_node": None,
                    "gpu_type": None,
                    "summary": ((f"cannot be read -- {why} -- write it "
                                 f"again with " if why
                                 else "no record here yet -- write one with ")
                                + fix)}
        # CORES: the RANGE the machines span, when the record lists them.
        #
        # `sockets x cores_per_socket` describes ONE node -- whichever
        # `sinfo` printed first (`_slurm_pick_node`) -- and a partition is a
        # queue, not a machine type (`scheduler.md` R0).
        #
        # A MACHINE WITH QUEUES IS SAID BY ITS QUEUES' MACHINES -- the core
        # and memory ranges across them, and the cards they carry; a queue
        # that lists no machines by its largest machine's figures (R3).  Its
        # `topology` is the one node the probe sampled, which describes a
        # machine with no queue and never the cluster (`gpu.md` § 1.3).
        from .quantities import core_range as _core_range
        from .quantities import mem_range as _mem_range
        bits = [env.scheduler.kind]
        cores_of, mem_of = [], []
        for d in env.domains or []:
            listed = [t for t in (d.node_types or []) if isinstance(t, dict)]
            cores_of += ([t.get("cores") for t in listed] if listed
                         else [d.max_cores])
            mem_of += ([t.get("mem_gb") for t in listed] if listed
                       else [d.max_mem_gb])
        if env.domains:
            spread = _core_range(cores_of)
            mem = _mem_range(mem_of)
            cards = sorted({c for d in env.domains for c in (d.gpu or {})})
            bits += [f"{spread} cores"] if spread else []
            bits += [mem] if mem else []
            bits += [", ".join(cards)] if cards else []
            bits.append(f"{len(env.domains)} domain(s)")
        else:
            cores = getattr(env.topology, "cores_per_socket", None)
            sockets = getattr(env.topology, "sockets", None)
            if cores and sockets:
                bits.append(f"{cores * sockets} cores")
            _mem = getattr(env.topology, "mem_total_gb", None)
            if _mem:
                bits.append(f"{float(_mem):g} GB")
            if getattr(env.topology, "gpus_per_node", None):
                bits.append(f"{env.topology.gpus_per_node}\u00d7 "
                            f"{env.topology.gpu_type or 'GPU'}")

        # WHAT WAS KEPT BUT NOT UNDERSTOOD (2026-09-06, user).  An
        # unrecognised column in a routing row is KEPT -- R10, "silently
        # dropping a column is the bug" -- and a retired key an operator
        # still writes, or a column of their own, must survive.  But that
        # makes a MISSPELLING indistinguishable from a deliberate extra:
        # `max_tme` lands in `extra` exactly like `node_type` does, and
        # `max_time` then reads as unstated, so a job asking more wall than
        # the queue allows is never said to be over it (R4).  `scheduler.md` § 4 predicted
        # it -- "a value that changes where a job lands is a field, or it is
        # a bug waiting for someone to misspell it".
        #
        # Neither refusing nor dropping is available, so the answer is to
        # SAY SO.  Nothing is rejected and nothing is lost; a typo simply
        # stops being invisible.  Here rather than in either surface, because
        # this list has two readers and they must not be able to disagree.
        uninterpreted = sorted({k for d in env.domains for k in d.extra})
        if uninterpreted:
            # `??` LEADS IT (user, 2026-09-06): this sits mid-way through a
            # dot-separated line of ordinary facts, and a reader scanning for
            # trouble needs something that is not a word.  The marker is the
            # question being asked -- did you mean to write this?
            bits.append("?? not understood: " + ", ".join(uninterpreted))
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
                # THE PER-CORE DEFAULT AS THE RECORD HOLDS IT (R13): a
                # number, or null when asked and unstated -- left out only
                # when the probe never asked.
                "domains": [{"name": d.name,
                             "partition": d.partition,
                             "qos": d.qos,
                             "max_time": d.max_time,
                             "max_time_s": domain_ceiling_s(d),
                             "cores": d.core_range,
                             "memory": d.mem_range,
                             "devices": ", ".join(f"{x.type} x{x.per_node}"
                                                  for x in d.devices),
                             **({} if d.default_mem_per_core_gb is UNSET
                                else {"default_mem_per_core_gb":
                                      d.default_mem_per_core_gb}),
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
    `runtime_config._SECTIONS` already uses for the config's sections.

    ``target`` names a machine explicitly and is resolved by the CALLER
    (:func:`machine_for`), because an unknown name is an error rather than a
    scope that failed to match.
    """
    scopes: List[Tuple[str, Path]] = []
    if bundle_dir is not None:
        scopes.append(("calculation", calculation_record(bundle_dir)))
    if target is not None:
        known = named_environments()
        if target in known:
            scopes.append((f"target:{target}", known[target]))
    scopes.append(("machine", machine_scope_path()))
    return scopes


def read_environment(path) -> Optional["Environment"]:
    """The record at ONE path, or ``None``.

    Takes the **file**, not the directory holding it: a named record's name
    IS its filename, so the join lives in :func:`record_scopes` where the
    paths are built.

    ``None`` covers every way there is no usable answer here — absent,
    unreadable, not JSON, wrong schema, malformed.  **Malformed is ``None``,
    not an exception**: a hand-edited file earns a fall-through to the next
    scope, and the caller has one thing to check instead of four.

    The narrowness of the second ``except`` is deliberate: a bad file earns
    tolerance; a bug does not.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Environment.from_dict(raw)
    except (ValueError, TypeError, KeyError):
        return None


def why_unread(path) -> Optional[str]:
    """Why the record at ``path`` does not read, in words -- ``None`` when
    it reads, or when nothing is there.

    :func:`read_environment` answers absent and unreadable alike with
    ``None``, so a caller has one thing to check; a REFUSAL says which, and
    names an older molbuilder's record as that -- written before this one's
    schema (`configuration.md` § 5) -- so its way out can be named.
    """
    return _unread(path)[0]


def _unread(path) -> Tuple[Optional[str], bool]:
    """``(why it does not read, whether an older molbuilder wrote it)`` --
    :func:`why_unread`'s answer, and whether its way out is `jobset
    migrate`, read once."""
    from ..persist import schema_major, schema_name
    p = Path(path)
    if not p.exists():
        return None, False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        return "it cannot be read here (permissions)", False
    except ValueError:
        return "it is not JSON", False
    got = str(raw.get("schema", "")) if isinstance(raw, dict) else ""
    if older_record(raw):
        return (f"an older molbuilder wrote it ({got}; this one reads "
                f"{SCHEMA})"), True
    try:
        Environment.from_dict(raw)
    except (ValueError, TypeError, KeyError) as exc:
        if schema_name(got) != schema_name(SCHEMA) or (
                schema_major(got) != schema_major(SCHEMA)):
            return (f"it is not a machine record this molbuilder reads "
                    f"({got})"), False
        return str(exc), False
    return None, False


def older_record(raw) -> bool:
    """Whether ``raw`` (a record's JSON) is a machine record an older
    molbuilder wrote: its schema's name, at a lower major."""
    from ..persist import schema_major, schema_name
    got = str(raw.get("schema", "")) if isinstance(raw, dict) else ""
    try:
        return (schema_name(got) == schema_name(SCHEMA)
                and int(schema_major(got)) < int(schema_major(SCHEMA)))
    except (ValueError, TypeError):
        return False


def write_environment(env: "Environment", path) -> Path:
    """The record to ONE path, atomically.  Returns the path written.

    Through ``persist.write_json``, the shared writer: 2-space indent,
    trailing newline, and the unique-temp + ``os.replace`` that keeps two
    writers from colliding.

    Takes the **file** for the same reason the reader does.
    """
    from ..persist import write_json
    return write_json(Path(path), env.to_dict())


class UnknownTarget(Exception):
    """``--target NAME`` cannot be honoured — unknown, or contradicted by the
    record this calculation was already prepared against."""

    def __init__(self, name: str, known):
        self.name, self.known = name, sorted(known)
        listed = ", ".join(self.known) or "(none)"
        super().__init__(
            f"no machine record named {name!r}.  Known targets: {listed}.  "
            f"Write one with `{probe_command(name)}` {probe_steps(name)}.")

    @classmethod
    def unreadable(cls, name: str, path) -> "UnknownTarget":
        """The record exists under that name but does not read.

        Absent, unreadable, not JSON, wrong schema -- `read_environment`
        answers all of them with ``None`` so a caller has one thing to
        check.  For a NAMED target that single answer must not mean "try
        the next scope": the user named this machine, so silently preparing
        for a different one is the mistake the flag exists to catch.
        """
        exc = cls.__new__(cls)
        exc.name, exc.known = name, []
        Exception.__init__(exc, (
            f"--target {name!r} names a record that cannot be read: {path} "
            f"-- {why_unread(path) or 'it is not there'}.\n"
            f"  Re-write it with `{probe_command(name)}` "
            f"{probe_steps(name)}."))
        return exc

    @classmethod
    def unreadable_copy(cls, bundle_dir) -> "UnknownTarget":
        """This calculation's own record is there and does not read.

        Skipped, it handed the calculation the next scope's record -- another
        machine's, silently -- which breaks what the copy is for
        (`configuration.md` § 5 M-3): two stages of one calculation cannot
        disagree about their own target.
        """
        exc = cls.__new__(cls)
        exc.name, exc.known = None, []
        path = calculation_record(bundle_dir)
        why, older = _unread(path)
        if older:
            # AN OLDER MOLBUILDER'S COPY is rewritten by the command that
            # rewrites what an older molbuilder wrote (`jobset migrate`).
            way = (f"`{migrate_as_typed(Path(bundle_dir).resolve())}` "
                   f"rewrites it, the old file kept beside the new.")
        else:
            way = (f"Fix it by hand -- or go back to the state saved before "
                   f"the calculation's first prep "
                   f"({checkpoint_words(bundle_dir)}) and prep anew.")
        Exception.__init__(exc, (
            f"this calculation's record of its machine cannot be read: "
            f"{path} -- {why}.\n  {way}"))
        return exc

    @classmethod
    def conflict(cls, name: str, set_to: str,
                 bundle_dir) -> "UnknownTarget":
        """This calculation is set to ANOTHER machine -- the one its first
        prep named, which does not change (`configuration.md` M-3).

        Silently keeping the snapshot would make ``--target`` a no-op on every
        folder after the first prep -- the flag would appear to work and
        change nothing.  Silently re-snapshotting would let stage 2 of a
        ladder resolve against a different machine than stage 1, which is the
        exact disagreement the once-per-bundle rule exists to prevent.  So
        neither: say which machine it is set to, and the way to another --
        a new prep from a saved state (user, 2026-10-02: *"we have
        persistency to roll back and start for a new prep if we need to"*).
        """
        exc = cls.__new__(cls)
        exc.name, exc.known = name, []
        Exception.__init__(exc, (
            f"this calculation is set to {set_to!r}, the machine of its first "
            f"prep, and that does not change -- --target {name} names "
            f"another.\n"
            f"  To go on with it: --target {set_to}.\n"
            f"  To prepare it for {name}: go back to the state saved before "
            f"its first prep ({checkpoint_words(bundle_dir)}) and prep again "
            f"with --target {name}."))
        return exc


class AmbiguousTarget(Exception):
    """Several machines could be meant and nobody said which.

    The cost is asymmetric, which is the whole argument: being asked costs
    one flag; being given the wrong machine costs a queue wait, an
    allocation, and a set of numbers that look plausible.  That is the
    failure ``--target`` was introduced for -- a benchmark prepared for a
    cluster, silently measured against the workstation it was prepared on.

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
        listed += f"\n    --target {LOCAL_TARGET}   # this machine"
        # "NOTHING NEEDS PROBING" ONLY WHEN IT IS SO: with no record of its
        # own, `--target this` is refused next, and the probe it needs is the
        # one command that helps.
        here = read_environment(machine_scope_path()) is not None
        # ...and a NAMED record that does not read needs probing again: it
        # is listed, so it is said.
        dead = sorted(n for n, path in named_environments().items()
                      if n in self.choices and read_environment(path) is None)
        tail = ("  These records already exist -- nothing needs probing."
                if here and not dead else "")
        if not here:
            tail = ("  This machine has no record yet, so `--target "
                    f"{LOCAL_TARGET}` needs one first:\n    "
                    + probe_command(None))
        for n in dead:
            tail += (f"\n  {n}'s record does not read: re-write it with "
                     f"`{probe_command(n)}` {probe_steps(n)}.")
        super().__init__(
            "several machines could be meant and none was named.  Say which "
            "this calculation is for:\n" + listed +
            "\n    (there is no default; name one of the above)\n"
            + tail.lstrip("\n"))


#: THE TYPEABLE NAME FOR THIS MACHINE.  ``known_machines`` displays
#: ``(this machine)``, which is a label and not something a person can type
#: at ``--target``.  C1 protects against SILENCE; naming this machine is not
#: silence.
LOCAL_TARGET = "this"


def probe_command(name: Optional[str] = None) -> str:
    """The command that writes the record for machine ``name`` -- THIS
    machine's with no name (:data:`LOCAL_TARGET` is reserved, and the probe
    refuses it), a named target's with ``--name``, run on that machine.  ONE
    spelling for every refusal that asks for a record (`configuration.md`
    M-4)."""
    if name in (None, LOCAL_TARGET):
        return "molbuilder jobset probe --write"
    return f"molbuilder jobset probe --write --name {name}"


def probe_steps(name: Optional[str] = None) -> str:
    """Where :func:`probe_command` runs, and what follows it, in words:
    ``here`` for this machine; for a named target, on that machine -- and
    then the record it writes copied into this machine's ``environments/``,
    without which the remedy leaves the same refusal."""
    if name in (None, LOCAL_TARGET):
        return "here"
    return (f"on that machine, then copy the {name}.json it writes into "
            f"{environments_dir()}/ here")


def probe_line(name: Optional[str] = None) -> str:
    """:func:`probe_command` as a line of its own, where it is run -- and,
    for a named target, the copy that follows -- said after ``#``: the
    shell reads no note (`job-system.md` § 5.3)."""
    return probe_command(name) + (
        "" if name in (None, LOCAL_TARGET) else f"   # {probe_steps(name)}")


def record_and_renewal(bundle_dir=None, target: Optional[str] = None
                       ) -> Tuple[str, str]:
    """The record a calculation reads, in words, and how it comes to read a
    re-probed one -- ``(which, renew)``, the ONE spelling for every refusal
    whose fix is a re-probe (`configuration.md` M-4).

    The calculation's own snapshot answers first, whatever the flag says
    (M-3): a calculation that holds one is set to that record, and a
    re-probed one reaches it only through a new prep, from a state saved
    before its first -- TWO steps, both said: one alone re-reads the
    same record.  One that holds none reads the target's record, and its
    probe line is the whole remedy."""
    snap = (calculation_record(Path(bundle_dir))
            if bundle_dir is not None else None)
    if snap is not None and snap.is_file():
        # WHICH machine: the one the copy names (M-3), else the one asked for.
        name = calculation_machine(bundle_dir) or target
        probe = (f"`{probe_command(None)}` here" if name in (None, LOCAL_TARGET)
                 else f"`{probe_command(name)}` {probe_steps(name)}")
        return ("this calculation's environment.json, snapshotted at its "
                "first prep",
                f"re-probe it -- {probe} -- then go back to the state saved "
                f"before its first prep ({checkpoint_words(bundle_dir)}) "
                f"and prep anew: a calculation is set to "
                f"the record it was first prepared with (configuration.md "
                f"M-3)")
    return ("this machine's" if target in (None, LOCAL_TARGET)
            else f"{target}'s",
            f"re-probe it --\n    {probe_line(target)}\n ")


def machine_for(bundle_dir=None, *, target: Optional[str] = None
                ) -> Optional["Environment"]:
    """**The precedence, entire** — the one function a caller asks.

    Walks :func:`record_scopes` and returns the first record that reads.  The
    first one found is the whole answer: **there is no field-level merge**, and
    that is the rule rather than an omission.  Two partial records blended at
    read time would describe a machine that exists in no file, and it would
    defeat the standing guarantee that two stages of one calculation cannot
    disagree about their own target.  A calculation is set to the machine of
    its first prep; another record reaches it only through a new prep, from a
    state saved before that one (M-3).

    **It never probes** (`configuration.md` M-4): when no scope answers it
    answers ``None``, and the caller refuses with :func:`probe_command`.
    """
    # A named target is validated FIRST, before any scope is consulted, so a
    # typo is never answered by whatever was snapshotted.
    # `LOCAL_TARGET` NAMES WHICH MACHINE; IT IS NOT A DIFFERENT WAY OF FINDING
    # ONE.  So it reads the same scopes saying nothing reads -- the
    # calculation's snapshot first (`workflow.md` § 5 step 1: *"read the
    # record -- the calculation's, else this machine's"*) -- and only the
    # NAMED-target machinery below is skipped, because there is no
    # `environments/this.json` to validate against: the name is reserved and
    # `jobset probe --name this` is refused.  C1 is skipped for the reason it
    # always was -- the question it guards, WHICH machine is meant, has just
    # been answered.
    #
    # A record stored beside the calculation IS a stored record, which is what
    # `running-a-job.md` § 3.1 asks for: *"read from a record, and nowhere
    # else"*, carving out no exception for the local box (user, 2026-09-02:
    # *"even for the current machine, the environment.json must be present"*,
    # and again: *"all environments have to be explicitly probed and stored.
    # no environment json, error"*).
    _by_name = target is not None and target != LOCAL_TARGET

    _named = named_environments()
    if _by_name and target not in _named:
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
    # first, and an already-prepared bundle returns there.  A check placed in
    # the loop would fire for a fresh bundle and stay silent for a prepared
    # one -- the same flag, two behaviours.
    _want = read_environment(_named[target]) if _by_name else None
    if _by_name and _want is None:
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
    # through to the machine the user is sitting at, which is the exact failure this refusal
    # exists to stop.
    if target is None and _named:
        _snapshot = calculation_record(bundle_dir) if bundle_dir is not None else None
        if _snapshot is None or not _snapshot.is_file():
            raise AmbiguousTarget(list(_named) + ["(this machine)"])

    for label, path in record_scopes(bundle_dir, target):
        env = read_environment(path)
        if env is None:
            # THE CALCULATION'S OWN COPY, there and unreadable, is an error,
            # not a miss: the next scope is another machine's answer.
            if label == "calculation" and Path(path).is_file():
                raise UnknownTarget.unreadable_copy(bundle_dir)
            continue
        if label == "calculation" and target is not None \
                and env.machine is not None and env.machine != target:
            # SET AT ITS FIRST PREP, AND IT DOES NOT CHANGE (M-3; user,
            # 2026-10-02).  The copy names its machine, so the check is a
            # name: `this` is checked like any other, and a re-probe of the
            # calculation's own machine is no conflict.
            raise UnknownTarget.conflict(target, env.machine, bundle_dir)
        return env
    return None


__all__ = [
    "SCHEMA", "FILENAME", "calculation_record", "calculation_machine",
    "Topology", "Site", "Domain", "Scheduler", "Environment",
    "detect_scheduler", "detect_topology", "detect_site",
    "resolve_environment",
    "machine_scope_path", "environments_dir", "named_environments",
    "LOCAL_TARGET", "probe_command", "probe_line", "probe_steps",
    "record_and_renewal",
    "record_scopes",
    "topology_field_types",
    "read_environment", "write_environment", "why_unread", "older_record",
    "machine_for", "UnknownTarget",
    "AmbiguousTarget",
]
