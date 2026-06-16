"""Workstation MPI/OMP/MPS advisor for ``molbuilder-siesta-gpu``.

Surfaces a single subcommand --

    molbuilder envs advise siesta-gpu [--n-atoms N] [--n-orbitals M]

-- that probes the host (lscpu / nvidia-smi / mps-control) and prints a
recommendation table.  This is the *user-facing* counterpart to the
runtime defaults policy living in
:func:`molbuilder.runwrap._gpu_runtime_defaults_block`: the wrapper
auto-picks one value at SCF time, while ``advise`` shows three sensible
presets side-by-side so the user can pick consciously.

Why three presets and not one?  ELPA-GPU performance is bimodal on a
single-GPU workstation:

* The published throughput optimum is 4 MPI ranks/GPU + small OMP
  (ELPA 2024.05 release notes; matched in BSC MareNostrum5 SIESTA
  ACC partition report).
* But if the eigenproblem doesn't fit in 5-6 GB VRAM/rank, you have
  to drop to 2 or 1 rank and lean on OMP for the missing parallelism.
* And if MPS isn't installed on the host, >1 rank serialises through
  the CUDA driver context anyway -- 1 rank + max OMP is best.

The advisor presents all three so the user understands what trade-off
they're making and can map their wall-clock observation to a config.

Probes are read-only and best-effort: failures degrade to "unknown"
fields rather than raising, so the output is always something useful
(e.g. "MPS unavailable, fallback preset recommended").
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


_PROBE_TIMEOUT_S = 4.0


# --------------------------------------------------------------------- #
#  Probe                                                                 #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class HostProbe:
    """Host hardware snapshot used to derive recommendations.

    Fields cluster into three groups:

    * CPU topology: ``phys_cores``, ``sockets``, ``cores_per_socket``
      from ``lscpu -p=Core,Socket`` (HT-aware physical-core count).
    * GPU snapshot: ``gpu_name``, ``gpu_vram_mb``, ``gpu_compute_cap``,
      ``gpu_numa`` from ``nvidia-smi --query-gpu`` and ``topo -m``.
    * Runtime affordances: ``mps_available`` checks
      ``nvidia-cuda-mps-control`` on PATH; ``numactl_available`` checks
      ``numactl`` on PATH (needed to NUMA-pin mpirun on multi-socket
      hosts).
    """

    phys_cores: int
    sockets: int
    cores_per_socket: int
    gpu_name: Optional[str]
    gpu_vram_mb: Optional[int]
    gpu_compute_cap: Optional[str]
    gpu_numa: Optional[int]
    mps_available: bool
    numactl_available: bool

    @property
    def can_numa_pin(self) -> bool:
        """True when the wrapper would actually wrap mpirun in
        ``numactl --cpunodebind=$_gpu_numa --membind=$_gpu_numa``.

        Same three conditions the bash side checks (kept in sync to
        keep the advisor's preset budgets aligned with what the
        wrapper will actually do at runtime).
        """
        return (
            self.sockets >= 2
            and self.gpu_numa is not None
            and self.numactl_available
        )

    def describe(self) -> str:
        lines = [
            f"host:    {self.phys_cores} phys cores "
            f"× {self.sockets} socket{'s' if self.sockets != 1 else ''} "
            f"({self.cores_per_socket} cores/socket)",
        ]
        if self.gpu_name:
            vram = (f"{self.gpu_vram_mb / 1024:.1f} GB"
                    if self.gpu_vram_mb else "?")
            cc = self.gpu_compute_cap or "?"
            numa = (f"NUMA={self.gpu_numa}"
                    if self.gpu_numa is not None else "NUMA=unknown")
            lines.append(
                f"gpu:     {self.gpu_name} ({vram}, sm_{cc}), {numa}"
            )
        else:
            lines.append("gpu:     (none detected via nvidia-smi)")
        lines.append(
            f"mps:     {'available' if self.mps_available else 'NOT installed'}"
        )
        lines.append(
            f"numactl: {'available' if self.numactl_available else 'NOT installed'}"
        )
        if self.can_numa_pin:
            lines.append(
                f"numa-pin: ON — ranks will bind to socket "
                f"{self.gpu_numa} (the {self.cores_per_socket}-core "
                f"GPU-proximate socket)"
            )
        elif self.sockets >= 2 and self.gpu_numa is not None:
            lines.append(
                "numa-pin: OFF (numactl not installed — "
                "ranks will spread across sockets, accepting UPI "
                "crossing latency)"
            )
        else:
            lines.append(
                "numa-pin: N/A (single-socket box or GPU NUMA unknown)"
            )
        return "\n".join(lines)


def _run_cmd(argv: List[str]) -> Tuple[int, str]:
    """Best-effort subprocess; returns (rc, combined output)."""
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError,
            PermissionError, OSError):
        return 127, ""
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _probe_cores() -> Tuple[int, int, int]:
    """Return (phys_cores, sockets, cores_per_socket).

    Prefers ``lscpu -p`` because it reports physical cores (HT-aware);
    falls back to ``nproc / 2`` if lscpu is unavailable.
    """
    rc, out = _run_cmd(["lscpu", "-p=Core,Socket"])
    cores: set = set()
    sockets: set = set()
    if rc == 0:
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                cores.add((parts[0], parts[1]))
                sockets.add(parts[1])
    if cores and sockets:
        phys = len(cores)
        nsock = len(sockets)
        return phys, nsock, max(1, phys // nsock)

    # Fallback: nproc, assume 2-way HT, single socket
    rc, out = _run_cmd(["nproc", "--all"])
    try:
        logical = int(out.strip()) if rc == 0 else 8
    except ValueError:
        logical = 8
    phys = max(1, logical // 2)
    return phys, 1, phys


_GPU_QUERY_FIELDS = ("name", "memory.total", "compute_cap")


def _probe_gpu() -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Return (name, vram_mb, compute_cap) for GPU 0 via nvidia-smi.

    Returns ``(None, None, None)`` if nvidia-smi is missing or returns
    no devices.
    """
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    rc, out = _run_cmd([
        "nvidia-smi",
        f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i", "0",
    ])
    if rc != 0 or not out.strip():
        return None, None, None
    first = out.strip().splitlines()[0]
    fields = [p.strip() for p in first.split(",")]
    if len(fields) < 3:
        return None, None, None
    name = fields[0] or None
    try:
        vram = int(fields[1])
    except ValueError:
        vram = None
    cc = fields[2] or None
    return name, vram, cc


def _probe_gpu_numa() -> Optional[int]:
    """Return the NUMA node GPU 0 is attached to (None if not detectable).

    Delegates to :func:`molbuilder.runwrap._probe_gpu0_numa`, which is
    the single source of truth across the project for this probe.  It
    uses NVML (via ``nvidia-ml-py``) + the kernel sysfs ABI
    (``/sys/bus/pci/devices/<id>/numa_node``) -- no parsing of
    ``nvidia-smi`` tabular output, which was the failure mode of an
    earlier in-house implementation that misread the "GPU NUMA ID"
    column as the "NUMA Affinity" column and broke a Stage-3 run
    on 2026-06-16.  Same call here keeps the advisor's reading
    consistent with what the wrapper actually bakes into the run
    script (any future change to the probe lights up both surfaces
    automatically).
    """
    from molbuilder.runwrap import _probe_gpu0_numa  # local import to avoid cycle
    return _probe_gpu0_numa()


def _probe_mps_available() -> bool:
    return shutil.which("nvidia-cuda-mps-control") is not None


def _probe_numactl_available() -> bool:
    return shutil.which("numactl") is not None


def probe_host() -> HostProbe:
    phys, nsock, cps = _probe_cores()
    name, vram, cc = _probe_gpu()
    numa = _probe_gpu_numa()
    mps = _probe_mps_available()
    nctl = _probe_numactl_available()
    return HostProbe(
        phys_cores=phys,
        sockets=nsock,
        cores_per_socket=cps,
        gpu_name=name,
        gpu_vram_mb=vram,
        gpu_compute_cap=cc,
        gpu_numa=numa,
        mps_available=mps,
        numactl_available=nctl,
    )


# --------------------------------------------------------------------- #
#  Recommendation                                                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Preset:
    name: str
    mpi_np: int
    omp: int
    mps: bool
    est_vram_per_rank_mb: Optional[int]
    notes: str


# ELPA-CUDA in our build doubles the dense Hermitian matrix in VRAM
# (one host-mirror + one device copy of the eigenproblem).  Per-rank
# VRAM ~ 16 bytes * N_orb^2 / mpi_np (complex double; halved for real)
# plus ~500 MB driver/MPS overhead.  Order-of-magnitude only -- ELPA
# 2024.05 also caches some workspaces.
_BYTES_PER_COMPLEX_MATRIX_ELEM = 16
_MPS_PER_RANK_OVERHEAD_MB = 500


def _est_vram_per_rank_mb(n_orbitals: Optional[int],
                          mpi_np: int) -> Optional[int]:
    if n_orbitals is None or n_orbitals <= 0 or mpi_np <= 0:
        return None
    matrix_bytes = _BYTES_PER_COMPLEX_MATRIX_ELEM * (n_orbitals ** 2)
    per_rank_mb = (matrix_bytes // (1024 * 1024)) // mpi_np
    return int(per_rank_mb + _MPS_PER_RANK_OVERHEAD_MB)


def recommend(probe: HostProbe,
              n_atoms: Optional[int] = None,
              n_orbitals: Optional[int] = None) -> List[Preset]:
    """Build the preset list given a probed host snapshot.

    Three presets are always returned (so the user can compare):

    * ``default`` -- ELPA 2024.05 published optimum (4 ranks × 2 OMP,
      MPS on).  Auto-downsized when ``phys_cores < 8`` or
      ``n_atoms < 4``.
    * ``memory``  -- 2 ranks × more OMP, MPS on.  For when the
      eigenproblem doesn't fit in ~5 GB / rank.
    * ``fallback`` -- 1 rank × max OMP, MPS off.  When MPS isn't
      installed, or for the smallest jobs.

    The ``notes`` line on each preset is the *primary* user-facing
    artefact: it tells them when to pick this preset.
    """
    phys = max(1, probe.phys_cores)
    cps = max(1, probe.cores_per_socket)
    # Cap rank count by atom count when known (same rule the wrapper
    # uses, kept in sync with _gpu_runtime_defaults_block).
    atom_cap = max(1, n_atoms) if n_atoms else None

    # Phys-core budget mirrors the runwrap.py policy exactly:
    #
    # * NUMA-pinned (multi-socket + GPU NUMA known + numactl available):
    #   only the GPU-proximate socket is usable, but its FULL capacity
    #   is available -- the other socket sits idle, leaving room for
    #   the kernel + ELPA-GPU host driver thread without reserving a
    #   core on the GPU socket.  Budget = cps.
    # * Single-socket OR NUMA unknown OR numactl missing: whole box
    #   in play, but the driver thread shares this socket -- leave 1
    #   core.  Budget = phys_cores - 1.
    #
    # See runwrap._gpu_runtime_defaults_block for the bash mirror.
    if probe.can_numa_pin:
        budget = cps
    else:
        budget = max(1, phys - 1)

    def _cap_np(target: int) -> int:
        # Cap by the BUDGET (cps when NUMA-pinned, else phys_cores).
        # On a NUMA-pinned 10-core socket, 4 ranks of OMP=2 is the
        # max throughput preset; capping by phys_cores=20 would lie.
        capped = min(target, budget)
        if atom_cap is not None:
            capped = min(capped, atom_cap)
        return max(1, capped)

    def _omp_for(np_: int) -> int:
        # Policy adopted 2026-06-16 after the OMP-per-rank correction:
        # OMP threads inside a rank DO accelerate ELPA's host-side
        # eigensolver stages (tridiag setup, back-transform) AND
        # SIESTA's non-solver host code, even when Diag.ELPA.GPU is on.
        # The "ELPA GPU choice at runtime not compatible with OpenMP"
        # docs sentence applies to the elpa_setup_gpu runtime-switch
        # API (2023.11+), which our build does not use -- SIESTA picks
        # GPU at SCF-setup time via Diag.ELPA.GPU.
        #
        # So: fill the budget, OMP = floor(budget / mpi_np), with
        # budget set above by the NUMA-pin decision.
        return max(1, budget // max(1, np_))

    default_np = _cap_np(4)
    default_omp = _omp_for(default_np)

    memory_np = _cap_np(2)
    memory_omp = _omp_for(memory_np)

    fallback_np = 1
    fallback_omp = _omp_for(fallback_np)

    def _notes_default() -> str:
        bits = ["ELPA 2024.05 published optimum"]
        if not probe.mps_available:
            bits.append("REQUIRES MPS -- install nvidia-cuda-mps-control")
        return "; ".join(bits)

    def _notes_memory() -> str:
        bits = ["balanced if VRAM ≥ 12 GB / rank"]
        if probe.gpu_vram_mb and probe.gpu_vram_mb < 12 * 1024:
            bits.append(f"your GPU has {probe.gpu_vram_mb / 1024:.1f} GB total")
        if not probe.mps_available:
            bits.append("requires MPS")
        return "; ".join(bits)

    def _notes_fallback() -> str:
        bits = ["single-rank fallback"]
        if not probe.mps_available:
            bits.append("MPS unavailable -- this is the only safe choice")
        else:
            bits.append("use if both presets above OOM")
        return "; ".join(bits)

    return [
        Preset(
            name="default",
            mpi_np=default_np, omp=default_omp,
            mps=probe.mps_available,
            est_vram_per_rank_mb=_est_vram_per_rank_mb(n_orbitals, default_np),
            notes=_notes_default(),
        ),
        Preset(
            name="memory",
            mpi_np=memory_np, omp=memory_omp,
            mps=probe.mps_available,
            est_vram_per_rank_mb=_est_vram_per_rank_mb(n_orbitals, memory_np),
            notes=_notes_memory(),
        ),
        Preset(
            name="fallback",
            mpi_np=fallback_np, omp=fallback_omp,
            mps=False,
            est_vram_per_rank_mb=_est_vram_per_rank_mb(n_orbitals, fallback_np),
            notes=_notes_fallback(),
        ),
    ]


def _recommend_picked(presets: List[Preset],
                      probe: HostProbe) -> Preset:
    """Pick the row to surface as the "use these env vars" snippet.

    Mirrors the wrapper's MPS-available default: pick ``default`` if MPS
    is available; otherwise ``fallback``.
    """
    if not probe.mps_available:
        return presets[-1]
    return presets[0]


# --------------------------------------------------------------------- #
#  Rendering                                                             #
# --------------------------------------------------------------------- #


def _fmt_vram(mb: Optional[int]) -> str:
    if mb is None:
        return "n/a"
    if mb >= 1024:
        return f"~{mb / 1024:.1f} GB"
    return f"~{mb} MB"


def format_report(probe: HostProbe,
                  presets: List[Preset],
                  *,
                  n_atoms: Optional[int] = None,
                  n_orbitals: Optional[int] = None) -> str:
    """Build the full user-facing text the CLI prints.

    Sections:
      1. host snapshot (probe.describe())
      2. problem-size echo (only when supplied)
      3. preset table (4 columns: name, mpi_np, omp, mps, VRAM/rank, notes)
      4. picked snippet (export MOLBUILDER_* lines for one preset)
      5. footer pointer to the runtime override knobs
    """
    lines: List[str] = []
    lines.append(probe.describe())
    if n_atoms or n_orbitals:
        bits = []
        if n_atoms:
            bits.append(f"{n_atoms} atoms")
        if n_orbitals:
            bits.append(f"~{n_orbitals} orbitals")
        lines.append(f"problem: {' × '.join(bits)}")
    lines.append("")

    # Table.
    header = ("preset", "mpi_np", "omp", "mps", "VRAM/rank", "notes")
    rows = [header]
    for p in presets:
        rows.append((
            p.name,
            str(p.mpi_np),
            str(p.omp),
            "on" if p.mps else "off",
            _fmt_vram(p.est_vram_per_rank_mb),
            p.notes,
        ))
    widths = [
        max(len(r[i]) for r in rows)
        for i in range(len(header) - 1)  # last col (notes) wraps; no pad
    ]
    sep = "  "
    head_line = sep.join(rows[0][i].ljust(widths[i])
                         for i in range(len(widths))) + sep + rows[0][-1]
    rule = sep.join("─" * widths[i] for i in range(len(widths))) + sep + \
        "─" * max(len(rows[0][-1]), 8)
    lines.append(head_line)
    lines.append(rule)
    for row in rows[1:]:
        body = sep.join(row[i].ljust(widths[i])
                        for i in range(len(widths)))
        lines.append(body + sep + row[-1])
    lines.append("")

    picked = _recommend_picked(presets, probe)
    lines.append(f"Recommended preset for this host: `{picked.name}`")
    lines.append(f"  export MOLBUILDER_MPI_NP={picked.mpi_np}")
    lines.append(f"  export MOLBUILDER_OMP_NUM_THREADS={picked.omp}")
    lines.append(f"  export MOLBUILDER_USE_MPS={1 if picked.mps else 0}")
    lines.append("")
    lines.append("Override at any time:")
    lines.append("  * env vars above")
    lines.append("  * wrapper flags: -np N -omp M --mps / --no-mps")
    lines.append("  * generator UI: SIESTA → Compute & budget")
    return "\n".join(lines)


__all__ = [
    "HostProbe",
    "Preset",
    "probe_host",
    "recommend",
    "format_report",
]
