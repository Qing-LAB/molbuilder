"""A PySCF run's own set-up: its threads, what it records about itself, and
its GPU.

THE SCRIPT IMPORTS IT, from ``mb_pyscf.pyz`` beside the job
(``runwrap.PYSCF_COMPANIONS``, `engines/pyscf.md` § 3), and calls it on its
first lines: :func:`cap_threads` sizes the run's threads and caps BLAS's
before numpy is imported -- the variables are read when numpy loads -- so
this module imports only the standard library at load: psutil, when the env
has it, inside :func:`physical_core_count`, and cupy and gpu4pyscf inside
:func:`probe_gpu`.  *(Until 2026-10-05 the thread set-up and the GPU probe
were written into every script as text, emitted from here; user, "yes to
#1".)*

The rule it keeps -- no oversubscription, and the run says what it did.  A
20-physical / 40-logical host ran at load 40 without it:

  * BLAS one thread (``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``) and
    OpenMP the run's count, set before numpy is imported;
  * the count the run's own, else the allocation's, the node's only last
    (:func:`threads_for`, :data:`THREAD_SOURCES`);
  * ``os.environ.setdefault``, so a value already exported wins;
  * PySCF's own pool sized after PySCF is imported -- the script's
    ``pyscf.lib.num_threads(N)``, the canonical post-import setter.

What a run records about itself is :func:`runtime_facts` -- the
:data:`RUNTIME_INFO_KEYS` -- which the result writers copy into the files
the Results tab reads, so a run shows the same CPU / GPU / host rows
whichever engine ran it.
"""
from __future__ import annotations

import os
import socket
from typing import Mapping, Optional, Tuple


# Canonical keys that every engine's runtime_info should populate.
# Documented here so the inspector panels can write to a single
# contract.  Missing keys render as "—" in the UI (the inspector
# tolerates absence; this is the source-of-truth list, not a hard
# schema).
RUNTIME_INFO_KEYS = (
    "n_threads_pyscf",        # actual pyscf.lib.num_threads() value
    "n_threads_omp",          # OMP_NUM_THREADS at script start
    "n_threads_blas",         # OPENBLAS_NUM_THREADS / MKL_NUM_THREADS (= 1)
    "physical_cores",         # host's physical core count
    "logical_cores",          # host's logical core count (HT-inclusive)
    "max_memory_mb",          # MB cap the script set on mol.max_memory
    "gpu_requested",          # bool: did the user ask for GPU?
    "gpu_used",               # bool: did GPU actually engage?
    "gpu_name",               # str: GPU device name OR fallback-reason string
    "gpu_compute_capability", # str: e.g. "8.9" (None if no GPU)
    "cuda_version",           # str: e.g. "12.4" (None if no GPU)
    "hostname",               # str: socket.gethostname()
)


def physical_core_count() -> int:
    """Return the host's physical (not logical) core count.

    Tries ``psutil`` first, then ``/proc/cpuinfo``, then
    ``os.cpu_count() // 2`` (a hyperthreaded box's best-guess).
    Final fallback: 1.  Never raises -- always returns >= 1.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fp:
            seen = set()
            phys = core = None
            for line in fp:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    if phys is not None:
                        seen.add((phys, core))
                        phys = core = None
            if seen:
                return len(seen)
    except Exception:
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical >= 2 else logical


#: WHERE A RUN'S THREAD COUNT COMES FROM when its settings state none, in
#: order: what the run script exported, else what the scheduler allocated.
#: ONE LIST, read by both chains (`execution/running-a-job.md` § 3.2): the
#: script's own, :func:`threads_for`, which ends on the node's cores, and the
#: run script's, which `runwrap` builds from it and ends on the count stated at
#: prep.  *(Each chain spelled it until 2026-10-05, held in step by a comment
#: -- which is how the run script came to lack the last two.)*
THREAD_SOURCES = ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "PBS_NCPUS",
                  "NSLOTS")


def threads_for(pinned: Optional[int] = None,
                environ: Optional[Mapping[str, str]] = None,
                physical: Optional[int] = None) -> Tuple[int, str]:
    """``(threads, where the count came from)`` for a PySCF run.

    The run's own ``threads`` when its settings state one; else the first of
    :data:`THREAD_SOURCES` set to a whole number of at least one -- read from
    ``environ``, the process's own by default; else the node's physical cores
    (``physical``, or :func:`physical_core_count`).

    The node is the last resort, never an early answer.  It is right on a
    workstation -- the node IS the allocation -- and wrong under a scheduler,
    expensively: a job given 8 cores of a 128-core node that counted the node
    started 128 OpenMP threads, which the cgroup then time-sliced onto the 8
    it granted -- slower than an honest 8, and the thrashing charged to it.
    """
    if pinned is not None:
        return int(pinned), f"config (threads={int(pinned)})"
    env = os.environ if environ is None else environ
    for var in THREAD_SOURCES:
        said = env.get(var)
        if not said:
            continue
        try:
            n = int(said)
        except ValueError:
            continue
        if n >= 1:
            return n, var
    return (physical if physical is not None else physical_core_count(),
            "node physical cores")


def cap_threads(pinned: Optional[int] = None) -> Tuple[int, int]:
    """Size this run's threads and cap BLAS's -- what a PySCF script does on
    its first lines, before numpy is imported (`engines/pyscf.md` § 3): the
    variables are read when numpy and PySCF load, so a cap set later caps
    nothing.

    ``OMP_NUM_THREADS`` and ``NUMEXPR_NUM_THREADS`` take the count
    :func:`threads_for` answers, and every BLAS one thread, so OpenMP's
    threads and BLAS's own do not multiply (a 20-core host ran at load 40
    without it) -- each by ``setdefault``, so a value already exported wins.
    Says on stdout how many, and where the count came from: a run that sized
    itself from the node when it should have read the allocation is otherwise
    indistinguishable in its log from one that was told 128.

    Returns ``(threads, the node's physical cores)`` -- the node probed once.
    PySCF's own pool is sized after its import, by the script
    (``pyscf.lib.num_threads``).
    """
    physical = physical_core_count()
    n, whence = threads_for(pinned, physical=physical)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # macOS Accelerate
    print(f"molbuilder: requested {n} PySCF threads from {whence} "
          f"(node physical={physical}, logical={os.cpu_count() or 1}, "
          f"BLAS=1).  Override via OMP_NUM_THREADS env.")
    return n, physical


def runtime_facts(threads: int, physical_cores: int, *,
                  max_memory_mb: Optional[int] = None,
                  gpu_requested: bool = False) -> dict:
    """What a run records about itself as it starts -- the
    :data:`RUNTIME_INFO_KEYS` -- read after :func:`cap_threads`, whose
    variables it reports.  The script keeps it as ``_RUNTIME_INFO`` and adds
    to it; the result writers (the progress log's header, the spectrum's
    file) copy it out.  The GPU's four are :func:`probe_gpu`'s to fill."""
    return {
        "n_threads_pyscf": int(threads),     # read again after PySCF loads
        "n_threads_omp": int(os.environ["OMP_NUM_THREADS"]),
        "n_threads_blas": int(os.environ["OPENBLAS_NUM_THREADS"]),
        "physical_cores": int(physical_cores),
        "logical_cores": os.cpu_count() or 1,
        "max_memory_mb": (int(max_memory_mb) if max_memory_mb is not None
                          else None),
        "gpu_requested": bool(gpu_requested),
        "gpu_used": False,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "cuda_version": None,
        "hostname": socket.gethostname(),
    }


# Minimum NVIDIA GPU compute capability gpu4pyscf supports.  7.0 = Volta;
# below that the runtime imports of gpu4pyscf raise.
GPU4PYSCF_MIN_COMPUTE_CAPABILITY = 7


def probe_gpu(facts: dict, min_compute_capability: int =
              GPU4PYSCF_MIN_COMPUTE_CAPABILITY) -> bool:
    """The GPU this run asked for, found -- or the run stops.

    cupy and gpu4pyscf importable, an NVIDIA device, its compute capability
    at least ``min_compute_capability``: ``True``, with the device's name,
    compute capability and CUDA version written into ``facts``
    (:func:`runtime_facts`) and one line on stdout.  Anything else is a
    ``SystemExit`` naming what is missing and the two ways out.

    **No CPU fallback** (user, 2026-08-17; `engines/overview.md` § 3a G-5): a
    run that silently changed where it ran would report a CPU time under a
    GPU label, and a benchmark would score it.  Asked when the run starts,
    never at prep: the device is the compute node's (`engines/pyscf.md`).
    """
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no NVIDIA GPU detected")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"(unknown)")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        major, minor = int(props.get("major", 0)), int(props.get("minor", 0))
        if major < min_compute_capability:
            raise RuntimeError(
                f"GPU {name} compute capability {major}.{minor}; "
                f"gpu4pyscf requires >= {min_compute_capability}.0")
        facts["gpu_used"] = True
        facts["gpu_name"] = name
        facts["gpu_compute_capability"] = f"{major}.{minor}"
        try:
            v = cp.cuda.runtime.runtimeGetVersion()
            facts["cuda_version"] = f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            pass
    except ImportError as exc:
        facts["gpu_name"] = f"gpu4pyscf not installed: {exc}"
        raise SystemExit(
            "molbuilder: this run asked for the GPU (use_gpu = true) and\n"
            f"  gpu4pyscf is not importable here: {exc}\n"
            "  There is no CPU fallback: a run that silently changed where it\n"
            "  executed would report a CPU time under a GPU label.\n"
            "  Fix: run in an env with gpu4pyscf + cupy, or set use_gpu = "
            "false.")
    except Exception as exc:
        facts["gpu_name"] = f"GPU unusable: {exc}"
        raise SystemExit(
            "molbuilder: this run asked for the GPU (use_gpu = true) and\n"
            f"  the local GPU is not usable: {exc}\n"
            "  There is no CPU fallback (see above).  Fix: run on a node with "
            "a\n"
            f"  supported NVIDIA GPU (compute capability >= "
            f"{min_compute_capability}.0), or set use_gpu = false.")
    print(f"GPU acceleration ON (gpu4pyscf, {name}, CC {major}.{minor}).")
    return True


def to_gpu(mf):
    """``mf`` moved onto the GPU -- gpu4pyscf's ``.to_gpu()`` -- once an
    optimization script has assembled it whole (density fitting, dispersion
    and solvent applied), so the copy on the device is all of it.  A
    promotion that fails stops the run: the probe announced GPU
    acceleration, and finishing on the CPU would make that false.  (A
    vibration script builds its SCF objects from gpu4pyscf's classes
    instead, and never calls this.)"""
    try:
        return mf.to_gpu()
    except Exception as exc:
        raise SystemExit(
            "molbuilder: this run asked for the GPU and the probe found one,\n"
            f"  but promoting the SCF object to gpu4pyscf failed: {exc}\n"
            "  There is no CPU fallback: the run announced GPU acceleration,\n"
            "  and finishing on the CPU would make that announcement false.")


__all__ = [
    "GPU4PYSCF_MIN_COMPUTE_CAPABILITY",
    "RUNTIME_INFO_KEYS",
    "THREAD_SOURCES",
    "cap_threads",
    "physical_core_count",
    "probe_gpu",
    "runtime_facts",
    "threads_for",
    "to_gpu",
]
