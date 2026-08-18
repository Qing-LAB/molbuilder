"""Shared runtime-info emission helpers.

Every script molbuilder emits that does real compute (PySCF geom opt,
PySCF spectra, future SIESTA wrappers) should run the SAME threading
setup BEFORE numpy/pyscf import and capture the SAME runtime facts
into a ``_RUNTIME_INFO`` dict that goes into the run's output file.

This is the cross-cutting "no oversubscription, and tell the user
what happened" rule.  The user's observed load=40 on a 20-physical /
40-logical host was the symptom of NOT having this in place; the
fix is canonical:

  * Set ``OPENBLAS_NUM_THREADS=1`` + ``MKL_NUM_THREADS=1`` BEFORE
    numpy import (env vars are read at import time).
  * Set ``OMP_NUM_THREADS = physical_cores`` -- hyperthreading
    rarely helps memory-bandwidth-bound QC kernels and can hurt
    cache locality.
  * Call ``pyscf.lib.num_threads(N)`` AFTER pyscf import (env vars
    don't re-thread already-imported modules; this is the
    canonical post-import setter per PySCF docs).
  * Use ``os.environ.setdefault`` so a user's explicit pre-export
    still wins -- our auto-detect is the default, not a hard cap.

Result-file format: every emitted script also captures a
``_RUNTIME_INFO`` dict with canonical keys (see
:data:`RUNTIME_INFO_KEYS`) that engine-specific result writers
copy into their on-disk output.  The /results inspector panels
read these keys uniformly so the user sees the SAME
"CPU / GPU / Host" rows regardless of which engine ran.
"""
from __future__ import annotations

import os
from typing import List, Optional


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


def emit_threading_setup_lines(threads: Optional[int]) -> List[str]:
    """Emit Python source lines for the BLAS / OMP threading setup.

    Inserted at the TOP of the generated script (before numpy /
    pyscf import).  ``threads=None`` -> auto-detect physical cores
    at run time.  ``threads=N`` -> use exactly N.

    Defines two module-level names on the running script:
      * ``_MB_REQUESTED_THREADS``    -- the N actually requested
      * ``_mb_count_physical_cores`` -- the detection helper (also
        used later in the runtime-info capture block)
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Threading setup -- runs BEFORE numpy/pyscf import.")
    out.append("# ============================================================")
    out.append("# Cap BLAS to 1 thread per worker so OMP threads * BLAS")
    out.append("# threads doesn't multiply.  PySCF does its own parallel")
    out.append("# loops; we size them to PHYSICAL cores (not logical).")
    out.append("# Hyperthreading rarely helps QC kernels (memory bandwidth")
    out.append("# bound) and can hurt cache locality.  Shared recipe lives")
    out.append("# in molbuilder/runtime_info.py.")
    out.append("import os")
    out.append("")
    out.append("def _mb_count_physical_cores():")
    out.append("    try:")
    out.append("        import psutil")
    out.append("        n = psutil.cpu_count(logical=False)")
    out.append("        if n: return int(n)")
    out.append("    except Exception:")
    out.append("        pass")
    out.append("    try:")
    out.append("        with open('/proc/cpuinfo') as fp:")
    out.append("            seen = set()")
    out.append("            phys, core = None, None")
    out.append("            for line in fp:")
    out.append("                if line.startswith('physical id'):")
    out.append("                    phys = line.split(':')[1].strip()")
    out.append("                elif line.startswith('core id'):")
    out.append("                    core = line.split(':')[1].strip()")
    out.append("                    if phys is not None:")
    out.append("                        seen.add((phys, core))")
    out.append("                        phys = core = None")
    out.append("            if seen: return len(seen)")
    out.append("    except Exception:")
    out.append("        pass")
    out.append("    logical = os.cpu_count() or 1")
    out.append("    return max(1, logical // 2) if logical >= 2 else logical")
    out.append("")
    if threads is None:
        out.append("# threads not pinned -> ask the ALLOCATION first, the")
        out.append("# machine only as a last resort.")
        out.append("#")
        out.append("# _mb_count_physical_cores() counts the whole NODE.  On a")
        out.append("# workstation that is right -- the node IS the allocation.")
        out.append("# Under a scheduler it is wrong and expensively so: a job")
        out.append("# given 8 of a 128-core node would start 128 OpenMP")
        out.append("# threads, which the cgroup then time-slices onto the 8")
        out.append("# cores it granted.  The job runs slower than an honest 8")
        out.append("# would, and the thrashing is charged to it.")
        out.append("#")
        out.append("# So: what the wrapper exported, else what the scheduler")
        out.append("# allocated, else the node.  Each step is a better")
        out.append("# statement of 'how much of this machine is mine'.")
        out.append("def _mb_resolve_threads():")
        out.append("    for _var in ('OMP_NUM_THREADS', 'SLURM_CPUS_PER_TASK',")
        out.append("                 'PBS_NCPUS', 'NSLOTS'):")
        out.append("        _v = os.environ.get(_var)")
        out.append("        if _v:")
        out.append("            try:")
        out.append("                _n = int(_v)")
        out.append("            except ValueError:")
        out.append("                continue")
        out.append("            if _n >= 1:")
        out.append("                return _n, _var")
        out.append("    return _mb_count_physical_cores(), 'node physical cores'")
        out.append("")
        out.append("_MB_REQUESTED_THREADS, _MB_THREADS_FROM = _mb_resolve_threads()")
    else:
        out.append(f"# threads explicitly = {threads}; honor user choice.")
        out.append(f"_MB_REQUESTED_THREADS = {int(threads)}")
        out.append(f"_MB_THREADS_FROM = 'config (threads={int(threads)})'")
    out.append("")
    out.append("# setdefault so a user's pre-exported env var wins.")
    out.append("os.environ.setdefault('OMP_NUM_THREADS',      str(_MB_REQUESTED_THREADS))")
    out.append("os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')")
    out.append("os.environ.setdefault('MKL_NUM_THREADS',      '1')")
    out.append("os.environ.setdefault('NUMEXPR_NUM_THREADS',  str(_MB_REQUESTED_THREADS))")
    out.append("os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')  # macOS Accelerate")
    out.append("")
    # Say WHERE the number came from, not just what it is.  A run that
    # sized itself from the node when it should have read the
    # allocation is otherwise indistinguishable in the log from one
    # that was told 128 -- and that is the failure this resolution
    # order exists to prevent.
    # Probe the node ONCE and reuse it.  The count is wanted in three
    # places -- the resolver's last resort, this banner, and
    # _RUNTIME_INFO['physical_cores'] -- and each call re-imports psutil
    # or re-reads /proc/cpuinfo for an answer that cannot change during
    # a run.
    out.append("_MB_PHYS_CORES = _mb_count_physical_cores()")
    out.append("print(")
    out.append("    f'molbuilder: requested {_MB_REQUESTED_THREADS} PySCF threads '")
    out.append("    f'from {_MB_THREADS_FROM} '")
    out.append("    f'(node physical={_MB_PHYS_CORES}, '")
    out.append("    f'logical={os.cpu_count() or 1}, BLAS=1).  '")
    out.append("    f'Override via OMP_NUM_THREADS env.'")
    out.append(")")
    out.append("")
    return out


def emit_pyscf_post_import_lines() -> List[str]:
    """Emit the ``pyscf.lib.num_threads(N)`` call -- the canonical
    post-import thread-pool sizer per PySCF docs.  Must come AFTER
    ``from pyscf import ...`` because env vars are read at import
    time and don't re-thread an already-imported module."""
    out: List[str] = []
    out.append("# PySCF post-import thread-pool size: env vars are read")
    out.append("# at import time, so lib.num_threads(N) is the canonical")
    out.append("# setter to use afterwards.")
    out.append("from pyscf import lib as _pyscf_lib")
    out.append("_pyscf_lib.num_threads(_MB_REQUESTED_THREADS)")
    out.append("")
    return out


def emit_runtime_info_capture_lines(use_gpu: bool,
                                    max_memory_mb: Optional[int] = None) -> List[str]:
    """Build the ``_RUNTIME_INFO`` dict in the running script.

    Assumes :func:`emit_threading_setup_lines` ran first (it relies
    on ``_MB_REQUESTED_THREADS`` and ``_mb_count_physical_cores`` being
    defined).  Engine-specific blocks (e.g. the GPU probe via
    :func:`emit_gpu_probe_lines`) later mutate this dict to record
    the GPU outcome.

    ``max_memory_mb``: the per-run PySCF memory cap (set on
    ``mol.max_memory`` in the script).  Recorded here so the
    /results inspector can show "this run was capped to N MB" --
    a useful resource-trace fact alongside CPU/GPU counts.  Pass
    None if the script doesn't impose a cap.
    """
    out: List[str] = []
    out.append("# Runtime-info bag -- result-file writers (the molwatch")
    out.append("# log header, the spectra.json final dump) copy this dict")
    out.append("# into their on-disk output so the /results inspectors")
    out.append("# can show 'this run used N threads, M MB, GPU ON/OFF, ...'.")
    out.append("import socket as _mb_socket")
    out.append("_RUNTIME_INFO = {")
    out.append("    'n_threads_pyscf':         _MB_REQUESTED_THREADS,  # may be re-read post-import")
    out.append("    'n_threads_omp':           int(os.environ['OMP_NUM_THREADS']),")
    out.append("    'n_threads_blas':          int(os.environ['OPENBLAS_NUM_THREADS']),")
    out.append("    'physical_cores':          _MB_PHYS_CORES,")
    out.append("    'logical_cores':           os.cpu_count() or 1,")
    out.append(f"    'max_memory_mb':           {int(max_memory_mb) if max_memory_mb is not None else None!r},")
    out.append(f"    'gpu_requested':           {bool(use_gpu)!r},")
    out.append("    'gpu_used':                False,        # set True by the GPU probe if it engages")
    out.append("    'gpu_name':                None,")
    out.append("    'gpu_compute_capability':  None,")
    out.append("    'cuda_version':            None,")
    out.append("    'hostname':                _mb_socket.gethostname(),")
    out.append("}")
    out.append("")
    return out


# Minimum NVIDIA GPU compute capability gpu4pyscf supports.  7.0 = Volta;
# below that the runtime imports of gpu4pyscf raise.  Defined as a module
# constant so engine emitters (pyscf/input.py + spectra/pyscf_engine.py)
# import from one place instead of duplicating the literal.  The previous
# arrangement had ``7`` hard-coded in pyscf/input.py + as a class constant
# on PySCFSpectraEngine + as a default kwarg here -- three sources, easy
# to drift.
GPU4PYSCF_MIN_COMPUTE_CAPABILITY = 7


def emit_gpu_probe_lines(use_gpu: bool,
                          min_compute_capability: int = GPU4PYSCF_MIN_COMPUTE_CAPABILITY) -> List[str]:
    """Emit the GPU probe + to_gpu helper.

    **No silent fallback** (user, 2026-08-17; `engines/overview.md` § 3a
    G-5).  When ``use_gpu`` is set and the GPU is missing, unusable or too
    old, the emitted script **raises SystemExit with an actionable
    message** rather than continuing on the CPU.  All three former fallback
    paths -- import failure, device failure, and a failed ``.to_gpu()``
    promotion -- now stop.

    The reason is a measurement one as much as a correctness one: a trial
    labelled *GPU* that quietly ran on the CPU puts a CPU time in a GPU
    column, and `bench/result.py` would score it.

    Defines these module-level names on the running script:
      * ``USE_GPU``    -- the literal config value (True/False).
      * ``_USING_GPU`` -- True iff gpu4pyscf + a usable NVIDIA GPU
        (compute capability >= ``min_compute_capability``) were both
        found at run start.  Since the failure paths now exit, this is
        equal to ``USE_GPU`` for any script that gets past the probe.
      * ``_mb_to_gpu_if_enabled(mf)`` -- helper used by callers to
        promote an mf object to its gpu4pyscf equivalent when
        ``_USING_GPU`` is True; pass-through when the user asked for CPU.

    Caller responsibility: invoke ``mf = _mb_to_gpu_if_enabled(mf)``
    AFTER the mf is fully assembled (density-fit + dispersion +
    PCM all applied) so ``.to_gpu()`` sees the complete CPU object.

    Side-effects: writes ``gpu_used``, ``gpu_name``,
    ``gpu_compute_capability``, ``cuda_version`` into the
    ``_RUNTIME_INFO`` dict (which :func:`emit_runtime_info_capture_lines`
    initialised with placeholder None / False values).
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  GPU probe (optional, NVIDIA via gpu4pyscf).")
    out.append("# ============================================================")
    out.append("# Probes cupy + gpu4pyscf at run start.  If both present AND")
    out.append("# the local GPU has compute capability >=")
    out.append(f"# {min_compute_capability}.0, sets _USING_GPU=True and the helper below")
    out.append("# promotes mf objects via .to_gpu().")
    out.append("#")
    out.append("# THERE IS NO CPU FALLBACK.  You asked for the GPU; if it is")
    out.append("# not here the run STOPS (no install, no GPU, too-old card).")
    out.append("# A run that silently changed where it executed would report a")
    out.append("# CPU time under a GPU label -- and a benchmark would score it.")
    out.append(f"USE_GPU = {bool(use_gpu)!r}")
    out.append("_USING_GPU = False")
    out.append("if USE_GPU:")
    out.append("    try:")
    out.append("        import cupy as _cp")
    out.append("        import gpu4pyscf  # noqa: F401")
    out.append("        _n_dev = _cp.cuda.runtime.getDeviceCount()")
    out.append("        if _n_dev == 0:")
    out.append("            raise RuntimeError('no NVIDIA GPU detected')")
    out.append("        _props = _cp.cuda.runtime.getDeviceProperties(0)")
    out.append("        _gpu_name = _props.get('name', b'(unknown)')")
    out.append("        if isinstance(_gpu_name, bytes):")
    out.append("            _gpu_name = _gpu_name.decode('utf-8', errors='replace')")
    out.append("        _maj = int(_props.get('major', 0))")
    out.append("        _min = int(_props.get('minor', 0))")
    out.append(f"        if _maj < {min_compute_capability}:")
    out.append("            raise RuntimeError(")
    out.append("                f'GPU {_gpu_name} compute capability "
               "{_maj}.{_min}; '")
    out.append(f"                f'gpu4pyscf requires >= {min_compute_capability}.0'")
    out.append("            )")
    out.append("        _USING_GPU = True")
    out.append("        _RUNTIME_INFO['gpu_used'] = True")
    out.append("        _RUNTIME_INFO['gpu_name'] = _gpu_name")
    out.append("        _RUNTIME_INFO['gpu_compute_capability'] = f'{_maj}.{_min}'")
    out.append("        try:")
    out.append("            _cv = _cp.cuda.runtime.runtimeGetVersion()")
    out.append("            _RUNTIME_INFO['cuda_version'] = f'{_cv // 1000}.{(_cv % 1000) // 10}'")
    out.append("        except Exception:")
    out.append("            pass")
    out.append("        print(f'GPU acceleration ON (gpu4pyscf, {_gpu_name}, "
               "CC {_maj}.{_min}).')")
    # NO SILENT FALLBACK (user, 2026-08-17; `engines/overview.md` § 3a G-5).
    #
    # These two branches printed "CPU fallback" and carried on.  That made a
    # GPU run and a CPU run indistinguishable except by reading the log, and
    # it made a BENCHMARK dishonest: a trial labelled *GPU* that quietly ran
    # on the CPU reports a CPU time in a GPU column, so the GPU looks slow for
    # a reason that has nothing to do with the GPU.
    #
    # The user asked for the GPU.  Either they get it or the run stops --
    # the same rule the boundary-condition contract states as *no silent
    # absorption of config* (`engines/overview.md` § 3).
    out.append("    except ImportError as _gpu_exc:")
    out.append("        _RUNTIME_INFO['gpu_name'] = f'gpu4pyscf not installed: {_gpu_exc}'")
    out.append("        raise SystemExit(")
    out.append("            'molbuilder: this run asked for the GPU "
               "(use_gpu = true) and\\n'")
    out.append("            f'  gpu4pyscf is not importable here: {_gpu_exc}\\n'")
    out.append("            '  There is no CPU fallback: a run that silently "
               "changed where it\\n'")
    out.append("            '  executed would report a CPU time under a GPU "
               "label.\\n'")
    out.append("            '  Fix: run in an env with gpu4pyscf + cupy, or "
               "set use_gpu = false.')")
    out.append("    except Exception as _gpu_exc:")
    out.append("        _RUNTIME_INFO['gpu_name'] = f'GPU unusable: {_gpu_exc}'")
    out.append("        raise SystemExit(")
    out.append("            'molbuilder: this run asked for the GPU "
               "(use_gpu = true) and\\n'")
    out.append("            f'  the local GPU is not usable: {_gpu_exc}\\n'")
    out.append("            '  There is no CPU fallback (see above).  Fix: run "
               "on a node with a\\n'")
    out.append(f"            '  supported NVIDIA GPU (compute capability >= "
               f"{min_compute_capability}.0), or set use_gpu = false.')")
    out.append("")
    out.append("def _mb_to_gpu_if_enabled(mf_obj):")
    out.append("    \"\"\"Promote an mf object to its gpu4pyscf equivalent when")
    out.append("    _USING_GPU is True; pass through unchanged on CPU.  Call")
    out.append("    AFTER mf is fully assembled (density_fit + disp + PCM all")
    out.append("    applied) so .to_gpu() mirrors the complete CPU state.\"\"\"")
    out.append("    if not _USING_GPU:")
    out.append("        return mf_obj")
    # The third silent fallback, and the least visible of the three: the probe
    # passed, so the run announced "GPU acceleration ON", and THEN the
    # promotion failed and the work went to the CPU anyway.
    out.append("    try:")
    out.append("        return mf_obj.to_gpu()")
    out.append("    except Exception as _e:")
    out.append("        raise SystemExit(")
    out.append("            'molbuilder: this run asked for the GPU and the "
               "probe found one,\\n'")
    out.append("            f'  but promoting the SCF object to gpu4pyscf "
               "failed: {_e}\\n'")
    out.append("            '  There is no CPU fallback: the run announced "
               "GPU acceleration,\\n'")
    out.append("            '  and finishing on the CPU would make that "
               "announcement false.')")
    out.append("")
    return out


__all__ = [
    "RUNTIME_INFO_KEYS",
    "physical_core_count",
    "emit_threading_setup_lines",
    "emit_pyscf_post_import_lines",
    "emit_runtime_info_capture_lines",
    "emit_gpu_probe_lines",
]
