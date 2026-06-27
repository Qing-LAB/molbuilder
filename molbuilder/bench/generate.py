"""``molbuilder bench generate`` -- CPU-only + GPU-only benchmark bundles.

Generalises the hand-built ``sol-bench-BDT-Au111-hex`` bundle: from ONE
input ``.fdf`` it emits two self-contained, comparable benchmark jobs so
the user can measure CPU vs GPU on their own system and decide which
mechanism the production run should use (molbuilder makes NO automatic
recommendation -- each bundle just reports its own wall/iter).

  * ``job-cpu.{fdf,run.sh,sbatch}`` -- plain ``diagon`` (no
    ``Diag.ELPA.GPU``); the run-wrapper auto-selects ``molbuilder-siesta``.
  * ``job-gpu.{fdf,run.sh,sbatch}`` -- ``Diag.ELPA.GPU .true.``; the
    wrapper auto-selects ``molbuilder-siesta-gpu``.

Both are made COLD and comparable (``MaxSCFIterations 5``,
``DM.UseSaveDM .false.``, MD/relaxation steps zeroed) -- everything else
is copied verbatim from the input fdf.  Tuning is done at submit time via
sbatch args (the launcher auto-adapts, slurm-integration.md § 7.3): the
CPU job scales with ``-n <np>``; the GPU job with ``--gres=gpu:a100:G``
(GPUs) x ``-n (K*G)`` (K ranks/GPU) x ``-c (cores_per_socket/K)``.  A
self-contained ``job-gpu-sweep.sh`` helper computes/emits that grid.

This module only does text surgery on the fdf + reuses
:func:`molbuilder.runwrap.write_run_wrapper` (which writes the
``.run.sh`` + ``.sbatch`` and ships ``mb_monitor.py``); it never
re-renders the structure, so the benchmarked input is byte-for-byte the
user's own fdf apart from the handful of flipped directives.

The directive surgery is SIESTA-label-normalized (``.``/``-``/``_``
stripped, case-insensitive), so a variant-spelled input directive (e.g.
``Diag-ELPA-GPU``) is matched and replaced in place rather than
duplicated -- it does not assume a molbuilder-canonical input.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from ..runwrap import write_run_wrapper

# GPU ranks-per-GPU values the sweep helper enumerates (the launcher
# turns on MPS when K >= 2).  K=16 keeps c = cores_per_socket/16 = 1
# (OMP off -- pure rank-parallelism via MPS), a distinct regime worth
# measuring; K beyond cores_per_socket is flagged invalid by the helper.
_SWEEP_KS = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------- #
#  fdf surgery                                                          #
# --------------------------------------------------------------------- #


def _norm_label(s: str) -> str:
    """SIESTA fdf-label normalization: case-insensitive and with ``.``
    ``-`` ``_`` stripped entirely (``DM.UseSaveDM`` == ``DM-UseSaveDM`` ==
    ``dmusesavedm``).  Matching on this is what makes the surgery robust
    to variant-spelled input rather than only the canonical form."""
    return s.lower().replace(".", "").replace("-", "").replace("_", "")


def _set_or_append(text: str, anchor: str, value: str, *,
                   only_if_present: bool = False) -> str:
    """Set ``<anchor> <value>`` SIESTA-normalized: replace the first
    directive line whose label normalizes to ``anchor`` (regardless of
    its `.`/`-`/`_`/case spelling) in place; else append it inside the
    engine body (before any user-custom block).

    ``only_if_present`` -> never append (used for MD step-count: absent
    means single-point already, nothing to zero).  Operating on
    normalized labels avoids the duplicate-conflicting-line bug a plain
    ``re.escape(anchor)`` match causes on legacy spellings (audit
    2026-06-27 B-2).
    """
    target = _norm_label(anchor)
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped[0] in "#%":
            continue
        tok = stripped.split(None, 1)[0]
        if _norm_label(tok) == target:
            indent = line[:len(line) - len(stripped)]
            eol = "\n" if line.endswith("\n") else ""
            lines[i] = (f"{indent}{anchor}          {value}"
                        f"    # bench override{eol}")
            return "".join(lines)
    if only_if_present:
        return text
    new_line = f"{anchor}          {value}    # bench override"
    if "user-custom BEGIN" in text:
        return text.replace(
            "# === molbuilder user-custom BEGIN ===",
            f"{new_line}\n\n# === molbuilder user-custom BEGIN ===",
            1,
        )
    return text.rstrip("\n") + "\n" + new_line + "\n"


def _remove_directive(text: str, anchor: str) -> str:
    """Drop every directive line whose label normalizes to ``anchor``
    (any `.`/`-`/`_`/case spelling) -- used to strip GPU directives from
    the CPU bundle so it is a genuine plain ``diagon`` run."""
    target = _norm_label(anchor)
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped and stripped[0] not in "#%":
            tok = stripped.split(None, 1)[0]
            if _norm_label(tok) == target:
                continue
        out.append(line)
    return "".join(out)


def transform_fdf(src_text: str, *, label: str, gpu: bool,
                  block_size: int, max_scf: int = 5) -> str:
    """Return the benchmark variant of ``src_text``.

    Common to both: relabel to ``label``, cap SCF at ``max_scf`` AND turn
    off ``SCF.MustConverge`` (so the capped run exits 0 / ``COMPLETED``
    instead of aborting non-zero -> SLURM ``FAILED``; audit 2026-06-27
    B-BENCH-1), force a cold start (``DM.UseSaveDM .false.``), zero any MD
    step count (single-point), set ``BlockSize``.  GPU adds
    ``Diag.Algorithm ELPA-1STAGE`` + ``Diag.ELPA.GPU .true.``; CPU strips
    any such directive so it runs a plain ``diagon``.  All edits are
    SIESTA-label-normalized, so variant-spelled input is replaced in
    place, never duplicated.
    """
    t = src_text
    t = _set_or_append(t, "SystemName", label)
    t = _set_or_append(t, "SystemLabel", label)
    t = _set_or_append(t, "MaxSCFIterations", str(max_scf))
    t = _set_or_append(t, "SCF.MustConverge", ".false.")
    t = _set_or_append(t, "DM.UseSaveDM", ".false.")
    t = _set_or_append(t, "MD.NumCGsteps", "0", only_if_present=True)
    t = _set_or_append(t, "BlockSize", str(block_size))
    if gpu:
        t = _set_or_append(t, "Diag.Algorithm", "ELPA-1STAGE")
        t = _set_or_append(t, "Diag.ELPA.GPU", ".true.")
    else:
        t = _remove_directive(t, "Diag.ELPA.GPU")
        t = _remove_directive(t, "Diag.ELPA.UseGPU")
        t = _remove_directive(t, "Diag.Algorithm")
    return t


# --------------------------------------------------------------------- #
#  GPU sweep helper + README                                            #
# --------------------------------------------------------------------- #


def render_gpu_sweep_helper(gpus_per_node: int, cores_per_socket: int) -> str:
    """Render the self-contained ``job-gpu-sweep.sh`` (stdlib bash, no
    molbuilder needed on the cluster).

    The node topology (GPUs/node + cores/socket) is baked in at generate
    time; the helper COMPUTES the valid ``-n``/``-c``/``--gres`` for each
    (G, K) and the README EXPLAINS the relationship.
    """
    ks = " ".join(str(k) for k in _SWEEP_KS)
    return f"""\
#!/usr/bin/env bash
# job-gpu-sweep.sh -- GPU benchmark sweep helper (self-contained).
# Generated by `molbuilder bench generate`.  Runs on the cluster with NO
# molbuilder install -- it only computes sbatch args and (optionally)
# submits job-gpu.sbatch.
#
# Node topology baked in at generate time:
#     GPUS_PER_NODE   = {gpus_per_node}
#     CORES_PER_SOCKET= {cores_per_socket}
#
# Two knobs, both overridden on the sbatch CLI (the launcher auto-adapts
# at runtime -- slurm-integration.md 7.3):
#     G = GPUs           -> --gres=gpu:a100:G
#     K = MPI ranks/GPU  -> -n (K*G)   (launcher derives K=ranks/GPU,
#                                        enables MPS when K>=2)
#     OMP width per rank -> -c (CORES_PER_SOCKET/K)   [K*c <= cores/socket]
#
# Usage:
#     ./job-gpu-sweep.sh            # print the valid G x K matrix
#     ./job-gpu-sweep.sh <G> <K>    # print the one sbatch line for (G,K)
#     ./job-gpu-sweep.sh --submit   # submit the whole valid sweep
set -u
GPUS_PER_NODE={gpus_per_node}
CORES_PER_SOCKET={cores_per_socket}
KS="{ks}"
SBATCH_FILE="job-gpu.sbatch"

# emit <G> <K>: print the sbatch line (or an INVALID comment); rc=1 if invalid.
emit() {{
  local G="$1" K="$2"
  local n=$(( K * G ))
  local c=$(( CORES_PER_SOCKET / K ))
  if [ "$c" -lt 1 ] || [ "$K" -gt "$CORES_PER_SOCKET" ]; then
    printf '# G=%s K=%s  INVALID: K=%s exceeds cores/socket=%s\\n' \\
           "$G" "$K" "$K" "$CORES_PER_SOCKET"
    return 1
  fi
  local note=""
  if [ "$G" -ge 2 ]; then
    note="   # multi-GPU: no NCCL -- MEASURE, do not assume speedup; do NOT add --gpu-bind"
  fi
  printf 'sbatch --gres=gpu:a100:%s -n %s -c %s %s%s\\n' \\
         "$G" "$n" "$c" "$SBATCH_FILE" "$note"
}}

if [ "$#" -eq 0 ]; then
  echo "# GPU sweep matrix (G=GPUs, K=ranks/GPU).  Paste a line to submit:"
  for G in $(seq 1 "$GPUS_PER_NODE"); do
    for K in $KS; do emit "$G" "$K"; done
  done
  echo "# './job-gpu-sweep.sh --submit' submits the whole valid sweep."
  exit 0
fi

if [ "$1" = "--submit" ]; then
  for G in $(seq 1 "$GPUS_PER_NODE"); do
    for K in $KS; do
      line="$(emit "$G" "$K")" || continue
      cmd="${{line%%   #*}}"          # strip trailing note comment
      echo "+ $cmd"
      eval "$cmd"
    done
  done
  exit 0
fi

if [ "$#" -eq 2 ]; then emit "$1" "$2"; exit "$?"; fi
echo "usage: $0 [--submit] | [<G> <K>]" >&2
exit 2
"""


def render_readme(*, cpu_np: int, gpu_gpus: int, gpu_k: int,
                  gpus_per_node: int, cores_per_socket: int,
                  max_scf: int) -> str:
    """Render the bundle ``README.md`` (explains both bundles + helper)."""
    gpu_c = max(1, cores_per_socket // gpu_k)
    return f"""\
# CPU-vs-GPU benchmark bundle

Generated by `molbuilder bench generate`.  Two self-contained, **cold**
(`MaxSCFIterations {max_scf}`, `SCF.MustConverge .false.`,
`DM.UseSaveDM .false.`, MD steps zeroed) SIESTA jobs built from the same
input fdf so you can compare CPU and GPU on this system and pick the
mechanism for your production run.  molbuilder makes **no automatic
recommendation** -- each job just reports its own wall time; you decide.

`SCF.MustConverge .false.` matters: it lets the deliberately-capped run
exit cleanly (`COMPLETED`) instead of aborting non-zero -- otherwise SLURM
marks every bench point `FAILED` and the accounting (`sacct MaxRSS`) is
unreliable.

## Self-contained on the target (no molbuilder install)

This bundle ships its own prep-lib (`mbbench/`) + executable drivers, so
the whole detect -> run -> decide flow runs on the target with nothing
installed:

```
./prep-bench        # detect this machine -> environment.json + a
                    #   topology-sized job-gpu-sweep.sh (each point isolated
                    #   in its own point-G<g>K<k>/ dir)
bash job-gpu-sweep.sh   # run the sweep (sbatch per point on SLURM;
                        #   sequential on a workstation)
./bench-summarize   # parse the points -> bench-result.json (ranked winner)
./prep-run --script-base <your-prod>   # winner -> run-production.sh,
                                       #   re-resolved for THIS machine
```

`prep-bench` accepts `--cores-per-socket` / `--gpus-per-node` / `--gpu-type`
/ `--scheduler` overrides when detection can't see the compute node. See
`docs/protocols/benchmark-workflow.md` for the full design.

Everything except the handful of flipped directives is **byte-for-byte
your input fdf**.

## job-cpu  -- plain `diagon` (`molbuilder-siesta`)

```
cd <this dir>
sbatch job-cpu.sbatch                 # default -n {cpu_np}
sbatch -n 32 job-cpu.sbatch           # try another rank count
```

`--mem` is auto-estimated from the system size (molbuilder/siesta/memory.py);
override with `sbatch --mem=<N>G`.  Scale with `-n`; the launcher reads
`SLURM_NTASKS` so `-n` and the `mpirun -np` agree by construction.

## job-gpu  -- ELPA-CUDA (`molbuilder-siesta-gpu`)

```
sbatch job-gpu.sbatch                 # default G={gpu_gpus} GPU, K={gpu_k} ranks/GPU (-n {gpu_k*gpu_gpus} -c {gpu_c})
./job-gpu-sweep.sh                    # print the valid G x K grid as sbatch lines
./job-gpu-sweep.sh 2 4                # the one line for 2 GPUs, 4 ranks/GPU
./job-gpu-sweep.sh --submit           # submit the whole valid sweep
```

Two knobs (both sbatch-CLI overrides; the launcher auto-adapts):

| knob | sbatch arg | meaning |
|------|------------|---------|
| **G** | `--gres=gpu:a100:G` | number of GPUs |
| **K** | `-n (K*G)` | MPI ranks **per GPU** (launcher derives K, enables MPS when K>=2) |
| OMP | `-c (cores_per_socket/K)` | threads/rank; keep `K*c <= {cores_per_socket}` (cores/socket) |

The sweep spans `K = 1,2,4,8,16`.  At **K=16** the OMP width is `c=1`
(OMP off) -- pure rank-parallelism via MPS, a distinct regime from the
ELPA host-OMP points; measure it to see whether more ranks/GPU still help
or have turned over.  `K = cores/socket` ({cores_per_socket}) is the
ceiling (c=1); beyond it `c` would be 0 (invalid).

**Caveats** (baked from this node's topology: {gpus_per_node} GPUs/node,
{cores_per_socket} cores/socket):

* **Multi-GPU (G>=2) is NOT guaranteed to be faster** -- ELPA-CUDA has no
  NCCL inter-GPU path here.  Measure; do not assume.
* **Do NOT add `--gpu-bind` for G>=2** -- it conflicts with the per-rank
  CUDA_VISIBLE_DEVICES launcher (slurm-integration.md 7.5.2) and breaks
  the K-ranks load balance.
* Cross-socket GPU/CPU placement bites harder at G>=2; for the cleanest
  numbers run the GPU job `--exclusive`.

## Reading the result

Each job stamps every `scf:` line into `<job>-run0.scf-timing.log`.  The
reliable metric is **total wall / N** (`slurm-integration.md` 11.0); the
background monitor reports the live running average and stays quiet when
stalled (11.0c).  Compare `job-cpu` vs the best `job-gpu` point and run
your production job with that mechanism.
"""


# --------------------------------------------------------------------- #
#  bundle generator                                                    #
# --------------------------------------------------------------------- #


def generate_bench_bundle(fdf_path, out_dir=None, *,
                          cpu_np: int = 64,
                          gpu_gpus: int = 1,
                          gpu_k: int = 4,
                          gpus_per_node: int = 4,
                          cores_per_socket: int = 24,
                          cpu_block_size: int = 8,
                          gpu_block_size: int = 256,
                          max_scf: int = 5,
                          cpu_time: Optional[str] = None,
                          gpu_time: Optional[str] = None
                          ) -> Tuple[Path, List[Path]]:
    """Generate the CPU + GPU benchmark bundle from ``fdf_path``.

    Returns ``(out_dir, [written paths])``.  The ``.sbatch`` files are
    emitted only when a ``scheduler`` block is configured (otherwise just
    the ``.run.sh`` launchers, per slurm-integration.md § 10).

    ``cpu_time`` / ``gpu_time`` set the per-bundle ``#SBATCH -t`` (else the
    scheduler default).  CPU diagon at a few-hundred atoms can exceed the
    4 h site default before finishing the capped iters -- pass a longer
    ``cpu_time`` (audit 2026-06-27 B-BENCH-2).
    """
    fdf_path = Path(fdf_path).resolve()
    if fdf_path.suffix.lower() != ".fdf":
        raise ValueError(f"input must be a .fdf, got {fdf_path.name!r}")
    src_text = fdf_path.read_text(encoding="utf-8")
    src_dir = fdf_path.parent

    out_dir = (Path(out_dir).resolve() if out_dir is not None
               else src_dir / f"{fdf_path.stem}.bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    # Pseudopotentials must sit beside the job fdfs (the run uses them and
    # the CPU --mem estimator reads valence configs from .psml).  Copy the
    # modern .psml AND the legacy formats SIESTA also accepts (.psf/.vps)
    # so a non-.psml project still produces a runnable bundle (B-3).
    for pat in ("*.psml", "*.psf", "*.vps", "*.psp8"):
        for pseudo in sorted(src_dir.glob(pat)):
            dst = out_dir / pseudo.name
            if pseudo.resolve() != dst.resolve():
                shutil.copy2(pseudo, dst)
            written.append(dst)

    cpu_fdf = out_dir / "job-cpu.fdf"
    cpu_fdf.write_text(transform_fdf(
        src_text, label="job-cpu", gpu=False,
        block_size=cpu_block_size, max_scf=max_scf), encoding="utf-8")
    written.append(cpu_fdf)

    gpu_fdf = out_dir / "job-gpu.fdf"
    gpu_fdf.write_text(transform_fdf(
        src_text, label="job-gpu", gpu=True,
        block_size=gpu_block_size, max_scf=max_scf), encoding="utf-8")
    written.append(gpu_fdf)

    # CPU launcher + sbatch: scales with -n; --mem auto-estimated.
    written.append(write_run_wrapper(cpu_fdf, mpi_np=cpu_np, time=cpu_time))

    # GPU launcher + sbatch: G GPUs (--gres) x K ranks/GPU (-n=K*G);
    # -c = cores/socket / K so K*c stays within one socket.
    gpu_c = max(1, cores_per_socket // gpu_k)
    written.append(write_run_wrapper(
        gpu_fdf, mpi_np=gpu_k * gpu_gpus,
        gres=f"a100:{gpu_gpus}", cpus_per_task=gpu_c, time=gpu_time))

    # write_run_wrapper returns only the .run.sh; pick up the .sbatch it
    # emits (when a scheduler is configured) + the shipped monitor so the
    # caller's file listing is complete.
    for extra in (out_dir / "job-cpu.sbatch", out_dir / "job-gpu.sbatch",
                  out_dir / "mb_monitor.py"):
        if extra.is_file() and extra not in written:
            written.append(extra)

    sweep = out_dir / "job-gpu-sweep.sh"
    sweep.write_text(render_gpu_sweep_helper(gpus_per_node, cores_per_socket),
                     encoding="utf-8")
    sweep.chmod(0o755)
    written.append(sweep)

    readme = out_dir / "README.md"
    readme.write_text(render_readme(
        cpu_np=cpu_np, gpu_gpus=gpu_gpus, gpu_k=gpu_k,
        gpus_per_node=gpus_per_node, cores_per_socket=cores_per_socket,
        max_scf=max_scf), encoding="utf-8")
    written.append(readme)

    # Ship the self-contained prep-lib so the target needs NO molbuilder
    # install (the on-target prep-bench/summarize/prep-run drivers).
    written.extend(_ship_prep_lib(out_dir))

    return out_dir, written


# Stdlib-only modules that form the shipped on-target prep-lib, and the
# executable shims that expose their main()s.  Copied VERBATIM (they use
# package-relative imports, so they work the same inside the shipped
# ``mbbench`` package as inside ``molbuilder.bench``).
_PREP_LIB_MODULES = ("environment", "adapters", "result", "prep",
                     "summarize", "prep_run")
_PREP_SHIMS = {"prep-bench": "prep", "bench-summarize": "summarize",
               "prep-run": "prep_run"}


def _ship_prep_lib(out_dir: Path) -> List[Path]:
    """Copy the stdlib-only prep-lib into ``<out>/mbbench/`` + write the
    ``prep-bench`` / ``bench-summarize`` / ``prep-run`` shims, so the
    bundle runs the whole on-target workflow with no molbuilder install
    (benchmark-workflow.md § 4.7 self-contained rule)."""
    src = Path(__file__).resolve().parent
    written: List[Path] = []

    pkg = out_dir / "mbbench"
    pkg.mkdir(exist_ok=True)
    init = pkg / "__init__.py"
    init.write_text(
        '"""Self-contained molbuilder benchmark prep-lib (stdlib-only).\n\n'
        "Copied verbatim by `molbuilder bench generate` so the target can "
        "run\nprep-bench / bench-summarize / prep-run with no molbuilder "
        'install."""\n', encoding="utf-8")
    written.append(init)
    for m in _PREP_LIB_MODULES:
        dst = pkg / f"{m}.py"
        shutil.copy2(src / f"{m}.py", dst)
        written.append(dst)

    for shim, mod in _PREP_SHIMS.items():
        p = out_dir / shim
        p.write_text(
            "#!/usr/bin/env python3\n"
            "# self-contained entry: adds this bundle dir to sys.path so the\n"
            "# shipped mbbench package imports without any molbuilder install.\n"
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            f"from mbbench.{mod} import main\n"
            "sys.exit(main())\n", encoding="utf-8")
        p.chmod(0o755)
        written.append(p)

    return written


__all__ = [
    "transform_fdf",
    "render_gpu_sweep_helper",
    "render_readme",
    "generate_bench_bundle",
]
