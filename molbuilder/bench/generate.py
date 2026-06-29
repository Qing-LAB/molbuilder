"""``molbuilder bench generate`` -- CPU-only + GPU-only benchmark bundles.

Generalises the hand-built ``sol-bench-BDT-Au111-hex`` bundle: from ONE
input ``.fdf`` it emits two self-contained, comparable benchmark jobs so
the user can measure CPU vs GPU on their own system and decide which
mechanism the production run should use (molbuilder makes NO automatic
recommendation -- each bundle just reports its own wall/iter).

Both points use the SAME eigensolver (``Diag.Algorithm ELPA-1STAGE``) so
the CPU-vs-GPU number isolates the *hardware*, not the solver
(slurm-integration.md § 11); only the GPU point adds ``Diag.ELPA.GPU .true.``.
ELPA lives only in ``molbuilder-siesta-gpu`` (the precompiled
``molbuilder-siesta`` has no ELPA), so BOTH bundles run there -- the CPU
one declares that env EXPLICITLY (visible as "Target env" in its
``.run.sh``), not by silent re-routing.

  * ``job-cpu.{fdf,run.sh,sbatch}`` -- ELPA-1STAGE, no ``Diag.ELPA.GPU``;
    env ``molbuilder-siesta-gpu`` (explicit).
  * ``job-gpu.{fdf,run.sh,sbatch}`` -- ELPA-1STAGE + ``Diag.ELPA.GPU
    .true.``; env ``molbuilder-siesta-gpu`` (auto-routed by the GPU flag).

Both are made COLD and comparable (``MaxSCFIterations 5``,
``DM.UseSaveDM .false.``, MD/relaxation steps zeroed) -- everything else
is copied verbatim from the input fdf.  Tuning is done at submit time via
sbatch args (the launcher auto-adapts, slurm-integration.md § 7.3): the
CPU job scales with ``-n <np>``; the GPU job with ``--gres=gpu:a100:G``
(GPUs) x ``-n (K*G)`` (K ranks/GPU) x ``-c (cores_per_socket/K)``.  The
GPU **sweep** over (G, K) is written on the target by ``prep-bench``
(``adapters.format_bench``), sized to the detected topology with each
point isolated in its own dir; ``bench generate`` only bakes a
placeholder that reminds the user to prep first.

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

import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from .. import runtime_config as _rc
from ..runwrap import write_run_wrapper


def _write_activation_config(out_dir: Path, activation: Optional[str],
                             preamble: Optional[str]) -> Path:
    """Merge ``script_generation.{activation,preamble}`` into the bundle's
    ``.molbuilder.json`` (creating the file, preserving anything else in
    it).  Only the fields that are given are written (a ``None`` leaves any
    existing value untouched).  This is the SINGLE place the activation
    decision is persisted: both the generate-time explicit-flag path and the
    prep-time target resolver (:func:`bake_target_wrappers`) write through
    here, so the on-disk shape is byte-identical regardless of which moment
    wrote it.  Returns the config path.  (job-execution.md § 7.3.)
    """
    cfg_path = out_dir / ".molbuilder.json"
    existing: dict = {}
    if cfg_path.is_file():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
    if not isinstance(existing, dict):           # malformed top-level JSON
        existing = {}
    sg_block = dict(existing.get("script_generation") or {})
    if activation is not None:
        sg_block["activation"] = activation
    if preamble:
        sg_block["preamble"] = preamble
    existing["script_generation"] = sg_block
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n",
                        encoding="utf-8")
    return cfg_path


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
    step count (single-point), set ``BlockSize``.

    **Both points use the SAME solver, ``Diag.Algorithm ELPA-1STAGE``** —
    only the GPU point adds ``Diag.ELPA.GPU .true.``.  So the CPU-vs-GPU
    number isolates the *hardware* (the CUDA toggle), not *solver* —
    ScaLAPACK-CPU vs ELPA-GPU would conflate the two
    (slurm-integration.md § 11).  ELPA lives only in ``molbuilder-siesta-gpu``
    (the CPU ``molbuilder-siesta`` conda package is built without it), so
    BOTH points route there: the CPU point declares that env EXPLICITLY
    (``env=`` to ``write_run_wrapper``), the GPU point auto-routes via its
    ``Diag.ELPA.GPU`` flag (``runwrap._fdf_requests_gpu``).  All edits are
    SIESTA-label-normalized, so variant-spelled input is replaced in place,
    never duplicated.
    """
    t = src_text
    t = _set_or_append(t, "SystemName", label)
    t = _set_or_append(t, "SystemLabel", label)
    t = _set_or_append(t, "MaxSCFIterations", str(max_scf))
    t = _set_or_append(t, "SCF.MustConverge", ".false.")
    t = _set_or_append(t, "DM.UseSaveDM", ".false.")
    t = _set_or_append(t, "MD.NumCGsteps", "0", only_if_present=True)
    t = _set_or_append(t, "BlockSize", str(block_size))
    # Same eigensolver on both points; only the CUDA toggle differs.
    t = _set_or_append(t, "Diag.Algorithm", "ELPA-1STAGE")
    if gpu:
        t = _set_or_append(t, "Diag.ELPA.GPU", ".true.")
    else:
        t = _remove_directive(t, "Diag.ELPA.GPU")
        t = _remove_directive(t, "Diag.ELPA.UseGPU")
    return t


# --------------------------------------------------------------------- #
#  GPU sweep helper + README                                            #
# --------------------------------------------------------------------- #


def render_sweep_placeholder() -> str:
    """A PLACEHOLDER ``job-gpu-sweep.sh``: it can't be the real sweep at
    generate time (the real one needs the TARGET's detected scheduler +
    topology + per-point isolation, which only ``prep-bench`` knows).  So
    the generated file just reminds the user to prep first; running
    ``./prep-bench`` OVERWRITES it with the real, isolated sweep
    (``adapters.format_bench``).  ``environment.json`` is the "prepared"
    flag.
    """
    return """\
#!/usr/bin/env bash
# job-gpu-sweep.sh -- PLACEHOLDER (not yet prepared for this machine).
#
# The real sweep depends on THIS machine's scheduler + topology, which is
# detected on the target -- so it can't be baked at generate time.
set -u
echo "This benchmark sweep has not been prepared for this machine." >&2
echo "Run ./prep-bench first: it detects the scheduler + topology, writes" >&2
echo "environment.json, and OVERWRITES this file with the real, per-point-" >&2
echo "isolated sweep sized to this node.  Then re-run: bash job-gpu-sweep.sh" >&2
exit 1
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

## Portable bundle -- prep on the target

This bundle is **target-neutral**: the run wrappers are NOT baked yet.  Copy
it to the machine it will run on, then prep THERE -- prep detects the machine,
resolves the conda activation for it, and bakes the standalone
`job-{{cpu,gpu}}.run.sh` (+ `.sbatch` on SLURM).  The SAME bundle works on a
workstation and on an HPC node; no per-target regeneration.

```
molbuilder bench prep        # detect machine + resolve activation + BAKE the
                             #   run wrappers + write a topology-sized
                             #   job-gpu-sweep.sh (each point isolated in its
                             #   own point-G<g>K<k>/ dir)
molbuilder envs doctor       # verify the env is ready before you submit
bash job-gpu-sweep.sh        # run the sweep (sbatch per point on SLURM;
                             #   sequential on a workstation)
./bench-summarize            # parse the points -> bench-result.json (winner)
./prep-run --script-base <your-prod>   # winner -> run-production.sh
```

- **Workstation:** `molbuilder bench prep` autodetects `conda activate`.
- **HPC:** ship a `.molbuilder.json` with `script_generation`
  (`preamble: module load mamba`, `activation: source activate`) in the bundle
  -- the job runs in a clean shell, so the toolchain can't be autodetected.

`molbuilder bench prep` accepts `--cores-per-socket` / `--gpus-per-node` /
`--gpu-type` / `--scheduler` overrides when detection can't see the compute
node.  (The shipped stdlib `./prep-bench` does detection + the sweep only; it
does not bake the run wrappers.)  See
`docs/protocols/benchmark-workflow.md` and `docs/job-execution.md § 7`.

Everything except the handful of flipped directives is **byte-for-byte
your input fdf**.

Both points use the SAME solver (`Diag.Algorithm ELPA-1STAGE`); only job-gpu
adds `Diag.ELPA.GPU`.  So the comparison is HARDWARE, not solver.  ELPA lives
only in `molbuilder-siesta-gpu`, so BOTH run there -- the `.run.sh` shows the
`Target env`.  This is a setup for YOU to measure and decide; molbuilder makes
no recommendation.

## job-cpu  -- ELPA on CPU (`molbuilder-siesta-gpu`)

```
cd <this dir>
sbatch job-cpu.sbatch                 # default -n {cpu_np}
sbatch -n 32 job-cpu.sbatch           # try another rank count
```

`--mem` is auto-estimated from the system size (molbuilder/siesta/memory.py);
override with `sbatch --mem=<N>G`.  Scale with `-n`; the launcher reads
`SLURM_NTASKS` so `-n` and the `mpirun -np` agree by construction.

## job-gpu  -- ELPA-CUDA (`molbuilder-siesta-gpu`)

A single GPU point (default G={gpu_gpus} GPU, K={gpu_k} ranks/GPU ->
`-n {gpu_k*gpu_gpus} -c {gpu_c}`):

```
sbatch job-gpu.sbatch
```

For the **GPU sweep** across (G, K), run `./prep-bench` first (it detects
this machine and writes the real, per-point-isolated `job-gpu-sweep.sh`),
then `bash job-gpu-sweep.sh`.  The knobs:

| knob | meaning |
|------|---------|
| **G** | number of GPUs (`--gres=gpu:<type>:G`) |
| **K** | MPI ranks **per GPU** (`-n = K*G`; launcher derives K, MPS when K>=2) |
| OMP | threads/rank = `cores_per_socket / K` (full-socket); prep-bench picks the K values that DIVIDE cores/socket so no cores sit idle |

**Caveats:**

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
                          gpu_time: Optional[str] = None,
                          cpu_cpus_per_task: int = 1,
                          gpu_exclusive: Optional[bool] = None,
                          activation: Optional[str] = None,
                          preamble: Optional[str] = None,
                          echo=None
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

    # Activation is resolved + baked at PREP time, on the target
    # (job-execution.md § 3.2 / § 7): ONE portable bundle then runs on a
    # workstation (prep autodetects conda) AND an HPC node (prep reads the
    # shipped config).  So generate does NOT autodetect the host's conda and
    # does NOT bake a target-locked wrapper.  It only PERSISTS an explicit
    # --activation / --preamble (the HPC case, § 4.4.2) for prep to consume;
    # with neither, the bundle stays target-neutral and prep decides on the
    # machine it will run on.
    if activation or preamble:
        cfg = _write_activation_config(out_dir, activation, preamble)
        if echo is not None:
            echo(f"activation/preamble written to {cfg.name} "
                 f"(used by `molbuilder bench prep` on the target).")

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

    # The run wrappers (.run.sh / .sbatch) are baked at PREP, on the target
    # (job-execution.md § 3.2 / § 7), so ONE bundle is portable.  generate
    # records the benchmark KNOBS it was asked for in a manifest; `molbuilder
    # bench prep` reads it, resolves activation for the target, and bakes the
    # wrappers via runwrap.write_run_wrapper -- with `gres` / `-c` derived
    # from the DETECTED gpu-type + cores there, not guessed here.
    #
    # CPU is MPI-only (1 core/rank); both points run ELPA-1STAGE in
    # molbuilder-siesta-gpu (ELPA lives only there) -- the env is declared
    # at bake time, not silently re-routed.  GPU: G GPUs x K ranks/GPU.
    # ``engine`` + per-job ``script`` are engine-NEUTRAL on purpose: the
    # job-execution layer (prep, bake, runwrap) dispatches by file extension
    # (.fdf -> siesta, .py -> pyscf), so a future PySCF bench reuses this
    # exact schema with engine="pyscf" + script="job-cpu.py" -- no special
    # casing.  Keep the key "script", not "fdf" (job-execution.md § 7.3).
    manifest = out_dir / "bench-manifest.json"
    manifest.write_text(json.dumps({
        "schema": "molbuilder/bench-manifest@1",
        "engine": "siesta",
        "jobs": {
            "cpu": {"script": cpu_fdf.name, "mpi_np": cpu_np,
                    "cpus_per_task": cpu_cpus_per_task, "time": cpu_time},
            "gpu": {"script": gpu_fdf.name, "gpu_gpus": gpu_gpus,
                    "gpu_k": gpu_k, "time": gpu_time,
                    "exclusive": gpu_exclusive},
        },
    }, indent=2) + "\n", encoding="utf-8")
    written.append(manifest)

    # The real sweep is written by `./prep-bench` on the target (it needs
    # the detected scheduler + topology + per-point isolation).  Bake a
    # placeholder that reminds the user to prep first; prep-bench
    # overwrites it.
    sweep = out_dir / "job-gpu-sweep.sh"
    sweep.write_text(render_sweep_placeholder(), encoding="utf-8")
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


def bake_target_wrappers(out_dir, env, *, echo=None) -> List[Path]:
    """Resolve the conda activation for THIS target and bake the run
    wrappers from the bundle's ``bench-manifest.json`` (job-execution.md
    § 7.4).  This is the **prep-time** step that makes one bundle portable;
    it runs in the molbuilder env the contract guarantees on every target
    (§ 3.4), while the stdlib ``prep-bench`` does only detection + the sweep.

    The detected ``env.scheduler`` gates activation resolution on BOTH
    branches (NOT a config>autodetect precedence, which would bake a shipped
    HPC activation on a workstation -- § 7.5):

      * **workstation** -> autodetect conda on PATH and write it into this
        dir's ``.molbuilder.json`` (specialising the copy for this machine);
      * **slurm / HPC** -> the activation MUST be the shipped explicit config
        (the job runs in a clean shell; nothing to autodetect -- § 3.3 row M).

    Then :func:`runwrap.write_run_wrapper` bakes the standalone ``.run.sh`` /
    ``.sbatch`` (it reads the activation from the config we just ensured).
    ``gres`` / ``-c`` come from the DETECTED topology, not a generate guess.
    Returns the written wrapper paths (+ ``.sbatch`` / ``mb_monitor.py``).
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "bench-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path.name} not found in {out_dir} -- run "
            f"`molbuilder bench generate` first to produce the bundle.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]

    # --- resolve activation for this target (scheduler-gated) ---
    if env.scheduler == "workstation":
        det = _rc.detect_conda_activation()
        if det is None:
            raise _rc.RuntimeConfigError(
                "no conda/mamba found on PATH on this workstation.\n"
                "Activate your conda, or ship an explicit script_generation "
                "in .molbuilder.json (see docs/job-execution.md § 4.4.1).")
        _write_activation_config(out_dir, det["activation"], det["preamble"])
        if echo is not None:
            echo(f'activation: autodetected "{det["activation"]}" '
                 f"for this workstation.")
    else:                                            # slurm / HPC
        sg = _rc.get_script_generation(project_dir=out_dir)
        if not sg.get("activation"):
            raise _rc.RuntimeConfigError(
                "HPC target: no activation configured.  Ship a "
                ".molbuilder.json with script_generation (preamble + "
                "activation) in the bundle -- see docs/job-execution.md "
                "§ 4.4.2.  The job runs in a clean shell, so the toolchain "
                "cannot be autodetected (§ 3.3 row M).")
        if echo is not None:
            echo(f'activation: "{sg["activation"]}" from the shipped config.')

    # --- bake the standalone wrappers (reuses write_run_wrapper) ---
    from ..diagnostics import get_capabilities
    elpa_env = get_capabilities().env_for_category("siesta-gpu")

    # Emit .sbatch only when the DETECTED scheduler is SLURM.  Otherwise a
    # bundle carrying an HPC scheduler block but prepped on a workstation
    # would emit stray SLURM .sbatch files that don't match the machine
    # (write_run_wrapper keys sbatch emission off the config block; the
    # detected scheduler is the truth -- § 7.5).
    emit_sbatch = env.scheduler == "slurm"

    written: List[Path] = []
    cpu = jobs["cpu"]
    written.append(write_run_wrapper(
        out_dir / cpu["script"], mpi_np=cpu["mpi_np"],
        cpus_per_task=cpu["cpus_per_task"], time=cpu.get("time"),
        env=elpa_env, emit_sbatch=emit_sbatch))

    gpu = jobs["gpu"]
    cores = env.topology.cores_per_socket or 24
    gtype = env.topology.gpu_type or "a100"
    gpu_c = max(1, cores // gpu["gpu_k"])
    written.append(write_run_wrapper(
        out_dir / gpu["script"], mpi_np=gpu["gpu_k"] * gpu["gpu_gpus"],
        gres=f"{gtype}:{gpu['gpu_gpus']}", cpus_per_task=gpu_c,
        time=gpu.get("time"), exclusive=gpu.get("exclusive"), env=elpa_env,
        emit_sbatch=emit_sbatch))

    # write_run_wrapper returns only the .run.sh; pick up the .sbatch it
    # emits (when a scheduler is configured) + the shipped monitor.
    for extra in (out_dir / "job-cpu.sbatch", out_dir / "job-gpu.sbatch",
                  out_dir / "mb_monitor.py"):
        if extra.is_file() and extra not in written:
            written.append(extra)
    return written


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
    "render_sweep_placeholder",
    "render_readme",
    "generate_bench_bundle",
]
