"""``molbuilder bench`` CLI group."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from ..diagnostics import get_capabilities
from . import (
    DEFAULT_POINTS, Point, parse_bench_marks,
    _resolve_project, _prepare_point_dir, run_point,
    write_results_csv,
)


@click.group("bench",
             context_settings={"help_option_names": ["-h", "--help"]})
def bench_group() -> None:
    """Benchmark CPU vs GPU on YOUR machine, then run production with the
    winner -- no hand-tuning of queue names, core counts, or bindings.

    \b
    WORKFLOW (run `molbuilder bench <cmd> --help` for each):
      generate    HOST    one .fdf -> a portable, self-contained bundle
      prep        TARGET  detect the machine -> environment.json + the sweep
      summarize   TARGET  read the sweep's outputs -> bench-result.json
      prep-run    TARGET  bench-result.json -> production run script
      siesta-gpu  (legacy) in-place np/omp/BlockSize sweep on one project

    \b
    KEY TERMS (these trip everyone up -- read once):
      rank   one MPI PROCESS = one running copy of SIESTA.  N ranks = N
             processes computing in parallel, exchanging data by messages.
      -n     the NUMBER OF RANKS (SLURM --ntasks; identical to `mpirun
             -np N`).  NOT the CPU count.
      -c     CPU cores per rank (OpenMP threads inside each process).
      ==>    total CPU cores = -n * -c.
      G      number of GPUs (--gres=gpu:<type>:G).
      K      MPI ranks PER GPU (so -n = K*G); K ranks share one GPU (MPS).
      c      cores per rank of a GPU point = cores_per_socket / K.

    \b
    TUNING the GPU point -- is -n (= K*G) too small or too big?
      * SWEEP K (e.g. `prep --gpu-ks 8,16`): if s/iter keeps dropping, -n
        was too small; if it rises, -n is too big (MPS contention, c->1).
      * the monitor's GPU sm% (util.csv / [UTIL-SUMMARY]; in
        bench-result.json as `bound`): sustained high sm% = GPU saturated
        (good); low sm% while cpu% is pegged = host-bound (more ranks
        won't help -- the CPU side feeding ELPA-CUDA is the limit).

    \b
    CPU point: SIESTA CPU is MPI-only, so 1 core/rank (-c 1); -n*-c must
    fit one node.  Scale with `sbatch -n <np>`; override -c per submission.

    \b
    PLACEMENT / GPU<->CPU BINDING (we do NOT hand-craft it):
      --gpu-exclusive   GPU job takes the WHOLE node (default, from config).
                        Clean timing; lets the launcher pin each rank to its
                        GPU's own socket (it owns all cores).
      --no-gpu-exclusive  pack jobs; placement is left to SLURM (no pin).
      We never use --gpu-bind (it conflicts with the per-rank launcher) and
      under a non-exclusive SLURM alloc we TRUST the scheduler's cpuset.
      Runtime: `MB_NO_SOCKET_PIN=1 sbatch job-gpu.sbatch` disables the
      auto socket-pin -- A/B it to see if the pin actually helps.
    """


@bench_group.command("siesta-gpu",
                      short_help="sweep np/omp/BlockSize on a SIESTA-GPU project")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False,
                                                resolve_path=True))
@click.option("--points", "points_str", default=None,
              help='space-separated 3/4/5-tuples, e.g. '
                   '"4,2,64 4,2,64,2s 20,1,64,1s,nopin".  Fields are '
                   'np,omp,bs[,diag[,pin]].  diag: "1s"/"2s" or '
                   '"ELPA-1STAGE"/"ELPA-2STAGE".  pin: "pin"/"nopin" '
                   '(default pin = GPU socket).  Default: 18-point sweep '
                   '(9 shapes × ELPA-1/2STAGE) covering all-cores '
                   'cross-socket + np10 big-block + ELPA 4-rank anchor '
                   '+ host-bound stress.  See docs/protocols/script-contract.md.')
@click.option("--iters", "iters", type=int, default=5, show_default=True,
              help="MaxSCFIterations cap per test point.")
@click.option("--cold", "cold", is_flag=True,
              help="don't carry over .DM/.CG/.XV warm-start from the project.")
@click.option("--verbose", "-v", "verbose", is_flag=True,
              help="show SIESTA stdout/stderr per point (slower output but "
                   "useful when debugging a failing run).")
def cmd_siesta_gpu(project_dir: str,
                    points_str: Optional[str],
                    iters: int,
                    cold: bool,
                    verbose: bool) -> None:
    """Sweep BlockSize / np / omp combinations on a SIESTA-GPU project.

    The project directory must contain a molbuilder-generated .fdf
    + .run.sh pair.  Refuses cleanly when the .fdf has no BENCH-MARKS
    block -- regenerate with the current molbuilder to add one.

    Output: per-point subdirectories under <basename>.bench/, plus a
    ``results.csv`` summarising wall time and effective values.
    """
    proj = Path(project_dir)

    # Gate on siesta-gpu env presence.
    caps = get_capabilities()
    siesta_gpu_env = caps.env_for_category("siesta-gpu")
    if siesta_gpu_env is None or not caps.env_available(siesta_gpu_env):
        click.echo(
            "ERROR: the `molbuilder-siesta-gpu` env is not present.  "
            "Install it first:\n"
            "    molbuilder envs install molbuilder-siesta-gpu",
            err=True,
        )
        raise SystemExit(2)

    # Resolve project artefacts.
    try:
        fdf_path, runsh_path, basename = _resolve_project(proj)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(2)

    fdf_text   = fdf_path.read_text(encoding="utf-8")
    runsh_text = runsh_path.read_text(encoding="utf-8")

    marks = parse_bench_marks(fdf_text)
    if marks is None:
        click.echo(
            f"ERROR: the .fdf at {fdf_path} has no BENCH-MARKS block.\n"
            f"  This contract block landed 2026-06-16; re-generate the\n"
            f"  script with the current molbuilder to add it.",
            err=True,
        )
        raise SystemExit(2)

    # Resolve the point grid.
    if points_str:
        try:
            points: List[Point] = [Point.parse(s) for s in points_str.split()]
        except ValueError as e:
            click.echo(f"ERROR: --points: {e}", err=True)
            raise SystemExit(2)
    else:
        points = list(DEFAULT_POINTS)

    # Plan banner.
    click.echo("==== molbuilder bench siesta-gpu ====")
    click.echo(f"project   : {proj}")
    click.echo(f"fdf       : {fdf_path.name}")
    click.echo(f"wrapper   : {runsh_path.name}")
    click.echo(f"env       : {siesta_gpu_env}")
    click.echo(f"iters     : {iters} per point")
    click.echo(f"cold      : {cold}")
    click.echo(f"sweep     : {len(points)} points")
    for i, p in enumerate(points, 1):
        pin_tag = "pin=GPU-socket" if p.pin else "pin=none (all cores)"
        gpu_tag = "GPU on" if p.gpu else "CPU only"
        click.echo(
            f"            {i}. np={p.np}  omp={p.omp}  bs={p.bs}  "
            f"diag={p.diag}  {pin_tag}  {gpu_tag}"
        )
    click.echo("")

    bench_root = proj / f"{basename}.bench"
    bench_root.mkdir(exist_ok=True)
    results = []
    csv = bench_root / "results.csv"

    # Run each point sequentially -- they share the GPU.
    for i, point in enumerate(points, 1):
        pin_tag = "pinned" if point.pin else "all-cores"
        gpu_tag = "gpu" if point.gpu else "cpu"
        click.echo(
            f"---- [{i}/{len(points)}] np={point.np} omp={point.omp} "
            f"bs={point.bs} diag={point.diag} {pin_tag} {gpu_tag} ----"
        )
        point_dir = _prepare_point_dir(
            proj, basename, fdf_text, runsh_text, point, iters, cold,
        )
        result = run_point(
            point_dir, basename, point,
            siesta_gpu_env=siesta_gpu_env,
            quiet=not verbose,
        )
        results.append(result)
        # Re-write the full CSV after every point so a killed bench
        # still leaves an up-to-date results.csv on disk.
        write_results_csv(results, bench_root)
        click.echo(
            f"     iters={result.iters_done} "
            f"first={result.first_iter_s} s "
            f"avg={result.avg_iter_s} s "
            f"wall={result.walltime_s} s"
            + (f" (mismatch eff_bs={result.effective_bs})"
               if (result.effective_bs is not None
                   and result.effective_bs != point.bs)
               else "")
            + (f" [{result.error}]" if result.error else "")
            + f"  -> {csv.name}"
        )

    # Print sorted summary.
    ranked = sorted(
        results,
        key=lambda r: (r.avg_iter_s if r.avg_iter_s is not None else 1e9),
    )
    click.echo("")
    click.echo("==== results sorted by avg s/iter (winner first) ====")
    header = f"{'point':<22} {'avg/iter':>10} {'wall':>8} {'iters':>6}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in ranked:
        avg = f"{r.avg_iter_s:.1f}" if r.avg_iter_s is not None else "n/a"
        click.echo(
            f"{r.point.slug:<22} {avg:>10} {r.walltime_s:>8} {r.iters_done:>6}"
        )
    click.echo("")
    click.echo(f"csv: {csv}")


_GENERATE_EPILOG = """\
\b
EXAMPLES:
\b
  # workstation: conda is on PATH, so activation is auto-detected -- no config
  molbuilder bench generate input.fdf --out bench/
\b
  # an HPC target (e.g. Sol): tell it the activation it will use THERE
  molbuilder bench generate input.fdf --out bench/ \\
      --activation "source activate" --preamble "module load mamba"
\b
Then ON THE TARGET (the bundle is self-contained -- no molbuilder needed):
  cd bench/
  ./prep-bench --gpu-ks 1,2,4,8     # detect machine -> environment.json + sweep
  bash job-gpu-sweep.sh             #   (SLURM: sbatch job-cpu.sbatch too)
  ./bench-summarize                 # rank CPU vs GPU points -> winner
  ./prep-run --script-base myprod   # winner -> run-production.sh for THIS machine
\b
CONFIG FILE -- .molbuilder.json (explicit; the flags above just write it):
  WHERE: the OUT dir (it travels with the bundle), merged over a server-wide
         ~/.config/molbuilder/molbuilder.json if present -- project wins.
  WHAT : a `scheduler` block (SLURM directives) + a `script_generation` block
         (how to load + activate conda). Workstation: optional (auto-detected).
  HPC example (what --activation/--preamble bake for you):
\b
    {
      "scheduler": { "kind": "slurm",
        "directives": { "partition": "public", "qos": "public",
                        "export": "NONE" } },
      "script_generation": { "preamble": "module load mamba",
                             "activation": "source activate" }
    }
\b
  Full reference: docs/protocols/deployment.md section 3.
\b
K is GPU processes/GPU (np = K*G); --gpu-ks may exceed cores/socket
(oversubscription is allowed + flagged) to find where np stops scaling.
"""


@bench_group.command("generate",
                     short_help="emit CPU-only + GPU-only benchmark bundles "
                                "from one .fdf",
                     epilog=_GENERATE_EPILOG)
@click.argument("fdf", type=click.Path(exists=True, dir_okay=False,
                                       resolve_path=True))
@click.option("--out", "out_dir", default=None,
              type=click.Path(file_okay=False, resolve_path=True),
              help="output directory (default: <fdf-stem>.bench next to the "
                   "input).")
@click.option("--cpu-np", type=int, default=64, show_default=True,
              help="default MPI rank count baked into job-cpu.sbatch "
                   "(override at submit: sbatch -n <np> job-cpu.sbatch).")
@click.option("--gpu-gpus", "gpu_gpus", type=int, default=1, show_default=True,
              help="default GPU count (G) baked into job-gpu.sbatch.")
@click.option("--gpu-k", type=int, default=4, show_default=True,
              help="default MPI ranks per GPU (K); job-gpu gets -n K*G, "
                   "-c cores_per_socket/K.")
@click.option("--gpus-per-node", type=int, default=4, show_default=True,
              help="FALLBACK GPUs/node for the default GPU point; the real "
                   "value + the sweep are detected on the target by "
                   "`bench prep`.")
@click.option("--cores-per-socket", type=int, default=24, show_default=True,
              help="FALLBACK cores/socket (sets the default GPU point's "
                   "-c = cores/K); `bench prep` detects the real value.")
@click.option("--cpu-block-size", type=int, default=8, show_default=True,
              help="ScaLAPACK BlockSize for the CPU bundle.")
@click.option("--gpu-block-size", type=int, default=256, show_default=True,
              help="BlockSize for the GPU (ELPA-CUDA) bundle.")
@click.option("--max-scf", type=int, default=5, show_default=True,
              help="MaxSCFIterations cap (both bundles run cold + capped, "
                   "with SCF.MustConverge .false. so the capped run exits "
                   "0 / COMPLETED).")
@click.option("--cpu-time", default=None,
              help="#SBATCH -t for the CPU bundle (e.g. 0-12:00:00); else "
                   "the scheduler default.  CPU diagon at a few-hundred "
                   "atoms can exceed a 4 h default -- set this generously.")
@click.option("--gpu-time", default=None,
              help="#SBATCH -t for the GPU bundle; else the scheduler "
                   "default.")
@click.option("--cpu-c", "cpu_cpus_per_task", type=int, default=1,
              show_default=True,
              help="cores (OMP threads) per CPU rank; CPU SIESTA is "
                   "MPI-only so 1 is right (-n*-c must fit one node). "
                   "Override per submission with `sbatch -c N`.")
@click.option("--gpu-exclusive/--no-gpu-exclusive", "gpu_exclusive",
              default=None,
              help="whether the GPU job takes the whole node (#SBATCH "
                   "--exclusive).  Default = the scheduler config "
                   "(gpu.exclusive).  --exclusive gives clean timing + "
                   "lets the launcher socket-pin (it owns all cores); "
                   "--no-gpu-exclusive packs jobs but the placement is "
                   "SLURM's (no pin).")
@click.option("--activation", default=None,
              help="conda activation for the run scripts, e.g. "
                   "'conda activate' or 'source activate'.  Default: an "
                   "existing molbuilder.json config, else auto-detected "
                   "from the local conda. Set this for a different run "
                   "target (e.g. an HPC cluster).")
@click.option("--preamble", default=None,
              help="shell line(s) to run before activation, e.g. "
                   "'module load mamba' on an HPC cluster.")
def cmd_generate(fdf: str, out_dir: Optional[str], cpu_np: int,
                 gpu_gpus: int, gpu_k: int, gpus_per_node: int,
                 cores_per_socket: int, cpu_block_size: int,
                 gpu_block_size: int, max_scf: int,
                 cpu_time: Optional[str], gpu_time: Optional[str],
                 cpu_cpus_per_task: int,
                 gpu_exclusive: Optional[bool],
                 activation: Optional[str],
                 preamble: Optional[str]) -> None:
    """Generate CPU-only + GPU-only benchmark bundles from one ``.fdf``.

    Emits ``job-cpu`` (plain diagon -> ``molbuilder-siesta``) and
    ``job-gpu`` (``Diag.ELPA.GPU`` -> ``molbuilder-siesta-gpu``), each a
    full ``.fdf`` + ``.run.sh`` (+ ``.sbatch`` when a scheduler is
    configured), plus a self-contained ``job-gpu-sweep.sh`` helper and a
    README.  Both are cold + SCF-capped so CPU and GPU are directly
    comparable; you measure and pick the mechanism for the real run.
    """
    from ..runtime_config import RuntimeConfigError
    from .generate import generate_bench_bundle

    if gpu_k < 1 or gpu_k > cores_per_socket:
        click.echo(f"ERROR: --gpu-k must be 1..{cores_per_socket} "
                   f"(cores/socket); got {gpu_k}.", err=True)
        raise SystemExit(2)

    try:
        out, written = generate_bench_bundle(
            fdf, out_dir,
            cpu_np=cpu_np, gpu_gpus=gpu_gpus, gpu_k=gpu_k,
            gpus_per_node=gpus_per_node, cores_per_socket=cores_per_socket,
            cpu_block_size=cpu_block_size, gpu_block_size=gpu_block_size,
            max_scf=max_scf, cpu_time=cpu_time, gpu_time=gpu_time,
            cpu_cpus_per_task=cpu_cpus_per_task, gpu_exclusive=gpu_exclusive,
            activation=activation, preamble=preamble, echo=click.echo,
        )
    except (ValueError, OSError, RuntimeConfigError) as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(2)

    gpu_c = max(1, cores_per_socket // gpu_k)
    click.echo("==== molbuilder bench generate ====")
    click.echo(f"input  : {fdf}")
    click.echo(f"out    : {out}")
    click.echo(f"job-cpu: plain diagon, default -n {cpu_np}")
    click.echo(f"job-gpu: ELPA-CUDA, default G={gpu_gpus} K={gpu_k} "
               f"(-n {gpu_k*gpu_gpus} -c {gpu_c})")
    has_sbatch = any(p.suffix == ".sbatch" for p in written)
    if not has_sbatch:
        click.echo("note   : no scheduler configured -> only .run.sh emitted "
                   "(no .sbatch); see slurm-integration.md.")
    click.echo("")
    for p in written:
        click.echo(f"  {p.relative_to(out)}")
    click.echo("")
    click.echo("Next: read README.md, then `sbatch job-cpu.sbatch` and "
               "`./job-gpu-sweep.sh` in the out dir.")


@bench_group.command("prep",
                     short_help="detect the target machine + format the "
                                "benchmark scripts for it")
@click.option("--out", default=".",
              type=click.Path(file_okay=False, resolve_path=True),
              help="bundle directory to write into (default: current dir).")
@click.option("--scheduler", type=click.Choice(["slurm", "workstation"]),
              default=None, help="force the scheduler (else auto-detected).")
@click.option("--cores-per-socket", type=int, default=None,
              help="override detected cores/socket.")
@click.option("--gpus-per-node", type=int, default=None,
              help="override detected GPUs/node.")
@click.option("--gpu-type", default=None,
              help="override detected GPU type (e.g. a100).")
@click.option("--gpu-ks", default=None,
              help="comma-separated K (ranks/GPU) values to sweep, e.g. "
                   "8,16 (default: cores/socket divisors).")
def cmd_prep(out: str, scheduler: Optional[str], cores_per_socket,
             gpus_per_node, gpu_type, gpu_ks) -> None:
    """Detect this machine (scheduler + topology) and format the benchmark
    scripts for it -- step 1 of the on-target workflow
    (docs/protocols/benchmark-workflow.md § 7.2).

    Writes ``environment.json`` + the topology-sized ``job-gpu-sweep.sh``.
    Run it in the bundle directory on the target; no hand-editing needed.
    """
    from .prep import (_overrides_from, _parse_ks, _summary, run_prep_bench,
                       utc_now_iso)

    env, written = run_prep_bench(
        out,
        overrides=_overrides_from(cores_per_socket, gpus_per_node, gpu_type),
        scheduler_override=scheduler,
        ks=_parse_ks(gpu_ks),
        now_iso=utc_now_iso())
    click.echo(_summary(env, written))


@bench_group.command("summarize",
                     short_help="read a sweep's outputs -> bench-result.json")
@click.option("--bundle", default=".",
              type=click.Path(file_okay=False, exists=True, resolve_path=True),
              help="bundle directory holding the point-G*K*/ run dirs.")
@click.option("--out", default=None, type=click.Path(),
              help="output path (default: <bundle>/bench-result.json).")
def cmd_summarize(bundle: str, out: Optional[str]) -> None:
    """Read the benchmark sweep's per-point outputs and write
    ``bench-result.json`` -- the portable verdict + ``choice`` that
    ``molbuilder bench prep-run`` consumes
    (docs/protocols/benchmark-workflow.md § 7.4).
    """
    from .prep import utc_now_iso
    from .summarize import run_summarize, summary_text

    res, out_path = run_summarize(bundle, out=out, now_iso=utc_now_iso())
    click.echo(summary_text(res, out_path))


@bench_group.command("prep-run",
                     short_help="bench-result.json -> production run-script "
                                "for this machine")
@click.option("--bench-result", "bench_result", default="bench-result.json",
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="the bench-result.json from `bench summarize`.")
@click.option("--script-base", default="job",
              help="basename of the production scripts "
                   "(<base>.fdf/.run.sh/.sbatch).")
@click.option("--out", default=None, type=click.Path(),
              help="output path (default: run-production.sh beside the "
                   "bench-result).")
@click.option("--scheduler", type=click.Choice(["slurm", "workstation"]),
              default=None, help="force the scheduler (else auto-detected).")
@click.option("--cores-per-socket", type=int, default=None)
@click.option("--gpus-per-node", type=int, default=None)
@click.option("--gpu-type", default=None)
def cmd_prep_run(bench_result: str, script_base: str, out: Optional[str],
                 scheduler: Optional[str], cores_per_socket, gpus_per_node,
                 gpu_type) -> None:
    """Format the production run from the benchmark verdict, re-resolved
    for THIS machine (docs/protocols/benchmark-workflow.md § 7.5).

    Reads ``bench-result.json``, applies the winning mechanism to your
    production scripts (``--script-base``), and writes ``run-production.sh``.
    The portable *choice* transfers; the concrete ``-n``/``-c``/``-G`` are
    re-resolved from this machine's topology (§ 5.4).
    """
    from .prep import _overrides_from, utc_now_iso
    from .prep_run import _summary, run_prep_run

    try:
        env, choice, out_path = run_prep_run(
            bench_result,
            script_base=script_base, out=out, scheduler_override=scheduler,
            overrides=_overrides_from(cores_per_socket, gpus_per_node,
                                      gpu_type),
            now_iso=utc_now_iso())
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(2)
    click.echo(_summary(env, choice, out_path))


__all__ = ["bench_group"]
