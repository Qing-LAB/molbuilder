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
  Full reference: docs/config.md sections 3-4 (config lookup + schema).
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

    Emits ``job-cpu`` (ELPA-1STAGE, no CUDA) and ``job-gpu`` (ELPA-1STAGE +
    ``Diag.ELPA.GPU``) -- same solver, only the hardware differs.  Both run
    in ``molbuilder-siesta-gpu`` (ELPA lives only there); each a
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

    click.echo("==== molbuilder bench generate ====")
    click.echo(f"input  : {fdf}")
    click.echo(f"out    : {out}  (portable bundle -- run wrappers baked at prep)")
    click.echo(f"job-cpu: ELPA-1STAGE (no CUDA), -n {cpu_np}")
    click.echo(f"job-gpu: ELPA-1STAGE + Diag.ELPA.GPU, G={gpu_gpus} K={gpu_k}")
    click.echo("")
    for p in written:
        click.echo(f"  {p.relative_to(out)}")
    click.echo("")
    click.echo("Next: copy the bundle to the target, then "
               "`molbuilder bench prep` THERE (it detects the machine, "
               "resolves activation, and bakes job-{cpu,gpu}.run.sh / "
               ".sbatch). Then run job-cpu + `./job-gpu-sweep.sh`.")


_PREP_EPILOG = """\
\b
RESOURCE MODEL -- think in three independent knobs, no arithmetic in your head:
\b
  --gpus-per-node N   G = how many GPUs                (sweeps 1..N)
  --gpu-ks K[,K...]   K = parallel apps (MPI ranks) PER GPU
  --gpu-cs c[,c...]   c = CPU cores (OMP threads) PER app
\b
The tool computes the rest:
    ranks (np)   = K x G          (apps-per-GPU times #GPUs)
    cores total  = K x c x G      (and cores PER GPU = K x c)
A point is named point-G<G>K<K>C<c>; its job shows as job-gpu-G<G>K<K>C<c>.
\b
EXAMPLES (read each as a sentence):
\b
  "2 GPUs; 12 apps on each; 2 cores per app"   (= np 24, 48 cores, whole A100 node)
      --gpus-per-node 2 --gpu-ks 12 --gpu-cs 2          -> point-G2K12C2
\b
  "1 GPU; 1 app; 4 cores for it"
      --gpus-per-node 1 --gpu-ks 1 --gpu-cs 4           -> point-G1K1C4
\b
  "1 GPU; 4 apps; 6 cores each"   (the published SIESTA-GPU optimum: ~4 apps/GPU + small OMP)
      --gpus-per-node 1 --gpu-ks 4 --gpu-cs 6           -> point-G1K4C6
\b
  Sweep a GRID (compare combinations) -- comma-separate any axis:
      --gpus-per-node 2 --gpu-ks 4,8,12 --gpu-cs 1,2,4  -> all G x K x c points
\b
NOTES:
  * cores PER GPU (K x c) is capped only by the node; the A100 node has 48
    cores (2x24), so e.g. 1 GPU can take at most 48 cores; 64 needs >1 node
    (multi-node is not supported in v1).
  * SIESTA CPU baseline is pure-MPI (OMP=1 helps nothing); the GPU/ELPA path
    benefits from a *small* OMP (c ~ 3-6). Let the c-sweep measure it.
  * Defaults (no flags): K = cores/socket divisors; c per K = the bracket
    {1, cores//K, 2*cores//K} (starved / one-socket / cross-socket).
"""


@bench_group.command("prep",
                     short_help="detect the target machine + format the "
                                "benchmark scripts for it",
                     epilog=_PREP_EPILOG)
@click.option("--out", default=".",
              type=click.Path(file_okay=False, resolve_path=True),
              help="bundle directory to write into (default: current dir).")
@click.option("--scheduler", type=click.Choice(["slurm", "workstation"]),
              default=None, help="force the scheduler (else auto-detected).")
@click.option("--cores-per-socket", type=int, default=None,
              help="override detected cores/socket.")
@click.option("--gpus-per-node", type=int, default=None,
              help="G = number of GPUs (sweeps 1..N). See examples below.")
@click.option("--gpu-type", default=None,
              help="override detected GPU type (e.g. a100).")
@click.option("--gpu-ks", default=None,
              help="K = parallel apps (MPI ranks) PER GPU; comma-separated to "
                   "sweep, e.g. 4,8,12 (default: cores/socket divisors). "
                   "np = K*G.")
@click.option("--gpu-cs", default=None,
              help="c = CPU cores (OMP threads) PER app; comma-separated to "
                   "sweep, e.g. 1,2,6 (default per K: {1, cores//K, "
                   "2*cores//K}). cores/GPU = K*c. See examples below.")
@click.option("--exclusive/--no-exclusive", "exclusive", default=None,
              help="EXCLUSIVE: each GPU job reserves a whole node (clean "
                   "timing, slower to schedule; configured --mem is ignored "
                   "-> all node RAM).  --no-exclusive: pack onto shared nodes "
                   "(faster, possible timing noise).  Default: the config's "
                   "scheduler.gpu.exclusive.  The resolved mode is the FIRST "
                   "line prep prints.")
def cmd_prep(out: str, scheduler: Optional[str], cores_per_socket,
             gpus_per_node, gpu_type, gpu_ks, gpu_cs, exclusive) -> None:
    """Detect this machine (scheduler + topology) and format the benchmark
    scripts for it -- step 1 of the on-target workflow
    (docs/protocols/benchmark-workflow.md § 7.2).

    Writes ``environment.json`` + the topology-sized ``job-gpu-sweep.sh``.
    Run it in the bundle directory on the target; no hand-editing needed.
    """
    import json
    from pathlib import Path

    from ..runtime_config import (RuntimeConfigError, get_execution,
                                   get_routing, get_scheduler)
    from .adapters import (divisors, fitting_domains, parse_walltime,
                           recommend_domain, resolve_launch_adapter,
                           resolve_mode)
    from .generate import (bake_run_bench, bake_target_wrappers,
                           render_bench_plan)
    from .prep import (_overrides_from, _parse_ks, _summary, run_prep_bench,
                       utc_now_iso)

    # Resolve the run-vs-submit LAUNCH policy (job-execution.md § 8.13) +
    # the submission-domain routing table (slurm-integration.md § 4.3) from
    # the bundle's .molbuilder.json -- applied uniformly to the CPU baseline
    # and the GPU sweep, independent of the detected scheduler.
    try:
        exec_cfg = get_execution(project_dir=Path(out))
        routing = get_routing(project_dir=Path(out))
        sched = get_scheduler(project_dir=Path(out)) or {}
    except RuntimeConfigError as e:
        click.echo(f"ERROR reading execution/routing config: {e}", err=True)
        raise SystemExit(2)
    cfg_mode, submit_via = exec_cfg["mode"], exec_cfg["submit_via"]
    exec_domain = exec_cfg["domain"]

    env, written = run_prep_bench(
        out,
        overrides=_overrides_from(cores_per_socket, gpus_per_node, gpu_type),
        scheduler_override=scheduler,
        ks=_parse_ks(gpu_ks),
        cs=_parse_ks(gpu_cs),
        mode=cfg_mode,
        submit_via=submit_via,
        now_iso=utc_now_iso())

    rmode = resolve_mode(env, cfg_mode)

    # Resolve GPU-node EXCLUSIVITY: explicit --exclusive/--no-exclusive wins,
    # else the config default (job-execution.md § 8.15).  Announce it FIRST --
    # it is too easy to miss in the config and decides queue time + whether
    # the --mem request is honored.
    gpu_excl = (exclusive if exclusive is not None
                else bool(sched.get("gpu", {}).get("exclusive", False)))
    gpu_mem = sched.get("gpu", {}).get("mem")
    if rmode == "submit":
        if gpu_excl:
            click.echo(
                "Allocation: EXCLUSIVE -- each GPU job reserves a WHOLE node; "
                f"configured --mem ({gpu_mem or 'unset'}) is IGNORED (all node "
                "RAM). Clean timing, slower to schedule.  Override: "
                "./prep-bench --no-exclusive")
        else:
            click.echo(
                f"Allocation: SHARED -- GPU jobs request --mem="
                f"{gpu_mem or 'partition default'} and pack onto shared nodes "
                "(faster scheduling; possible timing noise).  Override: "
                "./prep-bench --exclusive")
    else:
        click.echo("Allocation: DIRECT run (no scheduler) -- exclusivity / "
                   "--mem do not apply.")

    # Bake the run wrappers for THIS target (job-execution.md § 7.4): resolve
    # activation (workstation autodetect / HPC shipped config) + write
    # job-{cpu,gpu}.run.sh(.sbatch).  .sbatch is emitted iff mode=submit
    # (§ 8.13); GPU exclusivity is the resolved value above.  This is what
    # makes one bundle portable.
    try:
        written += bake_target_wrappers(
            out, env, submit=(rmode == "submit"), exclusive=gpu_excl,
            echo=click.echo)
    except (RuntimeConfigError, FileNotFoundError) as e:
        click.echo(_summary(env, written))
        click.echo(f"\nERROR baking run wrappers: {e}", err=True)
        raise SystemExit(2)

    # Bake run-bench so the CPU baseline launches the SAME way as the sweep
    # (§ 8.13): both submit under mode=submit, both bash under direct.  Under
    # submit + a routing table, bake the explicit domain-selection gate +
    # recommendation (§ 4.3, § 4.4).
    manifest = json.loads((Path(out) / "bench-manifest.json").read_text())
    adapter, _ = resolve_launch_adapter(env, mode=cfg_mode,
                                        submit_via=submit_via)
    cpu_np = manifest["points"]["cpu"]["mpi_np"]

    # Effective walltime for the recommendation/fit-check: the manifest
    # point time (usually null) → scheduler defaults.time → the safe 4h.
    job_time = (manifest["points"]["gpu"].get("time")
                or sched.get("defaults", {}).get("time")
                or "0-04:00:00")
    try:
        job_secs = parse_walltime(job_time)
    except ValueError:
        job_secs = parse_walltime("0-04:00:00")
    # Memory is not pinned in the manifest (null → partition default), so a
    # domain WITH a max_mem_gb cap can't be proven to fit (§ 4.3).
    recommend = recommend_domain(routing, job_secs, None) if routing else None
    fitting = fitting_domains(routing, job_secs, None) if routing else []

    written.append(bake_run_bench(
        out, adapter, cpu_np, rmode, routing=routing, exec_domain=exec_domain,
        recommend=recommend, fitting=fitting, job_time=job_time))

    # Write + PRINT the human-readable benchmark plan (job-execution.md § 8.4):
    # the enumerated matrix (CPU baseline + the GPU G×K×c grid), what's
    # measured, and how to change it.  K/c match the sweep.
    ks = _parse_ks(gpu_ks) or divisors(env.topology.cores_per_socket or 0)
    cs = _parse_ks(gpu_cs)
    plan = render_bench_plan(env, manifest, ks, cs, mode=cfg_mode,
                             submit_via=submit_via, routing=routing,
                             recommend=recommend, job_time=job_time)
    plan_path = Path(out) / "BENCH-PLAN.md"
    plan_path.write_text(plan + "\n", encoding="utf-8")
    written.append(plan_path)

    click.echo(_summary(env, written))
    click.echo("\n" + plan)


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
