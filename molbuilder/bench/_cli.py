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
                   '+ host-bound stress.  See docs/execution/job-contracts.md.')
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


# ``_GENERATE_EPILOG`` stood here until U19 (2026-08-12) -- the worked
# example for the deleted `generate` verb, orphaned when u5 deleted its
# command.  The live walkthrough is running-a-job.md's jobset flow.


# --------------------------------------------------------------------- #
#  `generate`, `prep`, `summarize` and `prep-run` were DELETED 2026-08-12 #
#  (plan step 6 u5) with the shipped-bundle lifecycle they drove.  The    #
#  loop is `jobset describe -> prep bench -> submit bench <trial> ->      #
#  summarize bench -> prep run` (job-system.md § 5.3); a described        #
#  calculation already travels, so there is no bundle to generate and     #
#  nothing to ship.  `siesta-gpu` (its own call pending, plan row 8) and  #
#  `probe-scheduler` (a config helper) remain.                            #
# --------------------------------------------------------------------- #

@bench_group.command("probe-scheduler",
                     short_help="probe sinfo/sacctmgr -> proposed scheduler "
                                "config block")
@click.option("--out", default=".",
              type=click.Path(file_okay=False, resolve_path=True),
              help="bundle dir whose .molbuilder.json to update with --write "
                   "(default: current dir).")
@click.option("--write", "do_write", is_flag=True, default=False,
              help="merge the proposed scheduler block into "
                   "<out>/.molbuilder.json (shows a diff + confirms).")
@click.option("--yes", is_flag=True, default=False,
              help="skip the confirmation prompt when --write.")
def cmd_probe_scheduler(out: str, do_write: bool, yes: bool) -> None:
    """Probe this SLURM cluster (sinfo/sacctmgr) and propose a `scheduler`
    config block -- partitions, GPU type, and the routing menu derived from
    the LIVE system (job-system.md § 7).  Run on the login node;
    every name/limit comes from the cluster, none is hardcoded.
    """
    import getpass
    import json
    from pathlib import Path

    from ..runtime_config import (RuntimeConfigError, get_scheduler,
                                   write_config_scope)
    from ..environment import _run
    from .probe import (derive_scheduler_block, parse_allowed_qos, parse_qos,
                        parse_sinfo)

    user = getpass.getuser()
    sinfo_txt = _run(["sinfo", "-h", "-o", "%P|%30l|%D|%40G"])
    if sinfo_txt is None:
        click.echo("ERROR: could not run sinfo -- run this on a SLURM login "
                   "node (sinfo/sacctmgr must be on PATH).", err=True)
        raise SystemExit(2)
    qos_txt = _run(["sacctmgr", "-nP", "show", "qos",
                    "format=Name,MaxWall,Flags"])
    assoc_txt = _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                      "format=QOS"])

    parts = parse_sinfo(sinfo_txt)
    qos = parse_qos(qos_txt or "")
    allowed = parse_allowed_qos(assoc_txt or "")
    block, notes = derive_scheduler_block(parts, qos, allowed)
    if block is None:
        click.echo("Could not derive a scheduler block:", err=True)
        for n in notes:
            click.echo(f"  - {n}", err=True)
        raise SystemExit(2)

    gpu_parts = [p.name for p in parts if p.has_gpu]
    click.echo(f"Probed (user={user}): GPU partitions {gpu_parts}; "
               f"allowed QoS: {', '.join(sorted(allowed)) or '(unknown)'}; "
               f"GPU type -> {block['gpu']['default_type']}")
    click.echo("\nRouting domains (pick at run: ./run-bench --domain <name>):")
    for d in block["routing"]:
        click.echo(f"  {d['name']:<10} <= {d['max_time']:<12} "
                   f"{d['partition']}/{d['qos']}")
    click.echo("\nProposed scheduler block:\n")
    click.echo(json.dumps({"scheduler": block}, indent=2))
    click.echo("\nNotes / assumptions (read before --write):")
    for n in notes:
        click.echo(f"  - {n}")

    if not do_write:
        click.echo(f"\n(dry run -- nothing written. Re-run with --write to "
                   f"merge into {Path(out) / '.molbuilder.json'}.)")
        return

    # --write: show a before/after of the key fields, then merge.
    try:
        before = get_scheduler(project_dir=Path(out)) or {}
    except RuntimeConfigError:
        before = {}
    old_names = [d.get("name") for d in (before.get("routing") or [])]
    new_names = [d["name"] for d in block["routing"]]
    click.echo(f"\nDIFF scheduler.routing: {old_names or '(none)'} -> "
               f"{new_names}")
    click.echo(f"DIFF directives: "
               f"{before.get('directives', {}).get('partition')}/"
               f"{before.get('directives', {}).get('qos')} -> "
               f"{block['directives']['partition']}/"
               f"{block['directives']['qos']}")
    if not yes:
        click.confirm(f"Merge this scheduler block into "
                      f"{Path(out) / '.molbuilder.json'}?", abort=True)
    write_config_scope(Path(out), {"scheduler": block})
    click.echo(f"wrote scheduler block to {Path(out) / '.molbuilder.json'} "
               "(execution / script_generation preserved).")


