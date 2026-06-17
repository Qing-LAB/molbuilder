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
    """Run small parameter sweeps over generated engine scripts.

    Reads the BENCH-MARKS block (see docs/protocols/script-contract.md)
    from a generated .fdf, sweeps a handful of (np, omp, BlockSize)
    test points, reports per-iter wall time.  Today: siesta-gpu only.
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


__all__ = ["bench_group"]
