"""``molbuilder transport`` CLI group (TranSIESTA workflow helpers)."""

from __future__ import annotations

import click

from .preflight import format_report, preflight_files


@click.group("transport",
             context_settings={"help_option_names": ["-h", "--help"]})
def transport_group() -> None:
    """TranSIESTA transport-workflow helpers.

    A conductance run is three coupled calculations -- relax the junction,
    a separate bulk-electrode `.TSHS`, then the NEGF device run -- whose
    correctness hinges on the device and electrode sharing ONE numerical
    contract + a geometric clone + commensurate k.  These commands enforce
    that consistency (the actual failure mode).  Scientific basis:
    docs/protocols/transiesta-workflow.md.
    """


@transport_group.command("preflight",
                         short_help="check device<->electrode .fdf "
                                    "consistency before a TranSIESTA run")
@click.option("--device", "device", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="the NEGF device .fdf (SolutionMethod transiesta).")
@click.option("--electrode", "electrode", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="the bulk-lead .fdf that writes the .TSHS.")
@click.option("--min-electrode-thickness", type=float, default=12.0,
              show_default=True,
              help="warn if the electrode z-length (Ang) is below this "
                   "(principal-layer heuristic).")
@click.option("--electrode-kz-warn", type=int, default=20, show_default=True,
              help="warn if the electrode kz is below this (bulk lead needs "
                   "a dense, converged kz).")
def cmd_preflight(device: str, electrode: str, min_electrode_thickness: float,
                  electrode_kz_warn: int) -> None:
    """Validate the cross-run consistency contract between a device and an
    electrode `.fdf` (docs/protocols/transiesta-workflow.md § 6.3).

    Checks: commensurate transverse k, device ``kz=1`` + dense electrode
    ``kz``, identical XC / MeshCutoff / EnergyShift / basis, matching
    lateral cell, electrode thickness, device z-vacuum, and that the
    electrode writes its `.TSHS`.  Exits non-zero on any ERROR.
    """
    report = preflight_files(
        device, electrode,
        min_electrode_thickness_ang=min_electrode_thickness,
        electrode_kz_warn=electrode_kz_warn)
    click.echo(format_report(report))
    if not report.ok():
        raise SystemExit(1)


__all__ = ["transport_group"]
