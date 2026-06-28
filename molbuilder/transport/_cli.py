"""``molbuilder transport`` CLI group (TranSIESTA workflow helpers)."""

from __future__ import annotations

from pathlib import Path

import click

from .preflight import format_report, preflight_files
from .wizard import DEFAULT_ELECTRODE_KZ, electrode_wizard, format_models


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


@transport_group.command("electrode",
                         short_help="derive bulk-lead .fdf(s) from a "
                                    "labeled device structure")
@click.option("--device", "device_xyz", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="device structure (.xyz); its .molstruct.json sidecar "
                   "(region labels) is auto-discovered alongside it.")
@click.option("--sidecar", "sidecar_path", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="explicit .molstruct.json (overrides auto-discovery).")
@click.option("--which", default="both", show_default=True,
              help="'both', or an electrode label/name (e.g. 'L-electrode' "
                   "or 'L').")
@click.option("--job-name", default="transport", show_default=True,
              help="device job name; electrode files are <job>_<label>.fdf "
                   "(matches the device's TS.Elec HS reference).")
@click.option("--mesh-cutoff", type=int, default=None,
              help="MeshCutoff (Ry) -- MUST match the device fdf "
                   "(default: TransportConfig default).")
@click.option("--kx", type=int, default=None,
              help="transverse kx -- MUST match the device.")
@click.option("--ky", type=int, default=None,
              help="transverse ky -- MUST match the device.")
@click.option("--electrode-kz", type=int, default=DEFAULT_ELECTRODE_KZ,
              show_default=True,
              help="dense kz for the periodic bulk lead (converge it, § 4.2).")
@click.option("--z-period", type=float, default=None,
              help="bulk repeat along transport (Å); override the "
                   "layer-spacing estimate.")
@click.option("--out-dir", type=click.Path(file_okay=False), default=".",
              show_default=True, help="where to write the electrode .fdf(s).")
def cmd_electrode(device_xyz, sidecar_path, which, job_name, mesh_cutoff,
                  kx, ky, electrode_kz, z_period, out_dir):
    """Derive the bulk-electrode `.fdf`(s) from a region-labeled device
    (docs/protocols/transiesta-workflow.md § 6.2).

    Clones the `*-electrode` region's exact atoms + the device lateral
    cell + the device numerical contract, so the device<->electrode
    invariants (§ 6.7) hold by construction.  Run `transport preflight`
    on the resulting pair to confirm.
    """
    from ..config.transport import TransportConfig
    from ..sidecars.molstruct import load as load_sidecar
    from ..sidecars.molstruct import apply_to_structure, sidecar_path_for
    from ..structure import Structure

    struct = Structure.from_xyz(device_xyz)
    sc = Path(sidecar_path) if sidecar_path else sidecar_path_for(device_xyz)
    if not Path(sc).exists():
        raise click.ClickException(
            f"no region sidecar found at {sc}; the device must carry "
            f"region labels (*-electrode). Pass --sidecar explicitly.")
    apply_to_structure(struct, load_sidecar(sc))

    cfg_kw = {"job_name": job_name}
    if mesh_cutoff is not None:
        cfg_kw["siesta_mesh_cutoff_ry"] = mesh_cutoff
    if kx is not None and ky is not None:
        cfg_kw["k_mesh_transverse"] = (kx, ky, 1)
    cfg = TransportConfig(**cfg_kw)

    try:
        models = electrode_wizard(
            struct, cfg, which=which, electrode_kz=electrode_kz,
            z_period=z_period)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for job, fdf, _model in models:
        (out / f"{job}.fdf").write_text(fdf)
    click.echo(format_models(models))
    click.echo(f"\nwrote {len(models)} electrode .fdf(s) to {out}/")
    click.echo("Next: run 'molbuilder transport preflight --device <device.fdf> "
               "--electrode <this.fdf>' to confirm the contract.")


__all__ = ["transport_group"]
