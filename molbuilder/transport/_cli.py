"""``molbuilder transport`` CLI group (TranSIESTA workflow helpers)."""

from __future__ import annotations

from pathlib import Path

import click

from .orchestrate import (
    DEFAULT_RELAX_STEPS,
    build_transport_bundle,
    format_bundle,
)
from .preflight import format_report, parse_fdf_params, preflight_files
from .wizard import DEFAULT_ELECTRODE_KZ, electrode_wizard, format_models


def _load_device(device_xyz, sidecar_path, cell_fdf):
    """Load a region-labeled device Structure from xyz + sidecar, and
    resolve its lattice (sidecar 'cell' if present, else a reference
    fdf's LatticeVectors via --cell-fdf).  Shared by electrode/bundle."""
    import numpy as np
    from pathlib import Path as _P
    from ..sidecars.molstruct import (apply_to_structure,
                                       load as load_sidecar, sidecar_path_for)
    from ..structure import Structure

    struct = Structure.from_xyz(device_xyz)
    sc = _P(sidecar_path) if sidecar_path else sidecar_path_for(device_xyz)
    if not _P(sc).exists():
        raise click.ClickException(
            f"no region sidecar found at {sc}; the device must carry "
            f"region labels (*-electrode). Pass --sidecar explicitly.")
    apply_to_structure(struct, load_sidecar(sc))

    # A --cell-fdf reference lattice overrides whatever the sidecar held
    # (lets the user point at an existing relaxed .fdf's hex cell).
    if cell_fdf:
        cell = parse_fdf_params(_P(cell_fdf).read_text()).cell_ang
        if not cell:
            raise click.ClickException(
                f"no LatticeVectors block found in {cell_fdf}")
        struct.cell = np.asarray(cell, dtype=float)
    if struct.cell is None:
        click.echo("WARNING: device has no lattice (no sidecar 'cell', no "
                   "--cell-fdf); the emitter will fabricate an orthorhombic "
                   "vacuum box (isolated-cluster model, NOT a periodic "
                   "surface). Supply --cell-fdf for a real Au(111) lead.",
                   err=True)
    return struct


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
@click.option("--cell-fdf", "cell_fdf", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="reference .fdf to read the device lattice (LatticeVectors) "
                   "from; preserves a hexagonal Au(111) cell.")
@click.option("--out-dir", type=click.Path(file_okay=False), default=".",
              show_default=True, help="where to write the electrode .fdf(s).")
def cmd_electrode(device_xyz, sidecar_path, which, job_name, mesh_cutoff,
                  kx, ky, electrode_kz, z_period, cell_fdf, out_dir):
    """Derive the bulk-electrode `.fdf`(s) from a region-labeled device
    (docs/protocols/transiesta-workflow.md § 6.2).

    Clones the `*-electrode` region's exact atoms + the device lateral
    cell + the device numerical contract, so the device<->electrode
    invariants (§ 6.7) hold by construction.  Run `transport preflight`
    on the resulting pair to confirm.
    """
    from ..config.transport import TransportConfig

    struct = _load_device(device_xyz, sidecar_path, cell_fdf)

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


@transport_group.command("bundle",
                         short_help="derive the full relax+electrode+device "
                                    "run bundle from a labeled device")
@click.option("--device", "device_xyz", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="device structure (.xyz); its .molstruct.json sidecar "
                   "(region labels) is auto-discovered alongside it.")
@click.option("--sidecar", "sidecar_path", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="explicit .molstruct.json (overrides auto-discovery).")
@click.option("--job-name", default="transport", show_default=True,
              help="device job name; the bundle files are named from it.")
@click.option("--mesh-cutoff", type=int, default=None,
              help="MeshCutoff (Ry) for all runs (default: config default).")
@click.option("--kx", type=int, default=None, help="transverse kx.")
@click.option("--ky", type=int, default=None, help="transverse ky.")
@click.option("--electrode-kz", type=int, default=DEFAULT_ELECTRODE_KZ,
              show_default=True, help="dense kz for the bulk leads (§ 4.2).")
@click.option("--relax-steps", type=int, default=DEFAULT_RELAX_STEPS,
              show_default=True, help="CG steps for the device relaxation.")
@click.option("--z-period", type=float, default=None,
              help="bulk repeat along transport (Å); override the estimate.")
@click.option("--cell-fdf", "cell_fdf", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="reference .fdf to read the device lattice from "
                   "(preserves a hexagonal Au(111) cell).")
@click.option("--out-dir", type=click.Path(file_okay=False), default=".",
              show_default=True, help="where to write the bundle.")
def cmd_bundle(device_xyz, sidecar_path, job_name, mesh_cutoff, kx, ky,
               electrode_kz, relax_steps, z_period, cell_fdf, out_dir):
    """Derive the full 3-run bundle (relax + electrodes + device + driver)
    from one region-labeled device (transiesta-workflow.md § 6.4).

    All four `.fdf`s share one numerical contract by construction; the
    `run-transport.sh` driver runs them in order and performs the
    relax→device and electrode→device file hand-offs automatically.
    Run it under `molbuilder-siesta`.
    """
    from ..config.transport import TransportConfig

    struct = _load_device(device_xyz, sidecar_path, cell_fdf)

    cfg_kw = {"job_name": job_name}
    if mesh_cutoff is not None:
        cfg_kw["siesta_mesh_cutoff_ry"] = mesh_cutoff
    if kx is not None and ky is not None:
        cfg_kw["k_mesh_transverse"] = (kx, ky, 1)
    cfg = TransportConfig(**cfg_kw)

    try:
        bundle = build_transport_bundle(
            struct, cfg, electrode_kz=electrode_kz, relax_steps=relax_steps,
            z_period=z_period)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in bundle.files.items():
        path = out / name
        path.write_text(content)
        if name.endswith(".sh"):
            path.chmod(0o755)
    click.echo(format_bundle(bundle))
    click.echo(f"\nwrote {len(bundle.files)} files to {out}/")
    click.echo(f"Run under molbuilder-siesta:  bash {out}/run-transport.sh")


__all__ = ["transport_group"]
